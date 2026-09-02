"""Source-only fine-tuning controls for the adapted Houston Full48 teacher.

This script starts from the selected Stage1 checkpoint and never reads
Houston18.  Use ``--mode partial`` for the last two transformer blocks or
``--mode full`` for all spatial/spectral transformer blocks.  Houston18
evaluation is intentionally implemented in a separate post-hoc audit script.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "HyperSIGMA" / "ImageClassification"))
import utils  # noqa: E402
from hypersigma_stage1_protocol import (  # noqa: E402
    CLASSES,
    IMG_SIZE,
    OUT_ROOT,
    SEED,
    disjoint_split,
    patches,
    seed_all,
)
from hypersigma_teacher_smoke_test import SSFusionFramework  # noqa: E402


STAGE1 = OUT_ROOT / "bands48" / "stage1_best.pth"
OUT_FT = OUT_ROOT
EPOCHS = 20
TRAIN_LR = 6e-5
PARTIAL_BLOCK_LR = 6e-6
FULL_LAST_BLOCK_LR = 3e-6
FULL_DECAY = 0.92
HALF = IMG_SIZE // 2


def block_index(name: str):
    match = re.search(r"(?:spat_encoder|spec_encoder)\.blocks\.(\d+)\.", name)
    return int(match.group(1)) if match else None


def configure_trainable(model: nn.Module, mode: str):
    """Freeze old backbone blocks and return optimizer parameter groups."""
    if mode not in {"partial", "full"}:
        raise ValueError(mode)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    groups = [{"params": [], "lr": TRAIN_LR, "name": "new_modules"}]
    block_groups = []
    for name, parameter in model.named_parameters():
        idx = block_index(name)
        if idx is None:
            parameter.requires_grad_(True)
            groups[0]["params"].append(parameter)
        elif mode == "full" or idx >= 10:
            parameter.requires_grad_(True)
            lr = FULL_LAST_BLOCK_LR * (FULL_DECAY ** (11 - idx)) if mode == "full" else PARTIAL_BLOCK_LR
            block_groups.append({"params": [parameter], "lr": lr, "name": name, "block": idx})

    # Grouping each block parameter separately makes the effective LR explicit
    # in config.json while retaining AdamW's normal parameter semantics.
    groups.extend(block_groups)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    summary = {
        "mode": mode,
        "trainable_parameters": trainable,
        "frozen_parameters": frozen,
        "new_module_lr": TRAIN_LR,
        "block_lrs": {},
    }
    for group in block_groups:
        summary["block_lrs"].setdefault(str(group["block"]), group["lr"])
    # Keep each transformer block in eval mode when it is frozen (there is no
    # stochastic dropout in the current model, but this makes the control
    # explicit and protects against future changes).
    for branch in (model.spat_encoder, model.spec_encoder):
        for idx, block in enumerate(branch.blocks):
            if not any(g.get("block") == idx for g in block_groups):
                block.eval()
    return groups, summary


def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loss_sum = correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            y = y.to(device)
            loss_sum += criterion(logits, y).item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
    return loss_sum / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("partial", "full"), required=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    seed_all(SEED)
    device = torch.device(args.device)
    out = OUT_FT / f"{args.mode}_ft"
    out.mkdir(parents=True, exist_ok=True)

    # Only Houston13 is loaded in this training script.  The split is copied
    # from Stage1 and checked against it to prevent accidental drift.
    source, src_gt = utils.load_data_houston(
        str(ROOT / "datasets/Houston/Houston13.mat"),
        str(ROOT / "datasets/Houston/Houston13_7gt.mat"),
    )
    split = np.load(OUT_ROOT / "bands48" / "source_split.npz")
    train_centers, val_centers, min_distance = disjoint_split(src_gt)
    if not np.array_equal(train_centers, split["train_centers"]) or not np.array_equal(val_centers, split["val_centers"]):
        raise RuntimeError("Stage1 split does not reproduce fixed seed-1174 centers")
    train_x = patches(source, train_centers, 48)
    val_x = patches(source, val_centers, 48)
    train_y = src_gt[train_centers[:, 0], train_centers[:, 1]].astype(np.int64) - 1
    val_y = src_gt[val_centers[:, 0], val_centers[:, 1]].astype(np.int64) - 1
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y)), batch_size=args.batch_size)

    model = SSFusionFramework(img_size=IMG_SIZE, in_channels=48, patch_size=2, classes=CLASSES, model_size="base")
    checkpoint = torch.load(STAGE1, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    groups, train_summary = configure_trainable(model, args.mode)
    model.to(device)
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.CrossEntropyLoss()
    history = []
    best = {"val_acc": -1.0}
    print(json.dumps({"mode": args.mode, "device": str(device), "bands": 48, "img_size": IMG_SIZE,
                      "patch_size": 2, "train": len(train_y), "val": len(val_y),
                      "min_center_distance": min_distance, "target_gt_used": False,
                      "train_summary": train_summary}, sort_keys=True))

    for epoch in range(1, args.epochs + 1):
        model.train()
        # Re-apply eval to frozen blocks after model.train().
        for branch in (model.spat_encoder, model.spec_encoder):
            for idx, block in enumerate(branch.blocks):
                if args.mode == "partial" and idx < 10:
                    block.eval()
        loss_sum = correct = total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)
        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": loss_sum / total, "train_acc": correct / total,
               "val_loss": val_loss, "val_acc": val_acc, "lr": scheduler.get_last_lr()[0]}
        history.append(row)
        print(json.dumps(row))
        if val_acc > best["val_acc"]:
            best = row.copy()
            torch.save({"model": model.state_dict(), "best": best, "mode": args.mode,
                        "bands": 48, "img_size": IMG_SIZE, "patch_size": 2,
                        "classes": CLASSES, "seed": SEED, "start_checkpoint": str(STAGE1)}, out / "best.pth")

    config = {"mode": args.mode, "bands": 48, "img_size": IMG_SIZE, "patch_size": 2,
              "classes": CLASSES, "seed": SEED, "epochs": args.epochs,
              "train_per_class": 180, "val_per_class": 30,
              "min_train_val_center_chebyshev_distance": min_distance,
              "target_gt_used_for_training_or_selection": False,
              "start_checkpoint": str(STAGE1), "train_summary": train_summary, "best": best}
    (out / "history.json").write_text(json.dumps(history, indent=2))
    (out / "config.json").write_text(json.dumps(config, indent=2))
    np.savez(out / "source_split.npz", train_centers=train_centers, val_centers=val_centers)
    print(json.dumps({"finished": True, "mode": args.mode, "best": best, "artifact": str(out)}, sort_keys=True))


if __name__ == "__main__":
    main()

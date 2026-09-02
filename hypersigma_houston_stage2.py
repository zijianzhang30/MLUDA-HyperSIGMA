"""Spatially-disjoint Houston HyperSIGMA adaptation experiments.

Stages:
  stage1: source-only adapter/SEM warm-up, transformer blocks frozen.
  control: source-only Stage 2, last two blocks of each backbone unfrozen.
  adapt: target-unlabeled Stage 2 using weak-view consistency and CORAL.

Houston18 ground truth is never read by this training script. Use the separate
post-hoc evaluation utilities only after all training choices are fixed.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import hdf5storage
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "HyperSIGMA" / "ImageClassification"))
import utils  # noqa: E402
from hypersigma_teacher_smoke_test import WEIGHT_DIR, SSFusionFramework, _load_official_transfer  # noqa: E402

OUT_ROOT = Path("/nas1/zhangzj26/HyperSIGMA_adapted/spatial_disjoint_seed1174")
SEED = 1174
NUM_CLASSES = 7
TRAIN_PER_CLASS = 180
VAL_PER_CLASS = 30
IMG_SIZE = 33
MARGIN = IMG_SIZE // 2
PCA_COMPONENTS = 30

# Each tuple is (axis, separating cut, train side). The 33-pixel gap around
# the cut guarantees no same-class train/val context overlap before the final
# all-class Chebyshev-distance verification.
CLASS_REGIONS = {
    1: (1, 498, "low"), 2: (1, 800, "high"), 3: (0, 89, "high"),
    4: (1, 505, "low"), 5: (1, 213, "high"), 6: (0, 119, "high"),
    7: (1, 648, "low"),
}


def seed_all(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def transform_source_only(source, target):
    pca = PCA(n_components=PCA_COMPONENTS, whiten=True, random_state=SEED)
    source_pca = pca.fit_transform(source.reshape(-1, source.shape[-1]).astype(np.float32))
    target_pca = pca.transform(target.reshape(-1, target.shape[-1]).astype(np.float32))
    return source_pca.reshape(*source.shape[:2], PCA_COMPONENTS).astype(np.float32), target_pca.reshape(*target.shape[:2], PCA_COMPONENTS).astype(np.float32), pca


def spatial_disjoint_split(gt):
    """Create a fixed 180/train + 30/val per class non-overlapping split."""
    rng = np.random.RandomState(SEED)
    train, val = [], []
    for cls in range(1, NUM_CLASSES + 1):
        points = np.argwhere(gt == cls)
        axis, cut, side = CLASS_REGIONS[cls]
        train_pool = points[points[:, axis] <= cut - MARGIN] if side == "low" else points[points[:, axis] >= cut + MARGIN + 1]
        val_pool = points[points[:, axis] >= cut + MARGIN + 1] if side == "low" else points[points[:, axis] <= cut - MARGIN]
        if len(train_pool) < TRAIN_PER_CLASS or len(val_pool) < VAL_PER_CLASS:
            raise RuntimeError(f"spatial split unavailable for class {cls}: train={len(train_pool)}, val={len(val_pool)}")
        train.extend(train_pool[rng.permutation(len(train_pool))[:TRAIN_PER_CLASS]].tolist())
        val.extend(val_pool[rng.permutation(len(val_pool))[:VAL_PER_CLASS]].tolist())
    train, val = np.asarray(train, dtype=np.int64), np.asarray(val, dtype=np.int64)
    # Enforce global disjointness across different semantic classes too.
    cheb = np.max(np.abs(val[:, None, :] - train[None, :, :]), axis=2).min(axis=1)
    if np.any(cheb < IMG_SIZE):
        raise RuntimeError(f"global split overlap: minimum Chebyshev distance={cheb.min()}")
    return train, val, int(cheb.min())


def extract_patches(padded, centers):
    out = np.empty((len(centers), PCA_COMPONENTS, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    for i, (r, c) in enumerate(centers):
        out[i] = padded[r:r + IMG_SIZE, c:c + IMG_SIZE].transpose(2, 0, 1)
    return out


class TargetPatchDataset(Dataset):
    """Unlabeled Houston18 patches. No labels/GT are loaded or retained."""
    def __init__(self, pca_cube):
        self.padded = np.pad(pca_cube, ((MARGIN, MARGIN), (MARGIN, MARGIN), (0, 0)), mode="constant")
        self.h, self.w = pca_cube.shape[:2]
    def __len__(self):
        return self.h * self.w
    def __getitem__(self, index):
        r, c = divmod(index, self.w)
        return torch.from_numpy(self.padded[r:r + IMG_SIZE, c:c + IMG_SIZE].transpose(2, 0, 1).copy())


def weak_hsi_augmentation(x):
    """Label-preserving HSI weak views: flip/90-degree rotation and tiny gain."""
    batch = []
    for sample in x:
        if torch.rand(()) < 0.5:
            sample = sample.flip(-1)
        if torch.rand(()) < 0.5:
            sample = sample.flip(-2)
        k = int(torch.randint(0, 4, ()).item())
        sample = torch.rot90(sample, k, (-2, -1))
        gain = 1.0 + 0.02 * torch.randn((sample.shape[0], 1, 1), device=sample.device)
        batch.append(sample * gain)
    return torch.stack(batch)


def coral(source, target):
    """Stable normalized CORAL alignment on task-head pre-classifier features."""
    source, target = F.normalize(source, dim=1), F.normalize(target, dim=1)
    cs = (source - source.mean(0)).T @ (source - source.mean(0)) / max(len(source) - 1, 1)
    ct = (target - target.mean(0)).T @ (target - target.mean(0)) / max(len(target) - 1, 1)
    return (cs - ct).pow(2).mean()


def features(teacher, x):
    b = x.shape[0]
    spatial = teacher.spat_encoder(x)
    spectral = teacher.spec_encoder(x)[0]
    spec_pool = teacher.pool(spectral).view(b, -1)
    weights = [layer(spec_pool).view(b, -1, 1, 1) for layer in (teacher.fc_spec1, teacher.fc_spec2, teacher.fc_spec3, teacher.fc_spec4)]
    reduced = [layer(f) for layer, f in zip((teacher.DR1, teacher.DR2, teacher.DR3, teacher.DR4), spatial)]
    sem = torch.cat([F.adaptive_avg_pool2d((1 + w) * f, 1).flatten(1) for w, f in zip(weights, reduced)], dim=1)
    return spatial, spectral, sem, teacher.classifier(sem)


def freeze_stage(teacher, stage):
    for name, p in teacher.named_parameters():
        if ".blocks." not in name:
            p.requires_grad_(True)
            continue
        # Stage 1: freeze every pretrained block. Stage 2: unfreeze blocks 10-11.
        p.requires_grad_(stage != "stage1" and ("blocks.10." in name or "blocks.11." in name))


def optimizer_for(teacher, stage):
    new, backbone = [], []
    for name, p in teacher.named_parameters():
        if not p.requires_grad:
            continue
        (backbone if ".blocks." in name else new).append(p)
    groups = [{"params": new, "lr": 6e-5, "name": "new_modules"}]
    if backbone:
        groups.append({"params": backbone, "lr": 6e-6, "name": "last_two_pretrained_blocks"})
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), weight_decay=0.05)


def evaluate_source(teacher, loader, device):
    teacher.eval(); total = correct = 0; loss_sum = 0.0; ce = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            logits = teacher(x.to(device)); y = y.to(device)
            loss_sum += ce(logits, y).item() * len(y)
            correct += (logits.argmax(1) == y).sum().item(); total += len(y)
    return loss_sum / total, correct / total


def model_from_pretraining(device):
    model = SSFusionFramework(img_size=IMG_SIZE, in_channels=PCA_COMPONENTS, patch_size=2, classes=NUM_CLASSES, model_size="base")
    _load_official_transfer(model, "spat", WEIGHT_DIR / "spat-vit-base-ultra-checkpoint-1599.pth")
    _load_official_transfer(model, "spec", WEIGHT_DIR / "spec-vit-base-ultra-checkpoint-1599.pth")
    return model.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("stage1", "control", "adapt"), required=True)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lambda-cons", type=float, default=1.0)
    ap.add_argument("--lambda-align", type=float, default=0.1)
    args = ap.parse_args()
    seed_all(SEED); device = torch.device(args.device); OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Target GT is deliberately not loaded. The input cube is all that Stage 2 sees.
    source, source_gt = utils.load_data_houston(str(ROOT / "datasets/Houston/Houston13.mat"), str(ROOT / "datasets/Houston/Houston13_7gt.mat"))
    target = hdf5storage.loadmat(str(ROOT / "datasets/Houston/Houston18.mat"))["ori_data"]
    src_pca, tgt_pca, pca = transform_source_only(source, target)
    train_centers, val_centers, min_distance = spatial_disjoint_split(source_gt)
    src_pad = np.pad(src_pca, ((MARGIN, MARGIN), (MARGIN, MARGIN), (0, 0)), mode="constant")
    train_x, val_x = extract_patches(src_pad, train_centers), extract_patches(src_pad, val_centers)
    train_y = source_gt[train_centers[:, 0], train_centers[:, 1]].astype(np.int64) - 1
    val_y = source_gt[val_centers[:, 0], val_centers[:, 1]].astype(np.int64) - 1
    if args.stage == "stage1":
        np.savez(OUT_ROOT / "spatial_disjoint_source_split.npz", train_centers=train_centers, val_centers=val_centers)
        with open(OUT_ROOT / "spatial_disjoint_source_pca.pkl", "wb") as f: pickle.dump(pca, f)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y)), batch_size=args.batch_size)
    target_loader = DataLoader(TargetPatchDataset(tgt_pca), batch_size=args.batch_size, shuffle=True, drop_last=True)

    if args.stage == "stage1":
        model = model_from_pretraining(device)
        base_name = "stage1_disjoint_best.pth"
    else:
        ckpt_path = OUT_ROOT / "stage1_disjoint_best.pth"
        if not ckpt_path.exists(): raise FileNotFoundError(f"Run stage1 first: {ckpt_path}")
        model = model_from_pretraining(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
        base_name = f"stage2_{args.stage}_best.pth"
    freeze_stage(model, args.stage)
    opt = optimizer_for(model, args.stage)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    ce = nn.CrossEntropyLoss(); history = []; best = {"val_acc": -1.0}
    print(f"stage={args.stage} device={device} split: train={len(train_y)}, val={len(val_y)}, min_center_distance={min_distance}; target_gt_used=False")
    for epoch in range(1, args.epochs + 1):
        model.train()
        # Frozen blocks remain deterministic; the two unfrozen final blocks train in Stage 2.
        if args.stage == "stage1":
            model.spat_encoder.blocks.eval(); model.spec_encoder.blocks.eval()
        total = correct = 0; sums = {"total": 0.0, "cls": 0.0, "cons": 0.0, "align": 0.0}
        target_iter = iter(target_loader)
        for sx, sy in train_loader:
            sx, sy = sx.to(device), sy.to(device)
            _, _, src_sem, logits = features(model, sx)
            cls = ce(logits, sy)
            cons = align = torch.zeros((), device=device)
            if args.stage == "adapt":
                try: tx = next(target_iter)
                except StopIteration: target_iter = iter(target_loader); tx = next(target_iter)
                tx = tx.to(device)
                _, _, target_sem_a, target_logits_a = features(model, weak_hsi_augmentation(tx))
                _, _, target_sem_b, target_logits_b = features(model, weak_hsi_augmentation(tx))
                cons = F.mse_loss(F.softmax(target_logits_a, 1), F.softmax(target_logits_b, 1))
                align = coral(src_sem, target_sem_a)
            loss = cls + args.lambda_cons * cons + args.lambda_align * align
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            n = len(sy); total += n; correct += (logits.argmax(1) == sy).sum().item()
            for key, value in (("total", loss), ("cls", cls), ("cons", cons), ("align", align)): sums[key] += value.item() * n
        train_loss, train_acc = sums["total"] / total, correct / total
        val_loss, val_acc = evaluate_source(model, val_loader, device)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc, "cls_loss": sums["cls"] / total, "cons_loss": sums["cons"] / total, "align_loss": sums["align"] / total, "lrs": [group["lr"] for group in opt.param_groups]}
        history.append(row); print(json.dumps(row))
        if val_acc > best["val_acc"]:
            best = row.copy()
            torch.save({"model": model.state_dict(), "stage": args.stage, "best": best, "seed": SEED, "img_size": IMG_SIZE, "pca_components": PCA_COMPONENTS, "classes": NUM_CLASSES}, OUT_ROOT / base_name)
    (OUT_ROOT / f"{args.stage}_history.json").write_text(json.dumps(history, indent=2))
    config = {"stage": args.stage, "seed": SEED, "target_gt_used_for_training_or_selection": False, "source_train_per_class": TRAIN_PER_CLASS, "source_val_per_class": VAL_PER_CLASS, "patch_size": IMG_SIZE, "min_train_val_center_chebyshev_distance": min_distance, "lambda_cons": args.lambda_cons if args.stage == "adapt" else 0.0, "lambda_align": args.lambda_align if args.stage == "adapt" else 0.0, "best": best}
    (OUT_ROOT / f"{args.stage}_config.json").write_text(json.dumps(config, indent=2))
    print("BEST", json.dumps(best))


if __name__ == "__main__":
    main()

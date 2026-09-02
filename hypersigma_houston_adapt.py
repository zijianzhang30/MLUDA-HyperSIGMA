"""Stage-1 Houston-only adaptation of the HyperSIGMA classification teacher.

The MLUDA code is intentionally not imported here.  Houston13 labels are used
for fitting PCA, supervised training, and source validation only. Houston18 is
read after training solely for an offline feature-quality analysis.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "HyperSIGMA" / "ImageClassification"))
import hdf5storage  # noqa: E402
import utils  # noqa: E402
from config_Houston import HalfWidth  # noqa: E402
from hypersigma_teacher_smoke_test import (  # noqa: E402
    WEIGHT_DIR, SSFusionFramework, _checkpoint_state, _load_official_transfer,
)

OUT = Path("/nas1/zhangzj26/HyperSIGMA_adapted")
SEED = 1174
NUM_CLASSES = 7
TRAIN_PER_CLASS = 180
IMG_SIZE = 33
PATCH_SIZE = 2
PCA_COMPONENTS = 30


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fit_pca(source):
    # Official ImageClassification preprocessing uses PCA(whiten=True).
    pca = PCA(n_components=PCA_COMPONENTS, whiten=True, random_state=SEED)
    flat = source.reshape(-1, source.shape[-1]).astype(np.float32)
    transformed = pca.fit_transform(flat).reshape(source.shape[0], source.shape[1], PCA_COMPONENTS)
    return transformed.astype(np.float32), pca


def make_center_patches(data, gt, indices):
    margin = IMG_SIZE // 2
    padded = np.pad(data, ((margin, margin), (margin, margin), (0, 0)), mode="constant")
    patches, labels = [], []
    for flat_index in indices:
        row, col = np.unravel_index(int(flat_index), gt.shape)
        patch = padded[row:row + IMG_SIZE, col:col + IMG_SIZE]
        patches.append(np.transpose(patch, (2, 0, 1)))
        labels.append(int(gt[row, col]) - 1)
    return np.asarray(patches, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def fixed_source_split(gt):
    """Exactly reproduce utils.get_sample_data(..., HalfWidth=3, 180)."""
    np.random.seed(SEED)
    padded = np.pad(gt, HalfWidth, mode="constant")
    rows, cols = np.nonzero(padded)
    train_local, val_local = [], []
    for cls in range(1, NUM_CLASSES + 1):
        positions = [j for j in range(len(rows)) if padded[rows[j], cols[j]] == cls]
        np.random.shuffle(positions)
        if len(positions) < TRAIN_PER_CLASS:
            raise RuntimeError(f"class {cls} has only {len(positions)} pixels")
        train_local.extend(positions[:TRAIN_PER_CLASS])
        val_local.extend(positions[TRAIN_PER_CLASS:])
    np.random.shuffle(train_local)
    np.random.shuffle(val_local)
    # Baseline stores locations in the padded map. Convert back to source-cube
    # flat indices for 33x33 extraction.
    train = [(rows[j] - HalfWidth) * gt.shape[1] + (cols[j] - HalfWidth) for j in train_local]
    val = [(rows[j] - HalfWidth) * gt.shape[1] + (cols[j] - HalfWidth) for j in val_local]
    return np.asarray(train, dtype=np.int64), np.asarray(val, dtype=np.int64)


def module_trainable_stage1(teacher):
    for name, parameter in teacher.named_parameters():
        # Freeze only pretrained transformer blocks in Stage 1. All adapters,
        # FPN/projection, SEM and classifier remain trainable.
        parameter.requires_grad_(not ("spat_encoder.blocks." in name or "spec_encoder.blocks." in name))


def load_stats(teacher):
    stats = {}
    for branch, filename, fragments in (
        ("spatial", "spat-vit-base-ultra-checkpoint-1599.pth", ("patch_embed.proj", "spat_map", "spat_output_maps", "pos_embed")),
        ("spectral", "spec-vit-base-ultra-checkpoint-1599.pth", ("patch_embed", "spat_map", "fpn1.0.weight")),
    ):
        state = _checkpoint_state(WEIGHT_DIR / filename)
        target_name = "spat_encoder" if branch == "spatial" else "spec_encoder"
        target = getattr(teacher, target_name).state_dict()
        filtered = {k: v for k, v in state.items() if not any(f in k for f in fragments)}
        loaded = {k: v for k, v in filtered.items() if k in target and tuple(v.shape) == tuple(target[k].shape)}
        stats[branch] = {
            "checkpoint_entries": len(state),
            "filtered_entries": len(filtered),
            "loaded_entries": len(loaded),
            "random_or_unloaded_target_entries": len(target) - len(loaded),
            "target_module": target_name,
        }
    return stats


def forward_features(teacher, x):
    b = x.shape[0]
    spatial = teacher.spat_encoder(x)
    spectral = teacher.spec_encoder(x)[0]
    spectral_pool = teacher.pool(spectral).view(b, -1)
    weights = [layer(spectral_pool).view(b, -1, 1, 1) for layer in (teacher.fc_spec1, teacher.fc_spec2, teacher.fc_spec3, teacher.fc_spec4)]
    reduced = [layer(feat) for layer, feat in zip((teacher.DR1, teacher.DR2, teacher.DR3, teacher.DR4), spatial)]
    parts = [F.adaptive_avg_pool2d((1 + w) * feat, 1).flatten(1) for w, feat in zip(weights, reduced)]
    sem = torch.cat(parts, dim=1)
    logits = teacher.classifier(sem)
    return spatial, spectral, sem, logits


def evaluate(teacher, loader, device):
    teacher.eval()
    total, correct, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for xb, yb in loader:
            logits = teacher(xb.to(device))
            yb = yb.to(device)
            loss_sum += criterion(logits, yb).item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)
    return loss_sum / total, correct / total


def quality_probe(teacher, source_data, source_gt, target_data, target_gt, device, max_per_class=100):
    def samples(data, gt):
        rng = np.random.default_rng(20260901)
        ids = []
        for cls in range(1, NUM_CLASSES + 1):
            cls_ids = np.flatnonzero(gt.reshape(-1) == cls)
            ids.extend(rng.permutation(cls_ids)[:max_per_class].tolist())
        return make_center_patches(data, gt, ids)

    def extract(data, gt):
        x, y = samples(data, gt)
        out = {"spatial": [], "spectral": [], "sem": []}
        with torch.no_grad():
            for i in range(0, len(x), 64):
                xb = torch.from_numpy(x[i:i + 64]).to(device)
                spatial, spectral, sem, _ = forward_features(teacher, xb)
                out["spatial"].append(F.adaptive_avg_pool2d(spatial[-1], 1).flatten(1).cpu().numpy())
                out["spectral"].append(spectral.mean(1).cpu().numpy())
                out["sem"].append(sem.cpu().numpy())
        return {k: np.concatenate(v) for k, v in out.items()}, y

    def stats(a, ya, b=None, yb=None):
        cross = b is not None
        if b is None:
            b, yb = a, ya
        aa = F.normalize(torch.from_numpy(a), dim=1).numpy()
        bb = F.normalize(torch.from_numpy(b), dim=1).numpy()
        sim = aa @ bb.T
        same = ya[:, None] == yb[None, :]
        if not cross:
            np.fill_diagonal(same, False)
            np.fill_diagonal(sim, np.nan)
        return float(np.nanmean(sim[same])), float(np.nanmean(sim[~same]))

    teacher.eval()
    src, sy = extract(source_data, source_gt)
    tgt, ty = extract(target_data, target_gt)
    result = {}
    for key in ("spatial", "spectral", "sem"):
        s13 = stats(src[key], sy)
        s18 = stats(tgt[key], ty)
        cross = stats(src[key], sy, tgt[key], ty)
        result[key] = {"source_same": s13[0], "source_diff": s13[1], "target_same": s18[0], "target_diff": s18[1], "cross_same": cross[0], "cross_diff": cross[1]}
    return result


def main():
    global SEED, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--seed-subdir", action="store_true", help="save artifacts under seed_<seed>/")
    args = ap.parse_args()
    SEED = args.seed
    if args.seed_subdir:
        OUT = OUT / f"seed_{SEED}"
    seed_all(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print(f"Stage 1 device={device}, epochs={args.epochs}, seed={SEED}")

    source, source_gt = utils.load_data_houston(str(ROOT / "datasets/Houston/Houston13.mat"), str(ROOT / "datasets/Houston/Houston13_7gt.mat"))
    target, target_gt = utils.load_data_houston(str(ROOT / "datasets/Houston/Houston18.mat"), str(ROOT / "datasets/Houston/Houston18_7gt.mat"))
    source_pca, pca = fit_pca(source)
    target_pca = pca.transform(target.reshape(-1, target.shape[-1]).astype(np.float32)).reshape(target.shape[0], target.shape[1], PCA_COMPONENTS).astype(np.float32)
    train_ids, val_ids = fixed_source_split(source_gt)
    train_x, train_y = make_center_patches(source_pca, source_gt, train_ids)
    val_x, val_y = make_center_patches(source_pca, source_gt, val_ids)
    print(f"source split: train={len(train_y)} ({TRAIN_PER_CLASS}/class), val={len(val_y)}, classes={np.bincount(train_y).tolist()}")
    np.savez(OUT / "houston13_source_split.npz", train_indices=train_ids, val_indices=val_ids)
    with open(OUT / "houston13_source_pca.pkl", "wb") as f:
        pickle.dump(pca, f)

    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)), batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y)), batch_size=args.batch_size, shuffle=False)
    teacher = SSFusionFramework(img_size=IMG_SIZE, in_channels=PCA_COMPONENTS, patch_size=PATCH_SIZE, classes=NUM_CLASSES, model_size="base")
    _load_official_transfer(teacher, "spat", WEIGHT_DIR / "spat-vit-base-ultra-checkpoint-1599.pth")
    _load_official_transfer(teacher, "spec", WEIGHT_DIR / "spec-vit-base-ultra-checkpoint-1599.pth")
    teacher.to(device)
    module_trainable_stage1(teacher)
    frozen_blocks = sum(not p.requires_grad for n, p in teacher.named_parameters() if "blocks." in n)
    total_blocks = sum("blocks." in n for n, _ in teacher.named_parameters())
    print(f"Stage 1 frozen transformer parameters: {frozen_blocks}/{total_blocks}")

    trainable = [p for p in teacher.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=6e-5, betas=(0.9, 0.999), weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.CrossEntropyLoss()
    history, best = [], {"val_acc": -1.0}
    for epoch in range(1, args.epochs + 1):
        teacher.train()
        # Keep frozen transformer blocks in eval mode so no stochastic layers
        # can change their output during adapter/head warm-up.
        teacher.spat_encoder.blocks.eval()
        teacher.spec_encoder.blocks.eval()
        total, correct, loss_sum = 0, 0, 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = teacher(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)
        train_loss, train_acc = loss_sum / total, correct / total
        val_loss, val_acc = evaluate(teacher, val_loader, device)
        scheduler.step()
        row = {"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc, "lr": scheduler.get_last_lr()[0]}
        history.append(row)
        print(json.dumps(row))
        if val_acc > best["val_acc"]:
            best = {**row}
            torch.save({"model": teacher.state_dict(), "best": best, "stage": 1, "seed": SEED, "img_size": IMG_SIZE, "patch_size": PATCH_SIZE, "pca_components": PCA_COMPONENTS, "classes": NUM_CLASSES}, OUT / "houston13_vitb_33x33_pca30_stage1_best.pth")

    with open(OUT / "houston13_vitb_33x33_pca30_stage1_history.json", "w") as f:
        json.dump(history, f, indent=2)
    config = {"stage": 1, "seed": SEED, "source": "Houston13", "target_gt_used_for_training_or_selection": False, "target_gt_used_for_offline_feature_probe": True, "img_size": IMG_SIZE, "patch_size": PATCH_SIZE, "pca_components": PCA_COMPONENTS, "train_per_class": TRAIN_PER_CLASS, "num_classes": NUM_CLASSES, "best": best, "checkpoint_load": load_stats(teacher)}
    with open(OUT / "houston13_vitb_33x33_pca30_stage1_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"BEST_SOURCE_VAL: {json.dumps(best)}")

    teacher.load_state_dict(torch.load(OUT / "houston13_vitb_33x33_pca30_stage1_best.pth", map_location=device)["model"])
    probe = quality_probe(teacher, source_pca, source_gt, target_pca, target_gt, device)
    with open(OUT / "houston13_vitb_33x33_pca30_stage1_feature_probe.json", "w") as f:
        json.dump(probe, f, indent=2)
    print("FEATURE_PROBE:")
    print(json.dumps(probe, indent=2))


if __name__ == "__main__":
    main()

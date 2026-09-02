"""Houston HyperSIGMA Stage-1 under the fixed spatially-disjoint protocol.

Run with ``--bands 48`` for the formal teacher. ``--bands 30`` is a matched
PCA30 ablation. Both use identical source centers, architecture, optimizer,
and training epochs. Houston18 is loaded without labels during training; its
GT is read only after the selected source-validation checkpoint is fixed.
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
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "HyperSIGMA" / "ImageClassification"))
import utils  # noqa: E402
from hypersigma_teacher_smoke_test import SSFusionFramework, _checkpoint_state  # noqa: E402

WEIGHTS = Path("/nas1/zhangzj26/HyperSIGMA_weights")
OUT_ROOT = Path("/nas1/zhangzj26/HyperSIGMA_adapted/protocol_stage1")
SEED = 1174
CLASSES = 7
TRAIN_PER_CLASS = 180
VAL_PER_CLASS = 30
IMG_SIZE = 33
HALF = IMG_SIZE // 2

# Fixed class-wise regions. A 33-pixel empty band separates train and val
# pools, and the global distance check below catches cross-class overlaps.
REGIONS = {
    1: (1, 498, "low"), 2: (1, 800, "high"), 3: (0, 89, "high"),
    4: (1, 505, "low"), 5: (1, 213, "high"), 6: (0, 119, "high"),
    7: (1, 648, "low"),
}


def seed_all(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def disjoint_split(gt):
    rng = np.random.RandomState(SEED)
    train, val = [], []
    val_pools = {}
    for cls in range(1, CLASSES + 1):
        points = np.argwhere(gt == cls)
        axis, cut, side = REGIONS[cls]
        if side == "low":
            train_pool = points[points[:, axis] <= cut - HALF]
            val_pool = points[points[:, axis] >= cut + HALF + 1]
        else:
            train_pool = points[points[:, axis] >= cut + HALF + 1]
            val_pool = points[points[:, axis] <= cut - HALF]
        if len(train_pool) < TRAIN_PER_CLASS or len(val_pool) < VAL_PER_CLASS:
            raise RuntimeError(f"class {cls}: train_pool={len(train_pool)}, val_pool={len(val_pool)}")
        train.extend(train_pool[rng.permutation(len(train_pool))[:TRAIN_PER_CLASS]].tolist())
        val_pools[cls] = val_pool[rng.permutation(len(val_pool))]
    train = np.asarray(train, np.int64)
    # Select validation points only after seeing every training center, so
    # cross-class train/val context overlap is excluded as well.
    for cls in range(1, CLASSES + 1):
        candidates = val_pools[cls]
        nearest = np.max(np.abs(candidates[:, None, :] - train[None, :, :]), axis=2).min(axis=1)
        selected = candidates[nearest >= IMG_SIZE]
        if len(selected) < VAL_PER_CLASS:
            raise RuntimeError(f"class {cls}: only {len(selected)} globally disjoint validation centers")
        val.extend(selected[:VAL_PER_CLASS].tolist())
    val = np.asarray(val, np.int64)
    nearest = np.max(np.abs(val[:, None, :] - train[None, :, :]), axis=2).min(axis=1)
    if nearest.min() < IMG_SIZE:
        raise RuntimeError(f"train/val context overlap, minimum Chebyshev distance={nearest.min()}")
    return train, val, int(nearest.min())


def source_fit_transform(source, target, bands):
    if bands == 48:
        return source.astype(np.float32), target.astype(np.float32), None
    pca = PCA(n_components=bands, whiten=True, random_state=SEED)
    src = pca.fit_transform(source.reshape(-1, source.shape[-1]).astype(np.float32)).reshape(source.shape[0], source.shape[1], bands)
    tgt = pca.transform(target.reshape(-1, target.shape[-1]).astype(np.float32)).reshape(target.shape[0], target.shape[1], bands)
    return src.astype(np.float32), tgt.astype(np.float32), pca


def patches(cube, centers, bands):
    padded = np.pad(cube, ((HALF, HALF), (HALF, HALF), (0, 0)), mode="constant")
    out = np.empty((len(centers), bands, IMG_SIZE, IMG_SIZE), np.float32)
    for i, (r, c) in enumerate(centers):
        out[i] = padded[r:r + IMG_SIZE, c:c + IMG_SIZE].transpose(2, 0, 1)
    return out


def interpolate_pos(value, target_shape, spectral=False):
    if tuple(value.shape) == tuple(target_shape):
        return value
    if spectral:
        # Spectral position tokens are a 1-D sequence (the MAE checkpoint has
        # 100 tokens and no class token).
        return F.interpolate(value.transpose(1, 2), size=target_shape[1], mode="linear", align_corners=False).transpose(1, 2)
    old_tokens, dim = value.shape[1], value.shape[2]
    old_size, new_size = int(old_tokens ** 0.5), int(target_shape[1] ** 0.5)
    image = value.reshape(1, old_size, old_size, dim).permute(0, 3, 1, 2)
    image = F.interpolate(image, size=(new_size, new_size), mode="bicubic", align_corners=False)
    return image.permute(0, 2, 3, 1).flatten(1, 2)


def load_pretrained(teacher):
    stats = {}
    for branch, filename, fragments in (
        ("spatial", "spat-vit-base-ultra-checkpoint-1599.pth", ("patch_embed.proj",)),
        ("spectral", "spec-vit-base-ultra-checkpoint-1599.pth", ("patch_embed", "spat_map")),
    ):
        state = _checkpoint_state(WEIGHTS / filename)
        module = teacher.spat_encoder if branch == "spatial" else teacher.spec_encoder
        target = module.state_dict()
        filtered = {k: v for k, v in state.items() if not any(f in k for f in fragments)}
        if "pos_embed" in filtered and "pos_embed" in target:
            filtered["pos_embed"] = interpolate_pos(filtered["pos_embed"], target["pos_embed"].shape, spectral=(branch == "spectral"))
        compatible = {k: v for k, v in filtered.items() if k in target and tuple(v.shape) == tuple(target[k].shape)}
        module.load_state_dict(compatible, strict=False)
        stats[branch] = {"checkpoint_entries": len(state), "filtered_entries": len(filtered), "loaded_entries": len(compatible), "target_entries": len(target), "random_or_unloaded": len(target) - len(compatible), "pos_embed_interpolated_or_loaded": "pos_embed" in compatible}
        print(f"{branch}: loaded {len(compatible)}/{len(target)}, pos_embed={stats[branch]['pos_embed_interpolated_or_loaded']}")
    return stats


def set_stage1(teacher):
    for name, parameter in teacher.named_parameters():
        parameter.requires_grad_(".blocks." not in name)


def forward_parts(teacher, x):
    b = x.shape[0]
    spat = teacher.spat_encoder(x)
    spec = teacher.spec_encoder(x)[0]
    pooled = teacher.pool(spec).view(b, -1)
    weights = [layer(pooled).view(b, -1, 1, 1) for layer in (teacher.fc_spec1, teacher.fc_spec2, teacher.fc_spec3, teacher.fc_spec4)]
    reduced = [layer(f) for layer, f in zip((teacher.DR1, teacher.DR2, teacher.DR3, teacher.DR4), spat)]
    sem = torch.cat([F.adaptive_avg_pool2d((1 + w) * f, 1).flatten(1) for w, f in zip(weights, reduced)], dim=1)
    return spat, spec, sem, teacher.classifier(sem)


def evaluate(teacher, loader, device):
    teacher.eval(); ce = nn.CrossEntropyLoss(); loss_sum = correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            logits = teacher(x.to(device)); y = y.to(device)
            loss_sum += ce(logits, y).item() * len(y); correct += (logits.argmax(1) == y).sum().item(); total += len(y)
    return loss_sum / total, correct / total


def probe(teacher, src_cube, src_gt, tgt_cube, tgt_gt, device, bands):
    rng = np.random.RandomState(20260901)
    def collect(cube, gt):
        centers = []
        for cls in range(1, CLASSES + 1):
            ids = np.flatnonzero(gt.reshape(-1) == cls)
            centers.extend([np.unravel_index(int(i), gt.shape) for i in rng.permutation(ids)[:100]])
        x = patches(cube, centers, bands); y = np.asarray([int(gt[r, c]) - 1 for r, c in centers])
        out = {"spatial": [], "spectral": [], "sem": []}
        with torch.no_grad():
            for i in range(0, len(x), 64):
                spat, spec, sem, _ = forward_parts(teacher, torch.from_numpy(x[i:i + 64]).to(device))
                out["spatial"].append(F.adaptive_avg_pool2d(spat[-1], 1).flatten(1).cpu().numpy())
                out["spectral"].append(spec.mean(1).cpu().numpy()); out["sem"].append(sem.cpu().numpy())
        return {k: np.concatenate(v) for k, v in out.items()}, y
    def sim(a, ya, b=None, yb=None):
        cross = b is not None
        if b is None: b, yb = a, ya
        aa = F.normalize(torch.from_numpy(a), dim=1).numpy(); bb = F.normalize(torch.from_numpy(b), dim=1).numpy(); matrix = aa @ bb.T
        same = ya[:, None] == yb[None, :]
        if not cross: np.fill_diagonal(same, False); np.fill_diagonal(matrix, np.nan)
        return float(np.nanmean(matrix[same])), float(np.nanmean(matrix[~same]))
    teacher.eval(); src, sy = collect(src_cube, src_gt); tgt, ty = collect(tgt_cube, tgt_gt); result = {}
    for key in ("spatial", "spectral", "sem"):
        a, b = sim(src[key], sy), sim(tgt[key], ty); c = sim(src[key], sy, tgt[key], ty)
        result[key] = {"source_same": a[0], "source_diff": a[1], "target_same": b[0], "target_diff": b[1], "cross_same": c[0], "cross_diff": c[1]}
    return result


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--bands", type=int, choices=(30, 48), required=True); ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu"); ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args(); seed_all(SEED); device = torch.device(args.device); out = OUT_ROOT / f"bands{args.bands}"; out.mkdir(parents=True, exist_ok=True)
    source, src_gt = utils.load_data_houston(str(ROOT / "datasets/Houston/Houston13.mat"), str(ROOT / "datasets/Houston/Houston13_7gt.mat")); target = hdf5storage.loadmat(str(ROOT / "datasets/Houston/Houston18.mat"))["ori_data"]
    src_cube, tgt_cube, pca = source_fit_transform(source, target, args.bands); train_centers, val_centers, min_distance = disjoint_split(src_gt)
    train_x, val_x = patches(src_cube, train_centers, args.bands), patches(src_cube, val_centers, args.bands); train_y = src_gt[train_centers[:, 0], train_centers[:, 1]].astype(np.int64) - 1; val_y = src_gt[val_centers[:, 0], val_centers[:, 1]].astype(np.int64) - 1
    np.savez(out / "source_split.npz", train_centers=train_centers, val_centers=val_centers)
    if pca is not None:
        with open(out / "source_pca.pkl", "wb") as f: pickle.dump(pca, f)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y)), batch_size=args.batch_size, shuffle=True); val_loader = DataLoader(TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y)), batch_size=args.batch_size)
    teacher = SSFusionFramework(img_size=IMG_SIZE, in_channels=args.bands, patch_size=2, classes=CLASSES, model_size="base"); load_stats = load_pretrained(teacher); teacher.to(device); set_stage1(teacher)
    opt = torch.optim.AdamW([p for p in teacher.parameters() if p.requires_grad], lr=6e-5, betas=(0.9, 0.999), weight_decay=0.05); scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs); ce = nn.CrossEntropyLoss(); history = []; best = {"val_acc": -1.0}
    print(f"bands={args.bands} train={len(train_y)} val={len(val_y)} min_center_distance={min_distance} target_gt_used=False")
    for epoch in range(1, args.epochs + 1):
        teacher.train(); teacher.spat_encoder.blocks.eval(); teacher.spec_encoder.blocks.eval(); total = correct = 0; loss_sum = 0.0
        for x, y in train_loader:
            logits = teacher(x.to(device)); y = y.to(device); loss = ce(logits, y); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); loss_sum += loss.item() * len(y); correct += (logits.argmax(1) == y).sum().item(); total += len(y)
        val_loss, val_acc = evaluate(teacher, val_loader, device); scheduler.step(); row = {"epoch": epoch, "train_loss": loss_sum / total, "train_acc": correct / total, "val_loss": val_loss, "val_acc": val_acc, "lr": scheduler.get_last_lr()[0]}; history.append(row); print(json.dumps(row))
        if val_acc > best["val_acc"]:
            best = row.copy(); torch.save({"model": teacher.state_dict(), "best": best, "bands": args.bands, "img_size": IMG_SIZE, "patch_size": 2, "classes": CLASSES, "seed": SEED}, out / "stage1_best.pth")
    (out / "history.json").write_text(json.dumps(history, indent=2)); config = {"bands": args.bands, "pca_components": args.bands if pca is not None else None, "img_size": IMG_SIZE, "patch_size": 2, "classes": CLASSES, "seed": SEED, "train_per_class": TRAIN_PER_CLASS, "val_per_class": VAL_PER_CLASS, "min_train_val_center_chebyshev_distance": min_distance, "target_gt_used_for_training_or_selection": False, "checkpoint_load": load_stats, "best": best}; (out / "config.json").write_text(json.dumps(config, indent=2))
    teacher.load_state_dict(torch.load(out / "stage1_best.pth", map_location=device)["model"]); metrics = probe(teacher, src_cube, src_gt, tgt_cube, hdf5storage.loadmat(str(ROOT / "datasets/Houston/Houston18_7gt.mat"))["map"], device, args.bands); (out / "feature_probe.json").write_text(json.dumps(metrics, indent=2)); print("FEATURE_PROBE", json.dumps(metrics, indent=2))


if __name__ == "__main__": main()

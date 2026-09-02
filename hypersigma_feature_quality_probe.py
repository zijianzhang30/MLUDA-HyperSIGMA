"""Offline HyperSIGMA feature quality and checkpoint provenance probe."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "HyperSIGMA" / "ImageClassification"))
from config_Houston import HalfWidth  # noqa: E402
import utils  # noqa: E402
from hypersigma_teacher_smoke_test import (  # noqa: E402
    WEIGHT_DIR, SSFusionFramework, _load_official_transfer,
)


def make_samples(data, labels, max_per_class):
    padded = np.pad(data, ((HalfWidth, HalfWidth),) * 2 + ((0, 0),), mode="constant")
    xs, ys = [], []
    rng = np.random.default_rng(20260901)
    for cls in range(1, int(labels.max()) + 1):
        rows, cols = np.where(labels == cls)
        order = rng.permutation(len(rows))[:max_per_class]
        for i in order:
            r, c = rows[i], cols[i]
            patch = padded[r:r + 2 * HalfWidth + 1, c:c + 2 * HalfWidth + 1]
            xs.append(np.transpose(patch, (2, 0, 1)))
            ys.append(cls - 1)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64)


def cosine_matrix_stats(a, ya, b=None, yb=None):
    """Return mean cosine for same and different labels, excluding diagonal."""
    if b is None:
        b, yb = a, ya
        same_domain = True
    else:
        same_domain = False
    a = F.normalize(torch.from_numpy(a), dim=1).numpy()
    b = F.normalize(torch.from_numpy(b), dim=1).numpy()
    vals = a @ b.T
    same = ya[:, None] == yb[None, :]
    if same_domain:
        np.fill_diagonal(same, False)
        np.fill_diagonal(vals, np.nan)
    different = ~same
    return float(np.nanmean(vals[same])), float(np.nanmean(vals[different]))


def provenance(teacher):
    print("\nCHECKPOINT PROVENANCE (loaded means shape-compatible checkpoint tensors)")
    for branch, module in (("spatial", teacher.spat_encoder), ("spectral", teacher.spec_encoder)):
        # Recompute the exact transfer set to classify each module.
        path = WEIGHT_DIR / ("spat-vit-base-ultra-checkpoint-1599.pth" if branch == "spatial" else "spec-vit-base-ultra-checkpoint-1599.pth")
        from hypersigma_teacher_smoke_test import _checkpoint_state
        state = _checkpoint_state(path)
        fragments = ("patch_embed.proj", "spat_map", "spat_output_maps", "pos_embed") if branch == "spatial" else ("patch_embed", "spat_map", "fpn1.0.weight")
        filtered = {k: v for k, v in state.items() if not any(f in k for f in fragments)}
        target = module.state_dict()
        loaded = {k for k, v in filtered.items() if k in target and tuple(v.shape) == tuple(target[k].shape)}
        groups = {
            "patch_embed": lambda k: k.startswith("patch_embed"),
            "transformer_blocks": lambda k: k.startswith("blocks."),
            "projection_layers": lambda k: any(k.startswith(x) for x in ("fpn", "l1", "norm", "pos_embed")),
        }
        print(f"{branch}:")
        for name, pred in groups.items():
            keys = [k for k in target if pred(k)]
            n = sum(k in loaded for k in keys)
            print(f"  {name}: loaded {n}/{len(keys)}, random_or_unloaded {len(keys)-n}")
    print("fusion/SEM (DR1-4, fc_spec1-4, conv_features, conv, classifier): loaded 0; all randomly initialized")
    print("task-head pre-feature (512-d): depends on frozen backbone plus RANDOM SEM/DR/fc_spec layers")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-per-class", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    device = torch.device(args.device)
    teacher = SSFusionFramework(img_size=7, in_channels=48, patch_size=1, classes=8, model_size="base")
    _load_official_transfer(teacher, "spat", WEIGHT_DIR / "spat-vit-base-ultra-checkpoint-1599.pth")
    _load_official_transfer(teacher, "spec", WEIGHT_DIR / "spec-vit-base-ultra-checkpoint-1599.pth")
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    provenance(teacher)

    feats: Dict[str, Dict[str, np.ndarray]] = {"Houston13": {}, "Houston18": {}}
    labels: Dict[str, np.ndarray] = {}
    for domain in ("Houston13", "Houston18"):
        data, gt = utils.load_data_houston(str(ROOT / "datasets/Houston" / f"{domain}.mat"), str(ROOT / "datasets/Houston" / f"{domain}_7gt.mat"))
        x, y = make_samples(data, gt, args.max_per_class)
        labels[domain] = y
        collected = {"spatial": [], "spectral": [], "sem": []}
        with torch.no_grad():
            for start in range(0, len(x), args.batch_size):
                xb = torch.from_numpy(x[start:start + args.batch_size]).to(device)
                spat = teacher.spat_encoder(xb)
                spec = teacher.spec_encoder(xb)[0]
                b = xb.shape[0]
                pooled_spat = F.adaptive_avg_pool2d(spat[-1], 1).flatten(1)
                pooled_spec = spec.mean(dim=1)
                spec_pool = teacher.pool(spec).view(b, -1)
                weights = [l(spec_pool).view(b, -1, 1, 1) for l in (teacher.fc_spec1, teacher.fc_spec2, teacher.fc_spec3, teacher.fc_spec4)]
                reduced = [l(f) for l, f in zip((teacher.DR1, teacher.DR2, teacher.DR3, teacher.DR4), spat)]
                parts = [F.adaptive_avg_pool2d((1+w)*f, 1).flatten(1) for w, f in zip(weights, reduced)]
                sem = torch.cat(parts, dim=1)
                collected["spatial"].append(pooled_spat.cpu().numpy())
                collected["spectral"].append(pooled_spec.cpu().numpy())
                collected["sem"].append(sem.cpu().numpy())
        for key in collected:
            feats[domain][key] = np.concatenate(collected[key])
        print(f"{domain}: extracted {len(y)} samples; spatial={feats[domain]['spatial'].shape[1]}, spectral={feats[domain]['spectral'].shape[1]}, SEM={feats[domain]['sem'].shape[1]}")

    print("\nCOSINE QUALITY PROBE (means; higher same-class and lower different-class are preferable)")
    for key in ("spatial", "spectral", "sem"):
        intra13 = cosine_matrix_stats(feats["Houston13"][key], labels["Houston13"])
        intra18 = cosine_matrix_stats(feats["Houston18"][key], labels["Houston18"])
        cross_same, cross_diff = cosine_matrix_stats(feats["Houston13"][key], labels["Houston13"], feats["Houston18"][key], labels["Houston18"])
        print(f"{key}: H13 intra_same={intra13[0]:.4f}, intra_diff={intra13[1]:.4f}; H18 intra_same={intra18[0]:.4f}, intra_diff={intra18[1]:.4f}; cross_same={cross_same:.4f}, cross_diff={cross_diff:.4f}")

    print("\nMLUDA baseline pre-classifier feature: NOT AVAILABLE for offline extraction.")
    print("No trained DSANSS checkpoint is present in the project/NAS paths; creating a random DSANSS would not be a valid baseline comparison.")
    print("TARGET RECOMMENDATION: defer SEM-512 KD until SEM/DR/fc_spec are trained or replaced; use final spatial backbone feature as the cleanest currently pretrained target.")


if __name__ == "__main__":
    main()

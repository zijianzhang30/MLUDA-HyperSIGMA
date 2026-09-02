"""Independent HyperSIGMA teacher smoke test on native Houston patches.

This script does not import or modify the MLUDA training entry points.  It
keeps the native MLUDA tensor layout (B, 48, 7, 7) and only adapts the
HyperSIGMA constructor enough to determine whether its forward graph can
consume that tensor.  Pretrained checkpoint transfer follows the official
ImageClassification notebook's prefix/filter/shape matching pattern.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
HS_MODEL_DIR = ROOT / "third_party" / "HyperSIGMA" / "ImageClassification"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HS_MODEL_DIR))

from config_Houston import HalfWidth  # noqa: E402
import utils  # noqa: E402
from model.ss_fusion_cls import SSFusionFramework  # noqa: E402


WEIGHT_DIR = Path("/nas1/zhangzj26/HyperSIGMA_weights")


def _checkpoint_state(path: Path) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(str(path), map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state = checkpoint
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint at {path}: {type(state)!r}")
    if state and next(iter(state)).startswith("module."):
        state = {k[7:]: v for k, v in state.items()}
    if state and sorted(state)[0].startswith("encoder"):
        state = {k.replace("encoder.", "", 1): v for k, v in state.items() if k.startswith("encoder.")}
    return state


def _load_official_transfer(model: SSFusionFramework, branch: str, path: Path) -> None:
    """Replicate the official classification notebook's branch transfer."""
    state = _checkpoint_state(path)
    if branch == "spat":
        # The official notebook removes these MAE/classification-transfer keys.
        remove_fragments = ("patch_embed.proj", "spat_map", "spat_output_maps", "pos_embed")
        target = model.spat_encoder
    elif branch == "spec":
        remove_fragments = ("patch_embed", "spat_map", "fpn1.0.weight")
        target = model.spec_encoder
    else:
        raise ValueError(branch)

    filtered = {
        k: v for k, v in state.items() if not any(fragment in k for fragment in remove_fragments)
    }
    target_state = target.state_dict()
    compatible = {k: v for k, v in filtered.items() if k in target_state and tuple(v.shape) == tuple(target_state[k].shape)}
    skipped = sorted(set(filtered) - set(compatible))
    missing, unexpected = target.load_state_dict(compatible, strict=False)
    print(
        f"{branch} checkpoint: entries={len(state)}, loaded={len(compatible)}, "
        f"missing_after_transfer={len(missing)}, unexpected={len(unexpected)}, "
        f"shape_or_filter_skipped={len(skipped)}"
    )
    if skipped:
        print(f"  skipped examples ({branch}): {skipped[:8]}")


def _native_samples(data: np.ndarray, labels: np.ndarray, count: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Extract non-background patches exactly as utils.get_all_data does."""
    rows, cols = np.nonzero(labels)
    if len(rows) < count:
        raise ValueError("Houston label map contains fewer non-background pixels than requested")
    padded = np.pad(data, ((HalfWidth, HalfWidth), (HalfWidth, HalfWidth), (0, 0)), mode="constant")
    patches = []
    ys = []
    for row, col in zip(rows[:count], cols[:count]):
        patch = padded[row : row + 2 * HalfWidth + 1, col : col + 2 * HalfWidth + 1, :]
        patches.append(np.transpose(patch, (2, 0, 1)))
        ys.append(int(labels[row, col]))
    return np.asarray(patches, dtype=np.float32), np.asarray(ys, dtype=np.int64)


def _finite(name: str, value) -> bool:
    tensors: Iterable[torch.Tensor]
    if isinstance(value, (tuple, list)):
        tensors = value
    else:
        tensors = (value,)
    ok = all(torch.isfinite(t).all().item() for t in tensors)
    print(f"{name} finite (no NaN/Inf): {ok}")
    return ok


def _forward_features(teacher: SSFusionFramework, x: torch.Tensor):
    """Run the official SSFusion SEM path while exposing its distillation point."""
    b = x.shape[0]
    spatial = teacher.spat_encoder(x)
    spectral = teacher.spec_encoder(x)[0]
    spectral_pooled = teacher.pool(spectral).view(b, -1)
    spec_weights = [layer(spectral_pooled).view(b, -1, 1, 1) for layer in (
        teacher.fc_spec1, teacher.fc_spec2, teacher.fc_spec3, teacher.fc_spec4)]
    spatial_reduced = [layer(feat) for layer, feat in zip((teacher.DR1, teacher.DR2, teacher.DR3, teacher.DR4), spatial)]
    sem_parts = [(1 + weight) * feat for weight, feat in zip(spec_weights, spatial_reduced)]
    sem_parts = [torch.nn.functional.avg_pool2d(part, part.shape[2:]).view(b, 128, -1) for part in sem_parts]
    sem = torch.cat(sem_parts, dim=1).unsqueeze(2)  # official task-head input before flatten
    task_head_feature = sem.view(b, -1)
    logits = teacher.classifier(task_head_feature)
    return spatial, spectral, sem, task_head_feature, logits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--samples-per-domain", type=int, default=3)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"device: {device}")
    print(f"HyperSIGMA weights: {WEIGHT_DIR}")
    print(f"MLUDA preprocessing: HalfWidth={HalfWidth}, patch_size={2 * HalfWidth + 1}, layout=(B,C,H,W)")

    sample_batches = []
    for domain in ("Houston13", "Houston18"):
        data_path = ROOT / "datasets" / "Houston" / f"{domain}.mat"
        label_path = ROOT / "datasets" / "Houston" / f"{domain}_7gt.mat"
        data, labels = utils.load_data_houston(str(data_path), str(label_path))
        print(f"{domain}: cube_shape={data.shape}, labels_shape={labels.shape}, dtype={data.dtype}, "
              f"value_range=({float(np.min(data)):.6g},{float(np.max(data)):.6g}), "
              f"classes={sorted(int(v) for v in np.unique(labels) if v != 0)}")
        patches, ys = _native_samples(data, labels, args.samples_per_domain)
        print(f"{domain}: selected input shape={patches.shape}, labels={ys.tolist()}")
        sample_batches.append(patches)

    inputs = torch.from_numpy(np.concatenate(sample_batches, axis=0)).to(device)
    if inputs.shape[1:] != (48, 7, 7):
        raise RuntimeError(f"Unexpected native Houston input shape: {tuple(inputs.shape)}")

    # img_size=7 and patch_size=1 are diagnostic constructor values: patch_size=16
    # from the official spatial setup cannot be applied to a 7x7 native patch.
    teacher = SSFusionFramework(img_size=7, in_channels=48, patch_size=1, classes=8, model_size="base")
    _load_official_transfer(teacher, "spat", WEIGHT_DIR / "spat-vit-base-ultra-checkpoint-1599.pth")
    _load_official_transfer(teacher, "spec", WEIGHT_DIR / "spec-vit-base-ultra-checkpoint-1599.pth")
    teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    print(f"teacher frozen: {all(not p.requires_grad for p in teacher.parameters())}; training={teacher.training}")

    with torch.no_grad():
        spatial, spectral, sem, task_head_feature, logits = _forward_features(teacher, inputs)

    print(f"input shape: {tuple(inputs.shape)}")
    print(f"spatial feature shapes: {[tuple(t.shape) for t in spatial]}")
    print(f"spectral feature shape: {tuple(spectral.shape)}")
    print(f"SEM fused feature shape: {tuple(sem.shape)}")
    print(f"task-head pre-feature shape: {tuple(task_head_feature.shape)}")
    print(f"task-head logits shape (smoke only): {tuple(logits.shape)}")
    all_finite = (
        _finite("input", inputs)
        and _finite("spatial", spatial)
        and _finite("spectral", spectral)
        and _finite("SEM", sem)
        and _finite("task-head feature", task_head_feature)
        and _finite("logits", logits)
    )
    print(f"OVERALL_SMOKE_PASS: {all_finite}")
    if device.type == "cuda":
        print(f"CUDA peak allocated: {torch.cuda.max_memory_allocated(device) / 1024**3:.3f} GiB")


if __name__ == "__main__":
    main()

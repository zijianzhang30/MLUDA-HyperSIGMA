"""Precompute Full48 HyperSIGMA F_spec for MLUDA center pixels.

This is a frozen-teacher cache only.  Labels are used to reproduce MLUDA's
non-background sample universe and are not stored in the training batches or
used by any KD loss.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import hdf5storage
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "third_party/HyperSIGMA/ImageClassification"))
import utils  # noqa: E402
from hypersigma_stage1_protocol import HALF, IMG_SIZE  # noqa: E402
from hypersigma_stage1_protocol import forward_parts  # noqa: E402
from hypersigma_teacher_smoke_test import SSFusionFramework  # noqa: E402

CKPT = Path("/nas1/zhangzj26/HyperSIGMA_adapted/protocol_stage1/bands48/stage1_best.pth")
OUT = Path("/nas1/zhangzj26/HyperSIGMA_adapted/mluda_fspec_full48_cache.npz")


def extract(model, cube, centers, device, batch_size):
    padded = np.pad(cube, ((HALF, HALF), (HALF, HALF), (0, 0)), mode="constant")
    result = []
    with torch.no_grad():
        for start in range(0, len(centers), batch_size):
            cs = centers[start:start + batch_size]
            xb = np.empty((len(cs), cube.shape[-1], IMG_SIZE, IMG_SIZE), np.float32)
            for j, (r, c) in enumerate(cs):
                xb[j] = padded[r:r + IMG_SIZE, c:c + IMG_SIZE].transpose(2, 0, 1)
            _, spec, _, _ = forward_parts(model, torch.from_numpy(xb).to(device))
            result.append(spec.mean(1).cpu().numpy().astype(np.float32))
    return np.concatenate(result)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu"); ap.add_argument("--batch-size", type=int, default=64); ap.add_argument("--output", type=Path, default=OUT)
    args = ap.parse_args(); device = torch.device(args.device)
    src, src_gt = utils.load_data_houston(str(ROOT / "datasets/Houston/Houston13.mat"), str(ROOT / "datasets/Houston/Houston13_7gt.mat"))
    tgt, tgt_gt = utils.load_data_houston(str(ROOT / "datasets/Houston/Houston18.mat"), str(ROOT / "datasets/Houston/Houston18_7gt.mat"))
    src_centers = np.argwhere(src_gt > 0).astype(np.int64)
    tgt_centers = np.argwhere(tgt_gt > 0).astype(np.int64)
    model = SSFusionFramework(img_size=IMG_SIZE, in_channels=48, patch_size=2, classes=7, model_size="base")
    ck = torch.load(CKPT, map_location="cpu"); model.load_state_dict(ck["model"], strict=True); model.to(device).eval()
    for p in model.parameters(): p.requires_grad_(False)
    print(f"teacher frozen={all(not p.requires_grad for p in model.parameters())} source={len(src_centers)} target={len(tgt_centers)}")
    src_f = extract(model, src, src_centers, device, args.batch_size); tgt_f = extract(model, tgt, tgt_centers, device, args.batch_size)
    if not np.isfinite(src_f).all() or not np.isfinite(tgt_f).all(): raise RuntimeError("teacher F_spec contains NaN/Inf")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, source_centers=src_centers, source_fspec=src_f, target_centers=tgt_centers, target_fspec=tgt_f)
    print(f"saved={args.output} source_fspec={src_f.shape} target_fspec={tgt_f.shape}")


if __name__ == "__main__": main()

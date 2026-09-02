"""Minimal Full48 HyperSIGMA F_spec distillation for the original MLUDA.

The MLUDA loss and data augmentations are kept intact.  KD is added only on
the 192-d spectral-only branch before concat/CrossAttention (MBCA), with a
192->128 student projection (HyperSIGMA's downstream F_spec dimension).
Target labels are never passed to a loss or
used for checkpoint selection; Houston18 metrics are computed after training.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import hdf5storage
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from config_Houston import (BATCH_SIZE, CLASS_NUM, HalfWidth, l2_decay, lr,
                            momentum, nBand, epochs, seeds)  # noqa: E402
from UtilsCMS import ILDA  # noqa: E402
import mmd  # noqa: E402
import utils  # noqa: E402
from contrastive_loss import SupConLoss  # noqa: E402
from net2 import DSANSS  # noqa: E402

HS_DIR = ROOT / "third_party" / "HyperSIGMA" / "ImageClassification"
sys.path.insert(0, str(HS_DIR))
from hypersigma_stage1_protocol import forward_parts  # noqa: E402
from hypersigma_teacher_smoke_test import SSFusionFramework  # noqa: E402

TEACHER_CKPT = Path("/nas1/zhangzj26/HyperSIGMA_adapted/protocol_stage1/bands48/stage1_best.pth")
CACHE = Path("/nas1/zhangzj26/HyperSIGMA_adapted/mluda_fspec_full48_cache.npz")
OUT_ROOT = Path("/nas1/zhangzj26/HyperSIGMA_adapted/mluda_fspec_kd")


def center_patches(cube, centers, width):
    half = width // 2
    padded = np.pad(cube, ((half, half), (half, half), (0, 0)), mode="constant")
    out = np.empty((len(centers), cube.shape[-1], width, width), dtype=np.float32)
    for i, (r, c) in enumerate(centers):
        out[i] = padded[r:r + width, c:c + width].transpose(2, 0, 1)
    return out


def paired_source_samples(adapted, raw, gt, seed):
    """Reproduce utils.get_sample_data's 180/class selection and ordering."""
    rng = np.random.RandomState(seed)
    padded_gt = np.pad(gt, HalfWidth, mode="constant")
    rows, cols = np.nonzero(padded_gt)
    train_indices, val_indices = [], []
    for cls in range(int(np.max(padded_gt))):
        indices = [j for j in range(len(rows)) if padded_gt[rows[j], cols[j]] == cls + 1]
        rng.shuffle(indices)
        train_indices += indices[:180]
        val_indices += indices[180:]
    rng.shuffle(train_indices); rng.shuffle(val_indices)
    train_c = np.asarray([(rows[j] - HalfWidth, cols[j] - HalfWidth) for j in train_indices], np.int64)
    val_c = np.asarray([(rows[j] - HalfWidth, cols[j] - HalfWidth) for j in val_indices], np.int64)
    train_x = center_patches(adapted, train_c, 7)
    val_x = center_patches(adapted, val_c, 7)
    train_teacher = center_patches(raw, train_c, 33)  # used only for verification/cache lookup
    val_teacher = center_patches(raw, val_c, 33)
    train_y = gt[train_c[:, 0], train_c[:, 1]].astype(np.int64) - 1
    val_y = gt[val_c[:, 0], val_c[:, 1]].astype(np.int64) - 1
    return train_c, train_x, train_teacher, train_y, val_c, val_x, val_teacher, val_y


def cosine_kd(projection, student, teacher):
    projected = F.normalize(projection(student), dim=1)
    target = F.normalize(teacher.detach(), dim=1)
    return (1.0 - (projected * target).sum(1)).mean()


def load_teacher(device):
    teacher = SSFusionFramework(img_size=33, in_channels=48, patch_size=2, classes=7, model_size="base")
    checkpoint = torch.load(TEACHER_CKPT, map_location="cpu")
    teacher.load_state_dict(checkpoint["model"], strict=True); teacher.to(device).eval()
    for parameter in teacher.parameters(): parameter.requires_grad_(False)
    return teacher


def teacher_features(teacher, x, device):
    with torch.no_grad():
        _, spec, _, _ = forward_parts(teacher, x.to(device))
        return spec.mean(1)


def eval_source(model, loader, device):
    model.eval(); ce = nn.CrossEntropyLoss(); loss_sum = correct = total = 0
    with torch.no_grad():
        # DSANSS needs both branches; using the same source view for the
        # second argument evaluates the first/source output only.
        for x, y in loader:
            out = model(x.to(device), x.to(device))[3]
            y = y.to(device); loss_sum += ce(out, y).item() * len(y); correct += (out.argmax(1) == y).sum().item(); total += len(y)
    return loss_sum / total, correct / total


def target_eval(model, target_x, target_y, source_ref, device):
    model.eval(); pred = []
    with torch.no_grad():
        for start in range(0, len(target_x), BATCH_SIZE):
            xb = target_x[start:start + BATCH_SIZE]
            ref = source_ref[:len(xb)]
            out = model(ref.to(device), xb.to(device))[8]
            pred.append(out.argmax(1).cpu().numpy())
    pred = np.concatenate(pred); y = target_y.astype(np.int64)
    cm = metrics.confusion_matrix(y, pred, labels=np.arange(CLASS_NUM)); pc = np.diag(cm) / np.maximum(cm.sum(1), 1)
    return {"n": int(len(y)), "oa": float(np.mean(y == pred)), "aa": float(np.mean(pc)), "kappa": float(metrics.cohen_kappa_score(y, pred, labels=np.arange(CLASS_NUM))), "per_class_accuracy": pc.tolist(), "prediction_distribution": np.bincount(pred, minlength=CLASS_NUM).tolist(), "confusion_matrix": cm.tolist()}


def run_seed(lam, seed, data_s, data_t, label_s, target_centers, source_map, teacher_cache, device, out, num_epochs):
    utils.set_seed(seed)
    train_c, train_x, _, train_y, val_c, val_x, _, val_y = paired_source_samples(data_s, data_s, label_s, seed)
    # Teacher cache is indexed by source center, generated from raw Full48.
    train_tf = np.stack([teacher_cache[int(r), int(c)] for r, c in train_c]).astype(np.float32)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y)), batch_size=BATCH_SIZE, shuffle=False)
    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_tf), torch.from_numpy(train_y)), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    target_x = center_patches(data_t, target_centers, 7)
    target_tf = np.stack([teacher_cache["target"][int(r), int(c)] for r, c in target_centers]).astype(np.float32)
    target_loader = DataLoader(TensorDataset(torch.from_numpy(target_x), torch.from_numpy(target_tf)), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    model = DSANSS(nBand, 7, CLASS_NUM).to(device)
    projection = nn.Linear(192, 128).to(device)
    ce = nn.CrossEntropyLoss().to(device); con_s = SupConLoss(temperature=0.1).to(device); con_t = SupConLoss(temperature=0.1).to(device); dsh = utils.Domain_Occ_loss().to(device)
    history = []; best = {"val_acc": -1.0}; source_ref = torch.from_numpy(train_x[:BATCH_SIZE])
    for epoch in range(1, num_epochs + 1):
        model.train(); projection.train();
        LEARNING_RATE = lr / math.pow((1 + 10 * (epoch - 1) / num_epochs), 0.75)
        params = [{"params": model.feature_layers.parameters()}, {"params": model.fc1.parameters(), "lr": LEARNING_RATE}, {"params": model.fc2.parameters(), "lr": LEARNING_RATE}, {"params": model.head1.parameters(), "lr": LEARNING_RATE}, {"params": model.head2.parameters(), "lr": LEARNING_RATE}]
        if lam > 0: params.append({"params": projection.parameters(), "lr": LEARNING_RATE})
        optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=momentum, weight_decay=l2_decay)
        target_iter = iter(target_loader); sums = {k: 0.0 for k in ("total", "cls", "scl", "lmmd", "kd")}; total = correct = 0
        for source_data, source_teacher, source_label in train_loader:
            try: target_data, target_teacher = next(target_iter)
            except StopIteration: target_iter = iter(target_loader); target_data, target_teacher = next(target_iter)
            source_data0 = utils.radiation_noise(source_data).type(torch.FloatTensor); source_data1 = utils.flip_augmentation(source_data)
            target_data0 = utils.radiation_noise(target_data).type(torch.FloatTensor); target_data1 = utils.flip_augmentation(target_data)
            if lam > 0:
                result = model.forward_with_spectral(source_data.cuda(), target_data.cuda())
                (source_features, source1, _, source_outputs, source_out, target_features, _, target1, target_outputs, target_out, source_spec, target_spec) = result
            else:
                (source_features, source1, _, source_outputs, source_out, target_features, _, target1, target_outputs, target_out) = model(source_data.cuda(), target_data.cuda())
            (_, source2, _, source_outputs2, _, _, _, target2, t1, _) = model(source_data0.cuda(), target_data0.cuda())
            (_, source3, _, source_outputs3, _, _, _, target3, t2, _) = model(source_data1.cuda(), target_data1.cuda())
            pseudo_label_t = torch.softmax(target_outputs, 1).detach().argmax(1)
            all_source_con = torch.cat([source2.unsqueeze(1), source3.unsqueeze(1)], 1); all_target_con = torch.cat([target2.unsqueeze(1), target3.unsqueeze(1)], 1)
            cls_loss = ce(source_outputs, source_label.cuda())
            lmmd_loss = mmd.lmmd(source_features, target_features, source_label, torch.softmax(target_outputs, 1), BATCH_SIZE=BATCH_SIZE, CLASS_NUM=CLASS_NUM)
            lambd = 2 / (1 + math.exp(-10 * epoch / num_epochs)) - 1
            scl_loss = con_s(all_source_con, source_label.cuda()) + con_t(all_target_con, pseudo_label_t)
            domain_loss = dsh(source_out, target_out)
            if lam > 0:
                kd_s = cosine_kd(projection, source_spec, source_teacher.to(device)); kd_t = cosine_kd(projection, target_spec, target_teacher.to(device)); kd_loss = kd_s + kd_t
            else: kd_loss = source_outputs.new_zeros(())
            total_loss = cls_loss + 0.01 * lambd * lmmd_loss + scl_loss + domain_loss + lam * kd_loss
            optimizer.zero_grad(); total_loss.backward(); optimizer.step()
            n = len(source_label); total += n; correct += (source_outputs.argmax(1) == source_label.cuda()).sum().item()
            for key, value in (("total", total_loss), ("cls", cls_loss), ("scl", scl_loss), ("lmmd", lmmd_loss), ("kd", kd_loss)): sums[key] += float(value.item()) * n
        val_loss, val_acc = eval_source(model, val_loader, device)
        row = {"epoch": epoch, "train_total_loss": sums["total"] / total, "train_cls_loss": sums["cls"] / total, "train_scl_loss": sums["scl"] / total, "train_lmmd_loss": sums["lmmd"] / total, "train_kd_loss": sums["kd"] / total, "train_acc": correct / total, "val_loss": val_loss, "val_acc": val_acc, "lr": LEARNING_RATE}
        history.append(row); print(json.dumps({"lambda_kd": lam, "seed": seed, **row}))
        if val_acc > best["val_acc"]:
            best = row.copy(); torch.save({"model": model.state_dict(), "projection": projection.state_dict(), "lambda_kd": lam, "seed": seed, "best": best}, out / f"seed_{seed}_best.pth")
    (out / f"seed_{seed}_history.json").write_text(json.dumps(history, indent=2))
    # Return the selected source reference for target post-hoc evaluation.
    return best, train_x, target_x, source_ref


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lambda-kd", type=float, required=True, choices=(0.0, 0.05, 0.1, 0.2)); ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu"); ap.add_argument("--epochs", type=int, default=epochs); ap.add_argument("--seeds", type=int, nargs="*", default=seeds); ap.add_argument("--cache", type=Path, default=CACHE)
    args = ap.parse_args(); device = torch.device(args.device); out = OUT_ROOT / f"lambda_{args.lambda_kd:g}"; out.mkdir(parents=True, exist_ok=True)
    cache_npz = np.load(args.cache, allow_pickle=False)
    src_centers = cache_npz["source_centers"]; tgt_centers = cache_npz["target_centers"]
    source_fspec = cache_npz["source_fspec"]; target_fspec = cache_npz["target_fspec"]
    # Dict lookups preserve center alignment while avoiding target labels in
    # the training dataset.
    source_map = {(int(r), int(c)): i for i, (r, c) in enumerate(src_centers)}
    target_map = {(int(r), int(c)): i for i, (r, c) in enumerate(tgt_centers)}
    class Cache(dict): pass
    teacher_cache = Cache({(int(r), int(c)): source_fspec[i] for i, (r, c) in enumerate(src_centers)})
    teacher_cache["target"] = Cache({(int(r), int(c)): target_fspec[i] for i, (r, c) in enumerate(tgt_centers)})
    source, label_s = utils.load_data_houston(str(ROOT / "datasets/Houston/Houston13.mat"), str(ROOT / "datasets/Houston/Houston13_7gt.mat")); target = hdf5storage.loadmat(str(ROOT / "datasets/Houston/Houston18.mat"))["ori_data"]
    data_s, data_t = ILDA(source, target, 2, 0.009)
    all_results = []
    for seed in args.seeds:
        best, train_x, target_x, source_ref = run_seed(args.lambda_kd, seed, data_s, data_t, label_s, tgt_centers, source_map, teacher_cache, device, out, args.epochs)
        all_results.append({"seed": seed, "best_source": best})
    (out / "source_training_summary.json").write_text(json.dumps({"lambda_kd": args.lambda_kd, "seeds": args.seeds, "target_gt_used_for_training_or_selection": False, "results": all_results}, indent=2))
    print(json.dumps({"finished": True, "lambda_kd": args.lambda_kd, "artifact": str(out), "seeds": args.seeds}))


if __name__ == "__main__": main()

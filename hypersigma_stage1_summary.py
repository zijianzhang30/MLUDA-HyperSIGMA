"""Create a compact Stage-1 accuracy/loss figure and seed summary from artifacts."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path("/nas1/zhangzj26/HyperSIGMA_adapted")
RUNS = [(1174, ROOT), (1370, ROOT / "seed_1370"), (1417, ROOT / "seed_1417"), (1418, ROOT / "seed_1418")]

summary = []
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
for seed, directory in RUNS:
    history = json.loads((directory / "houston13_vitb_33x33_pca30_stage1_history.json").read_text())
    probe = json.loads((directory / "houston13_vitb_33x33_pca30_stage1_feature_probe.json").read_text())
    best = json.loads((directory / "houston13_vitb_33x33_pca30_stage1_config.json").read_text())["best"]
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["train_loss"] for row in history], label=f"seed {seed} train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], linestyle="--", label=f"seed {seed} val")
    axes[1].plot(epochs, [row["train_acc"] for row in history], label=f"seed {seed} train")
    axes[1].plot(epochs, [row["val_acc"] for row in history], linestyle="--", label=f"seed {seed} val")
    spectral = probe["spectral"]
    summary.append({
        "seed": seed,
        "best_epoch": best["epoch"],
        "best_source_val_acc": best["val_acc"],
        "best_source_val_loss": best["val_loss"],
        "spectral_source_margin": spectral["source_same"] - spectral["source_diff"],
        "spectral_target_margin": spectral["target_same"] - spectral["target_diff"],
        "spectral_cross_margin": spectral["cross_same"] - spectral["cross_diff"],
    })
axes[0].set(title="Stage 1 Loss", xlabel="Epoch", ylabel="Cross-entropy")
axes[1].set(title="Stage 1 Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1.02))
for axis in axes:
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7, ncol=2)
fig.savefig(ROOT / "houston13_vitb_33x33_pca30_stage1_curves.png", dpi=160)
(ROOT / "houston13_vitb_33x33_pca30_stage1_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))

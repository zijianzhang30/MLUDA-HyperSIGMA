# MLUDA / HyperSIGMA Houston 实验记录

最后整理：2026-09-02。本文记录当前已经完成的复现、HyperSIGMA teacher 适配、fine-tuning、feature probe 和 F_spec 蒸馏实验，以及可复现启动命令。

## 1. 项目与环境

- 项目：`/home/zhangzj26/TGRS_MLUDA-2024`
- Python 环境：`/home/zhangzj26/TGRS_MLUDA-2024/.venv/bin/python`
- 数据：`datasets/Houston/`
  - `Houston13.mat`、`Houston13_7gt.mat`
  - `Houston18.mat`、`Houston18_7gt.mat`
- HyperSIGMA 官方代码：`third_party/HyperSIGMA/`
- 预训练权重：`/nas1/zhangzj26/HyperSIGMA_weights/`
  - `spat-vit-base-ultra-checkpoint-1599.pth`
  - `spec-vit-base-ultra-checkpoint-1599.pth`
- 适配 teacher 与实验产物：`/nas1/zhangzj26/HyperSIGMA_adapted/`

当前环境需要 `opencv-contrib-python==4.8.1.78`，因为 MLUDA 的 ILDA 预处理调用 `cv2.ximgproc.guidedFilter`。

## 2. 原始 MLUDA baseline

原始训练入口是 `MLUDA_hu.py`，配置位于 `config_Houston.py`：100 epochs、7 类、每类 180 个 source train 中心像素、10 个固定 seed：

`1174, 1370, 1417, 1418, 1421, 1535, 1546, 1599, 1610, 1631`

典型启动方式（用 GPU 掩码将物理卡映射为 `cuda:0`）：

```bash
cd /home/zhangzj26/TGRS_MLUDA-2024
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u MLUDA_hu.py | tee mluda_houston.log
```

此前 10-seed baseline 统计（Houston13→Houston18）为：OA `76.30 ± 2.60%`，AA `72.33 ± 4.81%`，Kappa `62.00 ± 4.51%`。AA 只对 7 个非 background 类别取算术平均，label 0 不参与。

## 3. HyperSIGMA teacher smoke test 与输入结论

官方 Spatial ViT / Spectral ViT 的权重加载、冻结状态和 Houston 样本 forward 检查在 `hypersigma_teacher_smoke_test.py`。MLUDA 原始输入是 `[B,48,7,7]`；raw teacher 在该输入下出现 feature collapse，因此没有直接把 raw feature 用作 KD target。

```bash
cd /home/zhangzj26/TGRS_MLUDA-2024
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u hypersigma_teacher_smoke_test.py --device cuda:0
```

正式 teacher 遵循论文的 full-band fine-tuning 原则：`[B,48,33,33]`、`patch_size=2`、7 类。PCA30 仅保留为 notebook 配置的 ablation，不是正式 teacher；PCA 只在 Houston13 source 上拟合。

## 4. HyperSIGMA Houston downstream adaptation

### Stage 1

脚本：`hypersigma_stage1_protocol.py`。固定 spatially-disjoint Houston13 train/val split（seed=1174，train/val 中心 Chebyshev 距离至少 33），只用 source label 训练；Houston18 GT 不用于训练、选 checkpoint 或 early stopping。

Stage 1 冻结 pretrained Transformer blocks，仅训练随机初始化的输入/投影/FPN/SEM/classifier 等模块；checkpoint 根据 source disjoint val accuracy 选择。

Full48 正式 checkpoint：

`/nas1/zhangzj26/HyperSIGMA_adapted/protocol_stage1/bands48/stage1_best.pth`

启动：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zhangzj26/TGRS_MLUDA-2024/.venv/bin/python -u \
  hypersigma_stage1_protocol.py --bands 48 --device cuda:0 --epochs 20 --batch-size 32 \
  | tee stage1_full48.log
```

PCA30 ablation 只需将 `--bands 48` 改为 `--bands 30`，产物分别在 `protocol_stage1/bands30/` 和 `protocol_stage1/bands48/`。

Stage 1 后处理评估：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zhangzj26/TGRS_MLUDA-2024/.venv/bin/python -u \
  eval_protocol_stage1.py --artifact /nas1/zhangzj26/HyperSIGMA_adapted/protocol_stage1/bands48 \
  --bands 48 --device cuda:0
```

feature quality probe：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zhangzj26/TGRS_MLUDA-2024/.venv/bin/python -u \
  hypersigma_feature_quality_probe.py --device cuda:0 --max-per-class 100
```

该 probe 比较 `F_spat`、`F_spec`、SEM fused `F*` 的 source/target/cross-domain same-different cosine。Full48 Stage 1 中 F_spec 的类别分离明显优于 F_spat 和 SEM fused，但仍有 source→target domain gap；因此后续 KD 第一版只试 F_spec，并保留其他两种表示作为后续候选。

### Stage 2 source-only 对照

脚本：`hypersigma_ft_protocol.py`。起点为 Full48 Stage 1 checkpoint；仍只用 Houston13 source label 和同一个 spatially-disjoint split，source-val 选 checkpoint。

Partial FT（仅最后 2 个 Spatial/Spectral blocks，约 6e-6；新模块 6e-5）：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zhangzj26/TGRS_MLUDA-2024/.venv/bin/python -u \
  hypersigma_ft_protocol.py --mode partial --device cuda:0 --epochs 20 --batch-size 32 \
  | tee hypersigma_partial_ft.log
```

Full FT（全部 blocks，layer-wise 小学习率；新模块 6e-5）：

```bash
CUDA_VISIBLE_DEVICES=1 /home/zhangzj26/TGRS_MLUDA-2024/.venv/bin/python -u \
  hypersigma_ft_protocol.py --mode full --device cuda:0 --epochs 20 --batch-size 32 \
  | tee hypersigma_full_ft.log
```

产物：`protocol_stage1/partial_ft/`、`protocol_stage1/full_ft/`。两组 source-val 约 0.50，Houston18 post-hoc OA 约 0.54；相比 Stage 1 没有形成可靠的跨域 teacher，因此暂不作为 KD teacher。

## 5. F_spec feature cache

teacher 冻结并 `eval()`，为 MLUDA source/target 中心像素预先提取 Full48 `F_spec`，缓存形状：source `(2530,128)`、target `(53200,128)`。

```bash
CUDA_VISIBLE_DEVICES=1 /home/zhangzj26/TGRS_MLUDA-2024/.venv/bin/python -u \
  prepare_hypersigma_fspec_cache.py --device cuda:0 --batch-size 64 \
  --output /nas1/zhangzj26/HyperSIGMA_adapted/mluda_fspec_full48_cache.npz
```

缓存：`/nas1/zhangzj26/HyperSIGMA_adapted/mluda_fspec_full48_cache.npz`。

## 6. MLUDA F_spec KD

实现文件：

- `net2.py`：增加可选 spectral branch 输出；默认 MLUDA forward 行为不变。
- `MLUDA_hu_fspec_kd.py`：在 MBCA 前对齐 MLUDA spectral branch pooled feature `[B,192]`，增加 `Linear(192,128)` projection，与 frozen teacher `F_spec [B,128]` 做 normalized cosine loss。
- `eval_mluDA_fspec_kd.py`：只在训练结束后读取 Houston18 GT，做 post-hoc audit。

总损失为原始 MLUDA loss 加 `lambda_kd * (L_kd_source + L_kd_target)`；MBCA、SCL、LMMD、classifier 和 augmentation 未修改。target GT 不进入 loss、checkpoint 选择或调参。

四组训练启动命令（每个 lambda、seed 顺序执行；也可以使用不同 GPU 并行）：

```bash
cd /home/zhangzj26/TGRS_MLUDA-2024
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u MLUDA_hu_fspec_kd.py \
  --lambda-kd 0.0 --device cuda:0 --epochs 100 \
  --cache /nas1/zhangzj26/HyperSIGMA_adapted/mluda_fspec_full48_cache.npz \
  | tee mluda_fspec_kd_lambda0.log

CUDA_VISIBLE_DEVICES=2 .venv/bin/python -u MLUDA_hu_fspec_kd.py \
  --lambda-kd 0.05 --device cuda:0 --epochs 100 \
  --cache /nas1/zhangzj26/HyperSIGMA_adapted/mluda_fspec_full48_cache.npz \
  | tee mluda_fspec_kd_lambda005.log

CUDA_VISIBLE_DEVICES=3 .venv/bin/python -u MLUDA_hu_fspec_kd.py \
  --lambda-kd 0.1 --device cuda:0 --epochs 100 \
  --cache /nas1/zhangzj26/HyperSIGMA_adapted/mluda_fspec_full48_cache.npz \
  | tee mluda_fspec_kd_lambda01.log

CUDA_VISIBLE_DEVICES=4 .venv/bin/python -u MLUDA_hu_fspec_kd.py \
  --lambda-kd 0.2 --device cuda:0 --epochs 100 \
  --cache /nas1/zhangzj26/HyperSIGMA_adapted/mluda_fspec_full48_cache.npz \
  | tee mluda_fspec_kd_lambda02.log
```

统一 post-hoc 评估（不改变 checkpoint）：

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -u eval_mluDA_fspec_kd.py \
  --device cuda:0 --lambdas 0 0.05 0.1 0.2 \
  --cache /nas1/zhangzj26/HyperSIGMA_adapted/mluda_fspec_full48_cache.npz
```

KD 产物位于 `mluda_fspec_kd/lambda_0/`、`lambda_0.05/`、`lambda_0.1/`、`lambda_0.2/`，每个 seed 保存 `seed_<seed>_best.pth` 和 `seed_<seed>_history.json`；checkpoint 包含 student 与 KD projection。

## 7. 当前统一结果

Houston18 仅作最终 post-hoc audit，10 seed mean ± std：

| 设置 | OA | AA | Kappa |
|---|---:|---:|---:|
| 原始 MLUDA baseline | 76.30 ± 2.60% | 72.33 ± 4.81% | 62.00 ± 4.51% |
| F_spec KD λ=0 | 77.97 ± 2.05% | 71.57 ± 3.96% | 63.65 ± 4.01% |
| F_spec KD λ=0.05 | 77.15 ± 2.24% | 71.74 ± 3.47% | 62.89 ± 3.95% |
| F_spec KD λ=0.1 | 77.79 ± 1.62% | 71.72 ± 3.00% | 63.74 ± 3.01% |
| F_spec KD λ=0.2 | 77.85 ± 1.58% | 71.01 ± 2.71% | 63.68 ± 2.72% |

结果文件：

- `mluda_fspec_kd/posthoc_target_summary_lambda0_005.json`
- `mluda_fspec_kd/posthoc_target_summary.json`

目前结论：F_spec KD 没有显示稳定、明确的增益。`lambda=0` 无蒸馏对照的 OA/Kappa 最高；`lambda=0.1/0.2` 相比历史 baseline 有约 1.5 个百分点 OA/Kappa 提升，但 AA 仍略低，主要类别不均衡和第 7 类波动仍在。原始 MLUDA 的 `L_cls/L_scl/L_lmmd` 没有训练崩溃，KD loss 约 `0.47–0.55`，但暂时不能宣称 foundation representation 已改善 UDA 泛化。

## 8. 当前边界与下一步

- 暂不加入 `F_spat`、SEM fused `F*` KD，也不改 MBCA/SCL/LMMD。
- 不使用 Houston18 GT 做训练、模型选择、early stopping、阈值或超参数调整。
- 如果继续研究，应先分析 per-class/cross-domain margin 和 loss 曲线，再决定是否改进 teacher adaptation 或尝试更稳健的 feature alignment；不能根据当前 post-hoc target 指标反向调参。

## 9. Source-only F_spec prototype KD（单 seed 初检）

新增 `MLUDA_hu_fspec_proto_kd.py` 和 `eval_mluDA_fspec_proto_kd.py`。该版本不再逐点拟合 teacher feature，而是从全部 Houston13 非 background source F_spec 按 7 类计算 teacher prototypes；source batch 的 teacher/student feature 分别对同一组 prototypes 计算 temperature-softmax distribution，再使用 `KL(q_T || q_S)`。KD 只用于 source labeled batch，Houston18 GT 不参与训练或 checkpoint 选择。

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u MLUDA_hu_fspec_proto_kd.py \
  --device cuda:0 --epochs 100 --seed 1174 --lambda-kd 0.1 --temperature 0.1 \
  --cache /nas1/zhangzj26/HyperSIGMA_adapted/mluda_fspec_full48_cache.npz \
  | tee /nas1/zhangzj26/TGRS_MLUDA-2024/mluda_fspec_proto_kd_seed1174.log

CUDA_VISIBLE_DEVICES=1 .venv/bin/python -u eval_mluDA_fspec_proto_kd.py \
  --device cuda:0 \
  --checkpoint /nas1/zhangzj26/HyperSIGMA_adapted/mluda_fspec_proto_kd/lambda_0.1/seed_1174_best.pth
```

seed 1174 初步结果：

| 设置 | OA | AA | Kappa |
|---|---:|---:|---:|
| 同 seed λ=0 对照 | 74.17% | 66.86% | 55.35% |
| Source-only prototype KD | 76.71% | 69.58% | 62.07% |

prototype KD 的 source-val best 为 epoch 96，val accuracy 93.94%；训练 KD loss 从约 0.76 降至 0.37。单 seed 的 OA/AA/Kappa 均高于同 seed λ=0，但不能据此宣称稳定提升。teacher prototype 的 source top-1 归属准确率仅约 39.4%，部分类别 prototype cosine 高达约 0.99，说明 F_spec prototype 本身仍有明显类别混叠；因此先停在单 seed 正确性检查，不据 Houston18 结果继续调 temperature 或 λ。

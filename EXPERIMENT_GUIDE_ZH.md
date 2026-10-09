# WallTopo-UQ 实验执行指南

## 1. 环境与验收

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
.\run_smoke.ps1
```

`run_smoke.ps1` 会生成合成数据、训练两轮、执行 TTA 校准、测试评估和预测可视化。合成结果只用于验证代码路径，不能写入论文性能表。

## 2. 准备真实数据

数据来源和下载建议见 [DATASETS_ZH.md](DATASETS_ZH.md)。构建统一 manifest 后，先检查数据：

```powershell
python scripts/check_dataset.py --config configs/full.yaml --split train
```

按墙体、建筑或连续视频序列划分数据，禁止把同一面墙的 patch 随机拆到 train 和 test：

```powershell
python scripts/make_splits.py `
  --manifest data/real/all.csv `
  --output-dir data/real `
  --group-column group `
  --train-ratio 0.7 `
  --val-ratio 0.15 `
  --seed 42
```

将 `configs/full.yaml` 中三个 manifest 修改为：

```text
data/real/train.csv
data/real/val.csv
data/real/test.csv
```

## 3. 主实验和基线

训练 WallTopo-UQ：

```powershell
python scripts/train.py --config configs/full.yaml --device auto
```

代码内置三个统一入口：

- `model_name: walltopo_uq`
- `model_name: unet`
- `model_name: deeplabv3plus`

先跑两轮 smoke 对照：

```powershell
python scripts/run_suite.py --suite core --device auto
```

全量基线实验应把各配置文件中的 `epochs`、`batch_size`、增强、优化器和数据划分保持一致，只改变模型结构。

## 4. 消融实验

```powershell
python scripts/run_suite.py --suite ablations --device auto
```

当前自动矩阵包含：

- 去掉 MDSM；
- 单延迟 `{1}`；
- 去掉形态条件门控；
- 去掉 topology tokens；
- 去掉 dual cross-attention；
- 去掉拓扑损失；
- 去掉几何一致性；
- 去掉证据不确定性。

正式论文实验建议至少运行 5 个随机种子，并保留每个 seed 的独立 `output_dir`。

## 5. 测试、跨域和预测

```powershell
python scripts/evaluate.py `
  --config configs/full.yaml `
  --checkpoint runs/full/checkpoints/best.pt `
  --split test `
  --device auto

python scripts/cross_domain.py `
  --config configs/full.yaml `
  --checkpoint runs/full/checkpoints/best.pt `
  --manifest data/real/test.csv `
  --split test `
  --device auto

python scripts/predict.py `
  --config configs/full.yaml `
  --checkpoint runs/full/checkpoints/best.pt `
  --split test `
  --save-panels `
  --device auto
```

主模型训练结束时会在验证集上校准：

- 训练域特征均值/标准差；
- domain-shift 距离尺度；
- 报告级不确定性的复核阈值。

若没有校准文件，预测结果仍可输出 pixel vacuity，但会关闭人工复核阈值判定。

## 5.1 对没有 mask 的真实图片推理

准备一个新文件夹，例如：

```text
data/raw/my_wall_photos/
```

把手机或现场照片直接放入该文件夹，然后执行：

```powershell
python scripts\infer_images.py `
  --input data\raw\my_wall_photos `
  --output runs\dais\my_wall_results `
  --config configs\dais.yaml `
  --checkpoint runs\dais\checkpoints\best.pt `
  --threshold 0.5 `
  --device auto
```

每张图片会输出：

- `*_overlay.png`：原图上的裂缝叠加结果；
- `*_mask.png`：裂缝二值区域；
- `*_skeleton.png`：预测中心线；
- `*_uncertainty.png`：像素不确定性热图；
- `*_panel.png`：原图、叠加、mask、骨架和不确定性的组合图。

`inference_summary.csv` 按视觉筛查分数和人工复核标记排序。`threshold` 默认是
`0.5`；漏检较多时可尝试 `0.35`，误检较多时可尝试 `0.65`。阈值只能在验证集上
选择，不能根据测试图片反复调整。

## 6. Leave-one-domain-out

生成每一折的 train/val/test manifest：

```powershell
python scripts/make_domain_folds.py `
  --manifest data/real/all.csv `
  --output-dir data/real/folds `
  --domain-column domain `
  --group-column group
```

对每个 `data/real/folds/fold_*`：

1. 复制一份 `full.yaml`；
2. 把 train/val/test manifest 指向该折；
3. 使用唯一 `output_dir` 训练和评估；
4. 最后汇总所有折的 `test_metrics.json`。

```powershell
python scripts/aggregate_results.py `
  --root runs `
  --pattern "test_metrics.json" `
  --output runs/aggregated_metrics.csv
```

## 7. 必须报告的实验层级

像素层：Dice、IoU、Precision、Recall、Boundary F1。

拓扑层：clDice、skeleton Dice、endpoint F1、junction F1、Betti-0 error、Betti-1 error、break rate、false-merge rate、length MAE。

几何层：width MAE/MAPE、orientation MAE；只有存在尺度参考时才报告毫米单位。

等级层：Accuracy、QWK、ordinal MAE、Spearman correlation。

不确定性层：ECE、Brier、NLL、AURC、error-detection AUROC、risk-coverage、selective F1、selective width MAE。

效率层：参数量、FLOPs、FPS、GPU 显存和模型大小。

## 7.1 重新生成论文图

训练 Dais 和 DeepCrack 完成后，可以一键重画当前论文中的训练曲线和定性图：

```powershell
python scripts\make_paper_figures.py --device auto
```

输出目录：

```text
..\figures\
```

脚本会生成：

```text
fig_training_curves.pdf
fig_training_curves.png
fig_qualitative_results.pdf
fig_qualitative_results.png
```

定性图会自动选取两个测试集中 Dice 位于中位数的样本，并绘制原图、真实
mask、预测 mask 和错误图。错误图中绿色为真正例，红色为假正例，蓝色为假负例。

## 8. 投稿前最低实验组合

1. Dais 主数据上的 5 seed 主实验和基线；
2. Crack500/DeepCrack 外部验证；
3. 拓扑专项和几何专项；
4. 至少三组 leave-one-domain-out；
5. 10%、25%、50%、100% 标注比例实验；
6. 不确定性校准和选择性报告；
7. 失败案例可视化；
8. 复杂度与速度表。

代码已经覆盖这些实验的运行入口；最终论文价值仍取决于数据划分、真实尺度和专家等级标注质量。

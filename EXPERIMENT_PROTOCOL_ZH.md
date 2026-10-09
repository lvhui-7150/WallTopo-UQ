# WallTopo-UQ 正式实验设计与执行说明

## 1. 实验目标

实验部分不再只回答“模型是否比 U-Net 高一点”，而是检验以下五个可证伪命题：

| 编号 | 命题 | 主要证据 |
|---|---|---|
| H1 | 延迟状态记忆提高遮挡和长裂缝的路径连续性 | 去掉延迟或多延迟后的 break rate、clDice、Dice |
| H2 | topology tokens 和双向交叉解码改善分支与端点结构 | 去掉 tokens/dual-cross 后的 endpoint F1、junction F1、false-merge rate |
| H3 | 几何一致性约束提高宽度、方向和长度稳定性 | 去掉 geometry consistency 后的 width MAE、orientation MAE、length MAE |
| H4 | 图级不确定性比单一像素置信度更适合选择性报告 | ECE、AURC、error-detection AUROC、selective error |
| H5 | 设计在精度、参数量和推理成本之间保持合理平衡 | Dice--参数量--FPS--显存 Pareto 图 |

“SOTA”不能通过预设数字声明，只能在相同数据划分、训练预算和评价代码下，与强基线比较并由多随机种子结果支持。

## 2. 实验层级

### 2.1 Smoke

用途：验证代码路径。禁止写入论文性能表。

```powershell
python scripts\run_benchmark_suite.py `
  --profile smoke `
  --suite core `
  --datasets dais `
  --seeds 3407 `
  --run-prefix benchmark_smoke `
  --device cuda
```

### 2.2 Pilot

用途：快速检查模型是否值得进行全量训练，并调整学习率、批大小和图像尺寸。

```powershell
python scripts\run_benchmark_suite.py `
  --profile pilot `
  --suite core `
  --datasets dais deepcrack `
  --seeds 3407 `
  --run-prefix benchmark_pilot `
  --device cuda
```

### 2.3 主实验

建议至少使用 5 个随机种子：

```powershell
python scripts\run_benchmark_suite.py `
  --profile full `
  --suite core `
  --datasets dais deepcrack crack500 crackforest `
  --seeds 3407 3408 3409 3410 3411 `
  --run-prefix benchmark `
  --resume `
  --device cuda
```

### 2.4 消融

```powershell
python scripts\run_benchmark_suite.py `
  --profile full `
  --suite ablations `
  --datasets dais deepcrack `
  --seeds 3407 3408 3409 3410 3411 `
  --run-prefix ablated `
  --resume `
  --device cuda
```

消融包含：

- 去掉 MDSM；
- 单延迟 `{1}`；
- 去掉形态门控；
- 去掉 topology tokens；
- 去掉 dual cross-attention；
- 去掉拓扑损失；
- 去掉几何一致性；
- 去掉证据不确定性。

## 3. 基线设置

核心基线由统一模型入口训练：

- U-Net；
- DeepLabV3+ ResNet50；
- 可选 U-Net++ / FPN / PAN / MAnet / LinkNet，通过 `segmentation-models-pytorch`。

安装可选强基线依赖：

```powershell
python -m pip install -r requirements-benchmark.txt
```

运行可选 U-Net++：

```powershell
python scripts\run_benchmark_suite.py `
  --profile full `
  --models smp_unetplusplus `
  --datasets dais deepcrack `
  --seeds 3407 3408 3409 3410 3411 `
  --run-prefix benchmark `
  --resume `
  --device cuda
```

所有模型使用相同数据划分、增强参数、损失权重、训练轮数和验证准则，只改变模型结构。DeepLabV3+ 默认使用 ImageNet 预训练 ResNet50，作为强基线。

## 4. 性能增强策略

代码已经加入以下正式训练策略：

1. 验证集 mask threshold 搜索。阈值只由 validation 选择，test 不参与调参。
2. EMA 权重平均。默认在训练后 10% 轮次开始更新，用于验证和最佳检查点。
3. 随机尺度、旋转、翻折、亮度、对比度、Gamma、模糊和传感器噪声增强。
4. 几何 TTA。支持 identity、水平翻转、垂直翻转和 180 度旋转。
5. 多随机种子统计。汇总均值和 95% 置信区间。
6. 多检查点概率集成。可使用 `ensemble_evaluate.py` 作为论文中的最终增强版本，但必须与单模型结果分开报告。

### 4.1 阈值搜索

训练流水线会自动在验证集上搜索阈值，并写入：

```text
runs/<prefix>/<dataset>/<model>/seed_<seed>/threshold.json
```

手工运行：

```powershell
python scripts\tune_threshold.py `
  --config configs\generated\benchmark\dais\walltopo_uq\seed_3407.yaml `
  --checkpoint runs\benchmark\dais\walltopo_uq\seed_3407\checkpoints\best.pt `
  --split val `
  --metric composite `
  --device cuda
```

### 4.2 多检查点集成

```powershell
python scripts\ensemble_evaluate.py `
  --config configs\generated\benchmark\dais\walltopo_uq\seed_3407.yaml `
  --checkpoints `
    runs\benchmark\dais\walltopo_uq\seed_3407\checkpoints\best.pt `
    runs\benchmark\dais\walltopo_uq\seed_3408\checkpoints\best.pt `
    runs\benchmark\dais\walltopo_uq\seed_3409\checkpoints\best.pt `
    runs\benchmark\dais\walltopo_uq\seed_3410\checkpoints\best.pt `
    runs\benchmark\dais\walltopo_uq\seed_3411\checkpoints\best.pt `
  --split test `
  --threshold 0.525 `
  --device cuda
```

## 5. 结果汇总和统计检验

```powershell
python scripts\collect_benchmark_results.py `
  --root runs\benchmark `
  --output results\benchmark
```

输出：

```text
results/benchmark/raw_metrics.csv
results/benchmark/summary.csv
results/benchmark/pairwise_vs_proposed.csv
results/benchmark/table_benchmark_main.tex
results/benchmark/table_benchmark_ablation.tex
results/benchmark/collection_summary.json
```

统计规则：

- 每个数据集、模型、指标报告 mean、std、95% CI；
- 同一 seed 下进行配对比较；
- `n >= 6` 时采用 Wilcoxon signed-rank test；
- `n < 6` 时采用 paired t-test；
- 没有配对样本时不报告显著性。

## 6. 效率实验

```powershell
python scripts\benchmark_models.py `
  --config configs\dais.yaml `
  --models walltopo_uq unet deeplabv3plus `
  --size 224 `
  --batch-size 2 `
  --iterations 10 `
  --warmup 3 `
  --device cuda `
  --output results\benchmark\efficiency.csv
```

报告：

- 参数量；
- 每次训练迭代时间；
- 每秒图像数；
- 峰值 CUDA 显存；
- 模型大小。

## 7. 出版级图

```powershell
python scripts\make_publication_figures.py `
  --results results\benchmark `
  --output ..\figures\publication
```

输出矢量 PDF 和 600 dpi PNG：

- 主性能对比图；
- 消融效应图；
- 可靠性与校准图；
- 精度--效率 Pareto 图。

定性对比图：

```powershell
python scripts\make_qualitative_comparison.py `
  --run-prefix benchmark `
  --datasets dais deepcrack `
  --models unet deeplabv3plus walltopo_uq `
  --seed 3407 `
  --samples 1 `
  --device cuda `
  --output ..\figures\publication\fig_qualitative.png
```

多模型、多样本画廊：

```powershell
python scripts\make_model_comparison_gallery.py `
  --run-prefix qualitative `
  --datasets dais deepcrack crack500 crackforest `
  --models unet deeplabv3plus walltopo_uq `
  --seed 3407 `
  --samples 3 `
  --device cuda `
  --output-dir ..\figures\qualitative
```

每一行是一个测试样本，列依次为输入、GT、U-Net、DeepLabV3+、WallTopo-UQ、
WallTopo-UQ 概率图和不确定度图。误差图颜色为：绿色真正例、红色假正例、蓝色假阴性。

## 8. 跨域实验

先生成 leave-one-domain-out folds，再对每一折训练和测试。正式实验至少报告：

- masonry -> concrete/pavement；
- pavement -> masonry；
- camera -> mobile/UAV；
- bright -> shadow/low-light。

不能把同一墙体或同一连续序列的 patch 分到不同 split。

## 9. 运行顺序

推荐顺序：

1. `smoke` 确认环境和代码；
2. `pilot` 在 Dais/DeepCrack 上检查收敛；
3. `full` 主实验；
4. `ablations`；
5. `efficiency`；
6. `cross-domain`；
7. `ensemble`；
8. `collect`；
9. `publication figures`。

## 10. SOTA 判定标准

只有同时满足以下条件，才在论文中写“achieves state-of-the-art”：

1. 与至少三个强基线在相同划分和协议下比较；
2. 主要指标在多个数据集上排名第一或统计上不劣于最佳基线；
3. 至少 5 个随机种子的均值和置信区间支持结论；
4. 所有阈值和超参数均在验证集选择；
5. 消融证明核心模块确实贡献性能；
6. 报告失败案例和跨域限制；
7. 不把集成结果冒充单模型结果。

如果没有达到上述条件，应写为 `competitive performance` 或 `improves topology/reliability at comparable accuracy`，不能写 SOTA。

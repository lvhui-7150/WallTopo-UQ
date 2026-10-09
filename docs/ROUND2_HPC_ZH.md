# WallTopo-UQ 第二轮审稿实验包

这个包只覆盖少量补丁文件，不安装新环境，不自动下载权重。

## 1. 上传

把整个 `hpc_round2` 目录上传到超算：

```bash
scp -r hpc_round2 <HPC_USER>@<HPC_HOST>:~/projects/WallTopo-UQ-HPC/
```

在超算上安装覆盖文件：

```bash
cd "$HOME/projects/WallTopo-UQ-HPC"
bash hpc_round2/install_round2.sh
```

## 2. 环境

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate aenet
unset PYTHONPATH || true
export KMP_DUPLICATE_LIB_OK=TRUE
```

U-Net++ 已提供纯 PyTorch fallback。若 `segmentation-models-pytorch` 不存在，
`scripts/run_benchmark_suite.py --models smp_unetplusplus` 会自动使用内置
`UNetPlusPlusBaseline`，不会下载 ImageNet 权重。

不要在 `login` 节点直接运行 `--device cuda`。登录节点没有 NVIDIA 驱动，
GPU 训练必须通过 `jsub` 投递。先提交一个 smoke 作业验证路径：

```bash
cd "$HOME/projects/WallTopo-UQ-HPC"
mkdir -p hpc_jobs/round2/smoke
cp hpc_jobs/round2/baseline_unetplusplus/unetplusplus__dais__seed_3407.sub \
  hpc_jobs/round2/smoke/
cd hpc_jobs/round2/smoke
jsub < unetplusplus__dais__seed_3407.sub
```

如果只想在登录节点做纯 CPU 的代码路径检查：

```bash
python scripts/run_benchmark_suite.py \
  --profile smoke \
  --datasets dais \
  --models smp_unetplusplus \
  --seeds 3407 \
  --run-prefix round2_smoke \
  --num-workers 0 \
  --batch-size 1 \
  --device cpu
```

GPU 正式作业仍应使用生成的 `.sub` 文件提交。

## 3. 生成作业

```bash
cd "$HOME/projects/WallTopo-UQ-HPC"
python scripts/generate_round2_hpc_jobs.py \
  --output-root hpc_jobs/round2 \
  --seeds 3407 3408 3409 3410 3411 \
  --batch-size 1 \
  --passes 10
```

生成的作业分为五组：

```text
hpc_jobs/round2/baseline_unetplusplus      20 jobs
hpc_jobs/round2/ablation_missing_datasets  80 jobs
hpc_jobs/round2/lodo                        60 jobs
hpc_jobs/round2/uncertainty_corruptions      4 jobs
hpc_jobs/round2/efficiency                   1 job
```

不要一次把 165 个作业全部提交。按下面的优先级分批提交。

## 4. 建议提交顺序

### 4.1 先补 U-Net++ 强基线

```bash
bash hpc_jobs/round2/baseline_unetplusplus/submit_baseline_unetplusplus.sh
```

结果位置：

```text
runs/benchmark/<dataset>/smp_unetplusplus/seed_<seed>/test_metrics.json
```

### 4.2 补 Crack500/CrackForest 消融

```bash
bash hpc_jobs/round2/ablation_missing_datasets/submit_ablation_missing_datasets.sh
```

结果位置：

```text
runs/reviewer_ablation/<dataset>/<ablation>/seed_<seed>/test_metrics.json
```

### 4.3 LODO 跨域

```bash
bash hpc_jobs/round2/lodo/submit_lodo.sh
```

结果位置：

```text
runs/reviewer_lodo/<scenario>/<model>/seed_<seed>/cross_domain_summary.json
```

四个场景为 `leave_crack500`、`leave_crackforest`、`leave_dais`、
`leave_deepcrack`。

### 4.4 不确定性扰动

这四个作业耗时最长，建议最后提交：

```bash
bash hpc_jobs/round2/uncertainty_corruptions/submit_uncertainty_corruptions.sh
```

每个数据集包含 clean、噪声、模糊、JPEG、亮度、对比度和阴影扰动。

### 4.5 效率与参数量

```bash
bash hpc_jobs/round2/efficiency/submit_efficiency.sh
```

输出：

```text
results/round2/efficiency.csv
results/round2/efficiency.json
```

## 5. 汇总

主表汇总：

```bash
python scripts/collect_benchmark_results.py \
  --root runs/benchmark \
  --output results/benchmark_round2
```

消融汇总：

```bash
python scripts/collect_benchmark_results.py \
  --root runs/reviewer_ablation \
  --output results/reviewer_ablation
```

LODO 和扰动汇总：

```bash
python scripts/collect_round2_results.py \
  --project-root "$PWD" \
  --output-root results/round2
```

输出：

```text
results/round2/lodo_raw.csv
results/round2/lodo_summary.csv
results/round2/corruption_raw.csv
results/round2/corruption_summary.csv
results/round2/collection_summary.json
```

## 6. GAF-Net 和 DCUFormer

这两个模型没有随包伪造实现。若拿到官方代码或权重，需要把它们转换成与
`walltopo_uq.models.build_model` 相同的输出字典后，才能进入同一套训练和评价流程。
否则论文中只能继续把它们写成相关工作，不能写数值 SOTA 对比。

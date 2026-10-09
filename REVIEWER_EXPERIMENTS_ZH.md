# WallTopo-UQ 评审补实验包

这个包对应审稿意见中的四类核心补充实验：

1. 模块消融：MDSM、morphology、TCD、topology tokens、dual cross、topology loss、geometry consistency、EUS。
2. 拓扑基线：U-Net + clDice，以及 U-Net + connectivity-aware topology loss。
3. 跨域实验：pavement -> masonry，以及 masonry -> pavement。
4. 不确定性基线：softmax confidence、MC Dropout、5-model Deep Ensemble。

## 一、你在超算上应该上传什么

你已经有 `$HOME/projects/WallTopo-UQ-HPC`，不要上传整个数据集或整个旧项目。只需要覆盖或新增以下内容：

```text
scripts/run_benchmark_suite.py
scripts/collect_benchmark_results.py
scripts/evaluate.py
scripts/run_cross_domain_suite.py
scripts/evaluate_uncertainty_baselines.py
scripts/generate_reviewer_hpc_jobs.py
scripts/collect_reviewer_results.py
hpc_jobs/reviewer/
```

原有以下目录必须保留：

```text
data/raw/
configs/generated/benchmark/
runs/benchmark/
```

不确定性基线会直接读取旧的 `runs/benchmark` checkpoints，所以不要删除已有 80 个 benchmark runs。

## 二、生成作业脚本

在新代码根目录运行：

```bash
cd "$HOME/projects/WallTopo-UQ-HPC"
export PATH="$HOME/miniconda3/envs/aenet/bin:$PATH"
unset PYTHONPATH

python scripts/generate_reviewer_hpc_jobs.py
```

默认会生成：

```text
reviewer_ablation:   80 jobs
reviewer_topology:   30 jobs
reviewer_crossdomain: 20 jobs
reviewer_uncertainty: 12 jobs
```

所有作业都在：

```text
hpc_jobs/reviewer/
```

## 三、建议按阶段提交

不要一次把四个阶段全部提交。先跑消融，再跑拓扑和跨域，最后跑不确定性评测。

### 阶段 1：模块消融

```bash
bash hpc_jobs/reviewer/ablation/submit_ablation.sh
```

结果写入：

```text
runs/reviewer_ablation/
```

### 阶段 2：拓扑基线

```bash
bash hpc_jobs/reviewer/topology/submit_topology.sh
```

结果写入：

```text
runs/reviewer_topology/
```

### 阶段 3：跨域实验

```bash
bash hpc_jobs/reviewer/crossdomain/submit_crossdomain.sh
```

结果写入：

```text
runs/reviewer_crossdomain/
```

跨域训练配置中：

- `pavement_to_masonry`：Crack500 + CrackForest + DeepCrack 训练，Dais 测试。
- `masonry_to_pavement`：Dais 训练，Crack500、CrackForest、DeepCrack 分别测试。

### 阶段 4：不确定性基线

```bash
bash hpc_jobs/reviewer/uncertainty/submit_uncertainty.sh
```

结果写入每个旧 benchmark run 目录：

```text
runs/benchmark/<dataset>/<model>/uncertainty_baselines_test.json
```

该阶段依赖已有 5 个 seeds 的 checkpoints。

## 四、查看队列

```bash
jjobs -q gpu4
```

## 五、阶段完成后收集结果

所有阶段结束后运行：

```bash
python scripts/collect_reviewer_results.py
```

结果汇总在：

```text
results/reviewer_summary/
```

其中包括：

- `ablation/summary.csv`
- `topology/summary.csv`
- `cross_domain_results.csv`
- `uncertainty_baselines.csv`
- `reviewer_summary.json`

## 六、如果队列太长

只生成 3 个 seeds：

```bash
python scripts/generate_reviewer_hpc_jobs.py --seeds 3407 3408 3409
```

只跑一个消融数据集：

```bash
python scripts/generate_reviewer_hpc_jobs.py \
  --ablation-datasets dais \
  --topology-datasets dais deepcrack \
  --seeds 3407 3408 3409
```

生成脚本只重建 `hpc_jobs/reviewer/`，不会删除已有 runs。

## 八、2026-10-08 失败任务修复

上一批结果中的失败来自三个问题：

1. `ablation_no_morphology`：MDSM 在关闭 morphology 时的通道数错误。
2. `ablation_no_dual_cross`：关闭 dual cross 后的特征尺寸未对齐。
3. DeepCrack 并发训练：部分任务发生 CUDA OOM；跨域任务还出现了共享缓存文件损坏。

修复已包含在当前包中：

- MDSM morphology-off 分支使用正确的三通道 morphology 占位。
- TCD bypass 分支在拼接前对齐空间尺寸。
- 每个 run 使用独立 `derived_cache`，避免并发写坏缓存。
- 新生成作业默认 `batch-size=1`。
- 新增 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

覆盖新代码后重新生成作业：

```bash
python scripts/generate_reviewer_hpc_jobs.py --batch-size 1
```

重新提交时，已完成的 run 会自动跳过：

```bash
bash hpc_jobs/reviewer/ablation/submit_ablation.sh
bash hpc_jobs/reviewer/topology/submit_topology.sh
bash hpc_jobs/reviewer/crossdomain/submit_crossdomain.sh
```

根据上一批结果，实际需要重跑的是：

- 总计 130 个训练任务，已完成 90 个。
- Ablation：44/80 完成，36 个失败，其中 10 个 MDSM 通道错误、10 个 dual-cross 尺寸错误、16 个 CUDA OOM。
- Topology：30/30 完成。
- Crossdomain：16/20 完成，4 个失败，其中 1 个缓存损坏、3 个 CUDA OOM。

提交脚本使用已有结果文件判断是否完成，因此不需要手工挑选失败任务。

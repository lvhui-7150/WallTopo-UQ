# Reviewer 任务进度检查

在超算项目根目录运行：

```bash
cd "$HOME/projects/WallTopo-UQ-HPC"
export PATH="$HOME/miniconda3/envs/aenet/bin:$PATH"
unset PYTHONPATH

python scripts/check_reviewer_progress.py \
  --project-root "$PWD" \
  --output "$PWD/results/reviewer_progress.csv"
```

输出内容包括：

- 已完成任务列表
- 未完成任务列表
- 每个未完成任务的错误类型
- 各阶段完成数量统计
- `results/reviewer_progress.csv`

只查看未完成任务：

```bash
python scripts/check_reviewer_progress.py \
  --project-root "$PWD" \
  --output "$PWD/results/reviewer_missing.csv" \
  --status missing
```

只查看某个阶段：

```bash
python scripts/check_reviewer_progress.py \
  --project-root "$PWD" \
  --output "$PWD/results/reviewer_ablation_progress.csv" \
  --stage ablation
```

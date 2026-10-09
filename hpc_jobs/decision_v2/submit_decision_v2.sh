#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

result_path="../../runs/benchmark/dais/walltopo_uq_v2/seed_3407/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed dais__walltopo_uq_v2__seed_3407.sub"
else
  echo "submit dais__walltopo_uq_v2__seed_3407.sub"
  jsub < "dais__walltopo_uq_v2__seed_3407.sub"
fi
result_path="../../runs/benchmark/dais/walltopo_uq_v2/seed_3408/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed dais__walltopo_uq_v2__seed_3408.sub"
else
  echo "submit dais__walltopo_uq_v2__seed_3408.sub"
  jsub < "dais__walltopo_uq_v2__seed_3408.sub"
fi
result_path="../../runs/benchmark/dais/walltopo_uq_v2/seed_3409/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed dais__walltopo_uq_v2__seed_3409.sub"
else
  echo "submit dais__walltopo_uq_v2__seed_3409.sub"
  jsub < "dais__walltopo_uq_v2__seed_3409.sub"
fi
result_path="../../runs/benchmark/deepcrack/walltopo_uq_v2/seed_3407/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed deepcrack__walltopo_uq_v2__seed_3407.sub"
else
  echo "submit deepcrack__walltopo_uq_v2__seed_3407.sub"
  jsub < "deepcrack__walltopo_uq_v2__seed_3407.sub"
fi
result_path="../../runs/benchmark/deepcrack/walltopo_uq_v2/seed_3408/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed deepcrack__walltopo_uq_v2__seed_3408.sub"
else
  echo "submit deepcrack__walltopo_uq_v2__seed_3408.sub"
  jsub < "deepcrack__walltopo_uq_v2__seed_3408.sub"
fi
result_path="../../runs/benchmark/deepcrack/walltopo_uq_v2/seed_3409/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed deepcrack__walltopo_uq_v2__seed_3409.sub"
else
  echo "submit deepcrack__walltopo_uq_v2__seed_3409.sub"
  jsub < "deepcrack__walltopo_uq_v2__seed_3409.sub"
fi
result_path="../../runs/benchmark/crack500/walltopo_uq_v2/seed_3407/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed crack500__walltopo_uq_v2__seed_3407.sub"
else
  echo "submit crack500__walltopo_uq_v2__seed_3407.sub"
  jsub < "crack500__walltopo_uq_v2__seed_3407.sub"
fi
result_path="../../runs/benchmark/crack500/walltopo_uq_v2/seed_3408/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed crack500__walltopo_uq_v2__seed_3408.sub"
else
  echo "submit crack500__walltopo_uq_v2__seed_3408.sub"
  jsub < "crack500__walltopo_uq_v2__seed_3408.sub"
fi
result_path="../../runs/benchmark/crack500/walltopo_uq_v2/seed_3409/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed crack500__walltopo_uq_v2__seed_3409.sub"
else
  echo "submit crack500__walltopo_uq_v2__seed_3409.sub"
  jsub < "crack500__walltopo_uq_v2__seed_3409.sub"
fi

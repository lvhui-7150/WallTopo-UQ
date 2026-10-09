#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

result_path="../../../runs/benchmark/crack500/walltopo_uq/uncertainty_baselines_test.json"
if [ -f "$result_path" ]; then
  echo "skip completed uncertainty__crack500__walltopo_uq.sub"
else
  echo "submit uncertainty__crack500__walltopo_uq.sub"
  jsub < "uncertainty__crack500__walltopo_uq.sub"
fi

result_path="../../../runs/benchmark/crack500/walltopo_uq_v2/uncertainty_baselines_test.json"
if [ -f "$result_path" ]; then
  echo "skip completed uncertainty__crack500__walltopo_uq_v2.sub"
else
  echo "submit uncertainty__crack500__walltopo_uq_v2.sub"
  jsub < "uncertainty__crack500__walltopo_uq_v2.sub"
fi

result_path="../../../runs/benchmark/crack500/unet/uncertainty_baselines_test.json"
if [ -f "$result_path" ]; then
  echo "skip completed uncertainty__crack500__unet.sub"
else
  echo "submit uncertainty__crack500__unet.sub"
  jsub < "uncertainty__crack500__unet.sub"
fi

result_path="../../../runs/benchmark/crackforest/walltopo_uq/uncertainty_baselines_test.json"
if [ -f "$result_path" ]; then
  echo "skip completed uncertainty__crackforest__walltopo_uq.sub"
else
  echo "submit uncertainty__crackforest__walltopo_uq.sub"
  jsub < "uncertainty__crackforest__walltopo_uq.sub"
fi

result_path="../../../runs/benchmark/crackforest/walltopo_uq_v2/uncertainty_baselines_test.json"
if [ -f "$result_path" ]; then
  echo "skip completed uncertainty__crackforest__walltopo_uq_v2.sub"
else
  echo "submit uncertainty__crackforest__walltopo_uq_v2.sub"
  jsub < "uncertainty__crackforest__walltopo_uq_v2.sub"
fi

result_path="../../../runs/benchmark/crackforest/unet/uncertainty_baselines_test.json"
if [ -f "$result_path" ]; then
  echo "skip completed uncertainty__crackforest__unet.sub"
else
  echo "submit uncertainty__crackforest__unet.sub"
  jsub < "uncertainty__crackforest__unet.sub"
fi

result_path="../../../runs/benchmark/dais/walltopo_uq/uncertainty_baselines_test.json"
if [ -f "$result_path" ]; then
  echo "skip completed uncertainty__dais__walltopo_uq.sub"
else
  echo "submit uncertainty__dais__walltopo_uq.sub"
  jsub < "uncertainty__dais__walltopo_uq.sub"
fi

result_path="../../../runs/benchmark/dais/walltopo_uq_v2/uncertainty_baselines_test.json"
if [ -f "$result_path" ]; then
  echo "skip completed uncertainty__dais__walltopo_uq_v2.sub"
else
  echo "submit uncertainty__dais__walltopo_uq_v2.sub"
  jsub < "uncertainty__dais__walltopo_uq_v2.sub"
fi

result_path="../../../runs/benchmark/dais/unet/uncertainty_baselines_test.json"
if [ -f "$result_path" ]; then
  echo "skip completed uncertainty__dais__unet.sub"
else
  echo "submit uncertainty__dais__unet.sub"
  jsub < "uncertainty__dais__unet.sub"
fi

result_path="../../../runs/benchmark/deepcrack/walltopo_uq/uncertainty_baselines_test.json"
if [ -f "$result_path" ]; then
  echo "skip completed uncertainty__deepcrack__walltopo_uq.sub"
else
  echo "submit uncertainty__deepcrack__walltopo_uq.sub"
  jsub < "uncertainty__deepcrack__walltopo_uq.sub"
fi

result_path="../../../runs/benchmark/deepcrack/walltopo_uq_v2/uncertainty_baselines_test.json"
if [ -f "$result_path" ]; then
  echo "skip completed uncertainty__deepcrack__walltopo_uq_v2.sub"
else
  echo "submit uncertainty__deepcrack__walltopo_uq_v2.sub"
  jsub < "uncertainty__deepcrack__walltopo_uq_v2.sub"
fi

result_path="../../../runs/benchmark/deepcrack/unet/uncertainty_baselines_test.json"
if [ -f "$result_path" ]; then
  echo "skip completed uncertainty__deepcrack__unet.sub"
else
  echo "submit uncertainty__deepcrack__unet.sub"
  jsub < "uncertainty__deepcrack__unet.sub"
fi

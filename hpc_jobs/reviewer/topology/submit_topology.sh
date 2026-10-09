#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

result_path="../../../runs/reviewer_topology/crackforest/unet_cldice/seed_3407/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed topology__crackforest__unet_cldice__seed_3407.sub"
else
  echo "submit topology__crackforest__unet_cldice__seed_3407.sub"
  jsub < "topology__crackforest__unet_cldice__seed_3407.sub"
fi

result_path="../../../runs/reviewer_topology/crackforest/unet_cldice/seed_3408/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed topology__crackforest__unet_cldice__seed_3408.sub"
else
  echo "submit topology__crackforest__unet_cldice__seed_3408.sub"
  jsub < "topology__crackforest__unet_cldice__seed_3408.sub"
fi

result_path="../../../runs/reviewer_topology/crackforest/unet_cldice/seed_3409/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed topology__crackforest__unet_cldice__seed_3409.sub"
else
  echo "submit topology__crackforest__unet_cldice__seed_3409.sub"
  jsub < "topology__crackforest__unet_cldice__seed_3409.sub"
fi

result_path="../../../runs/reviewer_topology/crackforest/unet_cldice/seed_3410/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed topology__crackforest__unet_cldice__seed_3410.sub"
else
  echo "submit topology__crackforest__unet_cldice__seed_3410.sub"
  jsub < "topology__crackforest__unet_cldice__seed_3410.sub"
fi

result_path="../../../runs/reviewer_topology/crackforest/unet_cldice/seed_3411/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed topology__crackforest__unet_cldice__seed_3411.sub"
else
  echo "submit topology__crackforest__unet_cldice__seed_3411.sub"
  jsub < "topology__crackforest__unet_cldice__seed_3411.sub"
fi

result_path="../../../runs/reviewer_topology/crackforest/unet_connectivity/seed_3407/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed topology__crackforest__unet_connectivity__seed_3407.sub"
else
  echo "submit topology__crackforest__unet_connectivity__seed_3407.sub"
  jsub < "topology__crackforest__unet_connectivity__seed_3407.sub"
fi

result_path="../../../runs/reviewer_topology/crackforest/unet_connectivity/seed_3408/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed topology__crackforest__unet_connectivity__seed_3408.sub"
else
  echo "submit topology__crackforest__unet_connectivity__seed_3408.sub"
  jsub < "topology__crackforest__unet_connectivity__seed_3408.sub"
fi

result_path="../../../runs/reviewer_topology/crackforest/unet_connectivity/seed_3409/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed topology__crackforest__unet_connectivity__seed_3409.sub"
else
  echo "submit topology__crackforest__unet_connectivity__seed_3409.sub"
  jsub < "topology__crackforest__unet_connectivity__seed_3409.sub"
fi

result_path="../../../runs/reviewer_topology/crackforest/unet_connectivity/seed_3410/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed topology__crackforest__unet_connectivity__seed_3410.sub"
else
  echo "submit topology__crackforest__unet_connectivity__seed_3410.sub"
  jsub < "topology__crackforest__unet_connectivity__seed_3410.sub"
fi

result_path="../../../runs/reviewer_topology/crackforest/unet_connectivity/seed_3411/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed topology__crackforest__unet_connectivity__seed_3411.sub"
else
  echo "submit topology__crackforest__unet_connectivity__seed_3411.sub"
  jsub < "topology__crackforest__unet_connectivity__seed_3411.sub"
fi

result_path="../../../runs/reviewer_topology/dais/unet_cldice/seed_3407/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed topology__dais__unet_cldice__seed_3407.sub"
else
  echo "submit topology__dais__unet_cldice__seed_3407.sub"
  jsub < "topology__dais__unet_cldice__seed_3407.sub"
fi

result_path="../../../runs/reviewer_topology/dais/unet_cldice/seed_3408/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed topology__dais__unet_cldice__seed_3408.sub"
else
  echo "submit topology__dais__unet_cldice__seed_3408.sub"
  jsub < "topology__dais__unet_cldice__seed_3408.sub"
fi

result_path="../../../runs/reviewer_topology/dais/unet_cldice/seed_3409/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed topology__dais__unet_cldice__seed_3409.sub"
else
  echo "submit topology__dais__unet_cldice__seed_3409.sub"
  jsub < "topology__dais__unet_cldice__seed_3409.sub"
fi

result_path="../../../runs/reviewer_topology/dais/unet_cldice/seed_3410/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed topology__dais__unet_cldice__seed_3410.sub"
else
  echo "submit topology__dais__unet_cldice__seed_3410.sub"
  jsub < "topology__dais__unet_cldice__seed_3410.sub"
fi

result_path="../../../runs/reviewer_topology/dais/unet_cldice/seed_3411/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed topology__dais__unet_cldice__seed_3411.sub"
else
  echo "submit topology__dais__unet_cldice__seed_3411.sub"
  jsub < "topology__dais__unet_cldice__seed_3411.sub"
fi

result_path="../../../runs/reviewer_topology/dais/unet_connectivity/seed_3407/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed topology__dais__unet_connectivity__seed_3407.sub"
else
  echo "submit topology__dais__unet_connectivity__seed_3407.sub"
  jsub < "topology__dais__unet_connectivity__seed_3407.sub"
fi

result_path="../../../runs/reviewer_topology/dais/unet_connectivity/seed_3408/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed topology__dais__unet_connectivity__seed_3408.sub"
else
  echo "submit topology__dais__unet_connectivity__seed_3408.sub"
  jsub < "topology__dais__unet_connectivity__seed_3408.sub"
fi

result_path="../../../runs/reviewer_topology/dais/unet_connectivity/seed_3409/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed topology__dais__unet_connectivity__seed_3409.sub"
else
  echo "submit topology__dais__unet_connectivity__seed_3409.sub"
  jsub < "topology__dais__unet_connectivity__seed_3409.sub"
fi

result_path="../../../runs/reviewer_topology/dais/unet_connectivity/seed_3410/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed topology__dais__unet_connectivity__seed_3410.sub"
else
  echo "submit topology__dais__unet_connectivity__seed_3410.sub"
  jsub < "topology__dais__unet_connectivity__seed_3410.sub"
fi

result_path="../../../runs/reviewer_topology/dais/unet_connectivity/seed_3411/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed topology__dais__unet_connectivity__seed_3411.sub"
else
  echo "submit topology__dais__unet_connectivity__seed_3411.sub"
  jsub < "topology__dais__unet_connectivity__seed_3411.sub"
fi

result_path="../../../runs/reviewer_topology/deepcrack/unet_cldice/seed_3407/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed topology__deepcrack__unet_cldice__seed_3407.sub"
else
  echo "submit topology__deepcrack__unet_cldice__seed_3407.sub"
  jsub < "topology__deepcrack__unet_cldice__seed_3407.sub"
fi

result_path="../../../runs/reviewer_topology/deepcrack/unet_cldice/seed_3408/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed topology__deepcrack__unet_cldice__seed_3408.sub"
else
  echo "submit topology__deepcrack__unet_cldice__seed_3408.sub"
  jsub < "topology__deepcrack__unet_cldice__seed_3408.sub"
fi

result_path="../../../runs/reviewer_topology/deepcrack/unet_cldice/seed_3409/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed topology__deepcrack__unet_cldice__seed_3409.sub"
else
  echo "submit topology__deepcrack__unet_cldice__seed_3409.sub"
  jsub < "topology__deepcrack__unet_cldice__seed_3409.sub"
fi

result_path="../../../runs/reviewer_topology/deepcrack/unet_cldice/seed_3410/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed topology__deepcrack__unet_cldice__seed_3410.sub"
else
  echo "submit topology__deepcrack__unet_cldice__seed_3410.sub"
  jsub < "topology__deepcrack__unet_cldice__seed_3410.sub"
fi

result_path="../../../runs/reviewer_topology/deepcrack/unet_cldice/seed_3411/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed topology__deepcrack__unet_cldice__seed_3411.sub"
else
  echo "submit topology__deepcrack__unet_cldice__seed_3411.sub"
  jsub < "topology__deepcrack__unet_cldice__seed_3411.sub"
fi

result_path="../../../runs/reviewer_topology/deepcrack/unet_connectivity/seed_3407/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed topology__deepcrack__unet_connectivity__seed_3407.sub"
else
  echo "submit topology__deepcrack__unet_connectivity__seed_3407.sub"
  jsub < "topology__deepcrack__unet_connectivity__seed_3407.sub"
fi

result_path="../../../runs/reviewer_topology/deepcrack/unet_connectivity/seed_3408/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed topology__deepcrack__unet_connectivity__seed_3408.sub"
else
  echo "submit topology__deepcrack__unet_connectivity__seed_3408.sub"
  jsub < "topology__deepcrack__unet_connectivity__seed_3408.sub"
fi

result_path="../../../runs/reviewer_topology/deepcrack/unet_connectivity/seed_3409/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed topology__deepcrack__unet_connectivity__seed_3409.sub"
else
  echo "submit topology__deepcrack__unet_connectivity__seed_3409.sub"
  jsub < "topology__deepcrack__unet_connectivity__seed_3409.sub"
fi

result_path="../../../runs/reviewer_topology/deepcrack/unet_connectivity/seed_3410/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed topology__deepcrack__unet_connectivity__seed_3410.sub"
else
  echo "submit topology__deepcrack__unet_connectivity__seed_3410.sub"
  jsub < "topology__deepcrack__unet_connectivity__seed_3410.sub"
fi

result_path="../../../runs/reviewer_topology/deepcrack/unet_connectivity/seed_3411/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed topology__deepcrack__unet_connectivity__seed_3411.sub"
else
  echo "submit topology__deepcrack__unet_connectivity__seed_3411.sub"
  jsub < "topology__deepcrack__unet_connectivity__seed_3411.sub"
fi

#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

result_path="../../../runs/reviewer_ablation/dais/ablation_no_mdsm/seed_3407/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__dais__ablation_no_mdsm__seed_3407.sub"
else
  echo "submit ablation__dais__ablation_no_mdsm__seed_3407.sub"
  jsub < "ablation__dais__ablation_no_mdsm__seed_3407.sub"
fi

result_path="../../../runs/reviewer_ablation/dais/ablation_no_mdsm/seed_3408/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__dais__ablation_no_mdsm__seed_3408.sub"
else
  echo "submit ablation__dais__ablation_no_mdsm__seed_3408.sub"
  jsub < "ablation__dais__ablation_no_mdsm__seed_3408.sub"
fi

result_path="../../../runs/reviewer_ablation/dais/ablation_no_mdsm/seed_3409/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__dais__ablation_no_mdsm__seed_3409.sub"
else
  echo "submit ablation__dais__ablation_no_mdsm__seed_3409.sub"
  jsub < "ablation__dais__ablation_no_mdsm__seed_3409.sub"
fi

result_path="../../../runs/reviewer_ablation/dais/ablation_no_mdsm/seed_3410/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__dais__ablation_no_mdsm__seed_3410.sub"
else
  echo "submit ablation__dais__ablation_no_mdsm__seed_3410.sub"
  jsub < "ablation__dais__ablation_no_mdsm__seed_3410.sub"
fi

result_path="../../../runs/reviewer_ablation/dais/ablation_no_mdsm/seed_3411/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__dais__ablation_no_mdsm__seed_3411.sub"
else
  echo "submit ablation__dais__ablation_no_mdsm__seed_3411.sub"
  jsub < "ablation__dais__ablation_no_mdsm__seed_3411.sub"
fi

result_path="../../../runs/reviewer_ablation/dais/ablation_single_delay/seed_3407/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__dais__ablation_single_delay__seed_3407.sub"
else
  echo "submit ablation__dais__ablation_single_delay__seed_3407.sub"
  jsub < "ablation__dais__ablation_single_delay__seed_3407.sub"
fi

result_path="../../../runs/reviewer_ablation/dais/ablation_single_delay/seed_3408/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__dais__ablation_single_delay__seed_3408.sub"
else
  echo "submit ablation__dais__ablation_single_delay__seed_3408.sub"
  jsub < "ablation__dais__ablation_single_delay__seed_3408.sub"
fi

result_path="../../../runs/reviewer_ablation/dais/ablation_single_delay/seed_3409/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__dais__ablation_single_delay__seed_3409.sub"
else
  echo "submit ablation__dais__ablation_single_delay__seed_3409.sub"
  jsub < "ablation__dais__ablation_single_delay__seed_3409.sub"
fi

result_path="../../../runs/reviewer_ablation/dais/ablation_single_delay/seed_3410/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__dais__ablation_single_delay__seed_3410.sub"
else
  echo "submit ablation__dais__ablation_single_delay__seed_3410.sub"
  jsub < "ablation__dais__ablation_single_delay__seed_3410.sub"
fi

result_path="../../../runs/reviewer_ablation/dais/ablation_single_delay/seed_3411/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__dais__ablation_single_delay__seed_3411.sub"
else
  echo "submit ablation__dais__ablation_single_delay__seed_3411.sub"
  jsub < "ablation__dais__ablation_single_delay__seed_3411.sub"
fi

result_path="../../../runs/reviewer_ablation/dais/ablation_no_morphology/seed_3407/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__dais__ablation_no_morphology__seed_3407.sub"
else
  echo "submit ablation__dais__ablation_no_morphology__seed_3407.sub"
  jsub < "ablation__dais__ablation_no_morphology__seed_3407.sub"
fi

result_path="../../../runs/reviewer_ablation/dais/ablation_no_morphology/seed_3408/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__dais__ablation_no_morphology__seed_3408.sub"
else
  echo "submit ablation__dais__ablation_no_morphology__seed_3408.sub"
  jsub < "ablation__dais__ablation_no_morphology__seed_3408.sub"
fi

result_path="../../../runs/reviewer_ablation/dais/ablation_no_morphology/seed_3409/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__dais__ablation_no_morphology__seed_3409.sub"
else
  echo "submit ablation__dais__ablation_no_morphology__seed_3409.sub"
  jsub < "ablation__dais__ablation_no_morphology__seed_3409.sub"
fi

result_path="../../../runs/reviewer_ablation/dais/ablation_no_morphology/seed_3410/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__dais__ablation_no_morphology__seed_3410.sub"
else
  echo "submit ablation__dais__ablation_no_morphology__seed_3410.sub"
  jsub < "ablation__dais__ablation_no_morphology__seed_3410.sub"
fi

result_path="../../../runs/reviewer_ablation/dais/ablation_no_morphology/seed_3411/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__dais__ablation_no_morphology__seed_3411.sub"
else
  echo "submit ablation__dais__ablation_no_morphology__seed_3411.sub"
  jsub < "ablation__dais__ablation_no_morphology__seed_3411.sub"
fi

result_path="../../../runs/reviewer_ablation/dais/ablation_no_topology_tokens/seed_3407/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__dais__ablation_no_topology_tokens__seed_3407.sub"
else
  echo "submit ablation__dais__ablation_no_topology_tokens__seed_3407.sub"
  jsub < "ablation__dais__ablation_no_topology_tokens__seed_3407.sub"
fi

result_path="../../../runs/reviewer_ablation/dais/ablation_no_topology_tokens/seed_3408/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__dais__ablation_no_topology_tokens__seed_3408.sub"
else
  echo "submit ablation__dais__ablation_no_topology_tokens__seed_3408.sub"
  jsub < "ablation__dais__ablation_no_topology_tokens__seed_3408.sub"
fi

result_path="../../../runs/reviewer_ablation/dais/ablation_no_topology_tokens/seed_3409/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__dais__ablation_no_topology_tokens__seed_3409.sub"
else
  echo "submit ablation__dais__ablation_no_topology_tokens__seed_3409.sub"
  jsub < "ablation__dais__ablation_no_topology_tokens__seed_3409.sub"
fi

result_path="../../../runs/reviewer_ablation/dais/ablation_no_topology_tokens/seed_3410/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__dais__ablation_no_topology_tokens__seed_3410.sub"
else
  echo "submit ablation__dais__ablation_no_topology_tokens__seed_3410.sub"
  jsub < "ablation__dais__ablation_no_topology_tokens__seed_3410.sub"
fi

result_path="../../../runs/reviewer_ablation/dais/ablation_no_topology_tokens/seed_3411/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__dais__ablation_no_topology_tokens__seed_3411.sub"
else
  echo "submit ablation__dais__ablation_no_topology_tokens__seed_3411.sub"
  jsub < "ablation__dais__ablation_no_topology_tokens__seed_3411.sub"
fi

result_path="../../../runs/reviewer_ablation/dais/ablation_no_dual_cross/seed_3407/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__dais__ablation_no_dual_cross__seed_3407.sub"
else
  echo "submit ablation__dais__ablation_no_dual_cross__seed_3407.sub"
  jsub < "ablation__dais__ablation_no_dual_cross__seed_3407.sub"
fi

result_path="../../../runs/reviewer_ablation/dais/ablation_no_dual_cross/seed_3408/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__dais__ablation_no_dual_cross__seed_3408.sub"
else
  echo "submit ablation__dais__ablation_no_dual_cross__seed_3408.sub"
  jsub < "ablation__dais__ablation_no_dual_cross__seed_3408.sub"
fi

result_path="../../../runs/reviewer_ablation/dais/ablation_no_dual_cross/seed_3409/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__dais__ablation_no_dual_cross__seed_3409.sub"
else
  echo "submit ablation__dais__ablation_no_dual_cross__seed_3409.sub"
  jsub < "ablation__dais__ablation_no_dual_cross__seed_3409.sub"
fi

result_path="../../../runs/reviewer_ablation/dais/ablation_no_dual_cross/seed_3410/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__dais__ablation_no_dual_cross__seed_3410.sub"
else
  echo "submit ablation__dais__ablation_no_dual_cross__seed_3410.sub"
  jsub < "ablation__dais__ablation_no_dual_cross__seed_3410.sub"
fi

result_path="../../../runs/reviewer_ablation/dais/ablation_no_dual_cross/seed_3411/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__dais__ablation_no_dual_cross__seed_3411.sub"
else
  echo "submit ablation__dais__ablation_no_dual_cross__seed_3411.sub"
  jsub < "ablation__dais__ablation_no_dual_cross__seed_3411.sub"
fi

result_path="../../../runs/reviewer_ablation/dais/ablation_no_topology_loss/seed_3407/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__dais__ablation_no_topology_loss__seed_3407.sub"
else
  echo "submit ablation__dais__ablation_no_topology_loss__seed_3407.sub"
  jsub < "ablation__dais__ablation_no_topology_loss__seed_3407.sub"
fi

result_path="../../../runs/reviewer_ablation/dais/ablation_no_topology_loss/seed_3408/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__dais__ablation_no_topology_loss__seed_3408.sub"
else
  echo "submit ablation__dais__ablation_no_topology_loss__seed_3408.sub"
  jsub < "ablation__dais__ablation_no_topology_loss__seed_3408.sub"
fi

result_path="../../../runs/reviewer_ablation/dais/ablation_no_topology_loss/seed_3409/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__dais__ablation_no_topology_loss__seed_3409.sub"
else
  echo "submit ablation__dais__ablation_no_topology_loss__seed_3409.sub"
  jsub < "ablation__dais__ablation_no_topology_loss__seed_3409.sub"
fi

result_path="../../../runs/reviewer_ablation/dais/ablation_no_topology_loss/seed_3410/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__dais__ablation_no_topology_loss__seed_3410.sub"
else
  echo "submit ablation__dais__ablation_no_topology_loss__seed_3410.sub"
  jsub < "ablation__dais__ablation_no_topology_loss__seed_3410.sub"
fi

result_path="../../../runs/reviewer_ablation/dais/ablation_no_topology_loss/seed_3411/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__dais__ablation_no_topology_loss__seed_3411.sub"
else
  echo "submit ablation__dais__ablation_no_topology_loss__seed_3411.sub"
  jsub < "ablation__dais__ablation_no_topology_loss__seed_3411.sub"
fi

result_path="../../../runs/reviewer_ablation/dais/ablation_no_geometry_consistency/seed_3407/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__dais__ablation_no_geometry_consistency__seed_3407.sub"
else
  echo "submit ablation__dais__ablation_no_geometry_consistency__seed_3407.sub"
  jsub < "ablation__dais__ablation_no_geometry_consistency__seed_3407.sub"
fi

result_path="../../../runs/reviewer_ablation/dais/ablation_no_geometry_consistency/seed_3408/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__dais__ablation_no_geometry_consistency__seed_3408.sub"
else
  echo "submit ablation__dais__ablation_no_geometry_consistency__seed_3408.sub"
  jsub < "ablation__dais__ablation_no_geometry_consistency__seed_3408.sub"
fi

result_path="../../../runs/reviewer_ablation/dais/ablation_no_geometry_consistency/seed_3409/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__dais__ablation_no_geometry_consistency__seed_3409.sub"
else
  echo "submit ablation__dais__ablation_no_geometry_consistency__seed_3409.sub"
  jsub < "ablation__dais__ablation_no_geometry_consistency__seed_3409.sub"
fi

result_path="../../../runs/reviewer_ablation/dais/ablation_no_geometry_consistency/seed_3410/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__dais__ablation_no_geometry_consistency__seed_3410.sub"
else
  echo "submit ablation__dais__ablation_no_geometry_consistency__seed_3410.sub"
  jsub < "ablation__dais__ablation_no_geometry_consistency__seed_3410.sub"
fi

result_path="../../../runs/reviewer_ablation/dais/ablation_no_geometry_consistency/seed_3411/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__dais__ablation_no_geometry_consistency__seed_3411.sub"
else
  echo "submit ablation__dais__ablation_no_geometry_consistency__seed_3411.sub"
  jsub < "ablation__dais__ablation_no_geometry_consistency__seed_3411.sub"
fi

result_path="../../../runs/reviewer_ablation/dais/ablation_no_uncertainty/seed_3407/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__dais__ablation_no_uncertainty__seed_3407.sub"
else
  echo "submit ablation__dais__ablation_no_uncertainty__seed_3407.sub"
  jsub < "ablation__dais__ablation_no_uncertainty__seed_3407.sub"
fi

result_path="../../../runs/reviewer_ablation/dais/ablation_no_uncertainty/seed_3408/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__dais__ablation_no_uncertainty__seed_3408.sub"
else
  echo "submit ablation__dais__ablation_no_uncertainty__seed_3408.sub"
  jsub < "ablation__dais__ablation_no_uncertainty__seed_3408.sub"
fi

result_path="../../../runs/reviewer_ablation/dais/ablation_no_uncertainty/seed_3409/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__dais__ablation_no_uncertainty__seed_3409.sub"
else
  echo "submit ablation__dais__ablation_no_uncertainty__seed_3409.sub"
  jsub < "ablation__dais__ablation_no_uncertainty__seed_3409.sub"
fi

result_path="../../../runs/reviewer_ablation/dais/ablation_no_uncertainty/seed_3410/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__dais__ablation_no_uncertainty__seed_3410.sub"
else
  echo "submit ablation__dais__ablation_no_uncertainty__seed_3410.sub"
  jsub < "ablation__dais__ablation_no_uncertainty__seed_3410.sub"
fi

result_path="../../../runs/reviewer_ablation/dais/ablation_no_uncertainty/seed_3411/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__dais__ablation_no_uncertainty__seed_3411.sub"
else
  echo "submit ablation__dais__ablation_no_uncertainty__seed_3411.sub"
  jsub < "ablation__dais__ablation_no_uncertainty__seed_3411.sub"
fi

result_path="../../../runs/reviewer_ablation/deepcrack/ablation_no_mdsm/seed_3407/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__deepcrack__ablation_no_mdsm__seed_3407.sub"
else
  echo "submit ablation__deepcrack__ablation_no_mdsm__seed_3407.sub"
  jsub < "ablation__deepcrack__ablation_no_mdsm__seed_3407.sub"
fi

result_path="../../../runs/reviewer_ablation/deepcrack/ablation_no_mdsm/seed_3408/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__deepcrack__ablation_no_mdsm__seed_3408.sub"
else
  echo "submit ablation__deepcrack__ablation_no_mdsm__seed_3408.sub"
  jsub < "ablation__deepcrack__ablation_no_mdsm__seed_3408.sub"
fi

result_path="../../../runs/reviewer_ablation/deepcrack/ablation_no_mdsm/seed_3409/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__deepcrack__ablation_no_mdsm__seed_3409.sub"
else
  echo "submit ablation__deepcrack__ablation_no_mdsm__seed_3409.sub"
  jsub < "ablation__deepcrack__ablation_no_mdsm__seed_3409.sub"
fi

result_path="../../../runs/reviewer_ablation/deepcrack/ablation_no_mdsm/seed_3410/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__deepcrack__ablation_no_mdsm__seed_3410.sub"
else
  echo "submit ablation__deepcrack__ablation_no_mdsm__seed_3410.sub"
  jsub < "ablation__deepcrack__ablation_no_mdsm__seed_3410.sub"
fi

result_path="../../../runs/reviewer_ablation/deepcrack/ablation_no_mdsm/seed_3411/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__deepcrack__ablation_no_mdsm__seed_3411.sub"
else
  echo "submit ablation__deepcrack__ablation_no_mdsm__seed_3411.sub"
  jsub < "ablation__deepcrack__ablation_no_mdsm__seed_3411.sub"
fi

result_path="../../../runs/reviewer_ablation/deepcrack/ablation_single_delay/seed_3407/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__deepcrack__ablation_single_delay__seed_3407.sub"
else
  echo "submit ablation__deepcrack__ablation_single_delay__seed_3407.sub"
  jsub < "ablation__deepcrack__ablation_single_delay__seed_3407.sub"
fi

result_path="../../../runs/reviewer_ablation/deepcrack/ablation_single_delay/seed_3408/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__deepcrack__ablation_single_delay__seed_3408.sub"
else
  echo "submit ablation__deepcrack__ablation_single_delay__seed_3408.sub"
  jsub < "ablation__deepcrack__ablation_single_delay__seed_3408.sub"
fi

result_path="../../../runs/reviewer_ablation/deepcrack/ablation_single_delay/seed_3409/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__deepcrack__ablation_single_delay__seed_3409.sub"
else
  echo "submit ablation__deepcrack__ablation_single_delay__seed_3409.sub"
  jsub < "ablation__deepcrack__ablation_single_delay__seed_3409.sub"
fi

result_path="../../../runs/reviewer_ablation/deepcrack/ablation_single_delay/seed_3410/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__deepcrack__ablation_single_delay__seed_3410.sub"
else
  echo "submit ablation__deepcrack__ablation_single_delay__seed_3410.sub"
  jsub < "ablation__deepcrack__ablation_single_delay__seed_3410.sub"
fi

result_path="../../../runs/reviewer_ablation/deepcrack/ablation_single_delay/seed_3411/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__deepcrack__ablation_single_delay__seed_3411.sub"
else
  echo "submit ablation__deepcrack__ablation_single_delay__seed_3411.sub"
  jsub < "ablation__deepcrack__ablation_single_delay__seed_3411.sub"
fi

result_path="../../../runs/reviewer_ablation/deepcrack/ablation_no_morphology/seed_3407/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__deepcrack__ablation_no_morphology__seed_3407.sub"
else
  echo "submit ablation__deepcrack__ablation_no_morphology__seed_3407.sub"
  jsub < "ablation__deepcrack__ablation_no_morphology__seed_3407.sub"
fi

result_path="../../../runs/reviewer_ablation/deepcrack/ablation_no_morphology/seed_3408/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__deepcrack__ablation_no_morphology__seed_3408.sub"
else
  echo "submit ablation__deepcrack__ablation_no_morphology__seed_3408.sub"
  jsub < "ablation__deepcrack__ablation_no_morphology__seed_3408.sub"
fi

result_path="../../../runs/reviewer_ablation/deepcrack/ablation_no_morphology/seed_3409/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__deepcrack__ablation_no_morphology__seed_3409.sub"
else
  echo "submit ablation__deepcrack__ablation_no_morphology__seed_3409.sub"
  jsub < "ablation__deepcrack__ablation_no_morphology__seed_3409.sub"
fi

result_path="../../../runs/reviewer_ablation/deepcrack/ablation_no_morphology/seed_3410/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__deepcrack__ablation_no_morphology__seed_3410.sub"
else
  echo "submit ablation__deepcrack__ablation_no_morphology__seed_3410.sub"
  jsub < "ablation__deepcrack__ablation_no_morphology__seed_3410.sub"
fi

result_path="../../../runs/reviewer_ablation/deepcrack/ablation_no_morphology/seed_3411/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__deepcrack__ablation_no_morphology__seed_3411.sub"
else
  echo "submit ablation__deepcrack__ablation_no_morphology__seed_3411.sub"
  jsub < "ablation__deepcrack__ablation_no_morphology__seed_3411.sub"
fi

result_path="../../../runs/reviewer_ablation/deepcrack/ablation_no_topology_tokens/seed_3407/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__deepcrack__ablation_no_topology_tokens__seed_3407.sub"
else
  echo "submit ablation__deepcrack__ablation_no_topology_tokens__seed_3407.sub"
  jsub < "ablation__deepcrack__ablation_no_topology_tokens__seed_3407.sub"
fi

result_path="../../../runs/reviewer_ablation/deepcrack/ablation_no_topology_tokens/seed_3408/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__deepcrack__ablation_no_topology_tokens__seed_3408.sub"
else
  echo "submit ablation__deepcrack__ablation_no_topology_tokens__seed_3408.sub"
  jsub < "ablation__deepcrack__ablation_no_topology_tokens__seed_3408.sub"
fi

result_path="../../../runs/reviewer_ablation/deepcrack/ablation_no_topology_tokens/seed_3409/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__deepcrack__ablation_no_topology_tokens__seed_3409.sub"
else
  echo "submit ablation__deepcrack__ablation_no_topology_tokens__seed_3409.sub"
  jsub < "ablation__deepcrack__ablation_no_topology_tokens__seed_3409.sub"
fi

result_path="../../../runs/reviewer_ablation/deepcrack/ablation_no_topology_tokens/seed_3410/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__deepcrack__ablation_no_topology_tokens__seed_3410.sub"
else
  echo "submit ablation__deepcrack__ablation_no_topology_tokens__seed_3410.sub"
  jsub < "ablation__deepcrack__ablation_no_topology_tokens__seed_3410.sub"
fi

result_path="../../../runs/reviewer_ablation/deepcrack/ablation_no_topology_tokens/seed_3411/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__deepcrack__ablation_no_topology_tokens__seed_3411.sub"
else
  echo "submit ablation__deepcrack__ablation_no_topology_tokens__seed_3411.sub"
  jsub < "ablation__deepcrack__ablation_no_topology_tokens__seed_3411.sub"
fi

result_path="../../../runs/reviewer_ablation/deepcrack/ablation_no_dual_cross/seed_3407/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__deepcrack__ablation_no_dual_cross__seed_3407.sub"
else
  echo "submit ablation__deepcrack__ablation_no_dual_cross__seed_3407.sub"
  jsub < "ablation__deepcrack__ablation_no_dual_cross__seed_3407.sub"
fi

result_path="../../../runs/reviewer_ablation/deepcrack/ablation_no_dual_cross/seed_3408/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__deepcrack__ablation_no_dual_cross__seed_3408.sub"
else
  echo "submit ablation__deepcrack__ablation_no_dual_cross__seed_3408.sub"
  jsub < "ablation__deepcrack__ablation_no_dual_cross__seed_3408.sub"
fi

result_path="../../../runs/reviewer_ablation/deepcrack/ablation_no_dual_cross/seed_3409/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__deepcrack__ablation_no_dual_cross__seed_3409.sub"
else
  echo "submit ablation__deepcrack__ablation_no_dual_cross__seed_3409.sub"
  jsub < "ablation__deepcrack__ablation_no_dual_cross__seed_3409.sub"
fi

result_path="../../../runs/reviewer_ablation/deepcrack/ablation_no_dual_cross/seed_3410/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__deepcrack__ablation_no_dual_cross__seed_3410.sub"
else
  echo "submit ablation__deepcrack__ablation_no_dual_cross__seed_3410.sub"
  jsub < "ablation__deepcrack__ablation_no_dual_cross__seed_3410.sub"
fi

result_path="../../../runs/reviewer_ablation/deepcrack/ablation_no_dual_cross/seed_3411/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__deepcrack__ablation_no_dual_cross__seed_3411.sub"
else
  echo "submit ablation__deepcrack__ablation_no_dual_cross__seed_3411.sub"
  jsub < "ablation__deepcrack__ablation_no_dual_cross__seed_3411.sub"
fi

result_path="../../../runs/reviewer_ablation/deepcrack/ablation_no_topology_loss/seed_3407/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__deepcrack__ablation_no_topology_loss__seed_3407.sub"
else
  echo "submit ablation__deepcrack__ablation_no_topology_loss__seed_3407.sub"
  jsub < "ablation__deepcrack__ablation_no_topology_loss__seed_3407.sub"
fi

result_path="../../../runs/reviewer_ablation/deepcrack/ablation_no_topology_loss/seed_3408/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__deepcrack__ablation_no_topology_loss__seed_3408.sub"
else
  echo "submit ablation__deepcrack__ablation_no_topology_loss__seed_3408.sub"
  jsub < "ablation__deepcrack__ablation_no_topology_loss__seed_3408.sub"
fi

result_path="../../../runs/reviewer_ablation/deepcrack/ablation_no_topology_loss/seed_3409/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__deepcrack__ablation_no_topology_loss__seed_3409.sub"
else
  echo "submit ablation__deepcrack__ablation_no_topology_loss__seed_3409.sub"
  jsub < "ablation__deepcrack__ablation_no_topology_loss__seed_3409.sub"
fi

result_path="../../../runs/reviewer_ablation/deepcrack/ablation_no_topology_loss/seed_3410/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__deepcrack__ablation_no_topology_loss__seed_3410.sub"
else
  echo "submit ablation__deepcrack__ablation_no_topology_loss__seed_3410.sub"
  jsub < "ablation__deepcrack__ablation_no_topology_loss__seed_3410.sub"
fi

result_path="../../../runs/reviewer_ablation/deepcrack/ablation_no_topology_loss/seed_3411/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__deepcrack__ablation_no_topology_loss__seed_3411.sub"
else
  echo "submit ablation__deepcrack__ablation_no_topology_loss__seed_3411.sub"
  jsub < "ablation__deepcrack__ablation_no_topology_loss__seed_3411.sub"
fi

result_path="../../../runs/reviewer_ablation/deepcrack/ablation_no_geometry_consistency/seed_3407/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__deepcrack__ablation_no_geometry_consistency__seed_3407.sub"
else
  echo "submit ablation__deepcrack__ablation_no_geometry_consistency__seed_3407.sub"
  jsub < "ablation__deepcrack__ablation_no_geometry_consistency__seed_3407.sub"
fi

result_path="../../../runs/reviewer_ablation/deepcrack/ablation_no_geometry_consistency/seed_3408/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__deepcrack__ablation_no_geometry_consistency__seed_3408.sub"
else
  echo "submit ablation__deepcrack__ablation_no_geometry_consistency__seed_3408.sub"
  jsub < "ablation__deepcrack__ablation_no_geometry_consistency__seed_3408.sub"
fi

result_path="../../../runs/reviewer_ablation/deepcrack/ablation_no_geometry_consistency/seed_3409/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__deepcrack__ablation_no_geometry_consistency__seed_3409.sub"
else
  echo "submit ablation__deepcrack__ablation_no_geometry_consistency__seed_3409.sub"
  jsub < "ablation__deepcrack__ablation_no_geometry_consistency__seed_3409.sub"
fi

result_path="../../../runs/reviewer_ablation/deepcrack/ablation_no_geometry_consistency/seed_3410/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__deepcrack__ablation_no_geometry_consistency__seed_3410.sub"
else
  echo "submit ablation__deepcrack__ablation_no_geometry_consistency__seed_3410.sub"
  jsub < "ablation__deepcrack__ablation_no_geometry_consistency__seed_3410.sub"
fi

result_path="../../../runs/reviewer_ablation/deepcrack/ablation_no_geometry_consistency/seed_3411/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__deepcrack__ablation_no_geometry_consistency__seed_3411.sub"
else
  echo "submit ablation__deepcrack__ablation_no_geometry_consistency__seed_3411.sub"
  jsub < "ablation__deepcrack__ablation_no_geometry_consistency__seed_3411.sub"
fi

result_path="../../../runs/reviewer_ablation/deepcrack/ablation_no_uncertainty/seed_3407/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__deepcrack__ablation_no_uncertainty__seed_3407.sub"
else
  echo "submit ablation__deepcrack__ablation_no_uncertainty__seed_3407.sub"
  jsub < "ablation__deepcrack__ablation_no_uncertainty__seed_3407.sub"
fi

result_path="../../../runs/reviewer_ablation/deepcrack/ablation_no_uncertainty/seed_3408/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__deepcrack__ablation_no_uncertainty__seed_3408.sub"
else
  echo "submit ablation__deepcrack__ablation_no_uncertainty__seed_3408.sub"
  jsub < "ablation__deepcrack__ablation_no_uncertainty__seed_3408.sub"
fi

result_path="../../../runs/reviewer_ablation/deepcrack/ablation_no_uncertainty/seed_3409/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__deepcrack__ablation_no_uncertainty__seed_3409.sub"
else
  echo "submit ablation__deepcrack__ablation_no_uncertainty__seed_3409.sub"
  jsub < "ablation__deepcrack__ablation_no_uncertainty__seed_3409.sub"
fi

result_path="../../../runs/reviewer_ablation/deepcrack/ablation_no_uncertainty/seed_3410/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__deepcrack__ablation_no_uncertainty__seed_3410.sub"
else
  echo "submit ablation__deepcrack__ablation_no_uncertainty__seed_3410.sub"
  jsub < "ablation__deepcrack__ablation_no_uncertainty__seed_3410.sub"
fi

result_path="../../../runs/reviewer_ablation/deepcrack/ablation_no_uncertainty/seed_3411/test_metrics.json"
if [ -f "$result_path" ]; then
  echo "skip completed ablation__deepcrack__ablation_no_uncertainty__seed_3411.sub"
else
  echo "submit ablation__deepcrack__ablation_no_uncertainty__seed_3411.sub"
  jsub < "ablation__deepcrack__ablation_no_uncertainty__seed_3411.sub"
fi

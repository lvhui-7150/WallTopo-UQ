#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

result_path="../../../runs/reviewer_crossdomain/pavement_to_masonry/walltopo_uq/seed_3407/cross_domain_summary.json"
if [ -f "$result_path" ]; then
  echo "skip completed crossdomain__pavement_to_masonry__walltopo_uq__seed_3407.sub"
else
  echo "submit crossdomain__pavement_to_masonry__walltopo_uq__seed_3407.sub"
  jsub < "crossdomain__pavement_to_masonry__walltopo_uq__seed_3407.sub"
fi

result_path="../../../runs/reviewer_crossdomain/pavement_to_masonry/walltopo_uq/seed_3408/cross_domain_summary.json"
if [ -f "$result_path" ]; then
  echo "skip completed crossdomain__pavement_to_masonry__walltopo_uq__seed_3408.sub"
else
  echo "submit crossdomain__pavement_to_masonry__walltopo_uq__seed_3408.sub"
  jsub < "crossdomain__pavement_to_masonry__walltopo_uq__seed_3408.sub"
fi

result_path="../../../runs/reviewer_crossdomain/pavement_to_masonry/walltopo_uq/seed_3409/cross_domain_summary.json"
if [ -f "$result_path" ]; then
  echo "skip completed crossdomain__pavement_to_masonry__walltopo_uq__seed_3409.sub"
else
  echo "submit crossdomain__pavement_to_masonry__walltopo_uq__seed_3409.sub"
  jsub < "crossdomain__pavement_to_masonry__walltopo_uq__seed_3409.sub"
fi

result_path="../../../runs/reviewer_crossdomain/pavement_to_masonry/walltopo_uq/seed_3410/cross_domain_summary.json"
if [ -f "$result_path" ]; then
  echo "skip completed crossdomain__pavement_to_masonry__walltopo_uq__seed_3410.sub"
else
  echo "submit crossdomain__pavement_to_masonry__walltopo_uq__seed_3410.sub"
  jsub < "crossdomain__pavement_to_masonry__walltopo_uq__seed_3410.sub"
fi

result_path="../../../runs/reviewer_crossdomain/pavement_to_masonry/walltopo_uq/seed_3411/cross_domain_summary.json"
if [ -f "$result_path" ]; then
  echo "skip completed crossdomain__pavement_to_masonry__walltopo_uq__seed_3411.sub"
else
  echo "submit crossdomain__pavement_to_masonry__walltopo_uq__seed_3411.sub"
  jsub < "crossdomain__pavement_to_masonry__walltopo_uq__seed_3411.sub"
fi

result_path="../../../runs/reviewer_crossdomain/pavement_to_masonry/unet/seed_3407/cross_domain_summary.json"
if [ -f "$result_path" ]; then
  echo "skip completed crossdomain__pavement_to_masonry__unet__seed_3407.sub"
else
  echo "submit crossdomain__pavement_to_masonry__unet__seed_3407.sub"
  jsub < "crossdomain__pavement_to_masonry__unet__seed_3407.sub"
fi

result_path="../../../runs/reviewer_crossdomain/pavement_to_masonry/unet/seed_3408/cross_domain_summary.json"
if [ -f "$result_path" ]; then
  echo "skip completed crossdomain__pavement_to_masonry__unet__seed_3408.sub"
else
  echo "submit crossdomain__pavement_to_masonry__unet__seed_3408.sub"
  jsub < "crossdomain__pavement_to_masonry__unet__seed_3408.sub"
fi

result_path="../../../runs/reviewer_crossdomain/pavement_to_masonry/unet/seed_3409/cross_domain_summary.json"
if [ -f "$result_path" ]; then
  echo "skip completed crossdomain__pavement_to_masonry__unet__seed_3409.sub"
else
  echo "submit crossdomain__pavement_to_masonry__unet__seed_3409.sub"
  jsub < "crossdomain__pavement_to_masonry__unet__seed_3409.sub"
fi

result_path="../../../runs/reviewer_crossdomain/pavement_to_masonry/unet/seed_3410/cross_domain_summary.json"
if [ -f "$result_path" ]; then
  echo "skip completed crossdomain__pavement_to_masonry__unet__seed_3410.sub"
else
  echo "submit crossdomain__pavement_to_masonry__unet__seed_3410.sub"
  jsub < "crossdomain__pavement_to_masonry__unet__seed_3410.sub"
fi

result_path="../../../runs/reviewer_crossdomain/pavement_to_masonry/unet/seed_3411/cross_domain_summary.json"
if [ -f "$result_path" ]; then
  echo "skip completed crossdomain__pavement_to_masonry__unet__seed_3411.sub"
else
  echo "submit crossdomain__pavement_to_masonry__unet__seed_3411.sub"
  jsub < "crossdomain__pavement_to_masonry__unet__seed_3411.sub"
fi

result_path="../../../runs/reviewer_crossdomain/masonry_to_pavement/walltopo_uq/seed_3407/cross_domain_summary.json"
if [ -f "$result_path" ]; then
  echo "skip completed crossdomain__masonry_to_pavement__walltopo_uq__seed_3407.sub"
else
  echo "submit crossdomain__masonry_to_pavement__walltopo_uq__seed_3407.sub"
  jsub < "crossdomain__masonry_to_pavement__walltopo_uq__seed_3407.sub"
fi

result_path="../../../runs/reviewer_crossdomain/masonry_to_pavement/walltopo_uq/seed_3408/cross_domain_summary.json"
if [ -f "$result_path" ]; then
  echo "skip completed crossdomain__masonry_to_pavement__walltopo_uq__seed_3408.sub"
else
  echo "submit crossdomain__masonry_to_pavement__walltopo_uq__seed_3408.sub"
  jsub < "crossdomain__masonry_to_pavement__walltopo_uq__seed_3408.sub"
fi

result_path="../../../runs/reviewer_crossdomain/masonry_to_pavement/walltopo_uq/seed_3409/cross_domain_summary.json"
if [ -f "$result_path" ]; then
  echo "skip completed crossdomain__masonry_to_pavement__walltopo_uq__seed_3409.sub"
else
  echo "submit crossdomain__masonry_to_pavement__walltopo_uq__seed_3409.sub"
  jsub < "crossdomain__masonry_to_pavement__walltopo_uq__seed_3409.sub"
fi

result_path="../../../runs/reviewer_crossdomain/masonry_to_pavement/walltopo_uq/seed_3410/cross_domain_summary.json"
if [ -f "$result_path" ]; then
  echo "skip completed crossdomain__masonry_to_pavement__walltopo_uq__seed_3410.sub"
else
  echo "submit crossdomain__masonry_to_pavement__walltopo_uq__seed_3410.sub"
  jsub < "crossdomain__masonry_to_pavement__walltopo_uq__seed_3410.sub"
fi

result_path="../../../runs/reviewer_crossdomain/masonry_to_pavement/walltopo_uq/seed_3411/cross_domain_summary.json"
if [ -f "$result_path" ]; then
  echo "skip completed crossdomain__masonry_to_pavement__walltopo_uq__seed_3411.sub"
else
  echo "submit crossdomain__masonry_to_pavement__walltopo_uq__seed_3411.sub"
  jsub < "crossdomain__masonry_to_pavement__walltopo_uq__seed_3411.sub"
fi

result_path="../../../runs/reviewer_crossdomain/masonry_to_pavement/unet/seed_3407/cross_domain_summary.json"
if [ -f "$result_path" ]; then
  echo "skip completed crossdomain__masonry_to_pavement__unet__seed_3407.sub"
else
  echo "submit crossdomain__masonry_to_pavement__unet__seed_3407.sub"
  jsub < "crossdomain__masonry_to_pavement__unet__seed_3407.sub"
fi

result_path="../../../runs/reviewer_crossdomain/masonry_to_pavement/unet/seed_3408/cross_domain_summary.json"
if [ -f "$result_path" ]; then
  echo "skip completed crossdomain__masonry_to_pavement__unet__seed_3408.sub"
else
  echo "submit crossdomain__masonry_to_pavement__unet__seed_3408.sub"
  jsub < "crossdomain__masonry_to_pavement__unet__seed_3408.sub"
fi

result_path="../../../runs/reviewer_crossdomain/masonry_to_pavement/unet/seed_3409/cross_domain_summary.json"
if [ -f "$result_path" ]; then
  echo "skip completed crossdomain__masonry_to_pavement__unet__seed_3409.sub"
else
  echo "submit crossdomain__masonry_to_pavement__unet__seed_3409.sub"
  jsub < "crossdomain__masonry_to_pavement__unet__seed_3409.sub"
fi

result_path="../../../runs/reviewer_crossdomain/masonry_to_pavement/unet/seed_3410/cross_domain_summary.json"
if [ -f "$result_path" ]; then
  echo "skip completed crossdomain__masonry_to_pavement__unet__seed_3410.sub"
else
  echo "submit crossdomain__masonry_to_pavement__unet__seed_3410.sub"
  jsub < "crossdomain__masonry_to_pavement__unet__seed_3410.sub"
fi

result_path="../../../runs/reviewer_crossdomain/masonry_to_pavement/unet/seed_3411/cross_domain_summary.json"
if [ -f "$result_path" ]; then
  echo "skip completed crossdomain__masonry_to_pavement__unet__seed_3411.sub"
else
  echo "submit crossdomain__masonry_to_pavement__unet__seed_3411.sub"
  jsub < "crossdomain__masonry_to_pavement__unet__seed_3411.sub"
fi

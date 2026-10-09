# WallTopo-UQ Experiments

This repository contains the source code and experiment templates only.
Datasets, trained checkpoints, run outputs, and generated figures are excluded
by `.gitignore`.

This directory is a runnable PyTorch implementation of the WallTopo-UQ
manuscript:

1. MDSM: morphology-conditioned delayed state memory.
2. TCD: topology-cross decoder with four topology token groups.
3. MGE: mask, centerline, endpoint, junction, orientation, width, and ordinal
   severity heads.
4. EUS: evidential pixel uncertainty and report-level uncertainty utilities.

The code is intentionally self-contained. It uses no `timm`, `einops`, or
external segmentation framework.

Chinese guides:

- [DATASETS_ZH.md](DATASETS_ZH.md): dataset priorities, official sources, and
  manifest preparation.
- [EXPERIMENT_GUIDE_ZH.md](EXPERIMENT_GUIDE_ZH.md): full baseline, ablation,
  cross-domain, uncertainty, and reproducibility workflow.

## Environment

The current machine has been validated with:

- Python 3.13.5
- PyTorch 2.6.0+cu124
- NVIDIA RTX 3070 Laptop GPU

Install missing dependencies with:

```powershell
python -m pip install -r requirements.txt
```

## One-command Smoke Test

From this directory:

```powershell
.\run_smoke.ps1
```

The script generates deterministic synthetic wall images, trains for two
epochs, evaluates the checkpoint, and writes prediction panels. Synthetic data
is only for software validation and must not be used for publication claims.

The current two-epoch acceptance run reaches approximately `Dice=0.435` and
`clDice=0.700` on the synthetic test split. These numbers are not paper
results; they only confirm that the implementation is learning.

Public data has also been downloaded and prepared under `data/raw`:

- Dais masonry: 240 pairs;
- DeepCrack: 537 pairs;
- Crack500: 471 pairs from a public mirror;
- CrackForest: 118 pairs.

See `data/raw/README.md` for source and license notes. A one-epoch real-data
acceptance run on Dais reached approximately `Dice=0.603` on its held-out test
split; this is still an implementation check, not a final paper result.

## Manual Commands

```powershell
python scripts/make_synthetic_data.py --output data/synthetic --size 128 --train 64 --val 16 --test 16
python scripts/check_dataset.py --config configs/smoke.yaml --split train
python scripts/train.py --config configs/smoke.yaml --device cuda
python scripts/evaluate.py --config configs/smoke.yaml --checkpoint runs/smoke/checkpoints/best.pt --split test --device cuda
python scripts/predict.py --config configs/smoke.yaml --checkpoint runs/smoke/checkpoints/best.pt --split test --save-panels --device cuda
pytest -q
```

## Baselines And Ablations

The unified model registry supports:

```text
model_name: walltopo_uq
model_name: unet
model_name: deeplabv3plus
```

Run the small baseline matrix:

```powershell
python scripts/run_suite.py --suite core --device auto
python scripts/run_suite.py --suite ablations --device auto
```

The matrix writes generated configs under `configs/generated/smoke_matrix` and
results under `runs/smoke_matrix`. Formal runs should use `configs/full.yaml`,
the same split, identical augmentation, and at least five random seeds.

## Publication-Grade Benchmark Pipeline

The full experiment design, statistical rules, SOTA criteria, and run order are
documented in [EXPERIMENT_PROTOCOL_ZH.md](EXPERIMENT_PROTOCOL_ZH.md).

Pilot run:

```powershell
.\run_pilot.ps1
```

Full five-seed benchmark and ablations:

```powershell
.\run_full_experiments.ps1
```

The benchmark runner supports resumable jobs, validation-selected thresholds,
EMA checkpoints, paired statistical tests, LaTeX tables, and 600-dpi
publication figures. The full suite is computationally expensive on one 8 GB
GPU; run it over several sessions with `--resume`.

## Real Dataset Setup

The dataset contract is a CSV manifest with these columns:

```text
image,mask,severity,group,domain,material,device
```

`image` and `mask` are required. `severity` may be empty when expert labels are
unavailable. `group` must identify a wall, building, or sequence and is used to
prevent leakage between splits.

### Dais Masonry Public Subset

Clone or download:

```text
https://github.com/dimitrisdais/crack_detection_CNN_masonry
```

The public repository contains 240 image-mask pairs. Build a manifest:

```powershell
python scripts/build_manifest.py `
  --images path/to/crack_detection_CNN_masonry/dataset/crack_detection_224_images `
  --masks path/to/crack_detection_CNN_masonry/dataset/crack_detection_224_masks `
  --output data/real/all.csv
```

The public subset is too small for a final training run. Obtain the full Dais
dataset or combine it with additional masonry data.

### General Folder Dataset

```powershell
python scripts/build_manifest.py `
  --images data/raw/my_dataset/images `
  --masks data/raw/my_dataset/masks `
  --mask-suffix "" `
  --recursive `
  --output data/real/all.csv
```

Create leakage-aware splits:

```powershell
python scripts/make_splits.py `
  --manifest data/real/all.csv `
  --output-dir data/real `
  --group-column group
```

Then update `configs/full.yaml` so the three manifests point to:

```text
data/real/train.csv
data/real/val.csv
data/real/test.csv
```

For cross-domain metadata:

```powershell
python scripts/annotate_manifest.py `
  --manifest data/real/all.csv `
  --domain masonry `
  --material brick `
  --device camera `
  --group-prefix dais

python scripts/make_domain_folds.py `
  --manifest data/real/all.csv `
  --output-dir data/real/folds `
  --domain-column domain `
  --group-column group
```

### Optional Manual Targets

The manifest may include additional columns:

```text
skeleton,endpoint,junction,orientation,width
```

Use these columns when independent manual or scale-calibrated targets are
available. Automatic targets are suitable for training but are not a substitute
for an independent test standard.

## Ablation Experiments

The manuscript's key ablations can be run by editing a copied config:

- Remove MDSM: set `mdsm_stages: []`.
- Single delay: set `delays: [1]`.
- Reduce multi-delay memory: use `delays: [1, 2]`.
- Disable topology target groups: set the corresponding `use_*` flag in `data`.
- Disable geometry consistency: set `loss.width_consistency: 0`.
- Disable orientation loss: set `loss.orientation: 0`.
- Disable topology loss: set `loss.topo: 0`.
- Disable evidence calibration: set `loss.evidence: 0`.
- Replace evidential scoring with mask probability when reporting selective
  prediction; the utility layer supports both.
- Disable report-level calibration with `uncertainty.calibrate: false`.
- Change TTA cost with `uncertainty.tta_views`.

Run each variant into a unique `output_dir`, then aggregate:

```powershell
python scripts/aggregate_results.py --root runs --pattern "test_metrics.json"
```

## Report-Level Uncertainty

Training writes `report_calibration.npz/json`. The calibration uses only the
training split to fit the domain-reference feature distribution and only the
validation split to select `tau_review`. Evaluation and prediction then report:

- pixel vacuity;
- centerline/topology instability over identity, horizontal flip, vertical
  flip, and 180-degree rotation;
- geometric dispersion over width and centerline length;
- normalized feature-space domain shift;
- `U_report` and `manual_review`.

Evaluation also reports AURC, error-detection AUROC, selective Dice/error, and
selective width error.

## Cross-Domain Evaluation

```powershell
python scripts/cross_domain.py `
  --config configs/full.yaml `
  --checkpoint runs/full/checkpoints/best.pt `
  --manifest data/real/test.csv `
  --split test `
  --device auto
```

Every non-empty `domain` in the manifest is evaluated separately and a macro
mean is written to `cross_domain_metrics.json`.

## Inference On New Unlabeled Images

`scripts/infer_images.py` accepts an image file or folder and does not require a
ground-truth mask:

```powershell
python scripts/infer_images.py `
  --input path/to/wall_photos `
  --output runs/dais/inference_demo `
  --config configs/dais.yaml `
  --checkpoint runs/dais/checkpoints/best.pt `
  --threshold 0.5 `
  --device auto
```

Each input image produces:

```text
<name>_overlay.png
<name>_mask.png
<name>_skeleton.png
<name>_uncertainty.png
<name>_panel.png
```

The folder also contains `inference_report.json` and `inference_summary.csv`,
including crack area, length, width percentiles, component count, visual
screening score, uncertainty terms, and an optional manual-review flag.

## Manuscript Figures

After training Dais and DeepCrack, regenerate the current manuscript figures:

```powershell
python scripts/make_paper_figures.py --device auto
```

The script reads `runs/dais/history.csv` and `runs/deepcrack/history.csv`,
selects median test-Dice examples, and writes `fig_training_curves` and
`fig_qualitative_results` as PDF and 300-DPI PNG files to the paper-level
`figures` directory.

## Outputs

Each run contains:

```text
model_summary.json
history.csv
events.jsonl
training_summary.json
checkpoints/best.pt
checkpoints/last.pt
test_metrics.json
predictions/predictions.json
predictions/panels/*.png
```

Metrics include Dice, IoU, boundary F1, clDice, skeleton Dice, endpoint and
junction F1 and count errors, component errors, break/false-merge rates,
Betti errors, length error, width MAE, orientation MAE, ECE, Brier, NLL,
risk-coverage AURC, selective metrics, and ordinal severity metrics when labels
exist.

## Scientific Guardrails

- Do not randomly split patches from the same wall into different sets.
- Do not use mask-derived skeleton or width as an independent test claim.
- Do not present residual structural capacity, depth, or safety from RGB-only
  data.
- Do not claim millimeter units without a valid scale reference.
- Do not use synthetic data for publication performance claims.
- Report multiple random seeds and confidence intervals for final results.
- Keep report-level calibration weights, TTA set, and review thresholds fixed
  across compared models.

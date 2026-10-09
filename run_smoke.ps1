$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

function Invoke-Python {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE`: python $($Arguments -join ' ')"
    }
}

Invoke-Python scripts/make_synthetic_data.py `
    --output data/synthetic `
    --size 128 `
    --train 32 `
    --val 8 `
    --test 8

Invoke-Python scripts/check_dataset.py --config configs/smoke.yaml --split train
Invoke-Python scripts/train.py --config configs/smoke.yaml --device auto
Invoke-Python scripts/evaluate.py `
    --config configs/smoke.yaml `
    --checkpoint runs/smoke/checkpoints/best.pt `
    --split test `
    --device auto
Invoke-Python scripts/predict.py `
    --config configs/smoke.yaml `
    --checkpoint runs/smoke/checkpoints/best.pt `
    --split test `
    --save-panels `
    --device auto

Write-Host "Smoke experiment completed."

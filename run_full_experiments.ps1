$ErrorActionPreference = "Stop"

function Invoke-Checked {
    param([scriptblock]$Command)
    $text = $Command.ToString().Trim()
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $text"
    }
}

Set-Location $PSScriptRoot

Invoke-Checked { python -m pytest -q }
Invoke-Checked { python scripts\run_benchmark_suite.py `
    --profile full `
    --suite core `
    --datasets dais deepcrack crack500 crackforest `
    --seeds 3407 3408 3409 3410 3411 `
    --run-prefix benchmark `
    --resume `
    --device cuda }
Invoke-Checked { python scripts\run_benchmark_suite.py `
    --profile full `
    --suite ablations `
    --datasets dais deepcrack `
    --seeds 3407 3408 3409 3410 3411 `
    --run-prefix benchmark `
    --resume `
    --device cuda }
Invoke-Checked { python scripts\benchmark_models.py `
    --config configs\dais.yaml `
    --models walltopo_uq unet deeplabv3plus `
    --size 224 `
    --batch-size 2 `
    --iterations 10 `
    --warmup 3 `
    --device cuda `
    --output results\benchmark\efficiency.csv }
Invoke-Checked { python scripts\collect_benchmark_results.py `
    --root runs\benchmark `
    --output results\benchmark }
Invoke-Checked { python scripts\make_publication_figures.py `
    --results results\benchmark `
    --output ..\figures\publication }

Write-Host "Full suite completed."

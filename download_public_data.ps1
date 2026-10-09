param(
    [switch]$IncludeSDNET
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$SourceRoot = Join-Path $Root "data\raw\sources"
New-Item -ItemType Directory -Force -Path $SourceRoot | Out-Null

function Invoke-Curl {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$Output
    )
    if (Test-Path -LiteralPath $Output) {
        Write-Host "skip existing: $Output"
        return
    }
    & curl.exe -L --fail --connect-timeout 20 --max-time 3600 `
        --retry 3 --retry-delay 5 -o $Output $Url
    if ($LASTEXITCODE -ne 0) {
        throw "Download failed: $Url"
    }
}

$DaisZip = Join-Path $SourceRoot "dais_masonry.zip"
$DaisDir = Join-Path $SourceRoot "dais_masonry_archive"
Invoke-Curl `
    -Url "https://codeload.github.com/dimitrisdais/crack_detection_CNN_masonry/zip/refs/heads/main" `
    -Output $DaisZip
if (-not (Test-Path -LiteralPath $DaisDir)) {
    Expand-Archive -LiteralPath $DaisZip -DestinationPath $DaisDir
}

$DeepCrackZip = Join-Path $SourceRoot "deepcrack.zip"
$DeepCrackDir = Join-Path $SourceRoot "deepcrack_archive"
$DeepCrackDataset = Join-Path $SourceRoot "deepcrack_dataset_extracted"
Invoke-Curl `
    -Url "https://codeload.github.com/yhlleo/DeepCrack/zip/refs/heads/master" `
    -Output $DeepCrackZip
if (-not (Test-Path -LiteralPath $DeepCrackDir)) {
    Expand-Archive -LiteralPath $DeepCrackZip -DestinationPath $DeepCrackDir
}
$NestedDataset = Join-Path $DeepCrackDir "DeepCrack-master\dataset\DeepCrack.zip"
if (-not (Test-Path -LiteralPath $DeepCrackDataset)) {
    Expand-Archive -LiteralPath $NestedDataset -DestinationPath $DeepCrackDataset
}

$CrackForestDir = Join-Path $SourceRoot "crackforest"
if (-not (Test-Path -LiteralPath $CrackForestDir)) {
    & git clone --depth 1 `
        https://github.com/cuilimeng/CrackForest-dataset.git `
        $CrackForestDir
    if ($LASTEXITCODE -ne 0) {
        throw "CrackForest clone failed."
    }
}

$Crack500Zip = Join-Path $SourceRoot "crack500_mirror.zip"
$Crack500Dir = Join-Path $SourceRoot "crack500_mirror"
Invoke-Curl `
    -Url "https://codeload.github.com/Ennan010/crack-detection-unet/zip/refs/heads/master" `
    -Output $Crack500Zip
if (-not (Test-Path -LiteralPath $Crack500Dir)) {
    Expand-Archive -LiteralPath $Crack500Zip -DestinationPath $Crack500Dir
}

if ($IncludeSDNET) {
    $SDNETZip = Join-Path $SourceRoot "SDNET2018.zip"
    Invoke-Curl `
        -Url "https://digitalcommons.usu.edu/cgi/viewcontent.cgi?filename=2&article=1047&context=all_datasets&type=additional" `
        -Output $SDNETZip
    Write-Warning "SDNET2018 uses Cloudflare and may require browser download."
}

Write-Host "Public data archives are available under $SourceRoot"

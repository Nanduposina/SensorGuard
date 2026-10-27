param(
    [string]$InputPath = "data.txt/data.txt",
    [string]$Python = "python",
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$resolvedInput = Resolve-Path -LiteralPath $InputPath -ErrorAction SilentlyContinue
if (-not $resolvedInput) {
    throw "Dataset not found: $InputPath. Download the Intel Berkeley Research Lab dataset from https://db.csail.mit.edu/labdata/labdata.html and place the source file at data.txt/data.txt, or pass -InputPath. SensorGuard does not redistribute the dataset."
}

Push-Location $repoRoot
try {
    $outputDir = Join-Path $repoRoot "artifacts/phase4-demo"
    $experimentDir = Join-Path $outputDir "experiment"
    $reviewStore = Join-Path $repoRoot ".sensorguard/phase4-demo.sqlite3"
    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

    $source = Get-Item -LiteralPath $resolvedInput.Path
    Write-Host "Dataset: $($source.FullName) ($($source.Length) bytes)"
    Write-Host "Provenance: Intel Berkeley Research Lab; see https://db.csail.mit.edu/labdata/labdata.html"

    & $Python scripts/profile_dataset.py $source.FullName --output (Join-Path $outputDir "data_profile.md")
    if ($LASTEXITCODE -ne 0) { throw "Dataset profiling failed with exit code $LASTEXITCODE." }

    & $Python scripts/run_experiment.py `
        --input $source.FullName `
        --output-dir $experimentDir `
        --sensors 1,2,3,4 `
        --max-per-sensor 100 `
        --seed 42 `
        --epochs 2
    if ($LASTEXITCODE -ne 0) { throw "Bounded experiment failed with exit code $LASTEXITCODE." }

    $env:SENSORGUARD_DATA_PATH = $source.FullName
    $env:SENSORGUARD_EXPERIMENT_DIR = $experimentDir
    $env:SENSORGUARD_PROFILE_PATH = Join-Path $outputDir "data_profile.md"
    $env:SENSORGUARD_MODEL_DIR = Join-Path $experimentDir "models"
    $env:SENSORGUARD_REVIEW_STORE = $reviewStore
    Write-Host "Demo artifacts: $outputDir"
    Write-Host "Review database: $reviewStore"
    Write-Host "Open http://localhost:$Port and use the Alert review section."
    & $Python -m streamlit run app.py --server.port $Port
    if ($LASTEXITCODE -ne 0) { throw "Streamlit exited with code $LASTEXITCODE." }
}
finally {
    Pop-Location
}

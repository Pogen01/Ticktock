# Windows helper (PowerShell). Usage:
#   powershell -ExecutionPolicy Bypass -File run.ps1          # set up (first run) and start the demo
#   powershell -ExecutionPolicy Bypass -File run.ps1 train    # rebuild dataset and retrain models
#   powershell -ExecutionPolicy Bypass -File run.ps1 test     # run the tests
param([string]$Task = "serve")
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
    py -m venv .venv
    & $py -m pip install -r requirements.txt
}
switch ($Task) {
    "train" { & $py scripts\run_pipeline.py }
    "test"  { & $py -m pytest -q }
    default {
        if (-not (Test-Path reports\comparison.json)) { & $py scripts\build_comparison.py }
        Write-Host "TickTock demo: http://127.0.0.1:8000"; & $py -m uvicorn app.server:app --port 8000 }
}

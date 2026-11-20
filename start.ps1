$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$candidates = @(
    $env:POSECOACH_PYTHON,
    (Join-Path $env:LOCALAPPDATA "Programs\Python\Python312\python.exe"),
    "D:\python3.12.10\python.exe"
)

$pythonCommand = Get-Command python -ErrorAction SilentlyContinue
if ($pythonCommand) {
    $candidates += $pythonCommand.Source
}

$python = $candidates |
    Where-Object { $_ -and (Test-Path -LiteralPath $_ -PathType Leaf) } |
    Select-Object -First 1

if (-not $python) {
    Write-Host "Python 3.12 was not found." -ForegroundColor Red
    Write-Host "Set POSECOACH_PYTHON to the full path of python.exe." -ForegroundColor Yellow
    exit 1
}

try {
    $health = Invoke-WebRequest `
        -Uri "http://localhost:8501/_stcore/health" `
        -UseBasicParsing `
        -TimeoutSec 2
    if ($health.StatusCode -eq 200) {
        Write-Host "PoseCoach is already running. Opening the browser..." -ForegroundColor Green
        Start-Process "http://localhost:8501"
        Start-Sleep -Seconds 2
        exit 0
    }
}
catch {
    # No healthy server is listening on port 8501 yet.
}

$env:PYTHONUTF8 = "1"
Write-Host "Using Python: $python" -ForegroundColor DarkGray
& $python -m streamlit run app.py --server.address localhost --server.port 8501

if ($LASTEXITCODE -ne 0) {
    Write-Host "Streamlit exited with code $LASTEXITCODE." -ForegroundColor Red
    exit 1
}

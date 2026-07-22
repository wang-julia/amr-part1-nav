$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

$Python = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    if (Test-Path "C:\Python312\python.exe") {
        $Python = "C:\Python312\python.exe"
    } else {
        $Python = "python"
    }
}

& $Python server.py

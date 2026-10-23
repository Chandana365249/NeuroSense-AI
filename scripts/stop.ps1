<#
.SYNOPSIS
  Stop the NeuroSense AI backend and frontend started by scripts\start.ps1.

.DESCRIPTION
  Only stops the process IDs recorded in .run\, and only after checking that each one is still the expected
  NeuroSense process (uvicorn app.main or the Vite dev server). Other processes are never touched.
#>
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$RunDir = Join-Path $Root '.run'
$expected = @{ backend = 'uvicorn.*app\.main:create_app'; frontend = 'vite' }
$stopped = 0

foreach ($name in 'frontend', 'backend') {
    $pidFile = Join-Path $RunDir "$name.pid"
    if (-not (Test-Path $pidFile)) { Write-Host "${name}: not started by start.ps1 (no PID file)"; continue }
    $procId = [int](Get-Content $pidFile -Raw).Trim()
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$procId" -ErrorAction SilentlyContinue
    if (-not $proc) {
        Write-Host "${name}: already stopped"
    } elseif ($proc.CommandLine -notmatch $expected[$name]) {
        Write-Host "${name}: PID $procId is now a different program; leaving it alone" -ForegroundColor Yellow
    } else {
        # /T also stops child processes (the venv launcher starts a child Python interpreter).
        & taskkill.exe /PID $procId /T /F | Out-Null
        Write-Host "${name}: stopped (PID $procId)"
        $stopped++
    }
    Remove-Item $pidFile -Force
}
Write-Host "Done ($stopped stopped)."

<#
.SYNOPSIS
  Start the NeuroSense AI backend (FastAPI, 127.0.0.1:8000) and frontend (Vite, 127.0.0.1:5173) on this computer.

.DESCRIPTION
  Checks prerequisites first and reports anything missing without printing secrets. Both servers bind to
  127.0.0.1 only. Process IDs are written to .run\ so scripts\stop.ps1 stops only what this script started.
  No training or model activation is ever started.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\start.ps1
#>
[CmdletBinding()]
param(
    [switch]$BackendOnly,
    [switch]$FrontendOnly
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$RunDir = Join-Path $Root '.run'
$Python = Join-Path $Root '.venv\Scripts\python.exe'
$BackendDir = Join-Path $Root 'backend'
$FrontendDir = Join-Path $Root 'frontend'
$EnvFile = Join-Path $BackendDir '.env'
$problems = New-Object System.Collections.Generic.List[string]

function Test-PortFree([int]$Port) {
    return -not (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}

function Wait-Http([string]$Url, [int]$Seconds) {
    $deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $r = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 5
            if ($r.StatusCode -eq 200) { return $true }
        } catch { Start-Sleep -Milliseconds 700 }
    }
    return $false
}

# ---------- prerequisite checks (values are never printed) ----------
$wantBackend = -not $FrontendOnly
$wantFrontend = -not $BackendOnly
if ($wantBackend) {
    if (-not (Test-Path $Python)) { $problems.Add('Python virtual environment missing: create it with "python -m venv .venv" and "pip install -r backend\requirements.txt".') }
    if (-not (Test-Path $EnvFile)) {
        $problems.Add('backend\.env is missing: copy backend\.env.example to backend\.env and fill in the values.')
    } else {
        $envText = Get-Content $EnvFile -Raw
        foreach ($key in 'NEUROSENSE_JWT_SECRET_KEY', 'NEUROSENSE_DATABASE_URL') {
            if ($envText -notmatch "(?m)^\s*$key\s*=\s*\S+") { $problems.Add("backend\.env does not set $key.") }
        }
    }
    if (-not (Test-PortFree 8000)) { $problems.Add('Port 8000 is already in use (is the backend already running? see scripts\status.ps1).') }
}
if ($wantFrontend) {
    if (-not (Get-Command node -ErrorAction SilentlyContinue)) { $problems.Add('Node.js is not installed or not on PATH.') }
    if (-not (Test-Path (Join-Path $FrontendDir 'node_modules\vite\bin\vite.js'))) { $problems.Add('Frontend dependencies missing: run "npm ci" in the frontend folder.') }
    if (-not (Test-PortFree 5173)) { $problems.Add('Port 5173 is already in use (is the frontend already running? see scripts\status.ps1).') }
}
if ($problems.Count -gt 0) {
    Write-Host 'Cannot start NeuroSense AI:' -ForegroundColor Red
    $problems | ForEach-Object { Write-Host "  - $_" }
    exit 1
}

if ($wantBackend) {
    # Read-only connection check (SELECT 1). Prints only OK or a generic failure, never the connection string.
    Push-Location $BackendDir
    try {
        & $Python -c "import sys`nfrom sqlalchemy import create_engine, text`nfrom app.core.config import get_settings`ntry:`n    e = create_engine(get_settings().database_url)`n    c = e.connect(); c.execute(text('SELECT 1')); c.close(); print('database: reachable')`nexcept Exception:`n    print('database: NOT reachable'); sys.exit(3)"
        $dbExit = $LASTEXITCODE
    } finally { Pop-Location }
    if ($dbExit -ne 0) {
        Write-Host 'The database is not reachable. Start PostgreSQL and check NEUROSENSE_DATABASE_URL in backend\.env.' -ForegroundColor Red
        exit 1
    }
}

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

if ($wantBackend) {
    $backend = Start-Process -FilePath $Python -WorkingDirectory $BackendDir -WindowStyle Hidden -PassThru `
        -ArgumentList '-m', 'uvicorn', 'app.main:create_app', '--factory', '--host', '127.0.0.1', '--port', '8000' `
        -RedirectStandardOutput (Join-Path $RunDir 'backend.out.log') -RedirectStandardError (Join-Path $RunDir 'backend.log')
    Set-Content -Path (Join-Path $RunDir 'backend.pid') -Value $backend.Id
    Write-Host "Backend starting (PID $($backend.Id))..."
    if (Wait-Http 'http://127.0.0.1:8000/api/health' 90) {
        Write-Host '  backend healthy: http://127.0.0.1:8000/api/health' -ForegroundColor Green
    } else {
        Write-Host '  backend did not become healthy within 90 s; see .run\backend.log' -ForegroundColor Red
        exit 1
    }
}

if ($wantFrontend) {
    $node = (Get-Command node).Source
    $frontend = Start-Process -FilePath $node -WorkingDirectory $FrontendDir -WindowStyle Hidden -PassThru `
        -ArgumentList 'node_modules/vite/bin/vite.js', '--host', '127.0.0.1', '--port', '5173', '--strictPort' `
        -RedirectStandardOutput (Join-Path $RunDir 'frontend.log') -RedirectStandardError (Join-Path $RunDir 'frontend.err.log')
    Set-Content -Path (Join-Path $RunDir 'frontend.pid') -Value $frontend.Id
    Write-Host "Frontend starting (PID $($frontend.Id))..."
    if (Wait-Http 'http://127.0.0.1:5173/' 60) {
        Write-Host '  frontend ready: http://127.0.0.1:5173' -ForegroundColor Green
    } else {
        Write-Host '  frontend did not respond within 60 s; see .run\frontend.err.log' -ForegroundColor Red
        exit 1
    }
}

Write-Host ''
Write-Host 'NeuroSense AI is running on this computer only:'
if ($wantFrontend) { Write-Host '  App      http://127.0.0.1:5173' }
if ($wantBackend) {
    Write-Host '  API docs http://127.0.0.1:8000/api/docs'
    Write-Host '  Health   http://127.0.0.1:8000/api/health'
}
Write-Host 'Stop with: powershell -ExecutionPolicy Bypass -File scripts\stop.ps1'

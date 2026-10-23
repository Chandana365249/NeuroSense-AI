<#
.SYNOPSIS
  Show whether the NeuroSense AI backend and frontend are running, healthy and bound to 127.0.0.1 only.
#>
$ErrorActionPreference = 'Continue'
$ok = $true

function Show-Port([int]$Port, [string]$Name) {
    $listen = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if (-not $listen) { Write-Host ("{0,-9} port {1}: not running" -f $Name, $Port); return $false }
    $addresses = ($listen | Select-Object -ExpandProperty LocalAddress -Unique) -join ', '
    $loopback = -not ($listen | Where-Object { $_.LocalAddress -notin '127.0.0.1', '::1' })
    $note = if ($loopback) { 'loopback only' } else { 'WARNING: reachable beyond this computer' }
    Write-Host ("{0,-9} port {1}: listening on {2} (PID {3}, {4})" -f $Name, $Port, $addresses, ($listen[0].OwningProcess), $note)
    return $loopback
}

function Show-Http([string]$Url, [string]$Name) {
    try {
        $r = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec 5
        Write-Host ("{0,-9} {1} -> HTTP {2}" -f $Name, $Url, $r.StatusCode)
        return $r
    } catch {
        Write-Host ("{0,-9} {1} -> not responding" -f $Name, $Url)
        return $null
    }
}

$backendUp = Show-Port 8000 'backend'
$frontendUp = Show-Port 5173 'frontend'
$health = Show-Http 'http://127.0.0.1:8000/api/health' 'health'
if ($health) {
    $body = $health.Content | ConvertFrom-Json
    Write-Host ("          status={0}, API version {1}" -f $body.status, $body.version)
    $info = (Invoke-WebRequest -Uri 'http://127.0.0.1:8000/api/model-info' -UseBasicParsing -TimeoutSec 10).Content | ConvertFrom-Json
    if ($info.available) { Write-Host ("          active model: {0}" -f $info.version) } else { Write-Host '          no active model loaded' }
} else { $ok = $false }
if (-not (Show-Http 'http://127.0.0.1:5173/' 'frontend')) { $ok = $false }
if (-not ($backendUp -and $frontendUp)) { $ok = $false }
if ($ok) { exit 0 } else { exit 1 }

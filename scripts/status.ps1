<#
.SYNOPSIS
  Show whether the NeuroSense AI backend and frontend are running, healthy and bound to 127.0.0.1 only.
#>
$ErrorActionPreference = 'Continue'
$ok = $true

function Test-Connect([string]$Address, [int]$Port) {
    # A real connection attempt. The Windows connection table can omit live listeners, so it is not relied on alone.
    $client = New-Object System.Net.Sockets.TcpClient
    try { return ($client.ConnectAsync($Address, $Port).Wait(1500) -and $client.Connected) } catch { return $false } finally { $client.Close() }
}

function Show-Port([int]$Port, [string]$Name) {
    if (-not (Test-Connect '127.0.0.1' $Port)) { Write-Host ("{0,-9} port {1}: not running" -f $Name, $Port); return $false }
    # Loopback-only is proved by trying this computer's own network addresses, not by reading the connection table.
    $outside = @([System.Net.Dns]::GetHostAddresses($env:COMPUTERNAME) | Where-Object {
            $_.AddressFamily -eq 'InterNetwork' -and -not $_.ToString().StartsWith('127.') -and (Test-Connect $_.ToString() $Port) })
    $note = if ($outside.Count -eq 0) { 'loopback only' } else { "WARNING: reachable beyond this computer on $($outside -join ', ')" }
    Write-Host ("{0,-9} port {1}: listening on 127.0.0.1 ({2})" -f $Name, $Port, $note)
    return ($outside.Count -eq 0)
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

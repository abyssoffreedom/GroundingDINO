param(
    [string]$Configuration = "Release"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Source = Join-Path $RepoRoot "server\winsock_timestamp_udp_server.cpp"
$Output = Join-Path $RepoRoot "server\winsock_timestamp_udp_server.exe"

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    Write-Error "cl.exe was not found. Run this script from 'x64 Native Tools Command Prompt for VS' or a PowerShell where Visual Studio build tools are initialized."
}

$Flags = @("/std:c++17", "/EHsc", "/W4", "/DWIN32_LEAN_AND_MEAN")
if ($Configuration -ieq "Release") {
    $Flags += @("/O2", "/DNDEBUG")
} else {
    $Flags += @("/Od", "/Zi")
}

& cl.exe @Flags $Source "/Fe:$Output" "Ws2_32.lib"

Write-Host "Built $Output"

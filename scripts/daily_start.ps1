param([switch]$NoBrowser, [double]$Hours = 8,
      [switch]$InspectOnly, [string]$ProjectRoot = '')
$ErrorActionPreference = 'Stop'
if ([string]::IsNullOrWhiteSpace($ProjectRoot)) {
    $EntryPath = $PSCommandPath
    if (-not $EntryPath) { $EntryPath = $MyInvocation.MyCommand.Path }
    if (-not $EntryPath) { throw 'Cannot determine the script directory; pass -ProjectRoot explicitly.' }
    $ProjectRoot = Split-Path -Parent (Split-Path -Parent $EntryPath)
}
$ProjectRoot = [IO.Path]::GetFullPath($ProjectRoot)
if ($InspectOnly) {
    @{project_root=$ProjectRoot; script_path=$PSCommandPath; inspect_only=$true} | ConvertTo-Json -Compress
    exit 0
}
. (Join-Path $PSScriptRoot 'runtime_common.ps1')
$Settings = Get-StudioSettings $ProjectRoot
if ($Hours -le 0 -or $Hours -gt 168) { throw 'Hours must be above zero and no more than 168.' }
if (-not (Test-Path -LiteralPath $Settings.Python)) { throw "Missing web environment: $($Settings.Python). Run scripts/setup.ps1." }
$AlreadyRunning = Assert-StudioPort $Settings
$Watchdog = Join-Path $Settings.Root 'scripts\keep_alive.ps1'
$PowerShellExe = (Get-Process -Id $PID).Path
$Arguments = '-NoProfile -ExecutionPolicy Bypass -File "' + $Watchdog + '" -Hours ' + $Hours + ' -ProjectRoot "' + $Settings.Root + '"'
Start-Process -FilePath $PowerShellExe -ArgumentList $Arguments -WindowStyle Hidden
$Ready = $false
$Deadline = (Get-Date).AddSeconds(60)
while ((Get-Date) -lt $Deadline) {
    if (Test-StudioIdentity $Settings) { $Ready = $true; break }
    Start-Sleep -Seconds 1
}
if (-not $Ready) { throw "Web service did not become ready in 60 seconds. See $($Settings.Data)\frontend-watchdog.log and frontend.log.stderr.log." }
Save-StudioFrontendListener $Settings
if (-not $NoBrowser) { Start-Process $Settings.Url }
Write-Host "Ready: $($Settings.Url)"
Write-Host "Watchdog window requested for $Hours hours. An existing running service is not automatically reloaded."

param([switch]$InspectOnly, [string]$ProjectRoot = '')
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
function Stop-StudioOwned([string]$Filename, [string]$Role, [string]$Token) {
    $Path = Join-Path $Settings.Data $Filename
    $Receipt = Read-StudioReceipt $Path
    if (-not $Receipt) { return }
    if (Test-StudioReceipt $Receipt $Settings.Root $Role $Token) {
        Stop-Process -Id ([int]$Receipt.pid) -Force -ErrorAction Stop
        Write-Host "Stopped $Role (PID $($Receipt.pid))."
        Remove-Item -LiteralPath $Path -Force
    } else { Write-Warning "Ignored stale or foreign $Filename; no process was stopped." }
}
Stop-StudioOwned 'frontend-watchdog.json' 'watchdog' (Join-Path $Settings.Root 'scripts\keep_alive.ps1')
Stop-StudioOwned 'frontend-process.json' 'frontend' $Settings.Root
Stop-StudioOwned 'frontend-launcher.json' 'frontend-launcher' $Settings.Root
$EngineOutcome = Stop-StudioOwnedEngine $Settings
$Remaining = @(Get-StudioListeners $Settings.WebPort)
if ($Remaining.Count -gt 0) { Write-Warning "Port $($Settings.WebPort) still has a listener without a matching owned receipt. It was left running." }
if ($EngineOutcome -in @('unverified','incomplete')) { Write-Warning 'The engine was not fully stopped; check the ownership warnings above.' }
Write-Host 'Stop request finished. External engines and other installations were not stopped; user data was not changed.'

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
if (Assert-StudioPort $Settings) { Write-Host "Already running: $($Settings.Url)"; exit 0 }
$Child = Start-StudioFrontend $Settings (Join-Path $Settings.Data 'frontend.log')
Write-Host "Starting $($Settings.Url). Close with scripts/stop.ps1 or the stop BAT."
Wait-StudioReady $Settings $Child
Write-Host "Ready: $($Settings.Url)"
$Child.WaitForExit()
exit $Child.ExitCode

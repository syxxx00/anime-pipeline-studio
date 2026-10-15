param([double]$Hours = 8, [switch]$InspectOnly, [string]$ProjectRoot = '')
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
if ($Hours -le 0 -or $Hours -gt 168) { throw 'Invalid watchdog duration.' }
New-Item -ItemType Directory -Path $Settings.Data -Force | Out-Null
$StateFile = Join-Path $Settings.Data 'frontend-watchdog.json'
$PidFile = Join-Path $Settings.Data 'frontend-watchdog.pid'
$LogFile = Join-Path $Settings.Data 'frontend-watchdog.log'
$Hasher = [Security.Cryptography.SHA256]::Create()
try { $RootHash = [BitConverter]::ToString($Hasher.ComputeHash([Text.Encoding]::UTF8.GetBytes($Settings.Root.ToLowerInvariant()))).Replace('-', '') }
finally { $Hasher.Dispose() }
$MutexName = 'Local\AnimePipelineStudio-' + $RootHash
$Mutex = New-Object Threading.Mutex($false, $MutexName)
$Locked = $false
try {
    $Locked = $Mutex.WaitOne(5000)
    if (-not $Locked) { throw 'Another launcher is updating this installation. Retry shortly.' }
    $Existing = Read-StudioReceipt $StateFile
    if (Test-StudioReceipt $Existing $Settings.Root 'watchdog' $Settings.Root) {
        $Existing.minimum_until = (Get-Date).AddHours($Hours).ToString('o')
        Save-StudioReceipt $Existing $StateFile
        exit 0
    }
    $Receipt = New-StudioReceipt $PID $Settings.Root 'watchdog'
    $Receipt | Add-Member NoteProperty minimum_until (Get-Date).AddHours($Hours).ToString('o')
    $Receipt | Add-Member NoteProperty url $Settings.Url
    Save-StudioReceipt $Receipt $StateFile
    Set-Content -LiteralPath $PidFile -Value $PID -Encoding ASCII
} finally {
    if ($Locked) { $Mutex.ReleaseMutex() }
    $Mutex.Dispose()
}
Add-Content -LiteralPath $LogFile -Value "$(Get-Date -Format o) watchdog started: $($Settings.Root)"
try {
    while ($true) {
        $Current = Read-StudioReceipt $StateFile
        if (-not $Current -or $Current.pid -ne $PID -or [datetimeoffset]::Now -ge [datetimeoffset]::Parse($Current.minimum_until)) { break }
        try {
            if (Assert-StudioPort $Settings) { Save-StudioFrontendListener $Settings; Start-Sleep -Seconds 10; continue }
            $Child = Start-StudioFrontend $Settings (Join-Path $Settings.Data 'frontend.log')
            Add-Content -LiteralPath $LogFile -Value "$(Get-Date -Format o) frontend started: PID $($Child.Id)"
            Wait-StudioReady $Settings $Child
        } catch {
            Add-Content -LiteralPath $LogFile -Value "$(Get-Date -Format o) $($_.Exception.Message)"
            if (@(Get-StudioListeners $Settings.WebPort).Count -gt 0 -and -not (Test-StudioIdentity $Settings)) { break }
            Start-Sleep -Seconds 5
        }
    }
} finally {
    $Current = Read-StudioReceipt $StateFile
    if ($Current -and $Current.pid -eq $PID) {
        Remove-Item -LiteralPath $PidFile -Force -ErrorAction SilentlyContinue
        Remove-Item -LiteralPath $StateFile -Force -ErrorAction SilentlyContinue
    }
    Add-Content -LiteralPath $LogFile -Value "$(Get-Date -Format o) watchdog ended; a healthy frontend is not automatically stopped"
}

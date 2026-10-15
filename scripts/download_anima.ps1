param([switch]$Download, [string]$Revision = 'main',
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
$Hf = Join-Path $Settings.Root '.venv\Scripts\hf.exe'
if (-not (Test-Path -LiteralPath $Hf)) { throw 'Missing hf CLI. Prepare the web environment with scripts/setup.ps1 first.' }
$ModelRoot = Resolve-StudioPath $Settings.Root ([string]$Settings.Raw.model_root)
if (-not $ModelRoot) { throw 'model_root must be configured.' }
$Diffusion = 'anima-turbo-v1.1.safetensors'
$Clip = 'qwen_3_06b_base.safetensors'
$Vae = 'qwen_image_vae.safetensors'
if ($Settings.Raw.anima_diffusion_model) { $Diffusion = [string]$Settings.Raw.anima_diffusion_model }
if ($Settings.Raw.anima_clip) { $Clip = [string]$Settings.Raw.anima_clip }
if ($Settings.Raw.anima_vae) { $Vae = [string]$Settings.Raw.anima_vae }
$Files = @(
    @{Folder='diffusion_models'; Name=$Diffusion},
    @{Folder='text_encoders'; Name=$Clip},
    @{Folder='vae'; Name=$Vae}
)
$Pending = @()
foreach ($File in $Files) {
    if ($File.Name -match '[/\\]' -or $File.Name -notmatch '\.safetensors$') { throw 'Configured model filenames must be simple safetensors basenames.' }
    $Destination = Join-Path (Join-Path $ModelRoot $File.Folder) $File.Name
    if (Test-Path -LiteralPath $Destination) {
        if ((Get-Item -LiteralPath $Destination).Length -le 100MB) { throw "Existing file is too small; it was not overwritten: $Destination" }
        Write-Host "Keeping existing model: $Destination"
    } else { $Pending += $File }
}
if ($Pending.Count -eq 0) { Write-Host 'All configured Anima files already exist; nothing downloaded.'; exit 0 }
$DownloadRoot = Join-Path $Settings.Root 'runtime\downloads\Anima'
Write-Host "Source: https://huggingface.co/circlestone-labs/Anima (revision $Revision)"
Write-Host "Download cache: $DownloadRoot"
Write-Host "Destination: $ModelRoot"
Write-Host 'Several GiB may be downloaded, plus a second copy in model_root. No environment is created. Review the source license first.'
$Pending | ForEach-Object { Write-Host ("  split_files/" + $_.Folder + "/" + $_.Name) }
if (-not $Download) {
    $Confirmed = Read-Host 'Download these model files now? Type YES (anything else cancels)'
    if ($Confirmed -cne 'YES') { Write-Host 'Cancelled; no download was performed.'; exit 0 }
}
$env:HF_HOME = Join-Path $Settings.Root 'runtime\cache\huggingface'
$env:HF_HUB_CACHE = Join-Path $env:HF_HOME 'hub'
$env:HF_XET_CACHE = Join-Path $env:HF_HOME 'xet'
$env:TEMP = Join-Path $Settings.Root 'runtime\cache\temp'; $env:TMP = $env:TEMP
New-Item -ItemType Directory -Path $DownloadRoot,$env:TEMP -Force | Out-Null
$Arguments = @('download','circlestone-labs/Anima','--revision',$Revision,'--local-dir',$DownloadRoot,'--max-workers','3')
foreach ($File in $Pending) { $Arguments += ('split_files/' + $File.Folder + '/' + $File.Name) }
& $Hf @Arguments
if ($LASTEXITCODE -ne 0) { throw "Anima download failed ($LASTEXITCODE); nothing was installed into model_root." }
foreach ($File in $Pending) {
    $Source = Join-Path (Join-Path (Join-Path $DownloadRoot 'split_files') $File.Folder) $File.Name
    if (-not (Test-Path -LiteralPath $Source) -or (Get-Item -LiteralPath $Source).Length -le 100MB) { throw "Downloaded file is missing or too small: $Source" }
}
foreach ($File in $Pending) {
    $Folder = Join-Path $ModelRoot $File.Folder
    New-Item -ItemType Directory -Path $Folder -Force | Out-Null
    # Never overwrite a user's existing weight, including one added during download.
    [IO.File]::Copy((Join-Path (Join-Path (Join-Path $DownloadRoot 'split_files') $File.Folder) $File.Name), (Join-Path $Folder $File.Name), $false)
}
Write-Host 'Configured Anima files installed. Restart your own idle engine when convenient; no service was restarted.'

param(
    [string]$Python,
    [switch]$InstallDependencies,
    [switch]$WebOnly,
    [switch]$PrepareEngine,
    [switch]$UpdateEngineDependencies,
    [string]$ComfyRevision = 'v0.19.5',
    [string]$NewBieNodesRevision = '4b420741946d5243b2fed9561ccc9fa699675669',
    [string]$TorchIndexUrl = 'https://download.pytorch.org/whl/cu128',
    [switch]$InspectOnly, [string]$ProjectRoot = ''
)
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
$ProjectRoot = [IO.Path]::GetFullPath($ProjectRoot)
if ($WebOnly -and $PrepareEngine) { throw 'Choose either -WebOnly or -PrepareEngine.' }
if ($UpdateEngineDependencies -and -not $PrepareEngine) { throw '-UpdateEngineDependencies requires -PrepareEngine.' }

function Invoke-Checked([string]$Executable, [string[]]$Arguments) {
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $Executable $($Arguments -join ' ')" }
}
function Invoke-PythonCode([string]$Executable, [string]$Code) {
    # Windows PowerShell 5.1 strips embedded double quotes in native arguments.
    # Carry the source as ASCII base64, preserving quotes, Unicode and newlines.
    $Encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($Code))
    $Bootstrap = "import base64; exec(compile(base64.b64decode('$Encoded'), '<studio-setup>', 'exec'))"
    Invoke-Checked $Executable @('-c', $Bootstrap)
}
function Find-SetupPython {
    if ($Python) {
        $Candidate = Get-Command $Python -CommandType Application -ErrorAction Stop | Select-Object -First 1
        return $Candidate.Source
    }
    $PyLauncher = Get-Command py -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($PyLauncher) {
        foreach ($VersionFlag in @('-3.12', '-3.13')) {
            $Found = & $PyLauncher.Source $VersionFlag -c 'import sys; print(sys.executable)' 2>$null
            if ($LASTEXITCODE -eq 0 -and $Found -and (Test-Path -LiteralPath ([string]$Found))) { return [string]$Found }
        }
    }
    foreach ($Name in @('python3.12', 'python')) {
        $Candidate = Get-Command $Name -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($Candidate) {
            & $Candidate.Source -c 'import sys; raise SystemExit(0 if (3,12) <= sys.version_info[:2] < (3,14) else 1)' 2>$null
            if ($LASTEXITCODE -eq 0) { return $Candidate.Source }
        }
    }
    throw 'Python 3.12/3.13 was not found. Install Python yourself, or pass -Python C:\path\python.exe.'
}
$PythonBase = Find-SetupPython
Invoke-PythonCode $PythonBase 'import sys; assert (3,12) <= sys.version_info[:2] < (3,14), "Python 3.12/3.13 required"; print(sys.executable)'
$VenvPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if ($PrepareEngine) {
    # Read the current or first-run configuration before asking for downloads.
    # Resolve once so the displayed targets are the ones this run will use.
    $PreviewPath = Join-Path $ProjectRoot 'config\settings.json'
    if (-not (Test-Path -LiteralPath $PreviewPath)) { $PreviewPath = Join-Path $ProjectRoot 'config\settings.example.json' }
    $Preview = Get-Content -LiteralPath $PreviewPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $Core = Resolve-StudioPath $ProjectRoot ([string]$Preview.comfy_root)
    $EnginePython = Resolve-StudioPath $ProjectRoot ([string]$Preview.comfy_python)
    $EngineBase = Resolve-StudioPath $ProjectRoot ([string]$Preview.engine_base_dir)
    if (-not $Core -or -not $EnginePython -or -not $EngineBase) { throw 'Configure comfy_root, comfy_python and engine_base_dir before preparing the engine.' }
    $ExistingEngineEnvironment = Test-Path -LiteralPath $EnginePython
    $WriteEngineEnvironment = -not $ExistingEngineEnvironment -or $UpdateEngineDependencies
    if ($ExistingEngineEnvironment -and $EnginePython -ieq $VenvPython -and -not $UpdateEngineDependencies) {
        throw 'The web and existing inference environments use the same Python. Choose separate environments, or explicitly pass -PrepareEngine -UpdateEngineDependencies to allow package changes.'
    }
}
Write-Host "Project: $ProjectRoot"
Write-Host 'This installs the local web environment under .venv; models are NEVER downloaded by setup.'
Write-Host "Web environment to create/update: $VenvPython"
if ($PrepareEngine) {
    Write-Host "Core checkout target: $Core (existing source checkout is not updated)."
    Write-Host "NewBie node target: $(Join-Path $EngineBase 'custom_nodes\ComfyUI-Newbie-Nodes') (existing checkout is not updated)."
    if ($WriteEngineEnvironment) {
        Write-Host "Inference environment to create/update: $EnginePython"
        Write-Host 'CUDA/Python dependencies may download several GiB. Existing packages in this selected environment may change.'
    } else {
        Write-Host "Inference environment reused without pip changes: $EnginePython"
        Write-Host 'Missing dependencies must be repaired manually or with the additional -UpdateEngineDependencies option.'
    }
    Write-Host "Missing Core/node checkouts are fetched at $ComfyRevision / $NewBieNodesRevision. Weight files remain a separate manual step."
} else {
    Write-Host 'Web dependencies are typically tens of MiB. ComfyUI, CUDA runtime, llama-server and weights must be prepared separately or reused.'
}
if (-not $InstallDependencies) {
    $Confirmed = Read-Host 'Allow these dependency downloads/installations now? Type YES (anything else cancels)'
    if ($Confirmed -cne 'YES') { Write-Host 'Cancelled; no installation or download was performed.'; exit 0 }
}
$CacheRoot = Join-Path $ProjectRoot 'runtime\cache'
$InstallTemp = Join-Path $CacheRoot 'temp'
$PipCache = Join-Path $CacheRoot 'pip'
New-Item -ItemType Directory -Path $InstallTemp,$PipCache -Force | Out-Null
$env:TEMP=$InstallTemp; $env:TMP=$InstallTemp; $env:PIP_CACHE_DIR=$PipCache
if (-not (Test-Path -LiteralPath $VenvPython)) { Invoke-Checked $PythonBase @('-m','venv',(Join-Path $ProjectRoot '.venv')) }
Invoke-Checked $VenvPython @('-m','pip','install','--upgrade','pip')
Invoke-Checked $VenvPython @('-m','pip','install','-e',$ProjectRoot)
$ConfigPath = Join-Path $ProjectRoot 'config\settings.json'
if (-not (Test-Path -LiteralPath $ConfigPath)) { Copy-Item -LiteralPath (Join-Path $ProjectRoot 'config\settings.example.json') -Destination $ConfigPath }
$Settings = Get-StudioSettings $ProjectRoot

if ($PrepareEngine) {
    $Git = (Get-Command git -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
    if (Test-Path -LiteralPath $Core) {
        if (-not (Test-Path -LiteralPath (Join-Path $Core 'main.py'))) { throw "Engine directory exists but is not a ComfyUI checkout: $Core. It was not overwritten." }
        Write-Host "Reusing existing Core without updating it: $Core"
    } else {
        Invoke-Checked $Git @('clone','--branch',$ComfyRevision,'--depth','1','https://github.com/Comfy-Org/ComfyUI.git',$Core)
    }
    if (-not $ExistingEngineEnvironment) {
        $EngineVenv = Split-Path -Parent (Split-Path -Parent $EnginePython)
        if ((Split-Path -Leaf $EnginePython) -ine 'python.exe' -or (Split-Path -Leaf (Split-Path -Parent $EnginePython)) -ine 'Scripts' -or (Split-Path -Leaf $EngineVenv) -ine '.venv') {
            throw "For automatic environment creation comfy_python must end with .venv\Scripts\python.exe: $EnginePython"
        }
        Invoke-Checked $PythonBase @('-m','venv',$EngineVenv)
    }
    if ($WriteEngineEnvironment) {
        Invoke-Checked $EnginePython @('-m','pip','install','--upgrade','pip')
        Invoke-Checked $EnginePython @('-m','pip','install','torch','torchvision','torchaudio','--index-url',$TorchIndexUrl)
        Invoke-Checked $EnginePython @('-m','pip','install','-r',(Join-Path $Core 'requirements.txt'))
    }
    $Nodes = Join-Path $EngineBase 'custom_nodes\ComfyUI-Newbie-Nodes'
    if (-not (Test-Path -LiteralPath $Nodes)) {
        New-Item -ItemType Directory -Path (Split-Path -Parent $Nodes) -Force | Out-Null
        Invoke-Checked $Git @('clone','--depth','1','https://github.com/NewBieAI-Lab/ComfyUI-Newbie-Nodes.git',$Nodes)
        Invoke-Checked $Git @('-C',$Nodes,'fetch','--depth','1','origin',$NewBieNodesRevision)
        Invoke-Checked $Git @('-C',$Nodes,'checkout','--detach',$NewBieNodesRevision)
    } else { Write-Host "Reusing existing nodes without updating them: $Nodes" }
    try {
        Invoke-PythonCode $EnginePython 'import torch; print("torch:",torch.__version__); assert torch.cuda.is_available(), "CUDA is unavailable: check NVIDIA driver / TorchIndexUrl"'
    } catch {
        if ($ExistingEngineEnvironment -and -not $WriteEngineEnvironment) {
            throw "Existing inference environment was left unchanged. Check its dependencies and NVIDIA driver; repair manually or rerun with -PrepareEngine -UpdateEngineDependencies. Original check: $($_.Exception.Message)"
        }
        throw
    }
    Write-Host 'Core and selected inference environment checked. This does not install model weights or llama-server.'
}
Write-Host "Web environment ready. Review $ConfigPath before starting."
Write-Host 'Use the start BAT (or scripts/daily_start.ps1). No service was started by setup.'

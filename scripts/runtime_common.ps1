# Offline shared launcher helpers. Dot-sourcing never starts services.
$ErrorActionPreference = 'Stop'

function Get-StudioSettings([string]$Root) {
    $ResolvedRoot = [IO.Path]::GetFullPath($Root).TrimEnd('\', '/')
    $SettingsPath = Join-Path $ResolvedRoot 'config\settings.json'
    if (-not (Test-Path -LiteralPath $SettingsPath)) { throw "Missing config/settings.json. Run scripts/setup.ps1 first: $ResolvedRoot" }
    $Raw = Get-Content -LiteralPath $SettingsPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $WebPort = 8090
    if ($null -ne $Raw.web_port) { $WebPort = [int]$Raw.web_port }
    $EnginePort = 8190
    if ($null -ne $Raw.engine_port) { $EnginePort = [int]$Raw.engine_port }
    foreach ($Port in @($WebPort, $EnginePort)) { if ($Port -lt 1 -or $Port -gt 65535) { throw "Invalid configured port: $Port" } }
    $ComfyUrl = "http://127.0.0.1:$EnginePort"
    if ($Raw.comfy_url) { $ComfyUrl = [string]$Raw.comfy_url }
    return [pscustomobject]@{Root=$ResolvedRoot; Raw=$Raw; WebPort=$WebPort; EnginePort=$EnginePort
        Url="http://127.0.0.1:$WebPort"; ComfyUrl=$ComfyUrl
        Python=(Join-Path $ResolvedRoot '.venv\Scripts\python.exe'); Data=(Join-Path $ResolvedRoot 'data')}
}

function Resolve-StudioPath([string]$Root, [string]$Value) {
    if ([string]::IsNullOrWhiteSpace($Value)) { return $null }
    if ([IO.Path]::IsPathRooted($Value)) { return [IO.Path]::GetFullPath($Value) }
    return [IO.Path]::GetFullPath((Join-Path $Root $Value))
}

function Get-StudioEnvironmentValue([string]$Name, [EnvironmentVariableTarget]$Target) {
    return [Environment]::GetEnvironmentVariable($Name, $Target)
}

function Initialize-StudioAuthoringEnvironment($Settings) {
    # Read only the operator-named key. Never persist it, enumerate other user
    # variables, print its value, or replace an explicit process-level value.
    if (-not $Settings.Raw -or $Settings.Raw.authoring_backend -ne 'openai_compatible') { return }
    $Name = 'AI_COMIC_AUTHORING_API_KEY'
    if ($Settings.Raw.authoring_api_key_env) { $Name = [string]$Settings.Raw.authoring_api_key_env }
    if ($Name -notmatch '^[A-Za-z_][A-Za-z0-9_]*$') {
        throw 'authoring_api_key_env must be an environment variable name, not a key value.'
    }
    $ProcessValue = Get-StudioEnvironmentValue $Name ([EnvironmentVariableTarget]::Process)
    if (-not [string]::IsNullOrEmpty($ProcessValue)) { return }
    $UserValue = Get-StudioEnvironmentValue $Name ([EnvironmentVariableTarget]::User)
    if (-not [string]::IsNullOrEmpty($UserValue)) {
        [Environment]::SetEnvironmentVariable($Name, $UserValue, [EnvironmentVariableTarget]::Process)
    }
}

function Get-StudioListeners([int]$Port) {
    return @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique)
}

function Test-StudioIdentity($Settings) {
    try {
        $Identity = Invoke-RestMethod -Uri ($Settings.Url + '/api/app-identity') -TimeoutSec 2
        return ($Identity.application -eq 'anime-pipeline-studio' -and
            [IO.Path]::GetFullPath([string]$Identity.project_root).TrimEnd('\', '/') -ieq $Settings.Root)
    } catch { return $false }
}

function Assert-StudioPort($Settings) {
    $Listeners = @(Get-StudioListeners $Settings.WebPort)
    if ($Listeners.Count -gt 0 -and -not (Test-StudioIdentity $Settings)) {
        throw "Port $($Settings.WebPort) is occupied by another application or another installation. Nothing was stopped."
    }
    return ($Listeners.Count -gt 0)
}

function Get-StudioProcess([int]$ProcessId) {
    return Get-CimInstance Win32_Process -Filter "ProcessId=$ProcessId" -ErrorAction SilentlyContinue
}

function New-StudioReceipt([int]$ProcessId, [string]$Root, [string]$Role) {
    $ProcessInfo = Get-StudioProcess $ProcessId
    if (-not $ProcessInfo) { throw "Cannot record process identity: PID $ProcessId" }
    return [pscustomobject]@{pid=$ProcessId; project_root=$Root; role=$Role
        started_at=([datetimeoffset]$ProcessInfo.CreationDate).ToUniversalTime().ToString('o')}
}

function Test-StudioReceipt($Receipt, [string]$Root, [string]$Role, [string]$CommandToken) {
    if (-not $Receipt -or $Receipt.project_root -ine $Root -or $Receipt.role -ne $Role) { return $false }
    $ReceiptPid = 0
    if (-not [int]::TryParse([string]$Receipt.pid, [ref]$ReceiptPid) -or $ReceiptPid -le 0 -or -not $CommandToken) { return $false }
    $ProcessInfo = Get-StudioProcess $ReceiptPid
    if (-not $ProcessInfo -or -not $ProcessInfo.CommandLine -or -not $Receipt.started_at) { return $false }
    try {
        $Actual = ([datetimeoffset]$ProcessInfo.CreationDate).ToUniversalTime()
        $Expected = [datetimeoffset]::Parse([string]$Receipt.started_at).ToUniversalTime()
        if ([Math]::Abs(($Actual - $Expected).TotalSeconds) -gt 0.01) { return $false }
    } catch { return $false }
    $NormalizedCommand = $ProcessInfo.CommandLine.Replace('/', '\')
    $NormalizedToken = $CommandToken.Replace('/', '\')
    $TokenPattern = '(?i)(?:^|[\s"''])' + [regex]::Escape($NormalizedToken) + '(?=$|[\s"''])'
    return $NormalizedCommand -match $TokenPattern
}

function Read-StudioReceipt([string]$Path) {
    try { return Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json } catch { return $null }
}

function Save-StudioReceipt($Receipt, [string]$Path) {
    $Temporary = $Path + '.' + [guid]::NewGuid().ToString('N') + '.tmp'
    $Receipt | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $Temporary -Encoding UTF8
    Move-Item -LiteralPath $Temporary -Destination $Path -Force
}

function Start-StudioFrontend($Settings, [string]$LogPath) {
    if (-not (Test-Path -LiteralPath $Settings.Python)) { throw "Missing web environment: $($Settings.Python). Run scripts/setup.ps1." }
    if (Assert-StudioPort $Settings) { return $null }
    New-Item -ItemType Directory -Path $Settings.Data -Force | Out-Null
    $Arguments = '-m uvicorn app.main:app --host 127.0.0.1 --port ' + $Settings.WebPort + ' --app-dir "' + $Settings.Root + '"'
    $Options = @{FilePath=$Settings.Python; ArgumentList=$Arguments; WorkingDirectory=$Settings.Root; PassThru=$true; WindowStyle='Hidden'}
    if ($LogPath) { $Options.RedirectStandardOutput=$LogPath; $Options.RedirectStandardError=$LogPath + '.stderr.log' }
    Initialize-StudioAuthoringEnvironment $Settings
    $Child = Start-Process @Options
    Save-StudioReceipt (New-StudioReceipt $Child.Id $Settings.Root 'frontend-launcher') (Join-Path $Settings.Data 'frontend-launcher.json')
    return $Child
}

function Save-StudioFrontendListener($Settings) {
    if (-not (Test-StudioIdentity $Settings)) { return }
    foreach ($ListenerId in @(Get-StudioListeners $Settings.WebPort)) {
        $ProcessInfo = Get-StudioProcess ([int]$ListenerId)
        if ($ProcessInfo -and $ProcessInfo.CommandLine -match 'uvicorn\s+app\.main:app' -and
            $ProcessInfo.CommandLine.IndexOf($Settings.Root, [StringComparison]::OrdinalIgnoreCase) -ge 0) {
            Save-StudioReceipt (New-StudioReceipt $ListenerId $Settings.Root 'frontend') (Join-Path $Settings.Data 'frontend-process.json')
        }
    }
}

function Wait-StudioReady($Settings, $Child, [int]$Seconds = 60) {
    $Deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $Deadline) {
        if (Test-StudioIdentity $Settings) { Save-StudioFrontendListener $Settings; return }
        if ($Child -and $Child.HasExited) { throw "Web process exited with code $($Child.ExitCode). See data/frontend.log.stderr.log." }
        Start-Sleep -Seconds 1
    }
    throw 'Web process did not return this installation identity. See data/frontend.log and frontend.log.stderr.log.'
}

function Test-StudioEngineCommand($ProcessInfo, [string]$MainPath, [string]$BasePath) {
    if (-not $ProcessInfo -or $ProcessInfo.Name -notmatch '^python(?:w|\d+(?:\.\d+)*)?\.exe$' -or -not $ProcessInfo.CommandLine) { return $false }
    $Command = $ProcessInfo.CommandLine.Replace('/', '\')
    $Main = [regex]::Escape($MainPath.Replace('/', '\'))
    $Base = [regex]::Escape($BasePath.Replace('/', '\'))
    $MainArgument = '"' + $Main + '"'
    if ($MainPath -notmatch '\s') { $MainArgument = '(?:' + $MainArgument + '|' + $Main + ')' }
    $BaseArgument = '"' + $Base + '"'
    if ($BasePath -notmatch '\s') { $BaseArgument = '(?:' + $BaseArgument + '|' + $Base + ')' }
    return ($Command -match ('(?i)(?:^|\s)' + $MainArgument + '(?=\s|$)') -and
        $Command -match ('(?i)(?:^|\s)--base-directory\s+' + $BaseArgument + '(?=\s|$)'))
}

function Get-StudioProcessSnapshot {
    return @(Get-CimInstance Win32_Process -ErrorAction Stop)
}

function Stop-StudioOwnedEngine($Settings) {
    $ReceiptPath = Join-Path $Settings.Data 'engine-process.json'
    $Receipt = Read-StudioReceipt $ReceiptPath
    if (-not $Receipt) { return 'not_recorded' }
    $Core = Resolve-StudioPath $Settings.Root ([string]$Receipt.comfy_root)
    $Base = Resolve-StudioPath $Settings.Root ([string]$Receipt.engine_base_dir)
    if (-not $Core -or -not $Base) { Write-Warning 'Engine receipt has no complete path binding; engine was not stopped.'; return 'unverified' }
    $Main = Join-Path $Core 'main.py'
    $Arguments = @($Receipt.command_line)
    $BaseIndexes = @(for ($Index=0; $Index -lt $Arguments.Count; $Index++) { if ($Arguments[$Index] -eq '--base-directory') { $Index } })
    if ($Arguments.Count -lt 4 -or $BaseIndexes.Count -ne 1 -or $BaseIndexes[0] + 1 -ge $Arguments.Count -or
        (Resolve-StudioPath $Settings.Root ([string]$Arguments[1])) -ine $Main -or
        (Resolve-StudioPath $Settings.Root ([string]$Arguments[$BaseIndexes[0] + 1])) -ine $Base -or
        -not (Test-StudioReceipt $Receipt $Settings.Root 'engine' $Base)) {
        Write-Warning 'Engine parent ownership could not be verified; no descendant or port-based shutdown was attempted.'
        return 'unverified'
    }
    $Parent = Get-StudioProcess ([int]$Receipt.pid)
    if (-not (Test-StudioEngineCommand $Parent $Main $Base)) { Write-Warning 'Engine parent command does not match its receipt; nothing was stopped.'; return 'unverified' }
    $ParentTime = ([datetimeoffset]$Parent.CreationDate).ToUniversalTime()
    if ([Math]::Abs(($ParentTime - [datetimeoffset]::Parse([string]$Receipt.started_at).ToUniversalTime()).TotalSeconds) -gt 0.01) {
        Write-Warning 'Engine parent identity changed before descendant inspection; nothing was stopped.'
        return 'unverified'
    }
    $Family = @([pscustomobject]@{Process=$Parent;Depth=0;Created=$ParentTime})
    $Seen = @{([int]$Parent.ProcessId)=$true}
    $Snapshot = @(Get-StudioProcessSnapshot)
    # Traverse only matching Python-to-Python edges. A parent PID alone is not
    # ownership: each child must also be newer and carry these exact engine paths.
    for ($Index=0; $Index -lt $Family.Count; $Index++) {
        $Owner = $Family[$Index]
        foreach ($ProcessInfo in $Snapshot) {
            if ($ProcessInfo.ParentProcessId -ne $Owner.Process.ProcessId -or $Seen.ContainsKey([int]$ProcessInfo.ProcessId)) { continue }
            if (-not (Test-StudioEngineCommand $ProcessInfo $Main $Base)) { continue }
            try { $Created = ([datetimeoffset]$ProcessInfo.CreationDate).ToUniversalTime() } catch { continue }
            if ($Created -lt $Owner.Created) { continue }
            $Seen[[int]$ProcessInfo.ProcessId]=$true
            $Family += [pscustomobject]@{Process=$ProcessInfo;Depth=($Owner.Depth+1);Created=$Created}
        }
    }
    $Incomplete = $false
    foreach ($Member in @($Family | Sort-Object Depth -Descending)) {
        $ProcessInfo = Get-StudioProcess ([int]$Member.Process.ProcessId)
        if (-not $ProcessInfo) { continue }
        $StillSame = ([datetimeoffset]$ProcessInfo.CreationDate).ToUniversalTime() -eq $Member.Created
        if (-not $StillSame -or -not (Test-StudioEngineCommand $ProcessInfo $Main $Base)) { $Incomplete=$true; continue }
        Stop-Process -Id ([int]$ProcessInfo.ProcessId) -Force -ErrorAction Stop
        Write-Host "Stopped owned engine process (PID $($ProcessInfo.ProcessId))."
    }
    if ($Incomplete) { Write-Warning 'An engine process changed identity during shutdown and was left running.'; return 'incomplete' }
    Remove-Item -LiteralPath $ReceiptPath -Force
    return 'stopped'
}

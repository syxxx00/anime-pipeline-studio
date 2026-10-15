<#
.SYNOPSIS
Build an allowlisted public source directory from the CURRENT working tree.
.DESCRIPTION
OutputDirectory is mandatory and must be new or empty. No overwrite, deletion,
git export, dependency installation, application import or service startup occurs.
Documentation is mapped from docs/release, not the private local-use README.
.EXAMPLE
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\package_public.ps1 -OutputDirectory F:\staging\anime-pipeline-studio
#>
param(
    [Parameter(Mandatory=$true)][ValidateNotNullOrEmpty()][string]$OutputDirectory,
    [string]$ProjectRoot,
    [switch]$IncludeTests
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
# Windows PowerShell 5.1 can evaluate parameter defaults before PSScriptRoot
# is available. Resolve only an omitted value here, never replace caller input.
if (-not $ProjectRoot) {
    $ScriptDirectory = $PSScriptRoot
    if (-not $ScriptDirectory -and $PSCommandPath) { $ScriptDirectory = Split-Path -Parent $PSCommandPath }
    if (-not $ScriptDirectory) { throw 'Cannot determine the script directory; pass ProjectRoot explicitly.' }
    $ProjectRoot = Split-Path -Parent $ScriptDirectory
}
$Root = [IO.Path]::GetFullPath($ProjectRoot).TrimEnd([char[]]@('\', '/'))
$RootPrefix = $Root + [IO.Path]::DirectorySeparatorChar
$Output = [IO.Path]::GetFullPath($OutputDirectory).TrimEnd([char[]]@('\', '/'))
$Utf8 = [Text.UTF8Encoding]::new($false)

function Assert-NoLink([string]$Path) {
    $Current = [IO.Path]::GetFullPath($Path)
    while ($Current) {
        if (Test-Path -LiteralPath $Current) {
            $Item = Get-Item -LiteralPath $Current -Force
            if (($Item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                throw "Reparse points are refused: $Current"
            }
        }
        $Parent = [IO.Directory]::GetParent($Current)
        if ($null -eq $Parent) { break }
        $Current = $Parent.FullName
    }
}

function Assert-Relative([string]$Path) {
    if (-not $Path -or $Path.Contains('\') -or $Path.Contains(':') -or $Path.StartsWith('/')) {
        throw "Unsafe relative destination: $Path"
    }
    foreach ($Part in $Path.Split('/')) {
        if (-not $Part -or $Part -in @('.', '..') -or $Part.EndsWith('.') -or $Part.EndsWith(' ') -or
                $Part -match '^(?i:con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?$') {
            throw "Unsafe path component: $Path"
        }
    }
}

function Test-Excluded([string]$Relative) {
    foreach ($Part in $Relative.Split('/')) {
        if ($Part -match '^(?i:\.git|\.venv|venv|runtime|work|dist|__pycache__|node_modules|\.pytest_cache|\.ruff_cache|\.mypy_cache|logs?)$' -or
                $Part -match '^(?i:\.env)(?:\..*)?$' -or $Part -ieq 'settings.json' -or $Part -ieq 'AGENTS.md' -or
                $Part -match '(?i:credentials?|(?:^|[._-])(?:secrets?|api[._-]?keys?|private[._-]?keys?)(?:[._-]|$))') {
            return $true
        }
    }
    return [IO.Path]::GetExtension($Relative) -match '^(?i:\.pyc|\.pyo|\.gguf|\.safetensors|\.ckpt|\.pt|\.pth|\.onnx|\.bin|\.exe|\.dll|\.pdb|\.whl|\.zip|\.7z|\.tar|\.gz|\.log|\.jsonl|\.sqlite|\.sqlite3|\.db|\.key|\.pem|\.pfx|\.p12|\.kdbx)$'
}

function Get-Digest([string]$Path) {
    $Hasher = [Security.Cryptography.SHA256]::Create()
    $Stream = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::Read)
    try { return [BitConverter]::ToString($Hasher.ComputeHash($Stream)).Replace('-', '').ToLowerInvariant() }
    finally { $Stream.Dispose(); $Hasher.Dispose() }
}

function Add-Entry([string]$Source, [string]$Destination, $Records) {
    Assert-NoLink $Source
    Assert-Relative $Destination
    if (Test-Excluded $Destination) { return }
    $Item = Get-Item -LiteralPath $Source -Force
    if ($Item.PSIsContainer) { throw "Expected source file: $Source" }
    [void]$Records.Add([pscustomobject]@{source=$Item.FullName; path=$Destination;
        bytes=$Item.Length; sha256=(Get-Digest $Item.FullName)})
}

function Add-Tree([string]$Source, [string]$Destination, $Records) {
    Assert-NoLink $Source
    if (-not (Test-Path -LiteralPath $Source -PathType Container)) { throw "Required source directory is missing: $Source" }
    foreach ($Item in Get-ChildItem -LiteralPath $Source -Force) {
        if (($Item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw "Source link refused: $($Item.FullName)" }
        $Relative = $Destination + '/' + $Item.Name
        Assert-Relative $Relative
        if (Test-Excluded $Relative) { continue }
        if ($Item.PSIsContainer) { Add-Tree $Item.FullName $Relative $Records }
        else { Add-Entry $Item.FullName $Relative $Records }
    }
}

function From-CodePoints([int[]]$Points) { return -join ($Points | ForEach-Object { [char]$_ }) }

function Get-Inventory {
    $Records = [Collections.Generic.List[object]]::new()
    foreach ($Directory in @('app', 'static', 'engine_extensions', 'resources')) {
        Add-Tree (Join-Path $Root $Directory) $Directory $Records
    }
    $Files = @(
        'MODELS.md', 'LICENSE', 'THIRD_PARTY_NOTICES.md', '.gitignore', '.gitattributes', 'pyproject.toml',
        'config/settings.example.json', 'config/extra_model_paths.yaml', 'config/style_presets.json', 'config/controlled_studies.json',
        'config/overnight_matrix_v1.json', 'config/pipeline_v2_audit_expectations.json',
        'data/check_style_rules.json', 'data/knowledge/arknights.json',
        'docs/EXISTING_EVIDENCE_2026-09-05.md',
        'scripts/setup.ps1', 'scripts/start.ps1', 'scripts/stop.ps1', 'scripts/daily_start.ps1',
        'scripts/keep_alive.ps1', 'scripts/runtime_common.ps1', 'scripts/package_public.ps1',
        'scripts/check_image.py', 'scripts/intent_infer.py', 'scripts/semantic_infer.py', 'scripts/text_detect.py',
        'scripts/download_anima.ps1',
        'scripts/build_creative_picture_decision_grammar.py', 'scripts/build_creative_research_decision_grammar.py',
        'scripts/build_creative_review_grammar.py', 'scripts/build_region_planner_grammar.py'
    )
    $Files += (From-CodePoints @(0x542f,0x52a8)) + 'NewBie Studio.bat'
    $Files += (From-CodePoints @(0x505c,0x6b62)) + 'NewBie Studio.bat'
    $Files += 'docs/research/reports/2026-09-20/' + (From-CodePoints @(0x5b9e,0x9a8c,0x5206,0x6790,0x4e0e,0x5faa,0x73af,0x8fed,0x4ee3)) + '.md'
    $Files += 'docs/research/reports/2026-09-25/' + (From-CodePoints @(0x521d,0x6b65,0x8bca,0x65ad)) + '.md'
    $Files += 'docs/research/reports/2026-09-26/' + (From-CodePoints @(0x7cfb,0x7edf,0x5206,0x6790)) + '.md'
    foreach ($Relative in $Files) { Add-Entry (Join-Path $Root $Relative) $Relative $Records }
    foreach ($Optional in @('scripts/install.ps1')) {
        $Source = Join-Path $Root $Optional
        if (Test-Path -LiteralPath $Source) { Add-Entry $Source $Optional $Records }
    }
    foreach ($Mapping in @(
        @{source='docs/release/README.md'; target='README.md'},
        @{source='docs/release/PIPELINE.md'; target='PIPELINE.md'},
        @{source='docs/release/SHOWCASE.md'; target='docs/SHOWCASE.md'})) {
        $DocumentSource = Join-Path $Root $Mapping.source
        # The development checkout keeps its private README: curated release
        # sources always win. An already-published tree carries the mapped
        # documents themselves, so it can be repackaged without private docs.
        if (-not (Test-Path -LiteralPath $DocumentSource -PathType Leaf)) {
            $DocumentSource = Join-Path $Root $Mapping.target
        }
        Add-Entry $DocumentSource $Mapping.target $Records
    }
    $Research = Join-Path $Root 'config/research_projects'
    Add-Tree $Research 'config/research_projects' $Records
    $Images = Join-Path $Root 'docs/release/images'
    if (Test-Path -LiteralPath $Images) { Add-Tree $Images 'images' $Records }
    if ($IncludeTests) {
        foreach ($Relative in @('tests/test_compatible_authoring_runtime.py', 'tests/test_portable_runtime.py',
                'tests/test_portable_settings.py', 'tests/test_model_connections.py', 'tests/test_model_connections_ui.cjs',
                'tests/test_public_grammar_builders.py', 'tests/test_launcher_authoring_environment.py',
                'tests/test_public_setup.py', 'tests/test_native_grammar_readiness.py',
                'tests/test_release_entry_guards.py')) {
            $Source = Join-Path $Root $Relative
            if (Test-Path -LiteralPath $Source) { Add-Entry $Source $Relative $Records }
        }
    }
    $Seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($Entry in $Records) {
        if (-not $Seen.Add($Entry.path)) { throw "Duplicate destination refused: $($Entry.path)" }
    }
    return @($Records.ToArray() | Sort-Object path)
}

Assert-NoLink $Root
Assert-NoLink $Output
if (-not (Test-Path -LiteralPath $Root -PathType Container)) { throw 'ProjectRoot must exist.' }
if (-not $Output -or $Output -eq [IO.Path]::GetPathRoot($Output) -or
        $Root.Equals($Output, [StringComparison]::OrdinalIgnoreCase) -or
        $RootPrefix.StartsWith($Output + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'OutputDirectory cannot be a drive root, ProjectRoot, or an ancestor of ProjectRoot.'
}
# Do not create an output inside any recursively collected source directory.
foreach ($Directory in @('app', 'static', 'engine_extensions', 'resources', 'config/research_projects', 'docs/release/images')) {
    $Source = [IO.Path]::GetFullPath((Join-Path $Root $Directory)).TrimEnd([char[]]@('\', '/'))
    if ($Output.Equals($Source, [StringComparison]::OrdinalIgnoreCase) -or
            $Output.StartsWith($Source + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'OutputDirectory cannot be inside an allowlisted source tree.'
    }
}
if (Test-Path -LiteralPath $Output) {
    if (-not (Test-Path -LiteralPath $Output -PathType Container) -or
            @(Get-ChildItem -LiteralPath $Output -Force).Count -ne 0) {
        throw 'OutputDirectory must be new or empty; existing content is never overwritten or removed.'
    }
}
$Inventory = @(Get-Inventory)
[IO.Directory]::CreateDirectory($Output) | Out-Null
foreach ($Entry in $Inventory) {
    $Target = [IO.Path]::GetFullPath((Join-Path $Output $Entry.path))
    if (-not $Target.StartsWith($Output + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Resolved destination escaped OutputDirectory.'
    }
    Assert-NoLink $Entry.source
    Assert-NoLink $Target
    [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($Target)) | Out-Null
    [IO.File]::Copy($Entry.source, $Target, $false)
    if ((Get-Item -LiteralPath $Target).Length -ne $Entry.bytes -or (Get-Digest $Target) -ne $Entry.sha256) {
        throw "Source changed during build or copy hash mismatched: $($Entry.path). Partial output was retained."
    }
}
$After = @(Get-Inventory)
if (($Inventory | Select-Object path,bytes,sha256 | ConvertTo-Json -Depth 4 -Compress) -cne
        ($After | Select-Object path,bytes,sha256 | ConvertTo-Json -Depth 4 -Compress)) {
    throw 'Allowlisted working tree changed during build. Partial output was retained; use a new empty directory after changes settle.'
}
$Manifest = [ordered]@{
    schema_version='anime-pipeline-public-source-v1'
    built_at_utc=[DateTimeOffset]::UtcNow.ToString('o')
    source_kind='current_working_tree_including_untracked_allowlisted_files'
    scope='builder_owned_source_files_only; later manually added assets are not covered'
    files=@($Inventory | Select-Object path,bytes,sha256)
}
$ManifestPath = Join-Path $Output 'PUBLIC_MANIFEST.json'
$Stream = [IO.File]::Open($ManifestPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
try {
    $Bytes = $Utf8.GetBytes(($Manifest | ConvertTo-Json -Depth 8) + "`n")
    $Stream.Write($Bytes, 0, $Bytes.Length)
} finally { $Stream.Dispose() }
Write-Output ("Public source built: {0} files; manifest covers allowlisted source bytes only." -f $Inventory.Count)

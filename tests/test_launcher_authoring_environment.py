"""Windows launch environment contract; no registry writes or real services."""
import base64
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
POWERSHELL = shutil.which('powershell.exe')
pytestmark = pytest.mark.skipif(os.name != 'nt' or not POWERSHELL, reason='Windows PowerShell launch contract')


def run_powershell(body):
    helper = str(ROOT / 'scripts/runtime_common.ps1').replace("'", "''")
    script = """
$ErrorActionPreference = 'Stop'
. '%s'
$script:Reads = [Collections.Generic.List[string]]::new()
function Get-StudioEnvironmentValue([string]$Name, [EnvironmentVariableTarget]$Target) {
    [void]$script:Reads.Add($Name + ':' + [string]$Target)
    if ($Target -eq [EnvironmentVariableTarget]::Process) { return $script:FakeProcess }
    return $script:FakeUser
}
$script:FakeProcess = $null
$script:FakeUser = $null
$Name = 'AI_COMIC_LAUNCH_TEST_ONLY_71638'
[Environment]::SetEnvironmentVariable($Name, $null, 'Process')
$Settings = [pscustomobject]@{Raw=[pscustomobject]@{
    authoring_backend='openai_compatible'; authoring_api_key_env=$Name}}
%s
Write-Output 'launch-environment-ok'
""" % (helper, body)
    result = subprocess.run([POWERSHELL, '-NoProfile', '-NonInteractive', '-EncodedCommand',
        base64.b64encode(script.encode('utf-16-le')).decode('ascii')],
        capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'launch-environment-ok'


def test_user_environment_is_imported_only_into_process_without_output():
    run_powershell("""
$script:FakeUser = 'mock-user-value-not-a-real-key'
$Output = @(Initialize-StudioAuthoringEnvironment $Settings)
if ($Output.Count -ne 0) { throw 'Initialization must not output credentials.' }
if ([Environment]::GetEnvironmentVariable($Name, 'Process') -cne $script:FakeUser) { throw 'User value was not inherited.' }
if (($script:Reads -join ',') -cne ($Name + ':Process,' + $Name + ':User')) { throw 'Unexpected environment reads.' }
""")


def test_explicit_process_environment_takes_priority():
    run_powershell("""
$script:FakeProcess = 'mock-explicit-process-value'
$script:FakeUser = 'mock-user-value-not-selected'
[Environment]::SetEnvironmentVariable($Name, $script:FakeProcess, 'Process')
Initialize-StudioAuthoringEnvironment $Settings
if ($script:Reads.Count -ne 1) { throw 'User store should not be read.' }
if ([Environment]::GetEnvironmentVariable($Name, 'Process') -cne $script:FakeProcess) { throw 'Explicit process value changed.' }
""")


@pytest.mark.parametrize('backend', ['local_gguf', ''])
def test_local_or_legacy_config_never_reads_api_environment(backend):
    run_powershell("""
$Settings.Raw.authoring_backend = '%s'
Initialize-StudioAuthoringEnvironment $Settings
if ($script:Reads.Count -ne 0) { throw 'Local mode must not read credentials.' }
""" % backend)


def test_service_without_auth_remains_usable_without_a_key():
    run_powershell("""
Initialize-StudioAuthoringEnvironment $Settings
if ([Environment]::GetEnvironmentVariable($Name, 'Process')) { throw 'Missing values must not invent a key.' }
if ($script:Reads.Count -ne 2) { throw 'Expected only the configured process/user reads.' }
""")


def test_default_key_name_remains_an_environment_name():
    run_powershell("""
$Settings.Raw.authoring_api_key_env = ''
Initialize-StudioAuthoringEnvironment $Settings
if (($script:Reads -join ',') -cne 'AI_COMIC_AUTHORING_API_KEY:Process,AI_COMIC_AUTHORING_API_KEY:User') { throw 'Default key name mismatch.' }
""")


def test_invalid_name_is_rejected_without_echoing_it_or_reading_user_store():
    run_powershell("""
$Settings.Raw.authoring_api_key_env = 'mock-invalid-secret=not-a-name'
$Rejected = $false
try { Initialize-StudioAuthoringEnvironment $Settings }
catch {
    $Rejected = $true
    if ($_.Exception.Message.Contains('mock-invalid-secret')) { throw 'Do not echo an invalid secret-like setting.' }
}
if (-not $Rejected -or $script:Reads.Count -ne 0) { throw 'Invalid name was not rejected before reading.' }
""")


def test_real_frontend_launch_hook_inherits_key_before_mocked_start_process():
    run_powershell("""
$script:FakeUser = 'mock-inherited-by-child'
$script:ChildSawKey = $false
$Settings | Add-Member NoteProperty Python 'X:\\test-only\\python.exe'
$Settings | Add-Member NoteProperty Data ([IO.Path]::GetTempPath())
$Settings | Add-Member NoteProperty Root 'X:\\test-only'
$Settings | Add-Member NoteProperty WebPort 8090
function Test-Path { return $true }
function Assert-StudioPort { return $false }
function New-Item { }
function Start-Process {
    param($FilePath, $ArgumentList, $WorkingDirectory, $PassThru, $WindowStyle)
    if ([Environment]::GetEnvironmentVariable($Name, 'Process') -cne 'mock-inherited-by-child') { throw 'Child started without key.' }
    $script:ChildSawKey = $true
    return [pscustomobject]@{Id=71638}
}
function New-StudioReceipt { return [pscustomobject]@{pid=71638} }
function Save-StudioReceipt { }
$Child = Start-StudioFrontend $Settings ''
if (-not $script:ChildSawKey -or $Child.Id -ne 71638) { throw 'Frontend launch hook did not run.' }
""")

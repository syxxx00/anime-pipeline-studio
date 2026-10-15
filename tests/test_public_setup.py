"""Setup execution checks: real PS5/Python quoting, fully mocked installers.

Native Python probes import only stdlib. Full setup copies replace only the
native invocation boundary: no real pip, git, CUDA, service or model runs.
"""
from __future__ import annotations
import base64
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "scripts" / "setup.ps1"
POWERSHELL = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
pytestmark = pytest.mark.skipif(os.name != "nt" or not POWERSHELL.is_file(), reason="Windows PowerShell 5.1 execution contract")


def quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def run_ps(code):
    prefix = "$ErrorActionPreference='Stop';$ProgressPreference='SilentlyContinue';[Console]::OutputEncoding=[Text.Encoding]::UTF8;"
    encoded = base64.b64encode((prefix + code).encode("utf-16-le")).decode("ascii")
    process = subprocess.run(
        [str(POWERSHELL), "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
        encoding="utf-8", errors="replace", capture_output=True, timeout=30,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert process.returncode == 0, process.stdout + process.stderr
    return process.stdout.strip()


def production_functions():
    return (
        "$tokens=$null;$errors=$null;"
        f"$ast=[System.Management.Automation.Language.Parser]::ParseFile({quote(SETUP)},[ref]$tokens,[ref]$errors);"
        "if($errors.Count){throw ($errors|Out-String)};"
        "foreach($name in @('Invoke-Checked','Invoke-PythonCode')) {"
        "$function=$ast.Find({param($node)$node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $name},$true);"
        "if(-not $function){throw ('Production function absent: '+$name)};"
        ". ([ScriptBlock]::Create($function.Extent.Text));}")


def test_real_ps51_python_preserves_quotes_unicode_spaces_and_newlines():
    value = {"quoted": 'she said "hello"', "words": "two words", "unicode": "中文 空格", "single": "don't strip"}
    code = "import json,sys\nvalue=" + repr(value) + "\nassert len(sys.argv)==1\nprint(json.dumps(value,ensure_ascii=True))"
    stdout = run_ps(production_functions() + f"Invoke-PythonCode {quote(sys.executable)} {quote(code)}")
    assert json.loads(stdout) == value


@pytest.mark.parametrize("code, exit_code", [("raise SystemExit(7)", 7), ('raise RuntimeError("中文 with spaces")', 1)])
def test_real_ps51_python_nonzero_exception_is_not_reported_as_success(code, exit_code):
    result = json.loads(run_ps(
        production_functions()
        + f"try {{ Invoke-PythonCode {quote(sys.executable)} {quote(code)}; $failed=$false;$message='' }}"
        + "catch {$failed=$true;$message=$_.Exception.Message};"
        + "@{failed=$failed;message=$message}|ConvertTo-Json -Compress"))
    assert result["failed"] is True
    assert f"Command failed ({exit_code})" in result["message"]


RECORDER = r"""
function Invoke-Checked([string]$Executable, [string[]]$Arguments) {
    $global:NativeCalls.Add([pscustomobject]@{executable=$Executable;arguments=@($Arguments)}) | Out-Null
    if ($Executable -notin @($global:ExpectedBase,$global:ExpectedWeb,$global:ExpectedEngine,'MOCK_GIT')) {
        throw "Unexpected native executable in fixture: $Executable"
    }
    if ($Arguments.Count -ge 3 -and $Arguments[0] -eq '-m' -and $Arguments[1] -eq 'venv') {
        $Directory = [IO.Path]::GetFullPath($Arguments[2])
        if (-not $Directory.StartsWith($global:FixtureRoot,[StringComparison]::OrdinalIgnoreCase)) { throw 'Venv target left fixture' }
        $Scripts = Join-Path $Directory 'Scripts'
        [void][IO.Directory]::CreateDirectory($Scripts)
        [IO.File]::WriteAllText((Join-Path $Scripts 'python.exe'),'mock-created-python')
    }
    if ($Executable -eq 'MOCK_GIT' -and $Arguments[0] -eq 'clone') {
        $Directory = [IO.Path]::GetFullPath($Arguments[-1])
        if (-not $Directory.StartsWith($global:FixtureRoot,[StringComparison]::OrdinalIgnoreCase)) { throw 'Clone target left fixture' }
        [void][IO.Directory]::CreateDirectory($Directory)
        if ($Directory -eq $global:ExpectedCore) {
            [IO.File]::WriteAllText((Join-Path $Directory 'main.py'),'# mock Core')
            [IO.File]::WriteAllText((Join-Path $Directory 'requirements.txt'),'# mock requirements')
        } else { [IO.File]::WriteAllText((Join-Path $Directory '__init__.py'),'# mock node') }
    }
    if ($global:FailEngineCheck -and $Executable -eq $global:ExpectedEngine -and $Arguments[0] -eq '-c') {
        throw 'Mock existing inference dependency is missing'
    }
    $global:LASTEXITCODE=0
}
"""


def run_mock_setup(tmp_path, *, existing=False, update=False, prepare=True, fail_engine=False, same_environment=False):
    fixture = tmp_path / "isolated studio"
    scripts = fixture / "scripts"
    scripts.mkdir(parents=True)
    config = fixture / "config"
    config.mkdir()
    core = fixture / "runtime" / "ComfyUI"
    web = fixture / ".venv" / "Scripts" / "python.exe"
    engine = (web if same_environment else
              (fixture / "shared inference" / ".venv" / "Scripts" / "python.exe"
               if existing else core / ".venv" / "Scripts" / "python.exe"))
    settings = {"comfy_root": str(core), "engine_base_dir": str(core),
                "comfy_python": str(engine), "model_root": "models", "web_port": 18090, "engine_port": 18190}
    (config / "settings.example.json").write_text(json.dumps(settings), encoding="utf-8")
    if existing:
        core.mkdir(parents=True)
        (core / "main.py").write_text("# original checkout", encoding="utf-8")
        (core / "requirements.txt").write_text("# requirements", encoding="utf-8")
        engine.parent.mkdir(parents=True)
        engine.write_text("original-python-marker", encoding="utf-8")
        nodes = core / "custom_nodes" / "ComfyUI-Newbie-Nodes"
        nodes.mkdir(parents=True)
        (nodes / "__init__.py").write_text("# original nodes", encoding="utf-8")
    source = SETUP.read_text(encoding="utf-8")
    start = source.index("function Invoke-Checked(")
    end = source.index("function Invoke-PythonCode(", start)
    local_setup = scripts / "setup.ps1"
    local_setup.write_text(source[:start] + RECORDER + "\n" + source[end:], encoding="ascii")
    shutil.copyfile(ROOT / "scripts" / "runtime_common.ps1", scripts / "runtime_common.ps1")
    flags = (" -PrepareEngine" if prepare else " -WebOnly") + (" -UpdateEngineDependencies" if update else "")
    code = (
        f"$global:FixtureRoot={quote(fixture)};$global:ExpectedBase={quote(sys.executable)};"
        f"$global:ExpectedWeb={quote(web)};$global:ExpectedEngine={quote(engine)};"
        f"$global:ExpectedCore={quote(core)};$global:FailEngineCheck=${str(fail_engine).lower()};"
        "$global:NativeCalls=New-Object 'System.Collections.Generic.List[object]';"
        "function Get-Command {param($Name,$CommandType)"
        "if($Name -eq 'git'){[pscustomobject]@{Source='MOCK_GIT'}}"
        "else{Microsoft.PowerShell.Core\\Get-Command -Name $Name -CommandType Application -ErrorAction Stop}};"
        "function Invoke-WebRequest {throw 'Network forbidden'};"
        "function Invoke-RestMethod {throw 'Network forbidden'};"
        "function Start-Process {throw 'Process startup forbidden'};"
        "function Read-Host {throw 'Unplanned prompt in fixture'};"
        f"try {{ & {quote(local_setup)} -Python {quote(sys.executable)} -InstallDependencies"
        f" -ProjectRoot {quote(fixture)}{flags};$failed=$false;$message='' }}"
        "catch {$failed=$true;$message=$_.Exception.Message};"
        "$result=@{failed=$failed;message=$message;calls=@($global:NativeCalls.ToArray())};"
        "Write-Output ('__TEST_RESULT__'+($result|ConvertTo-Json -Depth 12 -Compress))")
    stdout = run_ps(code)
    result = json.loads(stdout.split("__TEST_RESULT__")[-1])
    result.update(fixture=fixture, core=core, web=web, engine=engine, stdout=stdout)
    return result


def engine_pip_calls(result):
    return [call for call in result["calls"]
            if call["executable"].lower() == str(result["engine"]).lower()
            and call["arguments"][:2] == ["-m", "pip"]]


def test_entire_fresh_setup_installs_only_mocked_new_environments_and_checkouts(tmp_path):
    result = run_mock_setup(tmp_path)
    assert result["failed"] is False, result["message"]
    assert len(engine_pip_calls(result)) == 3
    clones = [call for call in result["calls"] if call["executable"] == "MOCK_GIT" and call["arguments"][0] == "clone"]
    assert len(clones) == 2
    assert (result["fixture"] / "config" / "settings.json").is_file()
    assert str(result["web"]) in result["stdout"]
    assert str(result["engine"]) in result["stdout"]
    assert "Inference environment to create/update:" in result["stdout"]


def test_entire_existing_shared_inference_is_not_pip_modified_by_general_install_consent(tmp_path):
    result = run_mock_setup(tmp_path, existing=True)
    assert result["failed"] is False, result["message"]
    assert engine_pip_calls(result) == []
    assert not [call for call in result["calls"] if call["executable"] == "MOCK_GIT"]
    assert result["engine"].read_text() == "original-python-marker"
    assert (result["core"] / "main.py").read_text() == "# original checkout"
    assert "Inference environment reused without pip changes:" in result["stdout"]


def test_entire_existing_engine_update_requires_the_additional_explicit_flag(tmp_path):
    result = run_mock_setup(tmp_path, existing=True, update=True)
    assert result["failed"] is False, result["message"]
    assert len(engine_pip_calls(result)) == 3
    assert "Existing packages in this selected environment may change." in result["stdout"]


def test_existing_missing_inference_dependencies_do_not_trigger_an_automatic_repair(tmp_path):
    result = run_mock_setup(tmp_path, existing=True, fail_engine=True)
    assert result["failed"] is True
    assert engine_pip_calls(result) == []
    assert "left unchanged" in result["message"]
    assert "-UpdateEngineDependencies" in result["message"]


def test_web_only_has_no_engine_pip_clone_or_cuda_check(tmp_path):
    result = run_mock_setup(tmp_path, prepare=False)
    assert result["failed"] is False, result["message"]
    assert engine_pip_calls(result) == []
    assert not [call for call in result["calls"] if call["executable"] == "MOCK_GIT"]
    assert not [call for call in result["calls"] if call["executable"] == str(result["engine"])]


def test_shared_web_inference_python_requires_opt_in_before_any_pip(tmp_path):
    result = run_mock_setup(tmp_path, existing=True, same_environment=True)
    assert result["failed"] is True
    assert "same Python" in result["message"]
    assert not [call for call in result["calls"] if call["arguments"][:2] == ["-m", "pip"]]

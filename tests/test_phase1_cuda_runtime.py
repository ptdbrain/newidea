"""Tests for Phase 1 CUDA runtime selection and validation."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from scripts.check_phase1_cuda import validate_runtime


PROJECT_ROOT = Path(__file__).parents[1]
CUDA_HELPER = PROJECT_ROOT / "scripts" / "phase1_cuda_runtime.sh"
PHASE1_LAUNCHER = PROJECT_ROOT / "scripts" / "run_phase1_main.sh"


class _FakeCuda:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available

    def get_device_name(self, _device: int = 0) -> str:
        return "Test GPU"

    def get_device_capability(self, _device: int = 0) -> tuple[int, int]:
        return (9, 0)


def _fake_torch(
    torch_version: str, cuda_runtime: str | None, *, available: bool
) -> SimpleNamespace:
    return SimpleNamespace(
        __version__=torch_version,
        version=SimpleNamespace(cuda=cuda_runtime),
        cuda=_FakeCuda(available),
    )


def _bash_path(value: Path | str) -> str:
    path = str(value)
    if os.name == "nt":
        drive, tail = os.path.splitdrive(path)
        if drive:
            return f"/{drive[0].lower()}{tail.replace(os.sep, '/')}"
    return Path(path).as_posix()


def _bash() -> str:
    executable = shutil.which("bash")
    if os.name == "nt":
        git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")
        if git_bash.is_file():
            executable = str(git_bash)
    if executable is None:
        raise RuntimeError("bash is required for the CUDA runtime tests")
    return executable


def _run_cuda_helper(
    variant: str, requirements_path: Path
) -> subprocess.CompletedProcess[str]:
    command = """
source "$1"
variant="$2"
requirements_path="$3"
runtime="$(phase1_cuda_runtime_version "$variant")" || exit $?
index="$(phase1_torch_index_url "$variant")" || exit $?
fingerprint="$(phase1_dependency_fingerprint "$requirements_path" "$variant")" || exit $?
printf 'torch=%s\n' "$PHASE1_TORCH_VERSION"
printf 'runtime=%s\n' "$runtime"
printf 'index=%s\n' "$index"
printf 'fingerprint=%s\n' "$fingerprint"
"""
    return subprocess.run(
        [
            _bash(),
            "-c",
            command,
            "phase1-cuda-test",
            _bash_path(CUDA_HELPER),
            variant,
            _bash_path(requirements_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def _run_launcher(
    *arguments: str, environment: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    launcher_environment = os.environ.copy()
    launcher_environment.update(environment or {})
    return subprocess.run(
        [_bash(), _bash_path(PHASE1_LAUNCHER), *arguments],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=launcher_environment,
    )


def _write_checkpoint_fixture(path: Path, config_name: str) -> None:
    path.mkdir(parents=True)
    (path / config_name).write_text("{}\n", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"checkpoint")


def test_cuda_variant_mapping_defaults_are_stable(tmp_path: Path) -> None:
    requirements_path = tmp_path / "requirements.txt"
    requirements_path.write_text("transformers==4.51.1\n", encoding="utf-8")

    result = _run_cuda_helper("cu128", requirements_path)

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[:3] == [
        "torch=2.9.1",
        "runtime=12.8",
        "index=https://download.pytorch.org/whl/cu128",
    ]


def test_cuda_variant_mapping_supports_cu130(tmp_path: Path) -> None:
    requirements_path = tmp_path / "requirements.txt"
    requirements_path.write_text("transformers==4.51.1\n", encoding="utf-8")

    result = _run_cuda_helper("cu130", requirements_path)

    assert result.returncode == 0, result.stderr
    assert "runtime=13.0" in result.stdout
    assert "index=https://download.pytorch.org/whl/cu130" in result.stdout


def test_cuda_variant_mapping_rejects_unknown_value(tmp_path: Path) -> None:
    requirements_path = tmp_path / "requirements.txt"
    requirements_path.write_text("transformers==4.51.1\n", encoding="utf-8")

    result = _run_cuda_helper("cu129", requirements_path)

    assert result.returncode != 0
    assert "Supported values: cu128, cu130" in result.stderr


def test_dependency_fingerprint_changes_with_cuda_variant(tmp_path: Path) -> None:
    requirements_path = tmp_path / "requirements.txt"
    requirements_path.write_text("transformers==4.51.1\n", encoding="utf-8")

    cu128 = _run_cuda_helper("cu128", requirements_path)
    cu130 = _run_cuda_helper("cu130", requirements_path)

    assert cu128.returncode == 0, cu128.stderr
    assert cu130.returncode == 0, cu130.stderr
    cu128_fingerprint = cu128.stdout.splitlines()[3]
    cu130_fingerprint = cu130.stdout.splitlines()[3]
    assert cu128_fingerprint.startswith("fingerprint=")
    assert cu130_fingerprint.startswith("fingerprint=")
    assert cu128_fingerprint != cu130_fingerprint


def test_validate_runtime_accepts_matching_cu128() -> None:
    info = validate_runtime(
        _fake_torch("2.9.1+cu128", "12.8", available=True),
        "cu128",
        "2.9.1",
    )

    assert info.torch_version == "2.9.1+cu128"
    assert info.cuda_runtime == "12.8"
    assert info.gpu_name == "Test GPU"
    assert info.compute_capability == (9, 0)


def test_validate_runtime_rejects_wrong_cuda_build() -> None:
    with pytest.raises(
        RuntimeError, match=r"expected CUDA runtime 13\.0, got 12\.8"
    ):
        validate_runtime(
            _fake_torch("2.9.1+cu128", "12.8", available=True),
            "cu130",
            "2.9.1",
        )


def test_validate_runtime_rejects_cpu_only_torch() -> None:
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        validate_runtime(
            _fake_torch("2.9.1+cpu", None, available=False),
            "cu128",
            "2.9.1",
        )


def test_validate_runtime_rejects_wrong_torch_version() -> None:
    with pytest.raises(RuntimeError, match=r"expected Torch 2\.9\.1, got 2\.6\.0"):
        validate_runtime(
            _fake_torch("2.6.0+cu126", "12.6", available=True),
            "cu128",
            "2.9.1",
        )


def test_validate_runtime_rejects_unknown_variant() -> None:
    with pytest.raises(ValueError, match="Supported values: cu128, cu130"):
        validate_runtime(
            _fake_torch("2.9.1+cu128", "12.8", available=True),
            "cu129",
            "2.9.1",
        )


def test_launcher_dry_run_defaults_to_cu128() -> None:
    result = _run_launcher(
        "--bootstrap-only",
        environment={"DRY_RUN": "1", "PYTORCH_CUDA_VARIANT": ""},
    )

    assert result.returncode == 0, result.stderr
    assert "PyTorch: 2.9.1 (cu128 / CUDA 12.8)" in result.stdout
    assert (
        "PyTorch index: https://download.pytorch.org/whl/cu128" in result.stdout
    )


def test_launcher_dry_run_supports_cu130() -> None:
    result = _run_launcher(
        "--bootstrap-only",
        environment={"DRY_RUN": "1", "PYTORCH_CUDA_VARIANT": "cu130"},
    )

    assert result.returncode == 0, result.stderr
    assert "PyTorch: 2.9.1 (cu130 / CUDA 13.0)" in result.stdout
    assert (
        "PyTorch index: https://download.pytorch.org/whl/cu130" in result.stdout
    )


def test_launcher_rejects_unsupported_cuda_variant() -> None:
    result = _run_launcher(
        "--bootstrap-only",
        environment={"DRY_RUN": "1", "PYTORCH_CUDA_VARIANT": "cu129"},
    )

    assert result.returncode == 2
    assert "Supported values: cu128, cu130" in result.stderr


def test_launcher_installs_torch_before_application_requirements(
    tmp_path: Path,
) -> None:
    venv_dir = tmp_path / "venv"
    python_path = venv_dir / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >> \"$PHASE1_PYTHON_LOG\"\n",
        encoding="utf-8",
    )
    python_path.chmod(0o755)

    model_path = tmp_path / "model"
    embedding_path = tmp_path / "embedding"
    _write_checkpoint_fixture(model_path, "config.json")
    _write_checkpoint_fixture(embedding_path, "modules.json")
    python_log = tmp_path / "python.log"

    result = _run_launcher(
        "--bootstrap-only",
        environment={
            "DRY_RUN": "0",
            "SKIP_BOOTSTRAP": "0",
            "PYTORCH_CUDA_VARIANT": "cu128",
            "VENV_DIR": _bash_path(venv_dir),
            "MODEL_PATH": _bash_path(model_path),
            "EMBEDDING_PATH": _bash_path(embedding_path),
            "PHASE1_PYTHON_LOG": _bash_path(python_log),
        },
    )

    assert result.returncode == 0, result.stderr
    invocations = python_log.read_text(encoding="utf-8").splitlines()
    torch_install = next(
        index
        for index, invocation in enumerate(invocations)
        if "torch==2.9.1" in invocation
    )
    requirements_install = next(
        index
        for index, invocation in enumerate(invocations)
        if "-r requirements.txt" in invocation
    )
    assert torch_install < requirements_install
    assert (
        "--index-url https://download.pytorch.org/whl/cu128"
        in invocations[torch_install]
    )


def test_launcher_runs_cuda_kivi_preflight_before_pipeline(tmp_path: Path) -> None:
    venv_dir = tmp_path / "venv"
    python_path = venv_dir / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >> \"$PHASE1_PYTHON_LOG\"\n"
        "if [[ \"$*\" == *\"-m scripts.check_phase1_cuda\"* ]]; then exit 23; fi\n",
        encoding="utf-8",
    )
    python_path.chmod(0o755)
    flock_path = venv_dir / "bin" / "flock"
    flock_path.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    flock_path.chmod(0o755)

    model_path = tmp_path / "model"
    embedding_path = tmp_path / "embedding"
    _write_checkpoint_fixture(model_path, "config.json")
    _write_checkpoint_fixture(embedding_path, "modules.json")
    python_log = tmp_path / "python.log"

    result = _run_launcher(
        environment={
            "DRY_RUN": "0",
            "SKIP_BOOTSTRAP": "1",
            "PYTORCH_CUDA_VARIANT": "cu130",
            "VENV_DIR": _bash_path(venv_dir),
            "MODEL_PATH": _bash_path(model_path),
            "EMBEDDING_PATH": _bash_path(embedding_path),
            "RUN_DIR": _bash_path(tmp_path / "run"),
            "PHASE1_PYTHON_LOG": _bash_path(python_log),
        },
    )

    assert result.returncode != 0
    assert python_log.is_file(), f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    invocations = python_log.read_text(encoding="utf-8").splitlines()
    assert any(
        invocation
        == "-m scripts.check_phase1_cuda --variant cu130 --torch-version 2.9.1 --device cuda:0"
        for invocation in invocations
    )
    assert not any("dataset.prepare_phase1" in invocation for invocation in invocations)

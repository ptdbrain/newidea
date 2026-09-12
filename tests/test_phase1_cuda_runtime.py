"""Tests for Phase 1 CUDA runtime selection and validation."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess


PROJECT_ROOT = Path(__file__).parents[1]
CUDA_HELPER = PROJECT_ROOT / "scripts" / "phase1_cuda_runtime.sh"


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

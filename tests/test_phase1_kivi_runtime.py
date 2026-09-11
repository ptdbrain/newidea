"""Tests for the offline, pre-staged KIVI runtime helpers."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess


KIVI_COMMIT = "876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6"


def _bash_path(value: Path | str) -> str:
    path = str(value)
    if os.name == "nt":
        drive, tail = os.path.splitdrive(path)
        if drive:
            return f"/{drive[0].lower()}{tail.replace(os.sep, '/')}"
    return Path(path).as_posix()


def _run_helper(helper: Path, *args: Path | str, path: str = "") -> subprocess.CompletedProcess[str]:
    bash = shutil.which("bash")
    if os.name == "nt":
        git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")
        if git_bash.is_file():
            bash = str(git_bash)
    if bash is None:
        raise RuntimeError("bash is required for the shell-helper tests")
    command = (
        "PATH=; export PATH; source \"$1\"; shift; "
        "phase1_kivi_prepare \"$@\"; phase1_kivi_commit \"$2\" \"$3\""
    )
    environment = os.environ.copy()
    environment["PATH"] = path
    return subprocess.run(
        [bash, "-c", command, "phase1-kivi-test", _bash_path(helper), *map(_bash_path, args)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_pre_staged_kivi_does_not_require_git(tmp_path: Path) -> None:
    repo_root = tmp_path
    kivi_root = repo_root / "third_party" / "KIVI"
    (kivi_root / "quant").mkdir(parents=True)
    (kivi_root / "quant" / "new_pack.py").write_text("# staged KIVI\n", encoding="utf-8")
    commit_file = repo_root / "third_party" / "KIVI.commit"
    commit_file.write_text(f"{KIVI_COMMIT}\n", encoding="utf-8")

    result = _run_helper(
        Path(__file__).parents[1] / "scripts" / "phase1_kivi_runtime.sh",
        repo_root,
        kivi_root,
        commit_file,
    )

    assert result.returncode == 0, result.stderr
    assert "pre-staged KIVI" in result.stdout
    assert KIVI_COMMIT in result.stdout


def test_missing_kivi_fails_without_git(tmp_path: Path) -> None:
    repo_root = tmp_path
    kivi_root = repo_root / "third_party" / "KIVI"
    kivi_root.mkdir(parents=True)
    commit_file = repo_root / "third_party" / "KIVI.commit"

    result = _run_helper(
        Path(__file__).parents[1] / "scripts" / "phase1_kivi_runtime.sh",
        repo_root,
        kivi_root,
        commit_file,
    )

    assert result.returncode != 0
    assert "pre-stage" in result.stderr


def test_launcher_delegates_kivi_setup_to_git_free_runtime_helper() -> None:
    launcher = (Path(__file__).parents[1] / "scripts" / "run_phase1_main.sh").read_text(
        encoding="utf-8"
    )

    assert 'source "$REPO_ROOT/scripts/phase1_kivi_runtime.sh"' in launcher
    assert 'phase1_kivi_prepare "$REPO_ROOT" "$KIVI_ROOT" "$KIVI_COMMIT_FILE"' in launcher
    assert "git submodule update --init --recursive" not in launcher
    assert "git -C third_party/KIVI rev-parse HEAD" not in launcher


def test_offline_entrypoint_extracts_bundle_and_skips_bootstrap() -> None:
    entrypoint = (Path(__file__).parents[1] / "scripts" / "run_phase1_offline.sh").read_text(
        encoding="utf-8"
    )

    assert "PHASE1_KIVI_BUNDLE" in entrypoint
    assert "phase1-kivi-bundle.tar.gz" in entrypoint
    assert "tar -xzf" in entrypoint
    assert "export SKIP_BOOTSTRAP=1" in entrypoint
    assert 'exec bash "$REPO_ROOT/scripts/run_phase1_main.sh" "$@"' in entrypoint

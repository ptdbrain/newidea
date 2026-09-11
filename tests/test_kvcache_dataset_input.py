"""Tests for generic prompt records used by Phase 1 datasets."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "inference"))

from get_kvcache import (  # noqa: E402
    extract_user_input,
    process_dataset_records,
    validate_local_checkpoint,
)


def _complete_checkpoint(root: Path) -> Path:
    root.mkdir()
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"weights")
    (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    return root


def test_validate_local_checkpoint_accepts_complete_layout(tmp_path: Path) -> None:
    validate_local_checkpoint(_complete_checkpoint(tmp_path / "model"))


def test_validate_local_checkpoint_reports_missing_tokenizer(tmp_path: Path) -> None:
    model_path = _complete_checkpoint(tmp_path / "model")
    (model_path / "tokenizer.json").unlink()

    with pytest.raises(FileNotFoundError, match="tokenizer.json or tokenizer.model"):
        validate_local_checkpoint(model_path)


def test_extract_user_input_prefers_normalized_prompt_field():
    assert extract_user_input({"prompt": "  keep this prompt  "}, "phase1.jsonl") == "keep this prompt"


def test_extract_user_input_supports_lmsys_messages():
    record = {"messages": [{"role": "user", "content": "hello"}]}
    assert extract_user_input(record, "phase1.jsonl") == "hello"


def test_process_dataset_records_propagates_sample_failure(tmp_path: Path) -> None:
    dataset = [{"sample_id": "prompt-1", "prompt": "hello"}]

    def failing_prefill(*args, **kwargs):
        raise ValueError("bad cache format")

    with pytest.raises(RuntimeError, match="prompt-1.*bad cache format") as raised:
        process_dataset_records(
            object(),
            object(),
            dataset,
            Path("phase1.jsonl"),
            tmp_path,
            minimal_cache=True,
            prefill_fn=failing_prefill,
        )

    assert isinstance(raised.value.__cause__, ValueError)


def test_process_dataset_records_rejects_missing_prompt(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="prompt-1.*no usable prompt"):
        process_dataset_records(
            object(),
            object(),
            [{"sample_id": "prompt-1"}],
            Path("phase1.jsonl"),
            tmp_path,
            minimal_cache=True,
        )


def test_process_dataset_records_requires_canonical_cache_file(tmp_path: Path) -> None:
    def prefill_without_artifact(*args, **kwargs):
        return (), None

    with pytest.raises(RuntimeError, match="produced no canonical cache"):
        process_dataset_records(
            object(),
            object(),
            [{"sample_id": "prompt-1", "prompt": "hello"}],
            Path("phase1.jsonl"),
            tmp_path,
            minimal_cache=True,
            prefill_fn=prefill_without_artifact,
        )

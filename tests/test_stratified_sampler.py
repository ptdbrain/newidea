"""Tests for deterministic, provenance-preserving prompt sampling."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from dataset.stratified_sampler import stratified_sample  # noqa: E402


class _Tokenizer:
    def __call__(self, text, add_special_tokens=False, truncation=False):
        del add_special_tokens, truncation
        return {"input_ids": list(range(len(text.split())))}


def _record(index, prompt):
    return {
        "conversation_id": f"conversation-{index}",
        "conversation": [{"role": "user", "content": prompt}],
        "language": "English",
    }


def test_stratified_sample_is_seeded_and_records_provenance():
    records = [
        _record(0, "short prompt"),
        _record(1, " ".join(f"medium{i}" for i in range(40))),
        _record(2, " ".join(f"long{i}" for i in range(70))),
        _record(3, "short prompt"),  # duplicate prompt must be removed
    ]

    first = stratified_sample(
        records,
        _Tokenizer(),
        strata_counts={"short": 1, "medium": 1, "long": 1},
        seed=42,
    )
    second = stratified_sample(
        records,
        _Tokenizer(),
        strata_counts={"short": 1, "medium": 1, "long": 1},
        seed=42,
    )

    assert [item["sample_id"] for item in first] == [item["sample_id"] for item in second]
    assert [item["prompt_token_length"] for item in first] == [2, 40, 70]
    assert [item["length_stratum"] for item in first] == ["short", "medium", "long"]
    assert all(item["source_row_id"].startswith("conversation-") for item in first)


def test_stratified_sample_fails_if_a_required_stratum_is_unavailable():
    with pytest.raises(ValueError, match="long"):
        stratified_sample(
            [_record(0, "short prompt")],
            _Tokenizer(),
            strata_counts={"short": 1, "long": 1},
            seed=42,
        )

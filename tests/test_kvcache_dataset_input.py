"""Tests for generic prompt records used by Phase 1 datasets."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "inference"))

from get_kvcache import extract_user_input  # noqa: E402


def test_extract_user_input_prefers_normalized_prompt_field():
    assert extract_user_input({"prompt": "  keep this prompt  "}, "phase1.jsonl") == "keep this prompt"


def test_extract_user_input_supports_lmsys_messages():
    record = {"messages": [{"role": "user", "content": "hello"}]}
    assert extract_user_input(record, "phase1.jsonl") == "hello"

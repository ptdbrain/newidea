import json
import hashlib

from dataset.prepare_phase1 import prepare_records, sha256_file


class TinyTokenizer:
    def __call__(self, text, add_special_tokens=False, truncation=False):
        return {"input_ids": text.split()}


def test_prepare_records_adds_manifest_fields_and_is_deterministic(tmp_path):
    records = [
        {"id": f"short-{i}", "messages": [{"role": "user", "content": f"short {i}"}]}
        for i in range(8)
    ] + [
        {"id": f"medium-{i}", "prompt": " ".join([f"medium-{i}"] * 40)}
        for i in range(8)
    ] + [
        {"id": f"long-{i}", "prompt": " ".join([f"long-{i}"] * 80)}
        for i in range(8)
    ]

    first = prepare_records(
        records,
        TinyTokenizer(),
        strata_counts={"short": 3, "medium": 3, "long": 3},
        seed=42,
    )
    second = prepare_records(
        records,
        TinyTokenizer(),
        strata_counts={"short": 3, "medium": 3, "long": 3},
        seed=42,
    )

    assert [item["sample_id"] for item in first] == [item["sample_id"] for item in second]
    assert {item["length_stratum"] for item in first} == {"short", "medium", "long"}
    assert all("prompt_token_length" in item for item in first)


def test_sha256_file(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps({"prompt": "hello"}) + "\n", encoding="utf-8")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert sha256_file(path) == expected

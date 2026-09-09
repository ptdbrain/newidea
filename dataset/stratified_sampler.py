"""Deterministic length-stratified sampling for Phase 1 prompt sets.

The sampler is deliberately independent of Hugging Face ``datasets`` so it
can operate on the repository's normalized JSONL files and preserve the
source row/provenance fields used by the experiment manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Mapping, Sequence


STRATA = {
    "short": (0, 31),
    "medium": (32, 63),
    "long": (64, 128),
}


def _extract_prompt(record: Mapping[str, Any]) -> str | None:
    prompt = record.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()

    for field in ("conversation", "messages"):
        messages = record.get(field)
        if not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)):
            continue
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                content = message["content"].strip()
                if content:
                    return content

    for field in ("question", "instruction"):
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _token_length(tokenizer: Any, prompt: str) -> int:
    encoded = tokenizer(prompt, add_special_tokens=False, truncation=False)
    input_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    if hasattr(input_ids, "shape") and len(input_ids.shape) == 2:
        return int(input_ids.shape[-1])
    return len(input_ids)


def _stratum(token_length: int) -> str | None:
    for name, (lower, upper) in STRATA.items():
        if lower <= token_length <= upper:
            return name
    return None


def _source_row_id(record: Mapping[str, Any], index: int) -> str:
    for field in ("source_row_id", "conversation_id", "id"):
        value = record.get(field)
        if value is not None and str(value):
            return str(value)
    return f"row-{index}"


def _sample_id(source_row_id: str, prompt: str) -> str:
    digest = hashlib.sha256(f"{source_row_id}\0{prompt}".encode("utf-8")).hexdigest()
    return f"prompt-{digest[:16]}"


def stratified_sample(
    records: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    strata_counts: Mapping[str, int],
    seed: int = 42,
    max_length: int = 128,
) -> list[dict[str, Any]]:
    """Select a deterministic, deduplicated prompt set by token-length stratum.

    ``strata_counts`` is intentionally explicit.  If a requested stratum does
    not contain enough candidates, the function raises instead of silently
    returning an imbalanced experiment set.
    """
    unknown = set(strata_counts) - set(STRATA)
    if unknown:
        raise ValueError(f"unknown length strata: {sorted(unknown)}")
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if any(count < 0 for count in strata_counts.values()):
        raise ValueError("strata counts must be non-negative")

    buckets: dict[str, list[dict[str, Any]]] = {name: [] for name in STRATA}
    seen_prompts: set[str] = set()
    for index, record in enumerate(records):
        prompt = _extract_prompt(record)
        if prompt is None or prompt in seen_prompts:
            continue
        length = _token_length(tokenizer, prompt)
        if length > max_length:
            continue
        stratum = _stratum(length)
        if stratum is None:
            continue
        seen_prompts.add(prompt)
        source_row_id = _source_row_id(record, index)
        item = dict(record)
        item.update(
            {
                "prompt": prompt,
                "sample_id": _sample_id(source_row_id, prompt),
                "source_row_id": source_row_id,
                "prompt_token_length": length,
                "length_stratum": stratum,
            }
        )
        buckets[stratum].append(item)

    missing = {
        name: int(count) - len(buckets[name])
        for name, count in strata_counts.items()
        if len(buckets[name]) < count
    }
    if missing:
        details = ", ".join(
            f"{name}: need {strata_counts[name]}, have {len(buckets[name])}"
            for name in missing
        )
        raise ValueError(f"insufficient prompts for requested strata ({details})")

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for name in STRATA:
        count = int(strata_counts.get(name, 0))
        selected.extend(rng.sample(buckets[name], count))
    return selected


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--short", type=int, default=0)
    parser.add_argument("--medium", type=int, default=0)
    parser.add_argument("--long", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    selected = stratified_sample(
        _read_jsonl(args.input),
        tokenizer,
        strata_counts={"short": args.short, "medium": args.medium, "long": args.long},
        seed=args.seed,
        max_length=args.max_length,
    )
    _write_jsonl(args.output, selected)

    if args.manifest:
        manifest = {
            "input": str(args.input),
            "output": str(args.output),
            "tokenizer": args.tokenizer,
            "seed": args.seed,
            "max_length": args.max_length,
            "strata_counts": {
                "short": args.short,
                "medium": args.medium,
                "long": args.long,
            },
            "selected_count": len(selected),
            "sample_ids": [item["sample_id"] for item in selected],
            "prompt_token_lengths": [item["prompt_token_length"] for item in selected],
        }
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(selected)} prompts to {args.output}")


if __name__ == "__main__":
    main()

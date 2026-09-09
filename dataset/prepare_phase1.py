"""Prepare a reproducible, length-stratified Phase 1 prompt set.

The default source is the pinned public mirror documented by the Phase 1
plan.  The command uses HF streaming so it does not download the complete
one-million-row dataset.  A local JSONL input is also supported for offline
reproduction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from dataset.stratified_sampler import STRATA, _stratum, _token_length, stratified_sample


DEFAULT_SOURCE_DATASET = "natong19/lmsys-chat-1m-filtered"
DEFAULT_SOURCE_REVISION = "1887528a022e25be62eb9bb15e62675f2a69353b"
DEFAULT_LICENSE = "CC-BY-4.0"
DEFAULT_COUNTS = {"short": 25, "medium": 50, "long": 25}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_records(
    records: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    strata_counts: Mapping[str, int] = DEFAULT_COUNTS,
    seed: int = 42,
    max_length: int = 128,
) -> list[dict[str, Any]]:
    """Select and annotate a deterministic prompt set from JSON-like rows."""
    return stratified_sample(
        records,
        tokenizer,
        strata_counts=dict(strata_counts),
        seed=seed,
        max_length=max_length,
    )


def _prompt_from_record(record: Mapping[str, Any], tokenizer: Any, max_length: int) -> str | None:
    """Return a prompt only when it belongs to a configured length stratum."""
    from dataset.stratified_sampler import _extract_prompt

    prompt = _extract_prompt(record)
    if prompt is None:
        return None
    length = _token_length(tokenizer, prompt)
    if length > max_length or _stratum(length) is None:
        return None
    return prompt


def collect_candidates(
    rows: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    *,
    strata_counts: Mapping[str, int],
    max_length: int,
    candidate_multiplier: int = 4,
    max_scan_rows: int = 100_000,
) -> tuple[list[dict[str, Any]], int]:
    """Collect a bounded candidate pool while preserving source order.

    The pool is intentionally larger than the requested sample.  This keeps
    the final sample deterministic but less sensitive to the first qualifying
    rows in a streaming source.
    """
    if candidate_multiplier < 1:
        raise ValueError("candidate_multiplier must be positive")
    if max_scan_rows <= 0:
        raise ValueError("max_scan_rows must be positive")

    targets = {
        name: int(strata_counts.get(name, 0)) * candidate_multiplier
        for name in STRATA
    }
    counts = {name: 0 for name in STRATA}
    candidates: list[dict[str, Any]] = []
    seen_prompts: set[str] = set()

    scanned = 0
    for row in rows:
        scanned += 1
        if scanned > max_scan_rows:
            break
        prompt = _prompt_from_record(row, tokenizer, max_length)
        if prompt is None or prompt in seen_prompts:
            continue
        from dataset.stratified_sampler import _stratum

        stratum = _stratum(_token_length(tokenizer, prompt))
        if stratum is None:
            continue
        seen_prompts.add(prompt)
        candidates.append(dict(row))
        counts[stratum] += 1
        if all(counts[name] >= targets[name] for name in STRATA):
            break
    return candidates, scanned


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def _load_stream(source_dataset: str, revision: str):
    from datasets import load_dataset

    return load_dataset(
        source_dataset,
        split="train",
        streaming=True,
        revision=revision,
    )


def build_provenance(
    *,
    source_dataset: str,
    source_revision: str,
    source_license: str,
    output: Path,
    tokenizer: str,
    seed: int,
    max_length: int,
    strata_counts: Mapping[str, int],
    candidate_rows_scanned: int,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    output = Path(output)
    return {
        "source_dataset": source_dataset,
        "source_url": f"https://huggingface.co/datasets/{source_dataset}",
        "source_revision": source_revision,
        "source_license": source_license,
        "local_file": str(output),
        "local_sha256": sha256_file(output),
        "sample_count": len(records),
        "selection": "deterministic deduplicated length-stratified sample",
        "seed": seed,
        "max_length": max_length,
        "strata_counts": dict(strata_counts),
        "candidate_rows_scanned": candidate_rows_scanned,
        "tokenizer": tokenizer,
        "sample_ids": [item["sample_id"] for item in records],
        "prompt_token_lengths": [item["prompt_token_length"] for item in records],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--source-dataset", default=DEFAULT_SOURCE_DATASET)
    parser.add_argument("--revision", default=DEFAULT_SOURCE_REVISION)
    parser.add_argument("--license", default=DEFAULT_LICENSE)
    parser.add_argument("--local-input", type=Path)
    parser.add_argument("--short", type=int, default=DEFAULT_COUNTS["short"])
    parser.add_argument("--medium", type=int, default=DEFAULT_COUNTS["medium"])
    parser.add_argument("--long", type=int, default=DEFAULT_COUNTS["long"])
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--candidate-multiplier", type=int, default=4)
    parser.add_argument("--max-scan-rows", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    counts = {"short": args.short, "medium": args.medium, "long": args.long}
    if args.local_input:
        rows = _read_jsonl(args.local_input)
        candidates = rows
        scanned = len(rows)
        source_dataset = args.source_dataset
        source_revision = args.revision
    else:
        rows = _load_stream(args.source_dataset, args.revision)
        candidates, scanned = collect_candidates(
            rows,
            tokenizer,
            strata_counts=counts,
            max_length=args.max_length,
            candidate_multiplier=args.candidate_multiplier,
            max_scan_rows=args.max_scan_rows,
        )
        source_dataset = args.source_dataset
        source_revision = args.revision

    selected = prepare_records(
        candidates,
        tokenizer,
        strata_counts=counts,
        seed=args.seed,
        max_length=args.max_length,
    )
    _write_jsonl(args.output, selected)
    provenance = build_provenance(
        source_dataset=source_dataset,
        source_revision=source_revision,
        source_license=args.license,
        output=args.output,
        tokenizer=args.tokenizer,
        seed=args.seed,
        max_length=args.max_length,
        strata_counts=counts,
        candidate_rows_scanned=scanned,
        records=selected,
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(selected)} prompts to {args.output}")
    print(f"Wrote provenance to {args.manifest}")


if __name__ == "__main__":
    main()

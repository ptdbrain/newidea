"""Normalize and aggregate Phase 1 Shadow attack outputs.

The upstream attack harness writes a legacy field named ``BERTScore`` even
though it computes cosine similarity with ``all-mpnet-base-v2``.  This module
renames that field in derived outputs so the report cannot overclaim the
metric.  Raw JSONL files are never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from src.provenance import KIVI_COMMIT, KVCLOAK_COMMIT


def _sha1_prompt(prompt: str) -> str:
    return hashlib.sha1(prompt.encode("utf-8")).hexdigest()


def _dataset_index(dataset_records: Iterable[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    index = {}
    for record in dataset_records:
        prompt = record.get("prompt")
        if isinstance(prompt, str):
            index[_sha1_prompt(prompt)] = record
    return index


def _layer_position(attack: str, layer: Any, num_layers: int | None = None) -> str:
    if attack == "injection" or layer is None:
        return "all"
    try:
        layer_index = int(layer)
    except (TypeError, ValueError):
        return str(layer)
    if layer_index == 0:
        return "first"
    if num_layers is not None and layer_index == num_layers - 1:
        return "last"
    return "mid"


def normalize_record(
    raw: Mapping[str, Any],
    *,
    condition: str,
    dataset_index: Mapping[str, Mapping[str, Any]],
    model: str | None = None,
    dataset: str | None = None,
    seed: int | None = None,
    num_layers: int | None = None,
) -> dict[str, Any]:
    """Convert one legacy attack record into the Phase 1 report vocabulary."""
    input_hash = str(raw.get("input hash", raw.get("input_hash", "")))
    sample = dataset_index.get(input_hash, {})
    attack = str(raw.get("attack type", raw.get("attack_type", "unknown")))
    semantic = raw.get("BERTScore", raw.get("semantic_cosine_mpnet"))
    rouge = raw.get("ROUGE_L[f1_score]", raw.get("rouge_l"))
    if isinstance(rouge, Mapping):
        rouge = rouge.get("f1", rouge.get("f1_score"))
    condition_config = {
        "FP16": {"kv_method": "none", "kv_bits": None, "mode": "identity", "residual_length": None},
        "KIVI4-STD": {"kv_method": "kivi", "kv_bits": 4, "mode": "standard", "residual_length": 32},
        "KIVI4-FQ": {"kv_method": "kivi", "kv_bits": 4, "mode": "full_prompt_quantized", "residual_length": 32},
        "KIVI2-STD": {"kv_method": "kivi", "kv_bits": 2, "mode": "standard", "residual_length": 32},
        "KIVI2-FQ": {"kv_method": "kivi", "kv_bits": 2, "mode": "full_prompt_quantized", "residual_length": 32},
    }.get(condition, {"kv_method": "kivi", "kv_bits": None, "mode": None, "residual_length": None})
    input_ids = raw.get("input token ids", raw.get("input_token_ids"))
    result_ids = raw.get("result token ids", raw.get("result_token_ids"))
    if isinstance(input_ids, list) and isinstance(result_ids, list):
        compared = min(len(input_ids), len(result_ids))
        matches = sum(input_ids[index] == result_ids[index] for index in range(compared))
        token_accuracy = matches / len(input_ids) if input_ids else 0.0
        full_exact = len(input_ids) == len(result_ids) and matches == len(input_ids)
    else:
        token_accuracy = None
        full_exact = None
    result = {
        "sample_id": sample.get("sample_id"),
        "source_row_id": sample.get("source_row_id"),
        "model": model or raw.get("target model"),
        "dataset": dataset or raw.get("dataset"),
        "input_hash": input_hash,
        "input_token_length": sample.get("prompt_token_length"),
        "length_stratum": sample.get("length_stratum"),
        "condition": condition,
        **condition_config,
        "attack": attack,
        "attacker_protocol": "quantization_mismatch",
        "layer": raw.get("layer"),
        "layer_position": _layer_position(attack, raw.get("layer"), num_layers),
        "semantic_cosine_mpnet": float(semantic) if semantic is not None else None,
        "rouge_l": float(rouge) if rouge is not None else None,
        "token_accuracy": token_accuracy,
        "exact_match": full_exact,
        "full_prompt_exact_match": full_exact,
        "attack_time_sec": float(raw.get("time", raw.get("time_sec", 0.0))),
        "candidate_evals": raw.get("candidate_evals"),
        "candidate_evals_per_token": raw.get("candidate_evals_per_token"),
        "seed": seed,
        "kivi_commit": KIVI_COMMIT,
        "kvcloak_commit": KVCLOAK_COMMIT,
    }
    if result["sample_id"] is None:
        # Preserve traceability when a legacy file contains a prompt not in the
        # current manifest instead of silently dropping the record.
        result["sample_id"] = f"unmapped-{input_hash or 'unknown'}"
    return result


def _bootstrap_mean(values: np.ndarray, reps: int, rng: np.random.Generator) -> tuple[float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"))
    if values.size == 1:
        return (float(values[0]), float(values[0]))
    draws = rng.choice(values, size=(reps, values.size), replace=True).mean(axis=1)
    return (float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)))


def aggregate_records(
    raw_by_condition: Mapping[str, Iterable[Mapping[str, Any]]],
    dataset_records: Iterable[Mapping[str, Any]],
    *,
    bootstrap_reps: int = 2000,
    seed: int = 42,
    model: str | None = None,
    dataset: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return normalized records and per-metric aggregate rows."""
    if bootstrap_reps <= 0:
        raise ValueError("bootstrap_reps must be positive")
    dataset_records = list(dataset_records)
    index = _dataset_index(dataset_records)
    normalized: list[dict[str, Any]] = []
    for condition, records in raw_by_condition.items():
        records = list(records)
        layer_indices = []
        for record in records:
            try:
                if record.get("layer") is not None:
                    layer_indices.append(int(record["layer"]))
            except (TypeError, ValueError):
                pass
        num_layers = max(layer_indices) + 1 if layer_indices else None
        normalized.extend(
            normalize_record(
                record,
                condition=condition,
                dataset_index=index,
                model=model,
                dataset=dataset,
                seed=seed,
                num_layers=num_layers,
            )
            for record in records
        )

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for record in normalized:
        for metric in ("semantic_cosine_mpnet", "rouge_l"):
            value = record.get(metric)
            if value is None or not np.isfinite(value):
                continue
            key = (record["condition"], record["attack"], record["layer_position"], metric)
            grouped.setdefault(key, []).append(record)

    rng = np.random.default_rng(seed)
    summary: list[dict[str, Any]] = []
    for (condition, attack, layer_position, metric), records in sorted(grouped.items()):
        values = np.asarray([float(record[metric]) for record in records], dtype=float)
        baseline_records = {
            record["sample_id"]: float(record[metric])
            for record in grouped.get(("FP16", attack, layer_position, metric), [])
        }
        paired = [
            float(record[metric]) - baseline_records[record["sample_id"]]
            for record in records
            if record["sample_id"] in baseline_records
        ]
        paired_values = np.asarray(paired, dtype=float)
        ci_low, ci_high = _bootstrap_mean(values, bootstrap_reps, rng)
        delta_low, delta_high = _bootstrap_mean(paired_values, bootstrap_reps, rng)
        summary.append(
            {
                "condition": condition,
                "attack": attack,
                "layer_position": layer_position,
                "metric": metric,
                "n": int(values.size),
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "paired_n": int(paired_values.size),
                "baseline_mean": (
                    float(np.asarray(list(baseline_records.values()), dtype=float).mean())
                    if baseline_records
                    else None
                ),
                "delta_mean": float(paired_values.mean()) if paired_values.size else None,
                "delta_ci95_low": delta_low if paired_values.size else None,
                "delta_ci95_high": delta_high if paired_values.size else None,
            }
        )
    return normalized, summary


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else ["condition", "attack", "metric"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["condition", "attack", "layer", "metric", "n", "mean", "delta", "delta 95% CI"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        delta = row.get("delta_mean")
        ci = (row.get("delta_ci95_low"), row.get("delta_ci95_high"))
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("condition", "")),
                    str(row.get("attack", "")),
                    str(row.get("layer_position", "")),
                    str(row.get("metric", "")),
                    str(row.get("n", "")),
                    f"{row.get('mean', float('nan')):.4f}",
                    "-" if delta is None else f"{delta:.4f}",
                    "-" if ci[0] is None else f"[{ci[0]:.4f}, {ci[1]:.4f}]",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--raw", action="append", required=True, help="CONDITION=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_by_condition = {}
    for item in args.raw:
        if "=" not in item:
            parser.error(f"--raw must be CONDITION=PATH, got {item!r}")
        condition, raw_path = item.split("=", 1)
        records = []
        for path in raw_path.split(","):
            records.extend(_read_jsonl(Path(path)))
        raw_by_condition.setdefault(condition, []).extend(records)
    normalized, summary = aggregate_records(
        raw_by_condition,
        _read_jsonl(args.dataset),
        bootstrap_reps=args.bootstrap_reps,
        seed=args.seed,
        model=args.model,
        dataset=args.dataset_name,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "normalized_records.jsonl").open("w", encoding="utf-8") as handle:
        for record in normalized:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_csv(args.output_dir / "summary.csv", summary)
    _write_markdown(args.output_dir / "summary.md", summary)
    print(f"Wrote {len(normalized)} normalized records and {len(summary)} aggregate rows to {args.output_dir}")


if __name__ == "__main__":
    main()

"""Create compact derived plots for the Phase 1 report."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def plot_report(normalized_path: Path, validation_path: Path, output_dir: Path) -> None:
    records = _read_jsonl(normalized_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = {}
    for record in records:
        if record["condition"] == "FP16":
            baseline[(record["sample_id"], record["attack"], record["layer_position"])] = record.get("semantic_cosine_mpnet")

    deltas = defaultdict(list)
    for record in records:
        value = record.get("semantic_cosine_mpnet")
        key = (record["sample_id"], record["attack"], record["layer_position"])
        if value is not None and record["condition"] != "FP16" and key in baseline:
            deltas[record["condition"]].append(float(value) - float(baseline[key]))

    labels = list(deltas)
    means = [sum(deltas[label]) / len(deltas[label]) for label in labels]
    plt.figure(figsize=(8, 4.5))
    plt.axhline(0, color="black", linewidth=0.8)
    plt.bar(labels, means)
    plt.ylabel("semantic_cosine_mpnet delta vs FP16")
    plt.title("Phase 1 paired leakage delta (Protocol A)")
    plt.tight_layout()
    plt.savefig(output_dir / "leakage_delta_vs_fp16.png", dpi=180)
    plt.close()

    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    storage = defaultdict(list)
    distortion = defaultdict(list)
    for row in validation:
        storage[row["condition"]].append(row["compression_ratio"])
        distortion[row["condition"]].append(row["k_mse_mean"] + row["v_mse_mean"])
    labels = list(storage)
    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, [sum(storage[label]) / len(storage[label]) for label in labels])
    plt.ylabel("origin bytes / native bytes")
    plt.title("Native KV storage compression")
    plt.tight_layout()
    plt.savefig(output_dir / "native_storage_compression.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, [sum(distortion[label]) / len(distortion[label]) for label in labels])
    plt.ylabel("mean K MSE + V MSE")
    plt.title("KV reconstruction distortion")
    plt.tight_layout()
    plt.savefig(output_dir / "kv_distortion.png", dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    plot_report(args.normalized, args.validation, args.output_dir)
    print(f"Wrote Phase 1 plots to {args.output_dir}")


if __name__ == "__main__":
    main()

# Phase 1 main experiment launcher

## Goal

Provide one reproducible shell command that prepares the pinned public dataset,
materializes the five Phase 1 cache conditions, runs the main 100-prompt
Protocol-A attack suite, records provenance, measures native storage, and
produces machine-readable aggregate outputs.

## Scope

- Main set: 100 deterministic prompts, 25 short / 50 medium / 25 long,
  tokenizer-capped at 128 tokens.
- Conditions: FP16, KIVI4-STD, KIVI4-FQ, KIVI2-STD, KIVI2-FQ.
- Attacks: inversion, collision, injection through the existing Shadow
  harness. Collision+ remains a separately calibrated, opt-in stage because
  its calibration is expensive and must be frozen before evaluation.
- Outputs: raw attack JSONL, cache correctness/storage JSON, provenance,
  aggregate CSV/JSON/Markdown and a run log.
- No model-weight quantization, fine-tuning, prompt optimization, or deletion
  of existing artifacts.

## Implementation tasks

1. Add a tested dataset preparation helper using a pinned streaming HF source
   and the existing deterministic stratified sampler.
2. Add a tested result aggregator that normalizes legacy Shadow metric names
   (`BERTScore` is reported as MPNet cosine), maps cache hashes to sample IDs,
   and computes per-condition/attack/layer mean, median, standard deviation,
   and paired bootstrap intervals against FP16 when available.
3. Add a tested shell launcher with strict mode, a single-run lock,
   environment-variable configuration, provenance snapshot, stage markers,
   resume support, and explicit attack-risk acknowledgement.
4. Extend the existing prefill/attack entrypoints only where needed for
   arbitrary local model paths and deterministic output paths; preserve their
   default behavior and attack algorithms.
5. Add documentation with the one-line invocation, expected runtime/storage,
   artifact layout, resume/override examples, and the exact claim boundary
   (Protocol A does not establish quantization-as-defense).

## Verification

- Unit tests for dataset preparation/aggregation.
- `bash -n` and launcher `--help`/`--dry-run`.
- Existing full pytest, compileall, and `git diff --check`.
- Dry-run must not download, overwrite, or delete experiment artifacts.

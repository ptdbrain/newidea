# Phase 1 Prefill and Cache Hardening Design

## Context

The Phase 1 runner can report `DONE prefill` and `DONE materialize` even when
one or more samples have no `origin/past_key_values.pt`. The failure is only
discovered later by `validate_phase1_cache.py`. This happens because
`inference/get_kvcache.py` catches per-sample exceptions and continues, while
`defense/baseline/kivi_kvcache.py` treats missing source caches as warnings.

The prefill code also converts every model output through
`DynamicCache.from_legacy_cache`, although supported Transformers versions may
already return a modern cache object. The launcher validates only a shallow
subset of checkpoint files before beginning the expensive run.

## Goal

Make a Phase 1 run fail at the first invalid boundary with the original cause,
while accepting both modern Transformers caches and legacy layer tuples. A
successful prefill stage must prove that every selected dataset record has a
readable canonical Origin cache.

## Non-goals

- Downloading or reconstructing missing model files on an offline node.
- Skipping failed samples or reducing the required 100-sample experiment.
- Weakening cache validation to allow partial scientific results.
- Refactoring unrelated attack, aggregation, or plotting code.

## Design

### Local checkpoint validation

Add a lightweight checkpoint-layout validator used before model loading. The
base model directory must contain `config.json`, at least one supported weight
file, `tokenizer_config.json`, and either `tokenizer.json` or
`tokenizer.model`. The error lists the missing artifact class and the inspected
path.

`AutoTokenizer.from_pretrained` and `AutoModelForCausalLM.from_pretrained`
continue to load the supplied local directory. Offline execution must not
silently fall back to a Hub identifier when that directory is incomplete.

### Cache normalization

Introduce one focused conversion function in `inference/pdsplit.py`:

1. If the returned cache exposes `to_legacy_cache`, call it.
2. Otherwise, treat it as a legacy iterable of layers.
3. Convert each layer to an exact `(key, value)` pair.
4. Require tensors with shape `[batch, heads, tokens, head_dim]`, matching key
   and value shapes, floating-point dtype, and finite values.
5. Move validated tensors to CPU using the model dtype before serialization.

The canonical file remains `origin/past_key_values.pt` and contains a tuple of
layer tuples. No Transformers cache class is serialized into the experiment
artifact.

### Prefill failure semantics

`inference/get_kvcache.py` must not swallow a sample exception. It raises a
contextual error containing the sample index, sample ID when available, and
cache destination, preserving the original exception as the cause.

After processing, the command verifies that the number of generated canonical
Origin cache files equals the number of selected records. Zero or partial
output returns a non-zero exit code, so `run_stage` cannot write
`state/prefill.done`.

### Launcher boundary

After `inference/get_kvcache.py` returns, `stage_prefill` independently compares
the dataset record count with the number of
`*/origin/past_key_values.pt` files below the exact `CACHE_ROOT`. A mismatch
terminates the stage before KIVI materialization.

### KIVI materialization boundary

Add strict completeness behavior to the KIVI batch processor. The Phase 1
launcher enables strict mode, causing any missing source cache to return a
non-zero status. Existing non-Phase-1 callers retain warning-and-skip behavior
unless they opt into strict mode.

## Error handling

Errors must identify the failing boundary and path:

- incomplete checkpoint: missing model/tokenizer artifact and model path;
- incompatible cache: layer index and violated cache contract;
- failed sample: sample index/ID, destination path, and original exception;
- incomplete prefill: expected and actual Origin cache counts;
- incomplete materialization: number and representative paths of missing
  source caches.

No failed stage writes its `.done` marker. Existing partial files remain for
diagnosis and are not deleted automatically.

## Tests

Tests will cover:

- complete and incomplete local checkpoint layouts;
- modern cache objects exposing `to_legacy_cache`;
- legacy tuple caches;
- malformed layer count, tensor rank, shape, dtype, and non-finite values;
- propagation of a per-sample prefill exception;
- rejection of partial Origin cache output;
- strict KIVI materialization failure on a missing source cache;
- launcher contract requiring expected and actual cache counts to match.

The targeted tests must pass without loading a real model or requiring CUDA.
Shell syntax checks and the existing relevant test subset remain part of final
verification. Real GPU execution is still required to prove the server issue
is resolved end to end.

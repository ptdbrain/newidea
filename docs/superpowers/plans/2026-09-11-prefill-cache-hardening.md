# Phase 1 Prefill and Cache Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Phase 1 reject incomplete checkpoints and malformed or missing Origin KV caches at the prefill boundary, with actionable errors and no false `.done` markers.

**Architecture:** Validate the local checkpoint before loading, normalize both modern and legacy Transformers cache representations into one canonical tuple format, and propagate the first sample failure. Add independent completeness gates in the launcher and strict KIVI materialization so every stage verifies the artifact contract it consumes.

**Tech Stack:** Python 3.10, PyTorch 2.6, Transformers 4.51.1, pytest, Bash.

## Global Constraints

- Do not download or reconstruct missing model files on an offline node.
- Do not skip failed samples or reduce the required 100-sample experiment.
- Keep `origin/past_key_values.pt` as a CPU tuple of `(key, value)` tensors.
- Preserve partial artifacts for diagnosis; do not delete them automatically.
- Existing non-Phase-1 KIVI callers keep warning-and-skip behavior unless strict mode is requested.
- Unit tests must not require a real model, CUDA, or network access.

---

### Task 1: Validate the local base-model checkpoint

**Files:**
- Modify: `inference/get_kvcache.py:1-100`
- Test: `tests/test_kvcache_dataset_input.py`

**Interfaces:**
- Produces: `validate_local_checkpoint(model_path: Path) -> None`
- Consumes: a local Transformers checkpoint directory

- [ ] **Step 1: Write failing checkpoint-layout tests**

```python
from get_kvcache import validate_local_checkpoint


def _complete_checkpoint(root: Path) -> Path:
    root.mkdir()
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "model.safetensors").write_bytes(b"weights")
    (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    return root


def test_validate_local_checkpoint_accepts_complete_layout(tmp_path):
    validate_local_checkpoint(_complete_checkpoint(tmp_path / "model"))


def test_validate_local_checkpoint_reports_missing_tokenizer(tmp_path):
    model_path = _complete_checkpoint(tmp_path / "model")
    (model_path / "tokenizer.json").unlink()
    with pytest.raises(FileNotFoundError, match="tokenizer.json or tokenizer.model"):
        validate_local_checkpoint(model_path)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest tests/test_kvcache_dataset_input.py -q`

Expected: collection failure because `validate_local_checkpoint` does not exist.

- [ ] **Step 3: Implement the layout validator and local-only loading**

```python
def validate_local_checkpoint(model_path: Path) -> None:
    if not model_path.is_dir():
        raise FileNotFoundError(f"model checkpoint directory not found: {model_path}")
    missing = []
    if not (model_path / "config.json").is_file():
        missing.append("config.json")
    weight_files = [
        path
        for pattern in ("*.safetensors", "*.bin", "*.pt")
        for path in model_path.glob(pattern)
    ]
    if not weight_files:
        missing.append("model weights (*.safetensors, *.bin, or *.pt)")
    if not (model_path / "tokenizer_config.json").is_file():
        missing.append("tokenizer_config.json")
    if not any((model_path / name).is_file() for name in ("tokenizer.json", "tokenizer.model")):
        missing.append("tokenizer.json or tokenizer.model")
    if missing:
        raise FileNotFoundError(
            f"incomplete local model checkpoint at {model_path}: missing {', '.join(missing)}"
        )
```

Call it before model loading and pass `local_files_only=True` to both
`AutoTokenizer.from_pretrained` and `AutoModelForCausalLM.from_pretrained`.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `pytest tests/test_kvcache_dataset_input.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add inference/get_kvcache.py tests/test_kvcache_dataset_input.py
git commit -m "fix(model): validate offline checkpoint"
```

---

### Task 2: Normalize modern and legacy cache outputs

**Files:**
- Modify: `inference/pdsplit.py:1-75`
- Create: `tests/test_prefill_cache.py`

**Interfaces:**
- Produces: `normalize_past_key_values(cache: Any, *, dtype: torch.dtype) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]`
- Consumes: a modern cache exposing `to_legacy_cache()` or a legacy iterable of `(key, value)` layers

- [ ] **Step 1: Write failing normalization tests**

```python
class ModernCache:
    def __init__(self, layers):
        self.layers = layers

    def to_legacy_cache(self):
        return self.layers


def test_normalize_modern_cache_to_cpu_tuple():
    key = torch.ones(1, 2, 3, 4)
    value = torch.zeros(1, 2, 3, 4)
    result = normalize_past_key_values(ModernCache(((key, value),)), dtype=torch.float16)
    assert result[0][0].device.type == "cpu"
    assert result[0][0].dtype == torch.float16


def test_normalize_rejects_malformed_layer():
    with pytest.raises(ValueError, match="exactly key and value"):
        normalize_past_key_values(((torch.ones(1),),), dtype=torch.float16)


def test_normalize_rejects_non_finite_cache():
    key = torch.full((1, 1, 1, 1), float("nan"))
    value = torch.zeros_like(key)
    with pytest.raises(FloatingPointError, match="non-finite"):
        normalize_past_key_values(((key, value),), dtype=torch.float16)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest tests/test_prefill_cache.py -q`

Expected: collection failure because `normalize_past_key_values` does not exist.

- [ ] **Step 3: Implement cache normalization**

```python
def normalize_past_key_values(cache, *, dtype):
    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()
    layers = tuple(cache)
    if not layers:
        raise ValueError("past_key_values contains no layers")
    normalized = []
    for layer_index, layer in enumerate(layers):
        if len(layer) != 2:
            raise ValueError(f"cache layer {layer_index} must contain exactly key and value")
        key, value = layer
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise TypeError(f"cache layer {layer_index} must contain tensors")
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError(f"cache layer {layer_index} must have shape [B, H, T, D]")
        if key.shape != value.shape:
            raise ValueError(f"cache layer {layer_index} key/value shapes differ")
        if not key.dtype.is_floating_point or not value.dtype.is_floating_point:
            raise TypeError(f"cache layer {layer_index} must be floating point")
        if not torch.isfinite(key).all() or not torch.isfinite(value).all():
            raise FloatingPointError(f"cache layer {layer_index} contains non-finite values")
        normalized.append((key.detach().to(device="cpu", dtype=dtype), value.detach().to(device="cpu", dtype=dtype)))
    return tuple(normalized)
```

Replace `DynamicCache.from_legacy_cache(outputs.past_key_values)` in `prefill`
with this function. Leave decode conversion outside this task.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `pytest tests/test_prefill_cache.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add inference/pdsplit.py tests/test_prefill_cache.py
git commit -m "fix(cache): normalize prefill output"
```

---

### Task 3: Propagate sample failures and enforce completeness

**Files:**
- Modify: `inference/get_kvcache.py:93-150`
- Test: `tests/test_kvcache_dataset_input.py`

**Interfaces:**
- Produces: `process_dataset_records(model, tokenizer, dataset, dataset_path: Path, cache_root: Path, *, minimal_cache: bool, prefill_fn=prefill) -> int`
- Consumes: normalized dataset rows and the Task 2 `prefill` function

- [ ] **Step 1: Write failing fail-fast tests**

```python
def test_process_dataset_records_propagates_sample_failure(tmp_path):
    dataset = [{"sample_id": "prompt-1", "prompt": "hello"}]

    def failing_prefill(*args, **kwargs):
        raise ValueError("bad cache format")

    with pytest.raises(RuntimeError, match="prompt-1.*bad cache format"):
        process_dataset_records(
            object(), object(), dataset, Path("phase1.jsonl"), tmp_path,
            minimal_cache=True, prefill_fn=failing_prefill,
        )


def test_process_dataset_records_rejects_missing_prompt(tmp_path):
    with pytest.raises(RuntimeError, match="no usable prompt"):
        process_dataset_records(
            object(), object(), [{"sample_id": "prompt-1"}],
            Path("phase1.jsonl"), tmp_path, minimal_cache=True,
        )
```

- [ ] **Step 2: Run tests and verify RED**

Run: `pytest tests/test_kvcache_dataset_input.py -q`

Expected: import failure because `process_dataset_records` does not exist.

- [ ] **Step 3: Extract the record loop and fail on the first error**

```python
def process_dataset_records(
    model, tokenizer, dataset, dataset_path, cache_root, *, minimal_cache, prefill_fn=prefill
):
    generated = 0
    for index, item in enumerate(tqdm(dataset, desc=f"Processing {dataset_path.name}")):
        sample_id = str(item.get("sample_id", f"index-{index}"))
        user_input = extract_user_input(item, dataset_path.name)
        if not user_input:
            raise RuntimeError(f"prefill failed for {sample_id}: no usable prompt")
        input_hash = hashlib.sha1(user_input.encode("utf-8")).hexdigest()
        cache_dir = cache_root / input_hash
        try:
            prefill_fn(
                model, tokenizer, user_input, cache_dir,
                save_intermediates=not minimal_cache,
            )
        except Exception as error:
            raise RuntimeError(
                f"prefill failed for sample {sample_id} at {cache_dir}: {error}"
            ) from error
        if not (cache_dir / "origin" / "past_key_values.pt").is_file():
            raise RuntimeError(
                f"prefill produced no canonical cache for sample {sample_id}: {cache_dir}"
            )
        generated += 1
    if generated != len(dataset):
        raise RuntimeError(f"prefill incomplete: expected {len(dataset)}, generated {generated}")
    return generated
```

Use the function from `main` and remove the existing catch-and-continue loop.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `pytest tests/test_kvcache_dataset_input.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add inference/get_kvcache.py tests/test_kvcache_dataset_input.py
git commit -m "fix(prefill): fail on missing caches"
```

---

### Task 4: Add strict KIVI source-cache handling

**Files:**
- Modify: `defense/baseline/kivi_kvcache.py:84-183`
- Modify: `tests/test_kivi_batch_processor.py`

**Interfaces:**
- Produces: `process_cache_directory(..., strict: bool = False) -> list[Path]`
- Produces: CLI flag `--strict`
- Consumes: canonical Origin cache directories produced by Task 3

- [ ] **Step 1: Write a failing strict-mode test**

```python
def test_batch_processor_strict_mode_rejects_missing_source(tmp_path):
    cache_root = tmp_path / "cache"
    (cache_root / "sample-a").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="sample-a.*past_key_values.pt"):
        kivi_kvcache.process_cache_directory(
            cache_root,
            KIVIConfig(2, 2, 32, 32),
            device="cpu",
            dtype=torch.float16,
            strict=True,
        )
```

- [ ] **Step 2: Run test and verify RED**

Run: `pytest tests/test_kivi_batch_processor.py::test_batch_processor_strict_mode_rejects_missing_source -q`

Expected: failure because `strict` is not accepted.

- [ ] **Step 3: Implement strict mode without changing the default**

Add `strict: bool = False` to `process_cache_directory`. When a source file is
missing and `strict` is true, raise:

```python
raise FileNotFoundError(
    f"missing source cache for {sample_dir.name}: {source_path}"
)
```

Add `parser.add_argument("--strict", action="store_true")` and pass
`strict=args.strict` to the function.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `pytest tests/test_kivi_batch_processor.py -q`

Expected on a Linux Torch/KIVI environment: CPU tests pass; the real-KIVI
CUDA test skips when CUDA is absent. On Windows without Triton, run the strict
source-cache contract test separately from KIVI kernel import tests and report
the missing-Triton collection boundary.

- [ ] **Step 5: Commit**

```bash
git add defense/baseline/kivi_kvcache.py tests/test_kivi_batch_processor.py
git commit -m "fix(kivi): reject missing source caches"
```

---

### Task 5: Enforce the Phase 1 launcher artifact contract

**Files:**
- Modify: `scripts/phase1_kivi_runtime.sh`
- Modify: `scripts/run_phase1_main.sh:334-349`
- Modify: `tests/test_phase1_kivi_runtime.py`

**Interfaces:**
- Produces: `phase1_verify_origin_caches(dataset_path, cache_root)` Bash function
- Consumes: Task 3 canonical cache layout
- Consumes: Task 4 `--strict` CLI flag
- Produces: a prefill stage that succeeds only when expected and actual Origin counts match

- [ ] **Step 1: Write failing cache-contract behavior tests**

```python
def test_origin_cache_verifier_rejects_partial_prefill(tmp_path):
    dataset_path = tmp_path / "phase1.jsonl"
    dataset_path.write_text('{"prompt":"one"}\n{"prompt":"two"}\n')
    origin = tmp_path / "cache" / "sample-one" / "origin"
    origin.mkdir(parents=True)
    (origin / "past_key_values.pt").write_bytes(b"cache")

    result = run_cache_verifier(dataset_path, tmp_path / "cache")

    assert result.returncode != 0
    assert "expected 2 origin caches, found 1" in result.stderr
```

- [ ] **Step 2: Run test and verify RED**

Run: `pytest tests/test_phase1_kivi_runtime.py::test_launcher_requires_complete_origin_cache_set -q`

Expected: failure because `phase1_verify_origin_caches` does not exist.

- [ ] **Step 3: Add the launcher count gate and strict flags**

Implement `phase1_verify_origin_caches` with Bash built-ins and a nullglob array
over `"$cache_root"/*/origin/past_key_values.pt`. After
`inference/get_kvcache.py` returns in `stage_prefill`, call:

```bash
phase1_verify_origin_caches "$DATASET_PATH" "$CACHE_ROOT"
```

Append `--strict` to each of the four KIVI materialization commands.

- [ ] **Step 4: Run tests and shell syntax verification**

Run: `pytest tests/test_phase1_kivi_runtime.py -q`

Run: `bash -n scripts/run_phase1_main.sh`

Expected: all tests pass and Bash exits zero.

- [ ] **Step 5: Commit**

```bash
git add scripts/phase1_kivi_runtime.sh scripts/run_phase1_main.sh tests/test_phase1_kivi_runtime.py
git commit -m "fix(runner): gate incomplete prefill"
```

---

### Task 6: Verify the complete change and document server behavior

**Files:**
- Modify: `docs/phase1_offline_jobs.md`

**Interfaces:**
- Consumes: all prior task behavior
- Produces: operator guidance for a new run after a failed prefill

- [ ] **Step 1: Document the new failure boundaries**

Add the following operational guarantees:

```markdown
- Checkpoint validation fails before model loading when local model or tokenizer assets are incomplete.
- A sample prefill error stops the stage and preserves the original exception.
- `prefill.done` is written only after every dataset row has an Origin cache.
- Retry a corrected run with a new `RUN_ID`; do not resume a run whose prefill marker came from an older launcher.
```

- [ ] **Step 2: Run the targeted test suite**

Run:

```bash
pytest \
  tests/test_kvcache_dataset_input.py \
  tests/test_prefill_cache.py \
  tests/test_kivi_batch_processor.py \
  tests/test_phase1_kivi_runtime.py -q
```

Expected on a Linux Torch/KIVI environment: all CPU tests pass; CUDA-only tests
skip when CUDA is absent. On Windows without Triton, the checkpoint, prefill,
and launcher tests must pass; report KIVI test collection as an environment
boundary rather than a passing result.

- [ ] **Step 3: Run static verification**

```bash
python -m compileall inference/get_kvcache.py inference/pdsplit.py defense/baseline/kivi_kvcache.py
bash -n scripts/run_phase1_main.sh
git diff --check
```

Expected: every command exits zero.

- [ ] **Step 4: Record the external validation boundary**

Document in the handoff that local tests prove fail-fast and compatibility
logic only. A fresh server run with the actual Llama checkpoint and GPU is
required before claiming the original job failure is resolved end to end.

- [ ] **Step 5: Commit**

```bash
git add docs/phase1_offline_jobs.md
git commit -m "docs: explain prefill failure gates"
```

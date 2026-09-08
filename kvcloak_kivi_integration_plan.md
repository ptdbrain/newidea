# KV-Cloak + KIVI Integration Implementation Plan

> **For agentic workers:** Implement task-by-task, keep upstream Shadow/KV-Cloak and KIVI logic unchanged wherever possible. The project contribution should remain limited to integration, adapters, experiment orchestration, and reporting.

## Goal

Use `SiO-2/kvcloak` as the main baseline and evaluation harness, reuse the public Shadow/KV-Cloak implementation for KV extraction, privacy attacks, and evaluators, reuse the public `jy-yuan/KIVI` implementation for KV-cache quantization, and implement only the minimal glue/adapter code required to connect them.

## Architecture

The integration should preserve a strict separation of responsibilities:

- **Shadow / KV-Cloak**: KV extraction, baseline KV representation, inversion attack, collision attack, injection attack, privacy metrics, utility evaluators.
- **KIVI**: public quantization and dequantization primitives, packed KV representation, quantization parameters.
- **Our code**: adapter between Shadow's canonical `(K, V)` cache representation and KIVI's packed representation; experiment orchestration; storage and latency accounting; minimal evaluator hooks.

The central rule is:

> Do not reimplement Shadow attacks or KIVI quantization. Only write glue code.

## Tech Stack

- Python 3.10+
- PyTorch
- Hugging Face Transformers
- Triton / CUDA from KIVI
- Shadow / KV-Cloak public repository
- KIVI public repository
- pytest

## Upstream Repositories

- Shadow / KV-Cloak: `https://github.com/SiO-2/kvcloak`
- KIVI: `https://github.com/jy-yuan/KIVI`

Recommended pinned revisions used when preparing this plan:

```text
KV-Cloak:
6b40f36edb2f337557543e7e60b10022308883d4

KIVI:
876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6
```

---

# 1. Global Constraints

1. `inference/get_kvcache.py` remains the canonical KV extraction implementation.
2. `inference/pdsplit.py` remains unchanged unless an upstream compatibility fix is absolutely required.
3. `attack/inversion.py` remains unchanged.
4. `attack/collision.py` remains unchanged.
5. `attack/injection.py` remains unchanged.
6. `attack/attacks.py` should ideally remain unchanged.
7. KIVI quantization formulas must not be copied into our code.
8. KIVI dequantization formulas must not be independently reimplemented.
9. Our code should call the public KIVI functions directly.
10. All experiments must use the same source Shadow KV cache for fair comparison.
11. Privacy attacks must operate on the representation realistically available to a white-box attacker.
12. Native KIVI storage measurements must use packed KIVI tensors, not reconstructed FP tensors.
13. Utility comparisons should use a consistent dtype, preferably FP16 for the initial experiments.
14. Every generated artifact must contain provenance metadata including upstream commit hashes and KIVI configuration.

---

# 2. Experimental Conditions

The framework should support at least four cache conditions:

| Condition | Description |
|---|---|
| `origin` | Original Shadow FP16 KV cache |
| `kivi_k2_v2_g32_r32` | Original KV cache compressed with KIVI |
| `kvcloak` | KV-Cloak protected cache |
| `kvcloak_kivi_k2_v2_g32_r32` | KV-Cloak protected cache followed by KIVI compression |

This gives three primary research questions:

1. Does KIVI by itself reduce information leakage from the KV cache?
2. How much privacy improvement does KV-Cloak provide relative to Origin and KIVI?
3. Can KV-Cloak and KIVI be composed while preserving utility and memory efficiency?

High-level pipeline:

```text
                        SHADOW
                           |
                    original FP16 KV
                           |
          +----------------+----------------+
          |                |                |
          v                v                v
       Origin          KV-Cloak           KIVI
                           |                |
                           |          KIVI quantizer
                           |                |
                           v                v
                      Protected KV      Quantized KV
                           |
                           v
                     KIVI quantizer
                           |
                           v
                 KV-Cloak + KIVI
```

Attack-side pipeline for KIVI:

```text
KIVI native representation
        |
        | public KIVI dequantizer
        v
approximate FP16 K,V
        |
        v
Shadow inversion / collision / injection
```

Do **not** modify Shadow attacks so that they operate directly on packed `int32` KIVI values. A realistic white-box attacker knows the KIVI algorithm and its public metadata and can reconstruct the approximate K/V representation first.

---

# 3. Repository Setup and Upstream Pinning

## Task 1: Pin Shadow and KIVI

### Files

- Modify: `.gitmodules`
- Create: `third_party/KIVI/` as git submodule
- Create: `docs/upstream_versions.md`

### Steps

- [ ] Clone and pin KV-Cloak.

```bash
git clone https://github.com/SiO-2/kvcloak.git
cd kvcloak

git checkout 6b40f36edb2f337557543e7e60b10022308883d4
git checkout -b feat/kivi-adapter
```

- [ ] Add KIVI as a submodule.

```bash
mkdir -p third_party

git submodule add \
  https://github.com/jy-yuan/KIVI.git \
  third_party/KIVI

git -C third_party/KIVI checkout \
  876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6
```

- [ ] Record revisions in `docs/upstream_versions.md`.

```markdown
# Upstream Versions

- KV-Cloak: `6b40f36edb2f337557543e7e60b10022308883d4`
- KIVI: `876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6`
```

- [ ] Commit.

```bash
git add .gitmodules third_party/KIVI docs/upstream_versions.md
git commit -m "build: pin KIVI upstream dependency"
```

### Acceptance Criteria

- `git submodule status` reports the expected KIVI revision.
- No KIVI source files are copied into the main repository.

---

# 4. Environment Compatibility Strategy

KV-Cloak and KIVI currently use different dependency stacks. Do not blindly install both full requirements files into one environment.

Recommended strategy:

```bash
pip install -r requirements.txt
export PYTHONPATH=$PWD/third_party/KIVI:$PYTHONPATH
```

Do not start with:

```bash
pip install -r third_party/KIVI/requirements.txt
```

because KIVI's pinned Torch/Transformers versions may conflict with the newer versions used by KV-Cloak.

The first integration gate is therefore a compatibility smoke test for the actual KIVI quantization primitives.

---

# 5. KIVI Compatibility Smoke Test

## Task 2: Verify Public KIVI Quantization Functions Can Be Imported and Executed

### Files

- Create: `tests/test_kivi_import.py`

### Interface

The test must import the public KIVI functions directly:

```python
from quant.new_pack import (
    triton_quantize_and_pack_along_last_dim,
    unpack_and_dequant_vcache,
)
```

### Test Shape

```text
B = 1
H = 4
T = 32
D = 64
bits = 2
group_size = 32
```

### Steps

- [ ] Write a CUDA test that creates a small FP16 tensor.
- [ ] Call KIVI's public quantizer.
- [ ] Call KIVI's public dequantizer.
- [ ] Verify reconstructed shape.
- [ ] Verify all values are finite.

Example assertions:

```python
assert reconstructed.shape == source.shape
assert torch.isfinite(reconstructed).all()
```

Run:

```bash
pytest tests/test_kivi_import.py -v
```

### Acceptance Criteria

This is a **GO / NO-GO gate**.

If the public KIVI Triton implementation does not work with the current Torch/CUDA environment, resolve compatibility first. Do not replace KIVI with a newly written quantizer.

---

# 6. Canonical Adapter Design

## Task 3: Implement `KIVIConfig` and KIVI Adapter Boundary

### Files

- Create: `src/kivi_adapter.py`
- Create: `tests/test_kivi_adapter.py`

### Interfaces

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class KIVIConfig:
    k_bits: int
    v_bits: int
    group_size: int
    residual_length: int
```

Required functions:

```python
def quantize_cache(
    past_key_values,
    config: KIVIConfig,
):
    ...
```

```python
def dequantize_cache(
    quantized_cache,
    config: KIVIConfig,
):
    ...
```

```python
def roundtrip_cache(
    past_key_values,
    config: KIVIConfig,
):
    quantized = quantize_cache(past_key_values, config)
    return dequantize_cache(quantized, config)
```

### Responsibilities

`src/kivi_adapter.py` may:

- split full-precision and quantized regions;
- transpose tensors to the layouts expected by KIVI;
- call public KIVI quantization functions;
- collect codes, scales, minima, and residual tensors;
- call public KIVI dequantization functions;
- reconstruct Shadow-compatible `(K, V)` tuples;
- move tensors between CPU/GPU when needed;
- serialize/deserialise adapter metadata.

It must **not**:

- derive quantization scales itself;
- implement integer packing itself;
- implement unpacking formulas itself;
- independently implement KIVI's quantization algorithm.

---

# 7. Native KIVI Representation

Each layer should be represented by a structured payload similar to:

```python
{
    "key_code": ...,
    "key_scale": ...,
    "key_min": ...,
    "key_residual": ...,

    "value_code": ...,
    "value_scale": ...,
    "value_min": ...,
    "value_residual": ...,

    "seq_len": ...,
}
```

The exact field names can be slightly adjusted if required by implementation, but they must remain stable after the first adapter task.

Do not save only reconstructed FP cache. Both forms are required:

```text
native KIVI representation
```

and

```text
dequantized Shadow attack view
```

The native representation is required for real compression measurements.

---

# 8. KIVI Cache Partitioning

## Task 4: Implement Key and Value Partition Glue

### KIVI Default Reproduction Configuration

Start with:

```text
k_bits = 2
v_bits = 2
group_size = 32
residual_length = 32
```

Only after the basic pipeline is validated should additional parameter sweeps be introduced.

## Key Cache

KIVI quantizes Key cache per-channel. The quantizable prefix should therefore be transposed before calling the public quantizer.

Conceptual pipeline:

```text
original K
    |
    +-- quantizable prefix
    |       |
    |       v
    |  transpose(2, 3)
    |       |
    |       v
    |  KIVI public quantizer
    |
    +-- recent full-precision residual
```

Expected native components:

```text
key_code
key_scale
key_min
key_residual
```

## Value Cache

KIVI quantizes Value cache per-token.

Conceptual pipeline:

```text
original V
    |
    +-- old-token prefix
    |       |
    |       v
    |  KIVI public quantizer
    |
    +-- recent residual_length tokens in FP16
```

Expected native components:

```text
value_code
value_scale
value_min
value_residual
```

### Important Edge Cases

Tests must cover:

- sequence shorter than `residual_length`;
- sequence exactly equal to `residual_length`;
- sequence not divisible by `group_size`;
- sequence exactly divisible by `group_size`;
- long sequence with both quantized and full-precision regions.

Example case:

```text
T = 130
residual_length = 32
```

Expected conceptual behavior:

```text
Key:
quantized prefix = 128
FP tail = 2

Value:
quantized prefix = 98
FP tail = 32
```

The Key and Value partition rules are intentionally not identical.

---

# 9. KIVI Dequantization Adapter for Shadow Attacks

## Task 5: Reconstruct Shadow-Compatible K/V Using Public KIVI Functions

### Files

- Modify: `src/kivi_adapter.py`
- Test: `tests/test_kivi_adapter.py`

For Key, the adapter should call KIVI's public unpack/dequant function on the transposed packed representation, transpose the result back, then concatenate the full-precision residual.

Conceptual code:

```python
dequant_k_transposed = unpack_and_dequant_vcache(
    key_code,
    key_scale.unsqueeze(-1),
    key_min.unsqueeze(-1),
    config.group_size,
    config.k_bits,
)

dequant_k = dequant_k_transposed.transpose(2, 3)

K = torch.cat(
    [dequant_k, key_residual],
    dim=2,
)
```

For Value:

```python
dequant_v = unpack_and_dequant_vcache(
    value_code,
    value_scale.unsqueeze(-1),
    value_min.unsqueeze(-1),
    config.group_size,
    config.v_bits,
)

V = torch.cat(
    [dequant_v, value_residual],
    dim=2,
)
```

The exact tensor shapes must be validated against the pinned KIVI implementation.

### Acceptance Criteria

For every layer:

```python
assert reconstructed_k.shape == original_k.shape
assert reconstructed_v.shape == original_v.shape
assert torch.isfinite(reconstructed_k).all()
assert torch.isfinite(reconstructed_v).all()
```

---

# 10. Dtype Policy

Use FP16 for the initial controlled comparison.

Recommended experimental conditions:

```text
Origin FP16
KIVI FP16
KV-Cloak FP16
KV-Cloak + KIVI FP16
```

Avoid comparing:

```text
Origin FP32
vs
KIVI reconstructed FP16
```

because this mixes quantization error with dtype conversion error.

Run Shadow extraction initially with:

```bash
--dtype float16
```

---

# 11. Reuse Shadow KV Extraction Unchanged

## Task 6: Generate Canonical Source Cache

Do not modify:

```text
inference/get_kvcache.py
inference/pdsplit.py
```

Run:

```bash
python inference/get_kvcache.py \
  --model-name Llama-3.2-1B \
  --dataset ./dataset/lmsys-chat-1m_1k.jsonl \
  --dtype float16 \
  --device cuda:0 \
  --max-samples 20
```

Expected canonical structure:

```text
cache/float16/lmsys-chat-1m_1k/Llama-3.2-1B/<input_hash>/
├── decode.json
└── origin/
    └── past_key_values.pt
```

All derived cache conditions must originate from this same `origin/past_key_values.pt`.

---

# 12. Batch KIVI Cache Processor

## Task 7: Add KIVI Baseline Processor

### Files

- Create: `defense/baseline/kivi_kvcache.py`
- Test: `tests/test_kivi_batch_processor.py`

### CLI

```bash
python defense/baseline/kivi_kvcache.py \
  --model-name Llama-3.2-1B \
  --dataset-path ./dataset/lmsys-chat-1m_1k.jsonl \
  --source-protect-type origin \
  --k-bits 2 \
  --v-bits 2 \
  --group-size 32 \
  --residual-length 32 \
  --dtype float16 \
  --device cuda:0
```

### Behavior

For each sample:

1. Load:

```text
<sample>/origin/past_key_values.pt
```

2. Quantize through `src/kivi_adapter.py`.
3. Save native representation.
4. Dequantize native representation for Shadow attack view.
5. Save provenance metadata.

Expected output:

```text
<sample>/
└── kivi_k2_v2_g32_r32/
    ├── native.pt
    ├── past_key_values.pt
    └── metadata.json
```

### `native.pt`

Contains the actual compressed KIVI representation.

### `past_key_values.pt`

Contains the white-box attacker reconstruction:

```text
dequantize(native KIVI representation)
```

This file exists only to make the existing Shadow attack harness consume KIVI without modifying attack implementations.

### `metadata.json`

Example:

```json
{
  "method": "kivi",
  "k_bits": 2,
  "v_bits": 2,
  "group_size": 32,
  "residual_length": 32,
  "source": "origin",
  "kivi_commit": "876b4d2d08e3b1d5f70d0969c299d8c7c42ddfb6",
  "kvcloak_commit": "6b40f36edb2f337557543e7e60b10022308883d4",
  "attack_view": "dequantized_from_native"
}
```

---

# 13. Reuse Shadow Attacks Unchanged

## Task 8: Run Privacy Attacks on KIVI

Do not modify:

```text
attack/inversion.py
attack/collision.py
attack/injection.py
```

Prefer not to modify:

```text
attack/attacks.py
```

Because Shadow already loads:

```text
<sample>/<protect_type>/past_key_values.pt
```

KIVI can be introduced simply as a new `protect_type`.

Run:

```bash
python attack/attacks.py \
  --target-model-name Llama-3.2-1B \
  --dataset-path ./dataset/lmsys-chat-1m_1k.jsonl \
  --protect-type kivi_k2_v2_g32_r32 \
  --dtype float16 \
  --device cuda:0 \
  --run-inversion \
  --run-collision \
  --run-injection \
  --i-understand-risks
```

### Acceptance Criteria

- All three attacks execute without attack-specific modifications.
- Shadow result JSONL is generated successfully.
- Result records retain `protect_type = kivi_k2_v2_g32_r32`.

---

# 14. Collision+ Calibration

## Task 9: Generate Protection-Specific Collision+ Thresholds

Do not reuse Origin calibration for KIVI.

Generate separate configurations for:

```text
origin
kivi
kvcloak
kvcloak+kivi
```

Example:

```bash
python attack/get_collision_threshold.py \
  --model_path ~/model/Llama-3.2-1B \
  --target_data_path cache/.../<hash>/kivi_k2_v2_g32_r32/past_key_values.pt \
  --protect_type kivi_k2_v2_g32_r32 \
  --target_model_name Llama-3.2-1B \
  --dtype float16 \
  --device cuda:0
```

Then:

```bash
python attack/attacks.py \
  ... \
  --protect-type kivi_k2_v2_g32_r32 \
  --run-collision \
  --enhance \
  --i-understand-risks
```

### Acceptance Criteria

A protection-specific configuration exists under:

```text
attack/config/kivi_k2_v2_g32_r32/float16/Llama-3.2-1B.json
```

---

# 15. KV-Cloak + KIVI Composition

## Task 10: Compose Protection and Compression

Recommended order:

```text
Original KV
    |
    v
KV-Cloak obfuscate
    |
    v
KIVI quantize
    |
    v
Stored Native Cache
```

Inference-side reconstruction:

```text
Stored Native KIVI
    |
    v
KIVI dequantize
    |
    v
KV-Cloak deobfuscate
    |
    v
Model decode
```

Do **not** use:

```text
KIVI -> dequantize -> KV-Cloak -> store FP16
```

because this removes KIVI's storage advantage.

### Offline Privacy Attack View

For privacy attacks, generate:

```text
dequantized(KIVI(KV-Cloak(KV)))
```

and store it as:

```text
<sample>/kvcloak_kivi_k2_v2_g32_r32/past_key_values.pt
```

The attacker should see the obfuscated but KIVI-reconstructed K/V tensors.

### Native Output

```text
<sample>/kvcloak_kivi_k2_v2_g32_r32/
├── native.pt
├── past_key_values.pt
└── metadata.json
```

---

# 16. Generic Cache Pipeline for Utility Evaluation

## Task 11: Add Minimal Cache Pipeline Abstraction

### Files

- Create: `src/cache_pipeline.py`
- Create: `tests/test_cache_pipeline.py`

### Interface

```python
class CachePipeline:
    def roundtrip(self, past_key_values):
        raise NotImplementedError
```

Implementations:

```text
IdentityPipeline
KIVIPipeline
KVCloakKIVIPipeline
```

Optionally a small adapter can wrap the existing KV-Cloak flow rather than introducing another full abstraction.

### Behavior

```text
IdentityPipeline:
    KV -> KV

KIVIPipeline:
    KV -> KIVI quantize -> KIVI dequantize

KVCloakKIVIPipeline:
    KV
      -> KV-Cloak obfuscate
      -> KIVI quantize
      -> KIVI dequantize
      -> KV-Cloak deobfuscate
```

Keep this abstraction small. Do not redesign the entire Shadow defense framework.

---

# 17. Shadow Utility Evaluator Integration

## Task 12: Add Tiny Adapter Hooks to MMLU and SQuAD

### Files

- Modify: `defense/eval/mmlu_eval.py`
- Modify: `defense/eval/squad_eval.py`
- Test: integration tests where feasible

Do not change:

```text
prompt construction
dataset loading
metric implementation
prediction scoring
evaluation split
```

Add only a generic optional cache hook.

Example:

```python
class MMLUEvaluator:
    def __init__(
        self,
        ...,
        cache_adapter=None,
    ):
        ...
        self.cache_adapter = cache_adapter
```

Then immediately after obtaining `past_key_values`:

```python
if self.cache_adapter is not None:
    past_key_values = self.cache_adapter.roundtrip(
        past_key_values
    )
```

The mode mapping becomes:

```text
origin:
    identity

kivi:
    quantize -> dequantize

kvcloak:
    existing Shadow flow

kvcloak_kivi:
    kvcloak.obfuscate
    -> kivi.quantize
    -> kivi.dequantize
    -> kvcloak.deobfuscate
```

### Acceptance Criteria

- Original MMLU origin mode still produces identical behavior.
- Existing KV-Cloak mode still works.
- KIVI can be injected without changing evaluation metric logic.
- KV-Cloak + KIVI can be evaluated with the same dataset and scoring path.

---

# 18. Unit Tests

## Task 13: Adapter and Partition Tests

Required tests:

```text
test_kivi_quantized_cache_preserves_layer_count
test_kivi_roundtrip_preserves_shape
test_kivi_roundtrip_returns_finite_values
```

Partition tests:

```text
test_key_residual_partition
test_value_residual_partition
test_sequence_shorter_than_residual
test_exact_residual_multiple
```

Quantization quality sanity check:

```text
MSE(KIVI-4bit) <= MSE(KIVI-2bit)
```

Do not hard-code an overly strict absolute MSE threshold unless verified empirically across supported hardware.

---

# 19. Provenance Test

## Task 14: Verify the Adapter Actually Calls Public KIVI Functions

Use monkeypatch or mocking around:

```python
triton_quantize_and_pack_along_last_dim
unpack_and_dequant_vcache
```

Assert that:

- `quantize_cache()` calls KIVI's quantization function;
- `dequantize_cache()` calls KIVI's public reconstruction function.

This test prevents accidental drift toward a locally reimplemented quantizer.

---

# 20. Real Shadow Cache Smoke Test

## Task 15: Run Adapter Against One Real KV Cache

Use one actual file generated by Shadow:

```text
origin/past_key_values.pt
```

Pipeline:

```text
origin
-> KIVI quantize
-> native representation
-> KIVI dequantize
-> reconstructed Shadow cache
```

Verify for every layer:

```python
assert K_original.shape == K_reconstructed.shape
assert V_original.shape == V_reconstructed.shape
assert torch.isfinite(K_reconstructed).all()
assert torch.isfinite(V_reconstructed).all()
```

Then run all Shadow attacks on exactly one sample.

Acceptance criteria:

```text
Inversion completes
Collision completes
Injection completes
JSONL result is produced
```

At this stage, attack score quality is not the acceptance criterion. The objective is compatibility.

---

# 21. Native Storage Metrics

## Task 16: Measure Real KIVI Cache Compression

### Files

- Create: `defense/eval/kivi_storage_eval.py`
- Create: `tests/test_kivi_storage_eval.py`

Do not use reconstructed `past_key_values.pt` to claim compression.

Measure the native KIVI representation only.

For every tensor:

```python
bytes_used = tensor.numel() * tensor.element_size()
```

Include:

```text
packed key codes
key scales
key minima
key FP residual
packed value codes
value scales
value minima
value FP residual
```

Report:

```text
Origin KV bytes
KIVI native bytes
KV-Cloak native bytes if relevant
KV-Cloak + KIVI native bytes
```

Compression ratio:

```text
compression_ratio = origin_bytes / compressed_bytes
```

---

# 22. Quantize / Dequantize Latency

## Task 17: Add Adapter Micro-Benchmark

### Files

- Create: `defense/eval/kivi_adapter_benchmark.py`

Measure separately:

```text
T_quant
T_dequant
T_roundtrip
```

Use CUDA events:

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
```

Always:

1. warm up;
2. synchronize before measurement;
3. run multiple trials;
4. report median and mean;
5. report tensor shape and KIVI configuration.

Important reporting limitation:

> Offline adapter latency is not equivalent to native KIVI end-to-end model throughput.

Do not claim KIVI throughput improvement from this adapter benchmark alone.

If true end-to-end throughput is needed later, run the public `LlamaForCausalLM_KIVI` model path separately as an additional experiment.

---

# 23. Smoke Experiment Matrix

## Phase A: Integration Smoke

Model:

```text
Llama-3.2-1B
```

Dataset:

```text
LMSYS, first 20 samples
```

Dtype:

```text
float16
```

Seed:

```text
42
```

Conditions:

```text
Origin
KIVI-2bit
KV-Cloak
KV-Cloak + KIVI-2bit
```

KIVI config:

```text
k_bits = 2
v_bits = 2
group_size = 32
residual_length = 32
```

Run all privacy attacks:

```text
Inversion
Collision
Injection
```

Then run utility smoke tests on small MMLU/SQuAD subsets before full evaluation.

---

# 24. Quantization Ablation

## Phase B

Conditions:

```text
Origin

KIVI 2/2 bit
KIVI 4/4 bit

KV-Cloak

KV-Cloak + KIVI 2/2 bit
KV-Cloak + KIVI 4/4 bit
```

Hold constant initially:

```text
group_size = 32
residual_length = 32
```

Do not start with a large grid search over all KIVI parameters. First establish the main privacy/utility/compression effect.

---

# 25. Optional Residual-Length Ablation

Only after Phase B is stable, consider:

```text
residual_length = 16
residual_length = 32
residual_length = 64
residual_length = 128
```

Research question:

> Does preserving more recent FP16 tokens improve utility at the cost of privacy leakage and storage efficiency?

This is potentially an important result because recent unquantized tokens may preserve particularly attack-relevant information.

---

# 26. Optional Group-Size Ablation

After residual-length analysis:

```text
group_size = 16
group_size = 32
group_size = 64
```

Measure:

```text
attack leakage
utility
compression
quant/dequant latency
```

Avoid simultaneous large sweeps over bits, residual length, and group size unless compute budget allows it.

---

# 27. Result Tables

## Main Table

| Method | KV bits | Compression Ratio ↑ | MMLU ↑ | SQuAD ↑ | Inversion ↓ | Collision ↓ | Injection ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Origin | FP16 | 1.00x | | | | | |
| KIVI | 2/2 | | | | | | |
| KIVI | 4/4 | | | | | | |
| KV-Cloak | FP16 | | | | | | |
| KV-Cloak + KIVI | 2/2 | | | | | | |
| KV-Cloak + KIVI | 4/4 | | | | | | |

## Layer-Specific Privacy Table

Retain Shadow's attack convention:

```text
First layer
Middle layer
Last layer
```

Example:

| Method | Attack | First | Middle | Last |
|---|---|---:|---:|---:|
| Origin | Inversion | | | |
| KIVI-2 | Inversion | | | |
| KV-Cloak | Inversion | | | |
| KV-Cloak + KIVI-2 | Inversion | | | |

---

# 28. Metric Reporting Note

When reporting Shadow's attack evaluator results, verify the exact implementation used by the pinned repository.

If the code labels a metric as `BERTScore` but computes sentence-embedding cosine similarity internally, describe it precisely in the paper/report rather than presenting it as canonical BERTScore.

Recommended wording:

> Semantic similarity, reported as “BERTScore” by the public Shadow implementation.

Do not silently change the evaluator if the goal is an apples-to-apples reproduction of the public baseline.

---

# 29. Recommended Final Repository Structure

```text
kvcloak/
|
├── attack/
│   ├── attacks.py                 # UNCHANGED if possible
│   ├── inversion.py               # UNCHANGED
│   ├── collision.py               # UNCHANGED
│   └── injection.py               # UNCHANGED
│
├── inference/
│   ├── get_kvcache.py             # UNCHANGED
│   └── pdsplit.py                 # UNCHANGED
│
├── defense/
│   ├── core/
│   │   └── kvcloak.py             # UNCHANGED
│   │
│   ├── baseline/
│   │   └── kivi_kvcache.py        # NEW GLUE
│   │
│   └── eval/
│       ├── mmlu_eval.py            # MINIMAL ADAPTER HOOK
│       ├── squad_eval.py           # MINIMAL ADAPTER HOOK
│       ├── kivi_storage_eval.py    # NEW GLUE
│       └── kivi_adapter_benchmark.py
│
├── src/
│   ├── kivi_adapter.py             # NEW GLUE
│   └── cache_pipeline.py           # NEW GLUE
│
├── tests/
│   ├── test_kivi_import.py
│   ├── test_kivi_adapter.py
│   ├── test_kivi_batch_processor.py
│   ├── test_cache_pipeline.py
│   └── test_kivi_storage_eval.py
│
├── docs/
│   └── upstream_versions.md
│
└── third_party/
    └── KIVI/                       # GIT SUBMODULE, UNCHANGED
```

---

# 30. Implementation Order

Follow this sequence strictly:

```text
1. Pin Shadow + KIVI commits
        |
        v
2. KIVI import / Triton compatibility smoke
        |
        v
3. Synthetic KIVI quant/dequant adapter
        |
        v
4. Adapter unit tests
        |
        v
5. Real Shadow cache -> KIVI
        |
        v
6. Save native.pt + Shadow attack-view past_key_values.pt
        |
        v
7. Shadow attacks on KIVI
        |
        v
8. Collision+ calibration for KIVI
        |
        v
9. KV-Cloak -> KIVI composition
        |
        v
10. Shadow attacks on KV-Cloak + KIVI
        |
        v
11. Generic cache pipeline
        |
        v
12. MMLU / SQuAD integration
        |
        v
13. Native memory measurement
        |
        v
14. Quant/dequant latency measurement
        |
        v
15. 20-sample integration experiment
        |
        v
16. Full experiment
```

---

# 31. Suggested Commit Sequence

Keep commits small and independently reviewable.

```text
build: pin KIVI upstream dependency

test: add KIVI CUDA compatibility smoke test

feat: add KIVI cache adapter

test: validate KIVI cache partition and roundtrip

feat: add KIVI cache batch processor

test: verify KIVI adapter on Shadow cache

feat: support KIVI cache pipeline in utility evaluation

feat: compose KV-Cloak with KIVI cache compression

feat: measure native KIVI storage footprint

feat: benchmark KIVI adapter latency

docs: document Shadow and KIVI integration protocol
```

---

# 32. Definition of Done

The integration is considered complete only when all of the following hold.

## Upstream Integrity

- [ ] `inference/get_kvcache.py` remains unchanged.
- [ ] `attack/inversion.py` remains unchanged.
- [ ] `attack/collision.py` remains unchanged.
- [ ] `attack/injection.py` remains unchanged.
- [ ] KIVI source remains an untouched pinned submodule.

## Adapter Integrity

- [ ] Our adapter calls KIVI public quantization functions.
- [ ] Our adapter calls KIVI public dequantization functions.
- [ ] No local quantization formula is implemented.
- [ ] Provenance tests prove public KIVI functions are invoked.

## Functional Integration

The same source Shadow cache can be evaluated as:

```text
origin
kivi
kvcloak
kvcloak+kivi
```

- [ ] Inversion works for every condition.
- [ ] Collision works for every condition.
- [ ] Injection works for every condition.
- [ ] Collision+ has protection-specific calibration.

## Utility

- [ ] Origin evaluator still works.
- [ ] KIVI roundtrip evaluation works.
- [ ] KV-Cloak evaluation still works.
- [ ] KV-Cloak + KIVI evaluation works.

## Efficiency

- [ ] Native KIVI storage bytes are reported.
- [ ] Compression ratio is reported.
- [ ] Quantization latency is reported.
- [ ] Dequantization latency is reported.
- [ ] Roundtrip latency is reported.

## Research Output

Final experiments provide three result groups:

```text
Privacy
    Shadow attack scores

Utility
    MMLU / SQuAD

Efficiency
    Native KV memory + quant/dequant latency
```

---

# 33. Main Methodological Principle

The cleanest research framing is:

> Shadow/KV-Cloak defines the privacy threat model and evaluation environment. KIVI defines the cache compression mechanism. Our implementation contributes only the protocol and adapters necessary to evaluate how quantization and privacy protection interact under one controlled KV-cache pipeline.

This separation makes the result easier to reproduce, easier to audit, and easier to defend scientifically because differences in attack or quantization behavior cannot be attributed to locally rewritten implementations.

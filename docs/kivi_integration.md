# KV-Cloak + KIVI integration

This checkout is based on Shadow/KV-Cloak. The attack implementations and
canonical KV extraction remain upstream code; the new code only adapts
Shadow's `(K, V)` tuples to KIVI's packed representation and orchestrates
experiments.

## Environment

The Phase 1 launcher performs this setup automatically. For manual work,
install the KV-Cloak requirements first. KIVI has a separate, older
requirements file and should not be installed wholesale into the same
environment. The adapter adds the local submodule to the import path when it
is present, so these commands are also sufficient for a normal checkout:

```bash
pip install -r requirements.txt
git submodule update --init --recursive
```

The pinned public KIVI kernels require CUDA. Verify the compatibility gate:

```bash
pytest -q tests/test_kivi_import.py
```

## Cache conditions

The default configuration is `k_bits=2`, `v_bits=2`, `group_size=32`, and
`residual_length=32`. The adapter uses KIVI's partition contract:

- Key: quantized prefix is grouped over the sequence dimension after
  `transpose(2, 3)`; the recent suffix remains FP.
- Value: old tokens are quantized over the head dimension; the most recent
  `residual_length` tokens remain FP.

For example, with `T=130` and the default configuration, Key is split as
`128 + 2` and Value as `98 + 32`.

Each processed sample contains both representations:

```text
<sample>/kivi_k2_v2_g32_r32/
├── native.pt                 # packed KIVI tensors and FP residuals
├── past_key_values.pt        # dequantized Shadow attack view
└── metadata.json             # config and upstream provenance
```

The attack view is deliberately reconstructed with KIVI's public
`unpack_and_dequant_vcache`; attacks do not operate on packed `int32` values.

## Generate and process a canonical cache

First generate the source cache with the unchanged Shadow extractor:

```bash
python inference/get_kvcache.py \
  --model-name Llama-3.2-1B \
  --dataset ./dataset/lmsys-chat-1m_1k.jsonl \
  --dtype float16 \
  --device cuda:0 \
  --max-samples 20
```

Then materialize the KIVI condition from those exact `origin` files:

```bash
python defense/baseline/kivi_kvcache.py \
  --model-name Llama-3.2-1B \
  --dataset-path ./dataset/lmsys-chat-1m_1k.jsonl \
  --source-protect-type origin \
  --k-bits 2 --v-bits 2 \
  --group-size 32 --residual-length 32 \
  --dtype float16 --device cuda:0
```

To compose with an already materialized KV-Cloak cache, use
`--source-protect-type kvcloak`. The output is named
`kvcloak_kivi_k2_v2_g32_r32` and retains the obfuscated attack view.

The existing attack harness can consume the new attack view without changes:

```bash
python attack/attacks.py \
  --target-model-name Llama-3.2-1B \
  --dataset-path ./dataset/lmsys-chat-1m_1k.jsonl \
  --protect-type kivi_k2_v2_g32_r32 \
  --dtype float16 --device cuda:0 \
  --run-inversion --run-collision --run-injection \
  --i-understand-risks
```

If `--enhance` is used, calibrate a separate collision configuration for
every condition. For example:

```bash
python attack/get_collision_threshold.py \
  --model_path ~/model/Llama-3.2-1B \
  --target_data_path cache/float16/lmsys-chat-1m_1k/Llama-3.2-1B/<hash>/kivi_k2_v2_g32_r32/past_key_values.pt \
  --protect_type kivi_k2_v2_g32_r32 \
  --target_model_name Llama-3.2-1B \
  --dtype float16 --device cuda:0
```

## Utility evaluation

`MMLUEvaluator` and `SQuADEvaluator` accept an optional `cache_adapter`.
Supported CLI condition names include `kivi_*` and `kvcloak_kivi_*`; the
same prompt, split, metric, and scoring code is used for all conditions.

```bash
python defense/eval/mmlu_eval.py \
  --model-name Llama-3.2-1B \
  --protect-type kivi_k2_v2_g32_r32 \
  --k-bits 2 --v-bits 2 --group-size 32 --residual-length 32 \
  --dtype float16 --device cuda:0
```

## Efficiency measurements

Measure compression using `native.pt`, never the reconstructed attack view:

```bash
python defense/eval/kivi_storage_eval.py \
  --cache-root cache/float16/lmsys-chat-1m_1k/Llama-3.2-1B \
  --protect-type kivi_k2_v2_g32_r32
```

Benchmark quantization, dequantization, and roundtrip separately:

```bash
python defense/eval/kivi_adapter_benchmark.py \
  --device cuda:0 --dtype float16 \
  --k-bits 2 --v-bits 2 --group-size 32 --residual-length 32
```

These timings describe the offline adapter only. They are not an
end-to-end KIVI model-throughput claim.

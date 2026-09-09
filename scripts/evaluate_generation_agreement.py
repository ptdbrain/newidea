"""Compare greedy continuation tokens across cached Phase 1 conditions."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


def _load(path: Path) -> Any:
    try:
        return torch.load(path, weights_only=True)
    except TypeError:
        return torch.load(path)


def _next_tokens(
    model,
    input_ids: list[int],
    cache,
    device: str,
    max_new_tokens: int,
) -> list[int]:
    if not input_ids:
        return []
    cache = tuple(
        (key.to(device=device, dtype=model.dtype), value.to(device=device, dtype=model.dtype))
        for key, value in cache
    )
    sequence_length = cache[0][0].shape[2]
    if sequence_length < 1:
        return []
    # The stored cache includes the complete prompt. Re-run the final prompt
    # token against the prefix so that we can observe the first continuation
    # distribution without adding a duplicate token.
    prefix = tuple((key[:, :, :-1, :], value[:, :, :-1, :]) for key, value in cache)
    past = DynamicCache.from_legacy_cache(prefix)
    token = torch.tensor([[input_ids[-1]]], device=device, dtype=torch.long)
    generated: list[int] = []
    attention_mask = torch.ones((1, sequence_length), device=device, dtype=torch.long)
    with torch.no_grad():
        outputs = model(
            input_ids=token,
            attention_mask=attention_mask,
            past_key_values=past,
            use_cache=True,
        )
    for _ in range(max_new_tokens):
        next_token = int(torch.argmax(outputs.logits[:, -1, :], dim=-1).item())
        generated.append(next_token)
        token = torch.tensor([[next_token]], device=device, dtype=torch.long)
        past = outputs.past_key_values
        attention_mask = torch.ones(
            (1, past[0][0].shape[2] + 1), device=device, dtype=torch.long
        )
        with torch.no_grad():
            outputs = model(
                input_ids=token,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
            )
    return generated


def _prompt_ppl(
    model,
    input_ids: list[int],
    cache,
    device: str,
    max_tokens: int,
) -> dict[str, float | int]:
    """Compute teacher-forced NLL using the condition's cached prefixes.

    The stored cache contains the complete prompt. For target token ``t`` we
    use the cached prefix before ``t-1`` and feed token ``t-1`` to obtain its
    next-token logits. This keeps the condition in the scoring path instead
    of measuring an identical from-scratch model forward for every variant.
    """
    cache = tuple(
        (key.to(device=device, dtype=model.dtype), value.to(device=device, dtype=model.dtype))
        for key, value in cache
    )
    limit = min(len(input_ids), max_tokens)
    if limit < 2:
        return {"nll": 0.0, "ppl": 1.0, "token_count": 0}
    nll = 0.0
    token_count = 0
    for target_index in range(1, limit):
        prefix_end = target_index - 1
        if prefix_end:
            prefix = tuple(
                (key[:, :, :prefix_end, :], value[:, :, :prefix_end, :])
                for key, value in cache
            )
            past = DynamicCache.from_legacy_cache(prefix)
        else:
            past = None
        token = torch.tensor([[input_ids[prefix_end]]], device=device, dtype=torch.long)
        attention_mask = torch.ones(
            (1, prefix_end + 1), device=device, dtype=torch.long
        )
        with torch.no_grad():
            outputs = model(
                input_ids=token,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=False,
            )
        log_probability = torch.log_softmax(outputs.logits[:, -1, :], dim=-1)
        nll -= float(log_probability[0, input_ids[target_index]].item())
        token_count += 1
    mean_nll = nll / max(token_count, 1)
    return {"nll": nll, "ppl": float(torch.exp(torch.tensor(mean_nll)).item()), "token_count": token_count}


def evaluate(
    dataset_path: Path,
    cache_root: Path,
    model_path: Path,
    conditions: dict[str, str],
    device: str,
    max_new_tokens: int,
    dtype: torch.dtype = torch.float16,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, attn_implementation="eager"
    ).to(device, dtype)
    model.eval()
    with dataset_path.open("r", encoding="utf-8") as handle:
        dataset = [json.loads(line) for line in handle if line.strip()]

    outputs: list[dict[str, Any]] = []
    ppl_outputs: list[dict[str, Any]] = []
    for sample in dataset:
        input_ids = sample["input token ids"] if "input token ids" in sample else None
        if input_ids is None:
            # The source JSONL stores text; decode.json is the canonical record
            # produced by prefill and contains the exact BOS-adjusted IDs.
            sample_hash = hashlib.sha1(sample["prompt"].encode("utf-8")).hexdigest()
            decode = json.loads((cache_root / sample_hash / "decode.json").read_text())
            input_ids = decode["input token ids"]
        sample_hash = hashlib.sha1(sample["prompt"].encode("utf-8")).hexdigest()
        generated: dict[str, list[int]] = {}
        timings: dict[str, float] = {}
        for condition, directory_name in conditions.items():
            cache_path = cache_root / sample_hash / directory_name / "past_key_values.pt"
            start = time.time()
            generated[condition] = _next_tokens(
                model,
                input_ids,
                _load(cache_path),
                device,
                max_new_tokens,
            )
            ppl = _prompt_ppl(
                model,
                input_ids,
                _load(cache_path),
                device,
                max_tokens=len(input_ids),
            )
            timings[condition] = time.time() - start
            ppl_outputs.append(
                {
                    "sample_id": sample["sample_id"],
                    "source_row_id": sample.get("source_row_id"),
                    "input_hash": sample_hash,
                    "input_token_length": sample.get("prompt_token_length"),
                    "length_stratum": sample.get("length_stratum"),
                    "condition": condition,
                    "metric": "prompt_perplexity_teacher_forced",
                    **ppl,
                }
            )
        reference = generated["FP16"]
        for condition, tokens in generated.items():
            matches = sum(left == right for left, right in zip(reference, tokens))
            denominator = max(len(reference), 1)
            outputs.append(
                {
                    "sample_id": sample["sample_id"],
                    "source_row_id": sample.get("source_row_id"),
                    "input_hash": sample_hash,
                    "input_token_length": sample.get("prompt_token_length"),
                    "length_stratum": sample.get("length_stratum"),
                    "condition": condition,
                    "metric": "generation_agreement_vs_FP16",
                    "agreement": matches / denominator,
                    "matched_tokens": matches,
                    "reference_tokens": reference,
                    "condition_tokens": tokens,
                    "max_new_tokens": max_new_tokens,
                    "time_sec": timings[condition],
                }
            )
    return outputs, ppl_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    args = parser.parse_args()
    conditions = {
        "FP16": "origin",
        "KIVI4-STD": "kivi_k4_v4_g32_r32",
        "KIVI4-FQ": "kivi_k4_v4_g32_fq",
        "KIVI2-STD": "kivi_k2_v2_g32_r32",
        "KIVI2-FQ": "kivi_k2_v2_g32_fq",
    }
    records, ppl_records = evaluate(
        args.dataset,
        args.cache_root,
        args.model_path,
        conditions,
        args.device,
        args.max_new_tokens,
        getattr(torch, args.dtype),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    ppl_output = args.output.with_name("prompt_ppl.jsonl")
    with ppl_output.open("w", encoding="utf-8") as handle:
        for record in ppl_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} generation-agreement records to {args.output}")
    print(f"Wrote {len(ppl_records)} prompt-PPL records to {ppl_output}")


if __name__ == "__main__":
    main()

import argparse
import hashlib
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from pdsplit import prefill


def validate_local_checkpoint(model_path: Path) -> None:
    """Reject incomplete offline model checkpoints before loading begins."""
    model_path = Path(model_path)
    if not model_path.is_dir():
        raise FileNotFoundError(f"model checkpoint directory not found: {model_path}")

    missing: list[str] = []
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
    if not any(
        (model_path / name).is_file() for name in ("tokenizer.json", "tokenizer.model")
    ):
        missing.append("tokenizer.json or tokenizer.model")

    if missing:
        raise FileNotFoundError(
            f"incomplete local model checkpoint at {model_path}: "
            f"missing {', '.join(missing)}"
        )


def extract_user_input(item: dict, dataset_filename: str) -> str | None:
    """Extract the canonical prompt from normalized or legacy dataset rows."""
    prompt = item.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()

    if dataset_filename == "lmsys-chat-1m_1k.jsonl":
        messages = item.get("conversation", [])
        for turn in messages:
            if turn.get("role") == "user":
                value = turn.get("content")
                return value.strip() if isinstance(value, str) and value.strip() else None

    if dataset_filename == "gsm8k_1k.jsonl":
        value = item.get("question")
        return value.strip() if isinstance(value, str) and value.strip() else None

    if dataset_filename == "alpaca_1k.jsonl":
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        if input_text:
            return (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}"
            )
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}"
        )

    messages = item.get("messages", [])
    if isinstance(messages, list):
        for turn in messages:
            if isinstance(turn, dict) and turn.get("role") == "user":
                value = turn.get("content")
                return value.strip() if isinstance(value, str) and value.strip() else None

    for field in ("question", "instruction"):
        value = item.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def process_dataset_records(
    model,
    tokenizer,
    dataset: list[dict],
    dataset_path: Path,
    cache_root: Path,
    *,
    minimal_cache: bool,
    prefill_fn=prefill,
) -> int:
    """Generate one canonical Origin cache per record or fail immediately."""
    generated = 0
    for index, item in enumerate(
        tqdm(dataset, desc=f"Processing {dataset_path.name}")
    ):
        sample_id = str(item.get("sample_id", f"index-{index}"))
        user_input = extract_user_input(item, dataset_path.name)
        if not user_input:
            raise RuntimeError(
                f"prefill failed for sample {sample_id}: no usable prompt"
            )

        input_hash = hashlib.sha1(user_input.encode("utf-8")).hexdigest()
        cache_dir = cache_root / input_hash
        try:
            prefill_fn(
                model,
                tokenizer,
                user_input,
                cache_dir,
                save_intermediates=not minimal_cache,
            )
        except Exception as error:
            raise RuntimeError(
                f"prefill failed for sample {sample_id} at {cache_dir}: {error}"
            ) from error

        canonical_path = cache_dir / "origin" / "past_key_values.pt"
        if not canonical_path.is_file():
            raise RuntimeError(
                f"prefill produced no canonical cache for sample {sample_id}: "
                f"{canonical_path}"
            )
        generated += 1

    if generated != len(dataset):
        raise RuntimeError(
            f"prefill incomplete: expected {len(dataset)}, generated {generated}"
        )
    return generated


def main(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    dataset_name: str,
    max_samples: int = None,
    model_path: str | Path | None = None,
    minimal_cache: bool = False,
):
    """
    加载模型和数据集，为指定数据集中的每个用户输入生成并保存 KV 缓存。
    """
    model_path = (
        Path(model_path).expanduser()
        if model_path is not None
        else Path(f"~/model/{model_name}").expanduser()
    )
    dataset_path = Path(dataset_name)

    validate_local_checkpoint(model_path)
    if not dataset_path.is_file():
        print(f"Warning: Dataset path does not exist, skipping: {dataset_path}")
        return

    print(f"Loading tokenizer and model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        attn_implementation="eager",
        local_files_only=True,
    ).to(device, dtype)
    model.eval()
    print("Model loaded successfully.")

    print(f"Loading dataset from {dataset_path}...")
    dataset = []
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            dataset = [json.loads(line) for line in f if line.strip()]
    except Exception as e:
        print(f"Error loading dataset {dataset_path}: {e}")
        return

    if max_samples:
        dataset = dataset[:max_samples]
        print(f"Processing {len(dataset)} samples (limited to {max_samples}) from {dataset_path.name}...")
    else:
        print(f"Processing {len(dataset)} samples from {dataset_path.name}...")

    dtype_name = str(dtype).split(".")[-1]
    base_cache_dir_parent = Path(
        f"cache/{dtype_name}/{dataset_path.stem}/{model_name}/"
    )

    generated = process_dataset_records(
        model,
        tokenizer,
        dataset,
        dataset_path,
        base_cache_dir_parent,
        minimal_cache=minimal_cache,
    )

    print(
        f"\nFinished processing {generated} samples from {dataset_path.name}. "
        f"KV-Cache data saved in '{base_cache_dir_parent}'"
    )


if __name__ == "__main__":
    torch.manual_seed(42)
    parser = argparse.ArgumentParser(description="Generate KV-cache for a dataset.")
    parser.add_argument(
        "--model-name",
        default="Llama-3.2-1B",
        help="Model directory name under ~/model/.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional local model path; overrides ~/model/<model-name>.",
    )
    parser.add_argument(
        "--dataset",
        default="./dataset/lmsys-chat-1m_1k.jsonl",
        help="Path to dataset jsonl file.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Computation device (e.g., cuda:0 or cpu).",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "bfloat16", "float32"],
        help="Computation dtype.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing).",
    )
    parser.add_argument(
        "--minimal-cache",
        action="store_true",
        help="Save only KV tensors and decode metadata; omit attentions/hidden states.",
    )
    args = parser.parse_args()

    print(
        f"\n--- Processing model: {args.model_name} on dataset: {args.dataset} ---"
    )
    main(
        model_name=args.model_name,
        device=args.device,
        dtype=getattr(torch, args.dtype),
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        model_path=args.model_path,
        minimal_cache=args.minimal_cache,
    )

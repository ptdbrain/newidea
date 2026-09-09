"""Small contract tests for optional utility-evaluator cache hooks."""

import inspect
from pathlib import Path
from types import SimpleNamespace

import torch

from defense.eval.mmlu_eval import MMLUEvaluator
from defense.eval.squad_eval import SQuADEvaluator


def test_utility_evaluators_accept_cache_adapter():
    assert "cache_adapter" in inspect.signature(MMLUEvaluator.__init__).parameters
    assert "cache_adapter" in inspect.signature(SQuADEvaluator.__init__).parameters


class _Inputs(dict):
    def to(self, device):
        return self


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, prompt, return_tensors=None):
        return _Inputs(
            input_ids=torch.tensor([[1]]),
            attention_mask=torch.tensor([[1]]),
        )

    def encode(self, text, add_special_tokens=False, return_tensors=None):
        value = [1]
        return torch.tensor([value]) if return_tensors == "pt" else value

    def decode(self, token_ids, skip_special_tokens=True):
        return "A"


class _Model:
    device = torch.device("cpu")

    def __init__(self):
        self.calls = []

    def eval(self):
        return self

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        key = torch.ones(1, 1, 1, 2)
        value = torch.zeros(1, 1, 1, 2)
        return SimpleNamespace(
            past_key_values=((key, value),),
            logits=torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
        )


class _CacheAdapter:
    def __init__(self):
        self.seen = []

    def roundtrip(self, cache):
        self.seen.append(cache)
        return cache


def test_mmlu_applies_cache_adapter_after_prefill(tmp_path):
    model = _Model()
    adapter = _CacheAdapter()
    evaluator = MMLUEvaluator(
        model=model,
        tokenizer=_Tokenizer(),
        dataset={},
        device="cpu",
        output_filepath=Path(tmp_path) / "result.jsonl",
        shot_count=1,
        cache_adapter=adapter,
    )

    prediction, _ = evaluator._get_model_prediction("prompt", ":")

    assert prediction == "A"
    assert len(adapter.seen) == 1

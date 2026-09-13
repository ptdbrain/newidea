"""Regression tests for exact, bounded-memory Collision+ calibration."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from attack.collision_threshold_stats import (
    RunningStats,
    finalize_statistics,
    load_checkpoint,
    new_layer_statistics,
    save_checkpoint,
    stable_cache_digest,
    update_position_statistics,
    write_json_atomic,
)
from attack.get_collision_threshold import (
    _cache_key_values,
    _current_position_distances,
    calibrate_streaming,
    collect_position_distances,
    rebuild_prefix_cache,
)


class FakeCache:
    def __init__(
        self, key_cache: list[torch.Tensor], value_cache: list[torch.Tensor]
    ) -> None:
        self.key_cache = key_cache
        self.value_cache = value_cache


class FakeCandidateModel:
    device = torch.device("cpu")
    config = SimpleNamespace(vocab_size=5)

    def __init__(self) -> None:
        self.candidate_ids: list[int] = []

    def __call__(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        use_cache: bool,
        output_hidden_states: bool,
        past_key_values: FakeCache | None = None,
    ) -> SimpleNamespace:
        assert use_cache is True
        assert output_hidden_states is False
        assert attention_mask.shape[0] == input_ids.shape[0]
        candidates = input_ids[:, 0].to(dtype=torch.float32)
        self.candidate_ids.extend(int(item) for item in candidates.tolist())
        batch_size = candidates.shape[0]
        prefix_length = 0
        if past_key_values is not None:
            prefix_keys, _ = _cache_key_values(past_key_values)
            prefix_length = prefix_keys[0].shape[2]
        old_prefix = torch.full((batch_size, 1, prefix_length, 1), 999.0)
        key = torch.cat(
            (old_prefix, candidates.reshape(batch_size, 1, 1, 1)), dim=2
        )
        value = torch.cat(
            (old_prefix, (2 * candidates).reshape(batch_size, 1, 1, 1)), dim=2
        )
        return SimpleNamespace(past_key_values=FakeCache([key], [value]))


def test_running_stats_matches_direct_float64_statistics() -> None:
    stats = RunningStats()
    stats.update(torch.tensor([1.0, 2.0]))
    stats.update(torch.tensor([3.0, 4.0]))

    assert stats.count == 4
    assert stats.mean == pytest.approx(2.5)
    assert stats.std == pytest.approx(1.2909944487358056)
    assert stats.minimum == 1.0
    assert stats.maximum == 4.0


def test_running_stats_empty_and_singleton_have_zero_std() -> None:
    empty = RunningStats()
    singleton = RunningStats()
    singleton.update(torch.tensor([7.0]))

    assert empty.std == 0.0
    assert singleton.std == 0.0


def test_running_stats_rejects_non_finite_values() -> None:
    stats = RunningStats()

    with pytest.raises(ValueError, match="non-finite"):
        stats.update(torch.tensor([1.0, math.inf]))


def test_running_stats_serialization_roundtrip() -> None:
    stats = RunningStats()
    stats.update(torch.tensor([2.0, 4.0, 8.0]))

    restored = RunningStats.from_dict(stats.to_dict())

    assert restored.count == 3
    assert restored.mean == pytest.approx(14.0 / 3.0)
    assert restored.std == pytest.approx(3.0550504633038935)
    assert restored.minimum == 2.0
    assert restored.maximum == 8.0


def test_position_updates_exclude_exact_true_token() -> None:
    state = new_layer_statistics(1)
    update_position_statistics(
        state,
        [
            (
                torch.tensor([0.0, 1.0, 2.0]),
                torch.tensor([10.0, 11.0, 12.0]),
            )
        ],
        true_token_id=0,
    )
    update_position_statistics(
        state,
        [
            (
                torch.tensor([3.0, 4.0, 5.0]),
                torch.tensor([13.0, 14.0, 15.0]),
            )
        ],
        true_token_id=2,
    )

    config = finalize_statistics(state, sequence_length=2, vocab_size=3)

    assert config[0]["target_mean"] == pytest.approx([2.5, 12.5])
    assert config[0]["target_std"] == pytest.approx(
        [3.5355339059327378, 3.5355339059327378]
    )
    assert config[0]["target_max"] == pytest.approx([5.0, 15.0])
    assert config[0]["others_mean"] == pytest.approx([2.5, 12.5])
    assert config[0]["others_std"] == pytest.approx(
        [1.2909944487358056, 1.2909944487358056]
    )
    assert config[0]["others_min"] == pytest.approx([1.0, 11.0])


def test_position_update_requires_one_vector_per_layer() -> None:
    state = new_layer_statistics(2)

    with pytest.raises(ValueError, match="expected 2 layers, got 1"):
        update_position_statistics(
            state,
            [(torch.zeros(3), torch.zeros(3))],
            true_token_id=0,
        )


@pytest.mark.parametrize(
    ("distances", "message"),
    [
        ((torch.zeros(1, 3), torch.zeros(3)), "one-dimensional"),
        ((torch.zeros(3), torch.zeros(2)), "equal vocabulary length"),
    ],
)
def test_position_update_rejects_malformed_vectors(
    distances: tuple[torch.Tensor, torch.Tensor], message: str
) -> None:
    state = new_layer_statistics(1)

    with pytest.raises(ValueError, match=message):
        update_position_statistics(state, [distances], true_token_id=0)


def test_position_update_rejects_out_of_range_true_token() -> None:
    state = new_layer_statistics(1)

    with pytest.raises(ValueError, match="true token ID 3 is outside"):
        update_position_statistics(
            state,
            [(torch.zeros(3), torch.zeros(3))],
            true_token_id=3,
        )


def test_finalize_rejects_incomplete_observation_counts() -> None:
    state = new_layer_statistics(1)
    update_position_statistics(
        state,
        [(torch.tensor([0.0, 1.0, 2.0]), torch.tensor([0.0, 1.0, 2.0]))],
        true_token_id=0,
    )

    with pytest.raises(
        ValueError,
        match="layer 0 K target count: expected 2, got 1",
    ):
        finalize_statistics(state, sequence_length=2, vocab_size=3)


def test_current_position_distance_ignores_old_prefix() -> None:
    current = FakeCache(
        key_cache=[
            torch.tensor(
                [
                    [[[999.0], [3.0]]],
                    [[[999.0], [5.0]]],
                ]
            )
        ],
        value_cache=[
            torch.tensor(
                [
                    [[[999.0], [7.0]]],
                    [[[999.0], [11.0]]],
                ]
            )
        ],
    )
    target = (
        (
            torch.tensor([[[[3.0]]]]),
            torch.tensor([[[[7.0]]]]),
        ),
    )

    distances = _current_position_distances(current, target, seq_id=0)

    assert distances[0][0].tolist() == pytest.approx([0.0, 2.0])
    assert distances[0][1].tolist() == pytest.approx([0.0, 4.0])


def test_current_position_distance_rejects_layer_mismatch() -> None:
    current = FakeCache([torch.zeros(1, 1, 1, 1)], [torch.zeros(1, 1, 1, 1)])

    with pytest.raises(ValueError, match="current cache has 1 layers, target has 2"):
        _current_position_distances(
            current,
            (
                (torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1)),
                (torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1)),
            ),
            seq_id=0,
        )


def test_current_position_distance_rejects_target_position_out_of_bounds() -> None:
    current = FakeCache([torch.zeros(1, 1, 1, 1)], [torch.zeros(1, 1, 1, 1)])
    target = ((torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1)),)

    with pytest.raises(ValueError, match="sequence position 1 is outside"):
        _current_position_distances(current, target, seq_id=1)


def test_current_position_distance_rejects_head_shape_mismatch() -> None:
    current = FakeCache([torch.zeros(2, 2, 1, 3)], [torch.zeros(2, 2, 1, 3)])
    target = ((torch.zeros(1, 1, 1, 3), torch.zeros(1, 1, 1, 3)),)

    with pytest.raises(ValueError, match="layer 0 K shape mismatch"):
        _current_position_distances(current, target, seq_id=0)


@pytest.mark.parametrize("batch_size", [2, 3])
def test_full_vocabulary_collection_is_exact_and_cpu_bounded(batch_size: int) -> None:
    model = FakeCandidateModel()
    target = ((torch.tensor([[[[2.0]]]]), torch.tensor([[[[4.0]]]])),)

    distances = collect_position_distances(
        model,
        target,
        current_kvcache=None,
        seq_id=0,
        batch_size=batch_size,
    )

    assert model.candidate_ids == [0, 1, 2, 3, 4]
    assert distances[0][0].tolist() == pytest.approx([2.0, 1.0, 0.0, 1.0, 2.0])
    assert distances[0][1].tolist() == pytest.approx([4.0, 2.0, 0.0, 2.0, 4.0])
    for component in distances[0]:
        assert component.device.type == "cpu"
        assert component.ndim == 1
        assert component.numel() == 5
        assert torch.isfinite(component).all()


def test_full_vocabulary_collection_rejects_non_positive_batch() -> None:
    model = FakeCandidateModel()
    target = ((torch.tensor([[[[2.0]]]]), torch.tensor([[[[4.0]]]])),)

    with pytest.raises(ValueError, match="batch_size must be positive"):
        collect_position_distances(
            model,
            target,
            current_kvcache=None,
            seq_id=0,
            batch_size=0,
        )


def test_stable_cache_digest_tracks_tensor_content_shape_and_dtype() -> None:
    original = ((torch.tensor([[[[1.0, 2.0]]]]), torch.tensor([[[[3.0, 4.0]]]])),)
    equivalent = tuple((key.clone(), value.clone()) for key, value in original)
    changed_value = ((original[0][0].clone(), torch.tensor([[[[3.0, 5.0]]]])),)
    changed_shape = (
        (original[0][0].reshape(1, 1, 2, 1), original[0][1].reshape(1, 1, 2, 1)),
    )
    changed_dtype = tuple(
        (key.double(), value.double()) for key, value in original
    )

    digest = stable_cache_digest(original)

    assert stable_cache_digest(equivalent) == digest
    assert stable_cache_digest(changed_value) != digest
    assert stable_cache_digest(changed_shape) != digest
    assert stable_cache_digest(changed_dtype) != digest


def test_checkpoint_roundtrip_preserves_stats_and_records_batch(
    tmp_path,
) -> None:
    state = new_layer_statistics(1)
    update_position_statistics(
        state,
        [(torch.tensor([0.0, 1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0]))],
        true_token_id=2,
    )
    checkpoint_path = tmp_path / "streaming_stats_v2.pt"
    metadata = {"target_digest": "abc", "vocab_size": 3, "num_layers": 1}

    save_checkpoint(
        checkpoint_path,
        next_seq_id=1,
        layer_statistics=state,
        metadata=metadata,
        batch_size=128,
    )
    next_seq_id, restored = load_checkpoint(
        checkpoint_path,
        expected_metadata=metadata,
    )
    payload = torch.load(checkpoint_path, weights_only=True)

    assert next_seq_id == 1
    assert restored[0]["target"][0].count == 1
    assert restored[0]["target"][0].mean == 2.0
    assert restored[0]["others"][1].mean == pytest.approx(3.5)
    assert payload["execution"]["batch_size"] == 128
    assert not checkpoint_path.with_suffix(".pt.tmp").exists()


def test_checkpoint_allows_batch_change_but_rejects_semantic_mismatch(
    tmp_path,
) -> None:
    checkpoint_path = tmp_path / "streaming_stats_v2.pt"
    metadata = {"target_digest": "abc", "vocab_size": 3, "num_layers": 1}
    save_checkpoint(
        checkpoint_path,
        next_seq_id=0,
        layer_statistics=new_layer_statistics(1),
        metadata=metadata,
        batch_size=128,
    )

    next_seq_id, _ = load_checkpoint(
        checkpoint_path,
        expected_metadata=metadata,
    )
    assert next_seq_id == 0

    with pytest.raises(
        ValueError,
        match="checkpoint metadata mismatch for target_digest: expected def, got abc",
    ):
        load_checkpoint(
            checkpoint_path,
            expected_metadata={**metadata, "target_digest": "def"},
        )


def test_checkpoint_rejects_truncated_payload(tmp_path) -> None:
    checkpoint_path = tmp_path / "streaming_stats_v2.pt"
    checkpoint_path.write_bytes(b"not a torch checkpoint")

    with pytest.raises(RuntimeError, match=r"could not load Collision\+ checkpoint"):
        load_checkpoint(
            checkpoint_path,
            expected_metadata={"target_digest": "abc"},
        )


def test_atomic_json_writer_replaces_complete_document(tmp_path) -> None:
    output_path = tmp_path / "config.json"
    output_path.write_text("old", encoding="utf-8")

    write_json_atomic(output_path, [{"target_mean": [1.0, 2.0]}])

    assert output_path.read_text(encoding="utf-8").endswith("\n")
    assert output_path.read_text(encoding="utf-8").startswith("[\n")
    assert not output_path.with_suffix(".json.tmp").exists()


class NoCallModel:
    device = torch.device("cpu")
    config = SimpleNamespace(vocab_size=3)

    def __call__(self, **_kwargs):
        raise AssertionError("completed checkpoint must not call the model")


def test_completed_checkpoint_finalizes_without_model_calls(tmp_path) -> None:
    checkpoint_path = tmp_path / "streaming_stats_v2.pt"
    metadata = {"target_digest": "abc", "vocab_size": 3, "num_layers": 1}
    state = new_layer_statistics(1)
    update_position_statistics(
        state,
        [(torch.tensor([0.0, 1.0, 2.0]), torch.tensor([3.0, 4.0, 5.0]))],
        true_token_id=2,
    )
    save_checkpoint(
        checkpoint_path,
        next_seq_id=1,
        layer_statistics=state,
        metadata=metadata,
        batch_size=128,
    )

    config = calibrate_streaming(
        NoCallModel(),
        ((torch.zeros(1, 1, 1, 1), torch.zeros(1, 1, 1, 1)),),
        input_ids=torch.tensor([[2]]),
        batch_size=64,
        checkpoint_path=checkpoint_path,
        metadata=metadata,
    )

    assert config[0]["target_mean"] == pytest.approx([2.0, 5.0])
    assert config[0]["others_mean"] == pytest.approx([0.5, 3.5])


class PrefixModel:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.seen_input_ids: list[list[int]] = []

    def __call__(self, *, input_ids, attention_mask, use_cache, output_hidden_states):
        assert use_cache is True
        assert output_hidden_states is False
        assert attention_mask.shape == input_ids.shape
        self.seen_input_ids.append(input_ids[0].tolist())
        length = input_ids.shape[1]
        cache = FakeCache(
            [torch.zeros(1, 1, length, 1)],
            [torch.zeros(1, 1, length, 1)],
        )
        return SimpleNamespace(past_key_values=cache)


def test_rebuild_prefix_cache_uses_only_completed_positions() -> None:
    model = PrefixModel()

    cache = rebuild_prefix_cache(
        model,
        torch.tensor([[10, 20, 30, 40]]),
        next_seq_id=2,
    )

    assert model.seen_input_ids == [[10, 20]]
    assert cache.key_cache[0].shape[2] == 2


def test_partial_checkpoint_resumes_at_next_position(tmp_path) -> None:
    checkpoint_path = tmp_path / "streaming_stats_v2.pt"
    metadata = {"target_digest": "abc", "vocab_size": 5, "num_layers": 1}
    state = new_layer_statistics(1)
    update_position_statistics(
        state,
        [
            (
                torch.tensor([2.0, 1.0, 0.0, 1.0, 2.0]),
                torch.tensor([4.0, 2.0, 0.0, 2.0, 4.0]),
            )
        ],
        true_token_id=2,
    )
    save_checkpoint(
        checkpoint_path,
        next_seq_id=1,
        layer_statistics=state,
        metadata=metadata,
        batch_size=128,
    )
    model = FakeCandidateModel()
    target = (
        (
            torch.tensor([[[[2.0], [3.0]]]]),
            torch.tensor([[[[4.0], [6.0]]]]),
        ),
    )

    config = calibrate_streaming(
        model,
        target,
        input_ids=torch.tensor([[2, 3]]),
        batch_size=2,
        checkpoint_path=checkpoint_path,
        metadata=metadata,
    )

    assert model.candidate_ids == [2, 0, 1, 2, 3, 4, 3]
    assert config[0]["target_mean"] == pytest.approx([0.0, 0.0])
    assert config[0]["target_max"] == pytest.approx([0.0, 0.0])
    next_seq_id, restored = load_checkpoint(
        checkpoint_path,
        expected_metadata=metadata,
    )
    assert next_seq_id == 2
    assert restored[0]["target"][0].count == 2

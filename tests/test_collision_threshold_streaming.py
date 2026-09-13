"""Regression tests for exact, bounded-memory Collision+ calibration."""

from __future__ import annotations

import math

import pytest
import torch

from attack.collision_threshold_stats import (
    RunningStats,
    finalize_statistics,
    new_layer_statistics,
    update_position_statistics,
)


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

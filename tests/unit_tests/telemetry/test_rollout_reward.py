# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``gym.rollout.rewarded_total`` and ``gym.rollout.summed_reward``, against a real in-memory reader."""

import pytest

from nemo_gym import rollout_collection
from nemo_gym.telemetry import gym_metrics
from nemo_gym.telemetry.span_groups import GymSpanGroup
from tests.unit_tests.telemetry.test_sandbox_active import collected_metrics  # noqa: F401 - fixture


pytest.importorskip("opentelemetry.sdk.metrics")

REWARDED = gym_metrics.ROLLOUT_REWARDED_INSTRUMENT
SUMMED = gym_metrics.ROLLOUT_SUMMED_REWARD_INSTRUMENT
DISPATCH = gym_metrics.DISPATCH_NAME_ATTRIBUTE
MASKED = gym_metrics.ROLLOUT_MASKED_ATTRIBUTE
BUCKET = gym_metrics.REWARD_BUCKET_ATTRIBUTE


def _values(collected, name):
    return {tuple(sorted(p.attributes.items())): p.value for p in collected().get(name, [])}


def _key(masked=False, **attrs):
    return tuple(sorted({DISPATCH: "env", MASKED: masked, **attrs}.items()))


def _rollout_group_only(monkeypatch):
    monkeypatch.setattr(rollout_collection, "is_span_group_enabled", lambda group: group == GymSpanGroup.ROLLOUT)


def test_rewards_are_counted_by_bucket_and_summed_including_penalties(collected_metrics):  # noqa: F811
    rewards = (-100.0, 0.0, 0.0, 0.1, 0.25, 0.3, 0.6, 0.75, 0.999, 1.0, 1.0, 2.0)
    for reward in rewards:
        gym_metrics.record_rollout_reward(reward, dispatch_name="env", masked=False)
    assert _values(collected_metrics, REWARDED) == {
        _key(**{BUCKET: "<0"}): 1,
        _key(**{BUCKET: "0"}): 2,
        _key(**{BUCKET: "0-0.25"}): 2,
        _key(**{BUCKET: "0.25-0.5"}): 1,
        _key(**{BUCKET: "0.5-0.75"}): 2,
        _key(**{BUCKET: "0.75-1"}): 1,
        _key(**{BUCKET: "1"}): 2,
        _key(**{BUCKET: ">1"}): 1,
    }
    assert _values(collected_metrics, SUMMED) == {_key(): pytest.approx(sum(rewards))}


def test_driver_records_scored_rewards_only(collected_metrics, monkeypatch):  # noqa: F811
    _rollout_group_only(monkeypatch)
    rollout_collection._record_rollout_outcome({"reward": 1.0}, "scored", "env")
    rollout_collection._record_rollout_outcome({"reward": 0}, "scored", "env")
    rollout_collection._record_rollout_outcome(
        {"reward": 0.5, rollout_collection.MASK_SAMPLE_KEY: True}, "scored", "env"
    )
    rollout_collection._record_rollout_outcome(
        {"reward": 0.0, rollout_collection.NG_FAILURE_CLASS_KEY: "agent_run_error"}, "failed", "env"
    )
    rollout_collection._record_rollout_outcome(
        {"reward": 0.0, rollout_collection.NG_FAILURE_CLASS_KEY: "kill_shaped"}, "omitted", "env"
    )
    assert _values(collected_metrics, REWARDED) == {
        _key(**{BUCKET: "1"}): 1,
        _key(**{BUCKET: "0"}): 1,
        _key(masked=True, **{BUCKET: "0.25-0.5"}): 1,
    }
    assert _values(collected_metrics, SUMMED) == {_key(): 1.0, _key(masked=True): 0.5}


@pytest.mark.parametrize("result", [{}, {"reward": None}, {"reward": True}, {"reward": "1"}, {"reward": float("nan")}])
def test_a_result_without_a_numeric_reward_is_not_a_zero(collected_metrics, monkeypatch, result):  # noqa: F811
    _rollout_group_only(monkeypatch)
    rollout_collection._record_rollout_outcome(result, "scored", "env")
    assert _values(collected_metrics, REWARDED) == {}
    assert _values(collected_metrics, SUMMED) == {}
    # The attempt itself is still counted.
    assert len(collected_metrics()[gym_metrics.ROLLOUT_COMPLETED_INSTRUMENT]) == 1


def test_nothing_is_recorded_when_the_rollout_group_is_off(collected_metrics, monkeypatch):  # noqa: F811
    monkeypatch.setattr(rollout_collection, "is_span_group_enabled", lambda group: False)
    rollout_collection._record_rollout_outcome({"reward": 1.0}, "scored", "env")
    assert _values(collected_metrics, REWARDED) == {}

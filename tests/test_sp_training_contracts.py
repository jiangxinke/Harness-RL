from __future__ import annotations

import concurrent.futures
from types import SimpleNamespace

import pytest

from slime_plugins.stackplanner.config import (
    Experiment,
    TrainingConfig,
    TrainingMode,
)
from slime_plugins.stackplanner.rewards import (
    EpisodeOutcome,
    deterministic_reward,
)
from slime_plugins.stackplanner.routing import Backend, RoleRouter
from slime_plugins.stackplanner.sessions import SessionKey, SessionManager
from slime_plugins.stackplanner.trajectory import (
    annotate_episode_samples,
    infrastructure_failure,
    validate_sample,
)

POLICY_ROLES = {
    "researcher",
    "coder",
    "reporter",
    "outline",
    "perception",
    "memory_recaller",
}


def sample(*, response_length=2, mask=None, logprobs=None):
    return SimpleNamespace(
        tokens=[10, 11, 12, 13],
        response_length=response_length,
        loss_mask=list([1, 1] if mask is None else mask),
        rollout_log_probs=list([-0.1, -0.2] if logprobs is None else logprobs),
        metadata={},
        reward=None,
        remove_sample=False,
        group_index=None,
        rollout_id=None,
    )


@pytest.mark.parametrize("role", sorted(POLICY_ROLES))
def test_central_only_routes_every_runtime_subagent_frozen(role):
    router = RoleRouter(TrainingMode.CENTRAL_ONLY, POLICY_ROLES)
    assert router.route("central").backend is Backend.ACTOR
    assert router.route("central").trainable is True
    assert router.route(role).backend is Backend.FROZEN
    assert router.route(role).trainable is False


@pytest.mark.parametrize("role", ["central", *sorted(POLICY_ROLES)])
def test_shared_all_routes_every_runtime_policy_role_to_trainable_actor(role):
    route = RoleRouter(TrainingMode.SHARED_ALL_AGENTS, POLICY_ROLES).route(role)
    assert route.backend is Backend.ACTOR
    assert route.trainable is True


def test_non_policy_auxiliary_model_stays_frozen_in_shared_mode():
    router = RoleRouter(TrainingMode.SHARED_ALL_AGENTS, POLICY_ROLES)
    for role in ("title", "judge", "verifier", "preprocessor"):
        assert router.route(role).backend is Backend.FROZEN
        assert router.route(role).trainable is False


def test_backend_and_trainable_overrides_are_independent():
    router = RoleRouter(
        TrainingMode.SHARED_ALL_AGENTS,
        POLICY_ROLES,
        overrides={
            "researcher": {"trainable": False},
            "coder": {"backend": "frozen"},
        },
    )
    researcher = router.route("researcher")
    coder = router.route("coder")
    assert (researcher.backend, researcher.trainable) == (Backend.ACTOR, False)
    assert (coder.backend, coder.trainable) == (Backend.FROZEN, True)


def test_role_names_and_overrides_are_normalized():
    router = RoleRouter(
        TrainingMode.SHARED_ALL_AGENTS,
        ["memory_recaller"],
        {"MEMORY_RECALLER": {"trainable": False}},
    )
    route = router.route("memory-recaller")
    assert route == route.__class__("memory-recaller", Backend.ACTOR, False)


def test_session_key_separates_episode_role_task_attempt_and_continuation():
    base = {
        "episode_id": "episode-1",
        "role": "researcher",
        "action_id": "a1",
        "task_id": "t1",
    }
    ids = {
        SessionKey(**base).session_id(),
        SessionKey(**{**base, "episode_id": "episode-2"}).session_id(),
        SessionKey(**{**base, "role": "coder"}).session_id(),
        SessionKey(**{**base, "action_id": "a2"}).session_id(),
        SessionKey(**{**base, "task_id": "t2"}).session_id(),
        SessionKey(**base, attempt=1).session_id(),
        SessionKey(**base, continuation=1).session_id(),
    }
    assert len(ids) == 7


def test_same_role_concurrent_delegations_get_distinct_auditable_sessions():
    manager = SessionManager("episode-concurrent")

    def allocate(index):
        return manager.allocate(
            role="researcher", action_id=f"action-{index}", task_id=f"task-{index}"
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        allocations = list(executor.map(allocate, range(32)))

    session_ids = [sid for sid, _ in allocations]
    assert len(session_ids) == len(set(session_ids)) == 32
    assert set(manager.open_sessions()) == set(session_ids)
    for sid in session_ids:
        assert manager.close(sid) is not None
    assert manager.open_sessions() == {}


def test_exact_duplicate_session_allocation_is_rejected():
    manager = SessionManager("episode-duplicate")
    kwargs = {
        "role": "coder",
        "action_id": "action-1",
        "task_id": "task-1",
        "attempt": 0,
        "continuation": 0,
    }
    manager.allocate(**kwargs)
    with pytest.raises(ValueError, match="already allocated"):
        manager.allocate(**kwargs)


def test_session_id_cannot_be_reused_after_close():
    manager = SessionManager("episode-closed")
    sid, key = manager.allocate(role="central")
    assert manager.close(sid) == key
    with pytest.raises(ValueError, match="already allocated or closed"):
        manager.allocate(role="central")


def test_episode_annotation_preserves_full_team_reward_and_trace_metadata():
    samples = [sample(), sample()]
    annotated = annotate_episode_samples(
        samples,
        rollout_id=77,
        group_index=4,
        episode_id="episode-77",
        role="researcher",
        session_id="sid-77",
        mode=TrainingMode.SHARED_ALL_AGENTS.value,
        model_version="weights-3",
        trainable=True,
        reward=0.75,
        reward_components={"verifier": 0.75},
        action_id="action-77",
        task_id="task-77",
        parent_action_id="action-root",
    )

    for index, item in enumerate(annotated):
        assert item.rollout_id == 77
        assert item.group_index == 4
        assert item.reward == 0.75
        assert item.loss_mask == [1, 1]
        assert item.metadata == {
            "episode_id": "episode-77",
            "rollout_id": 77,
            "role": "researcher",
            "session_id": "sid-77",
            "training_mode": "shared-all-agents",
            "model_version": "weights-3",
            "action_id": "action-77",
            "task_id": "task-77",
            "parent_action_id": "action-root",
            "segment_index": index,
            "termination": "completed",
            "reward_components": {"verifier": 0.75},
        }


def test_frozen_annotation_masks_all_generated_tokens_but_keeps_exact_logprobs():
    item = sample()
    annotate_episode_samples(
        [item],
        rollout_id=1,
        group_index=1,
        episode_id="episode-1",
        role="researcher",
        session_id="sid-1",
        mode=TrainingMode.CENTRAL_ONLY.value,
        model_version="frozen-v1",
        trainable=False,
        reward=1.0,
    )
    assert item.loss_mask == [0, 0]
    assert item.rollout_log_probs == [-0.1, -0.2]
    validate_sample(item)


def test_trainable_annotation_rejects_fully_masked_trajectory():
    with pytest.raises(ValueError, match="expected at least one trainable"):
        annotate_episode_samples(
            [sample(mask=[0, 0])],
            rollout_id=1,
            group_index=0,
            episode_id="episode",
            role="central",
            session_id="session",
            mode=TrainingMode.CENTRAL_ONLY.value,
            model_version="v1",
            trainable=True,
            reward=1.0,
        )


@pytest.mark.parametrize(
    ("mask", "logprobs", "match"),
    [
        ([1], [-0.1, -0.2], "loss mask length"),
        ([1, 1], [-0.1], "logprob length"),
        ([1, 2], [-0.1, -0.2], "only 0/1"),
    ],
)
def test_invalid_training_sample_alignment_is_rejected(mask, logprobs, match):
    with pytest.raises(ValueError, match=match):
        validate_sample(sample(mask=mask, logprobs=logprobs))


def test_response_length_cannot_exceed_token_sequence():
    with pytest.raises(ValueError, match="invalid response_length"):
        validate_sample(sample(response_length=5, mask=[1] * 5, logprobs=[-0.1] * 5))


def test_infrastructure_failure_is_removed_and_never_policy_negative():
    item = infrastructure_failure(
        sample(), kind="sglang-disconnect", message="connection reset"
    )
    assert item.remove_sample is True
    assert item.reward == 0.0
    assert item.loss_mask == [0, 0]
    assert item.metadata["infrastructure_failure"] == {
        "kind": "sglang-disconnect",
        "message": "connection reset",
        "retryable": True,
    }


def test_deterministic_reward_separates_task_failure_from_infrastructure_failure():
    task_failure = deterministic_reward(
        EpisodeOutcome(
            episode_id="e1", prompt="p", final_answer="wrong", label="right"
        ),
        {},
    )
    infrastructure = deterministic_reward(
        EpisodeOutcome(
            episode_id="e2", prompt="p", final_answer="", status="adapter-error"
        ),
        {},
    )
    assert task_failure.value == 0.0 and task_failure.infrastructure_failure is False
    assert infrastructure.value == 0.0 and infrastructure.infrastructure_failure is True
    assert infrastructure.retryable is True
    success = deterministic_reward(
        EpisodeOutcome(
            episode_id="e3", prompt="p", final_answer="right", label="right"
        ),
        {},
    )
    assert success.value == 1.0 and success.infrastructure_failure is False


def test_deterministic_reward_normalizes_explanatory_multitoken_answers():
    result = deterministic_reward(
        EpisodeOutcome(
            episode_id="e4",
            prompt="p",
            final_answer="The answer is: Bank for International Settlements (BIS).",
            label="Bank for International Settlements (BIS)",
        ),
        {"metric": "token_f1"},
    )
    assert result.value == pytest.approx(5 / 6)
    assert result.components["token_recall"] == 1.0
    assert result.components["exact"] == 0.0


def test_deterministic_reward_uses_best_alias_and_rejects_unknown_metric():
    outcome = EpisodeOutcome(
        episode_id="e5",
        prompt="p",
        final_answer="NYC",
        label="New York City",
        verifier={"aliases": ["New York", "NYC"]},
    )
    assert deterministic_reward(outcome, {"metric": "token_f1"}).value == 1.0
    with pytest.raises(ValueError, match="unsupported deterministic reward metric"):
        deterministic_reward(outcome, {"metric": "not-a-metric"})


def test_deterministic_reward_supports_auditable_exact_token_f1_mix():
    outcome = EpisodeOutcome(
        episode_id="e6",
        prompt="p",
        final_answer="The answer is Paris",
        label="Paris",
    )

    result = deterministic_reward(
        outcome,
        {"metric": "exact_token_f1", "exact_weight": 0.75},
    )

    assert result.value == pytest.approx(0.125)
    assert result.components["exact"] == 0.0
    assert result.components["token_f1"] == pytest.approx(0.5)
    assert result.components["exact_weight"] == 0.75

    with pytest.raises(ValueError, match="exact_weight"):
        deterministic_reward(
            outcome,
            {"metric": "exact_token_f1", "exact_weight": 1.5},
        )


def test_training_config_rejects_conflicting_debug_modes_and_unknown_keys():
    with pytest.raises(ValueError, match="mutually exclusive"):
        TrainingConfig.from_mapping(
            {
                "run_id": "test-run",
                "rollout_only": True,
                "train_only": True,
                "replay_path": "x",
            }
        )
    with pytest.raises(ValueError, match="Unknown training config keys"):
        TrainingConfig.from_mapping({"not_a_real_option": 1})
    with pytest.raises(ValueError, match="requires replay_path"):
        TrainingConfig.from_mapping({"run_id": "test-run", "train_only": True})
    with pytest.raises(ValueError, match="training_force_initial_delegate"):
        TrainingConfig.from_mapping(
            {
                "run_id": "test-run",
                "training_force_initial_delegate": "yes",
            }
        )


def test_experiment_matrix_keeps_a_as_the_plain_baseline():
    a = TrainingConfig.from_mapping({"run_id": "a"})
    assert a.experiment is Experiment.CENTRAL_BASELINE


def test_experiment_b_action_output_decoupling_is_fail_closed_without_evidence():
    with pytest.raises(
        (FileNotFoundError, ValueError), match="taxonomy.json|unreadable"
    ):
        TrainingConfig.from_mapping(
            {
                "run_id": "b",
                "experiment": "B",
                "mode": "shared-all-agents",
                "capability_decoupling": True,
                "decoupling_scope": "all-agent-action-output",
                "capability_taxonomy_path": "taxonomy.json",
                "capability_probe_manifest": "probe.json",
                "capability_layer_map": "map.json",
                "capability_annotator_function": "plugin.annotate",
                "capability_loss_function": "plugin.loss",
                "capability_gradient_router_function": "plugin.route",
            }
        )


def test_experiment_b_allows_only_explicit_rollout_bootstrap():
    config = TrainingConfig.from_mapping(
        {
            "run_id": "b-probe-bootstrap",
            "experiment": "B",
            "mode": "shared-all-agents",
            "capability_decoupling": True,
            "decoupling_scope": "all-agent-action-output",
            "capability_probe_bootstrap": True,
            "rollout_only": True,
            "capability_annotator_function": "slime_plugins.stackplanner.capability.all_agent_output.annotate_turn",
        }
    )
    assert config.capability_probe_bootstrap is True
    with pytest.raises(ValueError, match="rollout-only"):
        TrainingConfig.from_mapping(
            {
                **config.as_dict(),
                "rollout_only": False,
            }
        )


def test_experiment_b_cannot_silently_revert_to_old_non_decoupled_contract():
    with pytest.raises(ValueError, match="experiment B requires"):
        TrainingConfig.from_mapping(
            {
                "run_id": "old-b",
                "experiment": "B",
                "mode": "shared-all-agents",
                "capability_decoupling": False,
            }
        )


def test_experiment_c_is_blocked_without_readable_probe_evidence():
    with pytest.raises(ValueError, match="capability evidence is unreadable"):
        TrainingConfig.from_mapping(
            {
                "run_id": "c",
                "experiment": "C",
                "mode": "central-only",
                "capability_decoupling": True,
                "decoupling_scope": "central-capability",
                "capability_taxonomy_path": "taxonomy.json",
                "capability_probe_manifest": "probe.json",
                "capability_layer_map": "map.json",
                "capability_annotator_function": "plugin.annotate",
                "capability_loss_function": "plugin.loss",
                "capability_gradient_router_function": "plugin.route",
            }
        )

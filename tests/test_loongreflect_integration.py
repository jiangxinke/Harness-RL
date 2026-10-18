"""Exercise real Slime conversion/loss code without installing GPU runtimes.

Only singleton Megatron topology queries and external actor services are
substituted. The policy loss, custom-loss dispatcher, response offsets and
autograd kernels are the production functions.
"""

from __future__ import annotations

import ast
import copy
from contextlib import nullcontext
import importlib
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from slime.utils.types import Sample
from slime_plugins.loongreflect.launcher import build_command, parse_args, validate_data
from slime_plugins.loongreflect.runtime import (
    initialize_teacher,
    save_teacher,
    update_teacher,
    validate_runtime,
)
from slime_plugins.loongreflect.self_distill import (
    build_update_objectives,
    materialize_update_objective,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def singleton_loss(monkeypatch):
    core = ModuleType("megatron.core")
    core.mpu = SimpleNamespace(
        get_context_parallel_world_size=lambda: 1,
        get_context_parallel_rank=lambda: 0,
        get_tensor_model_parallel_rank=lambda: 0,
        get_tensor_model_parallel_group=lambda: None,
        get_data_parallel_world_size=lambda **kwargs: 1,
        is_pipeline_last_stage=lambda: True,
    )
    package = ModuleType("megatron")
    package.core = core
    monkeypatch.setitem(sys.modules, "megatron", package)
    monkeypatch.setitem(sys.modules, "megatron.core", core)
    names = (
        "slime.backends.megatron_utils.cp_utils",
        "slime.backends.megatron_utils.loss",
    )
    # Ensure tests neither reuse a previous fixture's topology nor leak one.
    for name in names:
        monkeypatch.delitem(sys.modules, name, raising=False)
    loss = importlib.import_module(names[-1])
    yield loss
    for name in names:
        sys.modules.pop(name, None)


def loss_args(mode):
    return SimpleNamespace(
        reflect_self_distill=True,
        reflect_train_mode=mode,
        reflect_fast_updates_per_step=2,
        reflect_fast_distill_weight=1.0,
        reflect_distill_weight=0.5,
        reflect_distill_kl_clip=10.0,
        rollout_temperature=1.0,
        rollout_top_p=1.0,
        log_probs_chunk_size=-1,
        allgather_cp=False,
        use_rollout_logprobs=False,
        use_opsm=False,
        advantage_estimator="grpo",
        eps_clip=0.2,
        eps_clip_high=0.28,
        get_mismatch_metrics=False,
        use_tis=False,
        entropy_coef=0.0,
        use_kl_loss=False,
        calculate_per_token_loss=False,
        loss_type="custom_loss",
        custom_loss_function_path="slime_plugins.loongreflect.self_distill.reflect_joint_loss",
        recompute_loss_function=False,
        kl_coef=0.0,
        normalize_advantages=False,
        custom_advantage_function_path="slime_plugins.loongreflect.self_distill.dense_tree_advantages",
        use_opd=False,
        compute_advantages_and_returns=True,
    )


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 32)
        layer = torch.nn.TransformerEncoderLayer(
            32, 4, 64, dropout=0.0, batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(
            layer, 1, enable_nested_tensor=False
        )
        self.output = torch.nn.Linear(32, 32)

    def forward(self, tokens):
        mask = torch.triu(
            torch.ones(
                tokens.shape[1], tokens.shape[1], device=tokens.device, dtype=torch.bool
            ),
            1,
        )
        return self.output(self.transformer(self.embedding(tokens), mask=mask))


@pytest.mark.parametrize(
    "mode,expected_steps",
    [("joint", 1), ("fast_slow", 3), ("policy_only", 1), ("distill_only", 1)],
)
def test_real_slime_custom_loss_optimizes_and_keeps_teacher_fixed(
    singleton_loss, mode, expected_steps
):
    torch.manual_seed(17)
    device = torch.device(os.environ.get("LOONGREFLECT_TEST_DEVICE", "cpu"))
    policy = TinyPolicy().to(device)
    teacher = copy.deepcopy(policy).eval()
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.002)
    args = loss_args(mode)
    tokens = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], device=device)
    teacher_tokens = torch.tensor([9, 10, 11, 12, 4, 5, 6, 7, 8], device=device)
    batch = dict(
        tokens=tokens[None],
        unconcat_tokens=[tokens],
        total_lengths=[8],
        response_lengths=[5],
        loss_masks=[torch.tensor([1, 1, 1, 1, 0], device=device)],
        reflect_masks=[torch.tensor([0, 1, 1, 0, 0], device=device)],
        rollout_mask_sums=[torch.tensor(4.0, device=device)],
        rollout_reflect_mask_sums=[torch.tensor(2.0, device=device)],
        rewards=[1.0],
        position_rewards=[torch.tensor([0.4, 0.4, 0.4, 0.4, 0], device=device)],
    )
    with torch.no_grad():
        batch["log_probs"] = singleton_loss.get_log_probs_and_entropy(
            policy(tokens[None]).float(),
            args=args,
            unconcat_tokens=[tokens],
            total_lengths=[8],
            response_lengths=[5],
        )[1]["log_probs"]
        batch["teacher_log_probs"] = singleton_loss.get_log_probs_and_entropy(
            teacher(teacher_tokens[None]).float(),
            args=args,
            unconcat_tokens=[teacher_tokens],
            total_lengths=[9],
            response_lengths=[5],
        )[1]["log_probs"]
    fixed_teacher = batch["teacher_log_probs"][0].clone()
    fixed_old = batch["log_probs"][0].clone()
    singleton_loss.compute_advantages_and_returns(args, batch)
    initial = {
        name: value.detach().clone() for name, value in policy.named_parameters()
    }
    objectives = build_update_objectives(args)
    assert len(objectives) == expected_steps
    for objective in objectives:
        materialize_update_objective(batch, objective)
        optimizer.zero_grad()
        logits = policy(tokens[None]).float()
        logits.retain_grad()
        loss, normalizer, reported = singleton_loss.loss_function(
            args, batch, 1, 1, logits
        )
        assert torch.isfinite(loss) and normalizer.item() == 1
        assert "reflect_distill_loss" in reported["keys"]
        loss.backward()
        assert all(
            torch.isfinite(p.grad).all()
            for p in policy.parameters()
            if p.grad is not None
        )
        # Prompt logits and synthetic controller suffix receive no objective.
        assert torch.count_nonzero(logits.grad[:, :2]) == 0
        assert torch.count_nonzero(logits.grad[:, 6:]) == 0
        if objective.policy_weight == 0:
            # Fast loss only reaches the two reflection token predictions.
            assert torch.count_nonzero(logits.grad[:, [2, 5]]) == 0
        optimizer.step()
        torch.testing.assert_close(
            batch["teacher_log_probs"][0], fixed_teacher, rtol=0, atol=0
        )
        torch.testing.assert_close(batch["log_probs"][0], fixed_old, rtol=0, atol=0)
    assert any(
        not torch.equal(initial[name], value)
        for name, value in policy.named_parameters()
    )
    assert all(
        int(state["step"]) == expected_steps for state in optimizer.state.values()
    )
    assert all(parameter.grad is None for parameter in teacher.parameters())


def _method(path, class_name, method_name, namespace):
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    method = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )
    method.decorator_list = []
    module = ast.Module(body=[method], type_ignores=[])
    exec(
        compile(
            ast.fix_missing_locations(module),
            str(path),
            "exec",
            flags=__import__("__future__").annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace[method_name]


def test_real_rollout_converter_preserves_tree_fields_and_rollout_denominators():
    method = _method(
        ROOT / "slime/slime/ray/rollout.py",
        "RolloutManager",
        "_convert_samples_to_train_data",
        dict(Sample=Sample, get_source=lambda sample: "unit"),
    )
    manager = SimpleNamespace(
        args=SimpleNamespace(rollout_top_p=1.0),
        custom_train_sample_transform_func=None,
        custom_convert_samples_to_train_data_func=None,
        _post_process_rewards=lambda samples: (
            [s.reward for s in samples],
            [s.reward for s in samples],
        ),
    )
    samples = [
        Sample(
            index=i,
            rollout_id=4,
            tokens=[1, 2, 3],
            response_length=2,
            reward=0.5,
            loss_mask=[1, 0],
            rollout_log_probs=[-0.3, 0],
            position_rewards=[0.5, 0],
            reflect_mask=[1, 0],
            teacher_tokens=[8, 9, 2, 3],
        )
        for i in range(2)
    ]
    for sample in samples:
        sample._validate_response_metadata_lengths()
    result = method(manager, samples)
    assert result["rollout_mask_sums"] == [2, 2]
    assert result["rollout_reflect_mask_sums"] == [2, 2]
    assert result["teacher_total_lengths"] == [4, 4]
    assert result["position_rewards"] == [[0.5, 0], [0.5, 0]]
    assert result["teacher_tokens"] == [[8, 9, 2, 3], [8, 9, 2, 3]]
    samples[1].teacher_tokens = None
    with pytest.raises(ValueError, match="every compact rollout sample"):
        method(manager, samples)


def test_ema_checkpoint_roundtrip_and_missing_resume_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)

    class Backuper:
        def __init__(self):
            self.state = {"actor": {"weight": torch.tensor([2.0])}}

        def backup(self, tag):
            self.state[tag] = copy.deepcopy(self.state["actor"])

        def get(self, tag):
            return self.state[tag]

        def ema_update(self, *, src_tag, dst_tag, tau):
            for name, value in self.state[dst_tag].items():
                value.mul_(tau).add_(self.state[src_tag][name], alpha=1 - tau)

    actor = SimpleNamespace(
        args=SimpleNamespace(
            load=str(tmp_path),
            ref_load=str(tmp_path),
            save=str(tmp_path),
            start_rollout_id=0,
            num_rollout=4,
            reflect_ema_tau_base=0.5,
            reflect_ema_tau_final=1.0,
        ),
        weights_backuper=Backuper(),
    )
    initialize_teacher(actor, -1)
    actor.weights_backuper.state["actor"]["weight"].fill_(4)
    update_teacher(actor, 0)
    assert actor.weights_backuper.get("ema_teacher")["weight"].item() == 3.0
    save_teacher(actor, 0)
    actor.weights_backuper.get("ema_teacher")["weight"].zero_()
    actor.args.start_rollout_id = 1
    initialize_teacher(actor, 0)
    assert actor.weights_backuper.get("ema_teacher")["weight"].item() == 3.0
    with pytest.raises(FileNotFoundError, match="EMA teacher"):
        initialize_teacher(actor, 3)


def test_launcher_uses_existing_megatron_without_reference_lora_flags(tmp_path):
    data = tmp_path / "train.jsonl"
    data.write_text('{"query":"q","answer":["a"],"metadata":{"split":"train"}}\n')
    argv = ["--data", str(data), "--mode", "fast_slow"]
    for name in (
        "hf-checkpoint",
        "megatron-checkpoint",
        "reference-checkpoint",
        "save-dir",
    ):
        argv.extend((f"--{name}", str(tmp_path / name)))
    args = parse_args(argv)
    command, config = build_command(args)
    assert validate_data(data) == 1
    assert config["reflect_train_mode"] == "fast_slow"
    assert config["reflect_self_distill"] is True
    assert "--peft-lora-rank" not in command
    assert (
        command[command.index("--custom-generate-function-path") + 1]
        == "slime_plugins.loongreflect.slime_adapter.generate"
    )
    data.write_text('{"query":"q","answer":"a","split":"test"}\n')
    with pytest.raises(ValueError, match="evaluation split"):
        validate_data(data)


def test_runtime_rejects_teacher_context_parallel_misalignment():
    with pytest.raises(ValueError, match="context-parallel"):
        validate_runtime(loss_args("joint"), {"cp_size": 2})


@pytest.mark.parametrize(
    "mode,weights", [("joint", [(1, 0.5)]), ("fast_slow", [(0, 1), (0, 1), (1, 0)])]
)
def test_real_actor_reuses_rollout_for_ordered_optimizer_updates(mode, weights):
    args = loss_args(mode)
    args.keep_old_actor = args.use_critic = args.use_routing_replay = (
        args.use_rollout_routing_replay
    ) = False
    args.ref_update_interval = None
    args.num_rollout = 10
    data = {
        "num_microbatches": [1],
        "global_batch_sizes": [1],
        "response_lengths": [2],
        "loss_masks": [torch.ones(2)],
        "teacher_tokens": [torch.tensor([8, 9, 2, 3])],
        "teacher_total_lengths": [4],
        "tokens": [torch.tensor([1, 2, 3])],
        "total_lengths": [3],
    }
    updates, switches = [], []
    teacher_probs = torch.tensor([-0.3, -0.4])
    old_probs = torch.tensor([-0.5, -0.6])

    def train(
        rollout_id,
        model,
        optimizer,
        scheduler,
        iterators,
        microbatches,
        batch_sizes,
        **kwargs,
    ):
        assert iterators[0] is data
        assert data["teacher_log_probs"][0] is teacher_probs
        assert data["log_probs"][0] is old_probs
        assert "teacher_tokens" not in data and "teacher_total_lengths" not in data
        updates.append(
            (
                data["reflect_policy_weights"][0].item(),
                data["reflect_distill_weights"][0].item(),
            )
        )

    namespace = dict(
        get_data_iterator=lambda data: [data],
        inverse_timer=lambda *a: nullcontext(),
        timer=lambda *a: nullcontext(),
        compute_advantages_and_returns=lambda args, data: None,
        log_rollout_data=lambda *a: None,
        train=train,
        train_dump_utils=SimpleNamespace(save_debug_train_data=lambda *a, **kw: None),
        log_perf_data=lambda *a, **kw: None,
    )
    method = _method(
        ROOT / "slime/slime/backends/megatron_utils/actor.py",
        "MegatronTrainRayActor",
        "train_actor",
        namespace,
    )
    backuper = SimpleNamespace(
        backup_tags=["actor", "ema_teacher"],
        backup=lambda tag: None,
        ema_update=lambda **kw: None,
    )
    actor = SimpleNamespace(
        args=args,
        model=None,
        optimizer=None,
        opt_param_scheduler=None,
        weights_backuper=backuper,
        _active_model_tag="actor",
        _switch_model=lambda tag: switches.append(tag),
        compute_teacher_log_prob=lambda *a: {"teacher_log_probs": [teacher_probs]},
        compute_log_prob=lambda *a, **kw: {"log_probs": [old_probs]},
        rollout_data_postprocess=__import__(
            "slime_plugins.loongreflect.self_distill",
            fromlist=["schedule_rollout_loss"],
        ).schedule_rollout_loss,
        prof=SimpleNamespace(step=lambda **kw: None),
        weight_updater=SimpleNamespace(pop_metrics=lambda: {}),
    )
    method(actor, 0, data)
    assert updates == weights
    assert switches[0] == "ema_teacher" and switches[1] == "actor"


def test_real_teacher_forward_scores_exact_response_under_private_context():
    data = dict(
        tokens=[torch.tensor([1, 2, 3])],
        total_lengths=[3],
        response_lengths=[2],
        teacher_tokens=[torch.tensor([8, 9, 2, 3])],
        teacher_total_lengths=[4],
    )

    def forward_only(function, args, model, iterators, num_microbatches, **kwargs):
        assert iterators[0]["tokens"] is data["teacher_tokens"]
        assert iterators[0]["total_lengths"] == [4]
        assert iterators[0]["response_lengths"] == [2]
        assert kwargs == {"store_prefix": "teacher_", "use_rollout_top_p_replay": False}
        return {"teacher_log_probs": [torch.tensor([-0.2, -0.3])]}

    method = _method(
        ROOT / "slime/slime/backends/megatron_utils/actor.py",
        "MegatronTrainRayActor",
        "compute_teacher_log_prob",
        dict(
            get_data_iterator=lambda data: [data],
            forward_only=forward_only,
            get_log_probs_and_entropy=None,
        ),
    )
    result = method(SimpleNamespace(args=None, model=None), data, [1])
    assert result["teacher_log_probs"][0].shape == (2,)
    assert data["total_lengths"] == [3]

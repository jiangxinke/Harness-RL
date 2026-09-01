from argparse import Namespace

import torch

from slime.backends.megatron_utils import data


def test_log_rollout_data_ignores_structured_train_metadata(monkeypatch):
    monkeypatch.setattr(data.mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(data.mpu, "is_pipeline_last_stage", lambda: True)
    monkeypatch.setattr(data.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        data.mpu,
        "get_data_parallel_world_size",
        lambda with_context_parallel=False: 1,
    )
    captured = {}

    def fake_gather(metric_name, args, rollout_id, log_dict):
        captured.update(log_dict)
        return {}

    monkeypatch.setattr(data, "gather_log_data", fake_gather)

    rollout_data = {
        "tokens": [torch.tensor([1, 2])],
        "loss_masks": [torch.tensor([1.0])],
        "response_lengths": [1],
        "total_lengths": [2],
        "global_batch_sizes": [1],
        "metadata": [{"capability": "action_choice"}],
        "rewards": [0.5],
    }

    data.log_rollout_data(
        0,
        Namespace(
            ci_test=False,
            log_multi_turn=False,
            log_passrate=False,
            log_correct_samples=False,
        ),
        rollout_data,
    )

    assert "metadata" not in captured
    assert captured["rewards"] == (0.5, 1)

import pytest

from slime_plugins.dcc.eval_sampling import EvaluationSampling


@pytest.mark.parametrize("kwargs", [
    {"temperature": -1}, {"temperature": float("inf")}, {"temperature": True},
    {"temperature": 2.1}, {"top_p": 0}, {"top_p": 1.1}, {"top_k": 0},
    {"top_k": True}, {"min_p": -0.1}, {"min_p": 1.1}, {"presence_penalty": 2.1},
    {"frequency_penalty": -2.1}, {"repetition_penalty": 0}, {"top_p": "0.8"},
])
def test_invalid_sampling_is_rejected_before_gpu_loading(kwargs):
    with pytest.raises(ValueError):
        EvaluationSampling(**kwargs)

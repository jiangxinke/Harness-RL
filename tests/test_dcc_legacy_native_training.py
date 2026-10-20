"""Original TRAIN declaration compatibility with tiny real CPU tensor headers."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from slime_plugins.dcc.legacy_native_training_configuration import native_configuration as saved_configuration
from slime_plugins.dcc.native_budget import NativeBudget, BUDGET_PROTOCOL, approved_training_receipt
from slime_plugins.dcc.native_lora import validate_native_adapter
from slime_plugins.dcc.native_thinking_eval import native_configuration

ORIGINAL_CONFIG = Path(__file__).with_name("fixtures") / "dcc_original_native_training_config.json"
ORIGINAL_SOURCES = ["76b57e227b943fb34a446226fa67cc1420f40e52", "cf89170a00eb7c25bf5de038558c479228a87587"]


def fixture(tmp_path, source):
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file
    adapter = tmp_path / "adapter"; adapter.mkdir()
    config = json.loads(ORIGINAL_CONFIG.read_text())
    training = config["native_thinking"]["source_receipt"]
    training["source_commit"] = source
    config["native_thinking"]["rollout_native_thinking"]["source_receipt"] = deepcopy(training)
    evaluation = dict(training, source_commit="f" * 40)
    approved = approved_training_receipt(evaluation, source, BUDGET_PROTOCOL)
    (adapter / "training_config.json").write_text(json.dumps(config))
    (adapter / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "r": 2, "target_modules": ["q_proj"]}))
    save_file({"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": torch.zeros(2, 4),
               "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(4, 2)},
              str(adapter / "adapter_model.safetensors"))
    return adapter, config, {"source_receipt": evaluation, "approved_derivative_id": "synthetic-approved",
        "model_config": {"hidden_size": 4, "num_hidden_layers": 1,
                         "num_attention_heads": 1, "num_key_value_heads": 1},
        "approved_training_source_receipt": approved,
        "approved_training_protocol": BUDGET_PROTOCOL, "evaluation_protocol": BUDGET_PROTOCOL}


@pytest.mark.parametrize("source", ORIGINAL_SOURCES)
def test_original_saved_train_config_passes_without_relabelling(tmp_path, source):
    adapter, config, options = fixture(tmp_path, source)
    metadata, binding = validate_native_adapter(adapter, **options)
    saved = config["native_thinking"]["rollout_native_thinking"]
    assert metadata["config"]["r"] == 2 and binding["checked_actual_attention_shapes"] is True
    assert binding["training_rollout_native_thinking"] == saved
    assert saved["stage2_original_schema"] is True
    assert saved_configuration(options["approved_training_source_receipt"],
        budget=NativeBudget(16384, "thinking")) == saved
    current = native_configuration(options["source_receipt"], budget=NativeBudget(16384, "thinking"))
    assert current["stage2_original_schema"] is False
    assert current["stage2_original_public_schema_postvalidation"] is True
    assert binding["caller_approved_training"]["source_commit"] == source


@pytest.mark.parametrize("damage", ["budget", "sampling", "physical_schema", "receipt", "derivative", "rank"])
def test_legacy_compatibility_keeps_complete_training_validation(tmp_path, damage):
    adapter, config, options = fixture(tmp_path, ORIGINAL_SOURCES[1])
    saved = config["native_thinking"]["rollout_native_thinking"]
    if damage == "budget": saved["thought_max_tokens"] = 8192
    elif damage == "sampling": saved["thought_sampling"]["temperature"] = .7
    elif damage == "physical_schema": saved["stage2_original_schema"] = False
    elif damage == "receipt": options["approved_training_source_receipt"]["source_commit"] = "e" * 40
    elif damage == "derivative": options["approved_derivative_id"] = "unapproved"
    else: config["lora_rank"] = 3
    (adapter / "training_config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError): validate_native_adapter(adapter, **options)


def test_unknown_training_source_cannot_claim_legacy_backend(tmp_path):
    adapter, config, options = fixture(tmp_path, "e" * 40)
    with pytest.raises(ValueError, match="TRAIN rollout"):
        validate_native_adapter(adapter, **options)

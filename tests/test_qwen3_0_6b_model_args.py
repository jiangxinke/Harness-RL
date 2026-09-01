from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG = ROOT / "tests/fixtures/qwen3_0_6b_config.json"
SCRIPT = ROOT / "slime/scripts/models/qwen3-0.6B.sh"


def _args() -> dict[str, str | bool]:
    text = SCRIPT.read_text(encoding="utf-8")
    tokens = re.findall(r'--[a-z0-9-]+|"[^"]*"|[0-9.eE+-]+', text)
    result: dict[str, str | bool] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            index += 1
            continue
        if index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            result[token] = tokens[index + 1].strip('"')
            index += 2
        else:
            result[token] = True
            index += 1
    return result


def test_qwen3_0_6b_megatron_arguments_match_staged_hf_config():
    config = json.loads(MODEL_CONFIG.read_text(encoding="utf-8"))
    args = _args()
    assert int(args["--num-layers"]) == config["num_hidden_layers"]
    assert int(args["--hidden-size"]) == config["hidden_size"]
    assert int(args["--ffn-hidden-size"]) == config["intermediate_size"]
    assert int(args["--num-attention-heads"]) == config["num_attention_heads"]
    assert int(args["--num-query-groups"]) == config["num_key_value_heads"]
    assert int(args["--kv-channels"]) == config["head_dim"]
    assert int(args["--vocab-size"]) == config["vocab_size"]
    assert int(args["--rotary-base"]) == config["rope_theta"]
    assert float(args["--norm-epsilon"]) == config["rms_norm_eps"]
    assert args["--normalization"] == "RMSNorm"
    assert args["--qk-layernorm"] is True

"""Explicit DPO adoption metadata; default native accepted-SFT stays separate.

The caller independently approves the complete public saved configuration.
Receipts are provenance assertions, not cryptographic producer attestation.
Validation reads JSON and safetensors headers only, never pretrained tensors.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import re
from ..dcc_train_literal_reward_contract import (
    PROTOCOL as LITERAL_PROTOCOL, LITERAL_RECEIPT, LITERAL_DPO_COLLECTIONS, PER_COLLECTION_RECEIPTS,
    LITERAL_PROFILE_FIELDS, validate_approval,
)

DPO_APPROVAL = "native-final-dpo-adapter-approval-v1"
DPO_METHOD = "offline native final-decision DPO; not accepted-trajectory SFT; not RL"
DPO_ALGORITHM = "offline_native_final_DPO; not RL"
PAIR_CONTRACT = "native-final-best-worst-strict-gap-v1"
SCORING_VERSION = "raw-causal-full-completion-token-loglikelihood-sum-v1"
QUANTIZATION_ID = "qwen3-moe-nf4-double-quant-bf16-compute-transformers-4.57.6-v1"
COLLECTION_PROFILE_KEYS = {"collection_id", "experiment_id", "harness_id", "schema_policy",
    "collection_source_commit", "reward_version", "collection_native_identity", "collection_source_receipt",
    "generation_world_size", "generation_closed_before_scoring", "attempt_count", "valid_candidates",
    "eligible_pairs", "omitted_prefixes", "pair_selection", "training_sources", "frozen_prefix_ids",
    "frozen_task_ids", "attempt_trace_ids", "phase_request_ids", "selected_pairs", "public_freeze"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_dpo_approval(value):
    """Do not infer DPO approval from an adapter or a training config alone."""
    from .native_thinking_eval import MODEL_REVISION, validate_source_receipt
    from .native_budget import BUDGET_PROTOCOL
    require(isinstance(value, dict) and set(value) == {"version", "training_configuration", "checkpoint_optimizer_steps"}
        and value["version"] == DPO_APPROVAL, "Explicit independently approved DPO profile required")
    cfg = value["training_configuration"]
    require(isinstance(cfg, dict) and "native_thinking" not in cfg
        and cfg.get("training_method") == DPO_METHOD and cfg.get("algorithm") == DPO_ALGORITHM
        and cfg.get("fresh_run") is True and cfg.get("warm_start") is False
        and cfg.get("resume_supported") is False and cfg.get("reference_models_loaded") == 0
        and cfg.get("measured_KL") is False and cfg.get("gradient_weighting") == "global-equal-pair"
        and cfg.get("sequence_score") == "sum-all-actual-native-thought-close-JSON-EOS-token-logprobs"
        and cfg.get("candidate_contract") == "native-final-preference-candidate-v1"
        and cfg.get("reward_contract") in {"native-final-train-reward-receipt-v1", PER_COLLECTION_RECEIPTS}
        and cfg.get("pair_contract") == PAIR_CONTRACT and cfg.get("minimum_reward_gap") == .05
        and cfg.get("pair_selection") == "one-best-worst-per-frozen-prefix; reward-gap-strictly-greater-than-0.05",
        "Saved configuration is not the approved fresh raw-sum native final DPO method")
    require(type(cfg.get("steps")) is int and cfg["steps"] > 0
        and type(cfg.get("pairs_per_step")) is int and cfg["pairs_per_step"] > 0
        and cfg.get("lora_rank") == 16 and cfg.get("lora_alpha") == 32
        and cfg.get("lora_target_modules") == ["q_proj", "v_proj", "o_proj"]
        and cfg.get("lora_dropout") == 0 and cfg.get("weight_decay") == 0
        and cfg.get("max_grad_norm") == 1.0
        and all(type(cfg.get(key)) in (int, float) and math.isfinite(cfg[key]) and cfg[key] > 0
                for key in ("beta", "learning_rate")), "Invalid approved DPO optimizer/LoRA configuration")
    source = cfg.get("training_source")
    require(isinstance(source, dict) and set(source) == {"version", "source_commit", "tracked_runtime_clean", "training_entrypoint"}
        and source["version"] == "dcc-offline-native-dpo-training-source-v1"
        and isinstance(source["source_commit"], str) and re.fullmatch(r"[0-9a-f]{40}", source["source_commit"])
        and source["tracked_runtime_clean"] is True and source["training_entrypoint"] == "dcc_native_dpo_train.py",
        "Actual clean committed DPO training runtime identity required")
    require(type(value.get("checkpoint_optimizer_steps")) is int
        and 1 <= value["checkpoint_optimizer_steps"] <= cfg["steps"],
        "Explicit actual checkpoint optimizer steps within the configured total required")
    if "pair_sampling" in cfg:
        from .native_dpo_sampling import balanced_pair_sampling_configuration
        require(cfg["pair_sampling"] == balanced_pair_sampling_configuration(
            seed=cfg.get("seed"), global_draws=cfg["pairs_per_step"]),
            "Explicit balanced DPO sampling configuration differs from its optimizer batch")
        require(set(cfg.get("training_sources", [])) == {"cuad_train", "independent_curriculum"},
                "Two-domain balanced DPO requires both approved TRAIN domains")
    profiles = cfg.get("collections")
    require(cfg.get("collection_merge_contract") in {"native-final-dpo-collections-v1", LITERAL_DPO_COLLECTIONS}
        and isinstance(profiles, list) and bool(profiles), "Explicit approved collection merge contract required")
    literal_merge = cfg["collection_merge_contract"] == LITERAL_DPO_COLLECTIONS
    require(cfg["reward_contract"] == (PER_COLLECTION_RECEIPTS if literal_merge
        else "native-final-train-reward-receipt-v1"), "Reward receipt contract differs from approved collection mode")
    literal_profiles = 0
    seen = {key: set() for key in ("collection_id", "frozen_prefix_ids", "frozen_task_ids", "attempt_trace_ids", "phase_request_ids")}
    common = None
    for profile in profiles:
        literal = isinstance(profile, dict) and profile.get("reward_version") == LITERAL_PROTOCOL
        require(isinstance(profile, dict) and set(profile) == (COLLECTION_PROFILE_KEYS | LITERAL_PROFILE_FIELDS
            if literal else COLLECTION_PROFILE_KEYS) and (not literal or literal_merge)
            and isinstance(profile.get("collection_id"), str)
            and profile["collection_id"] not in seen["collection_id"], "Distinct approved collection identity required")
        if literal:
            validate_approval(profile["train_reward"], expected_protocol=profile["reward_version"])
            require(profile["reward_contract"] == LITERAL_RECEIPT
                and profile["train_metric"] == "train_exact_row_f1"
                and profile["training_sources"] == ["independent_curriculum"],
                "Explicit WideSearch TRAIN literal reward profile required")
            literal_profiles += 1
        seen["collection_id"].add(profile["collection_id"])
        for field in ("frozen_prefix_ids", "frozen_task_ids", "attempt_trace_ids", "phase_request_ids"):
            values = profile.get(field)
            require(isinstance(values, list) and bool(values)
                and all(isinstance(item, str) and bool(item) for item in values)
                and len(set(values)) == len(values) and not seen[field].intersection(values),
                "Duplicated or missing approved collection public identities")
            seen[field].update(values)
        actual = _validate_collection_configuration(dict(cfg, **profile))
        freeze = profile["public_freeze"]
        require(isinstance(freeze, dict) and freeze.get("labels_loaded") is False
            and freeze.get("native_identity") == profile["collection_native_identity"]
            and freeze.get("source_receipt") == profile["collection_source_receipt"]
            and freeze.get("prefix_ids") == profile["frozen_prefix_ids"]
            and freeze.get("task_ids") == profile["frozen_task_ids"]
            and all(freeze.get(key) == profile[key] for key in ("experiment_id", "harness_id", "schema_policy"))
            and len(profile["frozen_prefix_ids"]) == len(profile["frozen_task_ids"])
            and profile["attempt_count"] == len(profile["attempt_trace_ids"]) == 4 * len(profile["frozen_prefix_ids"]),
            "Complete per-collection public freeze/attempt provenance required")
        selected = profile["selected_pairs"]
        require(isinstance(selected, list) and len(selected) == profile["eligible_pairs"]
            and all(isinstance(p, dict) for p in selected)
            and len({p.get("prefix_id") for p in selected}) == len(selected)
            and all(isinstance(p, dict) and set(p) == {"prefix_id", "chosen_trace_id", "rejected_trace_id",
                "chosen_seed", "rejected_seed"}
                and p["prefix_id"] in profile["frozen_prefix_ids"]
                and p["chosen_trace_id"] in profile["attempt_trace_ids"]
                and p["rejected_trace_id"] in profile["attempt_trace_ids"]
                and p["chosen_trace_id"] != p["rejected_trace_id"]
                and type(p["chosen_seed"]) is int and type(p["rejected_seed"]) is int
                and p["chosen_seed"] != p["rejected_seed"] for p in selected),
            "Actual selected pair ownership differs from its approved collection")
        require(common is None or actual == common, "Merged collections differ in base/tokenizer/template/protocol")
        common = actual
    require(not literal_merge or literal_profiles > 0, "New merge contract must retain complete new TRAIN reward profiles")
    require(cfg.get("generation_closed_before_scoring") is True
        and cfg.get("eligible_pairs") == sum(p["eligible_pairs"] for p in profiles)
        and cfg.get("valid_candidates") == sum(p["valid_candidates"] for p in profiles)
        and cfg.get("attempt_count") == sum(p["attempt_count"] for p in profiles)
        and cfg.get("training_sources") == sorted({source for p in profiles for source in p["training_sources"]}),
        "Merged TRAIN pair/closure counts differ from approved per-collection provenance")
    return copy.deepcopy(value)


def approval_from_arguments(args):
    path = getattr(args, "native_lora_dpo_approval", None)
    if path is None:
        return None
    from .native_thinking_eval import _strict_json
    return validate_dpo_approval(_strict_json(Path(path).read_text(encoding="utf-8")))


def dpo_binding(approval, *, derivative_id, source_receipt, evaluation_protocol, tensor_pairs):
    from .native_budget import BUDGET_PROTOCOL
    from .native_thinking_eval import MODEL_REVISION, validate_source_receipt
    approval = validate_dpo_approval(approval)
    cfg = approval["training_configuration"]
    require(evaluation_protocol == BUDGET_PROTOCOL, "DPO adoption requires explicit native v3 evaluation")
    require(derivative_id == cfg["reference"]["base_derivative_id"], "DPO derivative differs from the approved base")
    require(type(tensor_pairs) is int and tensor_pairs > 0, "Actual DPO attention tensor pairs required")
    return {"derivative_id": derivative_id, "training_method": DPO_METHOD,
        "training_source": cfg["training_source"], "collection_source_receipts": [{"collection_id": p["collection_id"],
            "source_receipt": p["collection_source_receipt"]} for p in cfg["collections"]],
        "checkpoint_optimizer_steps": approval["checkpoint_optimizer_steps"],
        "model_revision": MODEL_REVISION, "lora_rank": 16, "tensor_pairs": tensor_pairs,
        "checked_actual_attention_shapes": True, "evaluation_source_receipt": validate_source_receipt(source_receipt),
        "evaluation_protocol": evaluation_protocol, "caller_approved_dpo": approval}


def validate_dpo_sidecar_binding(binding, approval, *, derivative_id, source_receipt, evaluation_protocol):
    require(isinstance(binding, dict), "Actual explicit DPO adapter binding required")
    expected = dpo_binding(approval, derivative_id=derivative_id, source_receipt=source_receipt,
                           evaluation_protocol=evaluation_protocol, tensor_pairs=binding.get("tensor_pairs"))
    require(binding == expected, "DPO native sidecar binding differs from the independently approved training")
    return expected


def validate_native_dpo_adapter(adapter, *, source_receipt, approved_derivative_id, model_config,
                                evaluation_protocol, approved_dpo):
    from .native_lora import validate_native_attention_headers
    from .native_thinking_eval import _strict_json
    from .vllm_policy import _adapter_metadata
    approval = validate_dpo_approval(approved_dpo)
    metadata = _adapter_metadata(adapter)
    require(metadata is not None, "Actual local DPO adapter required")
    path, cfg = Path(metadata["path"]), approval["training_configuration"]
    require(_strict_json((path / "training_config.json").read_text()) == cfg,
            "Actual saved DPO training configuration differs from caller approval")
    adapter_config = metadata["config"]
    require(adapter_config["r"] == 16 and adapter_config.get("lora_alpha") == 32
        and set(adapter_config["target_modules"]) == {"q_proj", "v_proj", "o_proj"}
        and adapter_config.get("lora_dropout", 0) == 0,
        "Actual DPO adapter must be standard rank16/alpha32 q/v/o LoRA")
    marker = _strict_json((path / "checkpoint_complete.json").read_text())
    require(marker.get("complete") is True and marker.get("optimizer_steps") == approval["checkpoint_optimizer_steps"],
            "Actual DPO checkpoint must be complete for its approved actual optimizer step")
    rank, tensor_pairs = validate_native_attention_headers(path, metadata, model_config)
    require(rank == 16 and _adapter_metadata(adapter) == metadata, "DPO adapter changed during header validation")
    return metadata, dpo_binding(approval, derivative_id=approved_derivative_id, source_receipt=source_receipt,
                                evaluation_protocol=evaluation_protocol, tensor_pairs=tensor_pairs)


def _validate_collection_configuration(cfg):
    from .native_thinking_eval import MODEL_REVISION, validate_source_receipt
    from .native_budget import BUDGET_PROTOCOL
    collection = cfg.get("collection_native_identity")
    receipt = validate_source_receipt(cfg.get("collection_source_receipt"))
    require(isinstance(collection, dict) and "generation_seed" not in collection
        and collection.get("source_commit") == receipt["source_commit"] == cfg.get("collection_source_commit")
        and collection.get("model_revision") == MODEL_REVISION
        and collection.get("protocol") == BUDGET_PROTOCOL and collection.get("thought_sampling_profile") == "thinking"
        and collection.get("thought_tokens") in {4096, 8192, 16384}
        and type(collection.get("max_prompt_tokens")) is int and collection["max_prompt_tokens"] > 0
        and type(collection.get("max_model_len")) is int
        and collection["max_model_len"] >= collection["max_prompt_tokens"] + collection["thought_tokens"] + 4096,
        "Separate actual native collection identity/source/budget required")
    reference = cfg.get("reference")
    require(isinstance(reference, dict) and set(reference) == {"base_derivative_id", "tokenizer_id", "quantization_id",
        "mode", "adapter_snapshot_id", "scoring_version"} and reference["mode"] == "base"
        and reference["adapter_snapshot_id"] is None and reference["scoring_version"] == SCORING_VERSION
        and reference["base_derivative_id"] == collection.get("derivative_id")
        and reference["quantization_id"] == QUANTIZATION_ID
        and isinstance(reference["tokenizer_id"], str) and bool(reference["tokenizer_id"].strip()),
        "Same actual NF4 base reference and explicit tokenizer identity required")
    report = cfg.get("quantization")
    require(isinstance(report, dict) and report.get("mode") == "nf4"
        and report.get("transformers_version") == "4.57.6" and report.get("expert_4bit_coverage") == 1.
        and report.get("expert_parameters", 0) > 0
        and report.get("expert_parameters") == report.get("expert_4bit_parameters")
        and report.get("expert_4bit_matrices", 0) > 0, "Actual complete NF4 expert coverage required")
    require(cfg.get("generation_closed_before_scoring") is True
        and type(cfg.get("eligible_pairs")) is int and cfg["eligible_pairs"] > 0
        and type(cfg.get("valid_candidates")) is int and cfg["valid_candidates"] >= 2 * cfg["eligible_pairs"]
        and type(cfg.get("attempt_count")) is int and cfg["attempt_count"] >= cfg["valid_candidates"]
        and cfg.get("schema_policy") in {"original-final-only", "final-decision-only-v1"}
        and isinstance(cfg.get("training_sources"), list) and bool(cfg["training_sources"])
        and set(cfg["training_sources"]) <= {"cuad_train", "independent_curriculum"}
        and all(isinstance(cfg.get(key), str) and bool(cfg[key].strip())
                for key in ("reward_version", "harness_id", "experiment_id")),
        "Closed TRAIN-only pair/reward/source provenance required")
    return {key: value for key, value in collection.items() if key not in {
        "source_commit", "max_prompt_tokens", "max_model_len", "thought_tokens",
        "thought_sampling_profile", "prompt_history"}}

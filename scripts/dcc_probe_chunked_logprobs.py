#!/usr/bin/env python3
"""Explicit short, single-GPU NF4 scorer comparison; never trains/evaluates.

Each backend loads the same real checkpoint in an isolated child. Controlled
nonzero rank-16 q/v/o adapters exercise both A and B gradients. This is a
likelihood/gradient probe, not a production optimizer or benchmark result.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parent),
                str(Path(__file__).resolve().parents[1] / "slime")]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Positive integer required")
    return value


def source_identity(expected):
    # Reuse the production clean-source check without weakening its gates.
    from dcc_native_dpo_train import training_source_identity
    return dict(training_source_identity(expected),
                probe_entrypoint="dcc_probe_chunked_logprobs.py")


def child(args):
    import gc
    import torch
    from safetensors.torch import save_file
    from slime_plugins.dcc.hf_policy import load_policy_model
    from slime_plugins.dcc.hf_chunked_logprobs import resolve_logprob_backend
    from dcc_native_dpo_train import validate_fresh_nf4_lora

    source = source_identity(args.expected_source_commit)
    if torch.cuda.device_count() != 1:
        raise ValueError("Reserve and expose exactly one GPU before invoking this probe")
    torch.manual_seed(args.seed)
    model, tokenizer = load_policy_model(args.model, trainable=True, lora_rank=16,
                                         rank=0, quantization="nf4")
    quantization = validate_fresh_nf4_lora(model)
    if model.config.model_type != "qwen3_moe":
        raise ValueError("Actual Qwen3 MoE checkpoint required")
    scorer, recipe = resolve_logprob_backend(args.child,
        args.head_chunk_tokens if args.child == "qwen3-moe-chunked" else None)

    expected = {f"base_model.model.model.layers.{layer}.self_attn.{projection}.lora_{matrix}.default.weight"
        for layer in range(model.config.num_hidden_layers)
        for projection in ("q_proj", "v_proj", "o_proj") for matrix in ("A", "B")}
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if set(trainable) != expected:
        raise ValueError("Trainable keys differ from all actual-layer q/v/o A/B keys")
    # A local CPU generator gives exactly reproducible adapters across children,
    # independent of loader RNG consumption. Nonzero B exercises A gradients.
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    initial = {}
    with torch.no_grad():
        for name, parameter in sorted(trainable.items()):
            value = torch.randn(tuple(parameter.shape), generator=generator) * .01
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))
            initial[name] = parameter.detach().float().cpu().contiguous()
    output = Path(args.output_dir) / args.child
    save_file(initial, str(output / "adapter_initial.safetensors"))
    del initial

    def tokens(text, length):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            raise ValueError("Tokenizer returned an empty controlled probe text")
        return (ids * ((length + len(ids) - 1) // len(ids)))[:length]
    prompt = tokens("Read this public source and identify the relevant contract clause.\n", args.prompt_tokens)
    response = tokens("The clause applies to the stated agreement and its conditions.\n", args.response_tokens)
    model.eval()
    with model.disable_adapter(), torch.no_grad():
        reference = scorer(model, prompt, response).detach().float().cpu()
    if not torch.isfinite(reference).all():
        raise FloatingPointError("Non-finite actual NF4 disabled-adapter reference")
    model.train()  # Enable the same decoder checkpointing used by DPO replay.
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    started = time.monotonic()
    values = scorer(model, prompt, response)
    loss = -values.sum()  # Same full-completion SUM semantics as the trainer.
    loss.backward()
    torch.cuda.synchronize()
    seconds = time.monotonic() - started
    peak_allocated, peak_reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    gradients = {name: p.grad.detach().float().cpu().contiguous()
        for name, p in trainable.items() if p.grad is not None}
    if (set(gradients) != expected
            or any(not torch.isfinite(g).all() for g in gradients.values())
            or any(not torch.count_nonzero(g) for g in gradients.values())):
        raise FloatingPointError("Missing, zero, or non-finite actual q/v/o adapter gradients")
    if not torch.isfinite(values).all() or not torch.isfinite(loss):
        raise FloatingPointError("Non-finite actual NF4 score/loss")
    save_file(gradients, str(output / "gradients.safetensors"))
    write_json(output / "result.actual.json", {
        "schema": "dcc-actual-nf4-chunked-scorer-child-v1", "backend": args.child,
        "source": source, "scoring_backend": recipe, "quantization": quantization,
        "real_checkpoint_loaded": True, "controlled_probe_tokens": True,
        "controlled_nonzero_adapter_initialization": True, "adapter_seed": args.seed,
        "lora_rank": 16, "lora_alpha": 32, "model_type": model.config.model_type,
        "decoder_layers": model.config.num_hidden_layers, "gradient_checkpointing": True,
        "backward_model_training_mode": model.training,
        "model_argument_recorded": False, "prompt_ids": prompt, "response_ids": response,
        "reference_scores": reference.tolist(), "scores": values.detach().float().cpu().tolist(),
        "loss": float(loss.detach().cpu()), "loss_semantics": "negative-full-response-raw-logprob-SUM",
        "expected_gradient_shapes": {name: list(p.shape) for name, p in sorted(trainable.items())},
        "gradient_nonzero_elements": {name: int(torch.count_nonzero(g)) for name, g in sorted(gradients.items())},
        "allocated_before": before, "peak_allocated": peak_allocated, "peak_reserved": peak_reserved,
        "forward_backward_seconds": seconds, "device": torch.cuda.get_device_name(),
        "optimizer_step_validated": False, "benchmark_improvement_claimed": False})


def compare(directory, *, score_atol=.02, gradient_rtol=.05, gradient_atol=1e-5):
    """CPU-only full-key/tensor comparison after both independent children exit."""
    import torch
    from safetensors.torch import load_file

    directory = Path(directory)
    left, right = (directory / name for name in ("original", "qwen3-moe-chunked"))
    a, b = (json.loads((p / "result.actual.json").read_text()) for p in (left, right))
    if (a.get("backend") != "original" or b.get("backend") != "qwen3-moe-chunked"
            or any(row.get("real_checkpoint_loaded") is not True
                or row.get("quantization", {}).get("mode") != "nf4"
                or row.get("quantization", {}).get("expert_4bit_coverage") != 1.
                for row in (a, b))):
        raise ValueError("Both explicit actual complete-NF4 backend results required")
    for key in ("source", "quantization", "prompt_ids", "response_ids", "expected_gradient_shapes",
                "adapter_seed", "decoder_layers", "lora_rank", "lora_alpha"):
        if a[key] != b[key]:
            raise ValueError("Independent child identity/input/layout mismatch: " + key)
    initial_a, initial_b = (load_file(str(p / "adapter_initial.safetensors"), device="cpu") for p in (left, right))
    grads_a, grads_b = (load_file(str(p / "gradients.safetensors"), device="cpu") for p in (left, right))
    expected = set(a["expected_gradient_shapes"])
    if not expected or any(set(value) != expected for value in (initial_a, initial_b, grads_a, grads_b)):
        raise ValueError("Missing or extra initialization/gradient keys")
    gradient_rows = []
    for name in sorted(expected):
        if not torch.equal(initial_a[name], initial_b[name]):
            raise ValueError("Independent adapter initialization differs: " + name)
        x, y = grads_a[name], grads_b[name]
        if list(x.shape) != a["expected_gradient_shapes"][name] or x.shape != y.shape or not x.numel():
            raise ValueError("Gradient shape/size mismatch: " + name)
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise FloatingPointError("Non-finite gradient: " + name)
        if not torch.count_nonzero(x) or not torch.count_nonzero(y):
            raise ValueError("Controlled nonzero adapters must exercise every gradient key: " + name)
        torch.testing.assert_close(x, y, rtol=gradient_rtol, atol=gradient_atol)
        diff = (x.double() - y.double())
        relative_l2 = float(diff.norm() / x.double().norm().clamp_min(1e-30))
        if relative_l2 > gradient_rtol:
            raise ValueError("Gradient relative L2 error exceeds tolerance: " + name)
        gradient_rows.append({"key": name, "numel": x.numel(),
            "baseline_nonzero": int(torch.count_nonzero(x)), "chunked_nonzero": int(torch.count_nonzero(y)),
            "max_abs_difference": float(diff.abs().max()), "relative_l2_difference": relative_l2})
    errors = {}
    for field in ("scores", "reference_scores"):
        x, y = torch.tensor(a[field], dtype=torch.float64), torch.tensor(b[field], dtype=torch.float64)
        if x.shape != y.shape or x.numel() != len(a["response_ids"]):
            raise ValueError("Per-token score length mismatch")
        torch.testing.assert_close(x, y, rtol=0, atol=score_atol)
        errors[field + "_max_abs_difference"] = float((x-y).abs().max())
    loss_difference = abs(a["loss"] - b["loss"])
    if not math.isfinite(loss_difference) or loss_difference > score_atol * len(a["response_ids"]):
        raise ValueError("Raw-SUM loss difference exceeds bounded per-token tolerance")
    return {"schema": "dcc-actual-nf4-chunked-scorer-comparison-v1", "source": a["source"],
        "scope": "short controlled values/gradients; not optimizer-step or benchmark validation",
        "actual_NF4_value_gradient_probe_passed": True,
        "baseline_scoring_backend": a["scoring_backend"], "chunked_scoring_backend": b["scoring_backend"],
        "adapter_initialization_exactly_equal": True, "expected_gradient_key_count": len(expected),
        "gradient_element_count": sum(row["numel"] for row in gradient_rows), "gradient_keys": gradient_rows,
        "gradient_max_abs_difference": max(row["max_abs_difference"] for row in gradient_rows),
        "gradient_max_relative_l2_difference": max(row["relative_l2_difference"] for row in gradient_rows),
        "score_atol": score_atol, "gradient_rtol_and_relative_l2_limit": gradient_rtol,
        "gradient_atol": gradient_atol, **errors,
        "baseline_loss": a["loss"], "chunked_loss": b["loss"], "loss_abs_difference": loss_difference,
        "baseline_allocated_before": a["allocated_before"], "chunked_allocated_before": b["allocated_before"],
        "baseline_peak_allocated": a["peak_allocated"], "chunked_peak_allocated": b["peak_allocated"],
        "baseline_peak_reserved": a["peak_reserved"], "chunked_peak_reserved": b["peak_reserved"],
        "baseline_seconds": a["forward_backward_seconds"], "chunked_seconds": b["forward_backward_seconds"],
        "memory_reduction_fraction": 1-b["peak_allocated"]/a["peak_allocated"],
        "optimizer_step_validated": False, "benchmark_improvement_claimed": False}


def run(args):
    source = source_identity(args.expected_source_commit)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "probe.started.json", {"source": source, "children": ["original", "qwen3-moe-chunked"],
        "synthetic_probe_only": False, "GPU_requested_by_this_wrapper": 1})
    try:
        for backend in ("original", "qwen3-moe-chunked"):
            directory = output / backend
            directory.mkdir()
            argv = [sys.executable, "-B", str(Path(__file__).resolve()), *sys.argv[1:], "--child", backend]
            with (directory / "stdout.log").open("x") as stdout, (directory / "stderr.log").open("x") as stderr:
                proc = subprocess.run(argv, stdout=stdout, stderr=stderr,
                    env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), check=False)
            write_json(directory / "terminal.actual.json", {"child": backend, "actual_exit_code": proc.returncode})
            if proc.returncode:
                raise RuntimeError("Actual NF4 probe child failed: " + backend)
        summary = compare(output)
        write_json(output / "comparison.actual.json", summary)
        write_json(output / "probe.terminal.json", {"completed": True, "actual_exit_code": 0})
        print(json.dumps(summary, allow_nan=False))
    except BaseException as error:
        write_json(output / "probe.terminal.json", {"completed": False, "actual_exit_code": 1,
                                                    "exception_type": type(error).__name__})
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", required=True, help="Explicit actual Qwen3 MoE checkpoint")
    result.add_argument("--output-dir", required=True, help="New durable artifact directory")
    result.add_argument("--expected-source-commit", required=True, help="Actual new clean committed source")
    result.add_argument("--prompt-tokens", type=positive_int, default=64)
    result.add_argument("--response-tokens", type=positive_int, default=64)
    result.add_argument("--head-chunk-tokens", type=positive_int, default=16)
    result.add_argument("--seed", type=int, default=59)
    result.add_argument("--child", choices=("original", "qwen3-moe-chunked"), help=argparse.SUPPRESS)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    child(arguments) if arguments.child else run(arguments)

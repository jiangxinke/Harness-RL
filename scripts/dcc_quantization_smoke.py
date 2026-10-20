#!/usr/bin/env python3
"""Measure opt-in NF4 model loading, full prefill and one synthetic LoRA step.

Run in the separate quantized environment on one explicitly selected GPU.
All durable results and adapter files go into --output-dir. This checks runtime
correctness and memory, not benchmark quality; it never reads benchmark labels.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "slime"))

import torch
from slime_plugins.dcc.hf_policy import completion_logprobs, load_policy_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--train-step", action="store_true")
    parser.add_argument("--training-prompt-tokens", type=int, default=256)
    parser.add_argument("--lora-rank", type=int, default=8)
    args = parser.parse_args()
    if min(args.prompt_tokens, args.max_new_tokens, args.training_prompt_tokens, args.lora_rank) < 1:
        parser.error("token budgets must be positive")
    destination = Path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    model, tokenizer = load_policy_model(args.model, quantization="nf4", trainable=args.train_step,
                                        lora_rank=args.lora_rank)

    def memory():
        torch.cuda.synchronize()
        return {
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }

    report = {
        "quantization": model.dcc_quantization_report,
        "lora_rank": args.lora_rank if args.train_step else None,
        "load_seconds": time.monotonic() - start,
        "load_memory": memory(),
    }

    def write_report():
        (destination / "smoke.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    write_report()
    # Exactly the requested length is passed to the model: no truncation or
    # silent shortening can make a long-prefill memory test appear to pass.
    seed_tokens = tokenizer.encode("Read this synthetic context carefully. ", add_special_tokens=False)
    prompt = (seed_tokens * ((args.prompt_tokens + len(seed_tokens) - 1) // len(seed_tokens)))[: args.prompt_tokens]
    ids = torch.tensor([prompt], device=model.device)
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    begin = time.monotonic()
    with torch.inference_mode():
        generated = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    report["generation"] = {
        "prompt_tokens": len(prompt),
        "completion_tokens": generated.shape[1] - len(prompt),
        "seconds": time.monotonic() - begin,
        "memory": memory(),
        "text": tokenizer.decode(generated[0, len(prompt) :], skip_special_tokens=True),
    }
    write_report()
    del ids, generated
    if args.train_step:
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not parameters or any("lora_" not in name for name, p in model.named_parameters() if p.requires_grad):
            raise RuntimeError("Only LoRA parameters may be trainable")
        optimizer = torch.optim.AdamW(parameters, lr=2e-5)
        model.train()
        completion = tokenizer.encode('{"answer": []}', add_special_tokens=False) + [tokenizer.eos_token_id]
        torch.cuda.reset_peak_memory_stats()
        begin = time.monotonic()
        loss = -completion_logprobs(model, prompt[: args.training_prompt_tokens], completion).mean()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(loss) or not torch.isfinite(norm) or norm <= 0:
            raise RuntimeError("Expected finite loss and a nonzero finite adapter gradient")
        optimizer.step()
        report["training_step"] = {
            "prompt_tokens": min(len(prompt), args.training_prompt_tokens),
            "loss": loss.item(),
            "gradient_norm": norm.item(),
            "trainable_parameters": sum(p.numel() for p in parameters),
            "seconds": time.monotonic() - begin,
            "memory": memory(),
        }
        model.save_pretrained(destination / "adapter")
    write_report()
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Optional replicated LoRA SFT + on-policy group-relative DCC training.

Launch with torchrun. Frozen model weights stay local to each GPU; only LoRA
gradients are averaged, avoiding synchronizing the 30B frozen base model.
All output locations are explicit arguments; no server defaults are embedded.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "slime"))

import torch
import torch.distributed as dist
from slime_plugins.dcc import EpisodeRunner, load_tasks
from slime_plugins.dcc.hf_policy import (
    HFPolicy,
    completion_logprobs,
    encode_messages,
    load_policy_model,
)


def grpo_loss(logprobs, old_logprobs, advantage: float, *, clip: float = 0.2):
    ratio = (logprobs - old_logprobs).exp()
    return -torch.minimum(
        ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage
    ).mean()


def average_gradients(parameters, world: int):
    if world == 1:
        return
    # Every rank participates even if its rollout had a search outage or
    # zero within-group reward variance.
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(world)


def supervised_example(task, runner, tokenizer, max_prompt: int, max_response: int):
    # Web curricula may have structured row labels; CUAD's verbatim span
    # auxiliary is inapplicable to them. They still participate in RL.
    if task.benchmark != "cuad":
        return None
    messages = runner.initial_messages(task)
    prompt = encode_messages(tokenizer, messages, max_prompt)
    visible = tokenizer.decode(prompt, skip_special_tokens=False)
    normalized_visible = " ".join(visible.split())
    if task.gold and not all(
        " ".join(span.split()) in normalized_visible
        or " ".join(json.dumps(span, ensure_ascii=False)[1:-1].split())
        in normalized_visible
        for span in task.gold
    ):
        return None
    target = json.dumps(
        {"answer": [{"text": span, "confidence": 0.9} for span in task.gold or []]},
        ensure_ascii=False,
    )
    completion = tokenizer.encode(target, add_special_tokens=False) + [
        tokenizer.eos_token_id
    ]
    if len(completion) > max_response:
        return None
    return prompt, completion


async def train(args):
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local)
    if world > 1:
        from datetime import timedelta

        dist.init_process_group("nccl", timeout=timedelta(hours=2))
    torch.manual_seed(args.seed)
    random.seed(args.seed + rank)
    tasks = load_tasks(args.train_data)
    if not tasks or any(task.split != "train" for task in tasks):
        raise ValueError("Trainer accepts nonempty training split only")
    if any(
        task.benchmark == "widesearch"
        and task.metadata.get("source") != "independent_curriculum"
        for task in tasks
    ):
        raise ValueError("Official WideSearch tasks must never enter training")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / f"train_rank{rank}.jsonl").open("a", buffering=1)

    def record(**value):
        line = json.dumps(value, allow_nan=False)
        log.write(line + "\n")
        if rank == 0:
            print(line, flush=True)

    record(
        event="start",
        world_size=world,
        tasks=len(tasks),
        algorithm="LoRA SFT warmup + on-policy GRPO",
        **{
            k: v
            for k, v in vars(args).items()
            if k not in {"output_dir", "train_data", "model", "adapter"}
        },
    )
    model, tokenizer = load_policy_model(
        args.model,
        adapter=args.adapter,
        trainable=True,
        rank=local,
        lora_rank=args.lora_rank,
    )
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.0)
    runner = EpisodeRunner(
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
        initial_chunks=args.initial_chunks,
        temperature=1.0,
    )
    policy = HFPolicy(
        model,
        tokenizer,
        batch_size=args.group_size,
        max_prompt_tokens=args.max_prompt_tokens,
        raw_sampling=True,
    )
    positives = [task for task in tasks if task.gold]
    negatives = [task for task in tasks if not task.gold]
    if not positives:
        raise ValueError("Training data must contain positive answers")
    started, optimizer_steps = time.monotonic(), 0

    def save(name):
        if world > 1:
            dist.barrier()
        if rank == 0:
            destination = out / name
            model.save_pretrained(destination)
            tokenizer.save_pretrained(destination)
            torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "optimizer_steps": optimizer_steps,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state(),
                    "python_rng": random.getstate(),
                },
                destination / "training_state.pt",
            )
            (destination / "training_config.json").write_text(
                json.dumps(
                    {
                        key: value
                        for key, value in vars(args).items()
                        if key not in {"model", "train_data", "output_dir", "adapter"}
                    },
                    indent=2,
                )
                + "\n"
            )
        if world > 1:
            dist.barrier()

    for step in range(args.sft_steps + args.rl_steps):
        optimizer.zero_grad(set_to_none=True)
        begin = time.monotonic()
        if step < args.sft_steps:
            example = None
            for _ in range(300):
                pool = negatives if negatives and (step + rank) % 3 == 0 else positives
                task = random.choice(pool)
                example = supervised_example(
                    task, runner, tokenizer, args.max_prompt_tokens, args.max_new_tokens
                )
                if example is not None:
                    break
            if example is None:
                raise RuntimeError("No grounded supervised example fits token budget")
            model.train()
            loss = -completion_logprobs(model, *example).mean()
            loss.backward()
            report = {
                "stage": "sft",
                "loss": loss.item(),
                "task": task.id,
                "response_tokens": len(example[1]),
            }
        else:
            task = random.choice(
                negatives if negatives and (step + rank) % 4 == 0 else positives
            )
            results = await asyncio.gather(
                *(runner.run(task, policy) for _ in range(args.group_size))
            )
            # Episode timeouts cancel a future, not a CUDA generation thread.
            # Wait for model access to finish before scoring or backpropagation.
            await policy.close()
            rewards = [result.reward for result in results]
            valid = [
                index
                for index, result in enumerate(results)
                if result.status != "infrastructure_error"
            ]
            average = sum(rewards[i] for i in valid) / len(valid) if valid else 0.0
            variance = (
                sum((rewards[i] - average) ** 2 for i in valid) / len(valid)
                if valid
                else 0.0
            )
            loss_total, segments = 0.0, 0
            # Fixed old policy scores before ANY optimizer mutation.
            scored = []
            model.eval()
            for index in valid:
                advantage = (rewards[index] - average) / (variance**0.5 + 1e-4)
                for segment in results[index].trainable_steps:
                    with torch.no_grad():
                        old = completion_logprobs(
                            model, segment.prompt_ids, segment.completion_ids
                        ).detach()
                    segment.logprobs = old.float().cpu().tolist()
                    scored.append((segment, old, advantage))
            model.train()
            for segment, old, advantage in scored:
                if advantage == 0:
                    continue
                current = completion_logprobs(
                    model, segment.prompt_ids, segment.completion_ids
                )
                loss = grpo_loss(current, old, advantage) / max(len(scored), 1)
                loss.backward()
                loss_total += loss.item()
                segments += 1
            # A small grounded training-only auxiliary prevents no-answer
            # collapse and supplies learning signal for equal-reward groups.
            auxiliary = supervised_example(
                task, runner, tokenizer, args.max_prompt_tokens, args.max_new_tokens
            )
            auxiliary_loss = 0.0
            if auxiliary is not None and args.sft_coefficient > 0:
                loss = (
                    -completion_logprobs(model, *auxiliary).mean()
                    * args.sft_coefficient
                )
                loss.backward()
                auxiliary_loss = loss.item()
            report = {
                "stage": "grpo",
                "loss": loss_total,
                "sft_auxiliary": auxiliary_loss,
                "rewards": rewards,
                "reward_std": variance**0.5,
                "policy_segments": segments,
                "task": task.id,
                "statuses": [result.status for result in results],
            }
            with (out / f"rollouts_rank{rank}.jsonl").open("a") as rollout_log:
                for result in results:
                    rollout_log.write(
                        json.dumps(
                            result.to_dict(include_tokens=True), ensure_ascii=False
                        )
                        + "\n"
                    )
        average_gradients(parameters, world)
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("Non-finite adapter gradient; refusing checkpoint")
        optimizer.step()
        optimizer_steps += 1
        record(
            step=optimizer_steps,
            seconds=time.monotonic() - begin,
            gradient_norm=norm.item(),
            elapsed=time.monotonic() - started,
            **report,
        )
        if args.save_every and optimizer_steps % args.save_every == 0:
            save(f"step_{optimizer_steps:04d}")
    await policy.close()
    save("final")
    record(
        event="finished",
        optimizer_steps=optimizer_steps,
        elapsed=time.monotonic() - started,
    )
    log.close()
    if world > 1:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--sft-steps", type=int, default=32)
    parser.add_argument("--rl-steps", type=int, default=24)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--sft-coefficient", type=float, default=0.1)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--initial-chunks", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (
        args.group_size < 2
        or min(args.sft_steps, args.rl_steps) < 0
        or args.sft_steps + args.rl_steps == 0
    ):
        parser.error("positive training steps and group-size >= 2 are required")
    asyncio.run(train(args))


if __name__ == "__main__":
    main()

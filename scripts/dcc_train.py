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
import math
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "slime"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.distributed as dist
from slime_plugins.dcc import EpisodeRunner, load_tasks
from slime_plugins.dcc.retrieval import load_retrieval_profile
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


def validate_grpo_dropout(model, *, rl_steps: int):
    """Require the same deterministic policy in eval and train mode for GRPO.

    Imported adapters retain their dropout configuration. SFT can use dropout,
    but this trainer samples and computes old scores in eval mode, so nonzero
    dropout during current-score replay would invalidate the on-policy ratio.
    Qwen attention uses functional dropout rather than an nn.Dropout module.
    """
    if rl_steps <= 0:
        return
    dropout_types = (
        torch.nn.Dropout,
        torch.nn.Dropout1d,
        torch.nn.Dropout2d,
        torch.nn.Dropout3d,
        torch.nn.AlphaDropout,
        torch.nn.FeatureAlphaDropout,
    )
    incompatible = []
    for name, module in model.named_modules():
        name = name or "<root>"
        if isinstance(module, dropout_types) and module.p != 0:
            incompatible.append(f"{name} (p={module.p})")
        attention_dropout = getattr(module, "attention_dropout", 0.0)
        if isinstance(attention_dropout, (int, float)) and attention_dropout != 0:
            incompatible.append(f"{name}.attention_dropout={attention_dropout}")
    if incompatible:
        details = "; ".join(incompatible[:8])
        if len(incompatible) > 8:
            details += f"; and {len(incompatible) - 8} more"
        raise ValueError(
            "On-policy GRPO requires zero dropout because rollouts/old logprobs "
            "use eval mode and current logprobs use train mode. Nonzero dropout: "
            f"{details}. Use a dropout-free base and adapter, or --rl-steps 0 "
            "for SFT-only training."
        )


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


def prepare_output_directory(out, rank, world):
    from dcc_training_io import prepare_training_output
    prepare_training_output(out, rank=rank, world_size=world, distributed=dist)


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
    defaults = dict(adv_estimator="grpo", cuad_protocol="legacy", task_order="random",
        credit_norm="mean_std_norm", invalid_action_penalty=.1, kl_coefficient=.001,
        graphgpo_omega=.2, graphpo_embedding=None, graphpo_branching="2,2,2,1,1,1,1,1",
        graphpo_similarity=.92, gigpo_gamma=.95, gigpo_step_weight=1.0,
        gigpo_similarity_threshold=.9, target_rollouts=0)
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    estimator = getattr(args, "adv_estimator", "grpo")
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
    prepare_output_directory(out, rank, world)
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
        algorithm=f"LoRA {estimator.upper()}",
        old_logprob_source="pre-update-replay",
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
        quantization=args.quantization,
    )
    validate_grpo_dropout(model, rl_steps=args.rl_steps)
    if hasattr(model, "dcc_quantization_report"):
        record(event="quantization", **model.dcc_quantization_report)
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.0)
    runner = EpisodeRunner(
        max_steps=args.max_steps,
        cuad_protocol=getattr(args, "cuad_protocol", "legacy"),
        max_new_tokens=args.max_new_tokens,
        initial_chunks=args.initial_chunks,
        temperature=1.0,
        retrieval_profile=load_retrieval_profile(args.retrieval_profile),
        auto_read_sources=getattr(args, "auto_read_sources", 0),
        retrieval_context_chars=getattr(args, "retrieval_context_chars", 0),
        retrieval_max_context_chars=getattr(args, "retrieval_max_context_chars", 16000),
    )
    policy = HFPolicy(
        model,
        tokenizer,
        batch_size=args.group_size,
        max_prompt_tokens=args.max_prompt_tokens,
        raw_sampling=True,
        allow_truncation=False,
    )
    graph_callbacks = None
    summary_policy = None
    if estimator == "graphpo":
        if not getattr(args, "graphpo_embedding", None):
            raise ValueError("GraphPO requires a fixed local sentence embedding model")
        from slime_plugins.dcc.graphpo_runtime import GraphPOSummarizer, SentenceEmbedder
        summary_policy = HFPolicy(model, tokenizer, batch_size=1,
            max_prompt_tokens=32768, allow_truncation=False, raw_sampling=True)
        graph_callbacks = (GraphPOSummarizer(summary_policy, tokenizer, protocol=runner.cuad_protocol), SentenceEmbedder(args.graphpo_embedding))
    positives = [task for task in tasks if task.gold]
    negatives = [task for task in tasks if not task.gold]
    if not positives:
        raise ValueError("Training data must contain positive answers")
    started, optimizer_steps, rollout_count = time.monotonic(), 0, 0

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
                    {"optimizer_steps": optimizer_steps, **{
                        key: value
                        for key, value in vars(args).items()
                        if key not in {"model", "train_data", "output_dir", "adapter"}
                    }},
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
            task = (tasks[(step - args.sft_steps) % len(tasks)]
                    if getattr(args, "task_order", "random") == "sequential" else
                    random.choice(negatives if negatives and (step + rank) % 4 == 0 else positives))
            if estimator == "graphpo":
                from slime_plugins.dcc.graphpo_collector import collect_graphpo
                branching = tuple(int(v) for v in args.graphpo_branching.split(","))
                summary_start, embedding_start = (len(callback.records) for callback in graph_callbacks)
                collection = await collect_graphpo(task, runner, policy, tokenizer=tokenizer,
                    summarize=graph_callbacks[0], embed=graph_callbacks[1], branching=branching,
                    threshold=args.graphpo_similarity, pooling_weight=.7, efficiency_weight=1.0)
                results = collection.results
                training_items = collection.training_items
                credit_diagnostics = collection.diagnostics
                advantages = None
                await summary_policy.close()
                with (out / f"graphpo_auxiliary_rank{rank}.jsonl").open("a", encoding="utf-8") as auxiliary_log:
                    auxiliary_log.write(json.dumps({"iteration": step + 1,
                        "summaries": graph_callbacks[0].records[summary_start:],
                        "embeddings": graph_callbacks[1].records[embedding_start:],
                        "embedding_provenance": graph_callbacks[1].provenance}, ensure_ascii=False) + "\n")
            else:
                results = await asyncio.gather(*(runner.run(task, policy) for _ in range(args.group_size)))
                from slime_plugins.dcc.credit_training import prepare_credit
                training_items, advantages, credit_diagnostics = prepare_credit(results,
                    tokenizer=tokenizer, method=estimator, mode=args.credit_norm,
                    invalid_penalty=args.invalid_action_penalty, gamma=args.gigpo_gamma,
                    weight=args.gigpo_step_weight, similarity_threshold=args.gigpo_similarity_threshold,
                    graph_omega=args.graphgpo_omega)
            await policy.close()
            rollout_count += len(results)
            rewards = [result.reward for result in results]
            raw_average = sum(rewards) / len(rewards) if rewards else 0.0
            raw_variance = sum((value - raw_average) ** 2 for value in rewards) / len(rewards) if rewards else 0.0
            eligible_rewards = ([rewards[i] for i, values in enumerate(advantages) if values]
                                if advantages is not None else rewards)
            average = sum(eligible_rewards) / len(eligible_rewards) if eligible_rewards else 0.0
            variance = (sum((value - average) ** 2 for value in eligible_rewards) / len(eligible_rewards)
                        if eligible_rewards else 0.0)
            loss_total, segments = 0.0, 0
            # Fixed old policy scores before ANY optimizer mutation.
            scored = []
            model.eval()
            for segment, advantage in training_items:
                with torch.no_grad():
                    old = completion_logprobs(model, segment.prompt_ids, segment.completion_ids).detach()
                    if args.kl_coefficient:
                        with model.disable_adapter():
                            reference = completion_logprobs(model, segment.prompt_ids, segment.completion_ids).detach()
                    else:
                        reference = None
                segment.logprobs = old.float().cpu().tolist()
                scored.append((segment, old, advantage, reference))
            model.train()
            kl_total = 0.0
            for segment, old, advantage, reference in scored:
                if advantage == 0 and not args.kl_coefficient:
                    continue
                current = completion_logprobs(
                    model, segment.prompt_ids, segment.completion_ids
                )
                policy_loss = grpo_loss(current, old, advantage)
                if reference is not None:
                    log_ratio = reference - current
                    kl = (log_ratio.exp() - log_ratio - 1).mean()
                    kl_total += float(kl.detach()) / max(len(scored), 1)
                    policy_loss = policy_loss + args.kl_coefficient * kl
                loss = policy_loss / max(len(scored), 1)
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
                "stage": estimator,
                "loss": loss_total,
                "sft_auxiliary": auxiliary_loss,
                "rewards": rewards,
                "reward_std": variance**0.5,
                "all_rollout_reward_std": raw_variance**0.5,
                "policy_segments": segments,
                "task": task.id,
                "statuses": [result.status for result in results],
                "advantages": advantages,
                "kl_mean": kl_total,
                "unique_training_decisions": len(training_items),
                "nonzero_policy_advantages": sum(value != 0 for _, value in training_items),
                "rollout_count": rollout_count,
                "credit": credit_diagnostics,
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
        applied = norm.item() > 0
        if applied:
            optimizer.step()
            optimizer_steps += 1
        record(
            step=optimizer_steps,
            iteration=step + 1,
            optimizer_step_applied=applied,
            seconds=time.monotonic() - begin,
            gradient_norm=norm.item(),
            elapsed=time.monotonic() - started,
            **report,
        )
        if args.save_every and applied and optimizer_steps % args.save_every == 0:
            save(f"step_{optimizer_steps:04d}")
        if (args.target_rollouts and rollout_count >= args.target_rollouts
                and (args.task_order != "sequential" or (step + 1 - args.sft_steps) % len(tasks) == 0)):
            break
    await policy.close()
    save("final")
    record(
        event="finished",
        optimizer_steps=optimizer_steps,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
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
    parser.add_argument("--target-rollouts", type=int, default=0)
    parser.add_argument("--adv-estimator", choices=("grpo", "gigpo", "graphgpo", "g2po", "graphpo"), default="grpo")
    parser.add_argument("--cuad-protocol", choices=("legacy", "cuad-audited-v1"), default="legacy")
    parser.add_argument("--task-order", choices=("random", "sequential"), default="random")
    parser.add_argument("--credit-norm", choices=("mean_norm", "mean_std_norm"), default="mean_std_norm")
    parser.add_argument("--invalid-action-penalty", type=float, default=.1)
    parser.add_argument("--kl-coefficient", type=float, default=.001)
    parser.add_argument("--graphgpo-omega", type=float, default=.2)
    parser.add_argument("--graphpo-embedding")
    parser.add_argument("--graphpo-branching", default="2,2,2,1,1,1,1,1")
    parser.add_argument("--graphpo-similarity", type=float, default=.92)
    parser.add_argument("--gigpo-gamma", type=float, default=0.95)
    parser.add_argument("--gigpo-step-weight", type=float, default=1.0)
    parser.add_argument("--gigpo-similarity-threshold", type=float, default=0.9,
                        help="GiGPO observation similarity threshold; 1 selects exact matching")
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--quantization", choices=("none", "nf4"), default="none")
    parser.add_argument("--retrieval-profile")
    parser.add_argument("--retrieval-context-chars", type=int, default=0)
    parser.add_argument("--retrieval-max-context-chars", type=int, default=16000)
    parser.add_argument("--auto-read-sources", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--sft-coefficient", type=float, default=0.1)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--initial-chunks", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if (not 0 < args.gigpo_gamma <= 1 or not math.isfinite(args.gigpo_step_weight)
            or args.gigpo_step_weight < 0 or not 0 < args.gigpo_similarity_threshold <= 1):
        parser.error("GiGPO requires gamma/threshold in (0, 1] and a finite nonnegative step weight")
    if (
        args.group_size < 2
        or min(args.sft_steps, args.rl_steps) < 0
        or args.sft_steps + args.rl_steps == 0
    ):
        parser.error("positive training steps and group-size >= 2 are required")
    if args.initial_chunks < 1:
        parser.error("Credit comparisons require an initial public source observation")
    if args.invalid_action_penalty < 0 or not math.isfinite(args.invalid_action_penalty):
        parser.error("Invalid-action penalty must be finite and nonnegative")
    if args.kl_coefficient < 0 or not math.isfinite(args.kl_coefficient):
        parser.error("KL coefficient must be finite and nonnegative")
    if args.adapter and args.kl_coefficient:
        parser.error("This reference KL uses the clean base; do not warm-start an adapter")
    asyncio.run(train(args))


if __name__ == "__main__":
    main()

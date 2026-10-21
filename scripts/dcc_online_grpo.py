#!/usr/bin/env python3
"""Opt-in TRAIN-only online LoRA GRPO for the native two-stage HF harness.

Sample raw-policy thoughts, execute greedy schema-constrained JSON, and train
only the sampled thought tokens. This is an independent HF runtime, not the
Slime/Megatron backend. There is no SFT auxiliary or offline preference loss.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

sys.path[:0] = [str(Path(__file__).resolve().parent),
                str(Path(__file__).resolve().parents[1] / "slime")]

import torch
import torch.distributed as dist

from dcc_training_io import prepare_training_output, publish_checkpoint
from dcc_train import validate_grpo_dropout
from slime_plugins.dcc import EpisodeRunner, load_tasks
from slime_plugins.dcc.durable_journal import OrderedDurableJSONLWriter, close_durable_journals
from slime_plugins.dcc.hf_chunked_logprobs import resolve_logprob_backend
from slime_plugins.dcc.hf_policy import encode_messages, load_policy_model
from slime_plugins.dcc.hf_compute import (COMPUTE_PROFILES, compute_profile_configuration,
    precision_observation, validate_model_compute_profile)
from slime_plugins.dcc.online_grpo import ALGORITHM, apply_policy_update, group_signal, thought_policy_loss
from slime_plugins.dcc.retrieval import load_retrieval_profile


def validate_source(expected):
    """Accept an actual clean checkout or its explicitly identified Git export."""
    root = Path(__file__).resolve().parents[1]
    receipt = root / ".dcc-online-grpo-source.json"
    if receipt.is_file():
        value = json.loads(receipt.read_text())
        if value != {"version": "dcc-online-grpo-git-export-v1", "source_commit": expected}:
            raise ValueError("Frozen runtime source receipt differs from the requested commit")
        return
    result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True)
    status = subprocess.run(["git", "-C", str(root), "status", "--porcelain",
                             "--untracked-files=normal", "--", "scripts", "slime/slime_plugins/dcc"],
                            capture_output=True, text=True, check=True)
    if result.stdout.strip() != expected or status.stdout:
        raise ValueError("Commit actual clean runtime sources before claiming their identity")


def validate_train_tasks(tasks):
    if not tasks or len({t.id for t in tasks}) != len(tasks):
        raise ValueError("Nonempty unique TRAIN tasks required")
    for task in tasks:
        if task.split != "train" or not isinstance(task.gold, list):
            raise ValueError("Only TRAIN tasks with explicit reward labels are allowed")
        if task.benchmark != "cuad":
            raise ValueError("This online pilot supports CUAD only; Web requires its own native schema backend")


def task_plan(tasks, *, steps, world_size, seed):
    """A fixed public task plan; CUAD negatives get one of every four draws."""
    validate_train_tasks(tasks)
    rng = random.Random(seed)
    positive = [i for i, task in enumerate(tasks) if task.gold]
    negative = [i for i, task in enumerate(tasks) if not task.gold]
    if not positive:
        raise ValueError("At least one positive reward-bearing TRAIN task required")
    decks = {"positive": [], "negative": []}
    def draw(name, pool):
        if not decks[name]:
            decks[name] = list(pool)
            rng.shuffle(decks[name])
        return decks[name].pop()
    flat = [draw("negative", negative) if negative and index % 4 == 0
            else draw("positive", positive) for index in range(steps * world_size)]
    return [flat[index:index + world_size] for index in range(0, len(flat), world_size)]


def make_runner(args):
    return EpisodeRunner(
        max_steps=args.max_steps, max_new_tokens=args.thought_tokens + args.json_tokens,
        temperature=1.0, evaluation_temperature=1.0,
        timeout_seconds=args.episode_timeout, initial_chunks=args.initial_chunks,
        retrieval_profile=load_retrieval_profile(args.retrieval_profile),
        retrieval_context_chars=args.retrieval_context_chars,
        retrieval_max_context_chars=args.retrieval_max_context_chars,
        retrieval_task_bound_expansion=args.retrieval_task_bound_expansion,
        cuad_structured_actions=True,
        auto_read_sources=args.auto_read_sources,
        web_structured_actions=True, web_structured_final=True,
        web_explicit_review=True, web_navigation_ledger=True,
        web_research_checkpoints=True, web_data_discovery_guidance=True,
        web_multi_citation=True, web_cached_rows=True,
    )


def replay_group(model, signal, scorer, *, group_size, max_steps, thought_tokens, clip):
    """One old-policy snapshot and one replay, before any optimizer mutation."""
    scored = []
    model.eval()
    with torch.no_grad():
        for segments, advantage in zip(signal.segments, signal.advantages, strict=True):
            if advantage == 0:
                continue
            for segment in segments:
                old = scorer(model, segment.prompt_ids, segment.completion_ids).detach()
                if not torch.isfinite(old).all():
                    raise FloatingPointError("Non-finite old thought scores")
                scored.append((segment, old, advantage))
    model.train()
    loss_total, maximum_score_difference, tokens = 0.0, 0.0, 0
    for segment, old, advantage in scored:
        current = scorer(model, segment.prompt_ids, segment.completion_ids)
        if not torch.isfinite(current).all():
            raise FloatingPointError("Non-finite current thought scores")
        difference = (current.detach() - old).abs().max().item()
        maximum_score_difference = max(maximum_score_difference, difference)
        loss = thought_policy_loss(current, old, advantage,
            group_size=group_size, max_steps=max_steps, thought_tokens=thought_tokens, clip=clip)
        loss.backward()
        loss_total += loss.item()
        tokens += len(segment.completion_ids)
    return dict(policy_loss=loss_total, scored_segments=len(scored), scored_tokens=tokens,
                maximum_old_current_logprob_difference=maximum_score_difference)


async def collect_group(runner, task, policy, *, group_size, rank):
    """Cancel/drain sibling episodes before draining model access on any error."""
    pending = [asyncio.create_task(runner.run(task, policy.bind_task(task.id, rank), evaluation=True))
               for _ in range(group_size)]
    failure = None
    try:
        return await asyncio.gather(*pending)
    except BaseException as error:
        failure = error
        raise
    finally:
        for worker in pending:
            if not worker.done():
                worker.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        try:
            await policy.close()
        except BaseException as error:
            if failure is None:
                raise
            failure.add_note(f"Policy drain: {type(error).__name__}: {error}")


def configuration(args, plan, tasks, world_size):
    return dict(algorithm=ALGORITHM, runtime="independent-HF; not Slime/Megatron",
        source_commit=args.source_commit, world_size=world_size,
        arguments={key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
        compute_profile=compute_profile_configuration(getattr(args, "compute_profile", "legacy")),
        training_split_only=True, sft_auxiliary=False, offline_preferences=False,
        behavior_policy=dict(thought="raw softmax T=1 top_p=1 top_k=0", json="greedy xgrammar"),
        policy_loss_tokens="sampled thought only, including actual closing token",
        json_policy_loss=False, model_failure_reward=0,
        objective_denominator=args.group_size * args.max_steps * args.thought_tokens,
        zero_global_gradient_optimizer_step=False,
        planned_task_ids=[[tasks[index].id for index in row] for row in plan])


def cpu_preflight(args, tasks, plan, world_size):
    from transformers import AutoTokenizer
    from slime_plugins.dcc.hf_native_policy import NativeHFPolicy
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    NativeHFPolicy.validate_tokenizer(tokenizer)
    runner = make_runner(args)
    lengths = []
    for index in sorted({index for row in plan for index in row}):
        messages = runner.initial_messages(tasks[index])
        # Native policy applies the same complete-message encoding. Final-turn
        # controls are measured separately so a pilot cannot silently omit them.
        from slime_plugins.dcc.episode import final_turn_message
        messages.append(final_turn_message(tasks[index], None)
                        if tasks[index].benchmark == "cuad" else
                        {"role": "user", "content": "Final answer now."})
        lengths.append(len(encode_messages(tokenizer, messages, args.max_prompt_tokens,
                                          allow_truncation=False)))
    return dict(cpu_preflight_only=True, CUDA_initialized=torch.cuda.is_initialized(),
                unique_planned_tasks=len(lengths), max_initial_final_prompt=max(lengths),
                training_config=configuration(args, plan, tasks, world_size))


async def train(args, tasks, plan, world_size):
    from slime_plugins.dcc.hf_native_policy import NativeHFPolicy
    rank, local = int(os.environ.get("RANK", "0")), int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    if world_size > 1:
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
    prepare_training_output(args.output_dir, rank=rank, world_size=world_size, distributed=dist)
    torch.manual_seed(args.seed)
    config = configuration(args, plan, tasks, world_size)
    events = OrderedDurableJSONLWriter(args.output_dir / f"train_rank{rank}.jsonl")
    traces = OrderedDurableJSONLWriter(args.output_dir / f"native_traces_rank{rank}.jsonl")
    rollouts = OrderedDurableJSONLWriter(args.output_dir / f"rollouts_rank{rank}.jsonl")
    def enqueue(writer, row):
        writer.write_line(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    def record(**row):
        enqueue(events, row)
        if rank == 0:
            print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
    policy = None
    failure = None
    started = time.monotonic()
    updates = 0
    try:
        record(event="start", **config)
        model, tokenizer = load_policy_model(args.model, adapter=args.adapter,
            trainable=True, rank=local, lora_rank=args.lora_rank, quantization=args.quantization,
            compute_profile=getattr(args, "compute_profile", "legacy"))
        validate_model_compute_profile(model, getattr(args, "compute_profile", "legacy"))
        config["precision"] = precision_observation(model)
        record(event="precision", **config["precision"])
        validate_grpo_dropout(model, rl_steps=args.steps)
        named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        if not named or any("lora_" not in name for name, _ in named):
            raise ValueError("Only LoRA parameters may be trained")
        parameters = [p for _, p in named]
        if world_size > 1:
            for parameter in parameters:
                dist.broadcast(parameter.data, src=0)
        initial = [p.detach().clone() for p in parameters]
        if hasattr(model, "dcc_quantization_report"):
            record(event="quantization", **model.dcc_quantization_report)
        optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.0)
        scorer, score_recipe = resolve_logprob_backend(args.logprob_backend, args.head_chunk_tokens)
        record(event="scorer", **score_recipe)
        policy = NativeHFPolicy(model, tokenizer, thought_tokens=args.thought_tokens,
            json_tokens=args.json_tokens, max_prompt_tokens=args.max_prompt_tokens,
            sampling_profile="raw", trace_sink=lambda value: enqueue(traces, value))
        runner = make_runner(args)
        # Identical initial replicas; independent sampled trajectories afterwards.
        torch.manual_seed(args.seed + rank * 100003)
        def save(name, iteration):
            if world_size > 1:
                dist.barrier()
            status = [None]
            if rank == 0:
                try:
                    def writer(destination):
                        model.save_pretrained(destination)
                        tokenizer.save_pretrained(destination)
                        torch.save(dict(optimizer=optimizer.state_dict(), optimizer_steps=updates,
                            rollout_iterations=iteration, torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state(), fresh_only_no_resume=True),
                            destination / "training_state.pt")
                        (destination / "training_config.json").write_text(
                            json.dumps(dict(config, rollout_iterations=iteration, optimizer_updates=updates),
                                       indent=2, ensure_ascii=False, allow_nan=False) + "\n")
                    publish_checkpoint(args.output_dir, name, updates, writer, sync=args.sync_artifacts)
                except Exception as error:
                    status[0] = f"{type(error).__name__}: {error}"
            if world_size > 1:
                dist.broadcast_object_list(status, src=0)
            if status[0] is not None:
                raise RuntimeError(f"Checkpoint publication failed: {status[0]}")
        save("initial", 0)
        for iteration, indices in enumerate(plan, 1):
            task = tasks[indices[rank]]
            begin = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            results = await collect_group(runner, task, policy, group_size=args.group_size, rank=rank)
            for sample, result in enumerate(results):
                enqueue(rollouts, dict(iteration=iteration, sample=sample,
                                       **result.to_dict(include_tokens=True)))
            signal = group_signal(results, group_size=args.group_size,
                                  max_steps=args.max_steps, thought_tokens=args.thought_tokens)
            report = replay_group(model, signal, scorer, group_size=args.group_size,
                max_steps=args.max_steps, thought_tokens=args.thought_tokens, clip=args.clip)
            norm, applied = apply_policy_update(parameters, optimizer, world_size=world_size,
                distributed=dist, max_grad_norm=args.max_grad_norm)
            updates += int(applied)
            record(event="step", iteration=iteration, optimizer_updates=updates,
                optimizer_step_applied=applied, global_gradient_norm=norm,
                task_id=task.id, statuses=[r.status for r in results], rewards=signal.rewards,
                reward_std=signal.reward_std, advantages=signal.advantages,
                discarded_reason=signal.discarded_reason,
                sampled_thought_tokens=sum(len(s.completion_ids) for group in signal.segments for s in group),
                seconds=time.monotonic() - begin, **report)
            if args.save_every and iteration % args.save_every == 0:
                save(f"iteration_{iteration:04d}", iteration)
        changed = sum(not torch.equal(parameter.detach(), before)
                      for parameter, before in zip(parameters, initial, strict=True))
        save("final", args.steps)
        summary = dict(rank=rank, rollout_iterations=args.steps, optimizer_updates=updates,
                       changed_adapter_tensors=changed, trainable_adapter_tensors=len(parameters))
        record(event="finished", elapsed=time.monotonic() - started, **summary)
        await policy.close()
        close_durable_journals((events, traces, rollouts))
        summary["closed_journals"] = dict(events=events.count, traces=traces.count, rollouts=rollouts.count)
        if rollouts.count != args.steps * args.group_size:
            raise RuntimeError("Closed rollout journal does not cover every sampled episode")
        summaries = [None] * world_size
        if world_size > 1:
            dist.all_gather_object(summaries, summary)
        else:
            summaries = [summary]
        if rank == 0:
            complete = dict(complete=True, algorithm=ALGORITHM, source_commit=args.source_commit,
                rollout_iterations=args.steps, optimizer_updates=updates, world_size=world_size,
                has_policy_gradient_updates=updates > 0 and all(s["changed_adapter_tensors"] > 0 for s in summaries),
                ranks=summaries, benchmark_improvement_claimed=False)
            with (args.output_dir / "training_complete.json").open("x") as stream:
                stream.write(json.dumps(complete, indent=2, allow_nan=False) + "\n")
    except BaseException as error:
        failure = error
        raise
    finally:
        if policy is not None:
            try:
                await policy.close()
            except BaseException as error:
                if failure is None:
                    failure = error
                else:
                    failure.add_note(f"Policy cleanup: {type(error).__name__}: {error}")
        close_durable_journals((events, traces, rollouts), failure=failure)
        if failure is not None and sys.exc_info()[0] is None:
            raise failure
    if world_size > 1:
        dist.destroy_process_group()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--thought-tokens", type=int, default=4096)
    parser.add_argument("--json-tokens", type=int, default=4096)
    parser.add_argument("--max-prompt-tokens", type=int, default=16384)
    parser.add_argument("--max-steps", type=int, default=2)
    parser.add_argument("--episode-timeout", type=float, default=7200)
    parser.add_argument("--initial-chunks", type=int, default=4)
    parser.add_argument("--retrieval-profile")
    parser.add_argument("--retrieval-context-chars", type=int, default=1200)
    parser.add_argument("--retrieval-max-context-chars", type=int, default=12000)
    parser.add_argument("--retrieval-task-bound-expansion", action="store_true")
    parser.add_argument("--auto-read-sources", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--quantization", choices=("none", "nf4"), default="nf4")
    parser.add_argument("--compute-profile", choices=COMPUTE_PROFILES, default="legacy")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--save-every", type=int, default=2)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--logprob-backend", choices=("original", "qwen3-moe-chunked"), default="qwen3-moe-chunked")
    parser.add_argument("--head-chunk-tokens", type=int)
    parser.add_argument("--sync-artifacts", action="store_true")
    parser.add_argument("--cpu-preflight", action="store_true")
    parser.add_argument("--plan-world-size", type=int, default=1)
    args = parser.parse_args(argv)
    import re
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        parser.error("--source-commit requires actual complete committed runtime identity")
    if (any(value < 1 for value in (args.steps, args.max_steps, args.thought_tokens, args.json_tokens,
            args.max_prompt_tokens, args.lora_rank, args.plan_world_size))
            or args.group_size < 2 or args.save_every < 0
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not math.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0
            or not math.isfinite(args.episode_timeout) or args.episode_timeout <= 0
            or not 0 < args.clip < 1):
        parser.error("Invalid online training budgets or optimizer settings")
    tasks = load_tasks(args.train_data)
    validate_source(args.source_commit)
    world_size = args.plan_world_size if args.cpu_preflight else int(os.environ.get("WORLD_SIZE", "1"))
    plan = task_plan(tasks, steps=args.steps, world_size=world_size, seed=args.seed)
    if args.cpu_preflight:
        print(json.dumps(cpu_preflight(args, tasks, plan, world_size), ensure_ascii=False, allow_nan=False))
    else:
        asyncio.run(train(args, tasks, plan, world_size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Opt-in TRAIN-only online LoRA GRPO for the native two-stage HF harness.

By default, sample raw-policy thoughts, execute greedy constrained JSON, and
train only the thoughts. --joint-json-policy opts CUAD into fresh stochastic
constrained JSON and a grammar-normalized two-phase objective. This is an
independent HF runtime, not Slime/Megatron; there is no SFT auxiliary loss.
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import timedelta
from functools import partial
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
from slime_plugins.dcc.cuad_reasoning import cuad_concise_reasoning_configuration
from slime_plugins.dcc.web_auto_read import auto_read_follow_docs_configuration
from slime_plugins.dcc.durable_journal import OrderedDurableJSONLWriter, close_durable_journals
from slime_plugins.dcc.hf_chunked_logprobs import resolve_logprob_backend
from slime_plugins.dcc.hf_policy import encode_messages, load_policy_model
from slime_plugins.dcc.hf_compute import (COMPUTE_PROFILES, compute_profile_configuration,
    precision_observation, validate_model_compute_profile)
from slime_plugins.dcc.online_grpo import (ALGORITHM, apply_policy_update, group_signal,
                                         thought_policy_loss, wait_for_update_ranks)
from slime_plugins.dcc.episode_gspo import (POLICY_SURROGATES, VARIANT as GSPO_VARIANT,
    gspo_configuration, validate_policy_surrogate)
from slime_plugins.dcc.reference_kl import (VARIANT as REFERENCE_KL_VARIANT,
    frozen_base_logprobs, reference_kl_configuration, thought_reference_kl_loss,
    validate_base_reference_model, validate_reference_kl_beta)
from slime_plugins.dcc.native_hf_context import context_handling_configuration
from slime_plugins.dcc.native_joint_grpo import (JSON_SURROGATE_VARIANT,
    json_surrogate_configuration, validate_json_surrogate_weight, weighted_joint_policy_loss)
from slime_plugins.dcc.web_observation_budget import web_observation_budget_configuration
from slime_plugins.dcc.online_web import (ONLINE_REWARD_VARIANTS, TABLE_GROUNDED_REWARD,
    online_reward_configuration)
from slime_plugins.dcc.retrieval import load_retrieval_profile
from slime_plugins.dcc.tools import WebTools


def validate_source(expected, *, online_web=False):
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
    checked_paths = ["scripts", "slime/slime_plugins/dcc"]
    if online_web:
        checked_paths.append("slime/slime_plugins/dcc_train_literal_reward_contract.py")
    status = subprocess.run(["git", "-C", str(root), "status", "--porcelain",
                             "--untracked-files=normal", "--", *checked_paths],
                            capture_output=True, text=True, check=True)
    if result.stdout.strip() != expected or status.stdout:
        raise ValueError("Commit actual clean runtime sources before claiming their identity")


def validate_train_tasks(tasks, *, require_labels=True):
    if not tasks or len({t.id for t in tasks}) != len(tasks):
        raise ValueError("Nonempty unique TRAIN tasks required")
    for task in tasks:
        if task.split != "train" or (require_labels and not isinstance(task.gold, list)):
            raise ValueError("Only TRAIN tasks with explicit reward labels are allowed")
        if task.benchmark != "cuad":
            raise ValueError("This online pilot supports CUAD only; Web requires its own native schema backend")


def task_plan(tasks, *, steps, world_size, seed, sampling="stratified-3to1"):
    """Original strata, opt-in capped span weights, or a label-blind TRAIN deck."""
    if sampling not in {"stratified-3to1", "uniform", "stratified-3to1-span-cap5"}:
        raise ValueError("Unknown CUAD task sampling mode")
    # A uniform public task plan must not inspect reward labels. The training
    # entrypoint still validates that every task has explicit TRAIN labels.
    validate_train_tasks(tasks, require_labels=sampling != "uniform")
    rng = random.Random(seed)
    if sampling == "uniform":
        flat, deck = [], []
        for _ in range(steps * world_size):
            if not deck:
                deck = list(range(len(tasks)))
                rng.shuffle(deck)
            flat.append(deck.pop())
        return [flat[index:index + world_size] for index in range(0, len(flat), world_size)]
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
    if sampling == "stratified-3to1-span-cap5":
        weights = [min(len(tasks[index].gold), 5) for index in positive]
        # One local seeded RNG drives both strata. Positive draws may repeat;
        # negative draws still exhaust a shuffled deck before its next cycle.
        flat = [draw("negative", negative) if negative and index % 4 == 0
                else rng.choices(positive, weights=weights, k=1)[0]
                for index in range(steps * world_size)]
        return [flat[index:index + world_size] for index in range(0, len(flat), world_size)]
    flat = [draw("negative", negative) if negative and index % 4 == 0
            else draw("positive", positive) for index in range(steps * world_size)]
    return [flat[index:index + world_size] for index in range(0, len(flat), world_size)]


def cuad_task_sampling_configuration(sampling):
    if sampling == "stratified-3to1":
        return dict(mode=sampling, stratification="TRAIN gold-label negative-vs-positive",
            negative_stride=4, positive_fallback_when_no_negatives=True,
            seeded_independent_decks=True, outcome_based_resampling=False,
            label_dependent=True, without_replacement_per_deck_cycle=True)
    if sampling == "uniform":
        return dict(mode=sampling, stratification=None, negative_stride=None,
            positive_fallback_when_no_negatives=False, seeded_independent_decks=False,
            outcome_based_resampling=False, label_dependent=False,
            without_replacement_per_deck_cycle=True,
            deck_scope="all-CUAD-TRAIN-task-indices", seeded_shuffled_deck=True)
    if sampling == "stratified-3to1-span-cap5":
        return dict(mode=sampling, stratification="TRAIN gold-label negative-vs-positive",
            negative_stride=4, positive_fallback_when_no_negatives=True,
            label_dependent=True, outcome_based_resampling=False,
            positive_sampling="weighted-with-replacement",
            positive_weight="min(TRAIN gold span count, 5)", positive_span_count_cap=5,
            positive_cap_recomputed=False,
            negative_sampling="shuffled-without-replacement-per-deck-cycle",
            seeded_rng="single-random.Random(seed)-shared-by-both-strata")
    raise ValueError("Unknown CUAD task sampling mode")


def make_web_tools(args):
    return WebTools(max_text_window_chars=getattr(args, "web_text_window_chars", None),
                    web_zip_member_fallback=getattr(args, "web_zip_member_fallback", False))


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
        cuad_concise_reasoning=getattr(args, "cuad_concise_reasoning", False),
        web_concise_reasoning=getattr(args, "web_concise_reasoning", False),
        web_observation_token_budget=getattr(args, "web_observation_token_budget", None),
        auto_read_sources=args.auto_read_sources,
        auto_read_follow_docs=getattr(args, "auto_read_follow_docs", False),
        web_tools_factory=partial(make_web_tools, args),
        web_structured_actions=True, web_structured_final=True,
        web_explicit_review=True, web_navigation_ledger=True,
        web_research_checkpoints=True, web_data_discovery_guidance=True,
        web_multi_citation=True, web_cached_rows=True,
    )


def replay_group(model, signal, scorer, *, group_size, max_steps, thought_tokens, clip,
                 replay_diagnostic="full", reference_kl_beta=0, reference_base_model=None,
                 reference_source_commit=None, joint_json_policy=False, json_tokens=None,
                 grammar_compiler=None, logprob_backend="qwen3-moe-chunked", head_chunk_tokens=None,
                 policy_surrogate="token-grpo", json_surrogate_weight=1):
    """Use actual cached-generation behavior as old; replay is diagnostic only.

    Cached generation and teacher-forced replay may differ numerically even
    before an update. Report that discrepancy; never reconstruct the behavior
    denominator from a teacher-forced snapshot.
    """
    if replay_diagnostic not in {"full", "first", "off"}:
        raise ValueError("Replay diagnostic must be full, first or off")
    validate_reference_kl_beta(reference_kl_beta)
    validate_policy_surrogate(policy_surrogate, reference_kl_beta)
    validate_json_surrogate_weight(json_surrogate_weight,
        joint_json_policy=joint_json_policy, policy_surrogate=policy_surrogate)
    objective_tokens = thought_tokens
    if joint_json_policy:
        if type(json_tokens) is not int or json_tokens < 1 or grammar_compiler is None:
            raise ValueError("Joint replay requires its actual grammar compiler and fixed JSON budget")
        objective_tokens += json_tokens
    def scorer_for(segment):
        if joint_json_policy:
            from slime_plugins.dcc.native_joint_grpo import segment_scorer
            return segment_scorer(segment, grammar_compiler,
                backend=logprob_backend, head_chunk_tokens=head_chunk_tokens)
        return scorer
    def old_values(segment):
        return segment.behavior_logprobs if joint_json_policy else segment.thought_logprobs
    if policy_surrogate == "episode-gspo":
        from slime_plugins.dcc.episode_gspo import replay_episode_gspo_group
        return replay_episode_gspo_group(model, signal, scorer_for=scorer_for, old_values=old_values,
            group_size=group_size, max_steps=max_steps, objective_tokens=objective_tokens,
            clip=clip, replay_diagnostic=replay_diagnostic, joint_json_policy=joint_json_policy)
    scored = [(segment, advantage)
              for segments, advantage in zip(signal.segments, signal.advantages, strict=True)
              if advantage != 0 for segment in segments]
    if reference_kl_beta > 0 and scored:
        validate_base_reference_model(model)
    diagnostics = {}
    if replay_diagnostic != "off":
        model.eval()
        with torch.no_grad():
            selected = scored if replay_diagnostic == "full" else scored[:1]
            for index, (segment, _) in enumerate(selected):
                replay = scorer_for(segment)(model, segment.prompt_ids, segment.completion_ids).detach()
                old = torch.tensor(old_values(segment), device=replay.device, dtype=torch.float32)
                if old.shape != replay.shape or not torch.isfinite(old).all() or not torch.isfinite(replay).all():
                    raise FloatingPointError("Aligned finite behavior and diagnostic replay scores required")
                diagnostics[index] = replay
    model.train()
    loss_total, tokens = 0.0, 0
    reference_kl_total, objective_total = 0.0, 0.0
    behavior_replay, replay_current = None, None
    behavior_current, diagnostic_tokens = 0.0, 0
    ratio_min, ratio_max, clipped_tokens = None, None, 0
    for index, (segment, advantage) in enumerate(scored):
        active_scorer = scorer_for(segment)
        if reference_kl_beta > 0:
            # Finish and restore the adapter context BEFORE creating a current
            # graph. Backward below completes before the next reference read.
            reference = frozen_base_logprobs(model, active_scorer, segment.prompt_ids, segment.completion_ids)
        current = active_scorer(model, segment.prompt_ids, segment.completion_ids)
        old = torch.tensor(old_values(segment), device=current.device, dtype=torch.float32)
        if old.shape != current.shape or not torch.isfinite(old).all() or not torch.isfinite(current).all():
            raise FloatingPointError("Aligned finite actual behavior and current thought scores required")
        observed = current.detach()
        behavior_current = max(behavior_current, (observed - old).abs().max().item())
        if index in diagnostics:
            replay = diagnostics[index]
            if old.shape != replay.shape or replay.device != current.device:
                raise FloatingPointError("Aligned behavior, current and diagnostic replay scores required")
            difference = (replay - old).abs().max().item()
            behavior_replay = max(behavior_replay if behavior_replay is not None else difference, difference)
            difference = (observed - replay).abs().max().item()
            replay_current = max(replay_current if replay_current is not None else difference, difference)
            diagnostic_tokens += len(segment.completion_ids)
        ratio = (observed - old).exp()
        if not torch.isfinite(ratio).all():
            raise FloatingPointError("Non-finite actual behavior importance ratio")
        lo, hi = ratio.min().item(), ratio.max().item()
        ratio_min, ratio_max = min(ratio_min if ratio_min is not None else lo, lo), max(ratio_max if ratio_max is not None else hi, hi)
        clipped_tokens += ((ratio < 1 - clip) | (ratio > 1 + clip)).sum().item()
        if json_surrogate_weight == 1:
            loss = thought_policy_loss(current, old, advantage,
                group_size=group_size, max_steps=max_steps, thought_tokens=objective_tokens, clip=clip)
        else:
            loss = weighted_joint_policy_loss(current, old, advantage,
                thought_count=segment.thought_count, group_size=group_size, max_steps=max_steps,
                thought_tokens=thought_tokens, json_tokens=json_tokens,
                json_surrogate_weight=json_surrogate_weight, clip=clip)
        if reference_kl_beta > 0:
            kl = thought_reference_kl_loss(current, reference,
                group_size=group_size, max_steps=max_steps, thought_tokens=objective_tokens)
            objective = loss + reference_kl_beta * kl
            if not torch.isfinite(objective):
                raise FloatingPointError("Non-finite reference-KL online objective; refusing update")
            objective.backward()
            reference_kl_total += kl.item()
            objective_total += objective.item()
        else:
            loss.backward()
        loss_total += loss.item()
        tokens += len(segment.completion_ids)
    report = dict(policy_loss=loss_total, scored_segments=len(scored), scored_tokens=tokens,
                replay_diagnostic_mode=replay_diagnostic,
                replay_diagnostic_scope={"full": "all-nonzero-advantage-segments",
                    "first": "first-nonzero-advantage-segment-per-group", "off": "not-collected"}[replay_diagnostic],
                diagnostic_scored_segments=len(diagnostics), diagnostic_scored_tokens=diagnostic_tokens,
                old_logprob_source="actual-sampled-thought-behavior",
                maximum_behavior_replay_logprob_difference=behavior_replay,
                maximum_behavior_current_logprob_difference=behavior_current if tokens else None,
                maximum_replay_current_logprob_difference=replay_current,
                importance_ratio_min=ratio_min, importance_ratio_max=ratio_max,
                importance_ratio_outside_clip_fraction=clipped_tokens / tokens if tokens else None)
    if joint_json_policy:
        report.update(old_logprob_source="actual-sampled-raw-thought-and-post-xgrammar-JSON-behavior",
            thought_scored_tokens=sum(segment.thought_count for segment, _ in scored),
            json_scored_tokens=sum(len(segment.json_logprobs) for segment, _ in scored))
    if reference_kl_beta > 0:
        report.update(algorithm_variant=REFERENCE_KL_VARIANT,
            reference_kl=reference_kl_configuration(reference_kl_beta, base_model=reference_base_model,
                source_commit=reference_source_commit),
            reference_kl_unweighted_loss=reference_kl_total,
            reference_kl_loss=reference_kl_beta * reference_kl_total,
            objective_loss=objective_total,
            reference_scored_segments=len(scored), reference_scored_tokens=tokens)
        if joint_json_policy:
            report["algorithm_variant"] = "frozen-base-reference-k3-nonzero-advantage-joint-v1"
            report["reference_kl"] = joint_reference_configuration(report["reference_kl"])
    if json_surrogate_weight != 1:
        report["json_surrogate"] = json_surrogate_configuration(json_surrogate_weight)
        report["algorithm_variant"] = "+".join(filter(None,
            (report.get("algorithm_variant"), JSON_SURROGATE_VARIANT)))
    return report


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
    from slime_plugins.dcc.web_archive import web_zip_member_fallback_configuration
    from slime_plugins.dcc.web_reasoning import web_concise_reasoning_configuration

    surrogate = getattr(args, "policy_surrogate", "token-grpo")
    validate_policy_surrogate(surrogate, getattr(args, "reference_kl_beta", 0))
    weight = getattr(args, "json_surrogate_weight", 1)
    observation_budget = getattr(args, "web_observation_token_budget", None)
    web_observation_budget_configuration(observation_budget)
    if observation_budget is not None and not getattr(args, "enable_online_web", False):
        raise ValueError("Web observation token budget requires --enable-online-web")
    validate_json_surrogate_weight(weight, joint_json_policy=getattr(args, "joint_json_policy", False),
        policy_surrogate=surrogate, online_web=getattr(args, "enable_online_web", False),
        benchmarks=(t.benchmark for t in tasks))
    if surrogate == "episode-gspo" and (getattr(args, "enable_online_web", False)
            or any(t.benchmark != "cuad" for t in tasks)):
        raise ValueError("episode-gspo currently supports CUAD only; online Web is unsupported")
    if getattr(args, "joint_json_policy", False) and (
            getattr(args, "enable_online_web", False) or any(t.benchmark != "cuad" for t in tasks)):
        raise ValueError("Joint JSON policy currently supports CUAD only; online Web is unsupported")
    web = make_web_tools(args)
    web_reasoning = web_concise_reasoning_configuration(getattr(args, "web_concise_reasoning", False))
    reference_kl = reference_kl_configuration(getattr(args, "reference_kl_beta", 0),
        base_model=args.model, source_commit=args.source_commit)
    config = dict(algorithm=ALGORITHM, runtime="independent-HF; not Slime/Megatron",
        source_commit=args.source_commit, world_size=world_size,
        arguments={key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()
                   if (key not in {"joint_json_policy", "web_zip_member_fallback", "web_concise_reasoning"} or value)
                   and (key != "policy_surrogate" or value != "token-grpo")
                   and (key != "json_surrogate_weight" or value != 1)
                   and (key != "web_observation_token_budget" or value is not None)},
        compute_profile=compute_profile_configuration(getattr(args, "compute_profile", "legacy")),
        cuad_concise_reasoning=cuad_concise_reasoning_configuration(getattr(args, "cuad_concise_reasoning", False)),
        auto_read_follow_docs=auto_read_follow_docs_configuration(getattr(args, "auto_read_follow_docs", False)),
        web_tool_limits=dict(max_page_chars=web.max_page_chars,
                             max_text_window_chars=web.max_text_window_chars),
        training_split_only=True, sft_auxiliary=False, offline_preferences=False,
        cuad_task_selection=cuad_task_sampling_configuration(
            getattr(args, "cuad_task_sampling", "stratified-3to1")),
        behavior_policy=dict(thought="raw softmax T=1 top_p=1 top_k=0", json="greedy xgrammar",
            old_logprob_source="streaming-generation-raw-logits-v1",
            teacher_forced_replay_is_behavior=False),
        policy_loss_tokens="sampled thought only, including actual closing token",
        replay_diagnostic_mode=getattr(args, "replay_diagnostic", "full"),
        advantage_normalization=getattr(args, "advantage_normalization", "std"),
        json_policy_loss=False, model_failure_reward=0,
        context_failure_handling=context_handling_configuration(),
        distributed_update_readiness=dict(backend="gloo", device_synchronize_before_wait=True,
            timeout_seconds=7200, before_gradient_collectives=True, enabled=world_size > 1),
        objective_denominator=args.group_size * args.max_steps * args.thought_tokens,
        zero_global_gradient_optimizer_step=False,
        planned_task_ids=[[tasks[index].id for index in row] for row in plan])
    config["reference_kl"] = reference_kl
    if reference_kl["enabled"]:
        config["algorithm"] = ALGORITHM + "+" + REFERENCE_KL_VARIANT
        config["algorithm_variant"] = REFERENCE_KL_VARIANT
    if getattr(args, "joint_json_policy", False):
        from slime_plugins.dcc.native_joint_grpo import ALGORITHM as JOINT_ALGORITHM, JSON_BEHAVIOR_SOURCE
        from slime_plugins.dcc.hf_native_policy import JOINT_PROTOCOL
        config.update(algorithm=JOINT_ALGORITHM,
            native_policy_protocol=JOINT_PROTOCOL,
            behavior_policy=dict(thought="raw softmax T=1 top_p=1 top_k=0",
                json="xgrammar-masked multinomial T=1 top_p=1 top_k=0",
                old_logprob_source=dict(thought="streaming-generation-raw-logits-v1", json=JSON_BEHAVIOR_SOURCE),
                teacher_forced_replay_is_behavior=False),
            policy_loss_tokens="actual sampled thought and grammar-normalized JSON, including real stops",
            json_policy_loss=True,
            objective_denominator=args.group_size * args.max_steps * (args.thought_tokens + args.json_tokens),
            reference_kl=joint_reference_configuration(reference_kl))
        if reference_kl["enabled"]:
            config["algorithm_variant"] = "frozen-base-reference-k3-nonzero-advantage-joint-v1"
            config["algorithm"] += "+" + config["algorithm_variant"]
    if surrogate == "episode-gspo":
        config["algorithm"] += "+" + GSPO_VARIANT
        config["algorithm_variant"] = GSPO_VARIANT
        config["policy_surrogate"] = gspo_configuration()
    if weight != 1:
        config["algorithm"] += "+" + JSON_SURROGATE_VARIANT
        config["algorithm_variant"] = "+".join(filter(None,
            (config.get("algorithm_variant"), JSON_SURROGATE_VARIANT)))
        config["json_surrogate"] = json_surrogate_configuration(weight)
    if getattr(args, "enable_online_web", False):
        from slime_plugins.dcc.online_web import binding_protocols
        if observation_budget is not None:
            config["web_observation_budget"] = web_observation_budget_configuration(observation_budget)
        if web_reasoning["enabled"]:
            config["web_concise_reasoning"] = web_reasoning
        if web.web_zip_member_fallback:
            config["web_zip_member_fallback"] = web_zip_member_fallback_configuration(True)
        config["online_web"] = dict(enabled=True, protocols=binding_protocols(),
            web_deck="public-ID-only-shuffled-cycles",
            cuad_plan=("existing-TRAIN-gold-positive-negative-stratification"
                       if config["cuad_task_selection"]["mode"] == "stratified-3to1"
                       else "TRAIN-gold-stratified-3to1-positive-span-cap5-with-replacement-negative-deck"
                       if config["cuad_task_selection"]["mode"] == "stratified-3to1-span-cap5"
                       else "uniform-label-blind-CUAD-TRAIN-index-deck"),
            draws_per_cycle=dict(cuad=args.cuad_draws_per_cycle,
                                 widesearch=args.web_draws_per_cycle),
            online_train_reward=online_reward_configuration(
                getattr(args, "web_train_reward_variant", TABLE_GROUNDED_REWARD)),
            labels="deserialized-only-after-own-group-durable-close")
    return config


def joint_reference_configuration(config):
    result = dict(config)
    result.update(json_tokens=True, denominator="group_size*max_steps*(thought_tokens+json_tokens)",
        behavior_denominator="actual-sampled-thought-and-post-xgrammar-JSON-behavior-unchanged")
    if result["enabled"]:
        result["active_scope"] = "nonzero-advantage-sampled-thought-and-JSON-segments"
    return result


def native_grammar_cpu_preflight(model_name, tokenizer):
    """Exercise the native compiler using checkpoint metadata, without weights."""
    import xgrammar as xgr
    from packaging.version import Version
    from transformers import AutoConfig, GenerationConfig, __version__
    from transformers.utils import GENERATION_CONFIG_NAME
    from transformers.utils.hub import cached_file
    from slime_plugins.dcc.cuad_structured import cuad_final_schema
    from slime_plugins.dcc.hf_native_policy import END_THINK
    from slime_plugins.dcc.hf_native_structured import compile_native_structured_schema

    config = AutoConfig.from_pretrained(model_name, local_files_only=True)
    # Only a genuinely absent file permits a fallback. HF wraps malformed JSON
    # and invalid UTF-8 in OSError too, so catching that error hides bad metadata.
    generation_file = cached_file(model_name, GENERATION_CONFIG_NAME,
        local_files_only=True, _raise_exceptions_for_missing_entries=False)
    if generation_file is not None:
        generation = GenerationConfig.from_pretrained(model_name, local_files_only=True)
        generation_source = "generation_config.json"
    else:
        # Match each loader's model-config fallback. TF5 removes generation
        # parameters from AutoConfig, so its loader rereads the original JSON.
        if Version(__version__).major >= 5:
            generation = GenerationConfig.from_pretrained(model_name,
                config_file_name="config.json", _from_model_config=True,
                local_files_only=True)
        else:
            generation = GenerationConfig.from_model_config(config)
        generation_source = "model_config"
    eos = generation.eos_token_id
    eos_ids = [eos] if type(eos) is int else list(eos or [])
    if not eos_ids or END_THINK in eos_ids or tokenizer.eos_token_id not in eos_ids:
        raise ValueError("Non-thinking EOS IDs including tokenizer primary EOS required")
    # The runtime loader supplies the primary EOS as padding when absent.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        raise ValueError("An explicit padding token is required")
    compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(
        tokenizer, vocab_size=config.vocab_size, stop_token_ids=eos_ids), max_threads=1)
    compile_native_structured_schema(compiler, cuad_final_schema())
    return dict(xgrammar_imported=True, tokenizer_info_created=True,
                grammar_compiler_created=True, public_schema="cuad_final",
                schema_compiled=True, vocab_size=config.vocab_size,
                stop_token_ids=eos_ids, generation_config_source=generation_source)


def cpu_preflight(args, tasks, plan, world_size):
    from transformers import AutoTokenizer
    from slime_plugins.dcc.hf_native_policy import NativeHFPolicy
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    NativeHFPolicy.validate_tokenizer(tokenizer)
    grammar = native_grammar_cpu_preflight(args.model, tokenizer)
    if getattr(args, "joint_json_policy", False):
        import xgrammar as xgr
        from slime_plugins.dcc.cuad_structured import cuad_final_schema
        from slime_plugins.dcc.native_joint_grpo import JSON_BEHAVIOR_SOURCE, json_prefix_bitmasks
        compiler = xgr.GrammarCompiler(xgr.TokenizerInfo.from_huggingface(tokenizer,
            vocab_size=grammar["vocab_size"], stop_token_ids=grammar["stop_token_ids"]), max_threads=1)
        ids = tokenizer.encode('{"plan":"review","checks":"observed","answer":[]}', add_special_tokens=False)
        ids.append(grammar["stop_token_ids"][0])
        masks = json_prefix_bitmasks(compiler, cuad_final_schema(), ids, eos_ids=grammar["stop_token_ids"])
        grammar["joint_json_support_preflight"] = dict(behavior_source=JSON_BEHAVIOR_SOURCE,
            actual_model_sampling=False, legal_complete_JSON_and_EOS=True, positions=masks.shape[0])
    runner = make_runner(args)
    lengths = []
    for index in sorted({index for row in plan for index in row}):
        messages = runner.initial_messages(tasks[index])
        # Native policy applies the same complete-message encoding. Final-turn
        # controls are measured separately so a pilot cannot silently omit them.
        from slime_plugins.dcc.episode import final_turn_message
        from slime_plugins.dcc.episode import _WebEvidence
        messages.append(final_turn_message(tasks[index], None if tasks[index].benchmark == "cuad"
                                           else _WebEvidence()))
        lengths.append(len(encode_messages(tokenizer, messages, args.max_prompt_tokens,
                                          allow_truncation=False)))
    return dict(cpu_preflight_only=True, CUDA_initialized=torch.cuda.is_initialized(),
                native_grammar_preflight=grammar,
                unique_planned_tasks=len(lengths), max_initial_final_prompt=max(lengths),
                training_config=configuration(args, plan, tasks, world_size))


async def train(args, tasks, plan, world_size, *, web_inputs=None):
    observation_budget = getattr(args, "web_observation_token_budget", None)
    web_observation_budget_configuration(observation_budget)
    if observation_budget is not None and not getattr(args, "enable_online_web", False):
        raise ValueError("Web observation token budget requires --enable-online-web")
    validate_policy_surrogate(getattr(args, "policy_surrogate", "token-grpo"),
                              getattr(args, "reference_kl_beta", 0))
    validate_json_surrogate_weight(getattr(args, "json_surrogate_weight", 1),
        joint_json_policy=getattr(args, "joint_json_policy", False),
        policy_surrogate=getattr(args, "policy_surrogate", "token-grpo"),
        online_web=getattr(args, "enable_online_web", False) or web_inputs is not None,
        benchmarks=(t.benchmark for t in tasks))
    if getattr(args, "policy_surrogate", "token-grpo") == "episode-gspo" and (
            getattr(args, "enable_online_web", False) or web_inputs is not None
            or any(t.benchmark != "cuad" for t in tasks)):
        raise ValueError("episode-gspo currently supports CUAD only; online Web is unsupported")
    if getattr(args, "joint_json_policy", False) and (
            getattr(args, "enable_online_web", False) or web_inputs is not None
            or any(t.benchmark != "cuad" for t in tasks)):
        raise ValueError("Joint JSON policy currently supports CUAD only; online Web is unsupported")
    from slime_plugins.dcc.hf_native_policy import NativeHFPolicy
    rank, local = int(os.environ.get("RANK", "0")), int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    control_group = None
    if world_size > 1:
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
        control_group = dist.new_group(backend="gloo", timeout=timedelta(hours=2))
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
        if config["reference_kl"]["enabled"]:
            validate_base_reference_model(model)
        parameters = [p for _, p in named]
        if world_size > 1:
            for parameter in parameters:
                dist.broadcast(parameter.data, src=0)
        initial = [p.detach().clone() for p in parameters]
        if hasattr(model, "dcc_quantization_report"):
            record(event="quantization", **model.dcc_quantization_report)
        optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.0)
        scorer, score_recipe = resolve_logprob_backend(args.logprob_backend, args.head_chunk_tokens)
        if getattr(args, "joint_json_policy", False):
            score_recipe = dict(score_recipe,
                implementation_version="joint-raw-thought-packed-grammar-JSON-" + args.logprob_backend + "-v1",
                json_support="actual-public-schema-and-sampled-prefix",
                json_normalization="differentiable-legal-token-logsumexp")
        record(event="scorer", **score_recipe)
        policy = NativeHFPolicy(model, tokenizer, thought_tokens=args.thought_tokens,
            json_tokens=args.json_tokens, max_prompt_tokens=args.max_prompt_tokens,
            batch_size=getattr(args, "native_batch_size", 4),
            sampling_profile="raw", trace_sink=lambda value: enqueue(traces, value),
            capture_thought_logprobs=True,
            mixed_schema_batching=getattr(args, "native_mixed_schema_batching", False),
            joint_json_policy=getattr(args, "joint_json_policy", False))
        runner = make_runner(args)
        if observation_budget is not None:
            runner.set_web_observation_tokenizer(tokenizer)
        # Identical initial replicas; independent sampled trajectories afterwards.
        torch.manual_seed(args.seed + rank * 100003)
        def save(name, iteration):
            save_started = time.monotonic()
            record(event="save_begin", checkpoint_name=name, iteration=iteration, optimizer_updates=updates)
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
            record(event="save_end", checkpoint_name=name, iteration=iteration, optimizer_updates=updates,
                   seconds=time.monotonic() - save_started)
        save("initial", 0)
        for iteration, indices in enumerate(plan, 1):
            task = tasks[indices[rank]]
            begin = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            record(event="collect_begin", iteration=iteration, task_id=task.id, group_size=args.group_size)
            if web_inputs is not None and task.benchmark == "widesearch":
                from slime_plugins.dcc.online_web import collect_public_group, score_closed_group
                closed = await collect_public_group(runner, task, policy,
                    group_size=args.group_size, rank=rank, iteration=iteration,
                    directory=args.output_dir / "web_groups" / f"iteration_{iteration:06d}_rank{rank}",
                    source_commit=args.source_commit)
                results, web_reward = score_closed_group(closed, inputs=web_inputs, tokenizer=tokenizer,
                    reward_variant=getattr(args, "web_train_reward_variant", TABLE_GROUNDED_REWARD))
                record(event="web_group_scored", iteration=iteration, closure=closed.closure,
                       **web_reward)
            else:
                results = await collect_group(runner, task, policy, group_size=args.group_size, rank=rank)
            record(event="collect_end", iteration=iteration, task_id=task.id, episodes=len(results),
                   seconds=time.monotonic() - begin)
            for sample, result in enumerate(results):
                enqueue(rollouts, dict(iteration=iteration, sample=sample,
                                       **result.to_dict(include_tokens=True)))
            if getattr(args, "joint_json_policy", False):
                from slime_plugins.dcc.native_joint_grpo import joint_group_signal
                signal = joint_group_signal(results, group_size=args.group_size,
                    max_steps=args.max_steps, thought_tokens=args.thought_tokens, json_tokens=args.json_tokens,
                    eos_ids=policy.eos_ids, advantage_normalization=config["advantage_normalization"])
            else:
                signal = group_signal(results, group_size=args.group_size,
                    max_steps=args.max_steps, thought_tokens=args.thought_tokens,
                    advantage_normalization=config["advantage_normalization"])
            record(event="signal", iteration=iteration, rewards=signal.rewards, reward_std=signal.reward_std,
                   advantages=signal.advantages, discarded_reason=signal.discarded_reason,
                   advantage_normalization=signal.advantage_normalization)
            replay_started = time.monotonic()
            record(event="replay_begin", iteration=iteration)
            report = replay_group(model, signal, scorer, group_size=args.group_size,
                max_steps=args.max_steps, thought_tokens=args.thought_tokens, clip=args.clip,
                replay_diagnostic=getattr(args, "replay_diagnostic", "full"),
                reference_kl_beta=getattr(args, "reference_kl_beta", 0), reference_base_model=args.model,
                reference_source_commit=args.source_commit,
                joint_json_policy=getattr(args, "joint_json_policy", False), json_tokens=args.json_tokens,
                grammar_compiler=policy.compiler if getattr(args, "joint_json_policy", False) else None,
                logprob_backend=args.logprob_backend,
                head_chunk_tokens=args.head_chunk_tokens,
                policy_surrogate=getattr(args, "policy_surrogate", "token-grpo"),
                json_surrogate_weight=getattr(args, "json_surrogate_weight", 1))
            record(event="replay_end", iteration=iteration, seconds=time.monotonic() - replay_started,
                   scored_tokens=report["scored_tokens"], scored_segments=report["scored_segments"])
            if world_size > 1:
                readiness_started = time.monotonic()
                record(event="update_readiness_begin", iteration=iteration)
                wait_for_update_ranks(world_size=world_size, distributed=dist,
                    control_group=control_group, synchronize=torch.cuda.synchronize)
                record(event="update_readiness_end", iteration=iteration,
                       seconds=time.monotonic() - readiness_started)
            apply_started = time.monotonic()
            record(event="apply_begin", iteration=iteration)
            norm, applied = apply_policy_update(parameters, optimizer, world_size=world_size,
                distributed=dist, max_grad_norm=args.max_grad_norm)
            record(event="apply_end", iteration=iteration, seconds=time.monotonic() - apply_started,
                   optimizer_step_applied=applied, global_gradient_norm=norm)
            updates += int(applied)
            sampled_thought_tokens = sum(
                (s.thought_count if getattr(args, "joint_json_policy", False) else len(s.completion_ids))
                for group in signal.segments for s in group)
            sampled_tokens = {}
            if getattr(args, "joint_json_policy", False):
                sampled_tokens["sampled_json_tokens"] = sum(
                    len(s.json_logprobs) for group in signal.segments for s in group)
            record(event="step", iteration=iteration, optimizer_updates=updates,
                optimizer_step_applied=applied, global_gradient_norm=norm,
                task_id=task.id, statuses=[r.status for r in results], rewards=signal.rewards,
                reward_std=signal.reward_std, advantages=signal.advantages,
                advantage_normalization=signal.advantage_normalization,
                discarded_reason=signal.discarded_reason,
                sampled_thought_tokens=sampled_thought_tokens,
                seconds=time.monotonic() - begin, **sampled_tokens, **report)
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
            complete = dict(complete=True, algorithm=config["algorithm"], source_commit=args.source_commit,
                rollout_iterations=args.steps, optimizer_updates=updates, world_size=world_size,
                has_policy_gradient_updates=updates > 0 and all(s["changed_adapter_tensors"] > 0 for s in summaries),
                ranks=summaries, benchmark_improvement_claimed=False)
            if config["reference_kl"]["enabled"]:
                complete.update(algorithm_variant=config.get("algorithm_variant", REFERENCE_KL_VARIANT),
                                reference_kl=config["reference_kl"])
            if "policy_surrogate" in config:
                complete.update(algorithm_variant=config["algorithm_variant"],
                                policy_surrogate=config["policy_surrogate"])
            if "json_surrogate" in config:
                complete.update(algorithm_variant=config["algorithm_variant"],
                                json_surrogate=config["json_surrogate"])
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
        dist.destroy_process_group(control_group)
        dist.destroy_process_group()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-data", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--enable-online-web", action="store_true",
                        help="Opt in to separately reviewed public-only Web TRAIN groups")
    parser.add_argument("--web-train-data", type=Path)
    parser.add_argument("--web-train-labels", type=Path)
    parser.add_argument("--web-binding", type=Path)
    parser.add_argument("--web-train-reward-variant", choices=ONLINE_REWARD_VARIANTS,
                        default=TABLE_GROUNDED_REWARD,
                        help="Online Web TRAIN only: original whole-table grounding (default), or per-row TP eligibility with full prediction/gold denominators")
    parser.add_argument("--cuad-draws-per-cycle", type=int, default=1)
    parser.add_argument("--web-draws-per-cycle", type=int, default=1)
    parser.add_argument("--cuad-task-sampling",
                        choices=("stratified-3to1", "uniform", "stratified-3to1-span-cap5"),
                        default="stratified-3to1",
                        help="Original TRAIN gold strata (default), a label-blind uniform deck, or opt-in 3:1 strata with positive min(gold span count,5) weights and replacement")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=2)
    parser.add_argument("--advantage-normalization", choices=("std", "centered"), default="std",
                        help="Group advantages: original (reward-mean)/(std+1e-4), or centered reward-mean without std/epsilon division")
    parser.add_argument("--policy-surrogate", choices=POLICY_SURROGATES, default="token-grpo",
                        help="Original token clipping (default), or opt-in fixed-budget episode GSPO; episode-gspo requires reference KL beta 0")
    parser.add_argument("--native-batch-size", type=int, default=4,
                        help="Maximum requests per physical native rollout batch on each rank; logical GRPO group size is unchanged")
    parser.add_argument("--native-mixed-schema-batching", action="store_true",
                        help="Batch native requests with distinct schemas using one grammar matcher per row")
    parser.add_argument("--joint-json-policy", action="store_true",
                        help="CUAD only: fresh raw-thought and grammar-masked stochastic-JSON GRPO; old greedy JSON traces are unsupported")
    parser.add_argument("--json-surrogate-weight", type=float, default=1,
                        help="Default 1 preserves the original loss; nondefault CUAD joint token-GRPO weights JSON policy surrogates only, with the original fixed denominator")
    parser.add_argument("--web-observation-token-budget", type=int, default=None,
                        help="Opt-in Web tool observation JSON token cap, including all auto-reads and metadata; requires --enable-online-web")
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
    parser.add_argument("--cuad-concise-reasoning", action="store_true",
                        help="Opt in to brief CUAD reasoning using only observed source identifiers/character ranges; answer spans and schema are unchanged")
    parser.add_argument("--web-concise-reasoning", action="store_true",
                        help="Opt in to concise Web reasoning focused on the next actual action; budgets, evidence and schemas are unchanged")
    parser.add_argument("--auto-read-sources", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--auto-read-follow-docs", action="store_true",
                        help="Within existing automatic-read budgets, prioritize one observed same-origin Help/API/documentation link per search")
    parser.add_argument("--web-zip-member-fallback", action="store_true",
                        help="Opt in to skipping whole ZIP members only at text character caps; all byte/security limits remain enforced")
    parser.add_argument("--web-text-window-chars", type=int,
                        help="Optional positive fetch_url body-window limit; cached source and original cached-row JSON budget are unchanged")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--quantization", choices=("none", "nf4"), default="nf4")
    parser.add_argument("--compute-profile", choices=COMPUTE_PROFILES, default="legacy")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--reference-kl-beta", type=float, default=0,
                        help="Opt-in frozen-loaded-base K3 on existing nonzero-advantage thought segments only; 0 preserves original GRPO")
    parser.add_argument("--save-every", type=int, default=2)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--logprob-backend", choices=("original", "qwen3-moe-chunked"), default="qwen3-moe-chunked")
    parser.add_argument("--head-chunk-tokens", type=int)
    parser.add_argument("--replay-diagnostic", choices=("full", "first", "off"), default="full",
                        help="Extra eval/no-grad replay diagnostics: every segment, first segment per group, or none; actual behavior/current loss is unchanged")
    parser.add_argument("--sync-artifacts", action="store_true")
    parser.add_argument("--cpu-preflight", action="store_true")
    parser.add_argument("--plan-world-size", type=int, default=1)
    args = parser.parse_args(argv)
    if args.joint_json_policy and args.enable_online_web:
        parser.error("--joint-json-policy supports CUAD only and cannot be combined with --enable-online-web")
    if args.policy_surrogate == "episode-gspo" and args.enable_online_web:
        parser.error("--policy-surrogate episode-gspo supports CUAD only and cannot be combined with --enable-online-web")
    try:
        validate_reference_kl_beta(args.reference_kl_beta)
        validate_policy_surrogate(args.policy_surrogate, args.reference_kl_beta)
        validate_json_surrogate_weight(args.json_surrogate_weight,
            joint_json_policy=args.joint_json_policy, policy_surrogate=args.policy_surrogate,
            online_web=args.enable_online_web)
        web_observation_budget_configuration(args.web_observation_token_budget)
        if args.web_observation_token_budget is not None and not args.enable_online_web:
            raise ValueError("Web observation token budget requires --enable-online-web")
    except ValueError as error:
        parser.error(str(error))
    if args.web_train_reward_variant != TABLE_GROUNDED_REWARD and not args.enable_online_web:
        parser.error("--web-train-reward-variant requires --enable-online-web")
    import re
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        parser.error("--source-commit requires actual complete committed runtime identity")
    if (any(value < 1 for value in (args.steps, args.max_steps, args.thought_tokens, args.json_tokens,
            args.max_prompt_tokens, args.lora_rank, args.plan_world_size, args.native_batch_size))
            or args.group_size < 2 or args.save_every < 0
            or (args.web_text_window_chars is not None and args.web_text_window_chars < 1)
            or not math.isfinite(args.learning_rate) or args.learning_rate <= 0
            or not math.isfinite(args.max_grad_norm) or args.max_grad_norm <= 0
            or not math.isfinite(args.episode_timeout) or args.episode_timeout <= 0
            or not 0 < args.clip < 1):
        parser.error("Invalid online training budgets or optimizer settings")
    tasks = load_tasks(args.train_data)
    if args.enable_online_web:
        validate_source(args.source_commit, online_web=True)
    else:
        validate_source(args.source_commit)
    world_size = args.plan_world_size if args.cpu_preflight else int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 1:
        parser.error("Positive world size required")
    web_paths = (args.web_train_data, args.web_train_labels, args.web_binding)
    web_inputs = None
    if args.enable_online_web:
        if not all(web_paths) or min(args.cuad_draws_per_cycle, args.web_draws_per_cycle) < 1:
            parser.error("Web opt-in requires separate public data, labels, reviewed binding and positive mixture")
        validate_train_tasks(tasks)
        from slime_plugins.dcc.online_web import WebTrainingInputs, mixed_task_plan
        web_inputs = WebTrainingInputs(*web_paths, source_commit=args.source_commit)
        plan = mixed_task_plan(tasks, web_inputs.tasks, steps=args.steps, world_size=world_size,
            seed=args.seed, cuad_draws_per_cycle=args.cuad_draws_per_cycle,
            web_draws_per_cycle=args.web_draws_per_cycle,
            cuad_plan=partial(task_plan, sampling=args.cuad_task_sampling))
        tasks = tasks + web_inputs.tasks
    else:
        if any(web_paths) or args.cuad_draws_per_cycle != 1 or args.web_draws_per_cycle != 1:
            parser.error("Web inputs and mixture settings require --enable-online-web")
        validate_train_tasks(tasks)
        plan = task_plan(tasks, steps=args.steps, world_size=world_size, seed=args.seed,
                         sampling=args.cuad_task_sampling)
    if args.cpu_preflight:
        print(json.dumps(cpu_preflight(args, tasks, plan, world_size), ensure_ascii=False, allow_nan=False))
    else:
        asyncio.run(train(args, tasks, plan, world_size, web_inputs=web_inputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

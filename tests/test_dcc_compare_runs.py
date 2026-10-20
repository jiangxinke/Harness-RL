"""CPU comparisons against official metrics and intentionally corrupted runs."""

from copy import deepcopy
import json
from pathlib import Path
import runpy

import numpy as np
import pytest

from slime_plugins.dcc import Task
from slime_plugins.dcc.evaluation import evaluate_results
from slime_plugins.dcc.metrics import evaluate_cuad

COMPARE = runpy.run_path(str(Path(__file__).parents[1] / "scripts/dcc_compare_runs.py"))


def contract(task_id, contract_id, gold):
    return Task(task_id, "cuad", "validation", "Find clause", document="alpha beta clause",
                gold=gold, metadata={"contract_id": contract_id})


def web(task_id):
    return Task(task_id, "widesearch", "validation", "Find entities", gold=[{"name": "alpha"}],
                metadata={"evaluation": {"required": ["name"], "unique_columns": ["name"],
                          "eval_pipeline": {"name": {"metric": ["exact_match"], "preprocess": ["norm_str"]}}}})


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def protocol(adapter):
    return {"schema": "dcc_paired_run_protocol_v1", "common": {
        "base_model": {"identity": "synthetic/base", "revision": "fixed"},
        "harness": {"revision": "fixed"}, "retrieval": {"source_snapshot": "fixed",
            "retrieval_context": {"context_chars": 0, "max_context_chars": 16000},
            "search_backend": "synthetic", "search_endpoint": None,
            "web_timeout_seconds": 30, "auto_read_sources": 0},
        "scoring": {"widesearch_protocol": "strict_exact_proxy", "table_input_protocol": "csv-inference-v2"},
        "generation": {"backend": "vllm", "quantization": "none", "batch_size": 2, "temperature": 0,
                       "timeout_seconds": 1800, "max_steps": 3, "max_new_tokens": 100,
                       "max_prompt_tokens": 1024, "initial_chunks": 4, "seed": 42}},
        "policy": {"kind": "lora" if adapter else "base", "adapter": adapter}}


def write_run(path, tasks, *, adapter=None, answers=None, statuses=None, world=2):
    path.mkdir()
    answers, statuses = answers or {}, statuses or {}
    identity = {"path": adapter, "config": {"peft_type": "LORA", "r": 8}} if adapter else None
    manifest = {"model": "synthetic/base", "adapter": adapter, "adapter_identity": identity,
                "data": ["synthetic-tasks.jsonl"], "world_size": world, "eval_num_shards": world,
                "max_steps": 3, "max_new_tokens": 100, "max_prompt_tokens": 1024,
                "initial_chunks": 4, "seed": 42, "limit": None, "inference_backend": "vllm",
                "timeout_seconds": 1800,
                "inference_parameters": {"batch_size": 2, "async_engine": True}}
    if any(task.benchmark == "widesearch" for task in tasks):
        manifest.update(search_backend="synthetic", search_endpoint=None, web_timeout_seconds=30)
    runtime = {"backend": "vllm", "model": manifest["model"], "adapter": identity,
               "vllm_version": "synthetic", "async_engine": True}
    predictions = [{"task_id": task.id, "status": statuses.get(task.id, "completed"),
                    "answer": answers.get(task.id, []), "benchmark": task.benchmark, "split": task.split}
                   for task in tasks]
    for rank in range(world):
        write_json(path / f"manifest_rank{rank}.json", manifest)
        write_json(path / f"rank{rank}.done", {"tasks": len(tasks[rank::world]), "world_size": world})
        if tasks[rank::world]:
            name = f"inference_runtime_rank{rank}.json" if world > 1 else "inference_runtime.json"
            write_json(path / name, runtime)
        (path / f"predictions_rank{rank}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in predictions[rank::world]))
    report = evaluate_results(tasks, predictions)
    report.update(generation_manifest=manifest, inference_runtime=runtime,
                  policy_quantization="none", policy_inference_backend="vllm")
    write_json(path / "scores_strict.json", report)
    return COMPARE["load_run"](path, tasks)


def pair(tmp_path, tasks, **trained_kwargs):
    base = write_run(tmp_path / "base", tasks)
    trained = write_run(tmp_path / "trained", tasks, adapter="synthetic/adapter", **trained_kwargs)
    return base, trained, protocol(None), protocol("synthetic/adapter")


def compare_pair(tasks, values, **kwargs):
    return COMPARE["compare"](tasks, *values, resamples=300, seed=47, **kwargs)


@pytest.mark.parametrize("damage", [None, "protocol", "runtime", "manifest", "version", "package", "limits", "browser", "extra", "enabled_type", "backend", "paired_mode"])
def test_browser_reader_requires_complete_matching_protocol_manifest_runtime(tmp_path, damage):
    from slime_plugins.dcc.browser_reader import BROWSER_READER_VERSION, browser_reader_configuration

    values = pair(tmp_path, [web("a")])
    config = {"version": BROWSER_READER_VERSION, "browser_version": "142.0.7444.23",
              "playwright_version": "1.56.0", "cpu_only": True,
              "configuration": browser_reader_configuration(True)}
    for run, declaration in zip(values[:2], values[2:], strict=True):
        for record in (run["manifest"], run["runtime"], declaration["common"]["generation"]):
            record["web_browser_reader"] = deepcopy(config)
    if damage in {"manifest", "runtime"}:
        values[0][damage].pop("web_browser_reader")
    elif damage == "protocol":
        values[2]["common"]["generation"].pop("web_browser_reader")
    elif damage == "version":
        values[0]["manifest"]["web_browser_reader"]["version"] = "unattested"
    elif damage == "package":
        values[0]["manifest"]["web_browser_reader"]["playwright_version"] = "unattested"
    elif damage == "limits":
        values[0]["manifest"]["web_browser_reader"]["configuration"]["limits"]["concurrency"] = 3
    elif damage == "browser":
        values[0]["runtime"]["web_browser_reader"]["browser_version"] = "different"
    elif damage == "extra":
        values[0]["manifest"]["web_browser_reader"]["path"] = "unattested"
    elif damage == "enabled_type":
        values[0]["manifest"]["web_browser_reader"]["configuration"]["enabled"] = 1
    elif damage == "backend":
        values[2]["common"]["generation"]["backend"] = "hf"
    elif damage == "paired_mode":
        for record in (values[0]["manifest"], values[0]["runtime"], values[2]["common"]["generation"]):
            record.pop("web_browser_reader")
    if damage:
        with pytest.raises(ValueError, match="[Bb]rowser|Paired protocols differ"):
            COMPARE["validate_protocols"](*values)
    else:
        assert COMPARE["validate_protocols"](*values) == values[2]["common"]


@pytest.mark.parametrize("damage", [None, "undeclared", "runtime", "manifest", "temperature", "missing_field"])
def test_explicit_sampling_requires_complete_matching_protocol_and_runtime(tmp_path, damage):
    from slime_plugins.dcc.eval_sampling import EvaluationSampling

    tasks = [contract("a", "A", [])]
    values = pair(tmp_path, tasks)
    sampling = EvaluationSampling(temperature=0.7, top_p=0.8, top_k=20, presence_penalty=0.5).to_dict()
    for run, declaration in zip(values[:2], values[2:], strict=True):
        run["manifest"]["evaluation_sampling"] = deepcopy(sampling)
        run["runtime"]["evaluation_sampling"] = deepcopy(sampling)
        declaration["common"]["generation"].update(temperature=0.7, sampling=deepcopy(sampling))
    if damage == "undeclared":
        values[2]["common"]["generation"].pop("sampling")
    elif damage == "runtime":
        values[0]["runtime"]["evaluation_sampling"]["presence_penalty"] = 0
    elif damage == "manifest":
        values[0]["manifest"]["evaluation_sampling"]["top_p"] = 1
    elif damage == "temperature":
        values[2]["common"]["generation"]["temperature"] = 0
    elif damage == "missing_field":
        values[0]["manifest"]["evaluation_sampling"].pop("min_p")
    if damage:
        with pytest.raises(ValueError, match="sampling"):
            COMPARE["validate_protocols"](*values)
    else:
        assert COMPARE["validate_protocols"](*values) == values[2]["common"]


@pytest.mark.parametrize("damage", [None, "protocol", "runtime", "manifest", "version", "backend", "extra", "enabled_type"])
def test_structured_final_requires_complete_matching_protocol_manifest_runtime(tmp_path, damage):
    from slime_plugins.dcc.web_structured import web_structured_configuration

    values = pair(tmp_path, [web("a")])
    for run, declaration in zip(values[:2], values[2:], strict=True):
        config = web_structured_configuration(True)
        run["manifest"]["web_structured_final"] = deepcopy(config)
        run["runtime"]["web_structured_final"] = deepcopy(config)
        declaration["common"]["generation"]["web_structured_final"] = deepcopy(config)
    if damage in {"manifest", "runtime"}:
        values[0][damage].pop("web_structured_final")
    elif damage == "protocol":
        values[2]["common"]["generation"].pop("web_structured_final")
    elif damage == "version":
        values[0]["manifest"]["web_structured_final"]["schema_version"] = "unattested"
    elif damage == "backend":
        values[0]["manifest"]["web_structured_final"]["backend"] = "auto"
    elif damage == "extra":
        values[0]["manifest"]["web_structured_final"]["unknown"] = True
    elif damage == "enabled_type":
        values[0]["manifest"]["web_structured_final"]["enabled"] = 1
    if damage:
        with pytest.raises(ValueError, match="Structured Web final"):
            COMPARE["validate_protocols"](*values)
    else:
        assert COMPARE["validate_protocols"](*values) == values[2]["common"]


@pytest.mark.parametrize("damage", [None, "protocol", "runtime", "manifest", "version", "backend", "extra", "enabled_type", "missing_final", "paired_mode"])
def test_structured_actions_require_matching_versioned_protocol_manifest_runtime(tmp_path, damage):
    from slime_plugins.dcc.web_structured import web_structured_actions_configuration, web_structured_configuration

    values = pair(tmp_path, [web("a")])
    for run, declaration in zip(values[:2], values[2:], strict=True):
        for record in (run["manifest"], run["runtime"], declaration["common"]["generation"]):
            record["web_structured_final"] = web_structured_configuration(True)
            record["web_structured_actions"] = web_structured_actions_configuration(True)
    if damage in {"manifest", "runtime"}:
        values[0][damage].pop("web_structured_actions")
    elif damage == "protocol":
        values[2]["common"]["generation"].pop("web_structured_actions")
    elif damage in {"version", "backend", "extra", "enabled_type"}:
        key, value = {"version": ("schema_version", "unknown"), "backend": ("backend", "auto"),
                      "extra": ("extra", True), "enabled_type": ("enabled", 1)}[damage]
        values[0]["manifest"]["web_structured_actions"][key] = value
    elif damage == "missing_final":
        for record in (values[0]["manifest"], values[0]["runtime"], values[2]["common"]["generation"]):
            record.pop("web_structured_final")
    elif damage == "paired_mode":
        for record in (values[0]["manifest"], values[0]["runtime"], values[2]["common"]["generation"]):
            record.pop("web_structured_actions")
    if damage:
        with pytest.raises(ValueError, match="Structured Web actions|Paired protocols differ"):
            COMPARE["validate_protocols"](*values)
    else:
        assert COMPARE["validate_protocols"](*values) == values[2]["common"]


def test_cuad_aggregate_matches_official_curve_including_parties_zero_confidence_and_nan():
    tasks = [contract("a_Parties", "A", ["ACME"]), contract("a2", "A", ["alpha beta", "beta clause"]),
             contract("b", "B", []), contract("c", "C", ["missing clause"])]
    answers = {
        "a_Parties": [{"text": "ACME extra words", "confidence": 0.9}],
        "a2": [{"text": "alpha beta clause", "confidence": 0.5}, {"text": "unrelated", "confidence": 0.0}],
        "b": [{"text": "false positive", "confidence": 1.0}], "c": []}
    predictions = {key: {"status": "completed", "answer": value} for key, value in answers.items()}
    groups, counts = COMPARE["cuad_cluster_counts"](tasks, predictions)
    assert groups == ["A", "B", "C"]
    # Compare whole set and duplicated-contract resamples to the public scorer,
    # rather than a second copy of the vectorized bootstrap implementation.
    for weights in ([1, 1, 1], [2, 0, 1], [0, 3, 0], [0, 0, 3]):
        gold, selected = {}, {}
        for group, repetitions in zip(groups, weights, strict=True):
            for repetition in range(repetitions):
                for task in tasks:
                    if task.metadata["contract_id"] == group:
                        key = f"rep{repetition}_{task.id}"
                        gold[key], selected[key] = task.gold, answers[task.id]
        expected = evaluate_cuad(gold, selected)
        actual = COMPARE["cuad_aggregate"](
            (np.asarray(weights) @ counts.reshape(3, -1)).reshape(counts.shape[1:]))
        assert actual["aupr"][0] == pytest.approx(expected["aupr"])
        assert actual["micro_f1"][0] == pytest.approx(expected["micro_f1"])


def test_paired_contract_resampling_keeps_related_questions_together(tmp_path, monkeypatch):
    tasks = [contract("a1", "A", ["alpha"]), contract("a2", "A", ["alpha"]), contract("b", "B", ["beta"])]
    values = pair(tmp_path, tasks, answers={"a1": ["alpha"], "a2": ["alpha"]})
    calls = []

    class FixedDraws:
        def multinomial(self, units, probabilities, size):
            calls.append((units, probabilities.tolist()))
            return np.asarray([[2, 0], [1, 1], [0, 2]] * (size // 3 + 1))[:size]

    monkeypatch.setattr(np.random, "default_rng", lambda seed: FixedDraws())
    report = compare_pair(tasks, values)
    result = report["benchmarks"]["cuad"]
    assert calls and all(units == 2 and probabilities == [0.5, 0.5] for units, probabilities in calls)
    assert result["tasks"] == 3 and result["units"] == 2 and result["resampling_unit"] == "contract"
    assert result["metrics"]["micro_f1"]["trained"] == pytest.approx(0.8)
    assert result["metrics"]["micro_f1"]["delta_ci"] == [0, 1]
    assert not result["primary_improvement_supported"]


def test_identical_policies_have_exact_zero_paired_intervals(tmp_path):
    tasks = [contract("a", "A", ["alpha"]), contract("b", "B", [])]
    values = pair(tmp_path, tasks)
    report = compare_pair(tasks, values)
    for metric in report["benchmarks"]["cuad"]["metrics"].values():
        assert metric["delta"] == 0 and metric["delta_ci"] == [0, 0]
    assert not report["all_primary_improvements_supported"]
    assert report == compare_pair(tasks, values)


def test_widesearch_task_pairing_and_model_failures_are_zero_even_if_saved_score_nonzero(tmp_path):
    tasks = [web("a"), web("b"), web("c")]
    values = pair(tmp_path, tasks, answers={task.id: [{"name": "alpha"}] for task in tasks},
                  statuses={"b": "invalid_answer", "c": "evidence_failure"})
    report = compare_pair(tasks, values)
    result = report["benchmarks"]["widesearch"]
    assert result["resampling_unit"] == "task" and result["units"] == 3
    assert result["metrics"]["f1_by_row"]["trained"] == pytest.approx(1 / 3)
    assert result["nonzero_failed_score_overrides"]["trained"] == 1
    assert report["runs"]["trained"]["status_counts"] == {"completed": 1, "invalid_answer": 1, "evidence_failure": 1}


def test_infrastructure_failure_stays_in_denominator_and_blocks_improvement_claim(tmp_path):
    tasks = [web("a"), web("b")]
    values = pair(tmp_path, tasks, answers={"a": [{"name": "alpha"}]}, statuses={"b": "infrastructure_error"})
    result = compare_pair(tasks, values)
    assert result["benchmarks"]["widesearch"]["metrics"]["f1_by_row"]["trained"] == 0.5
    assert result["infrastructure_errors"]["trained"] == 1
    assert not result["eligible_for_improvement_claim"]
    assert not result["all_primary_improvements_supported"]


def test_semantic_comparison_requires_same_complete_frozen_judge_and_declared_protocol(tmp_path):
    tasks = [web("a"), web("b")]
    values = pair(tmp_path, tasks)
    configuration = {"model": "synthetic/judge", "adapter": None, "context": 8192}
    for run, declared in ((values[0], values[2]), (values[1], values[3])):
        path = run["output"] / "scores_strict.json"
        report = json.loads(path.read_text())
        report.update(judge_configuration=configuration, judge_configuration_complete=True, judge_quantization="none")
        report["widesearch"].update(evaluation_protocol="official_protocol_custom_judge", judge_model="synthetic/judge",
                                     alignment_protocol="synthetic-fixed", alignment_protocol_complete=True)
        for row in report["widesearch"]["per_task"]:
            row["alignment_protocol"] = "synthetic-fixed"
        write_json(path, report)
        declared["common"]["scoring"].update(widesearch_protocol="official_protocol_custom_judge", judge_configuration=configuration)
    assert compare_pair(tasks, values)["benchmarks"]["widesearch"]["scoring"]["judge_configuration"] == configuration
    path = values[1]["output"] / "scores_strict.json"
    report = json.loads(path.read_text())
    report["judge_configuration_complete"] = False
    write_json(path, report)
    with pytest.raises(ValueError, match="incomplete judge/alignment"):
        compare_pair(tasks, values)


def test_widesearch_saved_scorer_must_match_declared_protocol(tmp_path):
    tasks = [web("a"), web("b")]
    values = pair(tmp_path, tasks)
    for declared in (values[2], values[3]):
        declared["common"]["scoring"]["widesearch_protocol"] = "official_protocol_custom_judge"
    with pytest.raises(ValueError, match="Declared WideSearch scoring protocol"):
        compare_pair(tasks, values)


@pytest.mark.parametrize("change,match", [
    ("missing", "require table_input_protocol"), ("mixed_rows", "Mixed WideSearch table"),
    ("mixed_runs", "scorer/judge protocols differ"), ("declaration", "Declared WideSearch table"),
])
def test_widesearch_table_ingestion_protocol_must_be_complete_and_matched(tmp_path, change, match):
    tasks = [web("a"), web("b")]
    values = pair(tmp_path, tasks)
    path = values[1]["output"] / "scores_strict.json"
    score = json.loads(path.read_text())
    if change == "missing": del score["widesearch"]["table_input_protocol"]
    elif change == "mixed_rows": score["widesearch"]["per_task"][0]["table_input_protocol"] = "legacy"
    elif change == "mixed_runs":
        score["widesearch"]["table_input_protocol"] = "legacy"
        for row in score["widesearch"]["per_task"]: row["table_input_protocol"] = "legacy"
    elif change == "declaration":
        for declared in values[2:]: declared["common"]["scoring"]["table_input_protocol"] = "legacy"
    write_json(path, score)
    with pytest.raises(ValueError, match=match): compare_pair(tasks, values)


@pytest.mark.parametrize("change,match", [
    ("common", "protocols differ"), ("budget", "saved manifest"), ("timeout", "saved manifest"),
    ("manifest", "beyond the adapter"), ("runtime", "runtime differs"),
    ("adapter", "adapter differs"), ("missing_generation", "include all budgets"),
])
def test_protocol_budget_harness_runtime_and_adapter_mismatches_rejected(tmp_path, change, match):
    tasks = [contract("a", "A", ["alpha"]), contract("b", "B", [])]
    base, trained, base_protocol, trained_protocol = pair(tmp_path, tasks)
    if change == "common": trained_protocol["common"]["harness"]["revision"] = "changed"
    elif change == "budget": trained_protocol["common"]["generation"]["max_new_tokens"] = 99
    elif change == "timeout": trained_protocol["common"]["generation"]["timeout_seconds"] = 10
    elif change == "manifest": trained["manifest"]["extra_setting"] = "changed"
    elif change == "runtime": trained["runtime"]["vllm_version"] = "changed"
    elif change == "adapter": trained_protocol["policy"]["adapter"] = "different/adapter"
    elif change == "missing_generation": del trained_protocol["common"]["generation"]["timeout_seconds"]
    with pytest.raises(ValueError, match=match):
        compare_pair(tasks, (base, trained, base_protocol, trained_protocol))


@pytest.mark.parametrize("change,match", [
    ("timeout", "timeout_seconds differs"), ("web_timeout", "web_timeout_seconds differs"),
    ("search_backend", "search_backend differs"), ("search_endpoint", "search_endpoint differs"),
    ("auto_read_sources", "auto_read_sources differs"), ("retrieval_context", "retrieval_context differs"),
    ("retrieval_profile", "retrieval_profile differs"), ("missing_web", "declare web timeout"),
])
def test_matching_declarations_cannot_override_saved_execution_settings(tmp_path, change, match):
    tasks = [web("a"), web("b")]
    values = pair(tmp_path, tasks)
    for declared in (values[2], values[3]):
        retrieval = declared["common"]["retrieval"]
        if change == "timeout": declared["common"]["generation"]["timeout_seconds"] = 10
        elif change == "web_timeout": retrieval["web_timeout_seconds"] = 10
        elif change == "search_backend": retrieval[change] = "different"
        elif change == "search_endpoint": retrieval[change] = "https://search.invalid"
        elif change == "auto_read_sources": retrieval[change] = 1
        elif change == "retrieval_context": retrieval[change]["context_chars"] = 100
        elif change == "retrieval_profile": retrieval[change] = {"context_chars": 100}
        elif change == "missing_web": del retrieval["web_timeout_seconds"]
    with pytest.raises(ValueError, match=match): compare_pair(tasks, values)


def test_legacy_missing_timeout_fields_are_explicit_operator_declarations(tmp_path):
    tasks = [web("a"), web("b")]
    values = pair(tmp_path, tasks)
    for run in values[:2]:
        for key in ("timeout_seconds", "web_timeout_seconds"):
            del run["manifest"][key]
        score_path = run["output"] / "scores_strict.json"
        score = json.loads(score_path.read_text())
        score["generation_manifest"] = run["manifest"]
        write_json(score_path, score)
    result = compare_pair(tasks, values)
    assert result["legacy_timeout_declarations"] == {
        "baseline": ["timeout_seconds", "web_timeout_seconds"],
        "trained": ["timeout_seconds", "web_timeout_seconds"]}


@pytest.mark.parametrize("change,match", [
    ("cross_shard", "across shards"), ("missing", "coverage"), ("done", "completion marker"),
    ("runtime", "Missing inference runtime"), ("status", "episode status"), ("extra", "coverage"),
])
def test_incomplete_duplicated_or_corrupt_persisted_predictions_are_rejected(tmp_path, change, match):
    tasks = [web("a"), web("b")]
    run = write_run(tmp_path / "run", tasks)
    root = run["output"]
    first = root / "predictions_rank0.jsonl"
    if change == "cross_shard":
        with (root / "predictions_rank1.jsonl").open("a") as stream: stream.write(first.read_text())
    elif change == "missing": first.write_text("")
    elif change == "done": write_json(root / "rank0.done", {"tasks": 9, "world_size": 2})
    elif change == "runtime": (root / "inference_runtime_rank0.json").unlink()
    elif change == "status": write_json(first, {"task_id": "a", "status": "running"})
    elif change == "extra":
        with first.open("a") as stream: stream.write(json.dumps({"task_id": "foreign", "status": "completed", "answer": []}) + "\n")
    with pytest.raises(ValueError, match=match): COMPARE["load_run"](root, tasks)


def test_same_shard_retry_keeps_latest_failure_and_reports_retry_count(tmp_path):
    tasks = [web("a"), web("b")]
    run = write_run(tmp_path / "run", tasks, answers={"a": [{"name": "alpha"}]})
    with (run["output"] / "predictions_rank0.jsonl").open("a") as stream:
        stream.write(json.dumps({"task_id": "a", "status": "evidence_failure", "answer": []}) + "\n")
    loaded = COMPARE["load_run"](run["output"], tasks)
    assert loaded["predictions"]["a"]["status"] == "evidence_failure"
    assert loaded["retry_records"] == 1 and loaded["retried_tasks"] == 1
    with pytest.raises(ValueError, match="status counts differ"):
        COMPARE["load_wide_scores"](loaded, tasks, "scores_strict.json")


@pytest.mark.parametrize("change,match", [
    ("judge_failure", "zero judge failures"), ("score_set", "score task sets"),
    ("duplicate_score", "Duplicate"), ("nan", "Non-finite JSON"),
    ("judge_protocol", "scorer/judge protocols differ"), ("aggregate", "aggregate differs"),
])
def test_widesearch_incomplete_or_mixed_scoring_rejected(tmp_path, change, match):
    tasks = [web("a"), web("b")]
    values = pair(tmp_path, tasks)
    path = values[1]["output"] / "scores_strict.json"
    report = json.loads(path.read_text())
    wide = report["widesearch"]
    if change == "judge_failure": wide["judge_failures"] = 1
    elif change == "score_set": wide["per_task"][0]["task_id"] = "foreign"
    elif change == "duplicate_score": wide["per_task"].append(deepcopy(wide["per_task"][0]))
    elif change == "nan": wide["per_task"][0]["f1_by_row"] = float("nan")
    elif change == "judge_protocol": wide["judge_model"] = "unexpected/judge"
    elif change == "aggregate": wide["f1_by_row"] = 0.7
    write_json(path, report)
    with pytest.raises(ValueError, match=match): compare_pair(tasks, values)


def test_missing_contract_identity_and_single_cluster_are_not_silent_question_bootstraps(tmp_path):
    tasks = [contract("a", "A", ["alpha"])]
    values = pair(tmp_path, tasks, answers={"a": ["alpha"]})
    report = compare_pair(tasks, values)
    assert not report["eligible_for_improvement_claim"]
    missing = [Task("a", "cuad", "validation", "Find clause", document="alpha", gold=["alpha"])]
    with pytest.raises(ValueError, match="contract_id"):
        compare_pair(missing, values)


def test_end_to_end_cli_reads_existing_artifacts_and_writes_finite_json(tmp_path):
    tasks = [contract("a", "A", ["alpha"]), contract("b", "B", ["alpha"])]
    values = pair(tmp_path, tasks, answers={task.id: ["alpha"] for task in tasks})
    data = tmp_path / "tasks.jsonl"
    data.write_text("".join(json.dumps(task.to_dict()) + "\n" for task in tasks))
    for name, value in (("base-protocol", values[2]), ("trained-protocol", values[3])):
        write_json(tmp_path / f"{name}.json", value)
    output = tmp_path / "comparison.json"
    assert COMPARE["main"](["--data", str(data), "--baseline-output", str(values[0]["output"]),
        "--trained-output", str(values[1]["output"]), "--baseline-protocol", str(tmp_path / "base-protocol.json"),
        "--trained-protocol", str(tmp_path / "trained-protocol.json"), "--resamples", "1000", "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["all_primary_improvements_supported"]
    assert report["benchmarks"]["cuad"]["metrics"]["aupr"]["delta_ci"] == [1, 1]
    assert "NaN" not in output.read_text()

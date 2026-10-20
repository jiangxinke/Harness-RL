from __future__ import annotations

import dataclasses
import asyncio
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest
from slime_plugins.dcc import EpisodeRunner, PolicyCompletion, Task
from slime_plugins.dcc.retrieval import RetrievalProfile, fit_retrieval_profile
from slime_plugins.dcc.tools import DocumentTools

SPEC = importlib.util.spec_from_file_location("dcc_prepare_cuad_tools", Path(__file__).resolve().parents[1] / "scripts/dcc_prepare_cuad_tools.py")
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)
QUESTION = 'Highlight the parts related to "Governing Law". Details: Which governing law applies?'
SPAN = "The relationship is controlled exclusively by the laws of Omega Province."


class CharacterTokenizer:
    eos_token_id = 1

    def encode(self, text, *, add_special_tokens=False):
        return [ord(char) + 3 for char in text]

    def decode(self, ids, *, skip_special_tokens=False):
        return "".join(chr(value - 3) for value in ids)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize and add_generation_prompt and not enable_thinking
        return self.encode("".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages))


def tasks(count=3):
    source = (("Governing law applies details reviewed lawyers. " * 34) + "\n") * 6 + SPAN
    return [Task(f"contract_{i}__Governing Law", "cuad", "train", QUESTION, document=source, gold=[SPAN], metadata={"category": "ignored"}) for i in range(count)]


def prepare(rows=None, *, profile=None, **kwargs):
    options = {"tokenizer": CharacterTokenizer(), "episodes": 2, "max_prompt_tokens": 100000, "max_response_tokens": 4096, "batch_tokens": 105000, **kwargs}
    return PREPARE.prepare_records(tasks() if rows is None else rows, profile, **options)


def observation(message):
    return json.loads(message["content"].split("\n", 1)[1].rsplit("\nRemaining turns:", 1)[0])


def test_hidden_positive_teachers_replay_real_tools_and_show_final_evidence():
    records, summary = prepare()
    assert summary["episodes"] == 2 and len(records) == 6
    assert Counter(row["stage"] for row in records) == {"search": 2, "read": 2, "final": 2}
    by_id = {task.id: task for task in tasks()}
    for record in records:
        task = by_id[record["task_id"]]
        tools = DocumentTools(task.document)
        assert record["source"] == "cuad_train" and record["benchmark"] == "cuad" and record["split"] == "train"
        assert record["metadata"]["positive"] is True
        assert record["metadata"]["category"] == "Governing Law"
        assert record["metadata"]["initial_visibility"] == "none"
        assert record["grounding_fields"] == [SPAN]
        initial = observation(record["messages"][2])
        assert all(SPAN not in hit["text"] for hit in initial["hits"])
        pending, hits = None, []
        for message in record["messages"][3:]:
            if message["role"] == "assistant":
                pending = json.loads(message["content"])
            elif pending:
                actual = observation(message)
                if pending["tool"] == "search_document":
                    expected = tools.search_document(**pending["arguments"])
                    hits.extend(actual["hits"])
                else:
                    assert pending["arguments"] in [args for hit in hits for args in PREPARE.read_actions(hit)]
                    expected = tools.read_document(**pending["arguments"])
                assert actual == expected
                pending = None
        target = json.loads(record["completion"])
        if record["stage"] == "search":
            assert target["tool"] == "search_document"
            assert target["arguments"]["query"] in PREPARE.candidate_queries(task.prompt, None)
            assert "Omega Province" not in target["arguments"]["query"]
        elif record["stage"] == "read":
            assert target["tool"] == "read_document"
            assert target["arguments"] in [args for hit in hits for args in PREPARE.read_actions(hit)]
        else:
            assert target["answer"] == [{"text": SPAN, "confidence": 0.9}]
            assert list(target) == ["plan", "checks", "answer"]
            assert "Governing Law" in target["plan"]
            assert all(isinstance(target[key], str) and 0 < len(target[key].split()) <= 40 for key in ("plan", "checks"))
            assert all(SPAN not in target[key] and "ignored" not in target[key] for key in ("plan", "checks"))
            assert PREPARE.FINAL_TURN in record["messages"][-1]["content"]
            assert any(SPAN in observation(m).get("text", "") for m in record["messages"] if m["content"].startswith(PREPARE.OBSERVATION))
            assert SPAN in record["messages"][-2]["content"]
            assert SPAN not in record["messages"][-1]["content"]
            assert PREPARE.FINAL_TURN not in record["messages"][-2]["content"]
    assert summary["encoding_verified"] is True


def test_query_whitelist_uses_question_and_profile_and_is_deterministic():
    profile = RetrievalProfile({QUESTION: ("tribunals", "jurisdiction", "construed")}, 60)
    queries = PREPARE.candidate_queries(QUESTION, profile)
    assert queries == PREPARE.candidate_queries(QUESTION, profile)
    assert len(queries) == len(set(queries))
    assert "tribunals jurisdiction construed" in queries
    assert not any("Omega" in query for query in queries)
    assert "tribunals" not in " ".join(PREPARE.candidate_queries("Unknown category question", profile))


@pytest.mark.parametrize("context_chars", [0, 400])
@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize("episode_max_steps", [3, 4, 6])
@pytest.mark.parametrize("with_profile", [False, True])
def test_teacher_history_matches_live_runner_with_context_on_every_search(context_chars, negative, episode_max_steps, with_profile):
    rows = tasks(1)
    if negative:
        rows = [dataclasses.replace(rows[0], gold=[])]
    profile = RetrievalProfile({QUESTION: ("reviewed",)}, 60) if with_profile else None
    records, summary = prepare(rows, profile=profile, episodes=int(not negative), negative_episodes=int(negative),
                               episode_max_steps=episode_max_steps, retrieval_context_chars=context_chars,
                               retrieval_max_context_chars=16000)
    assert len(records) == 3
    assert summary["retrieval_context_chars"] == context_chars
    assert summary["episode_max_steps"] == episode_max_steps
    assert f"at most {episode_max_steps} model turns" in records[0]["messages"][0]["content"]
    assert records[1]["messages"][-1]["content"].endswith(f"Remaining turns: {episode_max_steps - 1}.")
    final_messages = records[2]["messages"]
    if episode_max_steps == 3:
        assert final_messages[-1] == PREPARE.final_turn_message(rows[0])
        assert final_messages[-2]["content"].endswith("Remaining turns: 1.")
    else:
        assert final_messages[-1]["content"].endswith(f"Remaining turns: {episode_max_steps - 2}.")
        assert not any(PREPARE.FINAL_TURN in message["content"] for message in final_messages)
    assert all(record["metadata"]["retrieval_context_chars"] == context_chars for record in records)

    async def replay():
        calls = 0

        async def policy(messages, **_):
            nonlocal calls
            expected = records[calls]
            assert messages == expected["messages"]
            calls += 1
            return PolicyCompletion(expected["completion"])

        result = await EpisodeRunner(max_steps=episode_max_steps, initial_chunks=4, retrieval_profile=profile,
                                     retrieval_context_chars=context_chars,
                                     retrieval_max_context_chars=16000).run(rows[0], policy, evaluation=True)
        assert calls == 3 and result.status == "completed"
        assert len(result.tool_events) == 2
        assert [event["tool"] for event in result.tool_events] == ["search_document", "read_document"]
        assert len(result.steps) == 3
        if negative:
            assert result.answer == []
        else:
            assert result.answer[0]["text"] == SPAN

    asyncio.run(replay())


@pytest.mark.parametrize("split,benchmark", [("validation", "cuad"), ("test", "cuad"), ("train", "widesearch")])
def test_teacher_preparation_rejects_every_nontraining_input(split, benchmark):
    with pytest.raises(ValueError, match="CUAD train"):
        prepare([*tasks(), dataclasses.replace(tasks()[0], id="forbidden", split=split, benchmark=benchmark, gold=[])])


def test_no_fake_success_for_already_visible_negative_or_missing_labels():
    easy = dataclasses.replace(tasks()[0], document=SPAN)
    negative = dataclasses.replace(tasks()[0], id="negative", gold=[])
    absent = dataclasses.replace(tasks()[0], id="absent", gold=["A span absent from the real source."])
    records, summary = prepare([easy, negative, absent])
    assert not records and summary["episodes"] == 0
    assert summary["counters"]["already_fully_visible"] == 1


def test_final_must_fit_the_actual_encoded_context_and_completion_budget():
    assert prepare(max_prompt_tokens=100)[0] == []
    assert prepare(max_response_tokens=1)[0] == []
    assert prepare(batch_tokens=100)[0] == []

    class EvidenceDroppingTokenizer(CharacterTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return self.encode(messages[1]["content"])

    assert prepare(tokenizer=EvidenceDroppingTokenizer())[0] == []


def test_sampling_deterministic_without_using_arbitrary_category_metadata():
    first, first_summary = prepare(episodes=3)
    second, second_summary = prepare(list(reversed(tasks())), episodes=3)
    assert first == second and first_summary == second_summary
    assert len({record["task_id"] for record in first}) == 3
    with pytest.raises(ValueError, match="duplicate"):
        prepare([tasks()[0], tasks()[0]])


def negative_tasks(count=4):
    return [dataclasses.replace(
        tasks(1)[0], id=f"negative_{category}_{index}", gold=[],
        prompt=QUESTION.replace("Governing Law", category),
    ) for category in ("Governing Law", "Exclusivity") for index in range(count)]


def test_negative_teachers_replay_real_search_and_hit_anchored_read_before_abstaining():
    rows = negative_tasks()
    records, summary = prepare(rows, episodes=0, negative_episodes=6)
    assert summary["positive_episodes"] == 0 and summary["negative_episodes"] == 6
    assert summary["episodes"] == 6 and summary["records"] == 18
    assert summary["negative_categories"] == {"Exclusivity": 3, "Governing Law": 3}
    by_id = {task.id: task for task in rows}
    for record in records:
        assert record["metadata"]["positive"] is False and record["grounding_fields"] == []
        assert record["metadata"]["initial_visibility"] == "negative"
        assert record["split"] == "train"
        if record["stage"] != "final":
            continue
        target = json.loads(record["completion"])
        assert target["answer"] == [] and list(target) == ["plan", "checks", "answer"]
        assert record["metadata"]["category"] in target["plan"]
        assert all(isinstance(target[key], str) and 0 < len(target[key].split()) <= 40 for key in ("plan", "checks"))
        tools = DocumentTools(by_id[record["task_id"]].document)
        search = json.loads(record["messages"][3]["content"])
        searched = observation(record["messages"][4])
        assert searched == tools.search_document(**search["arguments"]) and searched["hits"]
        read = json.loads(record["messages"][5]["content"])
        assert read["arguments"] in [arguments for hit in searched["hits"] for arguments in PREPARE.read_actions(hit)]
        observed_read = observation(record["messages"][6])
        assert observed_read == tools.read_document(**read["arguments"]) and observed_read["text"].strip()
        assert PREPARE.FINAL_TURN not in record["messages"][6]["content"]
        assert record["messages"][7] == PREPARE.final_turn_message(by_id[record["task_id"]])
    repeat, repeated_summary = prepare(list(reversed(rows)), episodes=0, negative_episodes=6)
    assert repeat == records and repeated_summary == summary


def test_review_supervision_uses_public_question_and_keeps_labels_out_of_history():
    positive = tasks(1)[0]
    negative = dataclasses.replace(positive, gold=[])
    records, _ = prepare([positive], episodes=1)
    negatives, _ = prepare([negative], episodes=0, negative_episodes=1)
    positive_final = json.loads(records[-1]["completion"])
    negative_final = json.loads(negatives[-1]["completion"])
    assert positive_final["plan"] == negative_final["plan"]
    assert positive_final["checks"] != negative_final["checks"]
    assert records[-1]["messages"][-1] == negatives[-1]["messages"][-1]
    assert SPAN not in records[-1]["messages"][-1]["content"]
    assert all(set(json.loads(row["completion"])) == {"tool", "arguments"}
               for row in records + negatives if row["stage"] != "final")


def test_optional_negatives_preserve_positive_records_and_enforce_all_budgets():
    rows = tasks() + negative_tasks()
    original, _ = prepare(rows)
    explicit_zero, _ = prepare(rows, negative_episodes=0)
    combined, summary = prepare(rows, negative_episodes=2)
    assert original == explicit_zero == [record for record in combined if record["metadata"]["positive"]]
    assert summary["positive_episodes"] == 2 and summary["negative_episodes"] == 2
    for budgets in ({"max_prompt_tokens": 10}, {"max_response_tokens": 1}, {"batch_tokens": 10}):
        rejected, counts = prepare(negative_tasks(), episodes=0, negative_episodes=1, **budgets)
        assert not rejected and counts["negative_episodes"] == 0
        assert counts["counters"]["no_negative_tool_plan_fits_budgets"] == 8
    empty = dataclasses.replace(negative_tasks(1)[0], document=" \n ")
    assert prepare([empty], episodes=0, negative_episodes=1)[0] == []


@pytest.mark.parametrize("split", ["validation", "test"])
def test_negative_only_generation_rejects_nontraining_tasks(split):
    with pytest.raises(ValueError, match="CUAD train"):
        prepare([dataclasses.replace(negative_tasks(1)[0], split=split)], episodes=0, negative_episodes=1)


@pytest.mark.parametrize("counts", [{"episodes": 0}, {"episodes": -1}, {"negative_episodes": -1}])
def test_episode_counts_reject_negative_or_empty_requests(counts):
    with pytest.raises(ValueError, match="episode counts"):
        prepare(**counts)


def test_cli_outputs_external_records_and_manifest_and_refuses_overwrite(tmp_path):
    source, profile, output = tmp_path / "train.jsonl", tmp_path / "profile.json", tmp_path / "artifacts/teachers.jsonl"
    source.write_text("\n".join(json.dumps(dataclasses.asdict(task)) for task in tasks()))
    profile.write_text(json.dumps(fit_retrieval_profile(tasks(), min_examples=99)))
    args = ["--data", str(source), "--retrieval-profile", str(profile), "--tokenizer", "supplied-in-test", "--output", str(output), "--episodes", "2", "--max-prompt-tokens", "100000", "--max-response-tokens", "4096", "--batch-tokens", "105000"]
    assert PREPARE.main(args, tokenizer=CharacterTokenizer()) == 0
    assert len(output.read_text().splitlines()) == 6
    manifest = json.loads(output.with_suffix(".jsonl.manifest.json").read_text())
    assert manifest["episodes"] == 2 and manifest["split"] == "train"
    with pytest.raises(FileExistsError):
        PREPARE.main(args, tokenizer=CharacterTokenizer())


def test_cli_can_generate_negative_only_teachers(tmp_path):
    source, profile, output = tmp_path / "train.jsonl", tmp_path / "profile.json", tmp_path / "negative.jsonl"
    source.write_text("\n".join(json.dumps(dataclasses.asdict(task)) for task in negative_tasks()))
    profile.write_text(json.dumps(fit_retrieval_profile(tasks(), min_examples=99)))
    args = ["--data", str(source), "--retrieval-profile", str(profile), "--tokenizer", "supplied-in-test", "--output", str(output), "--episodes", "0", "--negative-episodes", "2", "--max-prompt-tokens", "100000", "--max-response-tokens", "4096", "--batch-tokens", "105000"]
    assert PREPARE.main(args, tokenizer=CharacterTokenizer()) == 0
    manifest = json.loads(output.with_suffix(".jsonl.manifest.json").read_text())
    assert manifest["positive_episodes"] == 0 and manifest["negative_episodes"] == 2
    assert len(output.read_text().splitlines()) == 6


@pytest.mark.parametrize("episode_max_steps", [0, 1, 2, -1, True, 3.5, "4"])
def test_teacher_turn_budget_rejects_values_that_cannot_hold_the_real_trace(episode_max_steps):
    with pytest.raises(ValueError, match="episode_max_steps"):
        prepare(episode_max_steps=episode_max_steps)


def test_default_three_turn_records_and_longer_episode_targets_stay_compatible():
    default, default_summary = prepare()
    explicit, explicit_summary = prepare(episode_max_steps=3)
    assert default == explicit and default_summary == explicit_summary
    longer, _ = prepare(episode_max_steps=4)
    assert [{key: value for key, value in record.items() if key != "messages"} for record in longer] == [
        {key: value for key, value in record.items() if key != "messages"} for record in default
    ]
    for old, new in zip(default, longer):
        assert [observation(message) for message in old["messages"] if message["content"].startswith(PREPARE.OBSERVATION)] == [
            observation(message) for message in new["messages"] if message["content"].startswith(PREPARE.OBSERVATION)
        ]


def test_cli_persists_requested_four_turn_budget_without_forcing_early_final(tmp_path):
    rows = [dataclasses.replace(tasks(1)[0], gold=[])]
    data = tmp_path / "train.jsonl"
    data.write_text("\n".join(json.dumps(row.to_dict()) for row in rows) + "\n")
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(fit_retrieval_profile(tasks(), min_examples=1)))
    output = tmp_path / "teachers.jsonl"
    assert PREPARE.main([
        "--data", str(data), "--retrieval-profile", str(profile), "--tokenizer", "unused-local-fixture",
        "--output", str(output), "--episodes", "0", "--negative-episodes", "1",
        "--episode-max-steps", "4", "--max-prompt-tokens", "100000",
        "--max-response-tokens", "4096", "--batch-tokens", "105000",
    ], tokenizer=CharacterTokenizer()) == 0
    manifest = json.loads(output.with_suffix(".jsonl.manifest.json").read_text())
    assert manifest["episode_max_steps"] == 4 and manifest["episodes"] == 1
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(records) == 3 and json.loads(records[-1]["completion"])["answer"] == []
    assert records[-1]["messages"][-1]["content"].endswith("Remaining turns: 2.")
    assert not any(PREPARE.FINAL_TURN in message["content"] for message in records[-1]["messages"])

import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest

from slime_plugins.dcc.episode import PolicyCompletion
from slime_plugins.dcc.graphpo_collector import chat_tokens
from slime_plugins.dcc.graphpo_runtime import (
    EmbeddingInputTooLong, GraphPOSummarizer, SentenceEmbedder, SummaryValidationError, SummaryCapacityError,
    BGE_REVISION, GTE_REVISION, verify_embedding_model,
    claim_candidates, state_candidates, remaining_decisions,
    summary_selection_schema, summary_selection_regex,
)


class Tokenizer:
    def encode(self, text, **kwargs):
        return [ord(character) for character in text]

    def apply_chat_template(self, messages, **kwargs):
        return self.encode(json.dumps(messages))


TOKENIZER = Tokenizer()


def _accepts_selection(schema, text):
    lmfe = pytest.importorskip("lmformatenforcer")
    parser = lmfe.RegexParser(summary_selection_regex(schema))
    for character in text:
        if character not in parser.get_allowed_characters():
            return False
        parser = parser.add_character(character)
    return parser.can_end()


def test_actual_lmfe_selection_enforces_domain_enum_empty_arrays_and_item_cap():
    candidates = {"facts": [{"id": index} for index in range(1, 70)],
                  "unresolved": [{"id": index} for index in range(1, 7)], "claims": []}
    schema = summary_selection_schema(candidates)
    assert _accepts_selection(schema, '{"fact_ids":[19,66],"unresolved_ids":[1,6],"claim_ids":[]}')
    assert _accepts_selection(schema, '{"fact_ids":[19,19],"unresolved_ids":[],"claim_ids":[]}')
    for text in (
        '{"fact_ids":[],"unresolved_ids":[19],"claim_ids":[]}',
        '{"fact_ids":[],"unresolved_ids":[],"claim_ids":[1]}',
        '{"fact_ids":[1,2,3,4,5,6,7,8,9],"unresolved_ids":[],"claim_ids":[]}',
        '{"fact_ids":[70],"unresolved_ids":[],"claim_ids":[]}',
        '{"fact_ids":[true],"unresolved_ids":[],"claim_ids":[]}',
        '{"fact_ids":[1.0],"unresolved_ids":[],"claim_ids":[]}',
        '{"fact_ids":[],"unresolved_ids":[],"claim_ids":[],"reward":1}',
    ):
        assert not _accepts_selection(schema, text), text


class ConstrainedSummaryPolicy:
    def __init__(self, output):
        self.output, self.calls = output, []
        self.summary_constraint_usage = dict(setup_seconds=0.0)

    async def __call__(self, *args, **kwargs):
        raise AssertionError("The constrained summary must not use the free actor call")

    async def generate_summary(self, messages, *, selection_schema, max_new_tokens, temperature):
        self.calls.append((messages, selection_schema))
        self.summary_constraint_usage["setup_seconds"] += 0.25
        return PolicyCompletion(self.output, chat_tokens(TOKENIZER, messages), TOKENIZER.encode(self.output))


def test_constrained_summary_logs_duplicate_selections_and_restores_source_once():
    messages = history()
    messages[1]["content"] = "Initial contract retrieval (source material):\n" + json.dumps({
        "hits": [{"text": "PROMOTION AGREEMENT.\nRenewal applies."}]})
    policy = ConstrainedSummaryPolicy('{"fact_ids":[1,1],"unresolved_ids":[],"claim_ids":[]}')
    summarizer = GraphPOSummarizer(policy, TOKENIZER)
    result = json.loads(asyncio.run(summarizer(messages)))
    assert result["facts"] == ["PROMOTION AGREEMENT."]
    record = summarizer.records[0]
    assert record["selected_fact_ids"] == [1, 1]
    assert record["selection_normalization"]["fact_ids"]["duplicate_ids"] == [1]
    assert record["selection_normalization"]["fact_ids"]["retained_ids"] == [1]
    assert record["selection_decoder_version"] == "0.11.3"
    assert record["selection_max_items_per_domain"] == 8
    assert record["constraint_setup_seconds"] == 0.25
    assert record["resolved_facts"] == result["facts"]
    assert policy.calls[0][1]["properties"]["claim_ids"]["maxItems"] == 0


def test_constrained_backend_violation_is_not_filtered_or_replaced():
    policy = ConstrainedSummaryPolicy('{"fact_ids":[1],"unresolved_ids":[19],"claim_ids":[]}')
    summarizer = GraphPOSummarizer(policy, TOKENIZER, max_attempts=1)
    with pytest.raises(SummaryValidationError, match="violated its exact constrained"):
        asyncio.run(summarizer(history()))
    assert summarizer.records[0]["output"] == policy.output
    assert not summarizer.records[0]["valid"]
    assert summarizer.usage["failed_states"] == 1


def history():
    return [{"role": "user", "content": "Find the contract title."},
            {"role": "user", "content": "Initial contract retrieval (source material):\n" + json.dumps({
                "hits": [{"start": 0, "end": 20, "text": "PROMOTION AGREEMENT."}]})}]


def valid(**updates):
    return json.dumps({"facts": ["PROMOTION AGREEMENT."], "unresolved": [], "agent_claims": [],
                       "last_action": "none", "stage": "initial", **updates})


def policy_with(outputs, calls):
    async def policy(messages, *, max_new_tokens, temperature):
        calls.append(dict(messages=messages, limit=max_new_tokens, temperature=temperature))
        value = outputs[len(calls) - 1]
        return PolicyCompletion(value, chat_tokens(TOKENIZER, messages), TOKENIZER.encode(value))
    return policy


def test_summary_uses_zero_temperature_exact_context_and_counts_actual_tokens():
    calls = []
    summary = GraphPOSummarizer(policy_with([valid()], calls), TOKENIZER)
    result = asyncio.run(summary(history()))
    assert json.loads(result)["facts"] == ["PROMOTION AGREEMENT."]
    assert calls[0]["temperature"] == 0
    assert summary.usage["calls"] == 1
    assert summary.usage["input_tokens"] == len(summary.records[0]["prompt_ids"])
    assert summary.usage["output_tokens"] == len(summary.records[0]["completion_ids"])
    assert summary.records[0]["valid"]
    assert json.loads(calls[0]["messages"][1]["content"])["public_history"] == history()


def test_initial_summary_does_not_ask_model_to_invent_nonexistent_agent_claims():
    calls = []
    output = json.dumps({"facts": ["PROMOTION AGREEMENT."], "unresolved": []})
    summary = GraphPOSummarizer(policy_with([output], calls), TOKENIZER)
    result = json.loads(asyncio.run(summary(history())))
    assert result["agent_claims"] == [] and result["submitted_answer"] is None
    instructions = calls[0]["messages"][0]["content"]
    assert '"fact_ids" and "unresolved_ids".' in instructions
    assert '"agent_claims" is a list' not in instructions
    assert json.loads(calls[0]["messages"][1]["content"])["latest_assistant_message"] is None


def test_invalid_summary_is_logged_and_retried_without_fallback():
    calls = []
    summary = GraphPOSummarizer(policy_with(["not JSON", valid()], calls), TOKENIZER)
    result = asyncio.run(summary(history()))
    assert json.loads(result)["facts"]
    assert summary.usage["validation_failures"] == 1
    assert summary.usage["calls"] == 2
    assert len(summary.records) == 2
    assert not summary.records[0]["valid"]
    assert summary.records[1]["valid"]
    assert "rejected" in calls[1]["messages"][-1]["content"]


@pytest.mark.parametrize("output", [
    valid(facts=["HALLUCINATED CONTRACT"]), valid(facts="PROMOTION AGREEMENT."),
    valid(unresolved=["x" * 161]), valid(reward=1),
    '{"facts":[],"facts":[],"unresolved":[],"last_action":"none","stage":"initial"}',
])
def test_unverified_or_malformed_summary_fails_closed(output):
    calls = []
    summary = GraphPOSummarizer(policy_with([output, output], calls), TOKENIZER)
    with pytest.raises(SummaryValidationError, match="after 2 attempts"):
        asyncio.run(summary(history()))
    assert summary.usage["failed_states"] == 1
    assert summary.usage["validation_failures"] == 2
    assert len(summary.records) == 2
    assert all(not record["valid"] for record in summary.records)


def test_assistant_assertion_is_not_a_source_fact():
    messages = history() + [{"role": "assistant", "content": "The contract is named HALLUCINATED CONTRACT."}]
    output = valid(facts=["HALLUCINATED CONTRACT"], last_action="invalid_output", stage="active")
    summary = GraphPOSummarizer(policy_with([output], []), TOKENIZER, max_attempts=1)
    with pytest.raises(SummaryValidationError, match="quote observed"):
        asyncio.run(summary(messages))


def test_unresolved_cannot_introduce_unobserved_36_month_requirement():
    output = valid(unresolved=["Verify that the warranty lasts 36 months."])
    summary = GraphPOSummarizer(policy_with([output], []), TOKENIZER, max_attempts=1)
    with pytest.raises(SummaryValidationError, match="original task question"):
        asyncio.run(summary(history()))


def test_unresolved_can_quote_the_actual_task_question():
    output = valid(unresolved=["contract title"])
    result = asyncio.run(GraphPOSummarizer(policy_with([output], []), TOKENIZER)(history()))
    assert json.loads(result)["unresolved"] == ["contract title"]


def test_audited_final_control_and_action_phase_are_preserved():
    from slime_plugins.dcc.cuad_audit_protocol import FINAL_CONTROL
    messages = history() + [
        {"role": "assistant", "content": '{"tool":"search_document","arguments":{"query":"title"}}'},
        {"role": "user", "content": FINAL_CONTROL},
    ]
    output = valid(last_action="search_document", stage="final")
    result = asyncio.run(GraphPOSummarizer(policy_with([output], []), TOKENIZER)(messages))
    assert json.loads(result)["stage"] == "final"


@pytest.mark.parametrize("action", [
    '[{"answer":[]}]', '{"answer":[],"answer":[]}',
    '{"tool":"search_document","arguments":{"query":"title","index":1}}',
])
def test_audited_rejected_actions_do_not_become_final_or_valid_summary_actions(action):
    messages = history() + [
        {"role": "assistant", "content": action},
        {"role": "user", "content": 'Tool observation (source material):\n{"error":"Invalid action"}\nRemaining turns: 2.'},
    ]
    output = valid(last_action="invalid_output", stage="active")
    calls = []
    result = asyncio.run(GraphPOSummarizer(policy_with([output], calls), TOKENIZER)(messages))
    assert json.loads(result)["last_action"] == "invalid_output"
    request = json.loads(calls[0]["messages"][1]["content"])
    assert request["observed_stage"] == "active"
    assert request["observed_last_action"] == "invalid_output"


@pytest.mark.parametrize("action", ['[{"answer":[]}]', '{"answer":[],"answer":[]}'])
def test_real_runner_rejection_leaves_summary_state_active(action):
    from slime_plugins.dcc.data import Task
    from slime_plugins.dcc.episode import EpisodeRunner
    from slime_plugins.dcc.graphpo_collector import expand_once
    from slime_plugins.dcc.credit_adapter import source_payload
    task = Task("real__Document Name", "cuad", "train", "Find the contract name.",
                "PROMOTION AGREEMENT.", ["PROMOTION AGREEMENT"])
    runner = EpisodeRunner(max_steps=3, initial_chunks=1, cuad_protocol="cuad-audited-v1")

    async def actor(messages, **kwargs):
        return PolicyCompletion(action, chat_tokens(TOKENIZER, messages), TOKENIZER.encode(action))

    expansion = asyncio.run(expand_once(task, runner, actor, tokenizer=TOKENIZER))
    assert expansion.result is None
    assert source_payload(expansion.next_messages[-1]["content"])["error"]
    output = valid(last_action="invalid_output", stage="active")
    result = asyncio.run(GraphPOSummarizer(policy_with([output], []), TOKENIZER)(expansion.next_messages))
    assert json.loads(result)["stage"] == "active"


def test_real_runner_tool_failure_is_distinct_from_successful_action():
    from slime_plugins.dcc.data import Task
    from slime_plugins.dcc.episode import EpisodeRunner
    from slime_plugins.dcc.graphpo_collector import expand_once
    task = Task("real__Document Name", "cuad", "train", "Find the contract name.",
                "PROMOTION AGREEMENT.", ["PROMOTION AGREEMENT"])
    runner = EpisodeRunner(max_steps=3, initial_chunks=1, cuad_protocol="cuad-audited-v1")
    action = '{"tool":"read_document","arguments":{"start":10000,"end":11000}}'

    async def actor(messages, **kwargs):
        return PolicyCompletion(action, chat_tokens(TOKENIZER, messages), TOKENIZER.encode(action))

    expansion = asyncio.run(expand_once(task, runner, actor, tokenizer=TOKENIZER))
    output = valid(last_action="tool_error:read_document", stage="active")
    result = asyncio.run(GraphPOSummarizer(policy_with([output], []), TOKENIZER)(expansion.next_messages))
    assert json.loads(result)["last_action"] == "tool_error:read_document"


@pytest.mark.parametrize("protocol", ["cuad-audited-v1", "cuad-audited-v2"])
@pytest.mark.parametrize("action", ["[]", '[{"text":"PROMOTION AGREEMENT","confidence":1}]'])
def test_answer_array_summary_stage_matches_real_runner_protocol(protocol, action):
    from slime_plugins.dcc.data import Task
    from slime_plugins.dcc.episode import EpisodeRunner
    from slime_plugins.dcc.graphpo_collector import expand_once
    sample = Task("real__Document Name", "cuad", "train", "Find the contract name.",
                  "PROMOTION AGREEMENT.", ["PROMOTION AGREEMENT"])
    runner = EpisodeRunner(max_steps=3, initial_chunks=1, cuad_protocol=protocol)

    async def actor(messages, **kwargs):
        return PolicyCompletion(action, chat_tokens(TOKENIZER, messages), TOKENIZER.encode(action))

    expansion = asyncio.run(expand_once(sample, runner, actor, tokenizer=TOKENIZER))
    accepted = protocol == "cuad-audited-v2"
    if accepted:
        assert expansion.result.status == "completed"
    else:
        assert expansion.result is None
    output = valid(last_action="submit_answer" if accepted else "invalid_output",
                   stage="final" if accepted else "active")
    summarizer = GraphPOSummarizer(policy_with([output], []), TOKENIZER, protocol=protocol)
    result = asyncio.run(summarizer(expansion.next_messages))
    assert json.loads(result)["stage"] == ("final" if accepted else "active")


@pytest.mark.parametrize("action", [
    '[{"tool":"search_document","arguments":{"query":"title"}}]',
    '["PROMOTION AGREEMENT"]', '[{"answer":[]}]', '{"answer":[],"answer":[]}',
])
def test_v2_tool_arrays_and_ambiguous_answers_stay_invalid_in_real_runner_and_summary(action):
    from slime_plugins.dcc.data import Task
    from slime_plugins.dcc.episode import EpisodeRunner
    from slime_plugins.dcc.graphpo_collector import expand_once
    sample = Task("real__Document Name", "cuad", "train", "Find the contract name.",
                  "PROMOTION AGREEMENT.", ["PROMOTION AGREEMENT"])
    runner = EpisodeRunner(max_steps=3, initial_chunks=1, cuad_protocol="cuad-audited-v2")

    async def actor(messages, **kwargs):
        return PolicyCompletion(action, chat_tokens(TOKENIZER, messages), TOKENIZER.encode(action))

    expansion = asyncio.run(expand_once(sample, runner, actor, tokenizer=TOKENIZER))
    assert expansion.result is None
    summarizer = GraphPOSummarizer(policy_with([valid(last_action="invalid_output", stage="active")], []),
                                  TOKENIZER, protocol="cuad-audited-v2")
    result = asyncio.run(summarizer(expansion.next_messages))
    assert json.loads(result)["last_action"] == "invalid_output"


@pytest.mark.parametrize("protocol", ["cuad-audited-v2", "cuad-audited-v3"])
@pytest.mark.parametrize("action", [
    'Finish: {"answer":[]}', '```json\n{"answer":[]}\n```', 'Finish: []',
])
def test_wrapped_finish_summary_matches_runner_protocol(protocol, action):
    from slime_plugins.dcc.data import Task
    from slime_plugins.dcc.episode import EpisodeRunner
    from slime_plugins.dcc.graphpo_collector import expand_once
    sample = Task("real__Document Name", "cuad", "train", "Find the contract name.",
                  "PROMOTION AGREEMENT.", ["PROMOTION AGREEMENT"])
    runner = EpisodeRunner(max_steps=3, initial_chunks=1, cuad_protocol=protocol)

    async def actor(messages, **kwargs):
        return PolicyCompletion(action, chat_tokens(TOKENIZER, messages), TOKENIZER.encode(action))

    expansion = asyncio.run(expand_once(sample, runner, actor, tokenizer=TOKENIZER))
    accepted = protocol == "cuad-audited-v3"
    assert (expansion.result is not None) is accepted
    output = valid(last_action="submit_answer" if accepted else "invalid_output",
                   stage="final" if accepted else "active")
    result = asyncio.run(GraphPOSummarizer(policy_with([output], []), TOKENIZER,
                                           protocol=protocol)(expansion.next_messages))
    assert json.loads(result)["stage"] == ("final" if accepted else "active")


@pytest.mark.parametrize("action", [
    'Search: {"answer":[]}', 'Finish: {"tool":"search_document","arguments":{"query":"title"}}',
    'Finish: {"answer":[]} {"answer":[]}', 'Answer follows: {"answer":[]}',
])
def test_v3_mismatched_or_ambiguous_wrapper_never_marks_summary_final(action):
    messages = history() + [
        {"role": "assistant", "content": action},
        {"role": "user", "content": 'Tool observation (source material):\n{"error":"Invalid action"}\nRemaining turns: 2.'},
    ]
    output = valid(last_action="invalid_output", stage="active")
    result = asyncio.run(GraphPOSummarizer(policy_with([output], []), TOKENIZER,
                                           protocol="cuad-audited-v3")(messages))
    assert json.loads(result)["stage"] == "active"


def test_summary_prompt_truncation_is_rejected_and_recorded():
    async def policy(messages, **kwargs):
        return PolicyCompletion(valid(), [1, 2, 3], TOKENIZER.encode(valid()))

    summary = GraphPOSummarizer(policy, TOKENIZER, max_attempts=1)
    with pytest.raises(SummaryValidationError, match="truncated"):
        asyncio.run(summary(history()))
    assert summary.records[0]["prompt_ids"] == [1, 2, 3]
    assert summary.usage["input_tokens"] == 3


def test_input_rejects_hidden_metadata_fields():
    summary = GraphPOSummarizer(policy_with([], []), TOKENIZER)
    messages = history()
    messages[0]["gold"] = "HIDDEN_REFERENCE"
    with pytest.raises(ValueError, match="public role/content"):
        asyncio.run(summary(messages))
    assert summary.usage["calls"] == 0


def fake_embedder(torch, *, overlong=False):
    class EmbedTokenizer:
        def encode(self, text, **kwargs):
            assert kwargs["truncation"] is False
            return list(range(513 if overlong else 4))

        def __call__(self, texts, **kwargs):
            assert kwargs["truncation"] is False
            return {"input_ids": torch.ones((len(texts), 4), dtype=torch.long)}

    class Model:
        def __call__(self, **encoded):
            values = torch.zeros((encoded["input_ids"].shape[0], 4, 2))
            values[:, 0] = torch.tensor([3.0, 4.0])
            values[:, 1:] = 1000.0
            return SimpleNamespace(last_hidden_state=values)

    embedder = SentenceEmbedder.__new__(SentenceEmbedder)
    embedder.tokenizer = EmbedTokenizer()
    embedder.model = Model()
    embedder.max_tokens = 512
    embedder.provenance = {"repository": "BAAI/bge-small-en-v1.5",
                          "model_card": "https://huggingface.co/BAAI/bge-small-en-v1.5"}
    embedder.usage = dict(input_tokens=0, calls=0, texts=0, seconds=0.0,
                         overlength_rejections=0, truncated_inputs=0)
    embedder.records = []
    embedder._lock = asyncio.Lock()
    return embedder


def test_bge_uses_cls_not_mean_pooling_and_l2_normalizes():
    torch = pytest.importorskip("torch")
    embedder = fake_embedder(torch)
    vectors = asyncio.run(embedder(["summary one", "summary two"]))
    assert vectors[0] == pytest.approx([0.6, 0.8])
    assert vectors[1] == pytest.approx([0.6, 0.8])
    assert embedder.records[0]["device"] == "cpu"
    assert embedder.usage["input_tokens"] == 8
    assert embedder.usage["truncated_inputs"] == 0


def test_bge_rejects_overlong_summary_without_truncation():
    torch = pytest.importorskip("torch")
    embedder = fake_embedder(torch, overlong=True)
    with pytest.raises(EmbeddingInputTooLong, match="513"):
        asyncio.run(embedder(["too long"]))
    assert embedder.usage["overlength_rejections"] == 1
    assert embedder.usage["input_tokens"] == 0
    assert embedder.usage["truncated_inputs"] == 0
    assert "error" in embedder.records[0]


def embedding_fixture(path, monkeypatch):
    from slime_plugins.dcc import graphpo_runtime
    files = {
        "model.safetensors": b"fixture weights",
        "config.json": json.dumps({"model_type": "bert", "hidden_size": 384,
                                   "max_position_embeddings": 512}).encode(),
        "tokenizer.json": b"fixture tokenizer",
    }
    pins = {name: hashlib.sha256(value).hexdigest() for name, value in files.items()}
    monkeypatch.setattr(graphpo_runtime, "BGE_FILE_HASHES", pins)
    for name, content in files.items():
        (path / name).write_bytes(content)
    manifest = {"repository": "BAAI/bge-small-en-v1.5", "revision": BGE_REVISION,
                "files": [{"file": name, "sha256": pins[name], "bytes": len(value)}
                          for name, value in files.items()]}
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_embedding_provenance_checks_actual_weight_and_config_content(tmp_path, monkeypatch):
    embedding_fixture(tmp_path, monkeypatch)
    provenance = verify_embedding_model(tmp_path)
    assert provenance["revision"] == BGE_REVISION
    assert len(provenance["verified_files"]) == 3
    (tmp_path / "model.safetensors").write_bytes(b"tampered weights")
    with pytest.raises(ValueError, match="file content"):
        verify_embedding_model(tmp_path)


def test_embedding_provenance_rejects_other_model_or_revision(tmp_path, monkeypatch):
    manifest = embedding_fixture(tmp_path, monkeypatch)
    manifest["revision"] = "wrong revision"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="pinned BGE"):
        verify_embedding_model(tmp_path)


def test_state_controls_are_derived_and_model_written_controls_are_logged():
    calls = []
    output = valid(last_action="read_document", stage="final")
    summarizer = GraphPOSummarizer(policy_with([output], calls), TOKENIZER)
    summary = json.loads(asyncio.run(summarizer(history())))
    assert summary["last_action"] == "none"
    assert summary["stage"] == "initial"
    assert summarizer.records[0]["ignored_model_controls"] == {"last_action": "read_document", "stage": "final"}
    assert summarizer.records[0]["derived_controls"] == {"last_action": "none", "stage": "initial", "submitted_answer": None,
                                                       "remaining_decisions": None}
    assert summarizer.records[0]["runtime_version"] == "graphpo-constrained-extractive-state-v7"
    assert summarizer.usage["validation_failures"] == 0


def test_model_can_return_only_semantic_fields():
    output = json.dumps({"facts": ["PROMOTION AGREEMENT."], "unresolved": ["contract title"], "agent_claims": []})
    summary = json.loads(asyncio.run(GraphPOSummarizer(policy_with([output], []), TOKENIZER)(history())))
    assert set(summary) == {"facts", "unresolved", "agent_claims", "last_action", "stage", "submitted_answer", "remaining_decisions"}
    assert summary["last_action"] == "none"


def test_long_grounded_quotation_uses_embedding_tokens_not_character_limit():
    quote = "The receiving party shall keep the confidential information secure. " * 6
    messages = history()
    messages[-1]["content"] = "Initial contract retrieval (source material):\n" + json.dumps({"text": quote})

    class WordTokenizer:
        def encode(self, text, *, add_special_tokens, truncation):
            assert add_special_tokens and truncation is False
            return list(range(len(text.split()) + 2))

    output = json.dumps({"facts": [quote], "unresolved": [], "agent_claims": []})
    summarizer = GraphPOSummarizer(policy_with([output], []), TOKENIZER,
                                   embedding_tokenizer=WordTokenizer(), embedding_max_tokens=512)
    summary = json.loads(asyncio.run(summarizer(messages)))
    assert len(quote) > 200
    assert summary["facts"] == [quote]
    assert summarizer.records[0]["embedding_tokens"] < 512
    assert summarizer.records[0]["embedding_capacity_checked"]


def test_embedding_overlength_retries_shorter_grounded_summary_without_truncation():
    quote = "PROMOTION AGREEMENT. " + "The parties agree to maintain confidentiality. " * 10
    messages = history()
    messages[-1]["content"] = "Initial contract retrieval (source material):\n" + json.dumps({"text": quote})

    class CharTokenizer:
        def __init__(self):
            self.seen = []

        def encode(self, text, *, add_special_tokens, truncation):
            assert add_special_tokens and truncation is False
            self.seen.append(text)
            return list(range(len(text) + 2))

    embed_tokenizer = CharTokenizer()
    outputs = [json.dumps({"facts": [quote], "unresolved": [], "agent_claims": []}),
               json.dumps({"facts": ["PROMOTION AGREEMENT."], "unresolved": [], "agent_claims": []})]
    summarizer = GraphPOSummarizer(policy_with(outputs, []), TOKENIZER,
                                   embedding_tokenizer=embed_tokenizer, embedding_max_tokens=220)
    result = asyncio.run(summarizer(messages))
    assert json.loads(result)["facts"] == ["PROMOTION AGREEMENT."]
    assert len(embed_tokenizer.seen) == 3
    assert quote in json.loads(embed_tokenizer.seen[1])["facts"]
    assert summarizer.records[0]["embedding_tokens"] > 220
    assert not summarizer.records[0]["valid"]
    assert summarizer.records[1]["embedding_tokens"] <= 220
    assert summarizer.records[1]["valid"]
    assert summarizer.usage["output_tokens"] == sum(len(record["completion_ids"]) for record in summarizer.records)


def test_same_evidence_different_submitted_answers_produce_different_states():
    outputs = []
    for text in ("PROMOTION AGREEMENT.", "MAINTENANCE AGREEMENT."):
        messages = history() + [{"role": "assistant", "content": json.dumps({
            "answer": [{"text": text, "confidence": 0.9}]})}]
        summary = asyncio.run(GraphPOSummarizer(policy_with([valid()], []), TOKENIZER,
                                                protocol="cuad-audited-v3")(messages))
        outputs.append(json.loads(summary))
    assert outputs[0]["facts"] == outputs[1]["facts"]
    assert outputs[0]["last_action"] == outputs[1]["last_action"] == "submit_answer"
    assert outputs[0]["submitted_answer"] == [{"text": "PROMOTION AGREEMENT.", "confidence": 0.9}]
    assert outputs[1]["submitted_answer"] == [{"text": "MAINTENANCE AGREEMENT.", "confidence": 0.9}]
    assert outputs[0] != outputs[1]


def test_empty_submission_is_distinct_from_no_submission():
    pending = json.loads(asyncio.run(GraphPOSummarizer(policy_with([valid()], []), TOKENIZER)(history())))
    ended = history() + [{"role": "assistant", "content": "[]"}]
    completed = json.loads(asyncio.run(GraphPOSummarizer(policy_with([valid()], []), TOKENIZER,
                                                       protocol="cuad-audited-v3")(ended)))
    assert pending["submitted_answer"] is None
    assert completed["submitted_answer"] == []


def test_public_agent_claim_is_retained_as_claim_but_not_source_fact():
    claim = "I believe this contract does not require further review."
    messages = history() + [{"role": "assistant", "content": claim}]
    summary = json.loads(asyncio.run(GraphPOSummarizer(policy_with([valid(agent_claims=[claim])], []),
                                                      TOKENIZER)(messages)))
    assert summary["agent_claims"] == [claim]
    assert claim not in summary["facts"]
    with pytest.raises(SummaryValidationError, match="observed contract"):
        asyncio.run(GraphPOSummarizer(policy_with([valid(facts=[claim])], []), TOKENIZER,
                                     max_attempts=1)(messages))


@pytest.mark.parametrize("claim", ["The warranty lasts 36 months.", "PROMOTION AGREEMENT."])
def test_claim_must_come_from_assistant_role_not_source_or_invention(claim):
    with pytest.raises(SummaryValidationError, match="actual public assistant"):
        asyncio.run(GraphPOSummarizer(policy_with([valid(agent_claims=[claim])], []), TOKENIZER,
                                     max_attempts=1)(history()))


def test_claim_can_quote_decoded_public_action_text():
    text = 'Clause "A" is applicable.'
    messages = history() + [{"role": "assistant", "content": json.dumps({
        "answer": [{"text": text, "confidence": 0.8}]})}]
    summary = json.loads(asyncio.run(GraphPOSummarizer(policy_with([valid(agent_claims=[text])], []),
                                                      TOKENIZER)(messages)))
    assert summary["agent_claims"] == [text]
    assert summary["submitted_answer"][0]["text"] == text


def test_model_cannot_replace_the_actual_submitted_answer():
    messages = history() + [{"role": "assistant", "content": '{"answer":[]}'}]
    output = valid(submitted_answer=[{"text": "invented replacement", "confidence": 1}])
    summarizer = GraphPOSummarizer(policy_with([output], []), TOKENIZER)
    summary = json.loads(asyncio.run(summarizer(messages)))
    assert summary["submitted_answer"] == []
    assert summarizer.records[0]["ignored_model_controls"]["submitted_answer"][0]["text"] == "invented replacement"


def test_required_answer_over_embedding_capacity_fails_before_generation_without_truncation():
    text = "a long publicly submitted answer " * 40
    messages = history() + [{"role": "assistant", "content": json.dumps({
        "answer": [{"text": text, "confidence": 1}]})}]

    class CharTokenizer:
        def encode(self, text, *, add_special_tokens, truncation):
            assert add_special_tokens and truncation is False
            return list(range(len(text) + 2))

    calls = []
    summarizer = GraphPOSummarizer(policy_with([], calls), TOKENIZER,
                                   embedding_tokenizer=CharTokenizer(), embedding_max_tokens=512)
    with pytest.raises(SummaryCapacityError, match="refusing to truncate"):
        asyncio.run(summarizer(messages))
    assert calls == []
    assert summarizer.usage["calls"] == 0
    assert summarizer.usage["capacity_failures"] == 1
    assert not summarizer.records[0]["generation_submitted"]
    assert summarizer.records[0]["derived_controls"]["submitted_answer"][0]["text"] == text


def selection_output(ids, **updates):
    return json.dumps({"facts": ["PROMOTION AGREEMENT."], "unresolved": [], "claim_ids": ids, **updates})


def test_claim_id_selection_restores_exact_excerpt_and_records_origin():
    statement = "I think the contract is long.\nI still need to verify the title."
    messages = history() + [{"role": "assistant", "content": statement}]
    candidates = claim_candidates(messages, "cuad-audited-v3")
    chosen = next(candidate for candidate in candidates if candidate["text"] == "I still need to verify the title.")
    calls = []
    summarizer = GraphPOSummarizer(policy_with([selection_output([chosen["id"]])], calls), TOKENIZER,
                                   protocol="cuad-audited-v3")
    result = json.loads(asyncio.run(summarizer(messages)))
    assert result["agent_claims"] == [chosen["text"]]
    request = json.loads(calls[0]["messages"][1]["content"])
    assert request["claim_candidates"] == [{key: candidate[key] for key in ("id", "text", "message_index", "role")}
                                           for candidate in candidates]
    record = summarizer.records[0]
    assert record["claim_candidates"] == candidates
    assert record["selected_claim_ids"] == [chosen["id"]]
    assert record["resolved_agent_claims"] == [chosen["text"]]
    assert record["claim_selection_mode"] == "ids"
    assert chosen["message_index"] == 3
    assert statement[chosen["source_start"]:chosen["source_end"]] == chosen["text"]
    assert chosen["source_text_sha256"] == hashlib.sha256(statement.encode()).hexdigest()


@pytest.mark.parametrize("ids", [[0], [999], [True], [1.0], ["1"], [1, 1], "1"])
def test_invalid_claim_ids_fail_without_text_fallback(ids):
    messages = history() + [{"role": "assistant", "content": "Need to check the title."}]
    summarizer = GraphPOSummarizer(policy_with([selection_output(ids)], []), TOKENIZER, max_attempts=1)
    with pytest.raises(SummaryValidationError, match="distinct integer IDs"):
        asyncio.run(summarizer(messages))
    assert summarizer.records[0]["requested_claim_ids_json"] == json.dumps(ids)
    assert not summarizer.records[0]["valid"]


def test_claim_ids_cannot_be_mixed_with_legacy_agent_claims():
    messages = history() + [{"role": "assistant", "content": "Need to check the title."}]
    summarizer = GraphPOSummarizer(policy_with([selection_output([1], agent_claims=[])], []), TOKENIZER,
                                   max_attempts=1)
    with pytest.raises(SummaryValidationError, match="Do not mix"):
        asyncio.run(summarizer(messages))


def test_candidates_exclude_user_sources_and_task_instructions():
    messages = history() + [{"role": "assistant", "content": "I need more evidence."}]
    candidates = claim_candidates(messages, "cuad-audited-v3")
    assert candidates
    assert all(messages[item["message_index"] - 1]["role"] == "assistant" for item in candidates)
    assert not any("PROMOTION AGREEMENT" in item["text"] or "Find the contract title" in item["text"] for item in candidates)


def test_parsed_action_candidates_keep_exact_unescaped_value_and_path():
    answer = 'Clause "A" applies. Another clause is unresolved.'
    message = json.dumps({"answer": [{"text": answer, "confidence": 1}]})
    messages = history() + [{"role": "assistant", "content": message}]
    candidates = claim_candidates(messages, "cuad-audited-v3")
    candidate = next(item for item in candidates if item["text"] == 'Clause "A" applies.'
                     and item["source_kind"] == "assistant_parsed_string")
    assert candidate["json_path"] == ["answer", 0, "text"]
    assert answer[candidate["source_start"]:candidate["source_end"]] == candidate["text"]
    summarizer = GraphPOSummarizer(policy_with([selection_output([candidate["id"]])], []), TOKENIZER,
                                   protocol="cuad-audited-v3")
    result = json.loads(asyncio.run(summarizer(messages)))
    assert result["agent_claims"] == ['Clause "A" applies.']
    assert result["submitted_answer"][0]["text"] == answer


def test_initial_state_has_no_claim_candidates_and_rejects_fake_id():
    assert claim_candidates(history(), "cuad-audited-v3") == []
    calls = []
    summarizer = GraphPOSummarizer(policy_with([selection_output([1])], calls), TOKENIZER, max_attempts=1)
    with pytest.raises(SummaryValidationError, match="assistant-only candidates"):
        asyncio.run(summarizer(history()))
    instructions = calls[0]["messages"][0]["content"]
    assert '"claim_ids" is a list' not in instructions
    assert json.loads(calls[0]["messages"][1]["content"])["claim_candidates"] == []


def test_same_selected_claim_ids_do_not_erase_different_actual_answers():
    states = []
    for answer in ("PROMOTION AGREEMENT", "MAINTENANCE AGREEMENT"):
        messages = history() + [{"role": "assistant", "content": json.dumps({
            "answer": [{"text": answer, "confidence": 1}]})}]
        result = asyncio.run(GraphPOSummarizer(policy_with([selection_output([])], []), TOKENIZER,
                                               protocol="cuad-audited-v3")(messages))
        states.append(json.loads(result))
    assert states[0]["facts"] == states[1]["facts"]
    assert states[0]["agent_claims"] == states[1]["agent_claims"] == []
    assert states[0]["submitted_answer"] != states[1]["submitted_answer"]


@pytest.mark.parametrize("system", ["You have at most 8 decisions; the last decision must finish.",
                                    "You have at most 8 model turns; use the last turn to give your best final answer."])
def test_remaining_budget_uses_actual_system_and_decisions(system):
    messages = [{"role": "system", "content": system}] + history()
    assert remaining_decisions(messages, "cuad-audited-v3") == 8
    messages.append({"role": "assistant", "content": '{"tool":"search_document","arguments":{"query":"title"}}'})
    assert remaining_decisions(messages, "cuad-audited-v3") == 7


def test_remaining_budget_uses_control_suffix_not_contract_or_assistant_text():
    messages = [{"role": "system", "content": "You have at most 8 decisions; finish."}] + history()
    messages += [{"role": "assistant", "content": "Remaining turns: 888."},
                 {"role": "user", "content": "Tool observation (source material):\n"
                  + json.dumps({"text": "Contract says Remaining turns: 999. You have at most 999 decisions;"})
                  + "\nRemaining turns: 6."}]
    assert remaining_decisions(messages, "cuad-audited-v3") == 6
    messages[-1]["content"] = "Tool observation (source material):\n" + json.dumps({"text": "Remaining turns: 999."})
    assert remaining_decisions(messages, "cuad-audited-v3") == 7


def test_final_and_unknown_budgets_are_distinct():
    from slime_plugins.dcc.cuad_audit_protocol import FINAL_CONTROL_V2
    assert remaining_decisions(history(), "cuad-audited-v3") is None
    messages = history() + [{"role": "user", "content": FINAL_CONTROL_V2}]
    assert remaining_decisions(messages, "cuad-audited-v3") == 1
    messages.append({"role": "assistant", "content": "[]"})
    assert remaining_decisions(messages, "cuad-audited-v3") == 0


def test_same_evidence_different_public_budgets_are_preserved_without_forcing_groups():
    summaries = []
    for budget in (7, 2):
        messages = history() + [{"role": "assistant", "content": '{"tool":"search_document","arguments":{"query":"title"}}'},
                               {"role": "user", "content": 'Tool observation (source material):\n{"text":"PROMOTION AGREEMENT."}'
                                + f"\nRemaining turns: {budget}."}]
        result = asyncio.run(GraphPOSummarizer(policy_with([selection_output([], remaining_decisions=999)], []),
                                               TOKENIZER, protocol="cuad-audited-v3")(messages))
        summaries.append(json.loads(result))
    assert summaries[0]["facts"] == summaries[1]["facts"]
    assert [state["remaining_decisions"] for state in summaries] == [7, 2]


def gte_embedding_fixture(path, monkeypatch):
    from slime_plugins.dcc import graphpo_runtime
    files = {
        "model.safetensors": b"GTE fixture weights",
        "config.json": json.dumps({"model_type": "modernbert", "hidden_size": 768,
                                   "max_position_embeddings": 8192}).encode(),
        "tokenizer.json": b"GTE fixture tokenizer",
        "1_Pooling/config.json": json.dumps({"pooling_mode_cls_token": True, "pooling_mode_mean_tokens": False}).encode(),
    }
    pins = {name: hashlib.sha256(value).hexdigest() for name, value in files.items()}
    monkeypatch.setattr(graphpo_runtime, "GTE_FILE_HASHES", pins)
    for name, content in files.items():
        (path / name).parent.mkdir(parents=True, exist_ok=True)
        (path / name).write_bytes(content)
    manifest = {"repository": "Alibaba-NLP/gte-modernbert-base", "revision": GTE_REVISION,
                "files": [{"file": name, "sha256": pins[name], "bytes": len(value)}
                          for name, value in files.items()]}
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_long_embedding_profile_verifies_fixed_revision_and_8192_config(tmp_path, monkeypatch):
    gte_embedding_fixture(tmp_path, monkeypatch)
    provenance = verify_embedding_model(tmp_path)
    assert provenance["revision"] == GTE_REVISION
    assert provenance["max_input_tokens"] == 8192
    assert provenance["embedding_dimension"] == 768
    assert provenance["pooling"] == "CLS"
    assert len(provenance["verified_files"]) == 4
    (tmp_path / "1_Pooling/config.json").write_bytes(b"tampered pooling")
    with pytest.raises(ValueError, match="file content"):
        verify_embedding_model(tmp_path)


@pytest.mark.parametrize("repository,architecture,dimension,context", [
    ("BAAI/bge-small-en-v1.5", "bert", 384, 512),
    ("Alibaba-NLP/gte-modernbert-base", "modernbert", 768, 8192),
])
def test_encoder_constructor_uses_native_cpu_sdpa_and_verified_context(tmp_path, monkeypatch, repository, architecture, dimension, context):
    import sys
    from slime_plugins.dcc import graphpo_runtime
    config = SimpleNamespace(max_position_embeddings=context)
    if architecture == "modernbert":
        config.reference_compile = None
    calls = {}
    model = SimpleNamespace(config=config)
    model.to = lambda device: calls.update(device=device) or model
    model.eval = lambda: calls.update(eval=True) or model
    model.requires_grad_ = lambda value: calls.update(requires_grad=value)

    def load_model(*args, **kwargs):
        calls["load"] = kwargs
        return model

    monkeypatch.setattr(graphpo_runtime, "verify_embedding_model", lambda path: {
        "repository": repository, "model_type": architecture, "embedding_dimension": dimension,
        "max_input_tokens": context, "model_card": "verified model card"})
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float32="FP32"))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **k: config),
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace(model_max_length=10**30)),
        AutoModel=SimpleNamespace(from_pretrained=load_model)))
    encoder = SentenceEmbedder(tmp_path)
    assert encoder.max_tokens == context
    assert calls["device"] == "cpu"
    assert calls["eval"] and calls["requires_grad"] is False
    assert calls["load"]["dtype"] == "FP32"
    assert calls["load"]["attn_implementation"] == "sdpa"
    assert calls["load"]["local_files_only"] and calls["load"]["trust_remote_code"] is False
    assert calls["load"]["use_safetensors"]
    if architecture == "modernbert":
        assert calls["load"]["config"].reference_compile is False


@pytest.mark.parametrize("context", [512, 8192])
def test_each_encoder_rejects_its_own_overlength_input_without_silent_truncation(context, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    encoder = SentenceEmbedder.__new__(SentenceEmbedder)
    encoder.max_tokens = context
    encoder.provenance = {"model_card": "verified", "repository": "verified"}
    seen = []

    def encode(text, *, add_special_tokens, truncation):
        assert add_special_tokens and truncation is False
        seen.append(text)
        return list(range(context + 1))

    encoder.tokenizer = SimpleNamespace(encode=encode)
    encoder.records = []
    encoder.usage = dict(calls=0, overlength_rejections=0, seconds=0, input_tokens=0, texts=0, truncated_inputs=0)
    with pytest.raises(EmbeddingInputTooLong, match=str(context)):
        encoder._encode(["retain this full text"])
    assert seen == ["retain this full text"]
    assert encoder.usage["overlength_rejections"] == 1
    assert encoder.usage["truncated_inputs"] == 0


def extractive_output(fact_ids=None, unresolved_ids=None, claim_ids=None, **updates):
    value = {"fact_ids": [1] if fact_ids is None else fact_ids,
             "unresolved_ids": [] if unresolved_ids is None else unresolved_ids, **updates}
    if claim_ids is not None:
        value["claim_ids"] = claim_ids
    return json.dumps(value)


def test_extractive_domains_restore_exact_public_text_and_full_audit_provenance():
    messages = history() + [{"role": "assistant", "content": "I need to verify the actual title."}]
    candidates = state_candidates(messages, "cuad-audited-v3")
    calls = []
    summarizer = GraphPOSummarizer(policy_with([extractive_output([1], [1], [1])], calls), TOKENIZER,
                                   protocol="cuad-audited-v3")
    state = json.loads(asyncio.run(summarizer(messages)))
    assert state["facts"] == ["PROMOTION AGREEMENT."]
    assert state["unresolved"] == ["Find the contract title."]
    assert state["agent_claims"] == ["I need to verify the actual title."]
    request = json.loads(calls[0]["messages"][1]["content"])
    for field in ("fact_candidates", "unresolved_candidates", "claim_candidates"):
        assert all(set(candidate) == {"id", "text", "message_index", "role"} for candidate in request[field])
        assert all("source_text_sha256" in candidate for candidate in summarizer.records[0][field])
    fact = candidates["facts"][0]
    assert fact["message_index"] == 2 and fact["role"] == "user"
    assert fact["json_path"] == ["hits", 0, "text"]
    assert fact["document_start"] == 0 and fact["document_end"] == 20
    assert fact["source_text_sha256"] == hashlib.sha256("PROMOTION AGREEMENT.".encode()).hexdigest()
    assert summarizer.records[0]["selected_fact_ids"] == [1]
    assert summarizer.records[0]["selected_unresolved_ids"] == [1]
    assert summarizer.records[0]["resolved_facts"] == state["facts"]


@pytest.mark.parametrize("field,value", [("fact_ids", [0]), ("fact_ids", [999]), ("fact_ids", [True]),
                                         ("unresolved_ids", [1.0]), ("unresolved_ids", [1, 1])])
def test_invalid_fact_or_question_ids_are_rejected(field, value):
    arguments = {"fact_ids": [1], "unresolved_ids": [], field: value}
    summarizer = GraphPOSummarizer(policy_with([json.dumps(arguments)], []), TOKENIZER, max_attempts=1)
    with pytest.raises(SummaryValidationError, match="distinct integer IDs"):
        asyncio.run(summarizer(history()))
    assert field in summarizer.records[0]["requested_selection_ids_json"]


@pytest.mark.parametrize("text_key,id_key", [("facts", "fact_ids"), ("unresolved", "unresolved_ids")])
def test_ids_and_legacy_text_cannot_mix_within_same_domain(text_key, id_key):
    arguments = {"fact_ids": [1], "unresolved_ids": [], text_key: []}
    summarizer = GraphPOSummarizer(policy_with([json.dumps(arguments)], []), TOKENIZER, max_attempts=1)
    with pytest.raises(SummaryValidationError, match=f"Do not mix {id_key}"):
        asyncio.run(summarizer(history()))


def test_task_question_and_assistant_claims_never_become_fact_candidates():
    messages = [{"role": "user", "content": "Find the warranty duration of 36 months."},
                {"role": "assistant", "content": "I believe the warranty lasts 24 months."}]
    pools = state_candidates(messages, "cuad-audited-v3")
    assert pools["facts"] == []
    assert pools["unresolved"][0]["text"] == messages[0]["content"]
    assert pools["claims"][0]["text"] == messages[1]["content"]
    summarizer = GraphPOSummarizer(policy_with([extractive_output([1], [], [])], []), TOKENIZER,
                                   max_attempts=1)
    with pytest.raises(SummaryValidationError, match="observed-source candidates"):
        asyncio.run(summarizer(messages))


def test_fact_candidate_offsets_are_contiguous_and_not_joined_across_sources():
    source1 = "The warranty lasts 24 months.\nExceptions apply."
    source2 = "The warranty lasts 36 months."
    messages = [{"role": "user", "content": "Check warranty."},
                {"role": "user", "content": "Initial contract retrieval (source material):\n" + json.dumps({
                    "hits": [{"text": source1, "start": 10, "end": 10 + len(source1)},
                             {"text": source2, "start": 200, "end": 200 + len(source2)}]})}]
    pools = state_candidates(messages, "cuad-audited-v3")
    for candidate in pools["facts"]:
        source = [source1, source2][candidate["json_path"][1]]
        assert source[candidate["source_start"]:candidate["source_end"]] == candidate["text"]
        assert candidate["source_text_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert [candidate["text"] for candidate in pools["facts"]] == [
        "The warranty lasts 24 months.", "Exceptions apply.", "The warranty lasts 36 months."]


def test_root_extractive_summary_needs_no_assistant_claim_field():
    calls = []
    summarizer = GraphPOSummarizer(policy_with([extractive_output([1], [1])], calls), TOKENIZER)
    result = json.loads(asyncio.run(summarizer(history())))
    assert result["agent_claims"] == []
    assert json.loads(calls[0]["messages"][1]["content"])["claim_candidates"] == []
    assert '"fact_ids" and "unresolved_ids".' in calls[0]["messages"][0]["content"]


def test_extractive_selection_cannot_rewrite_the_actual_final_answer():
    original = [{"text": "PROMOTION AGREEMENT", "confidence": 0.9}]
    messages = history() + [{"role": "assistant", "content": json.dumps({"answer": original})}]
    output = extractive_output([1], [], [], submitted_answer=[{"text": "invented", "confidence": 1}])
    summarizer = GraphPOSummarizer(policy_with([output], []), TOKENIZER, protocol="cuad-audited-v3")
    result = json.loads(asyncio.run(summarizer(messages)))
    assert result["submitted_answer"] == original
    assert summarizer.records[0]["ignored_model_controls"]["submitted_answer"][0]["text"] == "invented"

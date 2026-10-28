import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest

from slime_plugins.dcc.episode import PolicyCompletion
from slime_plugins.dcc.graphpo_collector import chat_tokens
from slime_plugins.dcc.graphpo_runtime import (
    EmbeddingInputTooLong, GraphPOSummarizer, SentenceEmbedder, SummaryValidationError, SummaryCapacityError,
    BGE_REVISION, verify_embedding_model,
)


class Tokenizer:
    def encode(self, text, **kwargs):
        return [ord(character) for character in text]

    def apply_chat_template(self, messages, **kwargs):
        return self.encode(json.dumps(messages))


TOKENIZER = Tokenizer()


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
    assert summarizer.records[0]["derived_controls"] == {"last_action": "none", "stage": "initial", "submitted_answer": None}
    assert summarizer.records[0]["runtime_version"] == "graphpo-source-and-agent-claims-v3"
    assert summarizer.usage["validation_failures"] == 0


def test_model_can_return_only_semantic_fields():
    output = json.dumps({"facts": ["PROMOTION AGREEMENT."], "unresolved": ["contract title"], "agent_claims": []})
    summary = json.loads(asyncio.run(GraphPOSummarizer(policy_with([output], []), TOKENIZER)(history())))
    assert set(summary) == {"facts", "unresolved", "agent_claims", "last_action", "stage", "submitted_answer"}
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
                                   embedding_tokenizer=embed_tokenizer, embedding_max_tokens=180)
    result = asyncio.run(summarizer(messages))
    assert json.loads(result)["facts"] == ["PROMOTION AGREEMENT."]
    assert len(embed_tokenizer.seen) == 3
    assert quote in json.loads(embed_tokenizer.seen[1])["facts"]
    assert summarizer.records[0]["embedding_tokens"] > 180
    assert not summarizer.records[0]["valid"]
    assert summarizer.records[1]["embedding_tokens"] <= 180
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

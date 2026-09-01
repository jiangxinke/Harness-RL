"""Tests for SP CentralAgent decision adapter and factory."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from deerflow.sp import CENTRAL_AGENT_ACTION_PROMPT, CentralAgentDecider, CentralDecisionRequest, SPAction, create_sp_action_loop


class FakeCentralModel:
    def __init__(self, content: str, *, reasoning_content: str | None = None):
        self.content = content
        self.reasoning_content = reasoning_content
        self.calls: list[list[object]] = []

    def invoke(self, messages):
        self.calls.append(messages)
        additional_kwargs = {"reasoning_content": self.reasoning_content} if self.reasoning_content is not None else {}
        return AIMessage(content=self.content, additional_kwargs=additional_kwargs)


def test_central_agent_decider_parses_action_json_and_uses_bounded_context():
    model = FakeCentralModel(
        """```json
{"action_id":"act-think","action_type":"THINK","reason":"Need a plan","task":"Plan next step"}
```"""
    )
    request = CentralDecisionRequest(
        system_prompt=CENTRAL_AGENT_ACTION_PROMPT,
        task_context="<sp-task-context>current_stage: planning</sp-task-context>",
        state={"messages": [HumanMessage(content="Please migrate StackPlanner")]},
        iteration=1,
        thread_id="thread-1",
        run_id="run-1",
    )

    action_dict = CentralAgentDecider(model=model).decide(request)
    action = SPAction.from_dict(action_dict)

    assert action.action_type == "THINK"
    assert action.action_id == "act-think"
    assert len(model.calls) == 1
    messages = model.calls[0]
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)
    assert "normal agent loop" in messages[0].content
    assert "sp_delegate" in messages[0].content
    assert "current_stage: planning" in messages[1].content
    assert "Please migrate StackPlanner" in messages[1].content


def test_central_agent_decider_accepts_free_form_output_for_direct_turns():
    model = FakeCentralModel("I should think first")
    request = CentralDecisionRequest(
        system_prompt=CENTRAL_AGENT_ACTION_PROMPT,
        task_context="<sp-task-context />",
        state={},
        iteration=1,
    )

    action = SPAction.from_dict(CentralAgentDecider(model=model).decide(request))

    assert action.action_type == "FINISH"
    assert action.task == "I should think first"


def test_central_agent_decider_uses_reasoning_content_when_vllm_content_is_empty():
    model = FakeCentralModel(
        "",
        reasoning_content='{"action_id":"act-finish","action_type":"FINISH","reason":"done","task":"Completed","metadata":{"allow_without_artifact":true}}',
    )
    request = CentralDecisionRequest(
        system_prompt=CENTRAL_AGENT_ACTION_PROMPT,
        task_context="<sp-task-context />",
        state={},
        iteration=1,
    )

    action = SPAction.from_dict(CentralAgentDecider(model=model).decide(request))

    assert action.action_id == "act-finish"
    assert action.action_type == "FINISH"


def test_central_agent_decider_selects_final_action_from_noisy_model_output():
    model = FakeCentralModel('Considered {"note":"not the action"}. Final: {"action_id":"act-final","action_type":"FINISH","reason":"done","task":"Completed","metadata":{"allow_without_artifact":true}}')
    request = CentralDecisionRequest(
        system_prompt=CENTRAL_AGENT_ACTION_PROMPT,
        task_context="<sp-task-context />",
        state={},
        iteration=1,
    )

    action = SPAction.from_dict(CentralAgentDecider(model=model).decide(request))

    assert action.action_id == "act-final"
    assert action.action_type == "FINISH"


def test_create_sp_action_loop_uses_toolless_central_decider_model():
    model = FakeCentralModel('{"action_id":"act-finish","action_type":"FINISH","reason":"Report is ready","task":"Done"}')
    loop = create_sp_action_loop(model=model, max_iterations=1)

    result = loop.run({"sp_current_artifact_refs": {"report": {"artifact_id": "report-1"}}})

    assert result.next_step == "finish"
    assert len(model.calls) == 1
    assert len(model.calls[0]) == 2

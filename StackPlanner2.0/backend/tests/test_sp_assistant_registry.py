"""Gateway assistant registration tests for StackPlanner 2.0."""

import pytest

from app.gateway.routers.assistants_compat import _list_assistants, get_assistant_graph, get_assistant_schemas


def test_stackplanner_is_registered_next_to_unchanged_lead_agent():
    by_id = {assistant.assistant_id: assistant for assistant in _list_assistants()}

    assert by_id["lead_agent"].graph_id == "lead_agent"
    assert by_id["stackplanner"].graph_id == "stackplanner"
    assert by_id["stackplanner"].name == "StackPlanner 2.0"
    assert by_id["stackplanner"].metadata["runtime"] == "deerflow"


@pytest.mark.asyncio
async def test_stackplanner_graph_and_schema_endpoints_report_sp_graph_id():
    graph = await get_assistant_graph("stackplanner")
    schemas = await get_assistant_schemas("stackplanner")

    assert graph["graph_id"] == "stackplanner"
    assert schemas["graph_id"] == "stackplanner"

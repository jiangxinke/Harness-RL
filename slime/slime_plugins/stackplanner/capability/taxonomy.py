"""Capability taxonomy shared by annotation, probing, and experiment C."""

from __future__ import annotations

CAPABILITY_TAXONOMY_VERSION = "stackplanner-central-action-output-v2"
CAPABILITIES = (
    "action_choice",
    "action_output",
)

ACTION_CHOICE_TOOLS = frozenset(
    {
        "sp_think",
        "sp_delegate",
        "sp_recall_memory",
        "sp_reflect",
        "sp_revise",
        "sp_backtrack",
        "sp_replan",
        "sp_summarize",
        "sp_ask_human",
        "sp_finish",
    }
)

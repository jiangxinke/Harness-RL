"""Select the hash-locked decoupling taxonomy for the active experiment."""

from __future__ import annotations

import os

from .all_agent_output import ALL_AGENT_OUTPUT_CAPABILITIES
from .taxonomy import CAPABILITIES


def active_capabilities() -> tuple[str, ...]:
    scope = os.environ.get("SP_DECOUPLING_SCOPE", "central-capability")
    if scope == "all-agent-action-output":
        return ALL_AGENT_OUTPUT_CAPABILITIES
    if scope == "central-capability":
        return CAPABILITIES
    raise RuntimeError(f"unsupported active decoupling scope: {scope!r}")

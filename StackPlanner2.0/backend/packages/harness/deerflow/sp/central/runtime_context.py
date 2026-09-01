"""DR2-owned personalization context for the SP CentralAgent."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deerflow.config.agents_config import AgentConfig, load_agent_config, load_agent_soul, validate_agent_name
from deerflow.config.app_config import AppConfig

logger = logging.getLogger(__name__)

DEFAULT_SOUL_MAX_CHARS = 6000
DEFAULT_SKILL_INDEX_MAX_CHARS = 7000
DEFAULT_SKILL_DESCRIPTION_MAX_CHARS = 320
BUILTIN_SP_SOUL_PATH = Path(__file__).with_name("SOUL.md")


@dataclass(frozen=True, slots=True)
class SPCentralRuntimeContext:
    """Bounded DR2 context attached to one SP graph run.

    ``system_prompt_section`` contains trusted runtime policy and explicit
    agent configuration. ``decision_context`` remains for compatibility with
    older graph factories but is intentionally empty: long-term memory enters
    CentralAgent only through the explicit RECALL_MEMORY action.
    """

    agent_name: str | None = None
    agent_model: str | None = None
    system_prompt_section: str = ""
    decision_context: str = ""
    available_skill_names: frozenset[str] | None = None


def _clip(value: Any, *, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    suffix = "...<truncated>"
    return f"{text[: max_chars - len(suffix)]}{suffix}"


def _load_builtin_soul() -> str:
    try:
        return BUILTIN_SP_SOUL_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        logger.warning("Failed to load built-in StackPlanner SOUL", exc_info=True)
        return ""


def _load_profile(agent_name: str | None, *, user_id: str | None) -> AgentConfig | None:
    validated_name = validate_agent_name(agent_name)
    if validated_name is None:
        return None
    return load_agent_config(validated_name, user_id=user_id)


def _load_soul(agent_name: str | None, *, user_id: str | None) -> str:
    try:
        configured_soul = load_agent_soul(agent_name, user_id=user_id)
    except OSError:
        logger.warning("Failed to load StackPlanner SOUL for agent %r", agent_name, exc_info=True)
        configured_soul = None
    return _clip(configured_soul or _load_builtin_soul(), max_chars=DEFAULT_SOUL_MAX_CHARS)


def _load_skills(
    app_config: AppConfig,
    *,
    user_id: str | None,
    profile: AgentConfig | None,
) -> tuple[list[Any], frozenset[str] | None]:
    if getattr(app_config, "skills", None) is None:
        return [], None
    try:
        from deerflow.agents.lead_agent.prompt import get_enabled_skills_for_config

        skills = get_enabled_skills_for_config(app_config, user_id=user_id)
    except Exception:
        logger.exception("Failed to load DR2 skills for StackPlanner")
        return [], frozenset()

    if profile is not None and profile.skills is not None:
        allowed = set(profile.skills)
        skills = [skill for skill in skills if skill.name in allowed]
    skills = sorted(skills, key=lambda skill: skill.name)
    return skills, frozenset(skill.name for skill in skills)


def _skill_index(skills: list[Any]) -> str:
    if not skills:
        return ""

    def render(description_limit: int) -> str:
        lines: list[str] = []
        for skill in skills:
            item: dict[str, str] = {"name": str(skill.name)}
            description = str(getattr(skill, "description", "") or "").strip()
            if description_limit > 0 and description:
                item["description"] = _clip(description, max_chars=description_limit)
            lines.append(f"- {json.dumps(item, ensure_ascii=False, sort_keys=True)}")
        return "\n".join(lines)

    # Skill names are routing capabilities, so losing the alphabetically last
    # names is worse than shortening descriptions. First reserve room for every
    # name, then use the largest shared description budget that fits the bound.
    # This keeps capabilities such as ``video-generation`` visible even when
    # several earlier Skills have long descriptions.
    names_only = render(0)
    if len(names_only) <= DEFAULT_SKILL_INDEX_MAX_CHARS:
        low, high = 0, DEFAULT_SKILL_DESCRIPTION_MAX_CHARS
        while low < high:
            candidate = (low + high + 1) // 2
            if len(render(candidate)) <= DEFAULT_SKILL_INDEX_MAX_CHARS:
                low = candidate
            else:
                high = candidate - 1
        return render(low)

    # Extremely large catalogs may not fit even as names only. Keep the legacy
    # hard ceiling as a final guard, but truncate only in that exceptional case.
    lines: list[str] = []
    total = 0
    for skill in skills:
        item = json.dumps({"name": str(skill.name)}, ensure_ascii=False, sort_keys=True)
        additional = len(item) + 3
        if total + additional > DEFAULT_SKILL_INDEX_MAX_CHARS:
            lines.append("- ...<skill-index-truncated>")
            break
        lines.append(f"- {item}")
        total += additional
    return "\n".join(lines)


def build_sp_central_runtime_context(
    app_config: AppConfig,
    *,
    agent_name: str | None = None,
    user_id: str | None = None,
) -> SPCentralRuntimeContext:
    """Build bounded SOUL and Skill-routing context for SP.

    It creates no SP-owned persistence or memory store. Long-term DR2 memory is
    deliberately not read here; ``sp_recall_memory`` owns that on-demand path.
    """
    profile = _load_profile(agent_name, user_id=user_id)
    resolved_agent_name = profile.name if profile is not None else None
    soul = _load_soul(resolved_agent_name, user_id=user_id)
    skills, available_skill_names = _load_skills(
        app_config,
        user_id=user_id,
        profile=profile,
    )
    skill_index = _skill_index(skills)

    system_parts = [
        "<sp-runtime-policy>",
        "Precedence: current user instructions and pinned task feedback override SOUL, long-term memory, and Skills.",
        "Long-term memory is not injected here. Use RECALL_MEMORY only when historical reusable context is absent from TaskMemoryStack.",
    ]
    if soul:
        system_parts.extend(["<soul>", soul, "</soul>"])
    if skill_index:
        system_parts.extend(
            [
                "<available_skills>",
                (
                    "The CentralAgent sees only this index. For DELEGATE or procedural RECALL_MEMORY, put selected "
                    "names in metadata.skill_names; the handler validates them and the subagent loads the Skill "
                    "bodies. For DELEGATE, metadata.tool_names may additionally carry a narrow list of known Tools "
                    "the specialist should receive; invalid or role-forbidden names are rejected. Pass source and "
                    "artifact references through input_refs. When the user request directly matches an indexed "
                    "Skill, delegate with that Skill immediately instead of searching for an alternative implementation."
                ),
                skill_index,
                "</available_skills>",
            ]
        )
    system_parts.append("</sp-runtime-policy>")

    return SPCentralRuntimeContext(
        agent_name=resolved_agent_name,
        agent_model=profile.model if profile is not None else None,
        system_prompt_section="\n".join(system_parts),
        decision_context="",
        available_skill_names=available_skill_names,
    )

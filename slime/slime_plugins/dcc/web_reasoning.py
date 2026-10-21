"""Opt-in Web thought guidance; actions, evidence and generation stay unchanged."""

WEB_CONCISE_REASONING_VERSION = "web-concise-next-action-v1"
WEB_CONCISE_REASONING_INSTRUCTIONS = (
    "\nKeep Web reasoning concise and focused on the next actual action. Identify the specific "
    "evidence gap, then choose a supported search/read action while research turns remain, or "
    "prepare a final answer from evidence already read. Do not repeatedly enumerate tables, "
    "reconsider the same empty-answer conclusion, simulate tool requests, or imagine unobserved "
    "values in reasoning. When another source is needed and research turns remain, finish "
    "reasoning and emit the required JSON tool action. Use the reserved final turn for the "
    "existing final JSON. Keep all original filters, missing-field rules and citation requirements; "
    "never invent rows or force a nonempty answer."
)


def web_concise_reasoning_configuration(enabled=False):
    if type(enabled) is not bool:
        raise ValueError("web_concise_reasoning must be a boolean")
    return {"enabled": enabled,
            "version": WEB_CONCISE_REASONING_VERSION if enabled else None}

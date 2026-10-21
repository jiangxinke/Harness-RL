"""Opt-in CUAD reasoning guidance; source navigation and answer schema stay separate."""

CUAD_CONCISE_REASONING_VERSION = "cuad-concise-source-offsets-v1"
CUAD_CONCISE_REASONING_INSTRUCTIONS = (
    "\nKeep reasoning concise: assess the question's exact legal conditions and locate candidate evidence "
    "using only source identifiers or source-character [start,end) ranges already provided in observations. "
    "Character ranges are zero-based positions in the original contract with end exclusive, not token or prompt offsets. "
    "Do not calculate new span boundaries or reproduce long contract passages in reasoning. "
    "Search/read when needed, then finish reasoning and emit the required JSON action or answer. "
    "Copy all qualifying spans verbatim only into answer[].text; keep the existing JSON schema."
)


def cuad_concise_reasoning_configuration(enabled=False):
    if type(enabled) is not bool:
        raise ValueError("cuad_concise_reasoning must be a boolean")
    return {"enabled": enabled,
            "version": CUAD_CONCISE_REASONING_VERSION if enabled else None}

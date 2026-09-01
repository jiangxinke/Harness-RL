"""Role/backend and role/loss policies for StackPlanner training modes."""

from __future__ import annotations

import dataclasses
import enum
from collections.abc import Iterable

from .config import TrainingMode


class Backend(str, enum.Enum):
    ACTOR = "actor"
    FROZEN = "frozen"


@dataclasses.dataclass(frozen=True, slots=True)
class RoleRoute:
    role: str
    backend: Backend
    trainable: bool


class RoleRouter:
    """Resolve routing from runtime-discovered policy roles.

    Backend selection and loss eligibility are intentionally independent.
    Overrides can therefore run a role through the actor with its tokens frozen,
    or through a frozen backend for an ablation, without changing presets.
    """

    def __init__(
        self,
        mode: TrainingMode,
        policy_roles: Iterable[str],
        overrides: dict[str, dict] | None = None,
    ) -> None:
        self.mode = TrainingMode(mode)
        self.policy_roles = frozenset(self._normalize(role) for role in policy_roles)
        self.overrides = {self._normalize(role): dict(value) for role, value in (overrides or {}).items()}

    @staticmethod
    def _normalize(role: str) -> str:
        return str(role).strip().lower().replace("_", "-")

    def route(self, role: str) -> RoleRoute:
        normalized = self._normalize(role)
        is_central = normalized in {"central", "central-agent"}
        is_policy_role = is_central or normalized in self.policy_roles
        if self.mode is TrainingMode.SHARED_ALL_AGENTS and is_policy_role:
            route = RoleRoute(role=role, backend=Backend.ACTOR, trainable=True)
        elif is_central:
            route = RoleRoute(role=role, backend=Backend.ACTOR, trainable=True)
        else:
            route = RoleRoute(role=role, backend=Backend.FROZEN, trainable=False)
        override = self.overrides.get(normalized)
        if override:
            return RoleRoute(
                role=role,
                backend=Backend(override.get("backend", route.backend.value)),
                trainable=bool(override.get("trainable", route.trainable)),
            )
        return route

    def table(self) -> list[dict[str, object]]:
        roles = ["central", *(role for role in sorted(self.policy_roles) if role != "central")]
        return [dataclasses.asdict(self.route(role)) for role in roles]

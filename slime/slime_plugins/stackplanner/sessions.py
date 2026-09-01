"""Collision-free episode-local adapter sessions for the slime plugin."""

from __future__ import annotations

import dataclasses
import hashlib
import re
import threading

_SAFE = re.compile(r"[^a-zA-Z0-9_.-]+")


@dataclasses.dataclass(frozen=True, slots=True)
class SessionKey:
    episode_id: str
    role: str
    action_id: str = "root"
    task_id: str = "root"
    attempt: int = 0
    continuation: int = 0

    def session_id(self) -> str:
        parts = (
            self.episode_id,
            self.role,
            self.action_id or "root",
            self.task_id or "root",
            str(self.attempt),
            str(self.continuation),
        )
        canonical = "|".join(parts)
        readable = ".".join(_SAFE.sub("-", part).strip("-.")[:28] or "x" for part in parts[:4])
        return f"sp.{readable}.{self.attempt}.{self.continuation}.{hashlib.sha256(canonical.encode()).hexdigest()[:16]}"


class SessionManager:
    def __init__(self, episode_id: str) -> None:
        if not episode_id:
            raise ValueError("episode_id must not be empty")
        self.episode_id = episode_id
        self._lock = threading.Lock()
        self._open: dict[str, SessionKey] = {}
        self._used: set[str] = set()

    def allocate(
        self,
        *,
        role: str,
        action_id: str | None = None,
        task_id: str | None = None,
        attempt: int = 0,
        continuation: int = 0,
    ) -> tuple[str, SessionKey]:
        key = SessionKey(
            episode_id=self.episode_id,
            role=role,
            action_id=action_id or "root",
            task_id=task_id or "root",
            attempt=attempt,
            continuation=continuation,
        )
        sid = key.session_id()
        with self._lock:
            if sid in self._used:
                raise ValueError(f"session already allocated or closed: {sid}")
            self._open[sid] = key
            self._used.add(sid)
        return sid, key

    def close(self, sid: str) -> SessionKey | None:
        with self._lock:
            return self._open.pop(sid, None)

    def open_sessions(self) -> dict[str, SessionKey]:
        with self._lock:
            return dict(self._open)

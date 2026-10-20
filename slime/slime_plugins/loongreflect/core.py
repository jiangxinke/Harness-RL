from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from html import escape, unescape
from typing import Any, Protocol
from collections.abc import Callable


class ProtocolError(ValueError):
    pass


PROTOCOL_REPAIR_GUIDANCE = "Produce exactly one permitted action. Do not generate an observation, analysis, or text outside the action tags."
POST_SEARCH_REFLECTION_GUIDANCE = (
    "Your next model turn must independently emit one complete, nonempty reflect "
    "action as its only public action. Use this separate model turn to assess the "
    "observed search evidence and remaining uncertainty without stating the final "
    "answer. After reflecting, you may search again if more evidence is needed, "
    "or answer in a later model turn."
)
SALVAGED_SEARCH_GUIDANCE = "Only the first valid search action was executed. Observations are added only by the harness; do not predict search results or continue after </search>."


class ActionKind(str, Enum):
    SEARCH = "search"
    REFLECT = "reflect"
    ANSWER = "answer"


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    text: str
    backtrack: str | None = None

    @classmethod
    def parse(cls, content: str, *, allow_reflect: bool = True) -> Action:
        """Parse exactly one public action (reflect may be followed by backtrack)."""
        content = content.strip()
        single = re.fullmatch(r"<(search|answer|reflect)>(.*?)</\1>", content, re.DOTALL)
        if single and single.group(2).strip():
            tag, text = single.group(1), unescape(single.group(2).strip())
            if re.search(r"</?[A-Za-z][^>]*>", text):
                raise ProtocolError("tags are not allowed inside plaintext actions")
            if tag == "answer" and ("\n" in text or len(text) > 200):
                raise ProtocolError("an answer must be one short phrase")
            action = cls(ActionKind(tag), text)
            if action.kind is ActionKind.REFLECT and not allow_reflect:
                raise ProtocolError("reflect/backtrack actions are disabled; expected <search> or <answer>")
            return action

        rollback = re.fullmatch(
            r"<reflect>(.*?)</reflect>\s*<backtrack>(.*?)</backtrack>",
            content,
            re.DOTALL,
        )
        if rollback and rollback.group(1).strip() and rollback.group(2).strip():
            if any(re.search(r"</?[A-Za-z][^>]*>", part) for part in rollback.groups()):
                raise ProtocolError("tags are not allowed inside plaintext actions")
            if not allow_reflect:
                raise ProtocolError("reflect/backtrack actions are disabled; expected <search> or <answer>")
            return cls(
                ActionKind.REFLECT,
                unescape(rollback.group(1).strip()),
                unescape(rollback.group(2).strip()),
            )
        raise ProtocolError("expected <search>, <answer>, <reflect>, or <reflect> followed by <backtrack>")

    @classmethod
    def leading_search(cls, content: str) -> Action | None:
        """Recover one complete leading search while leaving the invalid branch intact."""
        match = re.match(r"\s*<search>(.*?)</search>(?=\s*\S)", content, re.DOTALL)
        if match is None:
            return None
        text = unescape(match.group(1).strip())
        if not text or re.search(r"</?[A-Za-z][^>]*>", text):
            return None
        return cls(ActionKind.SEARCH, text)


@dataclass(frozen=True)
class ModelTurn:
    thinking: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResumeDecision:
    can_resume: bool
    reason: str = ""


@dataclass(frozen=True)
class AnswerValidation:
    accepted: bool
    feedback: str = ""


class Actor(Protocol):
    def act(self, trace: str, *, max_tokens: int) -> ModelTurn: ...


class BacktrackJudge(Protocol):
    def can_resume(self, *, query: str, prefix: str, backtrack: str) -> ResumeDecision: ...


class UpdateWriter(Protocol):
    def create_update(self, *, query: str, prefix: str, backtrack: str) -> str: ...


class SearchTool(Protocol):
    def search(self, question: str) -> str: ...


class AnswerValidator(Protocol):
    def validate(self, *, query: str, answer: str) -> AnswerValidation: ...


@dataclass
class Node:
    id: int
    kind: str
    text: str
    parent: int | None
    children: list[int] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class RolloutTree:
    """Append-only tree. A failed branch is never deleted."""

    def __init__(self, query: str):
        self.query = query
        self.nodes: dict[int, Node] = {0: Node(0, "query", query, None)}
        self.root_id = 0
        self._next_id = 1
        self.resume_audit: list[dict[str, Any]] = []
        self.status = "running"
        self.termination_reason: str | None = None
        self.model_turns = 0
        self.output_tokens = 0

    def append(self, parent: int, kind: str, text: str, metadata: dict[str, Any] | None = None) -> int:
        if parent not in self.nodes:
            raise KeyError(f"unknown parent node: {parent}")
        node_id = self._next_id
        self._next_id += 1
        self.nodes[node_id] = Node(node_id, kind, text, parent, metadata=metadata or {})
        self.nodes[parent].children.append(node_id)
        return node_id

    def path(self, node_id: int) -> list[int]:
        result: list[int] = []
        current: int | None = node_id
        while current is not None:
            result.append(current)
            current = self.nodes[current].parent
        return list(reversed(result))

    def render_path(self, node_id: int) -> str:
        return "".join(self._render(self.nodes[item]) for item in self.path(node_id))

    def leaves(self) -> list[int]:
        return [node.id for node in self.nodes.values() if not node.children]

    def traces(self) -> list[dict[str, Any]]:
        return [
            {
                "leaf_id": leaf,
                "terminal": self.nodes[leaf].kind,
                "trace": self.render_path(leaf),
            }
            for leaf in self.leaves()
        ]

    def to_dict(self, *, include_traces: bool = True) -> dict[str, Any]:
        result = {
            "query": self.query,
            "root_id": self.root_id,
            "nodes": [asdict(self.nodes[node_id]) for node_id in sorted(self.nodes)],
            "resume_audit": self.resume_audit,
            "status": self.status,
            "termination_reason": self.termination_reason,
            "model_turns": self.model_turns,
            "output_tokens": self.output_tokens,
        }
        if include_traces:
            result["traces"] = self.traces()
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RolloutTree:
        tree = cls(str(data["query"]))
        tree.nodes = {item["id"]: Node(**item) for item in data["nodes"]}
        tree.root_id = int(data.get("root_id", 0))
        tree._next_id = max(tree.nodes, default=-1) + 1
        tree.resume_audit = list(data.get("resume_audit", []))
        tree.status = str(data.get("status", "running"))
        tree.termination_reason = data.get("termination_reason")
        tree.model_turns = int(data.get("model_turns", 0))
        tree.output_tokens = int(data.get("output_tokens", 0))
        return tree

    @staticmethod
    def _render(node: Node) -> str:
        return f"<{node.kind}>{escape(node.text)}</{node.kind}>"


class Harness:
    def __init__(
        self,
        actor: Actor,
        search: SearchTool,
        *,
        backtrack_judge: BacktrackJudge,
        update_writer: UpdateWriter,
        max_steps: int = 64,
        max_output_tokens: int = 96 * 1024,
        max_tokens_per_turn: int = 8 * 1024,
        max_protocol_retries: int = 2,
        answer_validator: AnswerValidator | None = None,
        max_answer_retries: int = 2,
        allow_reflect: bool = True,
        require_post_search_reflection: bool = False,
        on_checkpoint: Callable[[RolloutTree], None] | None = None,
    ) -> None:
        if type(require_post_search_reflection) is not bool:
            raise ValueError("require_post_search_reflection must be a boolean")
        if require_post_search_reflection and not allow_reflect:
            raise ValueError("post-search reflection requires the full action protocol")
        self.require_post_search_reflection = require_post_search_reflection
        self.actor = actor
        self.search = search
        self.backtrack_judge = backtrack_judge
        self.update_writer = update_writer
        self.max_steps = max_steps
        self.max_output_tokens = max_output_tokens
        self.max_tokens_per_turn = max_tokens_per_turn
        self.max_protocol_retries = max_protocol_retries
        self.answer_validator = answer_validator
        self.max_answer_retries = max_answer_retries
        self.allow_reflect = allow_reflect
        self.on_checkpoint = on_checkpoint

    def run(self, query: str) -> RolloutTree:
        tree = RolloutTree(query)
        active = tree.root_id
        # A checkpoint is the active node immediately before a model thinking turn.
        checkpoints: list[int] = []
        consecutive_protocol_errors = 0
        answer_rejections = 0
        self._emit(tree)
        return self._continue(tree, active, checkpoints, consecutive_protocol_errors, answer_rejections)

    def resume(self, tree: RolloutTree) -> RolloutTree:
        """Continue an interrupted running tree without discarding prior branches."""
        if tree.status != "running":
            return tree
        active_leaves = [leaf for leaf in tree.leaves() if tree.nodes[leaf].kind not in {"answer", "backtrack"}]
        if len(active_leaves) > 1:
            raise ProtocolError("an interrupted tree has multiple active leaves")

        if active_leaves:
            active = active_leaves[0]
            checkpoints = self._checkpoints_on_path(tree, active)
        else:
            recovered = self._resume_unresolved_backtrack(tree)
            if recovered is None:
                raise ProtocolError("an interrupted tree has no resumable active leaf")
            active, checkpoints = recovered

        node = tree.nodes[active]
        consecutive_protocol_errors = int(node.metadata.get("attempt", 0)) if node.kind == "update" and node.metadata.get("source") == "protocol_repair" else 0
        answer_rejections = sum(1 for node in tree.nodes.values() if node.kind == "answer" and node.metadata.get("accepted") is False)
        self._emit(tree)
        return self._continue(tree, active, checkpoints, consecutive_protocol_errors, answer_rejections)

    def _continue(
        self,
        tree: RolloutTree,
        active: int,
        checkpoints: list[int],
        consecutive_protocol_errors: int,
        answer_rejections: int,
    ) -> RolloutTree:
        remaining_steps = max(0, self.max_steps - tree.model_turns)

        for _ in range(remaining_steps):
            checkpoints.append(active)
            remaining_tokens = self.max_output_tokens - tree.output_tokens
            if remaining_tokens <= 0:
                self._terminate(tree, active, "max_output_tokens_reached")
                self._emit(tree)
                return tree
            try:
                turn = self.actor.act(
                    tree.render_path(active),
                    max_tokens=min(self.max_tokens_per_turn, remaining_tokens),
                )
            except Exception as exc:
                self._terminate(tree, active, "actor_error", exc)
                self._emit(tree)
                return tree
            tree.model_turns += 1
            tree.output_tokens += _turn_output_tokens(turn)
            thinking = turn.thinking.strip()
            active = tree.append(active, "think", thinking, turn.metadata)

            try:
                action = Action.parse(turn.content, allow_reflect=self.allow_reflect)
            except ProtocolError as exc:
                if self.allow_reflect:
                    active = tree.append(
                        active,
                        "reflect",
                        "The model produced an invalid action.",
                        {"source": "harness"},
                    )
                    failed_leaf = tree.append(
                        active,
                        "backtrack",
                        str(exc),
                        {"source": "harness", "invalid_output": turn.content},
                    )
                else:
                    failed_leaf = tree.append(
                        active,
                        "invalid",
                        str(exc),
                        {"source": "harness", "invalid_output": turn.content},
                    )
                salvaged = Action.leading_search(turn.content)
                if salvaged is not None:
                    checkpoint_id = checkpoints[-1]
                    active = tree.append(
                        checkpoint_id,
                        "update",
                        SALVAGED_SEARCH_GUIDANCE,
                        {
                            "source": "protocol_salvage",
                            "source_backtrack_leaf": failed_leaf,
                        },
                    )
                    active = tree.append(
                        active,
                        "think",
                        thinking,
                        {
                            **turn.metadata,
                            "source": "protocol_salvage",
                            "source_backtrack_leaf": failed_leaf,
                        },
                    )
                    active = tree.append(active, "search", salvaged.text)
                    try:
                        observation = self.search.search(salvaged.text)
                    except Exception as search_exc:
                        self._terminate(tree, active, "search_error", search_exc)
                        self._emit(tree)
                        return tree
                    active = tree.append(active, "observation", observation, self._search_observation_metadata(observation))
                    active = self._request_post_search_reflection(tree, active)
                    consecutive_protocol_errors = 0
                    if self._output_limit_reached(tree, active):
                        self._emit(tree)
                        return tree
                    self._emit(tree)
                    continue
                consecutive_protocol_errors += 1
                if tree.output_tokens >= self.max_output_tokens:
                    # This invalid protocol branch is also the terminal branch
                    # that exhausted the rollout token budget. Mark it
                    # explicitly so reward assignment can distinguish a real
                    # length truncation from earlier protocol-repair leaves.
                    tree.nodes[failed_leaf].metadata.update(
                        {
                            "termination": True,
                            "termination_reason": "max_output_tokens_reached",
                        }
                    )
                    tree.status = "terminated"
                    tree.termination_reason = "max_output_tokens_reached"
                    self._emit(tree)
                    return tree
                if consecutive_protocol_errors > self.max_protocol_retries:
                    tree.status = "terminated"
                    tree.termination_reason = "invalid_action_retry_limit"
                    self._emit(tree)
                    return tree
                checkpoint_id = checkpoints[-1]
                active = tree.append(
                    checkpoint_id,
                    "update",
                    PROTOCOL_REPAIR_GUIDANCE,
                    {
                        "source": "protocol_repair",
                        "source_backtrack_leaf": failed_leaf,
                        "attempt": consecutive_protocol_errors,
                    },
                )
                self._emit(tree)
                continue

            consecutive_protocol_errors = 0

            if action.kind is ActionKind.SEARCH:
                active = tree.append(active, "search", action.text)
                try:
                    observation = self.search.search(action.text)
                except Exception as exc:
                    self._terminate(tree, active, "search_error", exc)
                    self._emit(tree)
                    return tree
                active = tree.append(active, "observation", observation, self._search_observation_metadata(observation))
                active = self._request_post_search_reflection(tree, active)
                if self._output_limit_reached(tree, active):
                    self._emit(tree)
                    return tree
                self._emit(tree)
                continue

            if action.kind is ActionKind.ANSWER:
                reflection_request = self._request_post_search_reflection(tree, active, deferred_answer=action.text)
                if reflection_request != active:
                    # Keep the real sampled turn and its token metadata intact.
                    # The model must independently reflect, then answer again.
                    active = reflection_request
                    if self._output_limit_reached(tree, active):
                        self._emit(tree)
                        return tree
                    self._emit(tree)
                    continue
                validation = self.answer_validator.validate(query=tree.query, answer=action.text) if self.answer_validator is not None else AnswerValidation(True)
                answer_id = tree.append(
                    active,
                    "answer",
                    action.text,
                    {
                        "accepted": validation.accepted,
                        "validator_feedback": validation.feedback,
                    },
                )
                if validation.accepted:
                    tree.status = "completed"
                    self._emit(tree)
                    return tree
                answer_rejections += 1
                if answer_rejections > self.max_answer_retries:
                    if self.allow_reflect:
                        reflect = tree.append(
                            active,
                            "reflect",
                            "The proposed answer was rejected by the external validator.",
                            {"source": "answer_validator"},
                        )
                        tree.append(
                            reflect,
                            "backtrack",
                            "The answer-validation retry limit was reached.",
                            {"source": "answer_validator", "termination": True},
                        )
                    else:
                        tree.append(
                            active,
                            "termination",
                            "The answer-validation retry limit was reached.",
                            {"source": "answer_validator", "termination": True},
                        )
                    tree.status = "terminated"
                    tree.termination_reason = "answer_validation_retry_limit"
                    self._emit(tree)
                    return tree
                # An externally rejected answer usually means the reasoning path, not
                # merely its last wording, is unreliable. Start a clean sibling branch
                # at the root. Do not repeat the rejected candidate: the candidate can
                # contain a private reference alias even when its complete wording was
                # rejected by an exact-match validator.
                checkpoints = [tree.root_id]
                guidance = "External validation rejected the proposed answer. Start a fresh solution from the query, verify every clue with direct evidence, and independently determine the exact entity, date, number, or formal wording."
                if validation.feedback:
                    guidance = f"{guidance} {validation.feedback}"
                active = tree.append(
                    tree.root_id,
                    "update",
                    guidance,
                    {
                        "source": "answer_validator",
                        "source_answer_leaf": answer_id,
                        "attempt": answer_rejections,
                    },
                )
                self._emit(tree)
                continue

            reflection_metadata = {"source": "model_action"} if self.require_post_search_reflection else None
            active = tree.append(active, "reflect", action.text, reflection_metadata)
            if action.backtrack is None:
                if self._output_limit_reached(tree, active):
                    self._emit(tree)
                    return tree
                self._emit(tree)
                continue

            failed_leaf = tree.append(active, "backtrack", action.backtrack)
            self._emit(tree)
            try:
                resumed = self._find_resume_point(
                    tree=tree,
                    checkpoints=checkpoints,
                    failed_leaf=failed_leaf,
                    backtrack=action.backtrack,
                )
            except Exception as exc:
                tree.status = "terminated"
                tree.termination_reason = f"backtrack_judge_error: {type(exc).__name__}: {exc}"
                self._emit(tree)
                return tree
            checkpoint_index, checkpoint_id = resumed
            # Truncate only the active checkpoint list; the failed tree branch remains.
            checkpoints = checkpoints[: checkpoint_index + 1]
            try:
                update = self.update_writer.create_update(
                    query=tree.query,
                    prefix=tree.render_path(checkpoint_id),
                    backtrack=action.backtrack,
                ).strip()
            except Exception as exc:
                tree.status = "terminated"
                tree.termination_reason = f"update_writer_error: {type(exc).__name__}: {exc}"
                self._emit(tree)
                return tree
            if not update:
                raise ProtocolError("the model produced an empty update")
            active = tree.append(
                checkpoint_id,
                "update",
                update,
                {
                    "source_backtrack_leaf": failed_leaf,
                    "source": "model_rewrite",
                },
            )
            if self._output_limit_reached(tree, active):
                self._emit(tree)
                return tree
            self._emit(tree)

        self._terminate(tree, active, "max_steps_reached")
        self._emit(tree)
        return tree

    def _request_post_search_reflection(
        self, tree: RolloutTree, active: int, *, deferred_answer: str | None = None,
    ) -> int:
        required_observation = self._reflection_required_after(tree, active)
        if required_observation is None:
            return active
        metadata = {
            "source": "post_search_reflection",
            "required_observation_id": required_observation,
        }
        if deferred_answer is not None:
            # Audit only: the candidate is neither rendered nor sent to the
            # validator, and cannot be automatically accepted after reflection.
            metadata["deferred_model_answer"] = deferred_answer
        return tree.append(active, "update", POST_SEARCH_REFLECTION_GUIDANCE, metadata)

    def _search_observation_metadata(self, observation: str) -> dict[str, Any] | None:
        if not self.require_post_search_reflection:
            return None
        # HttpSearch returns this fixed sentinel when there are no documents.
        # Only successful, nonempty tool output establishes an evidence stage.
        evidence = bool(observation.strip()) and observation.strip() != "No search results found."
        return {"source": "search_tool", "has_evidence": evidence}

    def _reflection_required_after(self, tree: RolloutTree, active: int) -> int | None:
        if not self.require_post_search_reflection:
            return None
        pending = None
        # Reconstruct from the actual active path for checkpoint recovery and
        # backtracking. Failed sibling branches and environment reflections do
        # not satisfy the model's independent post-evidence reflection stage.
        for node_id in tree.path(active):
            node = tree.nodes[node_id]
            if (node.kind == "observation" and node.metadata.get("source") == "search_tool"
                    and node.metadata.get("has_evidence") is True):
                pending = node_id
            elif (node.kind == "reflect" and node.metadata.get("source") == "model_action"
                    and node.parent is not None and tree.nodes[node.parent].kind == "think"):
                pending = None
        return pending

    @staticmethod
    def _checkpoints_on_path(tree: RolloutTree, node_id: int) -> list[int]:
        checkpoints: list[int] = [tree.root_id]
        for item in tree.path(node_id):
            node = tree.nodes[item]
            if node.kind == "think" and node.parent is not None and node.parent != tree.root_id:
                checkpoints.append(node.parent)
        return checkpoints

    def _resume_unresolved_backtrack(self, tree: RolloutTree) -> tuple[int, list[int]] | None:
        resolved = {node.metadata.get("source_backtrack_leaf") for node in tree.nodes.values() if node.kind == "update"}
        candidates = [leaf for leaf in tree.leaves() if tree.nodes[leaf].kind == "backtrack" and leaf not in resolved and tree.nodes[leaf].metadata.get("source") != "harness"]
        if not candidates:
            return None
        failed_leaf = max(candidates)
        backtrack = tree.nodes[failed_leaf].text
        checkpoints = self._checkpoints_on_path(tree, failed_leaf)
        resumed = self._find_resume_point(
            tree=tree,
            checkpoints=checkpoints,
            failed_leaf=failed_leaf,
            backtrack=backtrack,
        )
        checkpoint_index, checkpoint_id = resumed
        checkpoints = checkpoints[: checkpoint_index + 1]
        update = self.update_writer.create_update(query=tree.query, prefix=tree.render_path(checkpoint_id), backtrack=backtrack).strip()
        if not update:
            raise ProtocolError("the model produced an empty update")
        active = tree.append(
            checkpoint_id,
            "update",
            update,
            {
                "source_backtrack_leaf": failed_leaf,
                "source": "model_rewrite",
                "recovered": True,
            },
        )
        return active, checkpoints

    def _find_resume_point(
        self,
        *,
        tree: RolloutTree,
        checkpoints: list[int],
        failed_leaf: int,
        backtrack: str,
    ) -> tuple[int, int]:
        audit = {"failed_leaf": failed_leaf, "backtrack": backtrack, "candidates": []}
        for index in range(len(checkpoints) - 1, -1, -1):
            checkpoint_id = checkpoints[index]
            prefix = tree.render_path(checkpoint_id)
            decision = self.backtrack_judge.can_resume(query=tree.query, prefix=prefix, backtrack=backtrack)
            audit["candidates"].append(
                {
                    "checkpoint_id": checkpoint_id,
                    "can_resume": decision.can_resume,
                    "reason": decision.reason,
                }
            )
            if decision.can_resume:
                audit["selected_checkpoint"] = checkpoint_id
                tree.resume_audit.append(audit)
                return index, checkpoint_id
        # The root query is always a safe clean-slate fallback. If the learned
        # controller rejects every candidate (including root), preserve the
        # failed branch but start a new sibling branch from root with rewritten
        # forward guidance instead of terminating the rollout.
        root_index = checkpoints.index(tree.root_id) if tree.root_id in checkpoints else 0
        audit["selected_checkpoint"] = tree.root_id
        audit["root_fallback"] = True
        tree.resume_audit.append(audit)
        return root_index, tree.root_id

    def _output_limit_reached(self, tree: RolloutTree, active: int) -> bool:
        if tree.output_tokens < self.max_output_tokens:
            return False
        self._terminate(tree, active, "max_output_tokens_reached")
        return True

    def _emit(self, tree: RolloutTree) -> None:
        if self.on_checkpoint is not None:
            self.on_checkpoint(tree)

    def _terminate(self, tree: RolloutTree, active: int, reason: str, error: Exception | None = None) -> None:
        detail = reason if error is None else f"{reason}: {type(error).__name__}: {error}"
        if self.allow_reflect:
            reflect = tree.append(
                active,
                "reflect",
                "The rollout was terminated by the harness and its partial tree was preserved.",
                {"source": "harness", "termination_reason": reason},
            )
            tree.append(
                reflect,
                "backtrack",
                detail,
                {
                    "source": "harness",
                    "termination": True,
                    "termination_reason": reason,
                },
            )
        else:
            tree.append(
                active,
                "termination",
                detail,
                {
                    "source": "harness",
                    "termination": True,
                    "termination_reason": reason,
                },
            )
        tree.status = "terminated"
        tree.termination_reason = detail


def _turn_output_tokens(turn: ModelTurn) -> int:
    usage = turn.metadata.get("usage") if isinstance(turn.metadata, dict) else None
    if isinstance(usage, dict):
        value = usage.get("completion_tokens")
        if isinstance(value, int) and value >= 0:
            return value
    # Test doubles and non-token-reporting adapters get a conservative approximation.
    return max(1, (len(turn.thinking) + len(turn.content) + 3) // 4)

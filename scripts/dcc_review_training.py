"""Explicit TRAIN20 review/navigation approval and exact public request replay.

No trace-defined schemas, target edits, token synthesis, network or gold reads.
The legacy profile remains unchanged; this route requires its own profile ID.
"""
from __future__ import annotations
import json
from pathlib import Path
from jsonschema import Draft202012Validator

LEGACY_PROFILE = "dcc-native-train-profile-v1"
REVIEW20_PROFILE = "dcc-native-train-profile-web20-review-navigation-v1"
WEB32K_PROFILE = "dcc-native-train-profile-web32k-whole-turn-v2"
WEB32K_CACHED_PROFILE = "dcc-native-train-profile-web32k-cached-rows-v3"
WEB32K_CACHED_8K_PROFILE = "dcc-native-train-profile-web32k-cached-rows-8k-thought-v3"
CACHED_TRAIN_PROFILES = frozenset({WEB32K_CACHED_PROFILE, WEB32K_CACHED_8K_PROFILE})
CUAD_RETRIEVAL_PROFILE = "dcc-native-train-profile-cuad-retrieval-v1"
CUAD_TASKBOUND_UNION_PROFILE = "dcc-native-train-profile-cuad-taskbound-contiguous-v2"
JOINT_CACHED_CUAD_PROFILE = "dcc-native-train-profile-web32k-cached-rows-cuad-retrieval-v3"
JOINT_CACHED_CUAD_TASKBOUND_PROFILE = "dcc-native-train-profile-web32k-cached-rows-cuad-taskbound-contiguous-v4"
JOINT_CACHED_CUAD_TRAIN_PROFILES = frozenset({JOINT_CACHED_CUAD_PROFILE, JOINT_CACHED_CUAD_TASKBOUND_PROFILE})
WEB_STAGE_PROFILE = "final50-search25-fetch25"
WEB_STAGE_CYCLE = ("final", "search", "final", "fetch")


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def strict_json(text):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, "Duplicate public replay JSON key")
            out[key] = value
        return out
    return json.loads(text, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite replay JSON")))


def approved_web_profile(profile, profile_class):
    require(isinstance(profile, dict), "Trusted native TRAIN profile must be an object")
    version = profile.get("version")
    if version in {CUAD_RETRIEVAL_PROFILE, CUAD_TASKBOUND_UNION_PROFILE} | JOINT_CACHED_CUAD_TRAIN_PROFILES:
        # Reuse an explicitly declared, unchanged Web route. Retrieval belongs
        # to the CUAD harness and never alters the shared model identity.
        route = profile.get("web_profile_version")
        if version in JOINT_CACHED_CUAD_TRAIN_PROFILES:
            require(set(profile) == {"version", "web_profile_version", "collection_overlay_id", "identity", "web",
                                     "cuad", "cuad_identity", "cuad_identity_differences"},
                    "Joint cached TRAIN requires both complete original domain profiles")
            require(route == WEB32K_CACHED_PROFILE, "Joint cached TRAIN requires the original static16k Web route")
        else:
            require(route in {REVIEW20_PROFILE, WEB32K_PROFILE}, "Explicit existing Web profile version required")
        require("cuad" in profile, "Explicit retrieval profile requires CUAD configuration")
        from dcc_native_sft import CuadSchemaProfile
        approved_cuad_profile(profile, CuadSchemaProfile)
        original = dict(profile)
        original.pop("web_profile_version")
        original["version"] = route
        return approved_web_profile(original, profile_class)
    base = {"version", "identity", "web"}
    if version in {REVIEW20_PROFILE, WEB32K_PROFILE, WEB32K_CACHED_PROFILE, WEB32K_CACHED_8K_PROFILE}:
        base.add("collection_overlay_id")
        require(isinstance(profile.get("collection_overlay_id"), str)
                and bool(profile["collection_overlay_id"].strip()), "Explicit collection overlay ID required")
    else:
        require(version == LEGACY_PROFILE, "Unsupported trusted native TRAIN profile version")
    allowed = (base, base | {"cuad"})
    if version in {WEB32K_PROFILE, WEB32K_CACHED_PROFILE, WEB32K_CACHED_8K_PROFILE}:
        allowed = (base, base | {"cuad", "cuad_identity", "cuad_identity_differences"})
    require(set(profile) in allowed, "Unsupported trusted TRAIN profile fields")
    identity = profile["identity"]
    require(isinstance(identity, dict), "Trusted native identity must be an object")
    if version in {WEB32K_PROFILE, WEB32K_CACHED_PROFILE, WEB32K_CACHED_8K_PROFILE}:
        from slime_plugins.dcc.prompt_history import PromptHistoryConfig
        context = PromptHistoryConfig.from_dict(identity.get("prompt_history"))
        thought_tokens = 8192 if version == WEB32K_CACHED_8K_PROFILE else 16384
        minimum_model_len = 32768 + thought_tokens + 4096
        require(context.enabled and type(identity.get("max_prompt_tokens")) is int
                and identity["max_prompt_tokens"] == 32768
                and type(identity.get("max_model_len")) is int and identity["max_model_len"] >= minimum_model_len
                and type(identity.get("thought_tokens")) is int and identity["thought_tokens"] == thought_tokens
                and identity.get("thought_sampling_profile") == "thinking",
                "Explicit Web32k requires whole-turn context and its exact profile native thought/JSON budget")
        if "cuad_identity" in profile:
            cuad_identity = profile["cuad_identity"]
            require(isinstance(cuad_identity, dict)
                    and "prompt_history" not in cuad_identity
                    and type(cuad_identity.get("max_prompt_tokens")) is int
                    and cuad_identity["max_prompt_tokens"] == 16384
                    and type(cuad_identity.get("max_model_len")) is int
                    and cuad_identity["max_model_len"] == 36864,
                    "Preserved CUAD identity must retain its original unmanaged declaration")
            actual_differences = sorted(k for k in set(identity) | set(cuad_identity)
                                        if identity.get(k) != cuad_identity.get(k))
            require(isinstance(profile["cuad_identity_differences"], list)
                    and profile["cuad_identity_differences"] == actual_differences
                    and set(actual_differences) <= {"source_commit", "generation_seed",
                        "max_prompt_tokens", "max_model_len", "prompt_history"},
                    "Trusted profile must enumerate exactly the approved domain identity differences")
    else:
        require("prompt_history" not in identity, "Legacy profile cannot approve a new prompt history policy")
    web = profile["web"]
    keys = {"max_steps", "explicit_review", "cached_rows"}
    if version in {REVIEW20_PROFILE, WEB32K_PROFILE, WEB32K_CACHED_PROFILE, WEB32K_CACHED_8K_PROFILE}:
        keys.add("navigation_ledger")
    optional_web_keys = ({"data_discovery_guidance", "multi_citation"} if version in {
        REVIEW20_PROFILE, WEB32K_PROFILE, WEB32K_CACHED_PROFILE, WEB32K_CACHED_8K_PROFILE} else set())
    if version in CACHED_TRAIN_PROFILES:
        keys.add("cached_rows_configuration")
    require(isinstance(web, dict) and keys <= set(web)
            and set(web) <= keys | optional_web_keys, "Invalid approved Web profile fields")
    result = profile_class(**web, profile_version=version)
    result.validate()
    return result


def approved_cuad_profile(profile, profile_class):
    value = profile.get("cuad")
    bound = profile.get("version") in {CUAD_RETRIEVAL_PROFILE, CUAD_TASKBOUND_UNION_PROFILE} | JOINT_CACHED_CUAD_TRAIN_PROFILES
    if value is None:
        require(not bound, "Explicit retrieval profile requires CUAD configuration")
        return None
    keys = {"max_steps", "structured_actions", "schema_version"}
    if bound:
        keys.add("retrieval")
    require(isinstance(value, dict) and set(value) == keys, "Invalid approved CUAD profile fields")
    require(not bound or isinstance(value["retrieval"], dict), "Explicit CUAD retrieval binding required")
    if bound:
        from dcc_cuad_retrieval import VERSION, TASKBOUND_UNION_VERSION
        expected = (TASKBOUND_UNION_VERSION if profile.get("version") in {
            CUAD_TASKBOUND_UNION_PROFILE, JOINT_CACHED_CUAD_TASKBOUND_PROFILE} else VERSION)
        require(value["retrieval"].get("version") == expected,
                "CUAD binding version differs from explicitly approved harness profile")
    result = profile_class(**value)
    result.validate()
    return result


def bind_profile_source(profile, source_root):
    if profile["version"] in {REVIEW20_PROFILE, WEB32K_PROFILE, WEB32K_CACHED_PROFILE, WEB32K_CACHED_8K_PROFILE, CUAD_RETRIEVAL_PROFILE, CUAD_TASKBOUND_UNION_PROFILE} | JOINT_CACHED_CUAD_TRAIN_PROFILES:
        overlay = strict_json((Path(source_root) / ".dcc-training-code-overlay.json").read_text())
        require(overlay.get("overlay_id") == profile["collection_overlay_id"]
                and overlay.get("explicit_training_profile") == profile["version"]
                and overlay.get("baseline_source_unchanged") is False,
                "Explicit TRAIN profile differs from actual collection/training source overlay")
        if profile["version"] in CACHED_TRAIN_PROFILES | JOINT_CACHED_CUAD_TRAIN_PROFILES:
            from dcc_train_cached_rows import configuration, equal
            require(equal(overlay.get("web_cached_rows"), configuration()),
                    "Cached TRAIN source overlay requires complete current v3 configuration")


def review_request_state(profile, record, collector=None, *, cached_sources=None):
    """Rebuild exact runner messages/schema from public tool events, never targets.

    Returns only the trusted schema and indices of exact terminal controls.
    Original messages/targets stay intact for native token equality and training.
    """
    from slime_plugins.dcc.data import Task
    from slime_plugins.dcc.episode import EpisodeRunner, _WebEvidence, final_turn_message
    from slime_plugins.dcc.web_navigation import (WebNavigationLedger,
        WEB_NAVIGATION_INSTRUCTIONS, WEB_NAVIGATION_PREFIX)
    from slime_plugins.dcc.web_review import (WEB_EXPLICIT_REVIEW_INSTRUCTIONS,
        web_review_final_control, web_review_schema)
    from slime_plugins.dcc.web_structured import web_action_schema, web_final_schema

    profile.validate()
    schema_options = {"multi_citation": True} if profile.multi_citation else {}
    require(profile.profile_version in {REVIEW20_PROFILE, WEB32K_PROFILE, WEB32K_CACHED_PROFILE, WEB32K_CACHED_8K_PROFILE},
            "Only explicit TRAIN20 review/navigation profile may use replay controls")
    require(record.get("source") == "independent_curriculum" and record.get("split") == "train",
            "Review replay accepts independent Web TRAIN only")
    messages = record.get("messages")
    require(isinstance(messages, list) and len(messages) >= 3
            and all(isinstance(m, dict) and set(m) == {"role", "content"}
                    and m["role"] in {"system", "user", "assistant"}
                    and isinstance(m["content"], str) and m["content"].strip() for m in messages),
            "Exact public Web request messages required")
    require(messages[0]["role"] == "system" and messages[1]["role"] == "user",
            "Original public task anchors required")
    marker = "Required answer columns: "
    require(marker in messages[0]["content"], "Original public answer columns missing")
    columns, _ = json.JSONDecoder().raw_decode(messages[0]["content"].split(marker, 1)[1])
    require(isinstance(columns, list) and columns and len(columns) == len(set(columns))
            and all(isinstance(c, str) and c.strip() and c != "citations" for c in columns),
            "Invalid public required answer columns")
    task = Task(record["task_id"], "widesearch", "train", messages[1]["content"],
                metadata={"evaluation": {"required": columns}})
    anchors = EpisodeRunner(max_steps=20, initial_chunks=4).initial_messages(task)
    if profile.cached_rows:
        from slime_plugins.dcc.web_rows import WEB_CACHED_ROWS_INSTRUCTIONS
        anchors[0]["content"] += WEB_CACHED_ROWS_INSTRUCTIONS
    anchors[0]["content"] += WEB_EXPLICIT_REVIEW_INSTRUCTIONS + WEB_NAVIGATION_INSTRUCTIONS
    if profile.data_discovery_guidance:
        from slime_plugins.dcc.web_data_discovery import WEB_DATA_DISCOVERY_GUIDANCE

        anchors[0]["content"] += WEB_DATA_DISCOVERY_GUIDANCE
    require(messages[:2] == anchors, "Exact approved TRAIN20 review/navigation system and public task required")
    cached_replay = None
    if profile.cached_rows:
        from dcc_train_cached_rows import CachedReplay
        cached_replay = CachedReplay(collector, cached_sources, task_id=record["task_id"])
    evidence, events, turn, index = _WebEvidence(), [], 0, 2
    observation_prefix = "Tool observation (source material):\n"
    while index < len(messages) and messages[index]["role"] == "assistant":
        require(turn < 19 and index + 1 < len(messages), "Historical tool exceeds TRAIN20 budget")
        action = strict_json(messages[index]["content"])
        schema = web_review_schema(web_action_schema(columns, evidence.fetched_sources,
            successful_search=bool(evidence.searches), cached_rows=profile.cached_rows, **schema_options))
        Draft202012Validator(schema).validate(action)
        observation_message = messages[index + 1]
        if "answer" in action:
            # A rejected final consumes a model turn but executes no tool. Replay
            # only the runner's exact public feedback; this grants no native
            # trace or accepted-SFT eligibility to a later failed request.
            reasons = evidence.reject_reasons(task, action["answer"])
            require(bool(reasons), "Historical final was not rejected by the public evidence guard")
            require(observation_message == {"role": "user", "content":
                    evidence.feedback(reasons, 20 - turn - 1, task)},
                    "Rejected final feedback differs from exact public evidence replay")
            evidence.rejections.append({"turn": turn, "reasons": reasons})
            turn += 1
            index += 2
            continue
        require(action.get("tool") in {"web_search", "fetch_url"},
                "Historical actions must be exact reviewed tools or publicly rejected finals")
        require(observation_message["role"] == "user"
                and observation_message["content"].startswith(observation_prefix), "Actual tool observation required")
        text = observation_message["content"][len(observation_prefix):]
        suffix = f"\nRemaining turns: {20 - turn - 1}."
        require(text.endswith(suffix), "Actual remaining-turn counter differs from TRAIN20")
        observation = strict_json(text[:-len(suffix)])
        require(observation_message["content"] == observation_prefix
                + json.dumps(observation, ensure_ascii=False) + suffix,
                "Tool observation bytes differ from actual runner serialization")
        if cached_replay is not None:
            cached_replay.observe(action, observation, turn)
        evidence.observe(action["tool"], observation)
        events.append({"turn": turn, "tool": action["tool"], "arguments": action["arguments"],
                       "observation": observation})
        turn += 1
        index += 2
    require(turn < 20, "Current request exceeds approved TRAIN20 budget")
    if collector is not None:
        require(isinstance(collector, dict) and type(collector.get("turn")) is int
                and collector["turn"] == turn, "Recorded turn differs from exact public replay")
    if cached_replay is not None:
        cached_replay.finish(turn)
    navigation = WebNavigationLedger();navigation.observe_events(events)
    controls = [{"role": "user", "content": navigation.render()}]
    if turn == 19:
        controls.append(web_review_final_control(final_turn_message(task, evidence)))
    require(messages[index:] == controls,
            "Exact actual navigation snapshot and approved last-turn control required")
    require(controls[0]["content"].startswith(WEB_NAVIGATION_PREFIX), "Navigation renderer prefix changed")
    schema = (web_final_schema(columns, evidence.fetched_sources, **schema_options) if turn == 19 else
              web_action_schema(columns, evidence.fetched_sources,
                                successful_search=bool(evidence.searches), cached_rows=profile.cached_rows, **schema_options))
    return web_review_schema(schema), set(range(index, len(messages)))

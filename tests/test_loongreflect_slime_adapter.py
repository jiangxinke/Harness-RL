from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from slime_plugins.loongreflect import RolloutTree
from slime_plugins.loongreflect.self_distill import (
    TEACHER_SYSTEM_PROMPT,
    build_update_objectives,
    ema_tau,
    reflection_leaks_answer,
)
from slime_plugins.loongreflect.slime_adapter import (
    GeneratedTurn,
    LabelAnswerValidator,
    SGLangHarnessModel,
    _save_debug_tree_samples,
    actor_prompt_for_protocol,
    build_failed_accounting_sample,
    build_training_trajectory,
    build_tree_training_samples,
    compute_tree_node_reward,
    parse_actor_output,
    reward_func,
    sanitize_generated_turn,
    select_training_leaf,
    tree_metrics,
    truncate_training_trajectory,
)


class TinyTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(text.encode())}

    def decode(self, token_ids, skip_special_tokens=False):
        return bytes(token_ids).decode()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        prompt = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        return prompt + ("<assistant>" if add_generation_prompt else "")


class BoundaryMergeTokenizer(TinyTokenizer):
    """Tiny tokenizer whose ``><`` token reproduces the observed Qwen boundary."""

    def __call__(self, text, add_special_tokens=False):
        token_ids = []
        index = 0
        while index < len(text):
            if text.startswith("><", index):
                token_ids.append(256)
                index += 2
            else:
                token_ids.append(ord(text[index]))
                index += 1
        return {"input_ids": token_ids}

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(
            "><" if token_id == 256 else chr(token_id) for token_id in token_ids
        )


class SlimeAdapterTests(unittest.TestCase):
    def test_search_answer_prompt_has_only_search_and_answer_actions(self):
        prompt = actor_prompt_for_protocol("search_answer")
        self.assertIn("<search>", prompt)
        self.assertIn("<answer>", prompt)
        permitted_section = prompt.split(
            "Only these two public actions are permitted:", 1
        )[1]
        permitted_section = permitted_section.split("Never output", 1)[0]
        self.assertNotIn("<reflect>", permitted_section)
        self.assertNotIn("<backtrack>", permitted_section)

    def test_sglang_generate_posts_one_url_and_returns_sampled_tokens(self):
        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "text": "sampled",
                    "meta_info": {
                        "finish_reason": {"type": "stop"},
                        "output_token_logprobs": [[-0.1, 7], [-0.2, 8]],
                    },
                }

        class Client:
            def __init__(self):
                self.request = None

            def post(self, url, *, json):
                self.request = (url, json)
                return Response()

        model = object.__new__(SGLangHarnessModel)
        model.url = "http://router/generate"
        model.client = Client()
        model.args = SimpleNamespace(rollout_top_p=0.9)
        model.tokenizer = SimpleNamespace(
            decode=lambda token_ids, skip_special_tokens=False: "canonical-from-ids"
        )

        generated = model._generate(
            "prompt", max_tokens=32, stop=["</search>"], temperature=0.7
        )

        self.assertEqual(model.client.request[0], model.url)
        self.assertEqual(model.client.request[1]["text"], "prompt")
        self.assertEqual(generated.token_ids, [7, 8])
        self.assertEqual(generated.log_probs, [-0.1, -0.2])
        self.assertEqual(generated.raw, "canonical-from-ids")
        self.assertEqual(generated.server_raw, "sampled")

    def test_actor_output_requires_thinking_before_action(self):
        thinking, action = parse_actor_output(
            "<think>I need evidence.</think><search>who wrote it</search>"
        )
        self.assertEqual(thinking, "I need evidence.")
        self.assertEqual(action, "<search>who wrote it</search>")
        with self.assertRaises(ValueError):
            parse_actor_output("<search>who wrote it</search>")

    def test_label_validator_rejects_without_revealing_reference(self):
        validator = LabelAnswerValidator(["Paris"])
        rejected = validator.validate(query="capital?", answer="Lyon")
        accepted = validator.validate(query="capital?", answer="Paris")
        self.assertFalse(rejected.accepted)
        self.assertNotIn("Paris", rejected.feedback)
        self.assertTrue(accepted.accepted)

    def test_generated_turn_discards_predicted_observation_at_bpe_boundary(self):
        tokenizer = BoundaryMergeTokenizer()
        raw = "reason</think><search>query</search><observation>predicted by model</observation>"
        token_ids = tokenizer(raw)["input_ids"]
        generated = GeneratedTurn(raw, token_ids, [-0.1] * len(token_ids), "stop")

        sanitized, suffix, actor_output = sanitize_generated_turn(generated, tokenizer)

        self.assertEqual(actor_output, "reason</think><search>query</search>")
        self.assertEqual(sanitized.raw + suffix, actor_output)
        self.assertEqual(suffix, ">")
        self.assertNotIn("observation", sanitized.raw + suffix)
        self.assertEqual(len(sanitized.token_ids), len(sanitized.log_probs))

    def test_incomplete_backtrack_is_not_recast_as_plain_reflection(self):
        tokenizer = TinyTokenizer()
        raw = "reason</think><reflect>bad route</reflect><backtrack>unfinished"
        token_ids = tokenizer(raw)["input_ids"]
        generated = GeneratedTurn(raw, token_ids, [-0.1] * len(token_ids), "length")

        sanitized, suffix, actor_output = sanitize_generated_turn(generated, tokenizer)

        self.assertEqual(sanitized, generated)
        self.assertEqual(suffix, "")
        self.assertEqual(actor_output, raw)

    def test_trajectory_trains_generation_and_masks_observation(self):
        tree = RolloutTree("question")
        raw = "<think>reason</think><search>query</search>"
        generated = "reason</think><search>query</search>"
        think = tree.append(
            0,
            "think",
            "reason",
            {
                "raw_output": raw,
                "forced_prefix": "<think>",
                "generated_output": generated,
                "token_ids": list(generated.encode()),
                "log_probs": [-0.1] * len(generated),
            },
        )
        search = tree.append(think, "search", "query")
        observation = tree.append(search, "observation", "evidence")
        answer_raw = "<think>done</think><answer>answer</answer>"
        answer_generated = "done</think><answer>answer</answer>"
        answer_think = tree.append(
            observation,
            "think",
            "done",
            {
                "raw_output": answer_raw,
                "forced_prefix": "<think>",
                "generated_output": answer_generated,
                "token_ids": list(answer_generated.encode()),
                "log_probs": [-0.2] * len(answer_generated),
            },
        )
        tree.append(answer_think, "answer", "answer")
        tree.status = "completed"

        response, tokens, mask, log_probs = build_training_trajectory(
            tree, TinyTokenizer()
        )
        observation_text = "<observation>evidence</observation>"
        observation_start = response.index(observation_text)
        self.assertEqual(
            mask[observation_start : observation_start + len(observation_text)],
            [0] * len(observation_text),
        )
        self.assertEqual(len(tokens), len(mask))
        self.assertEqual(len(tokens), len(log_probs))
        self.assertEqual(response.count("<search>query</search>"), 1)

        metrics = tree_metrics(tree)
        sample = SimpleNamespace(
            metadata={"reflect_metrics": metrics}, label=["Answer"]
        )
        score = asyncio.run(reward_func(SimpleNamespace(), sample))
        self.assertAlmostEqual(score, 1.15)

    def test_metrics_only_score_the_selected_training_branch(self):
        tree = RolloutTree("question")
        failed_think = tree.append(
            0,
            "think",
            "bad branch",
            {"raw_output": "<think>bad branch</think><search>wrong</search>"},
        )
        failed_search = tree.append(failed_think, "search", "wrong")
        failed_observation = tree.append(failed_search, "observation", "irrelevant")
        failed_reflect = tree.append(failed_observation, "reflect", "wrong route")
        tree.append(failed_reflect, "backtrack", "restart")

        update = tree.append(0, "update", "try a direct answer")
        answer_think = tree.append(
            update,
            "think",
            "answer now",
            {"raw_output": "<think>answer now</think><answer>Paris</answer>"},
        )
        answer = tree.append(answer_think, "answer", "Paris", {"accepted": True})
        tree.status = "completed"

        metrics = tree_metrics(tree)
        self.assertEqual(select_training_leaf(tree), answer)
        self.assertEqual(metrics["training_leaf_id"], answer)
        self.assertEqual(metrics["unique_searches"], 0)
        self.assertEqual(metrics["reflections"], 0)
        self.assertEqual(metrics["answers"], ["Paris"])
        sample = SimpleNamespace(metadata={"reflect_metrics": metrics}, label=["Paris"])
        self.assertAlmostEqual(
            asyncio.run(reward_func(SimpleNamespace(), sample)),
            1.1,
        )

    def test_rejected_answer_is_not_selected_over_latest_terminal_branch(self):
        tree = RolloutTree("question")
        rejected_think = tree.append(0, "think", "guess")
        tree.append(rejected_think, "answer", "wrong", {"accepted": False})
        update = tree.append(0, "update", "try again")
        reflect = tree.append(update, "reflect", "stopped", {"source": "harness"})
        terminal = tree.append(reflect, "backtrack", "max_steps_reached")
        tree.status = "terminated"

        self.assertEqual(select_training_leaf(tree), terminal)
        self.assertEqual(tree_metrics(tree)["answers"], [])

    def test_training_trajectory_hard_caps_environment_tokens(self):
        tokenizer = TinyTokenizer()
        token_ids = list(b"generated-environment")
        response, tokens, mask, log_probs, truncated = truncate_training_trajectory(
            "generated-environment",
            token_ids,
            [1] * 9 + [0] * (len(token_ids) - 9),
            [-0.1] * 9 + [0.0] * (len(token_ids) - 9),
            tokenizer,
            12,
        )
        self.assertTrue(truncated)
        self.assertEqual(len(tokens), 12)
        self.assertEqual(len(mask), 12)
        self.assertEqual(len(log_probs), 12)
        self.assertEqual(response, tokenizer.decode(tokens))

    def test_compact_tree_trains_every_unique_model_turn_once(self):
        from slime.utils.types import Sample

        tokenizer = TinyTokenizer()
        tree = RolloutTree("capital?")
        failed_generated = (
            "bad</think><reflect>insufficient</reflect><backtrack>restart</backtrack>"
        )
        failed_think = tree.append(
            0,
            "think",
            "bad",
            {
                "model_turn_id": 0,
                "raw_output": "<think>" + failed_generated,
                "forced_prefix": "<think>",
                "generated_output": failed_generated,
                "generation_prompt": "student-a<think>",
                "generation_context": "",
                "token_ids": list(failed_generated.encode()),
                "log_probs": [-0.1] * len(failed_generated),
            },
        )
        failed_reflect = tree.append(failed_think, "reflect", "insufficient")
        tree.append(failed_reflect, "backtrack", "restart")
        update = tree.append(0, "update", "try direct evidence")
        answer_generated = "done</think><answer>Paris</answer>"
        answer_think = tree.append(
            update,
            "think",
            "done",
            {
                "model_turn_id": 1,
                "raw_output": "<think>" + answer_generated,
                "forced_prefix": "<think>",
                "generated_output": answer_generated,
                "generation_prompt": "student-b<think>",
                "generation_context": "<update>try direct evidence</update>",
                "token_ids": list(answer_generated.encode()),
                "log_probs": [-0.2] * len(answer_generated),
            },
        )
        tree.append(answer_think, "answer", "Paris", {"accepted": True})
        tree.status = "completed"
        tree.model_turns = 2
        source = Sample(index=42, group_index=3, prompt="capital?", label=["Paris"])
        args = SimpleNamespace(
            seq_length=4096,
            reflect_answer_path_weight=2.0,
            reflect_wrong_answer_reward=-0.75,
            reflect_truncation_reward=-0.5,
            reflect_failed_branch_reward=-0.3,
            reflect_partial_branch_reward=-0.2,
            reflect_search_bonus=0.05,
            reflect_action_bonus=0.08,
            reflect_answer_leak_penalty=1.0,
        )

        samples = build_tree_training_samples(args, source, tree, tokenizer)

        self.assertEqual(len(samples), tree.model_turns)
        self.assertEqual({sample.rollout_id for sample in samples}, {42})
        self.assertEqual(
            {sample.metadata["model_turn_id"] for sample in samples},
            {0, 1},
        )
        self.assertEqual(
            {sample.metadata["tree_node_id"] for sample in samples},
            {failed_think, answer_think},
        )
        for sample in samples:
            self.assertEqual(len(sample.loss_mask), sample.response_length)
            self.assertEqual(len(sample.position_rewards), sample.response_length)
            self.assertEqual(len(sample.reflect_mask), sample.response_length)
            self.assertEqual(len(sample.rollout_log_probs), sample.response_length)
            self.assertGreater(len(sample.teacher_tokens), sample.response_length)
            for reward, mask in zip(
                sample.position_rewards, sample.loss_mask, strict=True
            ):
                if mask:
                    self.assertNotEqual(reward, 0.0)
        failed_sample = next(
            sample
            for sample in samples
            if sample.metadata["tree_node_id"] == failed_think
        )
        answer_sample = next(
            sample
            for sample in samples
            if sample.metadata["tree_node_id"] == answer_think
        )
        self.assertTrue(any(failed_sample.reflect_mask))
        self.assertFalse(any(answer_sample.reflect_mask))
        self.assertGreater(
            answer_sample.metadata["tree_node_reward"],
            failed_sample.metadata["tree_node_reward"],
        )
        self.assertEqual(answer_sample.status, Sample.Status.COMPLETED)
        teacher_prompt = tokenizer.decode(
            answer_sample.teacher_tokens[: -answer_sample.response_length]
        )
        self.assertIn("Paris", teacher_prompt)
        self.assertIn("must never be exposed", teacher_prompt)

    def test_compact_status_distinguishes_budget_truncation_from_failures(self):
        from slime.utils.types import Sample

        tokenizer = TinyTokenizer()
        source = Sample(index=9, prompt="question", label="answer")
        args = SimpleNamespace(seq_length=4096)

        def build(reason):
            tree = RolloutTree("question")
            generated = "reason</think><reflect>check</reflect>"
            tree.append(
                tree.root_id,
                "think",
                "reason",
                {
                    "model_turn_id": 0,
                    "generated_output": generated,
                    "generation_prompt": "student<think>",
                    "generation_context": "",
                    "token_ids": list(generated.encode()),
                    "log_probs": [-0.1] * len(generated),
                },
            )
            tree.status = "terminated"
            tree.termination_reason = reason
            return build_tree_training_samples(args, source, tree, tokenizer)[0]

        self.assertEqual(build("max_steps_reached").status, Sample.Status.TRUNCATED)
        self.assertEqual(
            build("actor_error: TypeError: bad request").status,
            Sample.Status.FAILED,
        )

    def test_terminal_reward_distinguishes_budget_truncation_from_failures(self):
        args = SimpleNamespace(
            reflect_truncation_reward=-0.5,
            reflect_failed_branch_reward=-0.3,
        )

        def reward(reason, *, termination=True, reason_on_leaf=False):
            tree = RolloutTree("question")
            think = tree.append(
                tree.root_id,
                "think",
                "reason",
                {"raw_output": "<think>reason</think>"},
            )
            reflect = tree.append(
                think,
                "reflect",
                "terminated",
                {"source": "harness", "termination_reason": reason},
            )
            tree.append(
                reflect,
                "backtrack",
                reason,
                {
                    "source": "harness",
                    "termination": termination,
                    **({"termination_reason": reason} if reason_on_leaf else {}),
                },
            )
            return compute_tree_node_reward(args, tree, think, "answer")

        self.assertAlmostEqual(reward("max_steps_reached"), -0.48)
        self.assertAlmostEqual(reward("max_output_tokens_reached"), -0.48)
        self.assertAlmostEqual(
            reward("max_output_tokens_reached", reason_on_leaf=True), -0.48
        )
        self.assertAlmostEqual(reward("actor_error"), -0.28)
        self.assertAlmostEqual(reward("invalid_action", termination=False), -0.28)

    def test_pre_token_failure_writes_masked_accounting_debug_sample(self):
        from slime.utils.types import Sample

        tree = RolloutTree("question")
        reflect = tree.append(
            tree.root_id,
            "reflect",
            "terminated",
            {"source": "harness", "termination_reason": "actor_error"},
        )
        terminal = tree.append(
            reflect,
            "backtrack",
            "actor_error: RuntimeError: unavailable",
            {
                "source": "harness",
                "termination": True,
                "termination_reason": "actor_error",
            },
        )
        tree.status = "terminated"
        tree.termination_reason = "actor_error: RuntimeError: unavailable"
        source = Sample(index=7, prompt="question", label="answer")

        with tempfile.TemporaryDirectory() as output_dir:
            args = SimpleNamespace(
                reflect_debug_dir=output_dir,
                reflect_failed_branch_reward=-0.3,
            )
            sample = build_failed_accounting_sample(
                args, source, tree, TinyTokenizer(), "student prompt"
            )
            _save_debug_tree_samples(args, tree, [sample])

            node_sample = json.loads(
                (Path(output_dir) / "tree_7" / "node_samples.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            summary = json.loads(
                (Path(output_dir) / "tree_7" / "summary.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertTrue(sample.remove_sample)
        self.assertEqual(sample.status, Sample.Status.FAILED)
        self.assertEqual(sample.reward, -0.3)
        self.assertFalse(any(sample.loss_mask))
        self.assertEqual(sample.teacher_tokens, sample.tokens)
        self.assertEqual(
            sample.metadata["reflect_tree"]["nodes"][terminal]["metadata"][
                "termination_reason"
            ],
            "actor_error",
        )
        self.assertIsNone(node_sample["model_turn_id"])
        self.assertIsNone(node_sample["node_id"])
        self.assertTrue(node_sample["remove_sample"])
        self.assertTrue(node_sample["synthetic_accounting_sample"])
        self.assertEqual(summary["compact_samples"], 1)
        self.assertEqual(summary["training_samples"], 0)
        self.assertEqual(summary["accounting_samples"], 1)

    def test_wo_reflect_pre_token_failure_uses_no_reflect_action(self):
        from slime.utils.types import Sample

        tree = RolloutTree("question")
        terminal = tree.append(
            tree.root_id,
            "termination",
            "actor_error",
            {"source": "harness", "termination": True},
        )
        tree.status = "terminated"
        tree.termination_reason = "actor_error"
        sample = build_failed_accounting_sample(
            SimpleNamespace(
                reflect_action_protocol="search_answer",
                reflect_failed_branch_reward=-0.3,
            ),
            Sample(index=8, prompt="question", label="answer"),
            tree,
            TinyTokenizer(),
            "student prompt",
        )
        self.assertEqual(sample.response, "<answer>unknown</answer>")
        self.assertNotIn("<reflect>", sample.response)
        self.assertNotIn("<backtrack>", sample.response)
        self.assertEqual(
            sample.metadata["reflect_tree"]["nodes"][terminal]["kind"],
            "termination",
        )

    def test_teacher_policy_and_fast_slow_schedules(self):
        self.assertIn("NEVER quote, name, paraphrase, hint at", TEACHER_SYSTEM_PROMPT)
        self.assertTrue(reflection_leaks_answer("The answer is Paris", ["Paris"]))
        self.assertFalse(
            reflection_leaks_answer("Need stronger geographic evidence", ["Paris"])
        )
        self.assertAlmostEqual(ema_tau(0, 100, 0.996, 1.0), 0.996)
        self.assertAlmostEqual(ema_tau(99, 100, 0.996, 1.0), 1.0)
        objectives = build_update_objectives(
            SimpleNamespace(
                reflect_train_mode="fast_slow",
                reflect_fast_updates_per_step=2,
                reflect_fast_distill_weight=1.0,
            )
        )
        self.assertEqual(
            [objective.name for objective in objectives],
            ["reflect_1", "reflect_2", "verifier"],
        )
        self.assertEqual(objectives[0].policy_weight, 0.0)
        self.assertEqual(objectives[0].distill_weight, 1.0)
        self.assertEqual(objectives[-1].policy_weight, 1.0)
        self.assertEqual(objectives[-1].distill_weight, 0.0)

    def test_batched_compact_reward_contract(self):
        samples = [
            SimpleNamespace(metadata={"tree_node_reward": 0.25}),
            SimpleNamespace(metadata={"tree_node_reward": -0.5}),
        ]
        self.assertEqual(
            asyncio.run(reward_func(SimpleNamespace(), samples)),
            [0.25, -0.5],
        )


if __name__ == "__main__":
    unittest.main()

# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
"""WideSearch scoring, adapted from ByteDance-Seed/WideSearch.

The official protocol uses an LLM both for entity alignment and cell judgments.
Without a judge, scores are explicitly labeled as a strict, deterministic proxy.
Providing a custom judge follows the protocol, but does not reproduce the
published GPT-4.1 judge. The judge must only run after an episode has finished.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from io import StringIO
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pandas as pd

ALIGN_PROMPT = """Your task is to align two vocabularies. The inputs are the vocabulary to be aligned and the reference vocabulary respectively. Note that you need to perform semantic alignment (not positional alignment). If two strings are exactly the same, they must correspond to each other. These two strings are supposed to represent the same entity, with differences only in the expression forms and formats.


The vocabulary to be aligned is as follows:
{response}

The reference vocabulary is as follows:
{reference}

The alignment rules are as follows:
List the values in the vocabulary to be aligned one by one. If there is a value in the reference vocabulary that has the same meaning as this value, `transform` should be represented as the value from the reference vocabulary; otherwise, `transform` should be represented as the original value from the vocabulary to be aligned.

Note that `origin` must be taken from the vocabulary to be aligned keeping the original format, and `transform` must be taken from the reference vocabulary. For example: Some words in the vocabulary to be aligned might be the words in the reference vocabulary with Markdown formatting added, keep the to be aligned format in `origin` and the reference format in `transform`.

For the `origin`, first find the `transform` that is the closest in meaning and then judge whether they correspond to each other. Those entities not correspond to each other could not output.

Please output the alignment results in the following format:
```json
{{
    "origin_str1": "transform_str1",
    "origin_str2": "transform_str2"
}}
```
"""

CELL_PROMPT = """You are an expert in grading answers. Your task is to score the responses to a certain question. Below, you will be provided with a set of standard answers, a set of responses to be graded, and specific grading criteria.

Each answer and each response has an idx. Please score each pair of answers and responses in this set according to the following methods:
1. The scoring range is from 0 to 1. A score of 1 indicates a completely correct answer. For deduction items, please refer to the specific grading criteria section.
2. After reading the standard answers, responses to be graded, and grading criteria, please first analyze and judge them item by item according to the grading criteria.
3. The score can only be an integer of 0 or 1.
4. After the analysis and judgment, please provide the final scoring results. Each pair should have a score. Output in Markdown JSON format, as shown below:
```json
{{
    "idx_xxx": score,
    "idx_yyy": score,
    ...
}}
```

====== criterion-start ======
{criterion}
====== criterion-end ======

====== response-start ======
{response}
====== response-end ======

Now start scoring. Please make sure to analyze each item step by step before providing the final scoring results.

"""

Judge = Callable[[str], str | dict[str, Any]]


class JudgeError(RuntimeError):
    """Infrastructure or malformed judge output; never a model score of zero."""


class OpenAIJudge:
    """Small synchronous adapter for a local or hosted OpenAI chat endpoint."""

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY", timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def __call__(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 10240,
        }
        request = Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                content = json.load(response)["choices"][0]["message"]["content"]
        except Exception as exc:
            raise JudgeError("WideSearch judge request failed") from exc
        if not isinstance(content, str) or not content:
            raise JudgeError("WideSearch judge returned no content")
        return content


def norm_column(value: str) -> str:
    return str(value).strip().lower().replace(" ", "")


def _judge_json(judge: Judge, prompt: str) -> dict:
    try:
        result = judge(prompt)
        if isinstance(result, dict):
            return result
        matches = re.findall(r"```json\s*(\{.*?\})\s*```", result, re.DOTALL)
        result = json.loads(matches[-1] if matches else result)
        if not isinstance(result, dict):
            raise ValueError("expected JSON object")
        return result
    except JudgeError:
        raise
    except Exception as exc:
        raise JudgeError("Malformed WideSearch judge response") from exc


def _batches(values: list[Any], batch_size: int, max_chars: int) -> Iterator[tuple[int, list[Any]]]:
    """Partition without dropping an oversized item or changing global indices."""
    start, batch, chars = 0, [], 0
    for index, value in enumerate(values):
        size = len(str(value))
        if batch and (len(batch) >= batch_size or chars + size > max_chars):
            yield start, batch
            start, batch, chars = index, [], 0
        batch.append(value)
        chars += size
    if batch:
        yield start, batch


def _align(response: list[str], reference: list[str], judge: Judge, batch_size: int, max_chars: int) -> dict:
    mapping = {}
    for _, batch in _batches(response, batch_size, max_chars):
        # Every response batch sees the FULL reference vocabulary. Splitting
        # the reference could change the entity matching problem.
        result = _judge_json(judge, ALIGN_PROMPT.format(response=batch, reference=reference))
        if any(not isinstance(value, str) or (value not in reference and value != key) for key, value in result.items()):
            raise JudgeError("WideSearch alignment returned an unknown reference")
        mapping.update({key: value for key, value in result.items() if key in batch})
    return mapping


def _judge_cells(predicted: list[str], targets: list[str], criterion: str, judge: Judge, batch_size: int, max_chars: int) -> list[int]:
    pairs = [{"response": a, "target": b} for a, b in zip(predicted, targets, strict=True)]
    values = []
    for start, batch in _batches(pairs, batch_size, max_chars):
        indexed = {f"idx_{start + offset}": pair for offset, pair in enumerate(batch)}
        judgments = _judge_json(judge, CELL_PROMPT.format(criterion=criterion, response=indexed))
        scores = [judgments.get(f"idx_{index}") for index in range(start, start + len(batch))]
        if any(score not in (0, 1) for score in scores):
            raise JudgeError("WideSearch judge omitted a cell or returned a nonbinary score")
        values.extend(scores)
    return values


def extract_table(prediction: str | list[dict]) -> pd.DataFrame | None:
    """Accept native JSON rows or the benchmark's fenced Markdown table."""
    if isinstance(prediction, list):
        return pd.DataFrame(prediction)
    if not isinstance(prediction, str):
        return None
    try:
        parsed = json.loads(prediction)
        if isinstance(parsed, list):
            return pd.DataFrame(parsed)
    except (ValueError, TypeError):
        pass
    fenced = re.findall(r"```markdown(.*?)```", prediction, re.DOTALL)
    text = fenced[0] if fenced else prediction
    lines = [line.strip() for line in text.splitlines() if line.strip().startswith("|")]
    if len(lines) < 2:
        return None
    rows = []
    for line in lines:
        cells = re.split(r"(?<!\\)\|", line.strip().strip("|"))
        cells = [cell.strip().replace(r"\|", "|") for cell in cells]
        if all(re.fullmatch(r":?-+:?", cell.replace(" ", "")) for cell in cells):
            continue
        rows.append(cells)
    if len(rows) < 2 or any(len(row) != len(rows[0]) for row in rows):
        return None
    # Use CSV inference to match the reference loader's numeric handling.
    import csv

    stream = StringIO()
    csv.writer(stream).writerows(rows)
    stream.seek(0)
    return pd.read_csv(stream)


def _preprocess(value: str, name: str) -> str:
    if name == "norm_str":
        return str(value).lower().strip().replace(" ", "").replace("*", "")
    if name == "extract_number":
        numbers = re.findall(r"[-+]?\d*\.\d+%?|[-+]?\d+\.?\d*%?", str(value).replace(",", ""))
        return numbers[0] if numbers else "NULL"
    if name == "norm_date":
        import dateparser

        date = dateparser.parse(value, settings={"PREFER_DAY_OF_MONTH": "first"})
        return date.strftime("%Y-%m-%d") if date else value
    raise ValueError(f"Unknown WideSearch preprocess: {name}")


def _metric(response: str, target: str, name: str, criterion: Any = None) -> float:
    if name == "exact_match":
        return float(response.lower() == target.lower())
    if name == "in_match":
        return float(response in target)
    if name == "url_match":
        pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        return float({urlparse(u).netloc for u in re.findall(pattern, response)} == {urlparse(u).netloc for u in re.findall(pattern, target)})
    if name == "number_near":

        def number(value):
            return float(value.replace("%", "")) / (100 if "%" in value else 1)

        try:
            left, right = number(response), number(target)
        except (ValueError, TypeError):
            return float(response == target)
        return float(abs(left - right) <= abs(right) * criterion)
    if name == "date_near":
        import dateparser

        try:
            left = dateparser.parse(response, settings={"PREFER_DAY_OF_MONTH": "first"})
        except Exception:
            left = None
        try:
            right = dateparser.parse(target, settings={"PREFER_DAY_OF_MONTH": "first"})
        except Exception:
            right = None
        if left is None or right is None:
            return float(left is None and right is None)
        return float(abs((left - right).days) <= 31)
    raise ValueError(f"Unknown WideSearch metric: {name}")


def evaluate_widesearch(
    prediction: str | list[dict],
    gold: list[dict],
    evaluation: dict,
    *,
    judge: Judge | None = None,
    judge_batch_size: int = 32,
    judge_max_batch_chars: int = 32000,
) -> dict[str, Any]:
    """Return official row/item metrics, with the judge protocol recorded.

    Judge outages raise ``JudgeError`` instead of lowering a policy's score.
    With no judge, semantic cells receive exact-string proxy scores. Cell and
    response-vocabulary judgments are batched (32 by default), increasing the
    number of calls relative to upstream. References and metric denominators
    are retained in full. A single large item is never truncated here.
    """
    if judge_batch_size < 1 or judge_max_batch_chars < 1:
        raise ValueError("WideSearch judge batch limits must be positive")
    result: dict[str, Any] = {
        "score": 0.0,
        "precision_by_row": 0.0,
        "recall_by_row": 0.0,
        "f1_by_row": 0.0,
        "precision_by_item": 0.0,
        "recall_by_item": 0.0,
        "f1_by_item": 0.0,
        "evaluation_protocol": "official_protocol_custom_judge" if judge else "strict_exact_proxy",
        "judge_model": getattr(judge, "model", None),
        "judge_batch_size": judge_batch_size if judge else None,
        "judge_max_batch_chars": judge_max_batch_chars if judge else None,
        "msg": "",
    }
    required = evaluation["required"]
    unique = evaluation["unique_columns"]
    pipeline = evaluation["eval_pipeline"]
    answer = pd.DataFrame(gold)
    if isinstance(prediction, list) and "citations" not in required:
        # The harness attaches evidence separately to each native JSON row.
        # It is not part of the benchmark's answer table schema.
        prediction = [{key: value for key, value in row.items() if key != "citations"} for row in prediction]
    response = extract_table(prediction)
    if response is None or response.empty:
        result["msg"] = "No response table"
        return result
    answer.columns = [norm_column(c) for c in answer.columns]
    response.columns = [norm_column(c) for c in response.columns]
    if len(set(response.columns)) != len(response.columns):
        result["msg"] = "Duplicate response columns"
        return result
    if set(required) != set(response.columns) and judge:
        response.rename(columns=_align(response.columns.tolist(), required, judge, judge_batch_size, judge_max_batch_chars), inplace=True)
    if set(required) != set(response.columns):
        result["msg"] = "Response columns do not match required columns"
        return result
    if not set(required).issubset(answer.columns):
        raise ValueError("WideSearch gold is missing required columns")
    answer = answer[required].copy()
    response = response[required].copy()
    for column in required:
        if (answer[column].dtype == float and response[column].dtype == int) or (answer[column].dtype == int and response[column].dtype == float):
            answer[column] = answer[column].astype(float)
            response[column] = response[column].astype(float)
        answer[column] = answer[column].astype(str)
        response[column] = response[column].astype(str)
    response.drop_duplicates(subset=unique, inplace=True)
    answer.drop_duplicates(subset=unique, inplace=True)
    if judge:
        for column in unique:
            if {"llm_judge", "exact_match"}.intersection(pipeline.get(column, {}).get("metric", [])):
                mapping = _align(response[column].tolist(), answer[column].tolist(), judge, judge_batch_size, judge_max_batch_chars)
                response[column] = response[column].apply(lambda x, mapping=mapping: mapping.get(x, x))
    for column, rules in pipeline.items():
        for name in rules.get("preprocess", []):
            answer[column] = answer[column].apply(lambda x, name=name: _preprocess(x, name))
            response[column] = response[column].apply(lambda x, name=name: _preprocess(x, name))
    matched = pd.merge(answer, response, on=unique, how="inner", suffixes=("_query", "_response"))
    scores = pd.DataFrame(index=matched.index)
    for column in required:
        if column in unique:
            scores[column + "_exact_match"] = 1.0
            continue
        rules = pipeline[column]
        for metric in rules.get("metric", []):
            predicted = matched[column + "_response"].tolist()
            targets = matched[column + "_query"].tolist()
            if metric == "llm_judge" and judge and predicted:
                values = _judge_cells(predicted, targets, rules.get("criterion"), judge, judge_batch_size, judge_max_batch_chars)
            else:
                values = [_metric(a, b, "exact_match" if metric == "llm_judge" else metric, rules.get("criterion")) for a, b in zip(predicted, targets, strict=True)]
            scores[column + "_" + metric] = values
    row_true = float(scores.min(axis=1).sum())
    item_true = float(scores.sum().sum())
    n_pred, n_gold, width = len(response), len(answer), len(required)
    for label, true, denominator in [("row", row_true, 1), ("item", item_true, width)]:
        precision = true / (n_pred * denominator) if n_pred else 0.0
        recall = true / (n_gold * denominator) if n_gold else 0.0
        result[f"precision_by_{label}"] = precision
        result[f"recall_by_{label}"] = recall
        result[f"f1_by_{label}"] = 2 * precision * recall / (precision + recall) if precision + recall > 1e-9 else 0.0
    result["score"] = float(all(result[key] == 1 for key in result if key.startswith(("precision_", "recall_", "f1_"))))
    result["num_pred_rows"] = n_pred
    result["num_gold_rows"] = n_gold
    result["matched_rows"] = len(matched)
    return result

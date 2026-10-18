"""Independent web-tool curricula from public registries, never benchmark labels."""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass

from .data import Task


@dataclass(frozen=True)
class Registry:
    key: str
    title: str
    url: str
    columns: tuple[tuple[str, str], ...]
    ranges: tuple[tuple[int, int], ...]
    split: str


REGISTRIES = (
    Registry("http_status", "IANA HTTP Status Code Registry",
             "https://www.iana.org/assignments/http-status-codes/http-status-codes-1.csv",
             (("Value", "number"), ("Description", "name")),
             ((100, 199), (200, 299), (300, 399), (400, 419), (420, 499), (500, 599)), "train"),
    Registry("ip_protocols", "IANA Protocol Numbers Registry",
             "https://www.iana.org/assignments/protocol-numbers/protocol-numbers-1.csv",
             (("Decimal", "number"), ("Keyword", "name"), ("Protocol", "description")),
             tuple((start, start + 9) for start in range(0, 150, 10)), "train"),
    Registry("dns_types", "IANA DNS Resource Record TYPEs Registry",
             "https://www.iana.org/assignments/dns-parameters/dns-parameters-4.csv",
             (("Value", "number"), ("TYPE", "name"), ("Meaning", "description")),
             ((1, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 64), (65, 100), (101, 300)), "train"),
    Registry("icmp_types", "IANA ICMP Type Numbers Registry",
             "https://www.iana.org/assignments/icmp-parameters/icmp-parameters-types.csv",
             (("Type", "number"), ("Name", "name")),
             ((0, 10), (11, 20), (21, 40), (41, 255)), "validation"),
    Registry("arp_operations", "IANA ARP Operation Codes Registry",
             "https://www.iana.org/assignments/arp-parameters/arp-parameters-1.csv",
             (("Number", "number"), ("Operation Code (op)", "name")),
             ((1, 7), (8, 14), (15, 21), (22, 28)), "validation"),
)


def registry_tasks(registry: Registry, csv_text: str) -> list[Task]:
    """Create deterministic range questions; validation uses separate registries."""
    reader = csv.DictReader(io.StringIO(csv_text.lstrip("\ufeff")))
    needed = {original for original, _ in registry.columns}
    if not needed.issubset(reader.fieldnames or []):
        raise ValueError(f"Registry {registry.key} is missing required CSV columns")
    rows = []
    seen = set()
    number_column = registry.columns[0][0]
    for source in reader:
        number = (source.get(number_column) or "").strip()
        if not re.fullmatch(r"\d+", number):
            continue
        row = {target: (source.get(original) or "").strip() for original, target in registry.columns}
        row["number"] = str(int(number))
        if not all(row.values()) or re.match(r"^(?:unassigned|reserved|\(unused\))", row["name"], re.IGNORECASE):
            continue
        if row["number"] in seen:
            raise ValueError(f"Duplicate registry number in {registry.key}")
        seen.add(row["number"])
        rows.append(row)
    tasks = []
    required = [target for _, target in registry.columns]
    mapping = ", ".join(f"{original!r} → {target!r}" for original, target in registry.columns)
    for index, (lower, upper) in enumerate(registry.ranges):
        gold = sorted((row for row in rows if lower <= int(row["number"]) <= upper), key=lambda row: int(row["number"]))
        if len(gold) < 2:
            continue
        # Both languages appear in training. Validation has one question per
        # range, alternating languages, with no registry shared with training.
        languages = ("en", "zh") if registry.split == "train" else (("en",) if index % 2 == 0 else ("zh",))
        for language in languages:
            if language == "en":
                prompt = (f"Use the official {registry.title} and its CSV export. List every entry with an integer number "
                          f"from {lower} through {upper}, inclusive. Include only rows with nonempty required fields; "
                          "exclude names starting with Reserved, Unassigned, or (Unused). Preserve the registry's exact "
                          f"field text and use these column mappings: {mapping}. Return all matching rows, each with source citations.")
            else:
                prompt = (f"请查询官方 {registry.title} 及其 CSV 导出，列出编号为 {lower} 到 {upper}（含两端）的全部整数编号条目。"
                          "只保留所需字段均非空的行，排除名称以 Reserved、Unassigned 或 (Unused) 开头的条目。"
                          f"字段内容保留登记表原文，列名映射为：{mapping}。请返回全部匹配行，每行附上来源引用。")
            tasks.append(Task(
                id=f"independent-registry-{registry.key}-{lower}-{upper}-{language}",
                benchmark="widesearch", split=registry.split, prompt=prompt, gold=gold,
                metadata={
                    "source": "independent_curriculum", "curriculum_version": "iana-range-v1",
                    "registry": registry.key, "source_url": registry.url,
                    "teacher_query": f"{registry.title} CSV site:iana.org", "language": language,
                    "evaluation": {"required": required, "unique_columns": ["number"],
                                   "eval_pipeline": {column: {"preprocess": ["norm_str"], "metric": ["exact_match"]}
                                                     for column in required}},
                },
            ))
    return tasks

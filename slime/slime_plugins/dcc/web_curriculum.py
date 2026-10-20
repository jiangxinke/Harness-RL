"""Independent web-tool curricula from public registries, never benchmark labels."""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

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
                    "source_family": f"iana-{registry.key}",
                    "registry": registry.key, "source_url": registry.url,
                    "teacher_query": f"{registry.title} CSV site:iana.org", "language": language,
                    "evaluation": {"required": required, "unique_columns": ["number"],
                                   "eval_pipeline": {column: {"preprocess": ["norm_str"], "metric": ["exact_match"]}
                                                     for column in required}},
                },
            ))
    return tasks


@dataclass(frozen=True)
class PublicTable:
    """An independent source family and machine-readable public table."""

    key: str
    family: str
    title: str
    url: str
    split: str
    parser: str
    fields: tuple[str, ...]
    unique_columns: tuple[str, ...]
    # (field, numeric range or inclusive first-letter range, lower, upper).
    selections: tuple[tuple[str, str, str, str], ...]


_LETTER_RANGES = tuple(("name", "initial", lower, upper) for lower, upper in
                       (("A", "F"), ("G", "L"), ("M", "R"), ("S", "Z")))


PUBLIC_TABLES = (
    PublicTable(
        "pubchem-elements", "pubchem", "PubChem Periodic Table",
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/periodictable/CSV",
        "train", "pubchem", ("number", "symbol", "name", "atomic_mass", "electron_configuration", "group", "state", "discovered"), ("number",),
        tuple(("number", "integer", str(start), str(min(start + 9, 118))) for start in range(1, 119, 10)),
    ),
    PublicTable(
        "china-prefectures", "china-administrative-divisions", "Administrative divisions of China: prefecture-level cities",
        "https://raw.githubusercontent.com/modood/Administrative-divisions-of-China/master/dist/cities.json",
        "train", "china_cities", ("code", "name", "province_code"), ("code",),
        tuple(("province_code", "equal", code, code) for code in
              ("13", "14", "15", "21", "22", "23", "32", "33", "34", "35", "36", "37", "41", "42", "43", "44", "45", "51", "52", "53", "61", "62", "65")),
    ),
    *(PublicTable(
        f"nobel-{category}-{start}", "nobel-prize", f"Nobel Prize {title} laureates, {start}–{start + 19}",
        "https://api.nobelprize.org/2.1/nobelPrizes?limit=100&nobelPrizeCategory="
        f"{category}&nobelPrizeYear={start}&yearTo={start + 19}",
        "train", "nobel", ("year", "category", "name", "laureate_id", "share", "motivation"), ("year", "category", "name"),
        tuple(("year", "integer", str(year), str(year + 4)) for year in range(start, start + 20, 5)),
    ) for category, title in (("phy", "Physics"), ("che", "Chemistry")) for start in (1901, 1941, 1981)),
    *(PublicTable(
        f"worldbank-{region.lower()}", "worldbank-countries", f"World Bank countries and economies in {title}",
        f"https://api.worldbank.org/v2/region/{region}/country?format=json&per_page=100",
        "validation", "worldbank", ("code", "name", "capital", "region", "income_group", "lending_type", "longitude", "latitude"), ("code",), _LETTER_RANGES,
    ) for region, title in (
        ("LCN", "Latin America and the Caribbean"), ("ECS", "Europe and Central Asia"),
        ("SSF", "Sub-Saharan Africa"), ("EAS", "East Asia and Pacific"),
        ("MEA", "Middle East and North Africa"), ("SAS", "South Asia"),
    )),
    PublicTable(
        "gbif-bird-orders", "gbif-taxonomy", "GBIF Backbone Taxonomy: bird orders (Aves, taxon 212)",
        "https://api.gbif.org/v1/species/search?highertaxon_key=212&rank=ORDER&limit=100",
        "validation", "gbif", ("taxon_id", "name", "scientific_name", "rank", "kingdom", "phylum", "class", "parent", "status"), ("taxon_id",), _LETTER_RANGES,
    ),
)


def public_table_rows(source: PublicTable, source_text: str) -> list[dict[str, str]]:
    """Parse complete exports; never label an incomplete API page as a full list."""
    if source.parser == "pubchem" and not source_text.lstrip().startswith("{"):
        table = list(csv.reader(io.StringIO(source_text.lstrip("\ufeff"))))
        if not table:
            raise ValueError("PubChem table is empty")
        value = {"Table": {"Columns": {"Column": table[0]}, "Row": [{"Cell": cells} for cells in table[1:]]}}
    else:
        value = json.loads(source_text)
    rows: list[dict[str, Any]]
    if source.parser == "pubchem":
        table = value["Table"]
        columns = table["Columns"]["Column"]
        required = ("AtomicNumber", "Symbol", "Name", "AtomicMass", "ElectronConfiguration", "GroupBlock", "StandardState", "YearDiscovered")
        if any(column not in columns for column in required):
            raise ValueError("PubChem table is missing required columns")
        rows = []
        for row in table["Row"]:
            if len(row["Cell"]) != len(columns):
                raise ValueError("PubChem table has a malformed row")
            data = dict(zip(columns, row["Cell"], strict=True))
            rows.append(dict(zip(source.fields, (data[column] for column in required), strict=True)))
    elif source.parser == "china_cities":
        if not isinstance(value, list):
            raise ValueError("Chinese city source must contain an array")
        rows = [{"code": row["code"], "name": row["name"], "province_code": row["provinceCode"]} for row in value]
    elif source.parser == "nobel":
        prizes, meta = value["nobelPrizes"], value["meta"]
        if meta.get("offset", 0) != 0 or meta["count"] != len(prizes):
            raise ValueError("Nobel source is incomplete; all API pages are required")
        rows = [{"year": prize["awardYear"], "category": prize["category"]["en"],
                 "name": laureate.get("knownName", laureate.get("orgName", {})).get("en", ""),
                 "laureate_id": laureate["id"], "share": laureate["portion"], "motivation": laureate.get("motivation", {}).get("en", "")}
                for prize in prizes for laureate in prize.get("laureates", [])]
    elif source.parser == "worldbank":
        if not isinstance(value, list) or len(value) != 2 or value[0]["page"] != 1 or value[0]["pages"] != 1 or value[0]["total"] != len(value[1]):
            raise ValueError("World Bank source is incomplete; all API pages are required")
        rows = [{"code": row["id"], "name": row["name"], "capital": row["capitalCity"],
                 "region": row["region"]["value"], "income_group": row["incomeLevel"]["value"],
                 "lending_type": row["lendingType"]["value"], "longitude": row["longitude"], "latitude": row["latitude"]}
                for row in value[1] if row["region"]["id"] != "NA"]
    elif source.parser == "gbif":
        if value.get("offset", 0) != 0 or value.get("endOfRecords") is not True or value.get("count") != len(value["results"]):
            raise ValueError("GBIF source is incomplete; all API pages are required")
        rows = [{"taxon_id": row["key"], "name": row["canonicalName"], "scientific_name": row["scientificName"],
                 "rank": row["rank"], "kingdom": row.get("kingdom", ""), "phylum": row.get("phylum", ""),
                 "class": row.get("class", ""), "parent": row.get("parent", ""), "status": row["taxonomicStatus"]}
                for row in value["results"] if row.get("taxonomicStatus") == "ACCEPTED" and row.get("rank") == "ORDER"]
    else:
        raise ValueError(f"Unknown public-table parser: {source.parser}")
    output, seen = [], set()
    for row in rows:
        if any(row[name] is None for name in source.fields):
            continue
        normalized = {name: str(row[name]).strip() for name in source.fields}
        if not all(normalized.values()):
            continue
        key = tuple(normalized[name] for name in source.unique_columns)
        if key in seen:
            raise ValueError(f"Duplicate public-source row: {source.key}")
        seen.add(key)
        output.append(normalized)
    return output


def public_table_tasks(source: PublicTable, source_text: str) -> list[Task]:
    """Generate bilingual enumeration tasks from separate independent families."""
    rows = public_table_rows(source, source_text)
    tasks = []
    for index, (field, operation, lower, upper) in enumerate(source.selections):
        if field not in source.fields:
            raise ValueError("Selection field is not in the source schema")
        if operation == "integer":
            chosen = [row for row in rows if re.fullmatch(r"\d+", row[field]) and int(lower) <= int(row[field]) <= int(upper)]
            en = f"whose {field} is an integer from {lower} through {upper}, inclusive"
            zh = f"{field} 为 {lower} 至 {upper}（含两端）的整数"
        elif operation == "initial":
            chosen = [row for row in rows if lower <= row[field][:1].upper() <= upper]
            en = f"whose {field} starts with an English letter from {lower} through {upper}, inclusive (ignore case)"
            zh = f"{field} 以英文字母 {lower} 至 {upper}（含两端，忽略大小写）开头"
        elif operation == "equal":
            chosen = [row for row in rows if row[field] == lower]
            en, zh = f"whose {field} is exactly {lower!r}", f"{field} 恰好为 {lower!r}"
        else:
            raise ValueError("Unknown public-table selection")
        if not 2 <= len(chosen) <= 40:
            continue
        chosen.sort(key=lambda row: tuple(int(row[name]) if row[name].isdigit() else row[name] for name in source.unique_columns))
        languages = ("en", "zh") if source.split == "train" else (("en",) if index % 2 == 0 else ("zh",))
        fields = ", ".join(source.fields)
        for language in languages:
            if language == "en":
                prompt = (f"Research {source.title} using its public official export or maintained source dataset. "
                          f"List every record {en}. Return columns {fields}; preserve the exact source text, "
                          "omit rows with an empty required field, and cite the source for every row. "
                          "Return the complete filtered list, rather than examples or a prose summary.")
            else:
                prompt = (f"请查阅 {source.title} 的官方公开导出或持续维护的数据源，列出所有{zh}的记录。"
                          f"返回列 {fields}，字段保留来源原文，排除所需字段为空的行，每一行均附真实来源引用。"
                          "请返回完整筛选结果，不要只给部分示例或文字摘要。")
            tasks.append(Task(
                id=f"independent-table-v3-{source.key}-{field}-{lower}-{upper}-{language}",
                benchmark="widesearch", split=source.split, prompt=prompt, gold=chosen,
                metadata={
                    "source": "independent_curriculum", "curriculum_version": "public-tables-v3",
                    "source_family": source.family, "table": source.key, "source_url": source.url,
                    "teacher_query": f"{source.title} site:{urlsplit(source.url).hostname}",
                    "language": language, "selection": {"field": field, "operation": operation, "lower": lower, "upper": upper},
                    "evaluation": {"required": list(source.fields), "unique_columns": list(source.unique_columns),
                                   "eval_pipeline": {name: {"preprocess": ["norm_str"], "metric": ["exact_match"]}
                                                     for name in source.fields}},
                },
            ))
    return tasks


def verify_curriculum_rows(task: Task, source_text: str) -> None:
    """Reject stale or mixed-row teachers using freshly parsed source records."""
    if task.metadata.get("table"):
        source = next((source for source in PUBLIC_TABLES if source.key == task.metadata["table"]), None)
        if source is None or source.url != task.metadata.get("source_url") or source.split != task.split:
            raise ValueError("Unrecognized independent source provenance")
        candidates = public_table_tasks(source, source_text)
    else:
        source = next((source for source in REGISTRIES if source.key == task.metadata.get("registry")), None)
        if source is None:
            # A custom registry supplied by a caller still has field-level
            # evidence validation; shipped curricula always take the strict
            # source-row validation branch above or below.
            return
        if source.url != task.metadata.get("source_url") or source.split != task.split:
            raise ValueError("Unrecognized independent registry provenance")
        candidates = registry_tasks(source, source_text)
    match = next((candidate for candidate in candidates if candidate.id == task.id), None)
    if match is None or match.gold != task.gold or match.prompt != task.prompt:
        raise ValueError("Curriculum gold is no longer supported by complete live source rows")

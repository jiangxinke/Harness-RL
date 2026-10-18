#!/usr/bin/env python3
"""Check DCC runtime dependencies and tools on CPU before loading a model.

No network requests, model downloads, CUDA allocations or artifact writes occur.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import io
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "slime"))


def requirements(path):
    for line in path.read_text().splitlines():
        line = line.partition("#")[0].strip()
        if line.startswith("-r "):
            yield from requirements(path.parent / line[3:].strip())
        elif line:
            yield line


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=("harness", "bf16", "nf4"), default="bf16")
    parser.add_argument("--include-tests", action="store_true", help="Also require the declared pytest dependencies")
    args = parser.parse_args()
    # This process is intentionally CPU-only, even when the parent owns GPUs.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    filename = {"harness": "requirements.txt", "bf16": "requirements-training.txt", "nf4": "requirements-quantized.txt"}[args.runtime]
    declaration = ROOT / "configs" / "dcc" / filename
    report = {"runtime": args.runtime, "cuda_visible": False, "dependencies": {}, "checks": {}, "errors": []}
    try:
        from packaging.requirements import Requirement
    except ImportError:
        report["errors"].append("Version checks require packaging; install packaging before running preflight")
        print(json.dumps(report, indent=2))
        return 1
    for line in requirements(declaration):
        requirement = Requirement(line)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        if requirement.name.startswith("pytest") and not args.include_tests:
            continue
        try:
            version = importlib.metadata.version(requirement.name)
            report["dependencies"][requirement.name] = version
            if not requirement.specifier.contains(version, prereleases=True):
                report["errors"].append(f"{requirement.name}=={version} does not satisfy {requirement}")
            module = requirement.name.replace("-", "_")
            importlib.import_module(module)
        except Exception as exc:
            report["errors"].append(f"{requirement.name}: {type(exc).__name__}: {exc}")
    if not report["errors"]:
        try:
            from pypdf import PdfWriter
            from slime_plugins.dcc.metrics import evaluate_cuad
            from slime_plugins.dcc.tools import extract_page
            from slime_plugins.dcc.widesearch_metrics import evaluate_widesearch

            text = "Example contract page. The agreement is governed by California law."
            html = f"<html><body><article><p>{text}</p></article></body></html>".encode()
            if "California" not in extract_page(html, "text/html"):
                raise RuntimeError("HTML extraction lost source text")
            report["checks"]["html_extraction"] = True
            writer, buffer = PdfWriter(), io.BytesIO()
            writer.add_blank_page(width=72, height=72)
            writer.write(buffer)
            if extract_page(buffer.getvalue(), "application/pdf") != "":
                raise RuntimeError("Blank PDF extraction returned unexpected text")
            report["checks"]["pdf_extraction"] = True
            cuad = evaluate_cuad({"example": ["California law"]}, {"example": ["California law"]})
            if cuad["micro_f1"] != 1:
                raise RuntimeError("CUAD scorer failed its exact-span check")
            report["checks"]["cuad_scoring"] = True
            config = {
                "required": ["name", "date"],
                "unique_columns": ["name"],
                "eval_pipeline": {
                    "name": {"preprocess": ["norm_str"], "metric": ["exact_match"]},
                    "date": {"preprocess": ["norm_date"], "metric": ["date_near"]},
                },
            }
            rows = [{"name": "example", "date": "2020-01-15"}]
            wide = evaluate_widesearch(rows, rows, config)
            if wide["f1_by_row"] != 1 or wide["f1_by_item"] != 1:
                raise RuntimeError("WideSearch scorer failed its date-aware check")
            report["checks"]["widesearch_date_scoring"] = True
            if args.runtime != "harness":
                from transformers import Qwen3MoeForCausalLM

                if Qwen3MoeForCausalLM.config_class.model_type != "qwen3_moe":
                    raise RuntimeError("Qwen3 MoE model implementation is unavailable")
                report["checks"]["qwen3_moe_import"] = True
            if args.runtime == "nf4":
                from slime_plugins.dcc.hf_quantization import nf4_load_kwargs

                nf4_load_kwargs()
                report["checks"]["nf4_configuration"] = True
        except Exception as exc:
            report["errors"].append(f"Runtime smoke: {type(exc).__name__}: {exc}")
    report["ok"] = not report["errors"]
    print(json.dumps(report, indent=2))
    return int(not report["ok"])


if __name__ == "__main__":
    raise SystemExit(main())

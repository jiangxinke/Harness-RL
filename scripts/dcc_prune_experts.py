#!/usr/bin/env python3
"""Export a genuinely smaller Qwen3 MoE using training-only router statistics.

``plan`` reads only checkpoint headers; ``export`` removes expert tensors and
router rows and publishes a new Hugging Face checkpoint atomically. This never
uses masking or quantization as a substitute for reducing logical parameters.
Calibration JSON may be produced with :class:`RouterCounter` or merged with
``merge``. Benchmark test data must never be used to select experts.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import json
import math
from pathlib import Path
import re
import shutil
import struct
import sys
import time
import uuid

_SCHEMA = 1
_CONFIG_FIELDS = ("num_hidden_layers", "num_experts", "num_experts_per_tok", "hidden_size", "moe_intermediate_size")
_EXPERT = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
_FUSED = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$")
_GATE = re.compile(r"^model\.layers\.(\d+)\.mlp\.gate\.weight$")
_DTYPES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _positive_int(value, name, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < (0 if zero else 1):
        raise ValueError(f"{name} must be a {'nonnegative' if zero else 'positive'} integer")
    return value


def normalize_config(config):
    """Accept the HF 4.x num_experts and 5.x num_local_experts alias."""
    config = copy.deepcopy(config)
    if "num_local_experts" in config:
        if "num_experts" in config and config["num_experts"] != config["num_local_experts"]:
            raise ValueError("Conflicting num_experts/num_local_experts configuration")
        config["num_experts"] = config["num_local_experts"]
    return config


def sparse_layers(config):
    config = normalize_config(config)
    if config.get("model_type") != "qwen3_moe":
        raise ValueError("Only Qwen3 MoE checkpoints are supported")
    for field in _CONFIG_FIELDS:
        _positive_int(config.get(field), field)
    if config["num_experts_per_tok"] > config["num_experts"]:
        raise ValueError("num_experts_per_tok exceeds num_experts")
    step = _positive_int(config.get("decoder_sparse_step", 1), "decoder_sparse_step")
    dense = set(config.get("mlp_only_layers", []))
    return [i for i in range(config["num_hidden_layers"]) if i not in dense and (i + 1) % step == 0]


def checkpoint_metadata(source):
    """Read safetensors headers without mapping or loading large tensor data."""
    source = Path(source)
    config = normalize_config(read_json(source / "config.json"))
    sparse_layers(config)
    index_path = source / "model.safetensors.index.json"
    index = read_json(index_path) if index_path.exists() else None
    files = sorted(set(index["weight_map"].values())) if index else ["model.safetensors"]
    result = {}
    for filename in files:
        if Path(filename).name != filename or not filename.endswith(".safetensors"):
            raise ValueError("Checkpoint index must reference local safetensors basenames")
        path = source / filename
        with path.open("rb") as stream:
            size_bytes = stream.read(8)
            if len(size_bytes) != 8:
                raise ValueError(f"Incomplete safetensors header: {filename}")
            header_length = struct.unpack("<Q", size_bytes)[0]
            if header_length > 64 * 1024 * 1024:
                raise ValueError("Unreasonably large safetensors header")
            header = json.loads(stream.read(header_length))
        data_size = path.stat().st_size - 8 - header_length
        intervals = []
        for name, item in header.items():
            if name == "__metadata__":
                continue
            if name in result:
                raise ValueError(f"Duplicate checkpoint tensor: {name}")
            if item["dtype"] not in _DTYPES:
                raise ValueError(f"Expected floating, unquantized parameters: {name}")
            shape = item["shape"]
            if not shape or any(isinstance(d, bool) or not isinstance(d, int) or d < 1 for d in shape):
                raise ValueError(f"Invalid parameter shape: {name}")
            start, end = item["data_offsets"]
            if start < 0 or end - start != math.prod(shape) * _DTYPES[item["dtype"]] or end > data_size:
                raise ValueError(f"Invalid tensor data offsets: {name}")
            intervals.append((start, end))
            if index is not None and index["weight_map"].get(name) != filename:
                raise ValueError(f"Checkpoint index disagrees with shard: {name}")
            result[name] = {"shape": shape, "dtype": item["dtype"], "file": filename}
        intervals.sort()
        if not intervals or intervals[0][0] != 0 or intervals[-1][1] != data_size or any(a[1] != b[0] for a, b in zip(intervals, intervals[1:])):
            raise ValueError(f"Incomplete or overlapping safetensors shard: {filename}")
    if index is not None and set(result) != set(index["weight_map"]):
        raise ValueError("Checkpoint index references missing tensors")
    return config, result


def validate_calibration(calibration, config, *, source_model_id=None, source_revision=None):
    config = normalize_config(config)
    layers = sparse_layers(config)
    if calibration.get("schema_version") != _SCHEMA or calibration.get("kind") != "qwen3_moe_routing_calibration":
        raise ValueError("Unsupported routing calibration schema")
    if calibration.get("split") != "train":
        raise ValueError("Expert selection requires training-only calibration (split='train')")
    for key, expected in (("source_model_id", source_model_id), ("source_revision", source_revision)):
        if not isinstance(calibration.get(key), str) or not calibration[key]:
            raise ValueError(f"Calibration must bind {key}")
        if expected is not None and calibration[key] != expected:
            raise ValueError(f"Calibration {key} mismatch")
    if calibration.get("config") != {k: config[k] for k in _CONFIG_FIELDS}:
        raise ValueError("Calibration model dimensions do not match checkpoint")
    ids = calibration.get("task_ids")
    if not isinstance(ids, list) or not ids or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("Calibration requires unique, nonempty training task IDs")
    if set(calibration.get("layers", {})) != {str(i) for i in layers}:
        raise ValueError("Calibration must cover every sparse layer exactly once")
    for layer in layers:
        record = calibration["layers"][str(layer)]
        counts = record["counts"]
        if not isinstance(counts, list) or len(counts) != config["num_experts"]:
            raise ValueError(f"Expert count length mismatch in layer {layer}")
        for value in counts:
            _positive_int(value, "routing count", zero=True)
        tokens = _positive_int(record["total_tokens"], "total_tokens")
        if sum(counts) != tokens * config["num_experts_per_tok"]:
            raise ValueError(f"Routing counts do not sum to token top-k in layer {layer}")
    return calibration


def merge_calibrations(calibrations, config):
    if not calibrations:
        raise ValueError("At least one calibration is required")
    first = calibrations[0]
    result = copy.deepcopy(validate_calibration(first, config))
    for item in calibrations[1:]:
        validate_calibration(item, config, source_model_id=first["source_model_id"], source_revision=first["source_revision"])
        if set(result["task_ids"]) & set(item["task_ids"]):
            raise ValueError("Calibration shards contain overlapping task IDs")
        result["task_ids"].extend(item["task_ids"])
        for layer, record in result["layers"].items():
            other = item["layers"][layer]
            record["counts"] = [a + b for a, b in zip(record["counts"], other["counts"], strict=True)]
            record["total_tokens"] += other["total_tokens"]
    result["task_ids"].sort()
    return validate_calibration(result, config)


def parameter_count(metadata, config):
    """Count logical model parameters, counting tied embedding/head only once."""
    total = sum(math.prod(item["shape"]) for item in metadata.values())
    embed, head = "model.embed_tokens.weight", "lm_head.weight"
    if embed not in metadata:
        raise ValueError("Checkpoint lacks input embedding weights")
    if config.get("tie_word_embeddings", False):
        if head in metadata:
            if metadata[head]["shape"] != metadata[embed]["shape"]:
                raise ValueError("Tied embedding and head shapes differ")
            total -= math.prod(metadata[head]["shape"])
    elif head not in metadata:
        raise ValueError("Untied checkpoint lacks output head weights")
    if any("inv_freq" in name for name in metadata):
        raise ValueError("Cannot count persistent rotary buffers as logical parameters")
    return total


def pruning_plan(config, metadata, calibration, keep_experts, *, max_parameters=30_000_000_000, source_model_id=None, source_revision=None):
    config = normalize_config(config)
    validate_calibration(calibration, config, source_model_id=source_model_id, source_revision=source_revision)
    keep_experts = _positive_int(keep_experts, "keep_experts")
    if not config["num_experts_per_tok"] <= keep_experts < config["num_experts"]:
        raise ValueError("keep_experts must preserve top-k and remove at least one expert")
    _positive_int(max_parameters, "max_parameters")
    layers = sparse_layers(config)
    h, width, original = config["hidden_size"], config["moe_intermediate_size"], config["num_experts"]
    selections = {}
    layouts = {}
    expected_expert_keys = set()
    for layer in layers:
        prefix = f"model.layers.{layer}.mlp."
        gate = prefix + "gate.weight"
        if metadata.get(gate, {}).get("shape") != [original, h]:
            raise ValueError(f"Missing or invalid router matrix: {gate}")
        counts = calibration["layers"][str(layer)]["counts"]
        # Preserve original ordering after selecting by usage. Ties retain the
        # smaller original ID, so mappings are deterministic across machines.
        selected = sorted(sorted(range(original), key=lambda i: (-counts[i], i))[:keep_experts])
        selections[str(layer)] = selected
        fused = prefix + "experts.gate_up_proj"
        if fused in metadata:
            shapes = {fused: [original, 2 * width, h], prefix + "experts.down_proj": [original, h, width]}
            layouts[str(layer)] = "fused"
        else:
            shapes = {prefix + f"experts.{i}.{projection}.weight": shape for i in range(original) for projection, shape in (("gate_proj", [width, h]), ("up_proj", [width, h]), ("down_proj", [h, width]))}
            layouts[str(layer)] = "individual"
        for name, shape in shapes.items():
            if metadata.get(name, {}).get("shape") != shape:
                raise ValueError(f"Missing or invalid expert matrix: {name}")
        expected_expert_keys.update(shapes)
    actual_experts = {k for k in metadata if ".mlp.experts." in k}
    if actual_experts != expected_expert_keys:
        raise ValueError("Unrecognized, duplicated, or extra expert weights")
    if {int(m.group(1)) for name in metadata if (m := _GATE.fullmatch(name))} != set(layers):
        raise ValueError("Router matrices do not match configured sparse layers")
    projected = {}
    name_map = {}
    for name, item in metadata.items():
        item = copy.deepcopy(item)
        if match := _EXPERT.fullmatch(name):
            layer, expert = str(int(match[1])), int(match[2])
            if expert not in selections[layer]:
                continue
            new_id = selections[layer].index(expert)
            target = f"model.layers.{layer}.mlp.experts.{new_id}.{match[3]}.weight"
        else:
            target = name
            if _FUSED.fullmatch(name) or _GATE.fullmatch(name):
                item["shape"][0] = keep_experts
        if target in projected:
            raise ValueError(f"Duplicate output tensor: {target}")
        projected[target] = item
        name_map[name] = target
    before, after = parameter_count(metadata, config), parameter_count(projected, config)
    if after > max_parameters:
        raise ValueError(f"Pruned model has {after} logical parameters, exceeding limit {max_parameters}")
    expected_reduction = len(layers) * (original - keep_experts) * (3 * width * h + h)
    if before - after != expected_reduction:
        raise ValueError("Expert parameter reduction does not match exact shape accounting")
    return {
        "schema_version": _SCHEMA,
        "kind": "qwen3_moe_expert_pruning_plan",
        "source_model_id": calibration["source_model_id"],
        "source_revision": calibration["source_revision"],
        "source_logical_parameters": before,
        "logical_parameters": after,
        "removed_parameters": before - after,
        "max_parameters": max_parameters,
        "original_experts_per_layer": original,
        "experts_per_layer": keep_experts,
        "num_experts_per_tok": config["num_experts_per_tok"],
        "selected_experts": selections,
        "storage_layouts": layouts,
        "calibration_task_ids": calibration["task_ids"],
        "calibration_split": "train",
        "name_map": name_map,
        "output_metadata": projected,
    }


class RouterCounter:
    """Count actual top-k selections on unpadded training forward passes.

    Compatible with the Transformers 4.x Linear router and 5.x TopKRouter.
    Use ``with counter.valid_tokens(attention_mask): model(...)`` for batches
    with padding; otherwise every token passed to the model is counted.
    Hooks must be removed before generation, training, or exporting a model.
    """
    def __init__(self, model):
        import torch
        self.torch = torch
        self.config = normalize_config(model.config.to_dict())
        self.layers = sparse_layers(self.config)
        self.counts = {str(i): torch.zeros(self.config["num_experts"], dtype=torch.int64) for i in self.layers}
        self.total_tokens = {str(i): 0 for i in self.layers}
        self.handles = []
        self.mask = None
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        for layer in self.layers:
            gate = base.model.layers[layer].mlp.gate
            self.handles.append(gate.register_forward_hook(self._hook(str(layer))))

    @contextmanager
    def valid_tokens(self, attention_mask):
        if self.mask is not None:
            raise ValueError("Nested router masks are unsupported")
        self.mask = attention_mask.reshape(-1).bool()
        try:
            yield self
        finally:
            self.mask = None

    def _hook(self, layer):
        def record(module, inputs, output):
            torch = self.torch
            if isinstance(output, tuple):
                if len(output) != 3:
                    raise ValueError("Unknown Qwen router output format")
                if any(not torch.isfinite(value).all() for value in output[:2]):
                    raise ValueError("Nonfinite routing logits or weights")
                selected = output[2].detach()
            else:
                logits = output.detach().float()
                if not torch.isfinite(logits).all():
                    raise ValueError("Nonfinite routing logits")
                selected = torch.topk(torch.softmax(logits, dim=-1), self.config["num_experts_per_tok"], dim=-1).indices
            selected = selected.reshape(-1, self.config["num_experts_per_tok"])
            if self.mask is not None:
                if selected.shape[0] != self.mask.numel():
                    raise ValueError("Router token mask does not match forward pass")
                selected = selected[self.mask.to(selected.device)]
            if selected.numel() and ((selected < 0).any() or (selected >= self.config["num_experts"]).any()):
                raise ValueError("Router returned invalid expert indices")
            self.counts[layer] += torch.bincount(selected.reshape(-1).to("cpu"), minlength=self.config["num_experts"])
            self.total_tokens[layer] += selected.shape[0]
        return record

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def report(self, *, source_model_id, source_revision, task_ids):
        result = {
            "schema_version": _SCHEMA, "kind": "qwen3_moe_routing_calibration", "split": "train",
            "source_model_id": source_model_id, "source_revision": source_revision,
            "config": {k: self.config[k] for k in _CONFIG_FIELDS}, "task_ids": list(task_ids),
            "layers": {layer: {"counts": count.tolist(), "total_tokens": self.total_tokens[layer]} for layer, count in self.counts.items()},
        }
        return validate_calibration(result, self.config)


def export_checkpoint(source, destination, calibration, keep_experts, *, max_parameters=30_000_000_000, model_id=None, workers=1, source_model_id=None, source_revision=None):
    """Physically omit removed tensors; atomically publish after shape auditing."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    import torch
    _positive_int(workers, "workers")
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(f"Export destination already exists: {destination}")
    config, metadata = checkpoint_metadata(source)
    plan = pruning_plan(config, metadata, calibration, keep_experts, max_parameters=max_parameters, source_model_id=source_model_id, source_revision=source_revision)
    model_id = model_id or f"DCC-Qwen3MoE-{keep_experts}E-derivative"
    if not model_id or any(c in model_id for c in ("\n", "\r", "\\")) or model_id.startswith("/"):
        raise ValueError("Derivative model ID must be a public model name")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / (destination.name + ".partial-" + uuid.uuid4().hex)
    staging.mkdir()
    try:
        files = sorted({item["file"] for item in metadata.values()})
        def export_shard(filename):
            started = time.monotonic()
            tensors = {}
            shard_path = staging / filename
            temporary_shard = shard_path.with_name(shard_path.name + ".partial")
            with safe_open(source / filename, framework="pt", device="cpu") as stream:
                for name in stream.keys():
                    target = plan["name_map"].get(name)
                    if target is None:
                        continue
                    tensor = stream.get_tensor(name)
                    match = _GATE.fullmatch(name) or _FUSED.fullmatch(name)
                    if match:
                        tensor = tensor.index_select(0, torch.tensor(plan["selected_experts"][str(int(match[1]))])).contiguous()
                    tensors[target] = tensor
                if tensors:
                    save_file(tensors, temporary_shard, metadata={"format": "pt"})
                    temporary_shard.rename(shard_path)
            stored_bytes = sum(math.prod(output_meta_item["shape"]) * _DTYPES[output_meta_item["dtype"]]
                               for output_meta_item in plan["output_metadata"].values() if output_meta_item["file"] == filename)
            print(json.dumps({"event": "shard_exported", "shard": filename, "tensor_data_bytes": stored_bytes,
                              "tensors": len(tensors), "elapsed_seconds": round(time.monotonic() - started, 3)}),
                  file=sys.stderr, flush=True)
            return filename if tensors else None
        with ThreadPoolExecutor(max_workers=workers) as pool:
            exported_files = list(pool.map(export_shard, files))
        output_meta = plan["output_metadata"]
        weight_map = {key: value["file"] for key, value in output_meta.items()}
        if len([name for name in exported_files if name]) > 1 or (staging / "model.safetensors").exists() is False:
            write_json(staging / "model.safetensors.index.json", {"metadata": {"total_size": sum(math.prod(v["shape"]) * _DTYPES[v["dtype"]] for v in output_meta.values())}, "weight_map": weight_map})
        new_config = copy.deepcopy(config)
        new_config.update(num_experts=keep_experts, _name_or_path=model_id)
        if "num_local_experts" in new_config:
            new_config["num_local_experts"] = keep_experts
        manifest = {key: value for key, value in plan.items() if key not in {"name_map", "output_metadata"}}
        manifest.update(derivative_model_id=model_id, selection_rule="least-used training router top-k counts; stable original-ID tie break", quantized=False, physically_removed_experts=True)
        new_config["dcc_expert_pruning"] = {key: manifest[key] for key in ("source_model_id", "source_revision", "derivative_model_id", "source_logical_parameters", "logical_parameters", "original_experts_per_layer", "experts_per_layer", "calibration_split")}
        write_json(staging / "config.json", new_config)
        write_json(staging / "expert-pruning.json", manifest)
        write_json(staging / "routing-calibration.json", calibration)
        for asset in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt", "generation_config.json", "LICENSE", "LICENSE.txt"):
            if (source / asset).is_file():
                shutil.copyfile(source / asset, staging / asset)
        (staging / "README.md").write_text(f"# {model_id}\n\nDerivative of {manifest['source_model_id']} at revision {manifest['source_revision']}.\n\nThis checkpoint physically removes experts selected using training-only routing counts.\nIt is a distinct pruned model, not the unchanged upstream model.\n\nLogical base parameters: {manifest['logical_parameters']:,}; experts per sparse layer: {keep_experts}; top-k: {config['num_experts_per_tok']}.\nCalibration is declared training-only; the caller must verify the provided task IDs belong to the training split.\nSee expert-pruning.json for all per-layer original-to-retained expert selections and routing-calibration.json for declared training provenance.\n")
        observed_config, observed = checkpoint_metadata(staging)
        if set(observed) != set(output_meta) or any(observed[k]["shape"] != output_meta[k]["shape"] or observed[k]["dtype"] != output_meta[k]["dtype"] for k in observed):
            raise RuntimeError("Exported checkpoint metadata does not match pruning plan")
        if parameter_count(observed, observed_config) != plan["logical_parameters"]:
            raise RuntimeError("Exported logical parameter count does not match plan")
        staging.rename(destination)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "export"):
        sub = commands.add_parser(command)
        sub.add_argument("--source", required=True)
        sub.add_argument("--calibration", required=True)
        sub.add_argument("--keep-experts", type=int, default=125)
        sub.add_argument("--max-parameters", type=int, default=30_000_000_000)
        sub.add_argument("--source-model-id")
        sub.add_argument("--source-revision")
        if command == "export":
            sub.add_argument("--output", required=True)
            sub.add_argument("--model-id")
            sub.add_argument("--workers", type=int, default=1)
    merge = commands.add_parser("merge")
    merge.add_argument("--source", required=True)
    merge.add_argument("--calibration", nargs="+", required=True)
    merge.add_argument("--output", required=True)
    args = parser.parse_args()
    config = normalize_config(read_json(Path(args.source) / "config.json"))
    if args.command == "merge":
        result = merge_calibrations([read_json(p) for p in args.calibration], config)
        write_json(args.output, result)
    else:
        calibration = read_json(args.calibration)
        kwargs = {"max_parameters": args.max_parameters, "source_model_id": args.source_model_id, "source_revision": args.source_revision}
        if args.command == "export":
            result = export_checkpoint(args.source, args.output, calibration, args.keep_experts, model_id=args.model_id, workers=args.workers, **kwargs)
        else:
            config, metadata = checkpoint_metadata(args.source)
            result = pruning_plan(config, metadata, calibration, args.keep_experts, **kwargs)
            result = {key: value for key, value in result.items() if key not in {"name_map", "output_metadata"}}
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

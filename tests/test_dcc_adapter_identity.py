"""CPU-only original persisted identity versus actual tiny LoRA headers."""
from copy import deepcopy
from pathlib import Path
import json
import os
import runpy
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from slime_plugins.dcc import adapter_identity as identity_module
from slime_plugins.dcc.adapter_identity import adapter_content_identity, saved_adapter_matches
from slime_plugins.dcc.vllm_policy import _adapter_metadata

EVAL = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'scripts/dcc_model_evaluate.py'))


def adapter(tmp_path, value=1):
    path = tmp_path / 'adapter'; path.mkdir(exist_ok=True)
    (path / 'adapter_config.json').write_text(json.dumps({'peft_type': 'LORA', 'r': 2, 'target_modules': ['q_proj']}))
    save_file({'base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight': torch.full((2, 4), float(value)),
               'base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight': torch.ones(4, 2)},
              str(path / 'adapter_model.safetensors'))
    return path


def test_original_cli_content_identity_and_legacy_loader_metadata_are_both_checked(tmp_path):
    path = adapter(tmp_path); metadata = _adapter_metadata(path)
    saved = EVAL['_adapter_identity'](path)
    assert 'weights_sha256' in saved and 'weights_mtime_ns' not in saved
    assert saved_adapter_matches(saved, metadata)
    assert saved_adapter_matches(metadata, metadata)
    for key, value in [('weights_sha256', '0' * 64), ('weights_bytes', saved['weights_bytes'] + 1),
                       ('path', str(tmp_path / 'other')), ('unverified', True)]:
        wrong = deepcopy(saved); wrong[key] = value
        assert not saved_adapter_matches(wrong, metadata)


def test_same_size_same_shape_changed_weights_cannot_reuse_cached_content_identity(tmp_path):
    path = adapter(tmp_path); saved = EVAL['_adapter_identity'](path)
    assert saved_adapter_matches(saved, _adapter_metadata(path))
    adapter(tmp_path, value=0)
    actual = _adapter_metadata(path)
    assert saved['weights_bytes'] == actual['weights_bytes']
    assert not saved_adapter_matches(saved, actual)


def test_changed_file_during_header_and_content_validation_fails_closed(tmp_path):
    path = adapter(tmp_path); saved = EVAL['_adapter_identity'](path); metadata = _adapter_metadata(path)
    with (path / 'adapter_model.safetensors').open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ValueError, match='changed between header'):
        saved_adapter_matches(saved, metadata)


def test_writer_and_native_comparison_share_one_stable_file_digest(monkeypatch, tmp_path):
    path = adapter(tmp_path); calls = []
    original = identity_module.hashlib.sha256
    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(identity_module.hashlib, 'sha256', counted)
    saved = EVAL['_adapter_identity'](path)
    assert saved == adapter_content_identity(path)
    assert saved_adapter_matches(saved, _adapter_metadata(path))
    assert saved_adapter_matches(saved, _adapter_metadata(path))
    assert len(calls) == 1
    assert set(saved) == {'path', 'config', 'weights_file', 'weights_bytes', 'weights_sha256'}


def test_same_size_change_with_restored_mtime_invalidates_shared_cache(tmp_path):
    path = adapter(tmp_path); saved = EVAL['_adapter_identity'](path)
    weights = path / 'adapter_model.safetensors'; before = weights.stat()
    adapter(tmp_path, value=0)
    os.utime(weights, ns=(before.st_atime_ns, before.st_mtime_ns))
    changed = adapter_content_identity(path)
    assert weights.stat().st_size == before.st_size
    assert weights.stat().st_mtime_ns == before.st_mtime_ns
    assert changed['weights_sha256'] != saved['weights_sha256']
    assert not saved_adapter_matches(saved, _adapter_metadata(path))


def test_cached_digest_rechecks_final_file_fingerprint(monkeypatch, tmp_path):
    path = adapter(tmp_path); adapter_content_identity(path)
    weights = path / 'adapter_model.safetensors'; actual = weights.stat(); calls = []
    original = Path.stat
    def changed_stat(value, *args, **kwargs):
        if value == weights:
            calls.append(True)
            if len(calls) > 1:
                return SimpleNamespace(st_dev=actual.st_dev, st_ino=actual.st_ino,
                    st_ctime_ns=actual.st_ctime_ns + 1, st_mtime_ns=actual.st_mtime_ns, st_size=actual.st_size)
        return original(value, *args, **kwargs)
    monkeypatch.setattr(Path, 'stat', changed_stat)
    with pytest.raises(ValueError, match='changed while computing content identity'):
        EVAL['_adapter_identity'](path)


def test_mutation_during_hash_is_rejected_without_caching_failed_digest(monkeypatch, tmp_path):
    path = adapter(tmp_path); weights = path / 'adapter_model.safetensors'
    before = weights.stat()
    original = Path.open; changed = []
    class MutatingReader:
        def __init__(self, stream): self.stream = stream
        def __enter__(self): self.stream.__enter__(); return self
        def __exit__(self, *args): return self.stream.__exit__(*args)
        def fileno(self): return self.stream.fileno()
        def read(self, size):
            data = self.stream.read(size)
            if data and not changed:
                changed.append(True)
                with original(weights, 'r+b') as writer:
                    writer.seek(-1, os.SEEK_END); writer.write(b'\x01')
                # Some local filesystems coalesce timestamp updates within a tick.
                os.utime(weights, ns=(before.st_atime_ns, before.st_mtime_ns + 1))
            return data
    def open_file(value, mode='r', *args, **kwargs):
        stream = original(value, mode, *args, **kwargs)
        return MutatingReader(stream) if value == weights and mode == 'rb' else stream
    monkeypatch.setattr(Path, 'open', open_file)
    with pytest.raises(ValueError, match='changed while computing content identity'):
        adapter_content_identity(path)
    assert not any(key[0] == str(weights.resolve()) for key in identity_module._DIGEST_CACHE)
    monkeypatch.setattr(Path, 'open', original)
    assert adapter_content_identity(path)['weights_sha256'] == identity_module.hashlib.sha256(weights.read_bytes()).hexdigest()


def native_adapter(tmp_path):
    """Tiny real headers plus valid synthetic source declarations; no engine."""
    from slime_plugins.dcc.native_budget import approved_training_receipt
    from slime_plugins.dcc.native_lora import validate_native_adapter
    from slime_plugins.dcc.native_thinking_eval import BASE_SOURCE_COMMIT, MODEL_REVISION, PROTOCOL

    path = adapter(tmp_path)
    evaluation = {"version": "dcc-native-inference-source-v1", "source_commit": "f" * 40,
                  "base_source_commit": BASE_SOURCE_COMMIT, "protocol": PROTOCOL, "frozen": True}
    training = approved_training_receipt(evaluation, source_commit="e" * 40, protocol=PROTOCOL)
    native = {"enabled": True, "protocol": PROTOCOL, "source_receipt": training,
              "derivative_id": "synthetic/native-attention-r2", "model_revision": MODEL_REVISION,
              "all_files_validated_before_model_load": True, "warm_start": False,
              "raw_cuad_auxiliary": False, "validated_records": 1, "validated_cuad_records": 0}
    (path / "training_config.json").write_text(json.dumps({"steps": 1, "lora_rank": 2,
                                                         "native_thinking": native}))
    kwargs = {"source_receipt": evaluation, "approved_derivative_id": native["derivative_id"],
              "model_config": {"hidden_size": 4, "num_hidden_layers": 1, "num_attention_heads": 2,
                               "num_key_value_heads": 1, "head_dim": 2},
              "approved_training_source_receipt": training, "approved_training_protocol": PROTOCOL,
              "evaluation_protocol": PROTOCOL}
    return path, validate_native_adapter, kwargs


def test_content_identity_keeps_complete_actual_native_binding(tmp_path):
    path, validate, kwargs = native_adapter(tmp_path)
    content = EVAL['_adapter_identity'](path)
    metadata, binding = validate(path, **kwargs)
    assert set(metadata) == {'path', 'config', 'weights_file', 'weights_bytes', 'weights_mtime_ns'}
    assert saved_adapter_matches(content, metadata)
    assert binding == {"derivative_id": kwargs['approved_derivative_id'],
                       "training_source_receipt": kwargs['approved_training_source_receipt'],
                       "protocol": kwargs['approved_training_protocol'],
                       "model_revision": json.loads((path / 'training_config.json').read_text())['native_thinking']['model_revision'],
                       "lora_rank": 2, "tensor_pairs": 1, "checked_actual_attention_shapes": True,
                       "evaluation_source_receipt": kwargs['source_receipt'],
                       "evaluation_protocol": kwargs['evaluation_protocol'],
                       "caller_approved_training": {"source_commit": 'e' * 40,
                                                    "protocol": kwargs['approved_training_protocol']}}


@pytest.mark.parametrize('damage', ['source', 'protocol', 'derivative', 'rank', 'steps',
                                    'config_schema', 'shape', 'dtype', 'missing_pair', 'extra_tensor'])
def test_content_identity_never_exempts_actual_native_training_or_header_guards(tmp_path, damage):
    path, validate, kwargs = native_adapter(tmp_path)
    if damage == 'config_schema':
        config = json.loads((path / 'adapter_config.json').read_text())
        config['use_dora'] = True
        (path / 'adapter_config.json').write_text(json.dumps(config))
    elif damage in {'shape', 'dtype', 'missing_pair', 'extra_tensor'}:
        prefix = 'base_model.model.model.layers.0.self_attn.q_proj.lora_'
        tensors = {prefix + 'A.weight': torch.zeros(2, 3 if damage == 'shape' else 4),
                   prefix + 'B.weight': torch.ones(4, 2)}
        if damage == 'dtype': tensors[prefix + 'A.weight'] = torch.zeros(2, 4, dtype=torch.int32)
        if damage == 'missing_pair': tensors.pop(prefix + 'B.weight')
        if damage == 'extra_tensor': tensors['unverified.weight'] = torch.zeros(1)
        save_file(tensors, str(path / 'adapter_model.safetensors'))
    else:
        config = json.loads((path / 'training_config.json').read_text())
        if damage == 'source': config['native_thinking']['source_receipt']['source_commit'] = 'd' * 40
        elif damage == 'protocol': config['native_thinking']['protocol'] = 'unverified'
        elif damage == 'derivative': config['native_thinking']['derivative_id'] = 'unverified'
        elif damage == 'rank': config['lora_rank'] = 3
        elif damage == 'steps': config['steps'] = 0
        (path / 'training_config.json').write_text(json.dumps(config))
    # Even a freshly generated, matching SHA identity cannot bypass validation.
    content = EVAL['_adapter_identity'](path)
    assert len(content['weights_sha256']) == 64
    with pytest.raises(ValueError):
        validate(path, **kwargs)

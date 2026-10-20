"""CPU-only original persisted identity versus actual tiny LoRA headers."""
from copy import deepcopy
from pathlib import Path
import json
import runpy

import pytest
import torch
from safetensors.torch import save_file

from slime_plugins.dcc.adapter_identity import saved_adapter_matches
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

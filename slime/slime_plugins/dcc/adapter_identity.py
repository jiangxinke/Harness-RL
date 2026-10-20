"""Shared persisted content identity and original loader metadata comparison.

The vLLM loader records modification time; dcc_model_evaluate persists a SHA256
content identity instead. Both declarations must match the actual local file.
This module loads no model or adapter tensors. Digests are cached only while
all local file facts remain unchanged, like the original evaluation writer.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

_DIGEST_CACHE = {}


def _key(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(',', ':'))


def _fingerprint(stat):
    return (stat.st_dev, stat.st_ino, stat.st_ctime_ns, stat.st_mtime_ns, stat.st_size)


def adapter_content_identity(adapter):
    """Identify checkpoint content; local file facts only invalidate the cache.

    This is the exact persisted evaluator schema. Identical copied bytes keep
    their identity at the same caller path even when inode/timestamps change.
    One stable weight file is hashed once per process across writer and audit.
    """
    if adapter is None:
        return None
    path = Path(adapter).absolute()
    config = json.loads((path / 'adapter_config.json').read_text())
    weights = path / 'adapter_model.safetensors'
    before = weights.stat()
    cache_key = (str(weights.resolve()), *_fingerprint(before))
    digest = _DIGEST_CACHE.get(cache_key)
    if digest is None:
        hasher = hashlib.sha256()
        with weights.open('rb') as stream:
            if _fingerprint(os.fstat(stream.fileno())) != _fingerprint(before):
                raise ValueError('Adapter checkpoint changed while computing content identity')
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                hasher.update(block)
            if _fingerprint(os.fstat(stream.fileno())) != _fingerprint(before):
                raise ValueError('Adapter checkpoint changed while computing content identity')
        digest = hasher.hexdigest()
    # Recheck path identity even on cache hits; a cached digest does not close
    # a concurrent replacement between the initial stat and returning identity.
    if _fingerprint(weights.stat()) != _fingerprint(before):
        raise ValueError('Adapter checkpoint changed while computing content identity')
    _DIGEST_CACHE[cache_key] = digest
    return {'path': str(path), 'config': config, 'weights_file': weights.name,
            'weights_bytes': before.st_size, 'weights_sha256': digest}


def saved_adapter_matches(saved, actual_loader_metadata):
    """Accept exactly original loader metadata or actual saved content identity.

    The latter is the output of dcc_model_evaluate._adapter_identity. Extra
    fields, changed config/path/size/digest, and a changed file fail closed.
    Existing header-only callers remain supported without new hashing.
    """
    metadata = actual_loader_metadata
    if not isinstance(metadata, dict) or not isinstance(saved, dict):
        return False
    if _key(saved) == _key(metadata):
        return True
    expected_keys = (set(metadata) - {'weights_mtime_ns'}) | {'weights_sha256'}
    if set(saved) != expected_keys:
        return False
    common = {key: value for key, value in metadata.items() if key != 'weights_mtime_ns'}
    if _key({key: saved[key] for key in common}) != _key(common):
        return False
    weights = Path(metadata['path']) / metadata['weights_file']
    before = weights.stat()
    if (before.st_size != metadata['weights_bytes']
            or before.st_mtime_ns != metadata['weights_mtime_ns']):
        raise ValueError('Adapter changed between header validation and content identity validation')
    content = adapter_content_identity(metadata['path'])
    if _fingerprint(weights.stat()) != _fingerprint(before):
        raise ValueError('Adapter changed after checking its persisted content identity')
    return _key(saved) == _key(content)

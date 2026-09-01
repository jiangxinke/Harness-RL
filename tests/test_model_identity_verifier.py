from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.verify_hf_model_identity import verify

ROOT = Path(__file__).resolve().parent.parent


def _identity(path: Path, shard: Path) -> Path:
    digest = hashlib.sha256(shard.read_bytes()).hexdigest()
    line = f"{shard.name}\t{shard.stat().st_size}\t{digest}\n"
    payload = {
        "schema_version": 1,
        "model_id": "fixture",
        "base_checkpoint_sha256": hashlib.sha256(line.encode()).hexdigest(),
        "files": [
            {
                "relative_path": shard.name,
                "byte_size": shard.stat().st_size,
                "sha256": digest,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_verify_hf_model_identity_accepts_the_exact_locked_shard(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "model.safetensors"
    shard.write_bytes(b"locked-weights")
    report = verify(tmp_path, _identity(tmp_path / "identity.json", shard))
    assert report["verified"] is True


def test_verify_hf_model_identity_rejects_changed_weights(tmp_path: Path) -> None:
    shard = tmp_path / "model.safetensors"
    shard.write_bytes(b"locked-weights")
    identity = _identity(tmp_path / "identity.json", shard)
    shard.write_bytes(b"changed-weights")
    with pytest.raises(ValueError, match="checkpoint shard mismatch"):
        verify(tmp_path, identity)

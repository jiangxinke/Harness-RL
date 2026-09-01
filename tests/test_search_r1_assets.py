from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.verify_search_r1_assets import verify_assets


def _manifest(path: Path, payload: bytes) -> Path:
    manifest = path / "asset_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": {
                    "index.bin": {
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest


def test_search_r1_asset_verifier_accepts_exact_identity(tmp_path: Path) -> None:
    payload = b"locked-search-index"
    (tmp_path / "index.bin").write_bytes(payload)

    verified = verify_assets(tmp_path, _manifest(tmp_path, payload))

    assert verified == [
        {
            "path": str(tmp_path / "index.bin"),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    ]


@pytest.mark.parametrize("replacement", [b"short", b"changed-search-index"])
def test_search_r1_asset_verifier_rejects_changed_file(
    tmp_path: Path, replacement: bytes
) -> None:
    payload = b"locked-search-index"
    manifest = _manifest(tmp_path, payload)
    (tmp_path / "index.bin").write_bytes(replacement)

    with pytest.raises(ValueError, match="size mismatch|SHA-256 mismatch"):
        verify_assets(tmp_path, manifest)

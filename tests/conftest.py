from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
for source in (
    ROOT / "slime",
    ROOT / "StackPlanner2.0/backend/packages/harness",
):
    value = str(source)
    if value not in sys.path:
        sys.path.insert(0, value)

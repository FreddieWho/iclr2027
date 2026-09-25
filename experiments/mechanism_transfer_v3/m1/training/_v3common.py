"""Deterministic import of the v3 shared package.

The repository has a different module named ``common``
(``experiments/discovery_campaign/common.py``) and several legacy agents insert
that directory on ``sys.path``. Resolving the v3 package by file location makes
the import independent of ``sys.path`` order, so a training run can never
silently consume the legacy feature helpers instead of the v3 contracts.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

V3_ROOT = Path(__file__).resolve().parents[2]
COMMON_DIR = V3_ROOT / "common"


def load_common():
    """Load ``mechanism_transfer_v3/common`` under the name ``common``."""
    module = sys.modules.get("common")
    if module is not None and getattr(module, "__file__", None):
        if Path(module.__file__).resolve().parent == COMMON_DIR:
            return module
    if not (COMMON_DIR / "__init__.py").exists():
        raise ImportError(f"v3 shared package missing at {COMMON_DIR}")
    spec = importlib.util.spec_from_file_location(
        "common",
        COMMON_DIR / "__init__.py",
        submodule_search_locations=[str(COMMON_DIR)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["common"] = module
    spec.loader.exec_module(module)
    return module


common = load_common()
bank = common.bank
geom_features = common.geom_features
metrics = common.metrics
source_typed = common.source_typed
visual_pool = common.visual_pool

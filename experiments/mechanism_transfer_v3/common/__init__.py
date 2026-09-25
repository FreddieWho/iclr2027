"""Shared fail-closed primitives for mechanism-transfer v3.

Only fixed contracts live here. Legacy defective formulas are transcribed in
tests as negative controls; they are not reusable model/feature definitions.
"""

from . import bank, geom_features, metrics, source_typed, visual_pool

__all__ = ["bank", "geom_features", "metrics", "source_typed", "visual_pool"]

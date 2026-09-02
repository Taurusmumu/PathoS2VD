"""Faithful PathoS2VD evaluation utilities.

All generated 33-plane stacks are evaluated through their central z11--z21
block.  A legacy z00--z10 layout is available only by explicit configuration.
"""

from .io import GENERATED_LAYOUTS, load_generated_stack, load_gt_stack
from .structure_tensor import evaluate_st_pair, vz_descriptor

__all__ = [
    "GENERATED_LAYOUTS",
    "evaluate_st_pair",
    "load_generated_stack",
    "load_gt_stack",
    "vz_descriptor",
]

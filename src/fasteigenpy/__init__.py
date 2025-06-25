"""
Copyright (c) 2025 Simon Rothman. All rights reserved.

fasteigenpy: Python bindings for Eigen, accelerated as much as possible with compiler optimizations, blas/lapacke, etc
"""

from __future__ import annotations

from ._version import version as __version__

from ._core import CompleteOrthogonalDecomposition, LLT, LDLT
from ._core import CompleteOrthogonalDecompositionRowMajor, LLTRowMajor, LDLTRowMajor
from ._core import ComputationInfo

__all__ = ["__version__", 
           "CompleteOrthogonalDecomposition", "LLT", "LDLT",
           "CompleteOrthogonalDecompositionRowMajor", "LLTRowMajor", "LDLTRowMajor",
           "ComputationInfo"]

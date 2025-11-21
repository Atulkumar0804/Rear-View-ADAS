"""
Warnings module
---------------
TTC calculation and collision warning logic.
"""

from .ttc_calculation import compute_ttc, classify_ttc
from .warning_logic import WarningSystem

__all__ = ["compute_ttc", "classify_ttc", "WarningSystem"]

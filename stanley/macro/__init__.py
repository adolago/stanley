"""
Macro Module

Macroeconomic data analysis using DBnomics and other sources.
Provides access to GDP, inflation, employment, interest rates,
and other economic indicators from global statistical institutions.
"""

from .dbnomics_adapter import DBnomicsAdapter
from .indicators import (
    EconomicIndicator,
    IndicatorCategory,
    INDICATOR_REGISTRY,
)
from .macro_analyzer import MacroAnalyzer

__all__ = [
    "DBnomicsAdapter",
    "EconomicIndicator",
    "IndicatorCategory",
    "INDICATOR_REGISTRY",
    "MacroAnalyzer",
]

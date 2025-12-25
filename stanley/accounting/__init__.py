"""
Accounting Module

SEC filings analysis, financial statement parsing, and footnote extraction
using edgartools for comprehensive fundamental analysis.
"""

from .edgar_adapter import EdgarAdapter
from .financial_statements import FinancialStatements
from .footnotes import FootnoteAnalyzer
from .accounting_analyzer import AccountingAnalyzer

__all__ = [
    "EdgarAdapter",
    "FinancialStatements",
    "FootnoteAnalyzer",
    "AccountingAnalyzer",
]

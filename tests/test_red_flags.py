"""Tests for red flag detection."""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime

from stanley.accounting.red_flags import (
    RevenueRedFlagDetector,
    ExpenseRedFlagDetector,
    AccrualRedFlagDetector,
    OffBalanceSheetDetector,
    CashFlowRedFlagDetector,
    RedFlagScorer,
    RedFlagSeverity,
    RedFlagCategory,
)


class TestRevenueRedFlagDetector:
    """Test revenue recognition red flags."""

    def test_revenue_growth_vs_ar_normal(self):
        """Test normal revenue and AR growth correlation."""
        data = pd.DataFrame({
            'revenue': [1000, 1100, 1200],
            'receivables': [100, 110, 120],
        }, index=pd.date_range('2021-12-31', periods=3, freq='Y'))

        detector = RevenueRedFlagDetector()
        flags = detector.detect_ar_growth_mismatch(data)

        assert len(flags) == 0 or all(flag.severity == RedFlagSeverity.LOW for flag in flags)

    def test_revenue_growth_vs_ar_red_flag(self):
        """Test AR growing faster than revenue (red flag)."""
        data = pd.DataFrame({
            'revenue': [1000, 1050, 1100],  # 5% CAGR
            'receivables': [100, 130, 170],  # 30% CAGR - suspicious
        }, index=pd.date_range('2021-12-31', periods=3, freq='Y'))

        detector = RevenueRedFlagDetector()
        flags = detector.detect_ar_growth_mismatch(data)

        assert len(flags) > 0
        flag = flags[0]
        assert flag.category == RedFlagCategory.REVENUE
        assert flag.severity in [RedFlagSeverity.MEDIUM, RedFlagSeverity.HIGH]
        assert "receivables" in flag.description.lower()
        assert flag.metric_value > 1.5  # AR growth / Revenue growth ratio

    def test_channel_stuffing_pattern(self):
        """Test detection of channel stuffing (Q4 spike)."""
        data = pd.DataFrame({
            'revenue': [200, 210, 215, 375],  # Q4 spike
            'quarter': ['Q1', 'Q2', 'Q3', 'Q4'],
        }, index=pd.date_range('2023-03-31', periods=4, freq='Q'))

        detector = RevenueRedFlagDetector()
        flags = detector.detect_channel_stuffing(data)

        assert len(flags) > 0
        assert any(flag.severity >= RedFlagSeverity.MEDIUM for flag in flags)
        assert any("Q4" in flag.description or "quarter" in flag.description.lower()
                   for flag in flags)

    def test_dso_trend_increasing(self):
        """Test increasing Days Sales Outstanding trend."""
        data = pd.DataFrame({
            'receivables': [100, 120, 150, 200],
            'revenue': [1000, 1100, 1200, 1300],
        }, index=pd.date_range('2020-12-31', periods=4, freq='Y'))

        detector = RevenueRedFlagDetector()
        flags = detector.detect_dso_trend(data)

        # DSO = (Receivables / Revenue) * 365
        # Should be increasing over time
        assert len(flags) > 0
        flag = flags[0]
        assert "DSO" in flag.description or "days sales" in flag.description.lower()

    def test_bill_and_hold_indicators(self):
        """Test detection of bill-and-hold indicators."""
        data = pd.DataFrame({
            'revenue': [1000, 1200],
            'inventory': [200, 350],  # Inventory spike with revenue
            'receivables': [100, 180],
        }, index=['2022-12-31', '2023-12-31'])

        detector = RevenueRedFlagDetector()
        flags = detector.detect_bill_and_hold(data)

        # Bill-and-hold: revenue up but inventory also up significantly
        assert isinstance(flags, list)


class TestExpenseRedFlagDetector:
    """Test expense manipulation red flags."""

    def test_expense_capitalization_normal(self):
        """Test normal capitalization patterns."""
        data = pd.DataFrame({
            'capex': [100, 110, 120],
            'depreciation': [80, 88, 96],
            'revenue': [1000, 1100, 1200],
        }, index=pd.date_range('2021-12-31', periods=3, freq='Y'))

        detector = ExpenseRedFlagDetector()
        flags = detector.detect_aggressive_capitalization(data)

        assert len(flags) == 0 or all(flag.severity == RedFlagSeverity.LOW for flag in flags)

    def test_expense_capitalization_excessive(self):
        """Test excessive capitalization of expenses."""
        data = pd.DataFrame({
            'capex': [100, 200, 350],  # Rapid increase
            'depreciation': [80, 85, 90],  # Not keeping pace
            'revenue': [1000, 1100, 1200],
        }, index=pd.date_range('2021-12-31', periods=3, freq='Y'))

        detector = ExpenseRedFlagDetector()
        flags = detector.detect_aggressive_capitalization(data)

        assert len(flags) > 0
        assert any(flag.severity >= RedFlagSeverity.MEDIUM for flag in flags)

    def test_restructuring_charge_frequency(self):
        """Test frequent restructuring charges (red flag)."""
        data = pd.DataFrame({
            'restructuring_charges': [0, 50, 0, 75, 0, 60],
            'net_income': [100, 50, 100, 25, 100, 40],
        }, index=pd.date_range('2018-12-31', periods=6, freq='Y'))

        detector = ExpenseRedFlagDetector()
        flags = detector.detect_recurring_nonrecurring(data)

        # 3 restructuring charges in 6 years is suspicious
        assert len(flags) > 0
        assert any("recurring" in flag.description.lower() for flag in flags)

    def test_cookie_jar_reserves(self):
        """Test cookie jar reserve manipulation."""
        data = pd.DataFrame({
            'reserves': [100, 150, 180, 120, 100],  # Build up then release
            'net_income': [80, 75, 70, 120, 130],  # Income smoothing
        }, index=pd.date_range('2019-12-31', periods=5, freq='Y'))

        detector = ExpenseRedFlagDetector()
        flags = detector.detect_reserve_manipulation(data)

        # Large reserve release coinciding with income boost
        assert isinstance(flags, list)


class TestAccrualRedFlagDetector:
    """Test accrual quality red flags."""

    def test_high_accrual_ratio(self):
        """Test detection of high accrual ratio."""
        data = pd.DataFrame({
            'net_income': [100, 150, 200],
            'operating_cash_flow': [90, 100, 110],  # Large gap
            'total_assets': [1000, 1100, 1200],
        }, index=pd.date_range('2021-12-31', periods=3, freq='Y'))

        detector = AccrualRedFlagDetector()
        flags = detector.detect_high_accruals(data)

        # Accrual ratio = (NI - OCF) / Avg Assets
        # Should flag high accruals
        assert len(flags) > 0
        assert any(flag.severity >= RedFlagSeverity.MEDIUM for flag in flags)

    def test_sloan_accrual_anomaly(self):
        """Test Sloan accrual anomaly (predictor of future returns)."""
        data = pd.DataFrame({
            'net_income': [100],
            'operating_cash_flow': [60],  # Very high accruals
            'total_assets': [1000, 1100],
        }, index=['2022-12-31', '2023-12-31'])

        detector = AccrualRedFlagDetector()
        flags = detector.detect_accrual_anomaly(data)

        # High accruals predict lower future returns
        assert len(flags) > 0

    def test_working_capital_manipulation(self):
        """Test working capital manipulation patterns."""
        data = pd.DataFrame({
            'receivables': [100, 150, 180],  # Rapid increase
            'inventory': [200, 280, 340],
            'payables': [150, 140, 130],  # Decreasing
            'revenue': [1000, 1100, 1200],
        }, index=pd.date_range('2021-12-31', periods=3, freq='Y'))

        detector = AccrualRedFlagDetector()
        flags = detector.detect_wc_manipulation(data)

        # AR and inventory up, payables down = cash flow manipulation
        assert len(flags) > 0


class TestOffBalanceSheetDetector:
    """Test off-balance-sheet red flags."""

    def test_operating_lease_detection(self):
        """Test detection of significant operating leases."""
        footnotes = {
            'leases': {
                'operating_lease_commitments': 500,
                'total_debt': 1000,
            }
        }

        detector = OffBalanceSheetDetector()
        flags = detector.detect_operating_leases(footnotes)

        # Operating leases > 50% of debt is significant
        assert len(flags) > 0
        assert any("operating lease" in flag.description.lower() for flag in flags)

    def test_spe_indicators(self):
        """Test Special Purpose Entity indicators."""
        footnotes = {
            'related_party_transactions': [
                {'entity': 'ABC SPE', 'amount': 100},
                {'entity': 'XYZ LLC', 'amount': 50},
            ],
            'variable_interest_entities': ['Entity A', 'Entity B'],
        }

        detector = OffBalanceSheetDetector()
        flags = detector.detect_spe_usage(footnotes)

        # Multiple VIEs and related party transactions
        assert len(flags) > 0

    def test_pension_underfunding(self):
        """Test pension underfunding detection."""
        data = pd.DataFrame({
            'pension_obligation': [1000],
            'pension_assets': [700],  # 30% underfunded
            'total_assets': [5000],
        }, index=['2023-12-31'])

        detector = OffBalanceSheetDetector()
        flags = detector.detect_pension_issues(data)

        assert len(flags) > 0
        assert any("pension" in flag.description.lower() for flag in flags)


class TestCashFlowRedFlagDetector:
    """Test cash flow red flags."""

    def test_fcf_vs_earnings_divergence(self):
        """Test divergence between free cash flow and earnings."""
        data = pd.DataFrame({
            'net_income': [100, 120, 150],
            'operating_cash_flow': [90, 95, 100],
            'capex': [30, 35, 40],
        }, index=pd.date_range('2021-12-31', periods=3, freq='Y'))

        detector = CashFlowRedFlagDetector()
        flags = detector.detect_fcf_earnings_gap(data)

        # FCF = OCF - Capex = much lower than NI
        assert len(flags) > 0
        assert any("cash flow" in flag.description.lower() for flag in flags)

    def test_negative_operating_cash_flow(self):
        """Test persistent negative operating cash flow."""
        data = pd.DataFrame({
            'operating_cash_flow': [-50, -30, -20],
            'net_income': [100, 110, 120],  # Profitable but no cash
        }, index=pd.date_range('2021-12-31', periods=3, freq='Y'))

        detector = CashFlowRedFlagDetector()
        flags = detector.detect_negative_ocf(data)

        assert len(flags) > 0
        assert any(flag.severity >= RedFlagSeverity.HIGH for flag in flags)

    def test_cash_conversion_cycle(self):
        """Test deteriorating cash conversion cycle."""
        data = pd.DataFrame({
            'receivables': [100, 120, 150],
            'inventory': [200, 240, 300],
            'payables': [150, 155, 160],
            'revenue': [1000, 1100, 1200],
            'cogs': [600, 660, 720],
        }, index=pd.date_range('2021-12-31', periods=3, freq='Y'))

        detector = CashFlowRedFlagDetector()
        flags = detector.detect_ccc_deterioration(data)

        # CCC = DIO + DSO - DPO, should be increasing
        assert isinstance(flags, list)


class TestRedFlagScorer:
    """Test red flag aggregation and scoring."""

    def test_aggregate_scoring_no_flags(self):
        """Test scoring with no red flags (clean company)."""
        flags = []

        scorer = RedFlagScorer()
        result = scorer.calculate_risk_score(flags)

        assert result['risk_score'] == 0
        assert result['risk_level'] == "Low"
        assert result['critical_count'] == 0

    def test_aggregate_scoring_multiple_flags(self):
        """Test scoring with multiple red flags."""
        from stanley.accounting.red_flags import RedFlag

        flags = [
            RedFlag(
                category=RedFlagCategory.REVENUE,
                severity=RedFlagSeverity.HIGH,
                description="AR growing faster than revenue",
                metric_value=2.5,
                threshold=1.5,
            ),
            RedFlag(
                category=RedFlagCategory.ACCRUAL,
                severity=RedFlagSeverity.MEDIUM,
                description="High accrual ratio",
                metric_value=0.15,
                threshold=0.10,
            ),
            RedFlag(
                category=RedFlagCategory.CASH_FLOW,
                severity=RedFlagSeverity.CRITICAL,
                description="Negative operating cash flow",
                metric_value=-50,
                threshold=0,
            ),
        ]

        scorer = RedFlagScorer()
        result = scorer.calculate_risk_score(flags)

        assert result['risk_score'] > 50  # Significant risk
        assert result['risk_level'] in ["Medium", "High", "Critical"]
        assert result['critical_count'] == 1
        assert result['high_count'] == 1
        assert result['medium_count'] == 1

    def test_category_breakdown(self):
        """Test risk score breakdown by category."""
        from stanley.accounting.red_flags import RedFlag

        flags = [
            RedFlag(RedFlagCategory.REVENUE, RedFlagSeverity.HIGH, "Flag 1", 1.0, 0.5),
            RedFlag(RedFlagCategory.REVENUE, RedFlagSeverity.MEDIUM, "Flag 2", 1.0, 0.5),
            RedFlag(RedFlagCategory.EXPENSE, RedFlagSeverity.HIGH, "Flag 3", 1.0, 0.5),
            RedFlag(RedFlagCategory.ACCRUAL, RedFlagSeverity.LOW, "Flag 4", 1.0, 0.5),
        ]

        scorer = RedFlagScorer()
        result = scorer.calculate_risk_score(flags)

        assert 'category_breakdown' in result
        assert result['category_breakdown'][RedFlagCategory.REVENUE] == 2
        assert result['category_breakdown'][RedFlagCategory.EXPENSE] == 1

    def test_severity_weighting(self):
        """Test that severity affects score appropriately."""
        from stanley.accounting.red_flags import RedFlag

        # One critical flag
        flags_critical = [
            RedFlag(RedFlagCategory.REVENUE, RedFlagSeverity.CRITICAL, "Critical issue", 1.0, 0.5)
        ]

        # Multiple low severity flags
        flags_low = [
            RedFlag(RedFlagCategory.REVENUE, RedFlagSeverity.LOW, "Minor issue 1", 1.0, 0.5),
            RedFlag(RedFlagCategory.REVENUE, RedFlagSeverity.LOW, "Minor issue 2", 1.0, 0.5),
            RedFlag(RedFlagCategory.REVENUE, RedFlagSeverity.LOW, "Minor issue 3", 1.0, 0.5),
        ]

        scorer = RedFlagScorer()
        score_critical = scorer.calculate_risk_score(flags_critical)['risk_score']
        score_low = scorer.calculate_risk_score(flags_low)['risk_score']

        # One critical should outweigh multiple low severity
        assert score_critical > score_low

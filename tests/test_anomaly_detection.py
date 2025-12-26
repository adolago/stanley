"""Tests for anomaly detection."""

import pytest
import pandas as pd
import numpy as np
from scipy import stats
from unittest.mock import Mock, patch
from datetime import datetime

from stanley.accounting.anomaly_detection import (
    TimeSeriesAnomalyDetector,
    BenfordAnalyzer,
    PeerComparisonAnalyzer,
    FootnoteAnomalyDetector,
    DisclosureQualityScorer,
    AnomalyAggregator,
    AnomalyType,
)


class TestTimeSeriesAnomalyDetector:
    """Test time series anomaly detection."""

    def test_zscore_normal_data(self):
        """Test Z-score detection with normal data (no outliers)."""
        np.random.seed(42)
        data = pd.Series(
            np.random.normal(100, 10, 50),
            index=pd.date_range("2020-01-01", periods=50, freq="M"),
        )

        detector = TimeSeriesAnomalyDetector()
        anomalies = detector.detect_zscore_outliers(data, threshold=3.0)

        # Should have few or no anomalies with threshold=3
        assert len(anomalies) <= 2  # ~99.7% within 3 sigma

    def test_zscore_with_outliers(self):
        """Test Z-score detection with clear outliers."""
        data = pd.Series(
            [100, 102, 98, 101, 99, 250, 103, 97, 101, -50],  # 250 and -50 are outliers
            index=pd.date_range("2023-01-01", periods=10, freq="M"),
        )

        detector = TimeSeriesAnomalyDetector()
        anomalies = detector.detect_zscore_outliers(data, threshold=2.0)

        assert len(anomalies) >= 2
        assert any(a["value"] == 250 for a in anomalies)
        assert any(a["value"] == -50 for a in anomalies)
        assert all(a["type"] == AnomalyType.STATISTICAL for a in anomalies)

    def test_iqr_outliers(self):
        """Test IQR (Interquartile Range) outlier detection."""
        data = pd.Series([10, 12, 11, 13, 12, 100, 11, 13, 12, 11])

        detector = TimeSeriesAnomalyDetector()
        anomalies = detector.detect_iqr_outliers(data)

        # 100 should be detected as outlier
        assert len(anomalies) >= 1
        assert any(a["value"] == 100 for a in anomalies)

    def test_moving_average_deviation(self):
        """Test moving average deviation detection."""
        # Create data with sudden spike
        data = pd.Series(
            [100] * 10 + [200] + [100] * 10,
            index=pd.date_range("2023-01-01", periods=21, freq="D"),
        )

        detector = TimeSeriesAnomalyDetector()
        anomalies = detector.detect_ma_deviations(data, window=5, threshold=2.0)

        # The spike at position 10 should be detected
        assert len(anomalies) >= 1
        assert any(a["value"] == 200 for a in anomalies)

    def test_seasonal_decomposition(self):
        """Test seasonal pattern anomaly detection."""
        # Create data with seasonal pattern + one anomaly
        t = np.arange(48)
        seasonal = 10 * np.sin(2 * np.pi * t / 12)  # Monthly seasonality
        trend = t * 0.5
        noise = np.random.normal(0, 1, 48)
        data = pd.Series(
            100 + trend + seasonal + noise,
            index=pd.date_range("2020-01-01", periods=48, freq="M"),
        )
        data.iloc[24] += 50  # Add anomaly

        detector = TimeSeriesAnomalyDetector()
        anomalies = detector.detect_seasonal_anomalies(data)

        # Should detect the anomaly at position 24
        assert len(anomalies) >= 1


class TestBenfordAnalyzer:
    """Test Benford's Law analysis."""

    def test_benford_conforming_data(self):
        """Test data that conforms to Benford's Law."""
        # Generate data following Benford's Law distribution
        np.random.seed(42)
        benford_probs = [0.301, 0.176, 0.125, 0.097, 0.079, 0.067, 0.058, 0.051, 0.046]

        data = []
        for digit, prob in enumerate(benford_probs, 1):
            count = int(prob * 1000)
            data.extend(
                [float(f"{digit}.{np.random.randint(0, 99)}") for _ in range(count)]
            )

        analyzer = BenfordAnalyzer()
        result = analyzer.analyze_first_digit(pd.Series(data))

        assert result["conforms_to_benford"] == True
        assert result["chi_square_pvalue"] > 0.05
        assert result["anomaly_score"] < 0.3

    def test_benford_non_conforming_data(self):
        """Test data that does NOT conform to Benford's Law (suspicious)."""
        # Generate uniform distribution (not natural)
        data = pd.Series(np.random.uniform(100, 999, 1000))

        analyzer = BenfordAnalyzer()
        result = analyzer.analyze_first_digit(data)

        assert result["conforms_to_benford"] == False
        assert result["chi_square_pvalue"] < 0.05
        assert result["anomaly_score"] > 0.5

    def test_benford_manipulated_pattern(self):
        """Test detection of manipulated numbers (e.g., round numbers)."""
        # Create data with excessive round numbers
        data = pd.Series([100, 200, 300, 400, 500] * 100)

        analyzer = BenfordAnalyzer()
        result = analyzer.analyze_first_digit(data)

        # Should detect non-natural pattern
        assert result["conforms_to_benford"] == False
        assert "digit_distribution" in result

    def test_benford_second_digit(self):
        """Test second digit Benford analysis."""
        np.random.seed(42)
        data = pd.Series(np.random.lognormal(3, 1, 1000))

        analyzer = BenfordAnalyzer()
        result = analyzer.analyze_second_digit(data)

        assert "digit_distribution" in result
        assert len(result["digit_distribution"]) == 10  # 0-9


class TestPeerComparisonAnalyzer:
    """Test peer comparison anomaly detection."""

    def test_peer_comparison_normal(self):
        """Test company within normal range of peers."""
        company_metrics = {
            "profit_margin": 0.15,
            "roe": 0.18,
            "debt_to_equity": 0.5,
        }

        peer_data = pd.DataFrame(
            {
                "profit_margin": [0.12, 0.14, 0.16, 0.13, 0.17],
                "roe": [0.16, 0.19, 0.17, 0.18, 0.20],
                "debt_to_equity": [0.45, 0.52, 0.48, 0.55, 0.50],
            }
        )

        analyzer = PeerComparisonAnalyzer()
        anomalies = analyzer.detect_peer_anomalies(company_metrics, peer_data)

        # Should have no significant anomalies
        assert len(anomalies) == 0 or all(a["severity"] == "low" for a in anomalies)

    def test_peer_comparison_outlier(self):
        """Test company significantly different from peers."""
        company_metrics = {
            "profit_margin": 0.35,  # Much higher than peers
            "roe": 0.08,  # Much lower than peers
        }

        peer_data = pd.DataFrame(
            {
                "profit_margin": [0.12, 0.14, 0.13, 0.15, 0.14],
                "roe": [0.18, 0.19, 0.20, 0.17, 0.19],
            }
        )

        analyzer = PeerComparisonAnalyzer()
        anomalies = analyzer.detect_peer_anomalies(company_metrics, peer_data)

        # Should detect both anomalies
        assert len(anomalies) >= 2
        assert any(a["metric"] == "profit_margin" for a in anomalies)
        assert any(a["metric"] == "roe" for a in anomalies)

    def test_industry_benchmark_comparison(self):
        """Test comparison against industry benchmarks."""
        company_metrics = {
            "current_ratio": 0.8,  # Below healthy threshold
            "quick_ratio": 0.5,
            "debt_to_equity": 3.0,  # Very high
        }

        industry_benchmarks = {
            "current_ratio": {"median": 1.5, "std": 0.3},
            "quick_ratio": {"median": 1.0, "std": 0.2},
            "debt_to_equity": {"median": 1.0, "std": 0.5},
        }

        analyzer = PeerComparisonAnalyzer()
        anomalies = analyzer.compare_to_benchmarks(company_metrics, industry_benchmarks)

        assert len(anomalies) >= 2
        assert any("current_ratio" in a["metric"] for a in anomalies)


class TestFootnoteAnomalyDetector:
    """Test footnote and disclosure anomaly detection."""

    def test_footnote_length_change(self):
        """Test detection of significant footnote length changes."""
        footnotes_history = [
            {"year": 2020, "text": "A" * 1000},  # Normal
            {"year": 2021, "text": "B" * 1100},  # Normal growth
            {"year": 2022, "text": "C" * 3000},  # Sudden increase (red flag)
        ]

        detector = FootnoteAnomalyDetector()
        anomalies = detector.detect_length_changes(footnotes_history)

        assert len(anomalies) > 0
        assert any("2022" in str(a["year"]) for a in anomalies)

    def test_accounting_policy_changes(self):
        """Test detection of accounting policy changes."""
        current_policies = {
            "revenue_recognition": "ASC 606",
            "inventory_method": "LIFO",
            "depreciation": "Straight-line",
        }

        prior_policies = {
            "revenue_recognition": "ASC 605",  # Changed
            "inventory_method": "LIFO",
            "depreciation": "Straight-line",
        }

        detector = FootnoteAnomalyDetector()
        changes = detector.detect_policy_changes(current_policies, prior_policies)

        assert len(changes) == 1
        assert changes[0]["policy"] == "revenue_recognition"
        assert changes[0]["severity"] in ["medium", "high"]

    def test_related_party_transactions_increase(self):
        """Test detection of increasing related party transactions."""
        rpt_history = pd.DataFrame(
            {
                "total_rpt": [100, 110, 500],  # Sudden spike
                "revenue": [10000, 11000, 12000],
            },
            index=[2020, 2021, 2022],
        )

        detector = FootnoteAnomalyDetector()
        anomalies = detector.detect_rpt_anomalies(rpt_history)

        assert len(anomalies) > 0
        assert any(a["type"] == AnomalyType.DISCLOSURE for a in anomalies)

    def test_restatement_detection(self):
        """Test detection of financial restatements."""
        filings = [
            {"date": "2022-03-31", "type": "10-K", "restatement": False},
            {"date": "2022-06-30", "type": "10-Q", "restatement": False},
            {"date": "2022-09-30", "type": "10-Q", "restatement": True},  # Restatement
        ]

        detector = FootnoteAnomalyDetector()
        anomalies = detector.detect_restatements(filings)

        assert len(anomalies) > 0
        assert all(a["severity"] == "critical" for a in anomalies)


class TestDisclosureQualityScorer:
    """Test disclosure quality scoring."""

    def test_high_quality_disclosure(self):
        """Test scoring of high-quality disclosure."""
        disclosure = {
            "completeness_score": 0.95,
            "clarity_score": 0.90,
            "timeliness_score": 1.0,
            "consistency_score": 0.92,
        }

        scorer = DisclosureQualityScorer()
        result = scorer.calculate_quality_score(disclosure)

        assert result["overall_score"] > 0.90
        assert result["quality_grade"] == "A"

    def test_low_quality_disclosure(self):
        """Test scoring of low-quality disclosure."""
        disclosure = {
            "completeness_score": 0.60,
            "clarity_score": 0.55,
            "timeliness_score": 0.70,
            "consistency_score": 0.50,
        }

        scorer = DisclosureQualityScorer()
        result = scorer.calculate_quality_score(disclosure)

        assert result["overall_score"] < 0.65
        assert result["quality_grade"] in ["C", "D", "F"]

    def test_readability_analysis(self):
        """Test readability scoring (Flesch-Kincaid, etc.)."""
        simple_text = "The company earns money. It sells products. Customers pay cash."
        complex_text = """
        The corporation's operational paradigm leverages synergistic optimization
        to maximize stakeholder value through diversified revenue stream monetization
        and vertical integration of cross-functional enterprise solutions.
        """

        scorer = DisclosureQualityScorer()
        simple_score = scorer.analyze_readability(simple_text)
        complex_score = scorer.analyze_readability(complex_text)

        # Simple text should score better
        assert simple_score["readability_score"] > complex_score["readability_score"]

    def test_boilerplate_detection(self):
        """Test detection of excessive boilerplate language."""
        text_with_boilerplate = (
            """
        Forward-looking statements involve risks and uncertainties.
        Actual results may differ materially from those projected.
        We undertake no obligation to update forward-looking statements.
        """
            * 10
        )  # Repeated boilerplate

        scorer = DisclosureQualityScorer()
        result = scorer.detect_boilerplate(text_with_boilerplate)

        assert result["boilerplate_ratio"] > 0.5
        assert result["unique_content_ratio"] < 0.5


class TestAnomalyAggregator:
    """Test anomaly aggregation and prioritization."""

    def test_aggregate_multiple_sources(self):
        """Test aggregation of anomalies from multiple detectors."""
        anomalies = {
            "timeseries": [
                {
                    "type": AnomalyType.STATISTICAL,
                    "severity": "high",
                    "metric": "revenue",
                },
            ],
            "benford": [
                {
                    "type": AnomalyType.BENFORD,
                    "severity": "medium",
                    "metric": "expenses",
                },
            ],
            "peer": [
                {
                    "type": AnomalyType.PEER,
                    "severity": "high",
                    "metric": "profit_margin",
                },
            ],
            "footnote": [
                {
                    "type": AnomalyType.DISCLOSURE,
                    "severity": "critical",
                    "metric": "restatement",
                },
            ],
        }

        aggregator = AnomalyAggregator()
        result = aggregator.aggregate(anomalies)

        assert result["total_anomalies"] == 4
        assert result["critical_count"] == 1
        assert result["high_count"] == 2
        assert "risk_score" in result

    def test_prioritization(self):
        """Test anomaly prioritization by severity and impact."""
        anomalies = [
            {"severity": "low", "impact": 0.2, "metric": "A"},
            {"severity": "critical", "impact": 0.9, "metric": "B"},
            {"severity": "medium", "impact": 0.5, "metric": "C"},
            {"severity": "high", "impact": 0.7, "metric": "D"},
        ]

        aggregator = AnomalyAggregator()
        prioritized = aggregator.prioritize(anomalies)

        # Critical severity should be first
        assert prioritized[0]["severity"] == "critical"
        assert prioritized[-1]["severity"] == "low"

    def test_clustering_similar_anomalies(self):
        """Test clustering of related anomalies."""
        anomalies = [
            {"metric": "receivables", "category": "working_capital"},
            {"metric": "inventory", "category": "working_capital"},
            {"metric": "payables", "category": "working_capital"},
            {"metric": "revenue", "category": "income_statement"},
        ]

        aggregator = AnomalyAggregator()
        clusters = aggregator.cluster_anomalies(anomalies)

        assert "working_capital" in clusters
        assert len(clusters["working_capital"]) == 3
        assert len(clusters["income_statement"]) == 1

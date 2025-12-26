"""
Institutional Analytics Algorithms Specification
================================================

This module defines the data models, thresholds, and scoring algorithms
for enhanced institutional analytics in Stanley.

Author: Analyst Agent (Stanley Hive Mind)
Date: 2025-12-26

DESIGN PHILOSOPHY:
- All algorithms use only numpy for numerical operations (no sklearn dependency)
- Thresholds are derived from institutional investor behavior research
- Scores are normalized to [-1, 1] or [0, 1] ranges for consistency
- All algorithms are designed for real-time streaming updates
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# =============================================================================
# SECTION 1: WHALE DETECTION THRESHOLDS
# =============================================================================

class PositionChangeLevel(Enum):
    """
    Position change thresholds based on institutional behavior patterns.

    Rationale:
    - Minor (<5%): Normal portfolio rebalancing, noise
    - Moderate (5-10%): Tactical adjustment, worth monitoring
    - Significant (10-25%): Conviction-driven change, strong signal
    - Major (>25%): Strategic shift, potential alpha signal
    """
    MINOR = "minor"           # < 5% change
    MODERATE = "moderate"     # 5-10% change
    SIGNIFICANT = "significant"  # 10-25% change
    MAJOR = "major"           # > 25% change


class AUMTier(Enum):
    """
    Assets Under Management tiers for institution classification.

    Rationale:
    - Retail (<$100M): Individual/small RIA, limited market impact
    - Institutional ($100M-$1B): Mid-size funds, meaningful positions
    - Whale ($1B-$10B): Large funds, significant market impact
    - Mega ($10B+): Asset managers, market-moving capacity
    """
    RETAIL = "retail"         # < $100M
    INSTITUTIONAL = "institutional"  # $100M - $1B
    WHALE = "whale"           # $1B - $10B
    MEGA = "mega"             # > $10B


class AlertLevel(Enum):
    """
    Alert severity levels for institutional activity.
    """
    INFO = "info"             # Routine monitoring
    WARNING = "warning"       # Elevated attention
    SIGNIFICANT = "significant"  # Notable event
    CRITICAL = "critical"     # Immediate attention


@dataclass
class WhaleDetectionThresholds:
    """
    Threshold configuration for whale detection algorithms.
    """
    # Position change thresholds (as decimals)
    position_change_minor: float = 0.05      # 5%
    position_change_moderate: float = 0.10   # 10%
    position_change_significant: float = 0.25  # 25%

    # AUM tier thresholds (in USD)
    aum_retail_max: float = 100_000_000       # $100M
    aum_institutional_max: float = 1_000_000_000  # $1B
    aum_whale_max: float = 10_000_000_000     # $10B

    # Alert trigger thresholds
    alert_min_position_value: float = 10_000_000  # $10M minimum for alerts
    alert_min_aum: float = 500_000_000        # $500M minimum AUM for tracking

    def classify_position_change(self, change_percent: float) -> PositionChangeLevel:
        """Classify a position change by magnitude."""
        abs_change = abs(change_percent)
        if abs_change < self.position_change_minor:
            return PositionChangeLevel.MINOR
        elif abs_change < self.position_change_moderate:
            return PositionChangeLevel.MODERATE
        elif abs_change < self.position_change_significant:
            return PositionChangeLevel.SIGNIFICANT
        else:
            return PositionChangeLevel.MAJOR

    def classify_aum(self, aum: float) -> AUMTier:
        """Classify an institution by AUM tier."""
        if aum < self.aum_retail_max:
            return AUMTier.RETAIL
        elif aum < self.aum_institutional_max:
            return AUMTier.INSTITUTIONAL
        elif aum < self.aum_whale_max:
            return AUMTier.WHALE
        else:
            return AUMTier.MEGA

    def determine_alert_level(
        self,
        position_change: PositionChangeLevel,
        aum_tier: AUMTier,
        is_new_position: bool = False,
        is_exit: bool = False,
    ) -> AlertLevel:
        """
        Determine alert level based on position change and institution size.

        Algorithm:
        - New positions from whales/mega = SIGNIFICANT
        - Exits from whales/mega = CRITICAL
        - Major changes from whales/mega = CRITICAL
        - Significant changes from institutional+ = SIGNIFICANT
        - Moderate changes from whale+ = WARNING
        - Everything else = INFO
        """
        if is_exit and aum_tier in (AUMTier.WHALE, AUMTier.MEGA):
            return AlertLevel.CRITICAL

        if is_new_position and aum_tier in (AUMTier.WHALE, AUMTier.MEGA):
            return AlertLevel.SIGNIFICANT

        if position_change == PositionChangeLevel.MAJOR:
            if aum_tier in (AUMTier.WHALE, AUMTier.MEGA):
                return AlertLevel.CRITICAL
            elif aum_tier == AUMTier.INSTITUTIONAL:
                return AlertLevel.SIGNIFICANT

        if position_change == PositionChangeLevel.SIGNIFICANT:
            if aum_tier in (AUMTier.INSTITUTIONAL, AUMTier.WHALE, AUMTier.MEGA):
                return AlertLevel.SIGNIFICANT

        if position_change == PositionChangeLevel.MODERATE:
            if aum_tier in (AUMTier.WHALE, AUMTier.MEGA):
                return AlertLevel.WARNING

        return AlertLevel.INFO


# =============================================================================
# SECTION 2: SENTIMENT SCORE MODEL
# =============================================================================

@dataclass
class SentimentWeights:
    """
    Weight configuration for multi-factor sentiment model.

    Factor Descriptions:
    - ownership_trend (0.30): Direction and magnitude of total institutional ownership
    - buyer_seller_ratio (0.25): Balance between buyers and sellers
    - concentration_delta (0.20): Change in position concentration (HHI delta)
    - filing_momentum (0.25): Rate of change in filing activity

    Total weights sum to 1.0 for normalized output.
    """
    ownership_trend: float = 0.30
    buyer_seller_ratio: float = 0.25
    concentration_delta: float = 0.20
    filing_momentum: float = 0.25

    def validate(self) -> bool:
        """Ensure weights sum to 1.0."""
        total = (
            self.ownership_trend +
            self.buyer_seller_ratio +
            self.concentration_delta +
            self.filing_momentum
        )
        return abs(total - 1.0) < 1e-6


class InstitutionalSentimentModel:
    """
    Multi-factor model for calculating institutional sentiment.

    FORMULA:
        sentiment = w1*ownership_trend + w2*buyer_seller_ratio +
                   w3*concentration_delta + w4*filing_momentum

    Each factor is normalized to [-1, 1] before weighting.
    Final sentiment score ranges from -1 (extremely bearish) to +1 (extremely bullish).
    """

    def __init__(self, weights: Optional[SentimentWeights] = None):
        self.weights = weights or SentimentWeights()
        assert self.weights.validate(), "Weights must sum to 1.0"

    def calculate_ownership_trend(
        self,
        current_ownership: float,
        previous_ownership: float,
        lookback_quarters: int = 4,
    ) -> float:
        """
        Calculate normalized ownership trend factor.

        Algorithm:
        1. Calculate raw change: (current - previous) / previous
        2. Normalize to [-1, 1] using sigmoid-like transformation
        3. Threshold at +/-50% change for saturation

        Args:
            current_ownership: Current institutional ownership percentage
            previous_ownership: Previous period ownership percentage
            lookback_quarters: Number of quarters for trend calculation

        Returns:
            Normalized ownership trend in [-1, 1]
        """
        if previous_ownership <= 0:
            return 0.0

        raw_change = (current_ownership - previous_ownership) / previous_ownership

        # Normalize using tanh-like transformation
        # Changes of +/-50% saturate at +/-1
        saturation_threshold = 0.50
        normalized = np.tanh(raw_change / saturation_threshold)

        return float(np.clip(normalized, -1.0, 1.0))

    def calculate_buyer_seller_ratio(
        self,
        buyers: int,
        sellers: int,
    ) -> float:
        """
        Calculate normalized buyer/seller ratio factor.

        FORMULA:
            ratio = (buyers - sellers) / (buyers + sellers)

        This naturally produces values in [-1, 1]:
        - All buyers, no sellers: +1
        - Equal buyers/sellers: 0
        - All sellers, no buyers: -1

        Args:
            buyers: Number of institutions increasing positions
            sellers: Number of institutions decreasing positions

        Returns:
            Normalized buyer/seller ratio in [-1, 1]
        """
        total = buyers + sellers
        if total == 0:
            return 0.0

        return float((buyers - sellers) / total)

    def calculate_concentration_delta(
        self,
        current_hhi: float,
        previous_hhi: float,
    ) -> float:
        """
        Calculate normalized concentration change factor.

        Algorithm:
        - HHI decrease = more distributed = bullish (positive factor)
        - HHI increase = more concentrated = bearish (negative factor)

        Rationale: Decreasing concentration often signals broader
        institutional interest, which is bullish for price discovery.

        FORMULA:
            delta = -(current_hhi - previous_hhi) / max(previous_hhi, 0.01)

        The negative sign inverts the relationship so that
        decreasing HHI (more distributed) yields positive sentiment.

        Args:
            current_hhi: Current Herfindahl-Hirschman Index [0, 1]
            previous_hhi: Previous period HHI [0, 1]

        Returns:
            Normalized concentration delta in [-1, 1]
        """
        if previous_hhi <= 0.01:
            return 0.0

        # Raw change in HHI
        raw_delta = (current_hhi - previous_hhi) / previous_hhi

        # Invert: decreasing HHI is positive
        inverted_delta = -raw_delta

        # Normalize with saturation at +/-30% HHI change
        saturation_threshold = 0.30
        normalized = np.tanh(inverted_delta / saturation_threshold)

        return float(np.clip(normalized, -1.0, 1.0))

    def calculate_filing_momentum(
        self,
        position_changes: np.ndarray,
        performance_weights: Optional[np.ndarray] = None,
        lookback_quarters: int = 4,
    ) -> float:
        """
        Calculate filing momentum factor.

        Algorithm:
        1. Sum position changes weighted by optional performance scores
        2. Compare recent period to historical average
        3. Normalize to [-1, 1]

        FORMULA:
            momentum = sum(position_change_i * weight_i) for i in window
            normalized_momentum = tanh(momentum / scale_factor)

        Args:
            position_changes: Array of position change percentages per quarter
            performance_weights: Optional performance weights for each institution
            lookback_quarters: Number of quarters to consider

        Returns:
            Normalized filing momentum in [-1, 1]
        """
        if len(position_changes) == 0:
            return 0.0

        # Use lookback window
        changes = position_changes[-lookback_quarters:]

        if performance_weights is not None:
            weights = performance_weights[-lookback_quarters:]
            # Normalize weights
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                weights = np.ones_like(changes) / len(changes)
            momentum = np.sum(changes * weights)
        else:
            momentum = np.mean(changes)

        # Normalize with saturation at +/-25% average change
        saturation_threshold = 0.25
        normalized = np.tanh(momentum / saturation_threshold)

        return float(np.clip(normalized, -1.0, 1.0))

    def calculate_sentiment(
        self,
        ownership_trend: float,
        buyer_seller_ratio: float,
        concentration_delta: float,
        filing_momentum: float,
    ) -> Dict[str, float]:
        """
        Calculate overall institutional sentiment score.

        Args:
            ownership_trend: Normalized ownership trend [-1, 1]
            buyer_seller_ratio: Normalized buyer/seller ratio [-1, 1]
            concentration_delta: Normalized concentration change [-1, 1]
            filing_momentum: Normalized filing momentum [-1, 1]

        Returns:
            Dictionary with:
            - sentiment: Overall score [-1, 1]
            - components: Individual factor contributions
            - signal: Categorical signal (bullish/bearish/neutral)
            - strength: Signal strength (weak/moderate/strong)
        """
        # Calculate weighted sentiment
        sentiment = (
            self.weights.ownership_trend * ownership_trend +
            self.weights.buyer_seller_ratio * buyer_seller_ratio +
            self.weights.concentration_delta * concentration_delta +
            self.weights.filing_momentum * filing_momentum
        )

        # Determine signal and strength
        abs_sentiment = abs(sentiment)
        if abs_sentiment < 0.15:
            signal = "neutral"
            strength = "weak"
        elif abs_sentiment < 0.35:
            signal = "bullish" if sentiment > 0 else "bearish"
            strength = "moderate"
        else:
            signal = "bullish" if sentiment > 0 else "bearish"
            strength = "strong"

        return {
            "sentiment": float(sentiment),
            "components": {
                "ownership_trend": ownership_trend * self.weights.ownership_trend,
                "buyer_seller_ratio": buyer_seller_ratio * self.weights.buyer_seller_ratio,
                "concentration_delta": concentration_delta * self.weights.concentration_delta,
                "filing_momentum": filing_momentum * self.weights.filing_momentum,
            },
            "signal": signal,
            "strength": strength,
        }


# =============================================================================
# SECTION 3: POSITION CLUSTERING LOGIC
# =============================================================================

@dataclass
class ClusterResult:
    """Result from position clustering analysis."""
    cluster_id: int
    centroid_position_size: float
    centroid_performance_score: float
    member_count: int
    is_smart_money: bool
    total_aum: float
    avg_conviction: float


class PositionClusteringEngine:
    """
    Position clustering without sklearn dependency.

    Uses quartile-based clustering with numpy for grouping
    institutional positions by size and performance.

    ALGORITHM:
    1. Divide positions into quartiles by size
    2. Calculate cluster centroids using weighted mean
    3. Identify "smart money cluster" (Q4 size + high performance)
    4. Track cluster movement over time
    """

    def __init__(self):
        self.quartile_labels = ["Q1_Small", "Q2_Medium", "Q3_Large", "Q4_Whale"]

    def calculate_quartiles(
        self,
        positions: pd.DataFrame,
        size_column: str = "position_value",
    ) -> pd.DataFrame:
        """
        Assign positions to quartile-based clusters.

        Args:
            positions: DataFrame with position data
            size_column: Column name for position size

        Returns:
            DataFrame with added 'quartile' and 'cluster_id' columns
        """
        df = positions.copy()

        # Calculate quartile boundaries
        quartiles = np.percentile(df[size_column], [25, 50, 75])

        # Assign quartile labels
        def assign_quartile(value):
            if value < quartiles[0]:
                return 0  # Q1
            elif value < quartiles[1]:
                return 1  # Q2
            elif value < quartiles[2]:
                return 2  # Q3
            else:
                return 3  # Q4

        df["cluster_id"] = df[size_column].apply(assign_quartile)
        df["quartile_label"] = df["cluster_id"].map(
            dict(enumerate(self.quartile_labels))
        )

        return df

    def calculate_cluster_centroids(
        self,
        positions: pd.DataFrame,
        size_column: str = "position_value",
        performance_column: str = "performance_score",
    ) -> Dict[int, Dict[str, float]]:
        """
        Calculate centroids for each cluster.

        Centroid = weighted mean of cluster members
        Weight = position size (larger positions have more influence)

        Args:
            positions: DataFrame with position data and cluster assignments
            size_column: Column for position size
            performance_column: Column for performance score

        Returns:
            Dictionary mapping cluster_id to centroid coordinates
        """
        centroids = {}

        for cluster_id in range(4):
            cluster_mask = positions["cluster_id"] == cluster_id
            cluster_data = positions[cluster_mask]

            if len(cluster_data) == 0:
                centroids[cluster_id] = {
                    "position_size": 0.0,
                    "performance_score": 0.0,
                    "count": 0,
                }
                continue

            # Weighted centroid calculation
            weights = cluster_data[size_column].values
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(len(cluster_data)) / len(cluster_data)

            centroid_size = np.average(cluster_data[size_column], weights=weights)

            if performance_column in cluster_data.columns:
                centroid_perf = np.average(
                    cluster_data[performance_column], weights=weights
                )
            else:
                centroid_perf = 0.0

            centroids[cluster_id] = {
                "position_size": float(centroid_size),
                "performance_score": float(centroid_perf),
                "count": len(cluster_data),
                "total_aum": float(cluster_data[size_column].sum()),
            }

        return centroids

    def identify_smart_money_cluster(
        self,
        positions: pd.DataFrame,
        performance_threshold: float = 0.60,
        size_column: str = "position_value",
        performance_column: str = "performance_score",
    ) -> pd.DataFrame:
        """
        Identify the "smart money" cluster.

        DEFINITION:
        Smart money = Top quartile (Q4) positions + Above-threshold performance

        This identifies institutions that are both large AND have
        demonstrated strong historical performance.

        Args:
            positions: DataFrame with clustered positions
            performance_threshold: Minimum performance score (0-1)
            size_column: Column for position size
            performance_column: Column for performance score

        Returns:
            DataFrame containing only smart money positions
        """
        if "cluster_id" not in positions.columns:
            positions = self.calculate_quartiles(positions, size_column)

        # Smart money criteria:
        # 1. Top quartile (Q4, cluster_id=3)
        # 2. Performance above threshold
        is_top_quartile = positions["cluster_id"] == 3

        if performance_column in positions.columns:
            is_high_performer = positions[performance_column] >= performance_threshold
            smart_money_mask = is_top_quartile & is_high_performer
        else:
            smart_money_mask = is_top_quartile

        return positions[smart_money_mask].copy()

    def measure_cluster_movement(
        self,
        current_positions: pd.DataFrame,
        previous_positions: pd.DataFrame,
        manager_id_column: str = "manager_cik",
        size_column: str = "position_value",
    ) -> Dict[str, any]:
        """
        Measure how positions have moved between clusters over time.

        Algorithm:
        1. Match positions by manager ID
        2. Compare cluster assignments between periods
        3. Calculate net flows between clusters

        Args:
            current_positions: Current period positions
            previous_positions: Previous period positions
            manager_id_column: Column identifying managers
            size_column: Column for position size

        Returns:
            Dictionary with cluster movement metrics
        """
        # Calculate quartiles for both periods
        current = self.calculate_quartiles(current_positions.copy(), size_column)
        previous = self.calculate_quartiles(previous_positions.copy(), size_column)

        # Merge on manager ID
        merged = current.merge(
            previous[[manager_id_column, "cluster_id", size_column]],
            on=manager_id_column,
            how="outer",
            suffixes=("_current", "_previous"),
        ).fillna(-1)  # -1 indicates new/exited positions

        # Calculate transition matrix
        transition_matrix = np.zeros((5, 5))  # 4 clusters + "none" category

        for _, row in merged.iterrows():
            prev_cluster = int(row["cluster_id_previous"]) if row["cluster_id_previous"] >= 0 else 4
            curr_cluster = int(row["cluster_id_current"]) if row["cluster_id_current"] >= 0 else 4
            transition_matrix[prev_cluster, curr_cluster] += 1

        # Calculate key movement metrics
        upgrades = np.sum(np.triu(transition_matrix[:4, :4], k=1))
        downgrades = np.sum(np.tril(transition_matrix[:4, :4], k=-1))
        new_entries = np.sum(transition_matrix[4, :4])
        exits = np.sum(transition_matrix[:4, 4])

        return {
            "transition_matrix": transition_matrix.tolist(),
            "cluster_labels": self.quartile_labels + ["None"],
            "upgrades": int(upgrades),
            "downgrades": int(downgrades),
            "new_entries": int(new_entries),
            "exits": int(exits),
            "net_movement": int(upgrades - downgrades),
            "movement_signal": "bullish" if upgrades > downgrades else (
                "bearish" if downgrades > upgrades else "neutral"
            ),
        }


# =============================================================================
# SECTION 4: CROSS-FILING ANALYSIS
# =============================================================================

@dataclass
class CrossFilingMetrics:
    """Results from cross-filing analysis."""
    agreement_percentage: float  # % of institutions with same direction
    conviction_score: float      # AUM-weighted conviction
    consensus_signal: str        # strong_buy, buy, neutral, sell, strong_sell
    bullish_count: int
    bearish_count: int
    neutral_count: int
    total_bullish_aum: float
    total_bearish_aum: float


class CrossFilingAnalyzer:
    """
    Analyze agreement and conviction across multiple 13F filings.

    METRICS:
    1. Agreement: % of institutions moving in same direction
    2. Conviction: AUM-weighted magnitude of changes
    3. Consensus: Overall signal when agreement > threshold
    """

    def __init__(
        self,
        consensus_threshold: float = 0.60,
        strong_consensus_threshold: float = 0.75,
    ):
        """
        Initialize cross-filing analyzer.

        Args:
            consensus_threshold: Minimum agreement for signal (60%)
            strong_consensus_threshold: Threshold for strong signal (75%)
        """
        self.consensus_threshold = consensus_threshold
        self.strong_consensus_threshold = strong_consensus_threshold

    def calculate_agreement(
        self,
        position_changes: pd.DataFrame,
        change_column: str = "position_change_pct",
        minimum_change: float = 0.01,
    ) -> float:
        """
        Calculate percentage of institutions with same direction.

        Algorithm:
        1. Classify each position as bullish (increase), bearish (decrease), or neutral
        2. Calculate percentage of majority direction

        Args:
            position_changes: DataFrame with position changes
            change_column: Column with change percentage
            minimum_change: Minimum change to be considered (filter noise)

        Returns:
            Agreement percentage [0, 1]
        """
        if len(position_changes) == 0:
            return 0.0

        changes = position_changes[change_column].values

        # Classify directions
        bullish = np.sum(changes > minimum_change)
        bearish = np.sum(changes < -minimum_change)
        neutral = np.sum(np.abs(changes) <= minimum_change)

        total = len(changes)
        max_direction = max(bullish, bearish)

        # Agreement is the fraction in the majority direction
        # (excluding neutral positions from denominator)
        active_positions = bullish + bearish
        if active_positions == 0:
            return 0.0

        return float(max_direction / active_positions)

    def calculate_conviction(
        self,
        position_changes: pd.DataFrame,
        change_column: str = "position_change_pct",
        aum_column: str = "manager_aum",
    ) -> float:
        """
        Calculate AUM-weighted conviction score.

        FORMULA:
            conviction = sum(change_i * aum_i) / sum(aum_i)

        This weights changes by institution size, so larger
        institutions have more influence on the conviction score.

        Args:
            position_changes: DataFrame with position changes and AUM
            change_column: Column with change percentage
            aum_column: Column with manager AUM

        Returns:
            Conviction score [-1, 1]
        """
        if len(position_changes) == 0:
            return 0.0

        changes = position_changes[change_column].values
        aums = position_changes[aum_column].values

        total_aum = np.sum(aums)
        if total_aum == 0:
            return 0.0

        # Normalize AUMs to weights
        weights = aums / total_aum

        # Calculate weighted conviction
        conviction = np.sum(changes * weights)

        # Normalize to [-1, 1]
        # Assume max reasonable average change is 50%
        normalized = np.tanh(conviction / 0.50)

        return float(np.clip(normalized, -1.0, 1.0))

    def determine_consensus(
        self,
        agreement: float,
        conviction: float,
    ) -> str:
        """
        Determine consensus signal based on agreement and conviction.

        Signal Matrix:
        - Strong consensus (>75%) + High conviction (>0.3): strong_buy/strong_sell
        - Consensus (>60%) + Moderate conviction (>0.1): buy/sell
        - Below thresholds: neutral

        Args:
            agreement: Agreement percentage [0, 1]
            conviction: Conviction score [-1, 1]

        Returns:
            Signal string
        """
        # Check if we have sufficient agreement for a signal
        if agreement < self.consensus_threshold:
            return "neutral"

        # Determine direction and strength
        is_bullish = conviction > 0
        abs_conviction = abs(conviction)

        if agreement >= self.strong_consensus_threshold and abs_conviction > 0.30:
            return "strong_buy" if is_bullish else "strong_sell"
        elif abs_conviction > 0.10:
            return "buy" if is_bullish else "sell"
        else:
            return "neutral"

    def analyze_cross_filing(
        self,
        position_changes: pd.DataFrame,
        change_column: str = "position_change_pct",
        aum_column: str = "manager_aum",
    ) -> CrossFilingMetrics:
        """
        Perform complete cross-filing analysis.

        Args:
            position_changes: DataFrame with changes and AUM
            change_column: Column with change percentage
            aum_column: Column with manager AUM

        Returns:
            CrossFilingMetrics with all analysis results
        """
        if len(position_changes) == 0:
            return CrossFilingMetrics(
                agreement_percentage=0.0,
                conviction_score=0.0,
                consensus_signal="neutral",
                bullish_count=0,
                bearish_count=0,
                neutral_count=0,
                total_bullish_aum=0.0,
                total_bearish_aum=0.0,
            )

        changes = position_changes[change_column].values
        aums = position_changes[aum_column].values

        # Count by direction
        bullish_mask = changes > 0.01
        bearish_mask = changes < -0.01
        neutral_mask = np.abs(changes) <= 0.01

        bullish_count = int(np.sum(bullish_mask))
        bearish_count = int(np.sum(bearish_mask))
        neutral_count = int(np.sum(neutral_mask))

        # AUM by direction
        total_bullish_aum = float(np.sum(aums[bullish_mask]))
        total_bearish_aum = float(np.sum(aums[bearish_mask]))

        # Calculate metrics
        agreement = self.calculate_agreement(position_changes, change_column)
        conviction = self.calculate_conviction(position_changes, change_column, aum_column)
        consensus = self.determine_consensus(agreement, conviction)

        return CrossFilingMetrics(
            agreement_percentage=agreement,
            conviction_score=conviction,
            consensus_signal=consensus,
            bullish_count=bullish_count,
            bearish_count=bearish_count,
            neutral_count=neutral_count,
            total_bullish_aum=total_bullish_aum,
            total_bearish_aum=total_bearish_aum,
        )


# =============================================================================
# SECTION 5: SMART MONEY MOMENTUM
# =============================================================================

@dataclass
class SmartMoneyMomentumResult:
    """Result from smart money momentum calculation."""
    momentum: float           # Current momentum score [-1, 1]
    acceleration: float       # Change in momentum
    rolling_changes: List[float]  # Historical changes
    weighted_contributions: List[float]  # Per-manager contributions
    signal: str              # bullish/bearish/neutral
    confidence: float        # Signal confidence [0, 1]


class SmartMoneyMomentumCalculator:
    """
    Calculate smart money momentum with performance weighting.

    ALGORITHM:
    1. Track position changes over rolling 4-quarter windows
    2. Weight changes by manager performance scores
    3. Calculate momentum as weighted sum of position changes
    4. Calculate acceleration as current vs previous momentum

    FORMULA:
        momentum_t = sum(position_change_i * performance_weight_i) for quarter t
        acceleration = momentum_t - momentum_(t-1)
    """

    def __init__(
        self,
        window_quarters: int = 4,
        min_performance_weight: float = 0.10,
        max_performance_weight: float = 0.50,
    ):
        """
        Initialize smart money momentum calculator.

        Args:
            window_quarters: Rolling window size (default 4 quarters)
            min_performance_weight: Minimum weight for any manager
            max_performance_weight: Maximum weight for any manager
        """
        self.window_quarters = window_quarters
        self.min_weight = min_performance_weight
        self.max_weight = max_performance_weight

    def calculate_performance_weights(
        self,
        performance_scores: np.ndarray,
    ) -> np.ndarray:
        """
        Convert performance scores to normalized weights.

        Algorithm:
        1. Clip weights to [min_weight, max_weight]
        2. Normalize to sum to 1.0

        Args:
            performance_scores: Array of performance scores [0, 1]

        Returns:
            Normalized weight array summing to 1.0
        """
        if len(performance_scores) == 0:
            return np.array([])

        # Convert scores to weights with bounds
        weights = np.clip(performance_scores, self.min_weight, self.max_weight)

        # Normalize
        total = np.sum(weights)
        if total > 0:
            weights = weights / total
        else:
            weights = np.ones_like(weights) / len(weights)

        return weights

    def calculate_quarter_momentum(
        self,
        position_changes: np.ndarray,
        performance_scores: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """
        Calculate momentum for a single quarter.

        FORMULA:
            momentum = sum(position_change_i * weight_i)

        Args:
            position_changes: Array of position change percentages
            performance_scores: Array of performance scores

        Returns:
            Tuple of (momentum_score, weighted_contributions)
        """
        if len(position_changes) == 0:
            return 0.0, np.array([])

        weights = self.calculate_performance_weights(performance_scores)
        contributions = position_changes * weights
        momentum = float(np.sum(contributions))

        return momentum, contributions

    def calculate_rolling_momentum(
        self,
        quarterly_changes: List[np.ndarray],
        quarterly_performance: List[np.ndarray],
    ) -> List[float]:
        """
        Calculate momentum for each quarter in the rolling window.

        Args:
            quarterly_changes: List of position change arrays per quarter
            quarterly_performance: List of performance score arrays per quarter

        Returns:
            List of momentum values for each quarter
        """
        momentums = []

        for changes, scores in zip(quarterly_changes, quarterly_performance):
            momentum, _ = self.calculate_quarter_momentum(changes, scores)
            momentums.append(momentum)

        return momentums

    def calculate_acceleration(
        self,
        current_momentum: float,
        previous_momentum: float,
    ) -> float:
        """
        Calculate momentum acceleration.

        FORMULA:
            acceleration = current_momentum - previous_momentum

        Args:
            current_momentum: Current period momentum
            previous_momentum: Previous period momentum

        Returns:
            Acceleration value
        """
        return current_momentum - previous_momentum

    def determine_signal(
        self,
        momentum: float,
        acceleration: float,
    ) -> Tuple[str, float]:
        """
        Determine trading signal from momentum and acceleration.

        Signal Logic:
        - Positive momentum + positive acceleration = bullish
        - Positive momentum + negative acceleration = weakening bullish
        - Negative momentum + negative acceleration = bearish
        - Negative momentum + positive acceleration = weakening bearish

        Confidence is based on alignment of momentum and acceleration.

        Args:
            momentum: Current momentum score
            acceleration: Current acceleration

        Returns:
            Tuple of (signal_string, confidence)
        """
        # Determine direction
        if abs(momentum) < 0.05:
            signal = "neutral"
            confidence = 0.3
        elif momentum > 0:
            if acceleration > 0:
                signal = "bullish"
                confidence = min(0.9, 0.5 + abs(momentum) + abs(acceleration))
            else:
                signal = "weakening_bullish"
                confidence = max(0.3, 0.6 - abs(acceleration))
        else:
            if acceleration < 0:
                signal = "bearish"
                confidence = min(0.9, 0.5 + abs(momentum) + abs(acceleration))
            else:
                signal = "weakening_bearish"
                confidence = max(0.3, 0.6 - abs(acceleration))

        return signal, float(np.clip(confidence, 0.0, 1.0))

    def calculate_smart_money_momentum(
        self,
        quarterly_changes: List[np.ndarray],
        quarterly_performance: List[np.ndarray],
    ) -> SmartMoneyMomentumResult:
        """
        Calculate complete smart money momentum analysis.

        Args:
            quarterly_changes: List of position change arrays (oldest to newest)
            quarterly_performance: List of performance score arrays

        Returns:
            SmartMoneyMomentumResult with full analysis
        """
        # Ensure we have enough data
        if len(quarterly_changes) < 2:
            return SmartMoneyMomentumResult(
                momentum=0.0,
                acceleration=0.0,
                rolling_changes=[],
                weighted_contributions=[],
                signal="neutral",
                confidence=0.0,
            )

        # Use last window_quarters
        changes = quarterly_changes[-self.window_quarters:]
        performance = quarterly_performance[-self.window_quarters:]

        # Calculate rolling momentum
        rolling_momentum = self.calculate_rolling_momentum(changes, performance)

        # Current and previous momentum
        current_momentum = rolling_momentum[-1] if rolling_momentum else 0.0
        previous_momentum = rolling_momentum[-2] if len(rolling_momentum) >= 2 else 0.0

        # Calculate acceleration
        acceleration = self.calculate_acceleration(current_momentum, previous_momentum)

        # Calculate current quarter contributions
        _, contributions = self.calculate_quarter_momentum(
            changes[-1] if changes else np.array([]),
            performance[-1] if performance else np.array([]),
        )

        # Determine signal
        signal, confidence = self.determine_signal(current_momentum, acceleration)

        # Normalize momentum to [-1, 1]
        normalized_momentum = float(np.tanh(current_momentum / 0.50))
        normalized_acceleration = float(np.tanh(acceleration / 0.25))

        return SmartMoneyMomentumResult(
            momentum=normalized_momentum,
            acceleration=normalized_acceleration,
            rolling_changes=rolling_momentum,
            weighted_contributions=contributions.tolist() if len(contributions) > 0 else [],
            signal=signal,
            confidence=confidence,
        )


# =============================================================================
# SECTION 6: CONVENIENCE FUNCTIONS FOR INTEGRATION
# =============================================================================

def create_default_thresholds() -> WhaleDetectionThresholds:
    """Create default whale detection thresholds."""
    return WhaleDetectionThresholds()


def create_default_sentiment_model() -> InstitutionalSentimentModel:
    """Create default sentiment model with standard weights."""
    return InstitutionalSentimentModel()


def create_default_clustering_engine() -> PositionClusteringEngine:
    """Create default clustering engine."""
    return PositionClusteringEngine()


def create_default_cross_filing_analyzer() -> CrossFilingAnalyzer:
    """Create default cross-filing analyzer."""
    return CrossFilingAnalyzer()


def create_default_momentum_calculator() -> SmartMoneyMomentumCalculator:
    """Create default momentum calculator."""
    return SmartMoneyMomentumCalculator()


# =============================================================================
# SUMMARY OF ALGORITHMS
# =============================================================================

"""
ALGORITHM SUMMARY FOR IMPLEMENTATION:

1. WHALE DETECTION THRESHOLDS
   - Position changes: <5% minor, 5-10% moderate, 10-25% significant, >25% major
   - AUM tiers: <$100M retail, $100M-$1B institutional, $1B-$10B whale, $10B+ mega
   - Alerts: Matrix of position change level x AUM tier

2. SENTIMENT SCORE MODEL
   sentiment = 0.30*ownership_trend + 0.25*buyer_seller_ratio +
               0.20*concentration_delta + 0.25*filing_momentum

   Each factor normalized to [-1, 1] using tanh transformations.
   Final score in [-1, 1] with categorical signals.

3. POSITION CLUSTERING
   - Quartile-based clustering using numpy (no sklearn)
   - Smart money = Q4 (top 25%) + performance > 0.60
   - Track cluster movement with transition matrices

4. CROSS-FILING ANALYSIS
   - Agreement: % of institutions in same direction (threshold: 60%)
   - Conviction: AUM-weighted average change
   - Consensus: Agreement >= 60% triggers signal

5. SMART MONEY MOMENTUM
   momentum = sum(position_change * performance_weight)
   acceleration = momentum_t - momentum_(t-1)

   Rolling 4-quarter windows with performance weighting.
"""

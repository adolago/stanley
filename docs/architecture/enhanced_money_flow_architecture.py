"""
Enhanced Money Flow Analysis Architecture Design Document
=========================================================

This document defines the architecture for enhancing stanley/analytics/money_flow.py
to provide Bloomberg-competitive money flow analysis capabilities.

Author: System Architecture Agent
Date: 2025-12-26
Version: 1.0.0

OBJECTIVE: Design a comprehensive real-time money flow analysis system with:
1. Real-time dark pool alerts
2. Block trade detection with classification
3. Sector rotation signal generator
4. Smart money tracking aggregation
5. Unusual volume detection
6. Flow momentum indicators
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypedDict,
    Union,
)

import numpy as np
import pandas as pd

# =============================================================================
# SECTION 1: ENUMERATIONS AND CONSTANTS
# =============================================================================


class AlertSeverity(Enum):
    """Alert severity levels for prioritization."""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()


class AlertType(Enum):
    """Types of money flow alerts."""
    DARK_POOL_SURGE = auto()
    BLOCK_TRADE = auto()
    UNUSUAL_VOLUME = auto()
    SECTOR_ROTATION = auto()
    SMART_MONEY_ACCUMULATION = auto()
    SMART_MONEY_DISTRIBUTION = auto()
    FLOW_MOMENTUM_BREAKOUT = auto()
    FLOW_DIVERGENCE = auto()
    INSTITUTIONAL_CROSSOVER = auto()


class BlockTradeType(Enum):
    """Classification of block trades."""
    ACCUMULATION = auto()      # Large buy with minimal price impact
    DISTRIBUTION = auto()      # Large sell with minimal price impact
    MOMENTUM = auto()          # Trade aligned with price direction
    CONTRARIAN = auto()        # Trade against price direction
    CROSS = auto()             # Matched buyer/seller (usually institutional)
    ICEBERG = auto()           # Detected hidden large order
    SWEEP = auto()             # Aggressive order sweeping liquidity


class SectorRotationPhase(Enum):
    """Economic cycle sector rotation phases."""
    EARLY_CYCLE = auto()       # Recovery: Financials, Consumer Discretionary
    MID_CYCLE = auto()         # Expansion: Technology, Industrials
    LATE_CYCLE = auto()        # Peak: Energy, Materials
    RECESSION = auto()         # Contraction: Utilities, Healthcare, Staples


class FlowDirection(Enum):
    """Direction of money flow."""
    INFLOW = auto()
    OUTFLOW = auto()
    NEUTRAL = auto()


# =============================================================================
# SECTION 2: DATA STRUCTURES (TypedDict and Dataclasses)
# =============================================================================


class AlertThresholds(TypedDict):
    """Configuration for alert thresholds."""
    dark_pool_volume_zscore: float          # Z-score threshold for dark pool alerts
    block_trade_min_value: float            # Minimum USD value for block trade
    block_trade_min_shares: int             # Minimum shares for block trade
    unusual_volume_multiplier: float        # Multiple of average volume
    sector_rotation_min_flow: float         # Minimum net flow for rotation signal
    smart_money_min_score: float            # Minimum smart money score
    flow_momentum_lookback: int             # Days for momentum calculation
    flow_momentum_threshold: float          # Momentum breakout threshold


@dataclass
class AlertConfig:
    """
    Comprehensive alert configuration system.

    Allows fine-grained control over alert generation with sensible defaults.
    """
    # Dark Pool Thresholds
    dark_pool_volume_zscore: float = 2.0
    dark_pool_percentage_high: float = 0.40    # 40% dark pool is significant
    dark_pool_percentage_low: float = 0.10     # Below 10% is unusual

    # Block Trade Thresholds
    block_trade_min_value_usd: float = 1_000_000.0
    block_trade_min_shares: int = 10_000
    block_trade_price_impact_max: float = 0.005  # Max 0.5% impact = hidden

    # Volume Thresholds
    unusual_volume_multiplier: float = 2.5      # 2.5x average volume
    unusual_volume_lookback_days: int = 20

    # Sector Rotation Thresholds
    sector_rotation_min_flow_pct: float = 0.02  # 2% relative flow
    sector_rotation_confirmation_days: int = 3

    # Smart Money Thresholds
    smart_money_min_aum: float = 1_000_000_000  # $1B minimum AUM
    smart_money_min_performance: float = 0.10   # 10% annual outperformance
    smart_money_accumulation_threshold: float = 0.05  # 5% position increase

    # Flow Momentum Thresholds
    flow_momentum_lookback_days: int = 10
    flow_momentum_breakout_zscore: float = 2.0
    flow_momentum_divergence_threshold: float = 0.15

    # Alert Cooldown (prevent spam)
    alert_cooldown_minutes: int = 30
    max_alerts_per_symbol_per_hour: int = 5

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "dark_pool": {
                "volume_zscore": self.dark_pool_volume_zscore,
                "percentage_high": self.dark_pool_percentage_high,
                "percentage_low": self.dark_pool_percentage_low,
            },
            "block_trade": {
                "min_value_usd": self.block_trade_min_value_usd,
                "min_shares": self.block_trade_min_shares,
                "price_impact_max": self.block_trade_price_impact_max,
            },
            "volume": {
                "multiplier": self.unusual_volume_multiplier,
                "lookback_days": self.unusual_volume_lookback_days,
            },
            "sector_rotation": {
                "min_flow_pct": self.sector_rotation_min_flow_pct,
                "confirmation_days": self.sector_rotation_confirmation_days,
            },
            "smart_money": {
                "min_aum": self.smart_money_min_aum,
                "min_performance": self.smart_money_min_performance,
                "accumulation_threshold": self.smart_money_accumulation_threshold,
            },
            "flow_momentum": {
                "lookback_days": self.flow_momentum_lookback_days,
                "breakout_zscore": self.flow_momentum_breakout_zscore,
                "divergence_threshold": self.flow_momentum_divergence_threshold,
            },
            "rate_limiting": {
                "cooldown_minutes": self.alert_cooldown_minutes,
                "max_per_symbol_per_hour": self.max_alerts_per_symbol_per_hour,
            },
        }


@dataclass
class MoneyFlowAlert:
    """
    Represents a single money flow alert.

    Attributes:
        alert_id: Unique identifier for the alert
        timestamp: When the alert was generated
        symbol: Stock/ETF symbol
        alert_type: Type of alert (dark pool, block trade, etc.)
        severity: Alert severity level
        direction: Flow direction (inflow/outflow)
        metrics: Quantitative data supporting the alert
        context: Additional context and explanation
        confidence: Confidence score (0.0 to 1.0)
        expires_at: When the alert becomes stale
    """
    alert_id: str
    timestamp: datetime
    symbol: str
    alert_type: AlertType
    severity: AlertSeverity
    direction: FlowDirection
    metrics: Dict[str, float]
    context: str
    confidence: float
    expires_at: datetime
    related_alerts: List[str] = field(default_factory=list)

    def is_expired(self) -> bool:
        """Check if alert has expired."""
        return datetime.now() > self.expires_at

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API serialization."""
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "alert_type": self.alert_type.name,
            "severity": self.severity.name,
            "direction": self.direction.name,
            "metrics": self.metrics,
            "context": self.context,
            "confidence": self.confidence,
            "expires_at": self.expires_at.isoformat(),
            "related_alerts": self.related_alerts,
        }


@dataclass
class BlockTrade:
    """
    Represents a detected block trade.

    Block trades are large transactions that may indicate institutional activity.
    """
    trade_id: str
    timestamp: datetime
    symbol: str
    trade_type: BlockTradeType
    shares: int
    price: float
    value_usd: float
    price_impact_pct: float
    venue: str                        # Exchange or dark pool venue
    direction: FlowDirection
    is_dark_pool: bool
    confidence: float

    @property
    def is_significant(self) -> bool:
        """Check if trade is significant based on value."""
        return self.value_usd >= 1_000_000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trade_id": self.trade_id,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "trade_type": self.trade_type.name,
            "shares": self.shares,
            "price": self.price,
            "value_usd": self.value_usd,
            "price_impact_pct": self.price_impact_pct,
            "venue": self.venue,
            "direction": self.direction.name,
            "is_dark_pool": self.is_dark_pool,
            "confidence": self.confidence,
        }


@dataclass
class SectorRotationSignal:
    """
    Sector rotation signal indicating capital movement between sectors.
    """
    signal_id: str
    timestamp: datetime
    rotation_phase: SectorRotationPhase
    leading_sectors: List[str]        # Sectors receiving inflows
    lagging_sectors: List[str]        # Sectors seeing outflows
    sector_flows: Dict[str, float]    # Symbol -> net flow
    conviction: float                  # Signal strength (0.0 to 1.0)
    lookback_days: int
    context: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "signal_id": self.signal_id,
            "timestamp": self.timestamp.isoformat(),
            "rotation_phase": self.rotation_phase.name,
            "leading_sectors": self.leading_sectors,
            "lagging_sectors": self.lagging_sectors,
            "sector_flows": self.sector_flows,
            "conviction": self.conviction,
            "lookback_days": self.lookback_days,
            "context": self.context,
        }


@dataclass
class SmartMoneyActivity:
    """
    Aggregated smart money activity for a symbol.
    """
    symbol: str
    timestamp: datetime
    managers_accumulating: List[str]
    managers_distributing: List[str]
    net_smart_money_flow: float
    smart_money_score: float          # -1.0 to 1.0
    conviction_score: float           # Based on manager track records
    top_positions: List[Dict[str, Any]]
    context: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "managers_accumulating": self.managers_accumulating,
            "managers_distributing": self.managers_distributing,
            "net_smart_money_flow": self.net_smart_money_flow,
            "smart_money_score": self.smart_money_score,
            "conviction_score": self.conviction_score,
            "top_positions": self.top_positions,
            "context": self.context,
        }


@dataclass
class FlowMomentumIndicator:
    """
    Flow momentum indicator for trend analysis.
    """
    symbol: str
    timestamp: datetime
    momentum_score: float             # Normalized momentum (-1.0 to 1.0)
    momentum_zscore: float            # Z-score of recent momentum
    flow_acceleration: float          # Rate of change of flow
    price_flow_correlation: float     # Correlation between price and flow
    divergence_detected: bool         # Flow/price divergence
    divergence_type: Optional[str]    # "bullish" or "bearish" if divergence
    lookback_days: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "momentum_score": self.momentum_score,
            "momentum_zscore": self.momentum_zscore,
            "flow_acceleration": self.flow_acceleration,
            "price_flow_correlation": self.price_flow_correlation,
            "divergence_detected": self.divergence_detected,
            "divergence_type": self.divergence_type,
            "lookback_days": self.lookback_days,
        }


@dataclass
class UnusualVolumeEvent:
    """
    Unusual volume detection event.
    """
    event_id: str
    timestamp: datetime
    symbol: str
    current_volume: int
    average_volume: float
    volume_ratio: float               # current / average
    zscore: float
    price_change_pct: float
    is_dark_pool_related: bool
    block_trades_detected: int
    severity: AlertSeverity

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "current_volume": self.current_volume,
            "average_volume": self.average_volume,
            "volume_ratio": self.volume_ratio,
            "zscore": self.zscore,
            "price_change_pct": self.price_change_pct,
            "is_dark_pool_related": self.is_dark_pool_related,
            "block_trades_detected": self.block_trades_detected,
            "severity": self.severity.name,
        }


# =============================================================================
# SECTION 3: PROTOCOL DEFINITIONS (Interfaces)
# =============================================================================


class AlertHandler(Protocol):
    """Protocol for alert handlers (webhooks, notifications, etc.)."""

    async def handle_alert(self, alert: MoneyFlowAlert) -> bool:
        """
        Handle an incoming alert.

        Args:
            alert: The alert to handle

        Returns:
            True if handled successfully, False otherwise
        """
        ...


class DataSource(Protocol):
    """Protocol for data sources providing real-time flow data."""

    async def get_realtime_volume(
        self, symbol: str
    ) -> Dict[str, Any]:
        """Get real-time volume data for a symbol."""
        ...

    async def get_dark_pool_prints(
        self, symbol: str, since: datetime
    ) -> List[Dict[str, Any]]:
        """Get dark pool prints since a given time."""
        ...

    async def subscribe_to_trades(
        self, symbols: List[str]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Subscribe to real-time trade feed."""
        ...


# =============================================================================
# SECTION 4: ENHANCED MONEY FLOW ANALYZER CLASS DESIGN
# =============================================================================


class EnhancedMoneyFlowAnalyzer:
    """
    Enhanced Money Flow Analyzer with Bloomberg-competitive capabilities.

    This class extends the base MoneyFlowAnalyzer with:
    - Real-time dark pool alerts
    - Block trade detection and classification
    - Sector rotation signal generation
    - Smart money tracking aggregation
    - Unusual volume detection
    - Flow momentum indicators

    Architecture Principles:
    1. Event-driven alert system with configurable thresholds
    2. Async-first design for real-time data processing
    3. Pluggable data sources via Protocol interfaces
    4. Stateful tracking with in-memory caching
    5. Rate-limited alert generation to prevent spam
    """

    def __init__(
        self,
        data_manager: Optional[Any] = None,
        config: Optional[AlertConfig] = None,
        alert_handlers: Optional[List[AlertHandler]] = None,
    ):
        """
        Initialize enhanced money flow analyzer.

        Args:
            data_manager: DataManager instance for data access
            config: Alert configuration (uses defaults if None)
            alert_handlers: List of handlers for generated alerts
        """
        self.data_manager = data_manager
        self.config = config or AlertConfig()
        self.alert_handlers = alert_handlers or []

        # Internal state
        self._alert_history: Dict[str, List[MoneyFlowAlert]] = {}
        self._block_trade_cache: Dict[str, List[BlockTrade]] = {}
        self._volume_baseline: Dict[str, pd.DataFrame] = {}
        self._smart_money_tracker: Dict[str, SmartMoneyActivity] = {}
        self._last_alert_time: Dict[str, datetime] = {}

        self._logger = logging.getLogger(__name__)
        self._logger.info("EnhancedMoneyFlowAnalyzer initialized")

    # =========================================================================
    # DARK POOL ANALYSIS METHODS
    # =========================================================================

    async def analyze_dark_pool_realtime(
        self,
        symbol: str,
        lookback_minutes: int = 60,
    ) -> Dict[str, Any]:
        """
        Analyze real-time dark pool activity for a symbol.

        This method monitors dark pool prints and generates alerts when
        unusual activity is detected.

        Args:
            symbol: Stock symbol to analyze
            lookback_minutes: Minutes of data to analyze

        Returns:
            Dictionary with dark pool analysis including:
            - current_dark_pool_pct: Current dark pool percentage
            - dark_pool_zscore: Z-score compared to historical
            - large_prints: List of significant dark pool prints
            - net_dark_pool_flow: Net buy/sell from dark pools
            - alerts: Any generated alerts
        """
        ...

    async def get_dark_pool_alerts(
        self,
        symbol: str,
        since: Optional[datetime] = None,
    ) -> List[MoneyFlowAlert]:
        """
        Get dark pool alerts for a symbol.

        Args:
            symbol: Stock symbol
            since: Only return alerts after this time

        Returns:
            List of dark pool alerts
        """
        ...

    async def detect_dark_pool_accumulation(
        self,
        symbol: str,
        lookback_days: int = 5,
    ) -> Optional[MoneyFlowAlert]:
        """
        Detect sustained dark pool accumulation pattern.

        Looks for consistent dark pool buying over multiple days
        with minimal price impact (stealth accumulation).

        Args:
            symbol: Stock symbol
            lookback_days: Days to analyze

        Returns:
            Alert if accumulation pattern detected, None otherwise
        """
        ...

    # =========================================================================
    # BLOCK TRADE DETECTION METHODS
    # =========================================================================

    async def detect_block_trades(
        self,
        symbol: str,
        since: Optional[datetime] = None,
    ) -> List[BlockTrade]:
        """
        Detect and classify block trades for a symbol.

        Block trades are identified by:
        - Trade size exceeding configured threshold
        - Price impact analysis
        - Venue analysis (dark pool vs lit exchange)

        Args:
            symbol: Stock symbol
            since: Only analyze trades after this time

        Returns:
            List of detected block trades with classifications
        """
        ...

    def classify_block_trade(
        self,
        shares: int,
        price: float,
        price_before: float,
        price_after: float,
        is_dark_pool: bool,
        volume_context: Dict[str, float],
    ) -> BlockTradeType:
        """
        Classify a block trade based on its characteristics.

        Classification logic:
        - ACCUMULATION: Large buy with <0.5% price impact
        - DISTRIBUTION: Large sell with <0.5% price impact
        - MOMENTUM: Trade direction aligns with recent price trend
        - CONTRARIAN: Trade direction opposes recent price trend
        - CROSS: Matched institutional trade (minimal impact)
        - ICEBERG: Detected hidden order pattern
        - SWEEP: Aggressive order sweeping multiple levels

        Args:
            shares: Number of shares in the trade
            price: Execution price
            price_before: Price before trade
            price_after: Price after trade
            is_dark_pool: Whether trade occurred in dark pool
            volume_context: Volume statistics for context

        Returns:
            Classified block trade type
        """
        ...

    async def get_block_trade_summary(
        self,
        symbol: str,
        lookback_days: int = 5,
    ) -> Dict[str, Any]:
        """
        Get summary of block trade activity.

        Args:
            symbol: Stock symbol
            lookback_days: Days to analyze

        Returns:
            Summary with trade counts by type, net flow, and trends
        """
        ...

    # =========================================================================
    # SECTOR ROTATION SIGNAL METHODS
    # =========================================================================

    async def generate_sector_rotation_signal(
        self,
        sector_etfs: Optional[List[str]] = None,
        lookback_days: int = 21,
    ) -> SectorRotationSignal:
        """
        Generate sector rotation signal based on money flow.

        Analyzes relative fund flows across sector ETFs to identify
        capital rotation patterns that may indicate economic cycle changes.

        Default sector ETFs:
        - XLK (Technology), XLF (Financials), XLE (Energy)
        - XLV (Healthcare), XLY (Consumer Discretionary)
        - XLP (Consumer Staples), XLI (Industrials)
        - XLB (Materials), XLU (Utilities), XLRE (Real Estate)

        Args:
            sector_etfs: List of sector ETF symbols (uses defaults if None)
            lookback_days: Days to analyze

        Returns:
            Sector rotation signal with leading/lagging sectors
        """
        ...

    def determine_rotation_phase(
        self,
        sector_flows: Dict[str, float],
    ) -> SectorRotationPhase:
        """
        Determine economic cycle phase based on sector flows.

        Uses the classic sector rotation model:
        - Early Cycle: Financials, Consumer Discretionary lead
        - Mid Cycle: Technology, Industrials lead
        - Late Cycle: Energy, Materials lead
        - Recession: Utilities, Healthcare, Staples lead

        Args:
            sector_flows: Net flows by sector

        Returns:
            Detected rotation phase
        """
        ...

    async def get_sector_rotation_alerts(
        self,
        since: Optional[datetime] = None,
    ) -> List[MoneyFlowAlert]:
        """
        Get sector rotation alerts.

        Args:
            since: Only return alerts after this time

        Returns:
            List of sector rotation alerts
        """
        ...

    # =========================================================================
    # SMART MONEY TRACKING METHODS
    # =========================================================================

    async def track_smart_money(
        self,
        symbol: str,
        min_aum: Optional[float] = None,
        min_performance: Optional[float] = None,
    ) -> SmartMoneyActivity:
        """
        Track smart money activity for a symbol.

        Smart money is defined as institutional managers with:
        - AUM above threshold (default $1B)
        - Historical outperformance above threshold

        Args:
            symbol: Stock symbol
            min_aum: Minimum AUM for smart money (uses config default)
            min_performance: Minimum performance for smart money

        Returns:
            Aggregated smart money activity
        """
        ...

    async def aggregate_smart_money_flow(
        self,
        universe: List[str],
    ) -> pd.DataFrame:
        """
        Aggregate smart money flow across a universe of stocks.

        Args:
            universe: List of stock symbols

        Returns:
            DataFrame with smart money metrics for each symbol
        """
        ...

    async def get_smart_money_conviction_picks(
        self,
        min_managers: int = 3,
        min_conviction: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Get stocks with high smart money conviction.

        Conviction picks are stocks where multiple smart money
        managers are building significant positions.

        Args:
            min_managers: Minimum number of managers accumulating
            min_conviction: Minimum conviction score

        Returns:
            List of conviction picks with details
        """
        ...

    # =========================================================================
    # UNUSUAL VOLUME DETECTION METHODS
    # =========================================================================

    async def detect_unusual_volume(
        self,
        symbol: str,
        current_volume: Optional[int] = None,
    ) -> Optional[UnusualVolumeEvent]:
        """
        Detect unusual volume for a symbol.

        Compares current volume against historical average and
        generates event if threshold exceeded.

        Args:
            symbol: Stock symbol
            current_volume: Current volume (fetches if None)

        Returns:
            UnusualVolumeEvent if unusual, None otherwise
        """
        ...

    async def scan_unusual_volume(
        self,
        universe: List[str],
    ) -> List[UnusualVolumeEvent]:
        """
        Scan universe for unusual volume.

        Args:
            universe: List of stock symbols

        Returns:
            List of unusual volume events
        """
        ...

    async def correlate_volume_with_flow(
        self,
        symbol: str,
        lookback_days: int = 20,
    ) -> Dict[str, Any]:
        """
        Correlate unusual volume with flow indicators.

        Analyzes whether unusual volume is accompanied by:
        - Dark pool activity
        - Block trades
        - Institutional flow
        - Options activity

        Args:
            symbol: Stock symbol
            lookback_days: Days to analyze

        Returns:
            Correlation analysis results
        """
        ...

    # =========================================================================
    # FLOW MOMENTUM INDICATOR METHODS
    # =========================================================================

    async def calculate_flow_momentum(
        self,
        symbol: str,
        lookback_days: Optional[int] = None,
    ) -> FlowMomentumIndicator:
        """
        Calculate flow momentum indicator for a symbol.

        Flow momentum measures the rate and direction of institutional
        money flow, normalized for comparison across stocks.

        Args:
            symbol: Stock symbol
            lookback_days: Days for momentum calculation

        Returns:
            Flow momentum indicator with all metrics
        """
        ...

    async def detect_flow_divergence(
        self,
        symbol: str,
        lookback_days: int = 20,
    ) -> Optional[MoneyFlowAlert]:
        """
        Detect price/flow divergence.

        Divergence occurs when:
        - Bullish: Price falling but flow increasing (accumulation)
        - Bearish: Price rising but flow decreasing (distribution)

        Args:
            symbol: Stock symbol
            lookback_days: Days to analyze

        Returns:
            Alert if divergence detected, None otherwise
        """
        ...

    async def get_flow_momentum_leaders(
        self,
        universe: List[str],
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Get top flow momentum stocks from universe.

        Args:
            universe: List of stock symbols
            top_n: Number of leaders to return

        Returns:
            DataFrame with top momentum stocks
        """
        ...

    # =========================================================================
    # ALERT MANAGEMENT METHODS
    # =========================================================================

    async def generate_alert(
        self,
        symbol: str,
        alert_type: AlertType,
        metrics: Dict[str, float],
        context: str,
        severity: Optional[AlertSeverity] = None,
    ) -> Optional[MoneyFlowAlert]:
        """
        Generate a money flow alert.

        Handles:
        - Rate limiting (cooldown between alerts)
        - Severity calculation if not provided
        - Alert storage and dispatch to handlers

        Args:
            symbol: Stock symbol
            alert_type: Type of alert
            metrics: Quantitative metrics supporting alert
            context: Human-readable context
            severity: Alert severity (calculated if None)

        Returns:
            Generated alert if not rate-limited, None otherwise
        """
        ...

    def register_alert_handler(
        self,
        handler: AlertHandler,
    ) -> None:
        """
        Register an alert handler.

        Args:
            handler: Alert handler to register
        """
        self.alert_handlers.append(handler)

    async def dispatch_alert(
        self,
        alert: MoneyFlowAlert,
    ) -> None:
        """
        Dispatch alert to all registered handlers.

        Args:
            alert: Alert to dispatch
        """
        for handler in self.alert_handlers:
            try:
                await handler.handle_alert(alert)
            except Exception as e:
                self._logger.error(
                    f"Alert handler failed: {e}",
                    exc_info=True,
                )

    async def get_active_alerts(
        self,
        symbol: Optional[str] = None,
        alert_type: Optional[AlertType] = None,
        min_severity: Optional[AlertSeverity] = None,
    ) -> List[MoneyFlowAlert]:
        """
        Get active (non-expired) alerts.

        Args:
            symbol: Filter by symbol
            alert_type: Filter by alert type
            min_severity: Minimum severity level

        Returns:
            List of active alerts matching criteria
        """
        ...

    # =========================================================================
    # COMPOSITE ANALYSIS METHODS
    # =========================================================================

    async def get_comprehensive_flow_analysis(
        self,
        symbol: str,
    ) -> Dict[str, Any]:
        """
        Get comprehensive money flow analysis for a symbol.

        Combines all analysis types into a single report:
        - Dark pool activity
        - Block trades
        - Smart money tracking
        - Volume analysis
        - Flow momentum
        - Active alerts

        Args:
            symbol: Stock symbol

        Returns:
            Comprehensive analysis dictionary
        """
        ...

    async def get_market_flow_dashboard(
        self,
        universe: List[str],
    ) -> Dict[str, Any]:
        """
        Get market-wide flow dashboard.

        Provides overview of:
        - Sector rotation status
        - Top flow momentum stocks
        - Unusual volume events
        - Smart money activity
        - Active alerts summary

        Args:
            universe: List of stock symbols

        Returns:
            Market flow dashboard data
        """
        ...

    # =========================================================================
    # STREAMING / REAL-TIME METHODS
    # =========================================================================

    async def stream_alerts(
        self,
        symbols: List[str],
    ) -> AsyncGenerator[MoneyFlowAlert, None]:
        """
        Stream real-time alerts for symbols.

        Args:
            symbols: List of symbols to monitor

        Yields:
            MoneyFlowAlert objects as they are generated
        """
        ...

    async def start_realtime_monitoring(
        self,
        symbols: List[str],
        data_source: DataSource,
    ) -> None:
        """
        Start real-time monitoring for symbols.

        This method runs continuously, processing real-time data
        and generating alerts.

        Args:
            symbols: List of symbols to monitor
            data_source: Real-time data source
        """
        ...

    async def stop_realtime_monitoring(self) -> None:
        """Stop real-time monitoring."""
        ...

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def update_config(
        self,
        config: AlertConfig,
    ) -> None:
        """
        Update alert configuration.

        Args:
            config: New configuration
        """
        self.config = config
        self._logger.info("Configuration updated")

    def clear_alert_history(
        self,
        symbol: Optional[str] = None,
    ) -> None:
        """
        Clear alert history.

        Args:
            symbol: Clear only for this symbol (all if None)
        """
        if symbol:
            self._alert_history.pop(symbol, None)
        else:
            self._alert_history.clear()

    async def health_check(self) -> bool:
        """Check if analyzer is operational."""
        return True


# =============================================================================
# SECTION 5: INTEGRATION POINTS WITH EXISTING DATAMANAGER
# =============================================================================


class DataManagerIntegration:
    """
    Integration specification for EnhancedMoneyFlowAnalyzer with DataManager.

    The DataManager needs the following new methods to support enhanced analysis:
    """

    async def get_realtime_trades(
        self,
        symbol: str,
        since: datetime,
    ) -> pd.DataFrame:
        """
        Get real-time trade data.

        Required columns:
        - timestamp: Trade timestamp
        - price: Execution price
        - size: Number of shares
        - venue: Exchange or dark pool venue
        - side: 'buy' or 'sell' (if available)

        Args:
            symbol: Stock symbol
            since: Get trades since this time

        Returns:
            DataFrame with trade data
        """
        ...

    async def get_dark_pool_prints(
        self,
        symbol: str,
        lookback_days: int = 5,
    ) -> pd.DataFrame:
        """
        Get dark pool print data.

        Required columns:
        - timestamp: Print timestamp
        - price: Execution price
        - size: Number of shares
        - venue: Dark pool venue name
        - premium_discount: Premium/discount to NBBO

        Args:
            symbol: Stock symbol
            lookback_days: Days of data

        Returns:
            DataFrame with dark pool prints
        """
        ...

    async def get_institutional_flow(
        self,
        symbol: str,
        lookback_days: int = 20,
    ) -> pd.DataFrame:
        """
        Get institutional money flow data.

        Required columns:
        - date: Date
        - institutional_buys: Buy volume from institutions
        - institutional_sells: Sell volume from institutions
        - net_flow: Net institutional flow
        - num_buyers: Number of institutional buyers
        - num_sellers: Number of institutional sellers

        Args:
            symbol: Stock symbol
            lookback_days: Days of data

        Returns:
            DataFrame with institutional flow
        """
        ...

    async def get_smart_money_managers(
        self,
        min_aum: float = 1_000_000_000,
    ) -> pd.DataFrame:
        """
        Get list of smart money managers.

        Required columns:
        - cik: SEC CIK identifier
        - name: Manager name
        - aum: Assets under management
        - performance_1y: 1-year performance
        - performance_3y: 3-year performance
        - sharpe_ratio: Sharpe ratio

        Args:
            min_aum: Minimum AUM filter

        Returns:
            DataFrame with smart money managers
        """
        ...


# =============================================================================
# SECTION 6: API ENDPOINT SPECIFICATIONS
# =============================================================================


class APIEndpointSpec:
    """
    Specification for new API endpoints in stanley/api/main.py.
    """

    # GET /api/money-flow/dark-pool/{symbol}
    # Returns: Dark pool analysis for symbol

    # GET /api/money-flow/dark-pool/{symbol}/alerts
    # Returns: Dark pool alerts for symbol

    # GET /api/money-flow/block-trades/{symbol}
    # Returns: Block trade analysis for symbol

    # GET /api/money-flow/sector-rotation
    # Returns: Current sector rotation signal

    # GET /api/money-flow/smart-money/{symbol}
    # Returns: Smart money activity for symbol

    # GET /api/money-flow/smart-money/conviction
    # Returns: High conviction smart money picks

    # GET /api/money-flow/unusual-volume
    # Query params: universe (comma-separated symbols)
    # Returns: Unusual volume events

    # GET /api/money-flow/momentum/{symbol}
    # Returns: Flow momentum indicator for symbol

    # GET /api/money-flow/momentum/leaders
    # Query params: universe (comma-separated symbols), top_n
    # Returns: Top flow momentum stocks

    # GET /api/money-flow/comprehensive/{symbol}
    # Returns: Comprehensive flow analysis for symbol

    # GET /api/money-flow/dashboard
    # Query params: universe (comma-separated symbols)
    # Returns: Market flow dashboard

    # GET /api/money-flow/alerts
    # Query params: symbol, alert_type, min_severity
    # Returns: Active alerts

    # WebSocket /ws/money-flow/alerts
    # Streams real-time alerts
    pass


# =============================================================================
# SECTION 7: EVENT-DRIVEN ALERT PATTERN DESIGN
# =============================================================================


class AlertEventBus:
    """
    Event bus for distributing alerts to subscribers.

    Implements publish-subscribe pattern for decoupled alert handling.
    """

    def __init__(self):
        self._subscribers: Dict[AlertType, List[Callable]] = {}
        self._global_subscribers: List[Callable] = []

    def subscribe(
        self,
        alert_type: Optional[AlertType],
        callback: Callable[[MoneyFlowAlert], None],
    ) -> None:
        """
        Subscribe to alerts.

        Args:
            alert_type: Specific alert type (None for all)
            callback: Function to call with alert
        """
        if alert_type is None:
            self._global_subscribers.append(callback)
        else:
            if alert_type not in self._subscribers:
                self._subscribers[alert_type] = []
            self._subscribers[alert_type].append(callback)

    async def publish(
        self,
        alert: MoneyFlowAlert,
    ) -> None:
        """
        Publish alert to subscribers.

        Args:
            alert: Alert to publish
        """
        # Notify type-specific subscribers
        if alert.alert_type in self._subscribers:
            for callback in self._subscribers[alert.alert_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(alert)
                    else:
                        callback(alert)
                except Exception:
                    pass  # Log error

        # Notify global subscribers
        for callback in self._global_subscribers:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception:
                pass  # Log error


# =============================================================================
# SECTION 8: IMPLEMENTATION PRIORITY AND DEPENDENCIES
# =============================================================================

"""
IMPLEMENTATION PRIORITY:

Phase 1 - Core Infrastructure (Week 1):
    1. Data structures (dataclasses, enums, TypedDicts)
    2. AlertConfig and threshold system
    3. Basic alert generation and storage
    4. Integration with existing MoneyFlowAnalyzer

Phase 2 - Detection Algorithms (Week 2):
    1. Unusual volume detection
    2. Block trade detection and classification
    3. Dark pool analysis (using existing get_dark_pool_activity)
    4. Flow momentum calculation

Phase 3 - Advanced Analysis (Week 3):
    1. Sector rotation signal generation
    2. Smart money tracking aggregation
    3. Flow divergence detection
    4. Comprehensive analysis methods

Phase 4 - Real-time Capabilities (Week 4):
    1. Alert event bus and streaming
    2. WebSocket endpoint for alerts
    3. Real-time monitoring loop
    4. Alert handlers (webhook, notification)

Phase 5 - API and Integration (Week 5):
    1. New API endpoints
    2. DataManager integration methods
    3. GUI integration (stanley-gui)
    4. Documentation and testing

DEPENDENCIES:
    - pandas >= 1.5.0
    - numpy >= 1.24.0
    - asyncio (stdlib)
    - dataclasses (stdlib)
    - typing (stdlib)

EXTERNAL DATA SOURCES (for full functionality):
    - Dark pool data: FINRA ADF, IEX, or specialized providers
    - Block trade data: TAQ or consolidated tape
    - 13F data: SEC EDGAR (already integrated)
    - Real-time quotes: OpenBB or specialized feed
"""

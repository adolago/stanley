"""
Money Flow Actor for NautilusTrader

Wraps Stanley's MoneyFlowAnalyzer in a NautilusTrader Actor interface
to provide real-time money flow analysis and signal generation.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import deque

import numpy as np
import pandas as pd

# NautilusTrader imports
from nautilus_trader.config import ActorConfig
from nautilus_trader.core.datetime import unix_nanos_to_dt
from nautilus_trader.core.uuid import UUID4
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.events import Event
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.trading.strategy import Actor

# Stanley imports
from stanley.analytics.money_flow import MoneyFlowAnalyzer


logger = logging.getLogger(__name__)


@dataclass
class MoneyFlowSignalEvent(Event):
    """
    Custom event emitted when a significant money flow signal is detected.
    """

    symbol: str
    signal_type: str  # 'accumulation', 'distribution', 'dark_pool', 'smart_money'
    signal_strength: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    money_flow_score: float
    institutional_sentiment: float
    smart_money_activity: float
    dark_pool_signal: Optional[int]  # -1, 0, 1
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> UUID4:
        return UUID4()

    @property
    def ts_event(self) -> int:
        return int(self.timestamp.timestamp() * 1e9)

    @property
    def ts_init(self) -> int:
        return int(datetime.now().timestamp() * 1e9)


class MoneyFlowActorConfig(ActorConfig):
    """
    Configuration for the MoneyFlowActor.
    """

    # Symbols to monitor
    symbols: List[str] = field(default_factory=list)

    # Analysis parameters
    lookback_bars: int = 20  # Number of bars for analysis
    update_frequency: int = 1  # Update every N bars

    # Dark pool analysis
    enable_dark_pool: bool = True
    dark_pool_lookback_days: int = 20

    # Signal thresholds
    signal_threshold: float = 0.3  # Minimum signal strength to emit event
    confidence_threshold: float = 0.5  # Minimum confidence to emit event

    # Sector ETFs for sector flow analysis
    sector_etfs: List[str] = field(
        default_factory=lambda: [
            "XLK",
            "XLF",
            "XLE",
            "XLV",
            "XLI",
            "XLP",
            "XLY",
            "XLB",
            "XLU",
            "XLRE",
            "XLC",
        ]
    )

    # Async data refresh interval (seconds)
    data_refresh_interval: int = 300  # 5 minutes


class MoneyFlowActor(Actor):
    """
    NautilusTrader Actor that wraps Stanley's MoneyFlowAnalyzer.

    Provides real-time money flow analysis, dark pool tracking,
    and emits custom events for institutional signals.
    """

    def __init__(self, config: MoneyFlowActorConfig):
        """
        Initialize the MoneyFlowActor.

        Args:
            config: Actor configuration
        """
        super().__init__(config)

        self._config = config
        self._analyzer = MoneyFlowAnalyzer()

        # Bar data buffers per instrument
        self._bar_buffers: Dict[InstrumentId, deque] = {}
        self._bar_counts: Dict[InstrumentId, int] = {}

        # Cached analysis results
        self._flow_cache: Dict[str, Dict] = {}
        self._dark_pool_cache: Dict[str, pd.DataFrame] = {}
        self._sector_flow_cache: Optional[pd.DataFrame] = None

        # Last analysis timestamps
        self._last_flow_analysis: Dict[str, datetime] = {}
        self._last_sector_analysis: Optional[datetime] = None

        # Running state
        self._is_running = False
        self._refresh_task: Optional[asyncio.Task] = None

    def on_start(self) -> None:
        """
        Called when the actor starts.
        Subscribe to bar data for configured symbols.
        """
        logger.info(f"MoneyFlowActor starting with {len(self._config.symbols)} symbols")

        self._is_running = True

        # Initialize buffers for each symbol
        for symbol in self._config.symbols:
            instrument_id = InstrumentId.from_str(symbol)
            self._bar_buffers[instrument_id] = deque(
                maxlen=self._config.lookback_bars * 2
            )
            self._bar_counts[instrument_id] = 0

            # Subscribe to bar data
            # Note: In production, you'd specify the bar type
            self.subscribe_bars(BarType.from_str(f"{symbol}-1-DAY-LAST-EXTERNAL"))

        # Start async data refresh task
        self._refresh_task = asyncio.create_task(self._periodic_data_refresh())

        # Initial sector flow analysis
        self._update_sector_flow()

        logger.info("MoneyFlowActor started successfully")

    def on_stop(self) -> None:
        """
        Called when the actor stops.
        """
        self._is_running = False

        if self._refresh_task:
            self._refresh_task.cancel()

        logger.info("MoneyFlowActor stopped")

    def on_bar(self, bar: Bar) -> None:
        """
        Process new bar data and update money flow analysis.

        Args:
            bar: New bar data
        """
        instrument_id = bar.bar_type.instrument_id
        symbol = str(instrument_id.symbol)

        # Add bar to buffer
        if instrument_id not in self._bar_buffers:
            self._bar_buffers[instrument_id] = deque(
                maxlen=self._config.lookback_bars * 2
            )
            self._bar_counts[instrument_id] = 0

        self._bar_buffers[instrument_id].append(bar)
        self._bar_counts[instrument_id] += 1

        # Check if we should update analysis
        if self._bar_counts[instrument_id] % self._config.update_frequency != 0:
            return

        # Only analyze if we have enough data
        if len(self._bar_buffers[instrument_id]) < self._config.lookback_bars:
            return

        # Perform money flow analysis
        self._analyze_equity_flow(symbol, bar)

    def on_event(self, event: Event) -> None:
        """
        Handle incoming events.

        Args:
            event: The event to process
        """
        # Handle custom events or order events if needed
        pass

    def _analyze_equity_flow(self, symbol: str, latest_bar: Bar) -> None:
        """
        Analyze money flow for a single equity and emit signals.

        Args:
            symbol: Stock symbol
            latest_bar: Most recent bar data
        """
        try:
            # Get money flow analysis from Stanley
            flow_analysis = self._analyzer.analyze_equity_flow(
                symbol=symbol, lookback_days=self._config.lookback_bars
            )

            # Get dark pool analysis if enabled
            dark_pool_signal = None
            if self._config.enable_dark_pool:
                dark_pool_data = self._analyzer.get_dark_pool_activity(
                    symbol=symbol, lookback_days=self._config.dark_pool_lookback_days
                )
                self._dark_pool_cache[symbol] = dark_pool_data

                if (
                    not dark_pool_data.empty
                    and "dark_pool_signal" in dark_pool_data.columns
                ):
                    dark_pool_signal = int(dark_pool_data["dark_pool_signal"].iloc[-1])

            # Cache the analysis
            self._flow_cache[symbol] = flow_analysis
            self._last_flow_analysis[symbol] = datetime.now()

            # Calculate signal strength and type
            signal_strength = flow_analysis.get("money_flow_score", 0.0)
            confidence = flow_analysis.get("confidence", 0.0)

            # Determine signal type
            signal_type = self._determine_signal_type(flow_analysis, dark_pool_signal)

            # Emit event if thresholds are met
            if (
                abs(signal_strength) >= self._config.signal_threshold
                and confidence >= self._config.confidence_threshold
            ):

                event = MoneyFlowSignalEvent(
                    symbol=symbol,
                    signal_type=signal_type,
                    signal_strength=signal_strength,
                    confidence=confidence,
                    money_flow_score=flow_analysis.get("money_flow_score", 0.0),
                    institutional_sentiment=flow_analysis.get(
                        "institutional_sentiment", 0.0
                    ),
                    smart_money_activity=flow_analysis.get("smart_money_activity", 0.0),
                    dark_pool_signal=dark_pool_signal,
                    timestamp=unix_nanos_to_dt(latest_bar.ts_event),
                    metadata={
                        "accumulation_distribution": flow_analysis.get(
                            "accumulation_distribution", 0.0
                        ),
                        "short_pressure": flow_analysis.get("short_pressure", 0.0),
                    },
                )

                self.publish_event(event)
                logger.info(
                    f"Emitted MoneyFlowSignalEvent for {symbol}: {signal_type} "
                    f"(strength={signal_strength:.2f}, confidence={confidence:.2f})"
                )

        except Exception as e:
            logger.error(f"Error analyzing money flow for {symbol}: {e}")

    def _determine_signal_type(
        self, flow_analysis: Dict, dark_pool_signal: Optional[int]
    ) -> str:
        """
        Determine the type of money flow signal.

        Args:
            flow_analysis: Money flow analysis results
            dark_pool_signal: Dark pool signal (-1, 0, 1)

        Returns:
            Signal type string
        """
        acc_dist = flow_analysis.get("accumulation_distribution", 0.0)
        smart_money = flow_analysis.get("smart_money_activity", 0.0)

        # Check dark pool signal first (high priority)
        if dark_pool_signal is not None and dark_pool_signal != 0:
            return "dark_pool"

        # Check smart money activity
        if abs(smart_money) > 0.5:
            return "smart_money"

        # Check accumulation/distribution
        if acc_dist > 0.3:
            return "accumulation"
        elif acc_dist < -0.3:
            return "distribution"

        return "neutral"

    def _update_sector_flow(self) -> None:
        """
        Update sector flow analysis.
        """
        try:
            self._sector_flow_cache = self._analyzer.analyze_sector_flow(
                sectors=self._config.sector_etfs, lookback_days=63  # 3 months
            )
            self._last_sector_analysis = datetime.now()

            logger.debug("Updated sector flow analysis")

        except Exception as e:
            logger.error(f"Error updating sector flow: {e}")

    async def _periodic_data_refresh(self) -> None:
        """
        Periodically refresh data that requires async fetching.
        """
        while self._is_running:
            try:
                await asyncio.sleep(self._config.data_refresh_interval)

                # Refresh sector flow analysis
                self._update_sector_flow()

                logger.debug("Completed periodic data refresh")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic data refresh: {e}")

    # Public API methods

    def get_flow_analysis(self, symbol: str) -> Optional[Dict]:
        """
        Get the latest money flow analysis for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Money flow analysis dictionary or None
        """
        return self._flow_cache.get(symbol)

    def get_dark_pool_analysis(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Get the latest dark pool analysis for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Dark pool analysis DataFrame or None
        """
        return self._dark_pool_cache.get(symbol)

    def get_sector_flow(self) -> Optional[pd.DataFrame]:
        """
        Get the latest sector flow analysis.

        Returns:
            Sector flow analysis DataFrame or None
        """
        return self._sector_flow_cache

    def get_signal_strength(self, symbol: str) -> float:
        """
        Get the current signal strength for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Signal strength (-1.0 to 1.0)
        """
        analysis = self._flow_cache.get(symbol)
        if analysis:
            return analysis.get("money_flow_score", 0.0)
        return 0.0

    def get_confidence(self, symbol: str) -> float:
        """
        Get the current confidence score for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Confidence score (0.0 to 1.0)
        """
        analysis = self._flow_cache.get(symbol)
        if analysis:
            return analysis.get("confidence", 0.0)
        return 0.0

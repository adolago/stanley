"""
Institutional Actor for NautilusTrader

Wraps Stanley's InstitutionalAnalyzer in a NautilusTrader Actor interface
to provide real-time institutional positioning analysis and 13F tracking.
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
from stanley.analytics.institutional import InstitutionalAnalyzer


logger = logging.getLogger(__name__)


@dataclass
class InstitutionalSignalEvent(Event):
    """
    Custom event emitted when significant institutional activity is detected.
    """

    symbol: str
    signal_type: str  # 'accumulation', 'distribution', '13f_change', 'smart_money'
    signal_strength: float  # -1.0 to 1.0
    confidence: float  # 0.0 to 1.0
    institutional_ownership: float
    ownership_trend: float
    smart_money_score: float
    concentration_risk: float
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


@dataclass
class Filing13FEvent(Event):
    """
    Event emitted when new 13F filing data is processed.
    """

    manager_cik: str
    manager_name: str
    filing_date: datetime
    new_positions: List[str]
    closed_positions: List[str]
    significant_increases: List[Dict[str, Any]]
    significant_decreases: List[Dict[str, Any]]
    timestamp: datetime

    @property
    def id(self) -> UUID4:
        return UUID4()

    @property
    def ts_event(self) -> int:
        return int(self.timestamp.timestamp() * 1e9)

    @property
    def ts_init(self) -> int:
        return int(datetime.now().timestamp() * 1e9)


class InstitutionalActorConfig(ActorConfig):
    """
    Configuration for the InstitutionalActor.
    """

    # Universe of symbols to monitor
    universe: List[str] = field(default_factory=list)

    # Institutional managers to track (by CIK)
    tracked_managers: List[str] = field(default_factory=list)

    # Minimum AUM for smart money tracking
    minimum_aum: float = 1e9  # $1 billion

    # Analysis parameters
    update_frequency: int = 5  # Update every N bars
    lookback_bars: int = 20

    # Signal thresholds
    signal_threshold: float = 0.3
    confidence_threshold: float = 0.5
    ownership_change_threshold: float = 0.05  # 5% change is significant

    # 13F filing check interval (seconds)
    filing_check_interval: int = 3600  # 1 hour

    # Async data refresh interval (seconds)
    holdings_refresh_interval: int = 900  # 15 minutes


class InstitutionalActor(Actor):
    """
    NautilusTrader Actor that wraps Stanley's InstitutionalAnalyzer.

    Provides real-time institutional positioning analysis, 13F tracking,
    and smart money monitoring with custom event emission.
    """

    def __init__(self, config: InstitutionalActorConfig):
        """
        Initialize the InstitutionalActor.

        Args:
            config: Actor configuration
        """
        super().__init__(config)

        self._config = config
        self._analyzer = InstitutionalAnalyzer()

        # Bar data buffers per instrument
        self._bar_buffers: Dict[InstrumentId, deque] = {}
        self._bar_counts: Dict[InstrumentId, int] = {}

        # Cached analysis results
        self._holdings_cache: Dict[str, Dict] = {}
        self._sentiment_cache: Optional[Dict] = None
        self._smart_money_cache: Optional[pd.DataFrame] = None
        self._13f_cache: Dict[str, pd.DataFrame] = {}

        # Last analysis timestamps
        self._last_holdings_update: Dict[str, datetime] = {}
        self._last_sentiment_update: Optional[datetime] = None
        self._last_13f_check: Optional[datetime] = None

        # Processed 13F filings to avoid duplicates
        self._processed_filings: set = set()

        # Running state
        self._is_running = False
        self._refresh_task: Optional[asyncio.Task] = None
        self._filing_task: Optional[asyncio.Task] = None

    def on_start(self) -> None:
        """
        Called when the actor starts.
        Subscribe to bar data and start background tasks.
        """
        logger.info(
            f"InstitutionalActor starting with {len(self._config.universe)} symbols"
        )

        self._is_running = True

        # Initialize buffers for each symbol
        for symbol in self._config.universe:
            instrument_id = InstrumentId.from_str(symbol)
            self._bar_buffers[instrument_id] = deque(
                maxlen=self._config.lookback_bars * 2
            )
            self._bar_counts[instrument_id] = 0

            # Subscribe to bar data
            self.subscribe_bars(BarType.from_str(f"{symbol}-1-DAY-LAST-EXTERNAL"))

        # Start async background tasks
        self._refresh_task = asyncio.create_task(self._periodic_holdings_refresh())
        self._filing_task = asyncio.create_task(self._periodic_13f_check())

        # Initial data load
        self._update_universe_sentiment()
        self._update_smart_money_tracking()

        logger.info("InstitutionalActor started successfully")

    def on_stop(self) -> None:
        """
        Called when the actor stops.
        """
        self._is_running = False

        if self._refresh_task:
            self._refresh_task.cancel()
        if self._filing_task:
            self._filing_task.cancel()

        logger.info("InstitutionalActor stopped")

    def on_bar(self, bar: Bar) -> None:
        """
        Process new bar data and update institutional analysis.

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

        # Perform institutional analysis
        self._analyze_institutional_holdings(symbol, bar)

    def on_event(self, event: Event) -> None:
        """
        Handle incoming events.

        Args:
            event: The event to process
        """
        # Handle custom events if needed
        pass

    def _analyze_institutional_holdings(self, symbol: str, latest_bar: Bar) -> None:
        """
        Analyze institutional holdings for a symbol and emit signals.

        Args:
            symbol: Stock symbol
            latest_bar: Most recent bar data
        """
        try:
            # Get institutional holdings from Stanley
            holdings = self._analyzer.get_holdings(symbol)

            # Cache the holdings
            previous_holdings = self._holdings_cache.get(symbol)
            self._holdings_cache[symbol] = holdings
            self._last_holdings_update[symbol] = datetime.now()

            # Calculate signal strength based on institutional metrics
            signal_strength = self._calculate_signal_strength(
                holdings, previous_holdings
            )
            confidence = self._calculate_confidence(holdings)

            # Determine signal type
            signal_type = self._determine_signal_type(holdings, previous_holdings)

            # Emit event if thresholds are met
            if (
                abs(signal_strength) >= self._config.signal_threshold
                and confidence >= self._config.confidence_threshold
            ):

                event = InstitutionalSignalEvent(
                    symbol=symbol,
                    signal_type=signal_type,
                    signal_strength=signal_strength,
                    confidence=confidence,
                    institutional_ownership=holdings.get(
                        "institutional_ownership", 0.0
                    ),
                    ownership_trend=holdings.get("ownership_trend", 0.0),
                    smart_money_score=holdings.get("smart_money_score", 0.0),
                    concentration_risk=holdings.get("concentration_risk", 0.0),
                    timestamp=unix_nanos_to_dt(latest_bar.ts_event),
                    metadata={
                        "number_of_institutions": holdings.get(
                            "number_of_institutions", 0
                        ),
                        "top_holders": self._serialize_top_holders(
                            holdings.get("top_holders")
                        ),
                    },
                )

                self.publish_event(event)
                logger.info(
                    f"Emitted InstitutionalSignalEvent for {symbol}: {signal_type} "
                    f"(strength={signal_strength:.2f}, confidence={confidence:.2f})"
                )

        except Exception as e:
            logger.error(f"Error analyzing institutional holdings for {symbol}: {e}")

    def _calculate_signal_strength(
        self, holdings: Dict, previous_holdings: Optional[Dict]
    ) -> float:
        """
        Calculate signal strength based on holdings changes.

        Args:
            holdings: Current holdings data
            previous_holdings: Previous holdings data

        Returns:
            Signal strength (-1.0 to 1.0)
        """
        # Base signal on smart money score
        smart_money = holdings.get("smart_money_score", 0.0)

        # Add ownership trend contribution
        ownership_trend = holdings.get("ownership_trend", 0.0)

        # If we have previous holdings, factor in the change
        change_factor = 0.0
        if previous_holdings:
            prev_ownership = previous_holdings.get("institutional_ownership", 0.0)
            curr_ownership = holdings.get("institutional_ownership", 0.0)
            ownership_change = curr_ownership - prev_ownership

            if abs(ownership_change) > self._config.ownership_change_threshold:
                change_factor = np.sign(ownership_change) * 0.5

        # Combine factors
        signal = 0.4 * smart_money + 0.3 * ownership_trend + 0.3 * change_factor

        return max(-1.0, min(1.0, signal))

    def _calculate_confidence(self, holdings: Dict) -> float:
        """
        Calculate confidence score based on data quality.

        Args:
            holdings: Holdings data

        Returns:
            Confidence score (0.0 to 1.0)
        """
        # Base confidence on number of institutions
        institution_count = holdings.get("number_of_institutions", 0)
        count_confidence = min(1.0, institution_count / 100)

        # Factor in concentration risk (lower is better for confidence)
        concentration_risk = holdings.get("concentration_risk", 0.5)
        concentration_confidence = 1.0 - concentration_risk

        # Factor in ownership level
        ownership = holdings.get("institutional_ownership", 0.0)
        ownership_confidence = min(
            1.0, ownership / 0.5
        )  # 50% ownership = full confidence

        # Combine factors
        confidence = (
            0.4 * count_confidence
            + 0.3 * concentration_confidence
            + 0.3 * ownership_confidence
        )

        return max(0.0, min(1.0, confidence))

    def _determine_signal_type(
        self, holdings: Dict, previous_holdings: Optional[Dict]
    ) -> str:
        """
        Determine the type of institutional signal.

        Args:
            holdings: Current holdings data
            previous_holdings: Previous holdings data

        Returns:
            Signal type string
        """
        smart_money = holdings.get("smart_money_score", 0.0)
        ownership_trend = holdings.get("ownership_trend", 0.0)

        # Strong smart money signal
        if abs(smart_money) > 0.5:
            return "smart_money"

        # Check for significant ownership changes
        if previous_holdings:
            prev_ownership = previous_holdings.get("institutional_ownership", 0.0)
            curr_ownership = holdings.get("institutional_ownership", 0.0)
            ownership_change = curr_ownership - prev_ownership

            if ownership_change > self._config.ownership_change_threshold:
                return "accumulation"
            elif ownership_change < -self._config.ownership_change_threshold:
                return "distribution"

        # Use ownership trend
        if ownership_trend > 0.3:
            return "accumulation"
        elif ownership_trend < -0.3:
            return "distribution"

        return "neutral"

    def _serialize_top_holders(self, top_holders: Any) -> List[Dict]:
        """
        Serialize top holders for event metadata.

        Args:
            top_holders: Top holders DataFrame or similar

        Returns:
            List of dictionaries
        """
        if top_holders is None:
            return []

        if isinstance(top_holders, pd.DataFrame):
            return top_holders.to_dict("records")

        return []

    def _update_universe_sentiment(self) -> None:
        """
        Update institutional sentiment for the entire universe.
        """
        try:
            self._sentiment_cache = self._analyzer.get_institutional_sentiment(
                universe=self._config.universe
            )
            self._last_sentiment_update = datetime.now()

            logger.debug("Updated universe institutional sentiment")

        except Exception as e:
            logger.error(f"Error updating universe sentiment: {e}")

    def _update_smart_money_tracking(self) -> None:
        """
        Update smart money tracking data.
        """
        try:
            self._smart_money_cache = self._analyzer.track_smart_money(
                minimum_aum=self._config.minimum_aum
            )
            logger.debug("Updated smart money tracking")

        except Exception as e:
            logger.error(f"Error updating smart money tracking: {e}")

    def _check_13f_filings(self) -> None:
        """
        Check for new 13F filings from tracked managers.
        """
        for manager_cik in self._config.tracked_managers:
            try:
                # Get 13F changes
                changes = self._analyzer.analyze_13f_changes(manager_cik)

                # Cache the changes
                self._13f_cache[manager_cik] = changes

                # Create a filing identifier
                filing_id = f"{manager_cik}_{datetime.now().strftime('%Y%m%d')}"

                if filing_id not in self._processed_filings:
                    self._processed_filings.add(filing_id)

                    # Extract significant changes
                    new_positions = changes[changes["change_type"] == "new"][
                        "symbol"
                    ].tolist()
                    closed_positions = changes[changes["change_type"] == "closed"][
                        "symbol"
                    ].tolist()

                    significant_increases = changes[changes["change_percentage"] > 0.1][
                        ["symbol", "change_percentage", "value_change"]
                    ].to_dict("records")

                    significant_decreases = changes[
                        changes["change_percentage"] < -0.1
                    ][["symbol", "change_percentage", "value_change"]].to_dict(
                        "records"
                    )

                    # Emit 13F event if there are significant changes
                    if (
                        new_positions
                        or closed_positions
                        or significant_increases
                        or significant_decreases
                    ):
                        event = Filing13FEvent(
                            manager_cik=manager_cik,
                            manager_name=self._get_manager_name(manager_cik),
                            filing_date=datetime.now(),
                            new_positions=new_positions,
                            closed_positions=closed_positions,
                            significant_increases=significant_increases,
                            significant_decreases=significant_decreases,
                            timestamp=datetime.now(),
                        )

                        self.publish_event(event)
                        logger.info(f"Emitted Filing13FEvent for manager {manager_cik}")

            except Exception as e:
                logger.error(f"Error checking 13F filing for {manager_cik}: {e}")

    def _get_manager_name(self, manager_cik: str) -> str:
        """
        Get manager name from CIK.

        Args:
            manager_cik: SEC CIK identifier

        Returns:
            Manager name or CIK if not found
        """
        # Known major managers
        known_managers = {
            "0000102909": "Vanguard Group",
            "0001390777": "BlackRock",
            "0000093751": "State Street",
            "0000315066": "Fidelity",
            "0000080227": "T. Rowe Price",
        }

        return known_managers.get(manager_cik, manager_cik)

    async def _periodic_holdings_refresh(self) -> None:
        """
        Periodically refresh holdings data.
        """
        while self._is_running:
            try:
                await asyncio.sleep(self._config.holdings_refresh_interval)

                # Update universe sentiment
                self._update_universe_sentiment()

                # Update smart money tracking
                self._update_smart_money_tracking()

                logger.debug("Completed periodic holdings refresh")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic holdings refresh: {e}")

    async def _periodic_13f_check(self) -> None:
        """
        Periodically check for new 13F filings.
        """
        while self._is_running:
            try:
                await asyncio.sleep(self._config.filing_check_interval)

                # Check 13F filings
                self._check_13f_filings()
                self._last_13f_check = datetime.now()

                logger.debug("Completed periodic 13F check")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic 13F check: {e}")

    # Public API methods

    def get_holdings(self, symbol: str) -> Optional[Dict]:
        """
        Get the latest holdings data for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Holdings dictionary or None
        """
        return self._holdings_cache.get(symbol)

    def get_universe_sentiment(self) -> Optional[Dict]:
        """
        Get the latest universe sentiment data.

        Returns:
            Sentiment dictionary or None
        """
        return self._sentiment_cache

    def get_smart_money_activity(self) -> Optional[pd.DataFrame]:
        """
        Get the latest smart money tracking data.

        Returns:
            Smart money DataFrame or None
        """
        return self._smart_money_cache

    def get_13f_changes(self, manager_cik: str) -> Optional[pd.DataFrame]:
        """
        Get the latest 13F changes for a manager.

        Args:
            manager_cik: SEC CIK identifier

        Returns:
            13F changes DataFrame or None
        """
        return self._13f_cache.get(manager_cik)

    def get_signal_strength(self, symbol: str) -> float:
        """
        Get the current signal strength for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Signal strength (-1.0 to 1.0)
        """
        holdings = self._holdings_cache.get(symbol)
        if holdings:
            return self._calculate_signal_strength(holdings, None)
        return 0.0

    def get_confidence(self, symbol: str) -> float:
        """
        Get the current confidence score for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Confidence score (0.0 to 1.0)
        """
        holdings = self._holdings_cache.get(symbol)
        if holdings:
            return self._calculate_confidence(holdings)
        return 0.0

    def get_institutional_ownership(self, symbol: str) -> float:
        """
        Get the institutional ownership percentage for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Ownership percentage (0.0 to 1.0)
        """
        holdings = self._holdings_cache.get(symbol)
        if holdings:
            return holdings.get("institutional_ownership", 0.0)
        return 0.0

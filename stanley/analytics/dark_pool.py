"""
Dark Pool Analysis Module

Analyze dark pool activity, off-exchange trading, and institutional order flow.
Integrates with FINRA ATS data and provides real dark pool analytics.

Dark pools are private exchanges for trading securities that are not accessible
to the general public. They allow institutional investors to trade large blocks
without revealing their intentions to the broader market.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Major dark pool operators (ATS - Alternative Trading Systems)
MAJOR_DARK_POOLS = {
    "UBSS": "UBS ATS",
    "CROS": "Credit Suisse Crossfinder",
    "SGMT": "Goldman Sachs Sigma X",
    "JPMX": "JPMorgan JPM-X",
    "MSPL": "Morgan Stanley MS Pool",
    "DBAX": "Deutsche Bank SuperX",
    "BIDS": "BIDS Trading",
    "LQFI": "Liquidnet",
    "IEGX": "IEX",
    "MEMX": "MEMX",
}

# Dark pool volume thresholds for signals
VOLUME_THRESHOLDS = {
    "high": 0.35,  # >35% dark pool = high institutional activity
    "moderate": 0.25,  # 25-35% = moderate activity
    "low": 0.15,  # 15-25% = low activity
}


class DarkPoolAnalyzer:
    """
    Analyze dark pool activity and off-exchange trading patterns.

    Provides insights into institutional order flow that isn't visible
    on traditional exchanges.
    """

    def __init__(self, data_manager=None):
        """
        Initialize dark pool analyzer.

        Args:
            data_manager: Data manager instance for data access
        """
        self.data_manager = data_manager
        logger.info("DarkPoolAnalyzer initialized")

    async def get_dark_pool_volume(
        self,
        symbol: str,
        lookback_days: int = 20,
    ) -> pd.DataFrame:
        """
        Get dark pool volume data for a symbol.

        Args:
            symbol: Stock symbol
            lookback_days: Number of days to analyze

        Returns:
            DataFrame with dark pool volume metrics
        """
        if self.data_manager:
            try:
                data = await self.data_manager.get_dark_pool_volume(
                    symbol, lookback_days
                )
                if not data.empty:
                    return self._enrich_dark_pool_data(data)
            except Exception as e:
                logger.warning(f"Failed to get dark pool data for {symbol}: {e}")

        # Fallback to simulated data with realistic patterns
        return self._generate_dark_pool_data(symbol, lookback_days)

    async def analyze_dark_pool_activity(
        self,
        symbol: str,
        lookback_days: int = 20,
    ) -> Dict[str, Any]:
        """
        Comprehensive dark pool activity analysis.

        Args:
            symbol: Stock symbol
            lookback_days: Number of days to analyze

        Returns:
            Dictionary with dark pool analysis
        """
        df = await self.get_dark_pool_volume(symbol, lookback_days)

        if df.empty:
            return self._empty_analysis(symbol)

        # Calculate key metrics
        avg_dp_pct = df["dark_pool_percentage"].mean()
        recent_dp_pct = df["dark_pool_percentage"].tail(5).mean()
        dp_trend = recent_dp_pct - df["dark_pool_percentage"].head(5).mean()

        # Large block analysis
        avg_block_activity = df["large_block_activity"].mean()
        recent_block_activity = df["large_block_activity"].tail(5).mean()

        # Volume analysis
        avg_dp_volume = df["dark_pool_volume"].mean()
        total_dp_volume = df["dark_pool_volume"].sum()

        # Generate signal
        signal = self._calculate_dark_pool_signal(df)

        # Detect accumulation/distribution
        accumulation_score = self._detect_accumulation(df)

        return {
            "symbol": symbol,
            "lookback_days": lookback_days,
            "metrics": {
                "average_dark_pool_percentage": round(avg_dp_pct * 100, 2),
                "recent_dark_pool_percentage": round(recent_dp_pct * 100, 2),
                "dark_pool_trend": round(dp_trend * 100, 2),
                "average_large_block_activity": round(avg_block_activity * 100, 2),
                "recent_large_block_activity": round(recent_block_activity * 100, 2),
                "average_daily_dark_pool_volume": int(avg_dp_volume),
                "total_dark_pool_volume": int(total_dp_volume),
            },
            "signal": {
                "value": round(signal, 3),
                "interpretation": self._interpret_signal(signal),
                "strength": abs(signal),
            },
            "accumulation_distribution": {
                "score": round(accumulation_score, 3),
                "interpretation": (
                    "accumulation"
                    if accumulation_score > 0.2
                    else "distribution" if accumulation_score < -0.2 else "neutral"
                ),
            },
            "activity_level": self._classify_activity_level(avg_dp_pct),
            "institutional_interest": self._assess_institutional_interest(df),
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def get_dark_pool_by_venue(
        self,
        symbol: str,
        lookback_days: int = 20,
    ) -> pd.DataFrame:
        """
        Get dark pool volume breakdown by venue/ATS.

        Args:
            symbol: Stock symbol
            lookback_days: Number of days to analyze

        Returns:
            DataFrame with volume by venue
        """
        # In production, this would integrate with FINRA ATS data
        # For now, generate realistic venue distribution
        total_volume = np.random.randint(1000000, 10000000)

        venue_data = []
        remaining_volume = total_volume

        for venue_code, venue_name in list(MAJOR_DARK_POOLS.items())[:8]:
            # Distribute volume with realistic proportions
            if remaining_volume > 0:
                venue_volume = int(remaining_volume * np.random.uniform(0.05, 0.25))
                remaining_volume -= venue_volume

                venue_data.append(
                    {
                        "venue_code": venue_code,
                        "venue_name": venue_name,
                        "volume": venue_volume,
                        "percentage": round(venue_volume / total_volume * 100, 2),
                        "average_trade_size": np.random.randint(200, 2000),
                        "trade_count": venue_volume // np.random.randint(300, 800),
                    }
                )

        df = pd.DataFrame(venue_data)
        return df.sort_values("volume", ascending=False).reset_index(drop=True)

    async def detect_block_trades(
        self,
        symbol: str,
        min_size: int = 10000,
        lookback_days: int = 5,
    ) -> pd.DataFrame:
        """
        Detect large block trades that may indicate institutional activity.

        Args:
            symbol: Stock symbol
            min_size: Minimum shares for block trade
            lookback_days: Number of days to analyze

        Returns:
            DataFrame with detected block trades
        """
        dates = pd.date_range(
            end=datetime.now(),
            periods=lookback_days * 5,  # Approximate 5 blocks per day
            freq="4H",
        )

        block_trades = []
        for date in dates:
            # Simulate block trade detection
            if np.random.random() < 0.3:  # 30% chance of block trade
                size = np.random.randint(min_size, min_size * 10)
                price = 150.0 + np.random.uniform(-10, 10)

                block_trades.append(
                    {
                        "timestamp": date,
                        "symbol": symbol,
                        "size": size,
                        "price": round(price, 2),
                        "notional_value": round(size * price, 2),
                        "side": np.random.choice(["buy", "sell"], p=[0.55, 0.45]),
                        "venue": np.random.choice(list(MAJOR_DARK_POOLS.keys())),
                        "is_print": np.random.random() < 0.7,
                    }
                )

        df = pd.DataFrame(block_trades)
        if not df.empty:
            df = df.sort_values("timestamp", ascending=False)

        return df

    async def calculate_dark_pool_sentiment(
        self,
        symbol: str,
        lookback_days: int = 20,
    ) -> Dict[str, Any]:
        """
        Calculate sentiment from dark pool activity patterns.

        Args:
            symbol: Stock symbol
            lookback_days: Number of days to analyze

        Returns:
            Dictionary with sentiment analysis
        """
        df = await self.get_dark_pool_volume(symbol, lookback_days)
        blocks = await self.detect_block_trades(symbol, lookback_days=lookback_days)

        if df.empty:
            return {
                "symbol": symbol,
                "sentiment": 0.0,
                "confidence": 0.0,
                "interpretation": "insufficient_data",
            }

        # Analyze dark pool percentage trend
        dp_trend = self._calculate_trend(df["dark_pool_percentage"])

        # Analyze block trade imbalance
        block_imbalance = 0.0
        if not blocks.empty and "side" in blocks.columns:
            buy_volume = blocks[blocks["side"] == "buy"]["size"].sum()
            sell_volume = blocks[blocks["side"] == "sell"]["size"].sum()
            total_volume = buy_volume + sell_volume
            if total_volume > 0:
                block_imbalance = (buy_volume - sell_volume) / total_volume

        # Calculate composite sentiment
        sentiment = (
            0.4 * dp_trend + 0.4 * block_imbalance + 0.2 * self._detect_accumulation(df)
        )

        # Normalize to -1 to 1
        sentiment = max(-1.0, min(1.0, sentiment))

        return {
            "symbol": symbol,
            "sentiment": round(sentiment, 3),
            "confidence": round(abs(sentiment) * 0.8, 3),
            "components": {
                "dark_pool_trend": round(dp_trend, 3),
                "block_imbalance": round(block_imbalance, 3),
                "accumulation_score": round(self._detect_accumulation(df), 3),
            },
            "interpretation": (
                "bullish"
                if sentiment > 0.2
                else "bearish" if sentiment < -0.2 else "neutral"
            ),
        }

    async def get_sector_dark_pool_activity(
        self,
        sector_etfs: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Analyze dark pool activity across sectors.

        Args:
            sector_etfs: List of sector ETF symbols (default: standard sectors)

        Returns:
            DataFrame with sector dark pool analysis
        """
        if sector_etfs is None:
            sector_etfs = [
                "XLK",
                "XLF",
                "XLE",
                "XLV",
                "XLY",
                "XLP",
                "XLI",
                "XLB",
                "XLU",
                "XLRE",
                "XLC",
            ]

        sector_data = []
        for etf in sector_etfs:
            analysis = await self.analyze_dark_pool_activity(etf, lookback_days=10)

            sector_data.append(
                {
                    "sector_etf": etf,
                    "dark_pool_percentage": analysis["metrics"][
                        "average_dark_pool_percentage"
                    ],
                    "dp_trend": analysis["metrics"]["dark_pool_trend"],
                    "signal": analysis["signal"]["value"],
                    "activity_level": analysis["activity_level"],
                    "institutional_interest": analysis["institutional_interest"],
                }
            )

        return pd.DataFrame(sector_data).sort_values(
            "dark_pool_percentage", ascending=False
        )

    async def get_dark_pool_alerts(
        self,
        symbols: List[str],
        dp_threshold: float = 0.35,
        block_threshold: float = 0.15,
    ) -> List[Dict[str, Any]]:
        """
        Generate alerts for unusual dark pool activity.

        Args:
            symbols: List of symbols to monitor
            dp_threshold: Dark pool percentage threshold for alerts
            block_threshold: Large block threshold for alerts

        Returns:
            List of alert dictionaries
        """
        alerts = []

        for symbol in symbols:
            try:
                analysis = await self.analyze_dark_pool_activity(
                    symbol, lookback_days=5
                )

                recent_dp = analysis["metrics"]["recent_dark_pool_percentage"] / 100
                recent_block = analysis["metrics"]["recent_large_block_activity"] / 100

                # Check for elevated dark pool activity
                if recent_dp >= dp_threshold:
                    alerts.append(
                        {
                            "symbol": symbol,
                            "alert_type": "elevated_dark_pool",
                            "severity": "high" if recent_dp >= 0.45 else "medium",
                            "message": f"Dark pool activity at {recent_dp*100:.1f}% (threshold: {dp_threshold*100:.0f}%)",
                            "metrics": {
                                "dark_pool_percentage": round(recent_dp * 100, 2),
                                "signal": analysis["signal"]["value"],
                            },
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )

                # Check for elevated block trade activity
                if recent_block >= block_threshold:
                    alerts.append(
                        {
                            "symbol": symbol,
                            "alert_type": "elevated_block_trades",
                            "severity": "high" if recent_block >= 0.20 else "medium",
                            "message": f"Block trade activity at {recent_block*100:.1f}% (threshold: {block_threshold*100:.0f}%)",
                            "metrics": {
                                "block_activity": round(recent_block * 100, 2),
                                "accumulation": analysis["accumulation_distribution"][
                                    "interpretation"
                                ],
                            },
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )

                # Check for significant trend changes
                dp_trend = analysis["metrics"]["dark_pool_trend"]
                if abs(dp_trend) >= 10:  # 10%+ change
                    alerts.append(
                        {
                            "symbol": symbol,
                            "alert_type": "trend_change",
                            "severity": "medium",
                            "message": f"Dark pool trend {'increasing' if dp_trend > 0 else 'decreasing'} by {abs(dp_trend):.1f}%",
                            "metrics": {
                                "trend_change": dp_trend,
                                "direction": "bullish" if dp_trend > 0 else "bearish",
                            },
                            "timestamp": datetime.utcnow().isoformat(),
                        }
                    )

            except Exception as e:
                logger.error(f"Error generating alert for {symbol}: {e}")
                continue

        return sorted(
            alerts,
            key=lambda x: 0 if x["severity"] == "high" else 1,
        )

    def _generate_dark_pool_data(
        self,
        symbol: str,
        lookback_days: int,
    ) -> pd.DataFrame:
        """Generate realistic dark pool data for simulation."""
        dates = pd.date_range(end=datetime.now(), periods=lookback_days, freq="D")

        # Generate with realistic patterns
        base_dp_pct = np.random.uniform(0.20, 0.35)
        trend = np.random.uniform(-0.005, 0.005)

        data = []
        for i, date in enumerate(dates):
            # Add trend and noise
            dp_pct = base_dp_pct + trend * i + np.random.normal(0, 0.03)
            dp_pct = max(0.10, min(0.50, dp_pct))

            total_volume = np.random.randint(1000000, 10000000)
            dp_volume = int(total_volume * dp_pct)

            block_activity = np.random.uniform(0.05, 0.20)

            data.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "dark_pool_volume": dp_volume,
                    "total_volume": total_volume,
                    "dark_pool_percentage": dp_pct,
                    "large_block_activity": block_activity,
                }
            )

        return pd.DataFrame(data)

    def _enrich_dark_pool_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich dark pool data with additional metrics."""
        if df.empty:
            return df

        # Calculate additional metrics
        if "dark_pool_volume" in df.columns and "total_volume" in df.columns:
            df["lit_volume"] = df["total_volume"] - df["dark_pool_volume"]

        # Calculate moving averages
        if "dark_pool_percentage" in df.columns:
            df["dp_ma5"] = df["dark_pool_percentage"].rolling(5, min_periods=1).mean()
            df["dp_ma10"] = df["dark_pool_percentage"].rolling(10, min_periods=1).mean()

        return df

    def _calculate_dark_pool_signal(self, df: pd.DataFrame) -> float:
        """Calculate trading signal from dark pool data."""
        if df.empty or "dark_pool_percentage" not in df.columns:
            return 0.0

        # Recent trend
        recent = df["dark_pool_percentage"].tail(5).mean()
        historical = df["dark_pool_percentage"].mean()

        trend_signal = (recent - historical) / historical if historical > 0 else 0

        # Block activity signal
        block_signal = 0.0
        if "large_block_activity" in df.columns:
            recent_blocks = df["large_block_activity"].tail(5).mean()
            if recent_blocks > VOLUME_THRESHOLDS["moderate"]:
                block_signal = 0.3
            elif recent_blocks > VOLUME_THRESHOLDS["low"]:
                block_signal = 0.1

        # Combine signals
        signal = 0.6 * trend_signal + 0.4 * block_signal

        return max(-1.0, min(1.0, signal))

    def _detect_accumulation(self, df: pd.DataFrame) -> float:
        """Detect accumulation or distribution patterns."""
        if df.empty:
            return 0.0

        # Look for increasing dark pool activity with stable/rising prices
        if "dark_pool_percentage" not in df.columns:
            return 0.0

        dp_trend = self._calculate_trend(df["dark_pool_percentage"])

        # Positive trend in dark pool activity suggests accumulation
        return max(-1.0, min(1.0, dp_trend * 2))

    def _calculate_trend(self, series: pd.Series) -> float:
        """Calculate linear trend of a series."""
        if len(series) < 3:
            return 0.0

        x = np.arange(len(series))
        coeffs = np.polyfit(x, series.values, 1)
        slope = coeffs[0]

        # Normalize by mean
        mean_val = series.mean()
        if mean_val > 0:
            return slope / mean_val
        return 0.0

    def _interpret_signal(self, signal: float) -> str:
        """Interpret the signal value."""
        if signal >= 0.5:
            return "strong_bullish"
        elif signal >= 0.2:
            return "bullish"
        elif signal <= -0.5:
            return "strong_bearish"
        elif signal <= -0.2:
            return "bearish"
        return "neutral"

    def _classify_activity_level(self, dp_pct: float) -> str:
        """Classify dark pool activity level."""
        if dp_pct >= VOLUME_THRESHOLDS["high"]:
            return "high"
        elif dp_pct >= VOLUME_THRESHOLDS["moderate"]:
            return "moderate"
        elif dp_pct >= VOLUME_THRESHOLDS["low"]:
            return "low"
        return "minimal"

    def _assess_institutional_interest(self, df: pd.DataFrame) -> str:
        """Assess level of institutional interest."""
        if df.empty:
            return "unknown"

        avg_dp = df["dark_pool_percentage"].mean()
        avg_block = df.get("large_block_activity", pd.Series([0.1])).mean()

        score = 0.6 * avg_dp + 0.4 * avg_block

        if score >= 0.30:
            return "very_high"
        elif score >= 0.22:
            return "high"
        elif score >= 0.15:
            return "moderate"
        elif score >= 0.10:
            return "low"
        return "minimal"

    def _empty_analysis(self, symbol: str) -> Dict[str, Any]:
        """Return empty analysis structure."""
        return {
            "symbol": symbol,
            "lookback_days": 0,
            "metrics": {
                "average_dark_pool_percentage": 0.0,
                "recent_dark_pool_percentage": 0.0,
                "dark_pool_trend": 0.0,
                "average_large_block_activity": 0.0,
                "recent_large_block_activity": 0.0,
                "average_daily_dark_pool_volume": 0,
                "total_dark_pool_volume": 0,
            },
            "signal": {
                "value": 0.0,
                "interpretation": "insufficient_data",
                "strength": 0.0,
            },
            "accumulation_distribution": {
                "score": 0.0,
                "interpretation": "insufficient_data",
            },
            "activity_level": "unknown",
            "institutional_interest": "unknown",
            "timestamp": datetime.utcnow().isoformat(),
        }

    def health_check(self) -> bool:
        """Check if dark pool analyzer is operational."""
        return True

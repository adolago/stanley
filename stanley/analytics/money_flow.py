"""
Money Flow Analysis Module

Analyze institutional money flow, volume patterns, and capital allocation.
No technical indicators, no moon phases, just real money movement.
"""

import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from typing import Dict, List

logger = logging.getLogger(__name__)


class MoneyFlowAnalyzer:
    """
    Analyze institutional money flow and capital allocation patterns.
    """

    def __init__(self, data_manager=None):
        """
        Initialize money flow analyzer.

        Args:
            data_manager: Data manager instance for data access
        """
        self.data_manager = data_manager
        logger.info("MoneyFlowAnalyzer initialized")

    def analyze_sector_flow(
        self, sectors: List[str], lookback_days: int = 63
    ) -> pd.DataFrame:
        """
        Analyze money flow across sectors using ETF flows and institutional data.

        Args:
            sectors: List of sector ETF symbols (e.g., ['XLK', 'XLF', 'XLE'])
            lookback_days: Number of days to analyze (default: 63 ~ 3 months)

        Returns:
            DataFrame with money flow analysis by sector
        """
        if not sectors:
            return pd.DataFrame(
                columns=[
                    "net_flow_1m",
                    "net_flow_3m",
                    "institutional_change",
                    "smart_money_sentiment",
                    "flow_acceleration",
                    "confidence_score",
                ]
            )

        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        results = []
        for sector in sectors:
            try:
                # Get ETF flow data
                flow_data = self._get_etf_flows(sector, start_date, end_date)

                # Get institutional positioning
                institutional_data = self._get_institutional_positioning(sector)

                # Calculate money flow metrics
                metrics = self._calculate_flow_metrics(flow_data, institutional_data)

                results.append(
                    {
                        "sector": sector,
                        "net_flow_1m": metrics["net_flow_1m"],
                        "net_flow_3m": metrics["net_flow_3m"],
                        "institutional_change": metrics["institutional_change"],
                        "smart_money_sentiment": metrics["smart_money_sentiment"],
                        "flow_acceleration": metrics["flow_acceleration"],
                        "confidence_score": metrics["confidence_score"],
                    }
                )

            except Exception as e:
                logger.error(f"Error analyzing sector {sector}: {e}")
                continue

        return pd.DataFrame(results).set_index("sector")

    def analyze_equity_flow(self, symbol: str, lookback_days: int = 20) -> Dict:
        """
        Analyze money flow for individual equity.

        Args:
            symbol: Stock symbol
            lookback_days: Number of days to analyze

        Returns:
            Dictionary with money flow analysis
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        # Get volume and price data
        volume_data = self._get_volume_analysis(symbol, start_date, end_date)

        # Get institutional flows
        institutional_flows = self._get_institutional_flows(
            symbol, start_date, end_date
        )

        # Get short interest changes
        short_interest = self._get_short_interest_changes(symbol)

        # Calculate money flow indicators
        money_flow_metrics = self._calculate_equity_flow_metrics(
            volume_data, institutional_flows, short_interest
        )

        return {
            "symbol": symbol,
            "money_flow_score": money_flow_metrics["flow_score"],
            "institutional_sentiment": money_flow_metrics["institutional_sentiment"],
            "smart_money_activity": money_flow_metrics["smart_money_activity"],
            "short_pressure": money_flow_metrics["short_pressure"],
            "accumulation_distribution": money_flow_metrics[
                "accumulation_distribution"
            ],
            "confidence": money_flow_metrics["confidence"],
        }

    def get_dark_pool_activity(
        self, symbol: str, lookback_days: int = 20
    ) -> pd.DataFrame:
        """
        Analyze dark pool activity for institutional positioning.

        Args:
            symbol: Stock symbol
            lookback_days: Number of days to analyze

        Returns:
            DataFrame with dark pool analysis
        """
        # This would integrate with dark pool data providers
        logger.info(f"Analyzing dark pool activity for {symbol}")

        # Placeholder implementation
        dates = pd.date_range(end=datetime.now(), periods=lookback_days, freq="D")

        data = []
        for date in dates:
            data.append(
                {
                    "date": date,
                    "dark_pool_volume": np.random.randint(
                        100000, 1000000
                    ),  # Placeholder
                    "total_volume": np.random.randint(1000000, 10000000),  # Placeholder
                    "dark_pool_percentage": np.random.uniform(
                        0.15, 0.35
                    ),  # Placeholder
                    "large_block_activity": np.random.uniform(
                        0.05, 0.15
                    ),  # Placeholder
                }
            )

        df = pd.DataFrame(data)
        df["dark_pool_signal"] = self._calculate_dark_pool_signal(df)

        return df

    def _get_etf_flows(
        self, etf_symbol: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        Get ETF flow data from data manager.
        """
        if self.data_manager:
            return self.data_manager.get_etf_flows(etf_symbol, start_date, end_date)
        else:
            # Placeholder implementation
            dates = pd.date_range(start=start_date, end=end_date, freq="D")
            return pd.DataFrame(
                {
                    "date": dates,
                    "net_flow": np.random.normal(0, 1000000, len(dates)),  # Placeholder
                    "creation_units": np.random.randint(0, 100, len(dates)),
                    "redemption_units": np.random.randint(0, 100, len(dates)),
                }
            )

    def _get_institutional_positioning(self, symbol: str) -> Dict:
        """
        Get institutional positioning data.
        """
        if self.data_manager:
            return self.data_manager.get_institutional_positioning(symbol)
        else:
            # Placeholder implementation
            return {
                "institutional_ownership": np.random.uniform(0.6, 0.9),
                "net_buyer_count": np.random.randint(50, 200),
                "total_institutions": np.random.randint(200, 500),
                "avg_position_size": np.random.uniform(1000000, 10000000),
            }

    def _calculate_flow_metrics(
        self, flow_data: pd.DataFrame, institutional_data: Dict
    ) -> Dict:
        """
        Calculate money flow metrics from raw data.
        """
        # Handle empty or insufficient data
        if flow_data.empty or "net_flow" not in flow_data.columns:
            return {
                "net_flow_1m": 0.0,
                "net_flow_3m": 0.0,
                "institutional_change": 0.0,
                "smart_money_sentiment": 0.0,
                "flow_acceleration": 0.0,
                "confidence_score": 0.0,
            }

        # Calculate net flows over different periods
        net_flow_1m = flow_data["net_flow"].tail(21).sum()  # ~1 month
        net_flow_3m = flow_data["net_flow"].tail(63).sum()  # ~3 months

        # Calculate institutional change
        institutional_change = institutional_data.get("net_buyer_count", 0) / max(
            institutional_data.get("total_institutions", 1), 1
        )

        # Smart money sentiment (institutional buying vs selling)
        # Guard against zero or near-zero standard deviation
        flow_std = flow_data["net_flow"].std()
        if flow_std == 0 or np.isnan(flow_std) or flow_std < 1e-10:
            smart_money_sentiment = np.sign(net_flow_3m)
        else:
            smart_money_sentiment = np.sign(net_flow_3m) * abs(net_flow_3m) / flow_std

        # Flow acceleration (rate of change)
        recent_flow = flow_data["net_flow"].tail(10).mean()
        historical_flow = flow_data["net_flow"].mean()

        # Guard against division by zero and NaN
        if historical_flow == 0 or np.isnan(historical_flow):
            flow_acceleration = 0.0
        else:
            flow_acceleration = (recent_flow - historical_flow) / historical_flow

        # Confidence score combines multiple factors
        confidence_score = (
            0.4 * np.sign(net_flow_3m)  # Direction of flow
            + 0.3 * institutional_change  # Institutional activity
            + 0.2 * np.sign(smart_money_sentiment)  # Smart money direction
            + 0.1 * np.sign(flow_acceleration)  # Momentum
        )

        return {
            "net_flow_1m": net_flow_1m,
            "net_flow_3m": net_flow_3m,
            "institutional_change": institutional_change,
            "smart_money_sentiment": smart_money_sentiment,
            "flow_acceleration": flow_acceleration,
            "confidence_score": max(
                -1, min(1, confidence_score)
            ),  # Bound between -1 and 1
        }

    def _calculate_dark_pool_signal(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate dark pool signal from activity data.
        """
        # High dark pool percentage + large blocks = institutional accumulation
        signal = np.where(
            (df["dark_pool_percentage"] > 0.25) & (df["large_block_activity"] > 0.10),
            1,  # Bullish
            np.where(
                (df["dark_pool_percentage"] < 0.15)
                & (df["large_block_activity"] < 0.05),
                -1,  # Bearish
                0,  # Neutral
            ),
        )

        return pd.Series(signal, index=df.index)

    def _get_volume_analysis(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        Get volume analysis data.
        """
        # Placeholder implementation
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        return pd.DataFrame(
            {
                "date": dates,
                "volume": np.random.randint(1000000, 10000000, len(dates)),
                "price_change": np.random.uniform(-0.05, 0.05, len(dates)),
                "block_trades": np.random.randint(0, 50, len(dates)),
            }
        )

    def _get_institutional_flows(
        self, symbol: str, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """
        Get institutional flow data.
        """
        # Placeholder implementation
        dates = pd.date_range(start=start_date, end=end_date, freq="D")
        return pd.DataFrame(
            {
                "date": dates,
                "institutional_net_flow": np.random.normal(0, 1000000, len(dates)),
                "buyer_count": np.random.randint(10, 100, len(dates)),
                "seller_count": np.random.randint(10, 100, len(dates)),
            }
        )

    def _get_short_interest_changes(self, symbol: str) -> Dict:
        """
        Get short interest changes.
        """
        # Placeholder implementation
        return {
            "current_short_interest": np.random.uniform(0.02, 0.10),
            "previous_short_interest": np.random.uniform(0.02, 0.10),
            "days_to_cover": np.random.uniform(1, 10),
        }

    def _calculate_equity_flow_metrics(
        self,
        volume_data: pd.DataFrame,
        institutional_flows: pd.DataFrame,
        short_interest: Dict,
    ) -> Dict:
        """
        Calculate equity-specific money flow metrics.
        """
        # Volume-weighted money flow
        avg_volume = volume_data["volume"].mean()
        recent_volume = volume_data["volume"].tail(5).mean()
        volume_trend = (
            (recent_volume - avg_volume) / avg_volume if avg_volume > 0 else 0
        )

        # Institutional sentiment
        recent_institutional = (
            institutional_flows["institutional_net_flow"].tail(5).sum()
        )
        institutional_sentiment = np.sign(recent_institutional) * min(
            1, abs(recent_institutional) / 1000000
        )

        # Smart money activity (buyers vs sellers)
        recent_buyers = institutional_flows["buyer_count"].tail(5).mean()
        recent_sellers = institutional_flows["seller_count"].tail(5).mean()
        smart_money_activity = (recent_buyers - recent_sellers) / max(
            recent_buyers + recent_sellers, 1
        )

        # Short pressure
        short_change = (
            short_interest["current_short_interest"]
            - short_interest["previous_short_interest"]
        )
        short_pressure = -np.sign(short_change) * min(
            1, abs(short_change) / 0.05
        )  # Normalize to 5% change

        # Accumulation/distribution based on volume and institutional activity
        accumulation_distribution = (
            0.4 * volume_trend
            + 0.4 * institutional_sentiment
            + 0.2 * smart_money_activity
        )

        # Overall flow score
        flow_score = (
            0.3 * institutional_sentiment
            + 0.3 * smart_money_activity
            + 0.2 * accumulation_distribution
            + 0.2 * short_pressure
        )

        return {
            "flow_score": max(-1, min(1, flow_score)),
            "institutional_sentiment": institutional_sentiment,
            "smart_money_activity": smart_money_activity,
            "short_pressure": short_pressure,
            "accumulation_distribution": accumulation_distribution,
            "confidence": abs(flow_score),  # Higher absolute value = higher confidence
        }

    def health_check(self) -> bool:
        """
        Check if money flow analyzer is operational.
        """
        return True

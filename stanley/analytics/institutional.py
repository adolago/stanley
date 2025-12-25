"""
Institutional Analysis Module

Analyze institutional positioning, 13F filings, and smart money activity.
No technical indicators, just institutional data analysis.
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd
from typing import Dict, List

logger = logging.getLogger(__name__)


class InstitutionalAnalyzer:
    """
    Analyze institutional positioning and smart money activity.
    """

    def __init__(self, data_manager=None):
        """
        Initialize institutional analyzer.

        Args:
            data_manager: Data manager instance for data access
        """
        self.data_manager = data_manager
        logger.info("InstitutionalAnalyzer initialized")

    def get_holdings(self, symbol: str) -> Dict:
        """
        Get institutional holdings data for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with institutional holdings information
        """
        # Get 13F data
        institutional_holdings = self._get_13f_holdings(symbol)

        # Get recent changes
        recent_changes = self._get_recent_institutional_changes(symbol)

        # Calculate institutional metrics
        metrics = self._calculate_institutional_metrics(
            institutional_holdings, recent_changes
        )

        return {
            "symbol": symbol,
            "institutional_ownership": metrics["ownership_percentage"],
            "number_of_institutions": metrics["institution_count"],
            "top_holders": metrics["top_holders"],
            "recent_changes": recent_changes,
            "ownership_trend": metrics["ownership_trend"],
            "concentration_risk": metrics["concentration_risk"],
            "smart_money_score": metrics["smart_money_score"],
        }

    def analyze_13f_changes(
        self, manager_cik: str, quarter_over_quarter: bool = True
    ) -> pd.DataFrame:
        """
        Analyze 13F filing changes for a specific institutional manager.

        Args:
            manager_cik: SEC CIK identifier for the manager
            quarter_over_quarter: Compare to previous quarter

        Returns:
            DataFrame with 13F changes analysis
        """
        # Get current and previous 13F filings
        current_filing = self._get_13f_filing(manager_cik, "current")
        previous_filing = self._get_13f_filing(manager_cik, "previous")

        # Calculate changes
        changes = self._calculate_13f_changes(current_filing, previous_filing)

        return changes

    def get_institutional_sentiment(self, universe: List[str]) -> Dict:
        """
        Get institutional sentiment for a universe of stocks.

        Args:
            universe: List of stock symbols

        Returns:
            Dictionary with institutional sentiment metrics
        """
        sentiment_data = []

        for symbol in universe:
            try:
                holdings = self.get_holdings(symbol)
                sentiment_data.append(
                    {
                        "symbol": symbol,
                        "institutional_ownership": holdings["institutional_ownership"],
                        "ownership_trend": holdings["ownership_trend"],
                        "smart_money_score": holdings["smart_money_score"],
                        "concentration_risk": holdings["concentration_risk"],
                    }
                )
            except Exception as e:
                logger.error(f"Error getting sentiment for {symbol}: {e}")
                continue

        df = pd.DataFrame(sentiment_data)

        # Handle empty DataFrame case
        if df.empty:
            return {
                "universe_size": len(universe),
                "average_institutional_ownership": 0.0,
                "percentage_trending_up": 0.0,
                "average_smart_money_score": 0.0,
                "institutional_sentiment": "neutral",
                "details": df,
            }

        # Calculate overall sentiment
        avg_ownership = df["institutional_ownership"].mean()
        trending_up = (df["ownership_trend"] > 0).sum() / len(df)
        avg_smart_money = df["smart_money_score"].mean()

        # Determine sentiment classification
        if avg_smart_money > 0.1:
            sentiment = "bullish"
        elif avg_smart_money < -0.1:
            sentiment = "bearish"
        else:
            sentiment = "neutral"

        return {
            "universe_size": len(universe),
            "average_institutional_ownership": avg_ownership,
            "percentage_trending_up": trending_up,
            "average_smart_money_score": avg_smart_money,
            "institutional_sentiment": sentiment,
            "details": df,
        }

    def track_smart_money(self, minimum_aum: float = 1e9) -> pd.DataFrame:
        """
        Track smart money managers with strong performance.

        Args:
            minimum_aum: Minimum AUM in dollars to consider (default: $1B)

        Returns:
            DataFrame with smart money tracking data
        """
        # Get top performing managers
        top_managers = self._get_top_performing_managers(minimum_aum)

        # Analyze their recent activity
        smart_money_activity = []

        for manager in top_managers:
            try:
                recent_changes = self.analyze_13f_changes(manager["cik"])

                # Focus on significant changes (top additions and reductions)
                significant_additions = recent_changes[
                    recent_changes["change_percentage"] > 0.1  # 10%+ increase
                ].head(5)

                significant_reductions = recent_changes[
                    recent_changes["change_percentage"] < -0.1  # 10%+ decrease
                ].head(5)

                smart_money_activity.append(
                    {
                        "manager_name": manager["name"],
                        "manager_cik": manager["cik"],
                        "aum": manager["aum"],
                        "performance_score": manager["performance_score"],
                        "top_additions": significant_additions,
                        "top_reductions": significant_reductions,
                        "new_positions": recent_changes[
                            recent_changes["change_type"] == "new"
                        ].head(3),
                        "closed_positions": recent_changes[
                            recent_changes["change_type"] == "closed"
                        ].head(3),
                    }
                )

            except Exception as e:
                logger.error(f"Error analyzing manager {manager['cik']}: {e}")
                continue

        return pd.DataFrame(smart_money_activity)

    def _get_13f_holdings(self, symbol: str) -> pd.DataFrame:
        """
        Get 13F holdings data for a symbol.
        """
        if self.data_manager:
            return self.data_manager.get_13f_holdings(symbol)
        else:
            # Placeholder implementation
            return pd.DataFrame(
                {
                    "manager_name": [
                        "Vanguard",
                        "BlackRock",
                        "State Street",
                        "Fidelity",
                        "T. Rowe Price",
                    ],
                    "manager_cik": [
                        "0000102909",
                        "0001390777",
                        "0000093751",
                        "0000315066",
                        "0000080227",
                    ],
                    "shares_held": [100000000, 80000000, 60000000, 40000000, 30000000],
                    "value_held": [
                        10000000000,
                        8000000000,
                        6000000000,
                        4000000000,
                        3000000000,
                    ],
                    "ownership_percentage": [0.05, 0.04, 0.03, 0.02, 0.015],
                }
            )

    def _get_recent_institutional_changes(self, symbol: str) -> pd.DataFrame:
        """
        Get recent institutional changes for a symbol.
        """
        if self.data_manager:
            return self.data_manager.get_institutional_changes(symbol)
        else:
            # Placeholder implementation
            dates = pd.date_range(end=datetime.now(), periods=5, freq="ME")
            return pd.DataFrame(
                {
                    "date": dates,
                    "net_institutional_change": [
                        1000000,
                        -500000,
                        2000000,
                        -100000,
                        1500000,
                    ],
                    "new_institutions": [5, 2, 8, 1, 6],
                    "closed_institutions": [2, 4, 1, 3, 2],
                    "total_institutions": [250, 253, 251, 258, 256],
                }
            )

    def _calculate_institutional_metrics(
        self, holdings: pd.DataFrame, changes: pd.DataFrame
    ) -> Dict:
        """
        Calculate institutional metrics from holdings and changes data.
        """
        # Handle empty holdings DataFrame
        if holdings.empty:
            return {
                "ownership_percentage": 0.0,
                "institution_count": 0,
                "top_holders": pd.DataFrame(
                    columns=["manager_name", "value_held", "ownership_percentage"]
                ),
                "ownership_trend": 0.0,
                "concentration_risk": 0.0,
                "smart_money_score": 0.0,
            }

        # Total institutional ownership
        total_ownership = holdings["ownership_percentage"].sum()

        # Number of institutions
        institution_count = len(holdings)

        # Top holders
        top_holders = holdings.nlargest(5, "value_held")[
            ["manager_name", "value_held", "ownership_percentage"]
        ]

        # Ownership trend from recent changes
        recent_change = (
            changes["net_institutional_change"].iloc[-1] if len(changes) > 0 else 0
        )
        ownership_trend = np.sign(recent_change) * min(
            1, abs(recent_change) / 1000000
        )  # Normalize

        # Concentration risk (normalized Herfindahl-Hirschman Index)
        # HHI = sum(s_i^2) where s_i is market share of institution i
        # Normalized HHI = (HHI - 1/n) / (1 - 1/n) to get value in [0, 1]
        ownership_shares = holdings["ownership_percentage"]
        n_institutions = len(holdings)
        if n_institutions == 0:
            concentration_risk = 0.0
        elif n_institutions == 1:
            concentration_risk = 1.0
        else:
            # Normalize shares to sum to 1 for proper HHI calculation
            normalized_shares = (
                ownership_shares / ownership_shares.sum()
                if ownership_shares.sum() > 0
                else ownership_shares
            )
            hhi = (normalized_shares**2).sum()
            # Normalize HHI to [0, 1] range
            min_hhi = 1.0 / n_institutions
            concentration_risk = (
                (hhi - min_hhi) / (1.0 - min_hhi) if (1.0 - min_hhi) > 0 else 0.0
            )
            concentration_risk = max(0.0, min(1.0, concentration_risk))

        # Smart money score based on institutional activity and concentration
        smart_money_score = (
            0.6 * ownership_trend
            + 0.2 * (institution_count / 500)  # Normalize to max 500 institutions
            + 0.2 * (1 - concentration_risk)  # Lower concentration = better
        )

        return {
            "ownership_percentage": min(1.0, total_ownership),  # Cap at 100%
            "institution_count": institution_count,
            "top_holders": top_holders,
            "ownership_trend": ownership_trend,
            "concentration_risk": concentration_risk,
            "smart_money_score": max(-1, min(1, smart_money_score)),
        }

    def _get_13f_filing(self, manager_cik: str, period: str) -> pd.DataFrame:
        """
        Get 13F filing data for a manager.
        """
        if self.data_manager:
            return self.data_manager.get_13f_filing(manager_cik, period)
        else:
            # Placeholder implementation
            symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
            return pd.DataFrame(
                {
                    "symbol": symbols,
                    "shares": [10000000, 8000000, 6000000, 5000000, 4000000],
                    "value": [1500000000, 1200000000, 900000000, 750000000, 600000000],
                    "weight": [0.25, 0.20, 0.15, 0.12, 0.10],
                }
            )

    def _calculate_13f_changes(
        self, current_filing: pd.DataFrame, previous_filing: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate changes between two 13F filings.
        """
        # Merge current and previous holdings
        merged = current_filing.merge(
            previous_filing,
            on="symbol",
            how="outer",
            suffixes=("_current", "_previous"),
        ).fillna(0)

        # Calculate changes
        merged["shares_change"] = merged["shares_current"] - merged["shares_previous"]
        merged["value_change"] = merged["value_current"] - merged["value_previous"]
        merged["change_percentage"] = np.where(
            merged["shares_previous"] > 0,
            merged["shares_change"] / merged["shares_previous"],
            np.where(merged["shares_current"] > 0, 1, 0),  # New position
        )

        # Determine change type
        merged["change_type"] = np.where(
            merged["shares_previous"] == 0,
            "new",
            np.where(merged["shares_current"] == 0, "closed", "existing"),
        )

        return merged.sort_values("change_percentage", ascending=False)

    def _get_top_performing_managers(self, minimum_aum: float) -> List[Dict]:
        """
        Get top performing institutional managers.
        """
        # Placeholder implementation
        return [
            {
                "cik": "0000102909",
                "name": "Vanguard Group",
                "aum": 7000000000000,  # $7T
                "performance_score": 0.85,
            },
            {
                "cik": "0001390777",
                "name": "BlackRock",
                "aum": 8000000000000,  # $8T
                "performance_score": 0.82,
            },
            {
                "cik": "0000093751",
                "name": "State Street",
                "aum": 3000000000000,  # $3T
                "performance_score": 0.78,
            },
        ]

    def health_check(self) -> bool:
        """
        Check if institutional analyzer is operational.
        """
        return True

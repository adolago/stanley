"""
Institutional Analysis Module

Analyze institutional positioning, 13F filings, and smart money activity.
No technical indicators, just institutional data analysis.

Enhanced with advanced 13F tracking capabilities:
- Filing calendar tracking
- Manager performance tracking
- Position clustering detection
- Conviction picks identification
- Portfolio overlap analysis
- New/exit position alerts
- Smart money flow calculation
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 13F filing deadlines (45 days after quarter end)
QUARTER_END_DATES = {
    1: (3, 31),  # Q1 ends March 31
    2: (6, 30),  # Q2 ends June 30
    3: (9, 30),  # Q3 ends September 30
    4: (12, 31),  # Q4 ends December 31
}

FILING_DEADLINE_DAYS = 45  # 13F deadline is 45 days after quarter end


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

    def get_holdings(
        self,
        symbol: str,
        include_changes: bool = False,
        include_performance: bool = False,
    ) -> Dict:
        """
        Get institutional holdings data for a symbol.

        Args:
            symbol: Stock symbol
            include_changes: Include quarter-over-quarter changes for each holder
            include_performance: Include manager performance metrics

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

        result = {
            "symbol": symbol,
            "institutional_ownership": metrics["ownership_percentage"],
            "number_of_institutions": metrics["institution_count"],
            "top_holders": metrics["top_holders"],
            "recent_changes": recent_changes,
            "ownership_trend": metrics["ownership_trend"],
            "concentration_risk": metrics["concentration_risk"],
            "smart_money_score": metrics["smart_money_score"],
        }

        # Add quarter-over-quarter changes if requested
        if include_changes and not institutional_holdings.empty:
            result["holder_changes"] = self._get_holder_qoq_changes(
                symbol, institutional_holdings
            )

        # Add manager performance metrics if requested
        if include_performance and not institutional_holdings.empty:
            result["manager_performance"] = self._get_managers_performance_summary(
                institutional_holdings
            )

        return result

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

    # =========================================================================
    # NEW ADVANCED 13F TRACKING METHODS
    # =========================================================================

    def get_13f_filing_calendar(self, quarters_ahead: int = 4) -> pd.DataFrame:
        """
        Get upcoming 13F filing deadlines for the next N quarters.

        Args:
            quarters_ahead: Number of quarters to look ahead (default: 4)

        Returns:
            DataFrame with quarter end dates and filing deadlines
        """
        calendar_entries = []
        current_date = datetime.now()

        for i in range(quarters_ahead):
            # Calculate the target quarter
            target_date = current_date + timedelta(days=90 * i)
            year = target_date.year
            quarter = (target_date.month - 1) // 3 + 1

            # Get quarter end date
            month, day = QUARTER_END_DATES[quarter]
            quarter_end = datetime(year, month, day)

            # If we're past this quarter end, move to next
            if quarter_end < current_date and i == 0:
                if quarter == 4:
                    quarter = 1
                    year += 1
                else:
                    quarter += 1
                month, day = QUARTER_END_DATES[quarter]
                quarter_end = datetime(year, month, day)

            # Calculate filing deadline (45 days after quarter end)
            filing_deadline = quarter_end + timedelta(days=FILING_DEADLINE_DAYS)

            # Determine status
            if current_date > filing_deadline:
                status = "past"
            elif current_date > quarter_end:
                status = "filing_period"
                days_remaining = (filing_deadline - current_date).days
            else:
                status = "upcoming"
                days_remaining = (filing_deadline - current_date).days

            calendar_entries.append(
                {
                    "quarter": f"Q{quarter} {year}",
                    "quarter_end": quarter_end,
                    "filing_deadline": filing_deadline,
                    "status": status,
                    "days_until_deadline": max(
                        0, (filing_deadline - current_date).days
                    ),
                }
            )

        return pd.DataFrame(calendar_entries)

    def track_manager_performance(
        self, manager_cik: str, lookback_quarters: int = 8
    ) -> Dict[str, Any]:
        """
        Track manager's historical performance based on 13F-reported positions.

        Args:
            manager_cik: SEC CIK identifier for the manager
            lookback_quarters: Number of quarters to analyze (default: 8)

        Returns:
            Dictionary with manager performance metrics
        """
        # Get historical filings
        historical_returns = []
        quarterly_data = []

        for quarter in range(lookback_quarters):
            try:
                # Get filing for this quarter
                filing = self._get_13f_filing(
                    manager_cik, f"Q-{quarter}" if quarter > 0 else "current"
                )

                if filing.empty:
                    continue

                # Calculate hypothetical portfolio return for the quarter
                quarter_return = self._calculate_portfolio_quarter_return(filing)
                historical_returns.append(quarter_return)

                quarterly_data.append(
                    {
                        "quarters_ago": quarter,
                        "portfolio_value": (
                            filing["value"].sum() if "value" in filing.columns else 0
                        ),
                        "num_positions": len(filing),
                        "top_holding_weight": (
                            filing["weight"].max() if "weight" in filing.columns else 0
                        ),
                        "quarter_return": quarter_return,
                    }
                )

            except Exception as e:
                logger.warning(f"Error getting filing for quarter -{quarter}: {e}")
                continue

        # Calculate performance metrics
        returns_array = (
            np.array(historical_returns) if historical_returns else np.array([0])
        )

        return {
            "manager_cik": manager_cik,
            "lookback_quarters": lookback_quarters,
            "total_return": (
                float(np.prod(1 + returns_array) - 1) if len(returns_array) > 0 else 0
            ),
            "annualized_return": (
                float(np.mean(returns_array) * 4) if len(returns_array) > 0 else 0
            ),
            "volatility": (
                float(np.std(returns_array) * 2) if len(returns_array) > 1 else 0
            ),  # Semi-annual vol
            "sharpe_ratio": self._calculate_sharpe_ratio(returns_array),
            "max_drawdown": self._calculate_max_drawdown(returns_array),
            "win_rate": (
                float(np.mean(returns_array > 0)) if len(returns_array) > 0 else 0
            ),
            "quarterly_data": (
                pd.DataFrame(quarterly_data) if quarterly_data else pd.DataFrame()
            ),
            "performance_rank": self._get_manager_performance_rank(manager_cik),
        }

    def detect_position_clustering(
        self, symbols: List[str], min_common_holders: int = 3
    ) -> pd.DataFrame:
        """
        Find stocks that are commonly held by the same institutional managers.

        Args:
            symbols: List of stock symbols to analyze
            min_common_holders: Minimum number of common holders to report

        Returns:
            DataFrame with clustering information
        """
        # Get holders for each symbol
        symbol_holders = {}
        for symbol in symbols:
            try:
                holdings = self._get_13f_holdings(symbol)
                if not holdings.empty and "manager_cik" in holdings.columns:
                    symbol_holders[symbol] = set(holdings["manager_cik"].tolist())
                else:
                    symbol_holders[symbol] = set()
            except Exception as e:
                logger.warning(f"Error getting holders for {symbol}: {e}")
                symbol_holders[symbol] = set()

        # Find common holders between pairs
        clustering_results = []
        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i + 1 :]:
                common_holders = symbol_holders[sym1] & symbol_holders[sym2]
                if len(common_holders) >= min_common_holders:
                    clustering_results.append(
                        {
                            "symbol_1": sym1,
                            "symbol_2": sym2,
                            "common_holders_count": len(common_holders),
                            "common_holders": list(common_holders)[:10],  # Top 10
                            "holder_overlap_pct": (
                                len(common_holders)
                                / min(
                                    len(symbol_holders[sym1]), len(symbol_holders[sym2])
                                )
                                if min(
                                    len(symbol_holders[sym1]), len(symbol_holders[sym2])
                                )
                                > 0
                                else 0
                            ),
                        }
                    )

        result = pd.DataFrame(clustering_results)
        if not result.empty:
            result = result.sort_values("common_holders_count", ascending=False)

        return result

    def get_conviction_picks(
        self, min_weight: float = 0.05, top_n_managers: int = 50
    ) -> pd.DataFrame:
        """
        Get high-conviction positions (stocks with significant portfolio weight).

        Args:
            min_weight: Minimum portfolio weight to consider (default: 5%)
            top_n_managers: Number of top managers to analyze

        Returns:
            DataFrame with conviction picks and their holders
        """
        conviction_positions = []
        top_managers = self._get_top_performing_managers(minimum_aum=1e9)[
            :top_n_managers
        ]

        for manager in top_managers:
            try:
                filing = self._get_13f_filing(manager["cik"], "current")
                if filing.empty:
                    continue

                # Filter for high-conviction positions
                if "weight" in filing.columns:
                    high_conviction = filing[filing["weight"] >= min_weight]
                    for _, row in high_conviction.iterrows():
                        conviction_positions.append(
                            {
                                "symbol": row.get("symbol", ""),
                                "manager_name": manager["name"],
                                "manager_cik": manager["cik"],
                                "weight": row.get("weight", 0),
                                "value": row.get("value", 0),
                                "shares": row.get("shares", 0),
                                "manager_aum": manager.get("aum", 0),
                                "manager_performance_score": manager.get(
                                    "performance_score", 0
                                ),
                            }
                        )

            except Exception as e:
                logger.warning(
                    f"Error getting conviction picks for {manager['cik']}: {e}"
                )
                continue

        result = pd.DataFrame(conviction_positions)
        if not result.empty:
            # Aggregate by symbol
            aggregated = (
                result.groupby("symbol")
                .agg(
                    {
                        "manager_name": lambda x: list(x),
                        "weight": ["mean", "max", "count"],
                        "value": "sum",
                        "manager_performance_score": "mean",
                    }
                )
                .reset_index()
            )
            aggregated.columns = [
                "symbol",
                "holders",
                "avg_weight",
                "max_weight",
                "holder_count",
                "total_value",
                "avg_manager_score",
            ]
            result = aggregated.sort_values("holder_count", ascending=False)

        return result

    def analyze_portfolio_overlap(self, manager_ciks: List[str]) -> Dict[str, Any]:
        """
        Compare portfolios of multiple managers to find overlap and divergence.

        Args:
            manager_ciks: List of manager CIK identifiers

        Returns:
            Dictionary with overlap analysis
        """
        portfolios = {}
        for cik in manager_ciks:
            try:
                filing = self._get_13f_filing(cik, "current")
                if not filing.empty and "symbol" in filing.columns:
                    portfolios[cik] = set(filing["symbol"].tolist())
            except Exception as e:
                logger.warning(f"Error getting portfolio for {cik}: {e}")
                portfolios[cik] = set()

        # Calculate overlap matrix
        overlap_matrix = []
        for cik1 in manager_ciks:
            row = []
            for cik2 in manager_ciks:
                if cik1 == cik2:
                    overlap = 1.0
                else:
                    common = portfolios.get(cik1, set()) & portfolios.get(cik2, set())
                    total = portfolios.get(cik1, set()) | portfolios.get(cik2, set())
                    overlap = len(common) / len(total) if total else 0
                row.append(overlap)
            overlap_matrix.append(row)

        # Find common positions across all managers
        all_positions = [portfolios.get(cik, set()) for cik in manager_ciks]
        common_to_all = set.intersection(*all_positions) if all_positions else set()

        # Find unique positions for each manager
        unique_positions = {}
        for cik in manager_ciks:
            others = (
                set.union(
                    *[
                        portfolios.get(other_cik, set())
                        for other_cik in manager_ciks
                        if other_cik != cik
                    ]
                )
                if len(manager_ciks) > 1
                else set()
            )
            unique_positions[cik] = list(portfolios.get(cik, set()) - others)

        return {
            "manager_ciks": manager_ciks,
            "overlap_matrix": pd.DataFrame(
                overlap_matrix, index=manager_ciks, columns=manager_ciks
            ),
            "common_positions": list(common_to_all),
            "common_positions_count": len(common_to_all),
            "unique_positions": unique_positions,
            "average_overlap": (
                float(
                    np.mean(
                        [
                            overlap_matrix[i][j]
                            for i in range(len(manager_ciks))
                            for j in range(i + 1, len(manager_ciks))
                        ]
                    )
                )
                if len(manager_ciks) > 1
                else 1.0
            ),
        }

    def get_new_positions_alert(
        self, lookback_days: int = 45, min_value: float = 1e7
    ) -> pd.DataFrame:
        """
        Alert on new institutional positions initiated recently.

        Args:
            lookback_days: Days to look back for new positions (default: 45)
            min_value: Minimum position value to report (default: $10M)

        Returns:
            DataFrame with new position alerts
        """
        new_positions = []
        top_managers = self._get_top_performing_managers(minimum_aum=1e9)

        for manager in top_managers:
            try:
                # Compare current to previous quarter
                current = self._get_13f_filing(manager["cik"], "current")
                previous = self._get_13f_filing(manager["cik"], "previous")

                if current.empty:
                    continue

                # Find new positions
                current_symbols = (
                    set(current["symbol"].tolist())
                    if "symbol" in current.columns
                    else set()
                )
                previous_symbols = (
                    set(previous["symbol"].tolist())
                    if not previous.empty and "symbol" in previous.columns
                    else set()
                )
                new_symbols = current_symbols - previous_symbols

                for _, row in current.iterrows():
                    if row.get("symbol", "") in new_symbols:
                        value = row.get("value", 0)
                        if value >= min_value:
                            new_positions.append(
                                {
                                    "symbol": row.get("symbol", ""),
                                    "manager_name": manager["name"],
                                    "manager_cik": manager["cik"],
                                    "position_value": value,
                                    "shares": row.get("shares", 0),
                                    "weight": row.get("weight", 0),
                                    "manager_performance_score": manager.get(
                                        "performance_score", 0
                                    ),
                                    "alert_type": "new_position",
                                    "significance": self._calculate_significance_score(
                                        value,
                                        row.get("weight", 0),
                                        manager.get("performance_score", 0),
                                    ),
                                }
                            )

            except Exception as e:
                logger.warning(
                    f"Error checking new positions for {manager['cik']}: {e}"
                )
                continue

        result = pd.DataFrame(new_positions)
        if not result.empty:
            result = result.sort_values("significance", ascending=False)

        return result

    def get_exit_positions_alert(
        self, lookback_days: int = 45, min_previous_value: float = 1e7
    ) -> pd.DataFrame:
        """
        Alert on institutional positions that were closed recently.

        Args:
            lookback_days: Days to look back for exits (default: 45)
            min_previous_value: Minimum previous value to report (default: $10M)

        Returns:
            DataFrame with exit position alerts
        """
        exit_positions = []
        top_managers = self._get_top_performing_managers(minimum_aum=1e9)

        for manager in top_managers:
            try:
                # Compare current to previous quarter
                current = self._get_13f_filing(manager["cik"], "current")
                previous = self._get_13f_filing(manager["cik"], "previous")

                if previous.empty:
                    continue

                # Find closed positions
                current_symbols = (
                    set(current["symbol"].tolist())
                    if not current.empty and "symbol" in current.columns
                    else set()
                )
                previous_symbols = (
                    set(previous["symbol"].tolist())
                    if "symbol" in previous.columns
                    else set()
                )
                closed_symbols = previous_symbols - current_symbols

                for _, row in previous.iterrows():
                    if row.get("symbol", "") in closed_symbols:
                        value = row.get("value", 0)
                        if value >= min_previous_value:
                            exit_positions.append(
                                {
                                    "symbol": row.get("symbol", ""),
                                    "manager_name": manager["name"],
                                    "manager_cik": manager["cik"],
                                    "previous_value": value,
                                    "previous_shares": row.get("shares", 0),
                                    "previous_weight": row.get("weight", 0),
                                    "manager_performance_score": manager.get(
                                        "performance_score", 0
                                    ),
                                    "alert_type": "position_exit",
                                    "significance": self._calculate_significance_score(
                                        value,
                                        row.get("weight", 0),
                                        manager.get("performance_score", 0),
                                    ),
                                }
                            )

            except Exception as e:
                logger.warning(
                    f"Error checking exit positions for {manager['cik']}: {e}"
                )
                continue

        result = pd.DataFrame(exit_positions)
        if not result.empty:
            result = result.sort_values("significance", ascending=False)

        return result

    def calculate_smart_money_flow(self, symbol: str) -> Dict[str, Any]:
        """
        Calculate net buying/selling by top-performing managers.

        Args:
            symbol: Stock symbol to analyze

        Returns:
            Dictionary with smart money flow metrics
        """
        # Get top performing managers
        top_managers = self._get_top_performing_managers(minimum_aum=1e9)
        manager_ciks = [m["cik"] for m in top_managers]
        manager_scores = {
            m["cik"]: m.get("performance_score", 0.5) for m in top_managers
        }

        buying_activity = []
        selling_activity = []
        net_flow = 0
        weighted_flow = 0

        for manager in top_managers:
            try:
                changes = self.analyze_13f_changes(manager["cik"])
                if changes.empty:
                    continue

                # Find this symbol in changes
                symbol_changes = changes[changes["symbol"] == symbol]
                if symbol_changes.empty:
                    continue

                for _, row in symbol_changes.iterrows():
                    shares_change = row.get("shares_change", 0)
                    value_change = row.get("value_change", 0)
                    performance_weight = manager_scores.get(manager["cik"], 0.5)

                    if shares_change > 0:
                        buying_activity.append(
                            {
                                "manager_name": manager["name"],
                                "manager_cik": manager["cik"],
                                "shares_added": shares_change,
                                "value_added": value_change,
                                "performance_score": performance_weight,
                            }
                        )
                        net_flow += value_change
                        weighted_flow += value_change * performance_weight
                    elif shares_change < 0:
                        selling_activity.append(
                            {
                                "manager_name": manager["name"],
                                "manager_cik": manager["cik"],
                                "shares_sold": abs(shares_change),
                                "value_sold": abs(value_change),
                                "performance_score": performance_weight,
                            }
                        )
                        net_flow += value_change  # Negative for selling
                        weighted_flow += value_change * performance_weight

            except Exception as e:
                logger.warning(f"Error calculating flow for {manager['cik']}: {e}")
                continue

        # Determine smart money signal
        if weighted_flow > 0:
            signal = "accumulation"
            signal_strength = min(1.0, weighted_flow / 1e9)  # Normalize to $1B
        elif weighted_flow < 0:
            signal = "distribution"
            signal_strength = min(1.0, abs(weighted_flow) / 1e9)
        else:
            signal = "neutral"
            signal_strength = 0

        return {
            "symbol": symbol,
            "net_flow": net_flow,
            "weighted_flow": weighted_flow,
            "signal": signal,
            "signal_strength": signal_strength,
            "buyers_count": len(buying_activity),
            "sellers_count": len(selling_activity),
            "buying_activity": (
                pd.DataFrame(buying_activity) if buying_activity else pd.DataFrame()
            ),
            "selling_activity": (
                pd.DataFrame(selling_activity) if selling_activity else pd.DataFrame()
            ),
            "coordinated_buying": len(buying_activity) >= 3,
            "coordinated_selling": len(selling_activity) >= 3,
        }

    def get_significant_changes_alert(
        self, threshold_pct: float = 0.25
    ) -> pd.DataFrame:
        """
        Alert on significant position changes (>threshold increase/decrease).

        Args:
            threshold_pct: Change threshold to trigger alert (default: 25%)

        Returns:
            DataFrame with significant change alerts
        """
        significant_changes = []
        top_managers = self._get_top_performing_managers(minimum_aum=1e9)

        for manager in top_managers:
            try:
                changes = self.analyze_13f_changes(manager["cik"])
                if changes.empty:
                    continue

                # Filter for significant changes
                significant = changes[
                    (changes["change_percentage"].abs() >= threshold_pct)
                    & (changes["change_type"] == "existing")  # Exclude new/closed
                ]

                for _, row in significant.iterrows():
                    change_pct = row.get("change_percentage", 0)
                    significant_changes.append(
                        {
                            "symbol": row.get("symbol", ""),
                            "manager_name": manager["name"],
                            "manager_cik": manager["cik"],
                            "change_type": "increase" if change_pct > 0 else "decrease",
                            "change_percentage": change_pct,
                            "shares_change": row.get("shares_change", 0),
                            "value_change": row.get("value_change", 0),
                            "current_value": row.get("value_current", 0),
                            "manager_performance_score": manager.get(
                                "performance_score", 0
                            ),
                        }
                    )

            except Exception as e:
                logger.warning(
                    f"Error checking significant changes for {manager['cik']}: {e}"
                )
                continue

        result = pd.DataFrame(significant_changes)
        if not result.empty:
            result = result.sort_values("change_percentage", key=abs, ascending=False)

        return result

    def get_coordinated_buying_alert(
        self, min_buyers: int = 3, lookback_days: int = 45
    ) -> pd.DataFrame:
        """
        Alert on stocks being bought by multiple top managers.

        Args:
            min_buyers: Minimum number of coordinated buyers to report
            lookback_days: Days to look back for coordinated activity

        Returns:
            DataFrame with coordinated buying alerts
        """
        # Aggregate buying activity by symbol
        symbol_buyers = {}
        top_managers = self._get_top_performing_managers(minimum_aum=1e9)

        for manager in top_managers:
            try:
                changes = self.analyze_13f_changes(manager["cik"])
                if changes.empty:
                    continue

                # Find increases and new positions
                buying = changes[
                    (changes["change_percentage"] > 0.05)
                    | (changes["change_type"] == "new")
                ]

                for _, row in buying.iterrows():
                    symbol = row.get("symbol", "")
                    if symbol not in symbol_buyers:
                        symbol_buyers[symbol] = []
                    symbol_buyers[symbol].append(
                        {
                            "manager_name": manager["name"],
                            "manager_cik": manager["cik"],
                            "change_type": row.get("change_type", "existing"),
                            "change_percentage": row.get("change_percentage", 0),
                            "value_added": row.get("value_change", 0),
                            "performance_score": manager.get("performance_score", 0),
                        }
                    )

            except Exception as e:
                logger.warning(
                    f"Error checking coordinated buying for {manager['cik']}: {e}"
                )
                continue

        # Filter for coordinated buying
        coordinated = []
        for symbol, buyers in symbol_buyers.items():
            if len(buyers) >= min_buyers:
                total_value = sum(b.get("value_added", 0) for b in buyers)
                avg_performance = np.mean(
                    [b.get("performance_score", 0) for b in buyers]
                )
                coordinated.append(
                    {
                        "symbol": symbol,
                        "buyers_count": len(buyers),
                        "total_value_added": total_value,
                        "avg_buyer_performance": avg_performance,
                        "buyers": buyers,
                        "signal_strength": min(1.0, len(buyers) / 10) * avg_performance,
                    }
                )

        result = pd.DataFrame(coordinated)
        if not result.empty:
            result = result.sort_values("signal_strength", ascending=False)

        return result

    def get_concentration_risk_alert(self, threshold: float = 0.1) -> pd.DataFrame:
        """
        Alert on stocks with high concentration risk (few large holders).

        Args:
            threshold: Top holder ownership threshold to trigger alert

        Returns:
            DataFrame with concentration risk alerts
        """
        # This would analyze holdings across a universe
        # For now, return empty DataFrame as placeholder
        logger.info(f"Checking concentration risk with threshold {threshold}")
        return pd.DataFrame(
            columns=[
                "symbol",
                "top_holder_ownership",
                "top_3_ownership",
                "hhi_score",
                "risk_level",
            ]
        )

    # =========================================================================
    # HELPER METHODS FOR NEW FUNCTIONALITY
    # =========================================================================

    def _get_holder_qoq_changes(
        self, symbol: str, current_holdings: pd.DataFrame
    ) -> pd.DataFrame:
        """Get quarter-over-quarter changes for each holder."""
        # Placeholder - would fetch previous quarter data
        if current_holdings.empty:
            return pd.DataFrame()

        # Add change columns with mock data
        result = current_holdings.copy()
        result["shares_change"] = np.random.randint(-1000000, 1000000, len(result))
        result["change_percentage"] = result["shares_change"] / result["shares_held"]
        result["change_type"] = np.where(
            result["shares_change"] > 0,
            "increase",
            np.where(result["shares_change"] < 0, "decrease", "unchanged"),
        )
        return result

    def _get_managers_performance_summary(self, holdings: pd.DataFrame) -> pd.DataFrame:
        """Get performance summary for managers holding a stock."""
        if holdings.empty or "manager_cik" not in holdings.columns:
            return pd.DataFrame()

        performance_data = []
        for _, row in holdings.iterrows():
            cik = row.get("manager_cik", "")
            performance_data.append(
                {
                    "manager_name": row.get("manager_name", ""),
                    "manager_cik": cik,
                    "performance_score": np.random.uniform(0.5, 1.0),  # Placeholder
                    "1yr_return": np.random.uniform(-0.2, 0.4),  # Placeholder
                    "sharpe_ratio": np.random.uniform(0.5, 2.0),  # Placeholder
                }
            )

        return pd.DataFrame(performance_data)

    def _calculate_portfolio_quarter_return(self, filing: pd.DataFrame) -> float:
        """Calculate hypothetical return for a portfolio filing."""
        # Placeholder - would use actual price data
        if filing.empty:
            return 0.0

        # Simulate return based on filing weights
        weights = (
            filing["weight"].values
            if "weight" in filing.columns
            else np.ones(len(filing)) / len(filing)
        )
        # Mock quarterly return
        return float(np.random.normal(0.02, 0.08))

    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio from returns array."""
        if len(returns) < 2:
            return 0.0
        excess_returns = returns - 0.01  # Assume 4% annual risk-free / 4 quarters
        if np.std(returns) == 0:
            return 0.0
        return float(np.mean(excess_returns) / np.std(returns))

    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns array."""
        if len(returns) == 0:
            return 0.0
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        return float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0

    def _get_manager_performance_rank(self, manager_cik: str) -> str:
        """Get performance rank for a manager."""
        # Placeholder - would use actual ranking data
        ranks = [
            "top_decile",
            "top_quartile",
            "above_average",
            "average",
            "below_average",
        ]
        return np.random.choice(ranks)

    def _calculate_significance_score(
        self, value: float, weight: float, performance_score: float
    ) -> float:
        """Calculate significance score for an alert."""
        # Combine value, weight, and manager quality
        value_score = min(1.0, value / 1e9)  # Normalize to $1B
        weight_score = min(1.0, weight / 0.1)  # Normalize to 10% weight
        return float(0.4 * value_score + 0.3 * weight_score + 0.3 * performance_score)

    def health_check(self) -> bool:
        """
        Check if institutional analyzer is operational.
        """
        return True

    # =========================================================================
    # Enhanced Bloomberg-Beating Features
    # =========================================================================

    def detect_13f_changes(
        self,
        symbol: str,
        conviction_threshold: float = 0.05,
    ) -> pd.DataFrame:
        """
        Enhanced 13F change detection with conviction scoring.

        Compares current vs previous quarter holdings to detect:
        - New positions
        - Closed positions
        - Significant increases/decreases
        - Conviction scores based on position sizing changes

        Args:
            symbol: Stock symbol to analyze
            conviction_threshold: Minimum change to consider significant (default: 5%)

        Returns:
            DataFrame with columns:
            - manager_name: Institution name
            - change_type: 'new', 'closed', 'increased', 'decreased', 'unchanged'
            - shares_current: Current shares held
            - shares_previous: Previous quarter shares
            - change_magnitude: Absolute change in shares
            - change_percentage: Percentage change
            - conviction_score: Score from -1 (strong sell) to 1 (strong buy)
        """
        # Get current and previous holdings
        current_holdings = self._get_13f_holdings(symbol)
        previous_holdings = self._get_previous_quarter_holdings(symbol)

        if current_holdings.empty and previous_holdings.empty:
            return pd.DataFrame(
                columns=[
                    "manager_name",
                    "change_type",
                    "shares_current",
                    "shares_previous",
                    "change_magnitude",
                    "change_percentage",
                    "conviction_score",
                ]
            )

        # Merge current and previous holdings
        merged = current_holdings.merge(
            previous_holdings,
            on="manager_name",
            how="outer",
            suffixes=("_current", "_previous"),
        ).fillna(0)

        # Calculate changes
        merged["shares_current"] = merged.get("shares_held_current", 0)
        merged["shares_previous"] = merged.get("shares_held_previous", 0)
        merged["change_magnitude"] = (
            merged["shares_current"] - merged["shares_previous"]
        )

        # Calculate percentage change safely
        merged["change_percentage"] = np.where(
            merged["shares_previous"] > 0,
            merged["change_magnitude"] / merged["shares_previous"],
            np.where(merged["shares_current"] > 0, 1.0, 0.0),
        )

        # Determine change type
        def classify_change(row):
            if row["shares_previous"] == 0 and row["shares_current"] > 0:
                return "new"
            elif row["shares_current"] == 0 and row["shares_previous"] > 0:
                return "closed"
            elif row["change_percentage"] > conviction_threshold:
                return "increased"
            elif row["change_percentage"] < -conviction_threshold:
                return "decreased"
            else:
                return "unchanged"

        merged["change_type"] = merged.apply(classify_change, axis=1)

        # Calculate conviction score (-1 to 1)
        # Based on: magnitude of change, position size, and direction
        def calc_conviction(row):
            if row["change_type"] == "new":
                # New positions get score based on relative size
                max_shares = merged["shares_current"].max()
                if max_shares > 0:
                    return min(1.0, row["shares_current"] / max_shares)
                return 0.5
            elif row["change_type"] == "closed":
                return -1.0
            else:
                # Score based on percentage change, capped at [-1, 1]
                return max(-1.0, min(1.0, row["change_percentage"]))

        merged["conviction_score"] = merged.apply(calc_conviction, axis=1)

        # Select and order columns
        result = merged[
            [
                "manager_name",
                "change_type",
                "shares_current",
                "shares_previous",
                "change_magnitude",
                "change_percentage",
                "conviction_score",
            ]
        ].copy()

        return result.sort_values("conviction_score", ascending=False).reset_index(
            drop=True
        )

    def detect_whale_accumulation(
        self,
        symbol: str,
        min_position_change: float = 0.10,
        min_aum: float = 1e9,
    ) -> pd.DataFrame:
        """
        Detect whale activity alerts for large institutional position changes.

        Focuses on "smart money" by filtering institutions by AUM and
        detecting significant position changes.

        Args:
            symbol: Stock symbol to analyze
            min_position_change: Minimum position change to trigger alert (default: 10%)
            min_aum: Minimum AUM in dollars to consider "whale" (default: $1B)

        Returns:
            DataFrame with columns:
            - institution_name: Name of the institution
            - change_type: 'accumulating', 'distributing', 'new_position', 'exit'
            - magnitude: Percentage change in position
            - estimated_aum: Estimated AUM of institution
            - alert_level: 'critical', 'high', 'medium', 'low'
            - shares_changed: Number of shares changed
        """
        # Get 13F changes for the symbol
        changes = self.detect_13f_changes(symbol)

        if changes.empty:
            return pd.DataFrame(
                columns=[
                    "institution_name",
                    "change_type",
                    "magnitude",
                    "estimated_aum",
                    "alert_level",
                    "shares_changed",
                ]
            )

        # Get manager AUM data (placeholder or from data_manager)
        manager_aum = self._get_manager_aum_estimates()

        # Merge with AUM data
        changes = changes.merge(
            manager_aum,
            left_on="manager_name",
            right_on="name",
            how="left",
        )
        changes["estimated_aum"] = changes["aum"].fillna(0)

        # Filter by minimum AUM (whales only)
        whales = changes[changes["estimated_aum"] >= min_aum].copy()

        if whales.empty:
            return pd.DataFrame(
                columns=[
                    "institution_name",
                    "change_type",
                    "magnitude",
                    "estimated_aum",
                    "alert_level",
                    "shares_changed",
                ]
            )

        # Filter by significant changes
        significant = whales[
            (whales["change_type"].isin(["new", "closed"]))
            | (abs(whales["change_percentage"]) >= min_position_change)
        ].copy()

        if significant.empty:
            return pd.DataFrame(
                columns=[
                    "institution_name",
                    "change_type",
                    "magnitude",
                    "estimated_aum",
                    "alert_level",
                    "shares_changed",
                ]
            )

        # Classify change type for alerts
        def classify_whale_change(row):
            if row["change_type"] == "new":
                return "new_position"
            elif row["change_type"] == "closed":
                return "exit"
            elif row["change_percentage"] > 0:
                return "accumulating"
            else:
                return "distributing"

        significant["whale_change_type"] = significant.apply(
            classify_whale_change, axis=1
        )

        # Determine alert level based on AUM and magnitude
        def determine_alert_level(row):
            aum_factor = row["estimated_aum"] / 1e12  # Normalize to trillions
            magnitude = abs(row["change_percentage"])

            score = aum_factor * 0.4 + magnitude * 0.6

            if score > 0.5 or row["change_type"] in ["new", "closed"]:
                return "critical"
            elif score > 0.25:
                return "high"
            elif score > 0.1:
                return "medium"
            else:
                return "low"

        significant["alert_level"] = significant.apply(determine_alert_level, axis=1)

        # Build result DataFrame
        result = pd.DataFrame(
            {
                "institution_name": significant["manager_name"],
                "change_type": significant["whale_change_type"],
                "magnitude": significant["change_percentage"],
                "estimated_aum": significant["estimated_aum"],
                "alert_level": significant["alert_level"],
                "shares_changed": significant["change_magnitude"],
            }
        )

        # Sort by alert level priority
        alert_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        result["_sort"] = result["alert_level"].map(alert_order)
        result = result.sort_values("_sort").drop("_sort", axis=1)

        return result.reset_index(drop=True)

    def calculate_sentiment_score(
        self,
        symbol: str,
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate multi-factor institutional sentiment score.

        Combines multiple factors:
        - ownership_trend: Direction of institutional ownership changes
        - buyer_seller_ratio: Ratio of buyers to sellers
        - concentration_change: Change in ownership concentration
        - filing_momentum: Momentum in 13F filing activity

        Args:
            symbol: Stock symbol to analyze
            weights: Optional custom weights for factors (must sum to 1.0)
                    Default: trend(0.3), buyers(0.25), concentration(0.2), momentum(0.25)

        Returns:
            Dictionary with:
            - score: Overall sentiment score from -1 (bearish) to 1 (bullish)
            - contributing_factors: Dict with individual factor scores
            - confidence: Confidence level from 0 to 1
            - classification: 'strongly_bullish', 'bullish', 'neutral', 'bearish',
              'strongly_bearish'
        """
        # Default weights
        if weights is None:
            weights = {
                "ownership_trend": 0.30,
                "buyer_seller_ratio": 0.25,
                "concentration_change": 0.20,
                "filing_momentum": 0.25,
            }

        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}

        # Get holdings data
        holdings = self.get_holdings(symbol)
        changes = self._get_recent_institutional_changes(symbol)
        changes_13f = self.detect_13f_changes(symbol)

        # Calculate individual factors
        factors = {}

        # 1. Ownership trend (-1 to 1)
        factors["ownership_trend"] = holdings.get("ownership_trend", 0.0)

        # 2. Buyer/seller ratio (-1 to 1)
        if not changes.empty and "new_institutions" in changes.columns:
            buyers = changes["new_institutions"].sum()
            sellers = changes["closed_institutions"].sum()
            total = buyers + sellers
            if total > 0:
                # Normalize to [-1, 1]: (buyers - sellers) / total
                factors["buyer_seller_ratio"] = (buyers - sellers) / total
            else:
                factors["buyer_seller_ratio"] = 0.0
        else:
            factors["buyer_seller_ratio"] = 0.0

        # 3. Concentration change (-1 to 1)
        # Decreasing concentration often signals broader accumulation (bullish)
        # Increasing concentration often signals distribution to fewer hands
        current_concentration = holdings.get("concentration_risk", 0.5)
        # Invert: lower concentration = higher score
        factors["concentration_change"] = (0.5 - current_concentration) * 2

        # 4. Filing momentum (-1 to 1)
        if not changes_13f.empty:
            # Average conviction score from recent filings
            avg_conviction = changes_13f["conviction_score"].mean()
            factors["filing_momentum"] = max(-1.0, min(1.0, avg_conviction))
        else:
            factors["filing_momentum"] = 0.0

        # Calculate weighted score
        score = sum(factors[k] * weights[k] for k in weights.keys())
        score = max(-1.0, min(1.0, score))

        # Calculate confidence based on data availability and consistency
        # Higher confidence when factors agree
        factor_values = list(factors.values())
        factor_std = np.std(factor_values) if len(factor_values) > 1 else 0
        # Lower std = higher confidence (factors agree)
        confidence = max(0.0, min(1.0, 1.0 - factor_std))

        # Data availability factor
        data_completeness = sum(1 for v in factor_values if v != 0) / len(factor_values)
        confidence = confidence * 0.6 + data_completeness * 0.4

        # Classification
        if score > 0.5:
            classification = "strongly_bullish"
        elif score > 0.15:
            classification = "bullish"
        elif score > -0.15:
            classification = "neutral"
        elif score > -0.5:
            classification = "bearish"
        else:
            classification = "strongly_bearish"

        return {
            "score": round(score, 4),
            "contributing_factors": {k: round(v, 4) for k, v in factors.items()},
            "confidence": round(confidence, 4),
            "classification": classification,
            "weights_used": weights,
        }

    def cluster_positions(
        self,
        symbol: str,
        n_clusters: int = 4,
    ) -> Dict[str, Any]:
        """
        Position clustering analysis using statistical quartiles.

        Groups institutions by position sizes to identify if smart money
        clusters are accumulating or distributing. Uses quartile-based
        clustering (no sklearn dependency).

        Args:
            symbol: Stock symbol to analyze
            n_clusters: Number of clusters (default: 4 for quartiles)

        Returns:
            Dictionary with:
            - cluster_labels: DataFrame with institution -> cluster mapping
            - cluster_stats: Statistics for each cluster
            - smart_money_direction: 'accumulating', 'distributing', or 'neutral'
            - cluster_summary: Summary of cluster behavior
        """
        holdings = self._get_13f_holdings(symbol)
        changes = self.detect_13f_changes(symbol)

        if holdings.empty:
            return {
                "cluster_labels": pd.DataFrame(),
                "cluster_stats": {},
                "smart_money_direction": "neutral",
                "cluster_summary": "Insufficient data for clustering",
            }

        # Create position size column for clustering
        holdings = holdings.copy()
        if "value_held" in holdings.columns:
            holdings["position_size"] = holdings["value_held"]
        elif "shares_held" in holdings.columns:
            holdings["position_size"] = holdings["shares_held"]
        else:
            return {
                "cluster_labels": pd.DataFrame(),
                "cluster_stats": {},
                "smart_money_direction": "neutral",
                "cluster_summary": "Missing position data",
            }

        # Use quantile-based clustering
        try:
            holdings["cluster"] = pd.qcut(
                holdings["position_size"],
                q=min(n_clusters, len(holdings)),
                labels=False,
                duplicates="drop",
            )
        except ValueError:
            # If not enough unique values, use simple rank-based clustering
            holdings["cluster"] = (
                holdings["position_size"].rank(method="first").astype(int) % n_clusters
            )

        # Label clusters (0 = smallest, n-1 = largest)
        cluster_names = {
            0: "retail_sized",
            1: "small_institutional",
            2: "mid_institutional",
            3: "whale",
        }
        holdings["cluster_name"] = holdings["cluster"].map(
            lambda x: cluster_names.get(x, f"cluster_{x}")
        )

        # Calculate cluster statistics
        cluster_stats = {}
        for cluster_id in holdings["cluster"].unique():
            cluster_data = holdings[holdings["cluster"] == cluster_id]
            cluster_name = cluster_names.get(cluster_id, f"cluster_{cluster_id}")

            # Merge with change data to get direction
            if not changes.empty:
                cluster_changes = changes[
                    changes["manager_name"].isin(cluster_data["manager_name"])
                ]
                avg_change = (
                    cluster_changes["change_percentage"].mean()
                    if not cluster_changes.empty
                    else 0
                )
                avg_conviction = (
                    cluster_changes["conviction_score"].mean()
                    if not cluster_changes.empty
                    else 0
                )
            else:
                avg_change = 0
                avg_conviction = 0

            cluster_stats[cluster_name] = {
                "count": len(cluster_data),
                "total_position": float(cluster_data["position_size"].sum()),
                "avg_position": float(cluster_data["position_size"].mean()),
                "avg_change_pct": round(avg_change, 4),
                "avg_conviction": round(avg_conviction, 4),
                "direction": (
                    "accumulating"
                    if avg_change > 0.05
                    else ("distributing" if avg_change < -0.05 else "holding")
                ),
            }

        # Determine smart money direction (focus on whale cluster)
        whale_stats = cluster_stats.get("whale", {})
        mid_stats = cluster_stats.get("mid_institutional", {})

        # Weight whale behavior more heavily
        whale_direction = whale_stats.get("avg_conviction", 0)
        mid_direction = mid_stats.get("avg_conviction", 0)
        combined_signal = whale_direction * 0.7 + mid_direction * 0.3

        if combined_signal > 0.1:
            smart_money_direction = "accumulating"
        elif combined_signal < -0.1:
            smart_money_direction = "distributing"
        else:
            smart_money_direction = "neutral"

        # Create cluster labels DataFrame
        cluster_labels = holdings[["manager_name", "cluster", "cluster_name"]].copy()

        return {
            "cluster_labels": cluster_labels,
            "cluster_stats": cluster_stats,
            "smart_money_direction": smart_money_direction,
            "cluster_summary": (
                f"Whale cluster is {whale_stats.get('direction', 'unknown')} "
                f"with {whale_stats.get('count', 0)} institutions"
            ),
        }

    def analyze_cross_filing(
        self,
        symbol: str,
        min_filers: int = 3,
    ) -> Dict[str, Any]:
        """
        Cross-filing pattern analysis across multiple 13F filers.

        Analyzes the same stock across multiple institutional 13F filers
        to detect coordinated buying/selling patterns.

        Args:
            symbol: Stock symbol to analyze
            min_filers: Minimum number of filers required for analysis (default: 3)

        Returns:
            Dictionary with:
            - institution_count: Number of institutions analyzed
            - institution_agreement: Percentage of institutions in agreement
            - consensus_direction: 'buying', 'selling', 'mixed', or 'neutral'
            - cross_filing_score: Score from -1 (coordinated selling) to 1 (coordinated
              buying)
            - filing_breakdown: Detailed breakdown of filer activity
        """
        changes = self.detect_13f_changes(symbol)

        if changes.empty or len(changes) < min_filers:
            return {
                "institution_count": len(changes) if not changes.empty else 0,
                "institution_agreement": 0.0,
                "consensus_direction": "insufficient_data",
                "cross_filing_score": 0.0,
                "filing_breakdown": {},
            }

        # Count filers by action
        buyers = len(changes[changes["change_type"].isin(["new", "increased"])])
        sellers = len(changes[changes["change_type"].isin(["closed", "decreased"])])
        holders = len(changes[changes["change_type"] == "unchanged"])
        total = len(changes)

        # Calculate filing breakdown
        filing_breakdown = {
            "new_positions": int(len(changes[changes["change_type"] == "new"])),
            "increased": int(len(changes[changes["change_type"] == "increased"])),
            "decreased": int(len(changes[changes["change_type"] == "decreased"])),
            "closed": int(len(changes[changes["change_type"] == "closed"])),
            "unchanged": int(holders),
        }

        # Calculate agreement level
        # High agreement = most filers doing the same thing
        majority_action = max(buyers, sellers, holders)
        institution_agreement = majority_action / total if total > 0 else 0

        # Determine consensus direction
        if buyers > sellers + holders:
            consensus_direction = "buying"
        elif sellers > buyers + holders:
            consensus_direction = "selling"
        elif abs(buyers - sellers) < total * 0.1:
            consensus_direction = "mixed"
        else:
            consensus_direction = "neutral"

        # Calculate cross-filing score
        # Positive = coordinated buying, Negative = coordinated selling
        if total > 0:
            # Weight by conviction scores
            avg_conviction = changes["conviction_score"].mean()
            direction_factor = (buyers - sellers) / total
            cross_filing_score = direction_factor * 0.6 + avg_conviction * 0.4
            cross_filing_score = max(-1.0, min(1.0, cross_filing_score))
        else:
            cross_filing_score = 0.0

        return {
            "institution_count": total,
            "institution_agreement": round(institution_agreement, 4),
            "consensus_direction": consensus_direction,
            "cross_filing_score": round(cross_filing_score, 4),
            "filing_breakdown": filing_breakdown,
            "buyers_count": buyers,
            "sellers_count": sellers,
            "holders_count": holders,
        }

    def track_smart_money_momentum(
        self,
        symbol: str,
        window_quarters: int = 4,
        weight_by_performance: bool = True,
    ) -> Dict[str, Any]:
        """
        Enhanced smart money tracking with rolling momentum calculation.

        Tracks institutional momentum over configurable windows and
        detects momentum acceleration/deceleration patterns.

        Args:
            symbol: Stock symbol to analyze
            window_quarters: Number of quarters for momentum calculation (default: 4)
            weight_by_performance: Weight by manager performance scores (default: True)

        Returns:
            Dictionary with:
            - momentum_score: Current momentum from -1 to 1
            - trend_direction: 'accelerating', 'decelerating', 'stable'
            - acceleration: Rate of change in momentum
            - quarterly_momentum: List of momentum scores by quarter
            - top_movers: Top institutions driving momentum
            - signal_strength: Confidence in the momentum signal
        """
        # Get current holdings and changes
        holdings = self._get_13f_holdings(symbol)
        changes = self.detect_13f_changes(symbol)

        if holdings.empty:
            return {
                "momentum_score": 0.0,
                "trend_direction": "stable",
                "acceleration": 0.0,
                "quarterly_momentum": [],
                "top_movers": [],
                "signal_strength": 0.0,
            }

        # Get manager performance data for weighting
        if weight_by_performance:
            top_managers = self._get_top_performing_managers(minimum_aum=0)
            performance_map = {m["name"]: m["performance_score"] for m in top_managers}
        else:
            performance_map = {}

        # Calculate current momentum score
        if not changes.empty:
            # Weight conviction scores by manager performance
            def get_weighted_conviction(row):
                base_conviction = row["conviction_score"]
                if weight_by_performance:
                    perf_weight = performance_map.get(row["manager_name"], 0.5)
                    return base_conviction * (0.5 + perf_weight * 0.5)
                return base_conviction

            changes["weighted_conviction"] = changes.apply(
                get_weighted_conviction, axis=1
            )
            current_momentum = changes["weighted_conviction"].mean()
        else:
            current_momentum = 0.0

        current_momentum = max(-1.0, min(1.0, current_momentum))

        # Simulate historical momentum (in production, would use actual historical data)
        # For now, create a realistic momentum history based on current state
        np.random.seed(hash(symbol) % 2**32)  # Deterministic based on symbol
        historical_base = current_momentum * 0.8  # Historical tends toward current
        quarterly_momentum = [
            max(
                -1.0,
                min(
                    1.0,
                    historical_base
                    + np.random.normal(0, 0.15)
                    + (i - window_quarters / 2) * 0.05,
                ),
            )
            for i in range(window_quarters)
        ]
        quarterly_momentum.append(current_momentum)

        # Calculate acceleration (second derivative)
        if len(quarterly_momentum) >= 3:
            recent_change = quarterly_momentum[-1] - quarterly_momentum[-2]
            prior_change = quarterly_momentum[-2] - quarterly_momentum[-3]
            acceleration = recent_change - prior_change
        else:
            acceleration = 0.0

        # Determine trend direction
        if acceleration > 0.05:
            trend_direction = "accelerating"
        elif acceleration < -0.05:
            trend_direction = "decelerating"
        else:
            trend_direction = "stable"

        # Identify top movers
        top_movers = []
        if not changes.empty:
            sorted_changes = changes.nlargest(5, "conviction_score")
            for _, row in sorted_changes.iterrows():
                top_movers.append(
                    {
                        "institution": row["manager_name"],
                        "conviction": round(row["conviction_score"], 4),
                        "change_type": row["change_type"],
                        "magnitude": round(row["change_percentage"], 4),
                    }
                )

        # Calculate signal strength based on agreement and data quality
        if not changes.empty:
            # Higher strength when more institutions agree
            conviction_std = changes["conviction_score"].std()
            agreement_factor = max(0, 1 - conviction_std * 2)

            # Higher strength with more data points
            data_factor = min(1.0, len(changes) / 20)

            signal_strength = agreement_factor * 0.6 + data_factor * 0.4
        else:
            signal_strength = 0.0

        return {
            "momentum_score": round(current_momentum, 4),
            "trend_direction": trend_direction,
            "acceleration": round(acceleration, 4),
            "quarterly_momentum": [round(m, 4) for m in quarterly_momentum],
            "top_movers": top_movers,
            "signal_strength": round(signal_strength, 4),
            "window_quarters": window_quarters,
        }

    # =========================================================================
    # Helper Methods for Enhanced Bloomberg-Beating Features
    # =========================================================================

    def _get_previous_quarter_holdings(self, symbol: str) -> pd.DataFrame:
        """
        Get previous quarter holdings data for a symbol.
        """
        if self.data_manager and hasattr(
            self.data_manager, "get_13f_holdings_previous"
        ):
            return self.data_manager.get_13f_holdings_previous(symbol)
        else:
            # Placeholder: return slightly modified current holdings
            current = self._get_13f_holdings(symbol)
            if current.empty:
                return current

            previous = current.copy()
            # Simulate some changes
            np.random.seed(42)
            multipliers = np.random.uniform(0.85, 1.15, len(previous))
            previous["shares_held"] = (previous["shares_held"] * multipliers).astype(
                int
            )
            previous["value_held"] = (previous["value_held"] * multipliers).astype(int)

            # Remove one random institution (simulates closed position)
            if len(previous) > 3:
                drop_idx = np.random.choice(previous.index)
                previous = previous.drop(drop_idx)

            return previous

    def _get_manager_aum_estimates(self) -> pd.DataFrame:
        """
        Get estimated AUM for institutional managers.
        """
        if self.data_manager and hasattr(self.data_manager, "get_manager_aum"):
            return self.data_manager.get_manager_aum()
        else:
            # Placeholder with major institutions
            return pd.DataFrame(
                {
                    "name": [
                        "Vanguard",
                        "BlackRock",
                        "State Street",
                        "Fidelity",
                        "T. Rowe Price",
                    ],
                    "aum": [
                        7.0e12,
                        8.0e12,
                        3.0e12,
                        2.5e12,
                        1.3e12,
                    ],
                }
            )

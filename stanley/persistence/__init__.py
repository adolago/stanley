"""
Stanley Persistence Module

Local SQLite database for storing user data:
- Watchlists
- User settings
- Alert configurations
- Cached market data
- Historical analysis
- Portfolio holdings
- Trade journal
"""

from .database import StanleyDatabase
from .models import (
    Alert,
    AlertConfiguration,
    AnalysisRecord,
    CachedMarketData,
    PortfolioHolding,
    TradeEntry,
    UserSettings,
    Watchlist,
    WatchlistItem,
)
from .encryption import DataEncryptor
from .migration import MigrationManager
from .backup import BackupManager
from .sync import SyncManager

__all__ = [
    "StanleyDatabase",
    "Watchlist",
    "WatchlistItem",
    "UserSettings",
    "Alert",
    "AlertConfiguration",
    "CachedMarketData",
    "AnalysisRecord",
    "PortfolioHolding",
    "TradeEntry",
    "DataEncryptor",
    "MigrationManager",
    "BackupManager",
    "SyncManager",
]

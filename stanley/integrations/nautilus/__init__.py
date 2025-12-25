"""
NautilusTrader Integration for Stanley

Provides Actors, Indicators, and Data Clients that integrate Stanley's
institutional analytics with the NautilusTrader algorithmic trading framework.

Components:
- Actors: Event-driven components for institutional analysis
- Indicators: Custom indicators for smart money tracking
- Data Client: OpenBB-based data provider for NautilusTrader

Usage:
    from stanley.integrations.nautilus import (
        OpenBBDataClient,
        OpenBBDataClientConfig,
        MoneyFlowActor,
        SmartMoneyIndicator,
    )

    # Create data client
    config = OpenBBDataClientConfig(
        venue="OPENBB",
        openbb_token="your-token",
    )
    client = OpenBBDataClient(config=config, ...)

    # Request historical data
    bars = await client.request_bars(bar_type, start, end)
"""

from stanley.integrations.nautilus.actors import (
    MoneyFlowActor,
    MoneyFlowActorConfig,
    InstitutionalActor,
    InstitutionalActorConfig,
)
from stanley.integrations.nautilus.indicators import (
    SmartMoneyIndicator,
    InstitutionalMomentumIndicator,
)
from stanley.integrations.nautilus.config import (
    OpenBBDataClientConfig,
    OpenBBDataClientConfigFull,
    OpenBBProviderConfig,
    InstrumentConfig,
    BarSubscriptionConfig,
)
from stanley.integrations.nautilus.data_client import (
    OpenBBDataClient,
    OpenBBLiveDataClient,
)
from stanley.integrations.nautilus.data_types import (
    OpenBBBarConverter,
    OpenBBQuoteTickConverter,
    OpenBBTradeTickConverter,
    OpenBBInstrumentProvider,
    create_instrument_id,
    create_bar_type,
    parse_aggregation_string,
)

__all__ = [
    # Actors
    "MoneyFlowActor",
    "MoneyFlowActorConfig",
    "InstitutionalActor",
    "InstitutionalActorConfig",
    # Indicators
    "SmartMoneyIndicator",
    "InstitutionalMomentumIndicator",
    # Data Client
    "OpenBBDataClient",
    "OpenBBLiveDataClient",
    # Configuration
    "OpenBBDataClientConfig",
    "OpenBBDataClientConfigFull",
    "OpenBBProviderConfig",
    "InstrumentConfig",
    "BarSubscriptionConfig",
    # Data Types
    "OpenBBBarConverter",
    "OpenBBQuoteTickConverter",
    "OpenBBTradeTickConverter",
    "OpenBBInstrumentProvider",
    # Utilities
    "create_instrument_id",
    "create_bar_type",
    "parse_aggregation_string",
]

"""
Shared test fixtures for Stanley test suite.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock


@pytest.fixture
def sample_dates():
    """Generate sample date range for testing."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    return pd.date_range(start=start_date, end=end_date, freq="D")


@pytest.fixture
def sample_flow_data(sample_dates):
    """Generate sample flow data DataFrame."""
    np.random.seed(42)  # For reproducibility
    return pd.DataFrame(
        {
            "date": sample_dates,
            "net_flow": np.random.normal(0, 1000000, len(sample_dates)),
            "creation_units": np.random.randint(0, 100, len(sample_dates)),
            "redemption_units": np.random.randint(0, 100, len(sample_dates)),
        }
    )


@pytest.fixture
def empty_flow_data():
    """Empty flow data DataFrame."""
    return pd.DataFrame(
        columns=["date", "net_flow", "creation_units", "redemption_units"]
    )


@pytest.fixture
def sample_institutional_data():
    """Sample institutional positioning data."""
    return {
        "institutional_ownership": 0.75,
        "net_buyer_count": 50,
        "total_institutions": 200,
        "avg_position_size": 5000000,
    }


@pytest.fixture
def sample_holdings_df():
    """Sample holdings DataFrame."""
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
            "value_held": [10000000000, 8000000000, 6000000000, 4000000000, 3000000000],
            "ownership_percentage": [0.05, 0.04, 0.03, 0.02, 0.015],
        }
    )


@pytest.fixture
def sample_changes_df():
    """Sample institutional changes DataFrame."""
    dates = pd.date_range(end=datetime.now(), periods=5, freq="ME")
    return pd.DataFrame(
        {
            "date": dates,
            "net_institutional_change": [1000000, -500000, 2000000, -100000, 1500000],
            "new_institutions": [5, 2, 8, 1, 6],
            "closed_institutions": [2, 4, 1, 3, 2],
            "total_institutions": [250, 253, 251, 258, 256],
        }
    )


@pytest.fixture
def mock_data_manager():
    """Create a mock DataManager."""
    mock = Mock()
    mock.get_etf_flows = AsyncMock(
        return_value=pd.DataFrame(
            {
                "date": pd.date_range(end=datetime.now(), periods=10, freq="D"),
                "net_flow": np.random.normal(0, 1000000, 10),
            }
        )
    )
    mock.get_institutional_positioning = Mock(
        return_value={
            "institutional_ownership": 0.75,
            "net_buyer_count": 50,
            "total_institutions": 200,
            "avg_position_size": 5000000,
        }
    )
    mock.get_13f_holdings = Mock()
    mock.get_institutional_changes = Mock()
    return mock


@pytest.fixture
def sample_config():
    """Sample configuration dictionary."""
    return {
        "openbb": {"api_key": "test_key"},
        "sec": {"enabled": True},
        "database": {"postgresql": {"host": "localhost", "port": 5432}},
    }

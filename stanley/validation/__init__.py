"""
Stanley Validation Module

Comprehensive input/output validation for the Stanley API,
including data quality checks, outlier detection, and validation middleware.
"""

from .models import (
    # Base validators
    StanleyBaseModel,
    # Symbol validation
    SymbolValidator,
    SymbolField,
    # Financial validators
    PositiveFloat,
    NonNegativeFloat,
    Percentage,
    PriceField,
    VolumeField,
    SharesField,
    RatioField,
    # Date validators
    DateRangeValidator,
    TradingDateField,
    # Request models
    ValidatedMoneyFlowRequest,
    ValidatedPortfolioRequest,
    ValidatedPortfolioHolding,
    ValidatedResearchRequest,
    ValidatedCommoditiesRequest,
    ValidatedOptionsRequest,
    # Response validators
    ValidatedMarketData,
    ValidatedPortfolioAnalytics,
    ValidatedValuationMetrics,
)

from .data_quality import (
    DataQualityChecker,
    DataQualityReport,
    DataQualityLevel,
    DataQualityError,
    check_market_data_quality,
    check_ohlc_integrity,
    check_returns_quality,
)

from .outliers import (
    OutlierDetector,
    OutlierResult,
    detect_price_outliers,
    detect_volume_outliers,
    detect_return_outliers,
    calculate_zscore,
    calculate_iqr_bounds,
    detect_jump_discontinuity,
)

from .sanity_checks import (
    SanityChecker,
    SanityCheckResult,
    check_var_sanity,
    check_beta_sanity,
    check_valuation_sanity,
    check_portfolio_weights,
    check_returns_plausibility,
    check_dcf_assumptions,
)

from .temporal import (
    TemporalValidator,
    check_data_freshness,
    validate_date_range,
    validate_trading_hours,
    get_market_calendar,
    is_stale_data,
)

from .middleware import (
    ValidationMiddleware,
    RequestValidator,
    ResponseValidator,
    create_validation_error_response,
)

from .errors import (
    ValidationError,
    DataQualityError,
    OutlierError,
    SanityCheckError,
    TemporalValidationError,
    CrossFieldValidationError,
)

__all__ = [
    # Models
    "StanleyBaseModel",
    "SymbolValidator",
    "SymbolField",
    "PositiveFloat",
    "NonNegativeFloat",
    "Percentage",
    "PriceField",
    "VolumeField",
    "SharesField",
    "RatioField",
    "DateRangeValidator",
    "TradingDateField",
    # Request models
    "ValidatedMoneyFlowRequest",
    "ValidatedPortfolioRequest",
    "ValidatedPortfolioHolding",
    "ValidatedResearchRequest",
    "ValidatedCommoditiesRequest",
    "ValidatedOptionsRequest",
    # Response validators
    "ValidatedMarketData",
    "ValidatedPortfolioAnalytics",
    "ValidatedValuationMetrics",
    # Data quality
    "DataQualityChecker",
    "DataQualityReport",
    "DataQualityLevel",
    "DataQualityError",
    "check_market_data_quality",
    "check_ohlc_integrity",
    "check_returns_quality",
    # Outliers
    "OutlierDetector",
    "OutlierResult",
    "detect_price_outliers",
    "detect_volume_outliers",
    "detect_return_outliers",
    "calculate_zscore",
    "calculate_iqr_bounds",
    "detect_jump_discontinuity",
    # Sanity checks
    "SanityChecker",
    "SanityCheckResult",
    "check_var_sanity",
    "check_beta_sanity",
    "check_valuation_sanity",
    "check_portfolio_weights",
    "check_returns_plausibility",
    "check_dcf_assumptions",
    # Temporal
    "TemporalValidator",
    "check_data_freshness",
    "validate_date_range",
    "validate_trading_hours",
    "get_market_calendar",
    "is_stale_data",
    # Middleware
    "ValidationMiddleware",
    "RequestValidator",
    "ResponseValidator",
    "create_validation_error_response",
    # Errors
    "ValidationError",
    "DataQualityError",
    "OutlierError",
    "SanityCheckError",
    "TemporalValidationError",
    "CrossFieldValidationError",
]

# Stanley ML Architecture and Feature Roadmap

> **Related Documents**: See [System Architecture](./architecture/system_architecture.md) for current platform architecture, [Rust Patterns](./rust_financial_systems_architecture.md) for advanced Rust patterns.

## Executive Summary

This document outlines the machine learning integration strategy for Stanley, an institutional investment analysis platform. The ML capabilities will enhance Stanley's existing analytics with predictive models, pattern recognition, and automated signal generation.

**Status**: Roadmap document. ML dependencies are not yet added to requirements.txt.

## Current State Analysis

### Existing Analytics Infrastructure

Stanley currently provides robust rule-based analytics:

| Module | Capabilities | ML Enhancement Opportunity |
|--------|-------------|---------------------------|
| `analytics/money_flow.py` | Dark pool detection, block trades, flow momentum | Pattern recognition, anomaly detection |
| `analytics/alerts.py` | Threshold-based alerting | Adaptive thresholds via ML |
| `accounting/anomaly_detection.py` | Z-score, IQR, Benford's Law | Deep anomaly detection |
| `macro/regime_detector.py` | Rule-based regime classification | Hidden Markov Models, clustering |
| `signals/signal_generator.py` | Multi-factor signal generation | ML-enhanced factor scoring |
| `portfolio/risk_metrics.py` | VaR, beta, correlation | ML-based risk forecasting |
| `research/research_analyzer.py` | DCF, valuation metrics | Earnings prediction models |

### Current Dependencies (from requirements.txt)

```
pandas>=2.1.0
numpy>=1.25.0
scipy>=1.11.0
statsmodels>=0.14.0
```

### Proposed ML Dependencies

```
# Classical ML
scikit-learn>=1.4.0
xgboost>=2.0.0
lightgbm>=4.2.0
catboost>=1.2.0

# Deep Learning
torch>=2.1.0
transformers>=4.36.0  # For sentiment analysis

# Model Management
mlflow>=2.9.0
optuna>=3.5.0  # Hyperparameter tuning

# Feature Engineering
ta-lib>=0.4.28  # Technical indicators
featuretools>=1.28.0

# Model Serving
fastapi>=0.104.0  # Already present
redis>=5.0.0      # Already present - for caching
```

---

## ML Architecture Overview

```
+------------------------------------------------------------------+
|                     STANLEY ML ARCHITECTURE                       |
+------------------------------------------------------------------+
|                                                                   |
|  +-----------------------+     +-----------------------------+    |
|  |   Data Layer          |     |   Feature Store             |    |
|  +-----------------------+     +-----------------------------+    |
|  | DataManager           |<--->| Redis Cache                 |    |
|  | OpenBB Adapter        |     | Historical Features         |    |
|  | DBnomics Adapter      |     | Real-time Features          |    |
|  | Edgar Adapter         |     | Derived Features            |    |
|  +-----------------------+     +-----------------------------+    |
|            |                              |                       |
|            v                              v                       |
|  +-----------------------+     +-----------------------------+    |
|  |   Feature Engineering |     |   Model Registry            |    |
|  +-----------------------+     +-----------------------------+    |
|  | Technical Features    |     | MLflow Tracking             |    |
|  | Fundamental Features  |     | Model Versioning            |    |
|  | Alternative Data      |     | A/B Test Configurations     |    |
|  | Sentiment Features    |     | Model Metadata              |    |
|  +-----------------------+     +-----------------------------+    |
|            |                              |                       |
|            v                              v                       |
|  +-------------------------------------------------------+        |
|  |              ML Model Layer                            |        |
|  +-------------------------------------------------------+        |
|  | +---------------+ +---------------+ +---------------+ |        |
|  | | Anomaly       | | Pattern       | | Prediction    | |        |
|  | | Detection     | | Recognition   | | Models        | |        |
|  | | - Isolation   | | - Flow        | | - Earnings    | |        |
|  | |   Forest      | |   Patterns    | | - Risk        | |        |
|  | | - Autoencoders| | - Regime      | | - Volatility  | |        |
|  | | - Statistical | |   Clusters    | | - Direction   | |        |
|  | +---------------+ +---------------+ +---------------+ |        |
|  +-------------------------------------------------------+        |
|            |                                                      |
|            v                                                      |
|  +-------------------------------------------------------+        |
|  |              Model Serving Layer                       |        |
|  +-------------------------------------------------------+        |
|  | Real-time Inference | Batch Predictions | A/B Testing |        |
|  +-------------------------------------------------------+        |
|            |                                                      |
|            v                                                      |
|  +-------------------------------------------------------+        |
|  |              Integration Layer                         |        |
|  +-------------------------------------------------------+        |
|  | SignalGenerator | AlertAggregator | PortfolioAnalyzer |        |
|  +-------------------------------------------------------+        |
|                                                                   |
+------------------------------------------------------------------+
```

---

## Feature 1: Anomaly Detection in Market Data

### Overview
Enhance the existing `accounting/anomaly_detection.py` with ML-based anomaly detection for real-time market data.

### Architecture

```python
# stanley/ml/anomaly/market_anomaly_detector.py

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class AnomalyMethod(Enum):
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "lof"
    AUTOENCODER = "autoencoder"
    STATISTICAL = "statistical"
    ENSEMBLE = "ensemble"


@dataclass
class MarketAnomalyResult:
    """Result from ML-based anomaly detection."""

    symbol: str
    timestamp: datetime
    anomaly_score: float  # 0-1, higher = more anomalous
    is_anomaly: bool
    method: AnomalyMethod
    features_contributing: Dict[str, float]
    confidence: float


class MarketAnomalyDetector:
    """
    ML-enhanced market anomaly detection.

    Detects anomalies in:
    - Price movements
    - Volume patterns
    - Spread behavior
    - Order flow imbalances
    - Cross-asset correlations
    """

    def __init__(
        self,
        contamination: float = 0.01,
        n_estimators: int = 100,
        methods: List[AnomalyMethod] = None,
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.methods = methods or [AnomalyMethod.ENSEMBLE]

        # Initialize models
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self._is_fitted = False

    def fit(self, historical_data: pd.DataFrame) -> "MarketAnomalyDetector":
        """
        Fit anomaly detection models on historical data.

        Args:
            historical_data: DataFrame with columns:
                - return_1d, return_5d, return_20d
                - volume_ratio (current/avg)
                - spread_ratio
                - volatility_ratio
                - flow_imbalance
        """
        features = self._extract_features(historical_data)
        scaled_features = self.scaler.fit_transform(features)
        self.isolation_forest.fit(scaled_features)
        self._is_fitted = True
        return self

    def detect(
        self,
        current_data: pd.DataFrame,
        symbol: str,
    ) -> MarketAnomalyResult:
        """Detect anomalies in current market data."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted before detection")

        features = self._extract_features(current_data)
        scaled_features = self.scaler.transform(features)

        # Get anomaly scores
        scores = self.isolation_forest.decision_function(scaled_features)
        predictions = self.isolation_forest.predict(scaled_features)

        # Normalize score to 0-1 range
        anomaly_score = 1 - (scores[-1] - scores.min()) / (scores.max() - scores.min())

        return MarketAnomalyResult(
            symbol=symbol,
            timestamp=datetime.now(),
            anomaly_score=float(anomaly_score),
            is_anomaly=predictions[-1] == -1,
            method=AnomalyMethod.ISOLATION_FOREST,
            features_contributing=self._get_feature_contributions(
                features.iloc[-1], scaled_features[-1]
            ),
            confidence=self._calculate_confidence(scores[-1], scores),
        )

    def _extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Extract anomaly detection features."""
        features = pd.DataFrame()

        # Price-based features
        features['return_1d'] = data['close'].pct_change(1)
        features['return_5d'] = data['close'].pct_change(5)
        features['return_20d'] = data['close'].pct_change(20)
        features['return_volatility'] = features['return_1d'].rolling(20).std()

        # Volume features
        features['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
        features['volume_zscore'] = (
            (data['volume'] - data['volume'].rolling(20).mean()) /
            data['volume'].rolling(20).std()
        )

        # Spread and liquidity
        if 'high' in data and 'low' in data:
            features['spread'] = (data['high'] - data['low']) / data['close']
            features['spread_ratio'] = features['spread'] / features['spread'].rolling(20).mean()

        # Flow imbalance (if available)
        if 'buy_volume' in data and 'sell_volume' in data:
            total_volume = data['buy_volume'] + data['sell_volume']
            features['flow_imbalance'] = (
                (data['buy_volume'] - data['sell_volume']) / total_volume
            )

        return features.dropna()
```

### Integration Points

1. **MoneyFlowAnalyzer Enhancement**:
   - Add ML anomaly detection to `detect_dark_pool_alerts()`
   - Enhance `detect_unusual_volume()` with Isolation Forest

2. **AlertAggregator Enhancement**:
   - Add `ML_ANOMALY` alert type
   - Support confidence scores from ML models

### Training Pipeline

```python
# stanley/ml/training/anomaly_trainer.py

class AnomalyModelTrainer:
    """Training pipeline for anomaly detection models."""

    def __init__(
        self,
        data_manager,
        model_registry,
        experiment_name: str = "market_anomaly_detection",
    ):
        self.data_manager = data_manager
        self.model_registry = model_registry
        self.experiment_name = experiment_name

    async def train_and_register(
        self,
        symbols: List[str],
        lookback_days: int = 252,
        validation_split: float = 0.2,
    ) -> str:
        """
        Train anomaly detection model and register with MLflow.

        Returns:
            Model version ID
        """
        # Collect training data
        training_data = await self._collect_training_data(symbols, lookback_days)

        # Split data
        train_data, val_data = self._temporal_split(training_data, validation_split)

        # Train model
        detector = MarketAnomalyDetector()
        detector.fit(train_data)

        # Evaluate on validation set
        metrics = self._evaluate(detector, val_data)

        # Register model
        model_id = self.model_registry.register(
            model=detector,
            metrics=metrics,
            experiment_name=self.experiment_name,
        )

        return model_id
```

---

## Feature 2: Pattern Recognition in Flows

### Overview
Identify recurring patterns in institutional money flows that precede significant price movements.

### Architecture

```python
# stanley/ml/patterns/flow_pattern_recognizer.py

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import dtw
from sklearn.preprocessing import MinMaxScaler


class FlowPattern(Enum):
    """Recognized flow patterns."""

    ACCUMULATION = "accumulation"          # Gradual institutional buying
    DISTRIBUTION = "distribution"          # Gradual institutional selling
    CLIMAX_BUY = "climax_buy"             # Sudden surge in buying
    CLIMAX_SELL = "climax_sell"           # Sudden surge in selling
    ROTATION_IN = "rotation_in"           # Sector rotation entering
    ROTATION_OUT = "rotation_out"         # Sector rotation exiting
    SQUEEZE_SETUP = "squeeze_setup"       # Pre-squeeze accumulation
    INSTITUTIONAL_PIVOT = "institutional_pivot"  # Direction change
    UNKNOWN = "unknown"


@dataclass
class PatternMatch:
    """Pattern recognition result."""

    pattern: FlowPattern
    confidence: float
    start_date: datetime
    end_date: datetime
    historical_outcomes: Dict[str, float]  # avg_return, win_rate, etc.
    similar_instances: int


class FlowPatternRecognizer:
    """
    Recognize institutional flow patterns using:
    - Dynamic Time Warping for pattern matching
    - Clustering for pattern discovery
    - Template matching for known patterns
    """

    def __init__(
        self,
        pattern_length: int = 20,
        min_confidence: float = 0.7,
    ):
        self.pattern_length = pattern_length
        self.min_confidence = min_confidence
        self.scaler = MinMaxScaler()

        # Pattern templates (learned from historical data)
        self.pattern_templates: Dict[FlowPattern, np.ndarray] = {}
        self.historical_outcomes: Dict[FlowPattern, List[float]] = {}

    def learn_patterns(
        self,
        historical_flows: pd.DataFrame,
        price_data: pd.DataFrame,
        forward_returns: pd.DataFrame,
    ) -> None:
        """
        Learn pattern templates from historical data.

        Uses DBSCAN clustering to discover recurring patterns,
        then associates each cluster with forward returns.
        """
        # Extract flow sequences
        sequences = self._extract_sequences(historical_flows)

        # Cluster similar patterns
        clusters = self._cluster_patterns(sequences)

        # Associate clusters with outcomes
        self._associate_outcomes(clusters, forward_returns)

        # Build pattern templates
        self._build_templates(clusters)

    def recognize(
        self,
        current_flows: pd.DataFrame,
        symbol: str,
    ) -> List[PatternMatch]:
        """
        Recognize patterns in current flow data.

        Args:
            current_flows: Recent flow data (at least pattern_length days)
            symbol: Stock symbol

        Returns:
            List of recognized patterns with confidence scores
        """
        if len(current_flows) < self.pattern_length:
            return []

        # Extract current pattern
        current_sequence = self._extract_current_sequence(current_flows)

        matches = []

        # Match against templates using DTW
        for pattern_type, template in self.pattern_templates.items():
            distance = self._dtw_distance(current_sequence, template)
            similarity = 1 / (1 + distance)

            if similarity >= self.min_confidence:
                outcomes = self.historical_outcomes.get(pattern_type, {})

                matches.append(PatternMatch(
                    pattern=pattern_type,
                    confidence=similarity,
                    start_date=current_flows.index[0],
                    end_date=current_flows.index[-1],
                    historical_outcomes=outcomes,
                    similar_instances=len(outcomes.get('returns', [])),
                ))

        return sorted(matches, key=lambda x: x.confidence, reverse=True)

    def _extract_sequences(self, flows: pd.DataFrame) -> List[np.ndarray]:
        """Extract fixed-length sequences from flow data."""
        sequences = []

        for i in range(len(flows) - self.pattern_length):
            seq = flows.iloc[i:i + self.pattern_length].values
            scaled_seq = self.scaler.fit_transform(seq.reshape(-1, 1)).flatten()
            sequences.append(scaled_seq)

        return sequences

    def _cluster_patterns(self, sequences: List[np.ndarray]) -> Dict[int, List[int]]:
        """Cluster similar patterns using DTW-based distance."""
        # Compute pairwise DTW distances
        n = len(sequences)
        distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                dist = self._dtw_distance(sequences[i], sequences[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist

        # Cluster using DBSCAN
        clustering = DBSCAN(eps=0.3, min_samples=5, metric='precomputed')
        labels = clustering.fit_predict(distance_matrix)

        # Group by cluster
        clusters = {}
        for idx, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(idx)

        return clusters

    def _dtw_distance(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """Compute Dynamic Time Warping distance."""
        n, m = len(seq1), len(seq2)
        dtw_matrix = np.full((n + 1, m + 1), np.inf)
        dtw_matrix[0, 0] = 0

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(seq1[i-1] - seq2[j-1])
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i-1, j],
                    dtw_matrix[i, j-1],
                    dtw_matrix[i-1, j-1]
                )

        return dtw_matrix[n, m]
```

### Integration with MoneyFlowAnalyzer

```python
# Enhancement to stanley/analytics/money_flow.py

class MoneyFlowAnalyzer:
    def __init__(
        self,
        data_manager=None,
        thresholds: Optional[AlertThresholds] = None,
        pattern_recognizer: Optional[FlowPatternRecognizer] = None,
    ):
        # ... existing initialization ...
        self.pattern_recognizer = pattern_recognizer

    def get_comprehensive_analysis(
        self,
        symbol: str,
        lookback_days: int = TRADING_DAYS_1_MONTH,
    ) -> Dict[str, Any]:
        """Enhanced with pattern recognition."""
        # ... existing analysis ...

        # Add pattern recognition
        if self.pattern_recognizer:
            flow_data = self._get_flow_data_for_patterns(symbol, lookback_days)
            patterns = self.pattern_recognizer.recognize(flow_data, symbol)

            result["patterns"] = {
                "detected": [p.__dict__ for p in patterns],
                "primary_pattern": patterns[0].pattern.value if patterns else None,
                "pattern_confidence": patterns[0].confidence if patterns else 0.0,
            }

        return result
```

---

## Feature 3: Sentiment Analysis from News

### Overview
Extract sentiment signals from financial news and SEC filings using transformer models.

### Architecture

```python
# stanley/ml/sentiment/news_sentiment_analyzer.py

from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np


class SentimentSource(Enum):
    NEWS = "news"
    SEC_FILING = "sec_filing"
    EARNINGS_CALL = "earnings_call"
    SOCIAL_MEDIA = "social_media"


@dataclass
class SentimentResult:
    """Sentiment analysis result."""

    symbol: str
    source: SentimentSource
    sentiment_score: float  # -1 (bearish) to 1 (bullish)
    confidence: float
    key_phrases: List[str]
    timestamp: datetime
    source_text: str  # Original text (truncated)


class FinancialSentimentAnalyzer:
    """
    Financial sentiment analysis using FinBERT or similar models.

    Specialized for:
    - Earnings reports
    - SEC filings (10-K, 10-Q, 8-K)
    - Financial news
    - Analyst reports
    """

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: str = None,
        batch_size: int = 16,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size

        # Load FinBERT model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Label mapping for FinBERT
        self.label_map = {0: -1, 1: 0, 2: 1}  # negative, neutral, positive

    def analyze_text(
        self,
        text: str,
        symbol: str,
        source: SentimentSource,
    ) -> SentimentResult:
        """
        Analyze sentiment of a single text.

        Args:
            text: Text to analyze
            symbol: Associated stock symbol
            source: Source type of the text

        Returns:
            SentimentResult with sentiment score and metadata
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)

        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)

        # Calculate weighted sentiment score
        sentiment_score = self._calculate_sentiment_score(probs[0])
        confidence = self._calculate_confidence(probs[0])

        # Extract key phrases
        key_phrases = self._extract_key_phrases(text)

        return SentimentResult(
            symbol=symbol,
            source=source,
            sentiment_score=float(sentiment_score),
            confidence=float(confidence),
            key_phrases=key_phrases,
            timestamp=datetime.now(),
            source_text=text[:500] + "..." if len(text) > 500 else text,
        )

    def analyze_batch(
        self,
        texts: List[Tuple[str, str, SentimentSource]],  # (text, symbol, source)
    ) -> List[SentimentResult]:
        """Batch sentiment analysis for efficiency."""
        results = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_texts = [t[0] for t in batch]

            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self.device)

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)

            # Process each result
            for j, (text, symbol, source) in enumerate(batch):
                sentiment_score = self._calculate_sentiment_score(probs[j])
                confidence = self._calculate_confidence(probs[j])

                results.append(SentimentResult(
                    symbol=symbol,
                    source=source,
                    sentiment_score=float(sentiment_score),
                    confidence=float(confidence),
                    key_phrases=self._extract_key_phrases(text),
                    timestamp=datetime.now(),
                    source_text=text[:500] + "..." if len(text) > 500 else text,
                ))

        return results

    def _calculate_sentiment_score(self, probs: torch.Tensor) -> float:
        """Calculate weighted sentiment score from probabilities."""
        # probs: [negative, neutral, positive]
        weights = torch.tensor([-1, 0, 1], dtype=torch.float32)
        return (probs.cpu() * weights).sum().item()

    def _calculate_confidence(self, probs: torch.Tensor) -> float:
        """Calculate confidence as max probability."""
        return probs.max().item()

    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key financial phrases."""
        # Simple keyword extraction (could use more sophisticated NLP)
        keywords = [
            "revenue", "earnings", "profit", "loss", "growth",
            "decline", "beat", "miss", "guidance", "outlook",
            "upgrade", "downgrade", "bullish", "bearish",
        ]

        text_lower = text.lower()
        found = [kw for kw in keywords if kw in text_lower]
        return found[:5]  # Return top 5


class AggregateSentimentTracker:
    """
    Track and aggregate sentiment across multiple sources.

    Provides:
    - Rolling sentiment scores
    - Sentiment momentum
    - Cross-source agreement
    """

    def __init__(
        self,
        sentiment_analyzer: FinancialSentimentAnalyzer,
        decay_factor: float = 0.95,  # Daily decay
    ):
        self.sentiment_analyzer = sentiment_analyzer
        self.decay_factor = decay_factor

        # Sentiment history by symbol
        self.sentiment_history: Dict[str, List[SentimentResult]] = {}

    def update(self, results: List[SentimentResult]) -> None:
        """Update sentiment history with new results."""
        for result in results:
            if result.symbol not in self.sentiment_history:
                self.sentiment_history[result.symbol] = []
            self.sentiment_history[result.symbol].append(result)

    def get_aggregate_sentiment(
        self,
        symbol: str,
        lookback_days: int = 30,
    ) -> Dict[str, float]:
        """
        Get aggregate sentiment for a symbol.

        Returns:
            Dict with sentiment metrics:
            - current_sentiment: Decay-weighted average
            - sentiment_momentum: Rate of change
            - source_agreement: Cross-source correlation
            - confidence: Average confidence
        """
        if symbol not in self.sentiment_history:
            return {
                "current_sentiment": 0.0,
                "sentiment_momentum": 0.0,
                "source_agreement": 0.0,
                "confidence": 0.0,
            }

        history = self.sentiment_history[symbol]
        cutoff = datetime.now() - timedelta(days=lookback_days)
        recent = [r for r in history if r.timestamp > cutoff]

        if not recent:
            return {
                "current_sentiment": 0.0,
                "sentiment_momentum": 0.0,
                "source_agreement": 0.0,
                "confidence": 0.0,
            }

        # Decay-weighted average
        weights = []
        scores = []
        now = datetime.now()

        for r in recent:
            age_days = (now - r.timestamp).days
            weight = self.decay_factor ** age_days
            weights.append(weight)
            scores.append(r.sentiment_score)

        current_sentiment = np.average(scores, weights=weights)

        # Sentiment momentum (recent vs older)
        mid_point = len(recent) // 2
        if mid_point > 0:
            recent_avg = np.mean([r.sentiment_score for r in recent[:mid_point]])
            older_avg = np.mean([r.sentiment_score for r in recent[mid_point:]])
            momentum = recent_avg - older_avg
        else:
            momentum = 0.0

        # Source agreement
        by_source = {}
        for r in recent:
            source = r.source.value
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(r.sentiment_score)

        if len(by_source) > 1:
            source_means = [np.mean(v) for v in by_source.values()]
            agreement = 1 - np.std(source_means)
        else:
            agreement = 1.0

        # Average confidence
        avg_confidence = np.mean([r.confidence for r in recent])

        return {
            "current_sentiment": float(current_sentiment),
            "sentiment_momentum": float(momentum),
            "source_agreement": float(agreement),
            "confidence": float(avg_confidence),
        }
```

---

## Feature 4: Earnings Prediction

### Overview
Predict earnings surprises and estimate probability of beats/misses.

### Architecture

```python
# stanley/ml/prediction/earnings_predictor.py

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
import lightgbm as lgb


@dataclass
class EarningsPrediction:
    """Earnings prediction result."""

    symbol: str
    fiscal_period: str

    # EPS predictions
    predicted_eps: float
    eps_confidence_interval: Tuple[float, float]  # 95% CI
    consensus_estimate: float

    # Beat/miss probability
    beat_probability: float
    miss_probability: float
    inline_probability: float

    # Surprise magnitude
    predicted_surprise_pct: float

    # Feature importance
    top_features: Dict[str, float]

    # Model metadata
    model_version: str
    prediction_date: datetime


class EarningsPredictor:
    """
    Predict earnings outcomes using:
    - Historical earnings patterns
    - Fundamental ratios
    - Industry trends
    - Analyst revision patterns
    - Alternative data signals
    """

    def __init__(
        self,
        eps_model_type: str = "xgboost",
        classification_model_type: str = "xgboost",
    ):
        self.eps_model_type = eps_model_type
        self.classification_model_type = classification_model_type

        # Initialize models
        self.eps_regressor = self._create_regressor(eps_model_type)
        self.direction_classifier = self._create_classifier(classification_model_type)

        self.feature_names: List[str] = []
        self._is_fitted = False

    def _create_regressor(self, model_type: str):
        """Create regression model for EPS prediction."""
        if model_type == "xgboost":
            return XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
            )
        elif model_type == "lightgbm":
            return lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
            )
        else:
            return GradientBoostingRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                random_state=42,
            )

    def _create_classifier(self, model_type: str):
        """Create classifier for beat/miss prediction."""
        if model_type == "xgboost":
            return XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42,
            )
        elif model_type == "lightgbm":
            return lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                random_state=42,
            )
        else:
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=6,
                random_state=42,
            )

    def fit(
        self,
        historical_data: pd.DataFrame,
        eps_actuals: pd.Series,
        beat_labels: pd.Series,  # 1=beat, 0=inline, -1=miss
    ) -> "EarningsPredictor":
        """
        Train earnings prediction models.

        Args:
            historical_data: Feature matrix with columns:
                - prior_eps_growth: YoY EPS growth
                - estimate_revision_30d: 30-day estimate revision
                - estimate_revision_90d: 90-day estimate revision
                - beat_rate_4q: Beat rate over last 4 quarters
                - revenue_growth: Revenue growth rate
                - margin_trend: Operating margin trend
                - sector_momentum: Sector relative performance
                - money_flow_score: Institutional flow score
                - sentiment_score: News sentiment
                - guidance_change: Recent guidance revisions
            eps_actuals: Actual EPS values
            beat_labels: Beat/miss labels
        """
        self.feature_names = list(historical_data.columns)

        # Train EPS regressor
        self.eps_regressor.fit(historical_data, eps_actuals)

        # Train direction classifier
        # Convert to 3-class: 0=miss, 1=inline, 2=beat
        class_labels = beat_labels.map({-1: 0, 0: 1, 1: 2})
        self.direction_classifier.fit(historical_data, class_labels)

        self._is_fitted = True
        return self

    def predict(
        self,
        features: pd.DataFrame,
        symbol: str,
        fiscal_period: str,
        consensus_estimate: float,
    ) -> EarningsPrediction:
        """
        Predict earnings for upcoming quarter.

        Args:
            features: Feature vector for prediction
            symbol: Stock symbol
            fiscal_period: Fiscal period (e.g., "Q4 2024")
            consensus_estimate: Current analyst consensus

        Returns:
            EarningsPrediction with detailed forecast
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # EPS prediction
        eps_pred = self.eps_regressor.predict(features)[0]

        # Get prediction interval via bootstrapping
        ci_low, ci_high = self._bootstrap_confidence_interval(features)

        # Beat/miss probabilities
        probs = self.direction_classifier.predict_proba(features)[0]
        miss_prob, inline_prob, beat_prob = probs

        # Predicted surprise
        if consensus_estimate != 0:
            surprise_pct = (eps_pred - consensus_estimate) / abs(consensus_estimate) * 100
        else:
            surprise_pct = 0.0

        # Feature importance
        top_features = self._get_top_features(features)

        return EarningsPrediction(
            symbol=symbol,
            fiscal_period=fiscal_period,
            predicted_eps=float(eps_pred),
            eps_confidence_interval=(float(ci_low), float(ci_high)),
            consensus_estimate=consensus_estimate,
            beat_probability=float(beat_prob),
            miss_probability=float(miss_prob),
            inline_probability=float(inline_prob),
            predicted_surprise_pct=float(surprise_pct),
            top_features=top_features,
            model_version="1.0.0",
            prediction_date=datetime.now(),
        )

    def _bootstrap_confidence_interval(
        self,
        features: pd.DataFrame,
        n_bootstrap: int = 100,
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """Calculate confidence interval via bootstrapping."""
        predictions = []

        for _ in range(n_bootstrap):
            # Add small noise to features
            noisy_features = features + np.random.normal(0, 0.01, features.shape)
            pred = self.eps_regressor.predict(noisy_features)[0]
            predictions.append(pred)

        lower = np.percentile(predictions, (1 - confidence) / 2 * 100)
        upper = np.percentile(predictions, (1 + confidence) / 2 * 100)

        return lower, upper

    def _get_top_features(
        self,
        features: pd.DataFrame,
        top_n: int = 5,
    ) -> Dict[str, float]:
        """Get top contributing features."""
        if hasattr(self.eps_regressor, 'feature_importances_'):
            importances = self.eps_regressor.feature_importances_
        else:
            importances = np.ones(len(self.feature_names)) / len(self.feature_names)

        feature_importance = dict(zip(self.feature_names, importances))
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return dict(sorted_features[:top_n])
```

### Integration with ResearchAnalyzer

```python
# Enhancement to stanley/research/research_analyzer.py

class ResearchAnalyzer:
    def __init__(
        self,
        data_manager=None,
        accounting_analyzer=None,
        earnings_predictor: Optional[EarningsPredictor] = None,
    ):
        # ... existing initialization ...
        self.earnings_predictor = earnings_predictor

    async def get_earnings_forecast(
        self,
        symbol: str,
    ) -> Dict[str, Any]:
        """
        Get ML-enhanced earnings forecast.
        """
        if not self.earnings_predictor:
            return {"error": "Earnings predictor not configured"}

        # Build feature vector
        features = await self._build_earnings_features(symbol)

        # Get consensus estimate
        consensus = await self._get_consensus_estimate(symbol)

        # Get prediction
        prediction = self.earnings_predictor.predict(
            features=features,
            symbol=symbol,
            fiscal_period=self._get_next_fiscal_period(symbol),
            consensus_estimate=consensus,
        )

        return {
            "prediction": prediction.__dict__,
            "recommendation": self._generate_recommendation(prediction),
        }
```

---

## Feature 5: Regime Classification

### Overview
Enhance the existing `macro/regime_detector.py` with ML-based regime detection using Hidden Markov Models and clustering.

### Architecture

```python
# stanley/ml/regime/ml_regime_detector.py

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class MarketRegime(Enum):
    """ML-detected market regimes."""

    RISK_ON_HIGH_VOL = "risk_on_high_vol"
    RISK_ON_LOW_VOL = "risk_on_low_vol"
    RISK_OFF_ORDERLY = "risk_off_orderly"
    RISK_OFF_PANIC = "risk_off_panic"
    TRANSITION = "transition"
    RANGE_BOUND = "range_bound"


@dataclass
class RegimeState:
    """Current regime state from ML model."""

    regime: MarketRegime
    probability: float
    regime_duration_days: int
    transition_probabilities: Dict[str, float]
    regime_features: Dict[str, float]


class HMMRegimeDetector:
    """
    Hidden Markov Model for market regime detection.

    States represent unobservable market regimes,
    observed through market indicators.
    """

    def __init__(
        self,
        n_regimes: int = 4,
        covariance_type: str = "diag",
    ):
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type

        self.model = hmm.GaussianHMM(
            n_components=n_regimes,
            covariance_type=covariance_type,
            n_iter=100,
            random_state=42,
        )

        self.scaler = StandardScaler()
        self._is_fitted = False
        self.regime_mapping: Dict[int, MarketRegime] = {}

    def fit(
        self,
        observations: pd.DataFrame,
    ) -> "HMMRegimeDetector":
        """
        Fit HMM on historical market observations.

        Args:
            observations: DataFrame with columns:
                - vix_level
                - credit_spread
                - yield_curve_slope
                - equity_momentum
                - cross_asset_correlation
                - volume_ratio
        """
        # Standardize features
        scaled_obs = self.scaler.fit_transform(observations)

        # Fit HMM
        self.model.fit(scaled_obs)

        # Map HMM states to semantic regimes
        self._map_regimes(observations, scaled_obs)

        self._is_fitted = True
        return self

    def detect_regime(
        self,
        current_observations: pd.DataFrame,
    ) -> RegimeState:
        """
        Detect current market regime.

        Args:
            current_observations: Recent market observations

        Returns:
            RegimeState with current regime and probabilities
        """
        if not self._is_fitted:
            raise ValueError("Model must be fitted first")

        scaled_obs = self.scaler.transform(current_observations)

        # Get state probabilities
        state_probs = self.model.predict_proba(scaled_obs)[-1]
        current_state = np.argmax(state_probs)

        # Get transition matrix
        trans_matrix = self.model.transmat_

        # Calculate regime duration
        states = self.model.predict(scaled_obs)
        duration = self._calculate_duration(states)

        # Map to semantic regime
        regime = self.regime_mapping[current_state]

        return RegimeState(
            regime=regime,
            probability=float(state_probs[current_state]),
            regime_duration_days=duration,
            transition_probabilities={
                self.regime_mapping[i].value: float(trans_matrix[current_state, i])
                for i in range(self.n_regimes)
            },
            regime_features={
                col: float(current_observations[col].iloc[-1])
                for col in current_observations.columns
            },
        )

    def _map_regimes(
        self,
        observations: pd.DataFrame,
        scaled_obs: np.ndarray,
    ) -> None:
        """Map HMM states to semantic regimes based on characteristics."""
        states = self.model.predict(scaled_obs)

        # Analyze each state's characteristics
        for state in range(self.n_regimes):
            mask = states == state
            state_obs = observations[mask]

            if len(state_obs) == 0:
                self.regime_mapping[state] = MarketRegime.TRANSITION
                continue

            avg_vix = state_obs['vix_level'].mean() if 'vix_level' in state_obs else 20
            avg_return = state_obs['equity_momentum'].mean() if 'equity_momentum' in state_obs else 0

            # Classify based on characteristics
            if avg_vix > 30:
                if avg_return < -0.01:
                    self.regime_mapping[state] = MarketRegime.RISK_OFF_PANIC
                else:
                    self.regime_mapping[state] = MarketRegime.RISK_OFF_ORDERLY
            elif avg_vix < 15:
                if avg_return > 0.005:
                    self.regime_mapping[state] = MarketRegime.RISK_ON_LOW_VOL
                else:
                    self.regime_mapping[state] = MarketRegime.RANGE_BOUND
            else:
                if avg_return > 0.005:
                    self.regime_mapping[state] = MarketRegime.RISK_ON_HIGH_VOL
                else:
                    self.regime_mapping[state] = MarketRegime.TRANSITION

    def _calculate_duration(self, states: np.ndarray) -> int:
        """Calculate current regime duration."""
        if len(states) == 0:
            return 0

        current_state = states[-1]
        duration = 0

        for i in range(len(states) - 1, -1, -1):
            if states[i] == current_state:
                duration += 1
            else:
                break

        return duration
```

### Integration with MacroRegimeDetector

```python
# Enhancement to stanley/macro/regime_detector.py

class MacroRegimeDetector:
    def __init__(
        self,
        dbnomics_adapter=None,
        data_manager=None,
        ml_regime_detector: Optional[HMMRegimeDetector] = None,
    ):
        # ... existing initialization ...
        self.ml_regime_detector = ml_regime_detector

    async def get_regime_state(
        self,
        country: str = "USA",
        use_ml: bool = True,
    ) -> MacroRegimeState:
        """
        Get regime state with optional ML enhancement.
        """
        # Get rule-based regime (existing logic)
        rule_based_state = await self._get_rule_based_regime(country)

        if use_ml and self.ml_regime_detector:
            # Get ML regime
            observations = await self._build_ml_observations(country)
            ml_state = self.ml_regime_detector.detect_regime(observations)

            # Combine rule-based and ML regimes
            combined_state = self._combine_regimes(rule_based_state, ml_state)
            return combined_state

        return rule_based_state
```

---

## Feature 6: Risk Factor Modeling

### Overview
Build ML-based risk factor models for better portfolio risk decomposition.

### Architecture

```python
# stanley/ml/risk/factor_model.py

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler


@dataclass
class FactorExposure:
    """Factor exposure for a security or portfolio."""

    symbol: str
    factor_loadings: Dict[str, float]
    residual_risk: float
    r_squared: float
    factor_contribution: Dict[str, float]  # Contribution to total risk


@dataclass
class FactorModel:
    """Complete factor model specification."""

    factor_names: List[str]
    factor_returns: pd.DataFrame
    factor_covariance: pd.DataFrame
    model_r_squared: float


class MLFactorModel:
    """
    Machine learning enhanced factor model.

    Features:
    - PCA-based factor extraction
    - Dynamic factor selection
    - Non-linear factor interactions
    - Regime-dependent loadings
    """

    def __init__(
        self,
        n_statistical_factors: int = 5,
        use_macro_factors: bool = True,
        use_style_factors: bool = True,
    ):
        self.n_statistical_factors = n_statistical_factors
        self.use_macro_factors = use_macro_factors
        self.use_style_factors = use_style_factors

        self.pca = PCA(n_components=n_statistical_factors)
        self.scaler = StandardScaler()

        # Factor models for each security
        self.security_models: Dict[str, ElasticNet] = {}

        # Factor definitions
        self.macro_factors = [
            'market_return',
            'interest_rate_change',
            'credit_spread_change',
            'vix_change',
            'oil_return',
            'dollar_return',
        ]

        self.style_factors = [
            'size',  # Market cap
            'value',  # Book-to-market
            'momentum',  # 12-1 month return
            'quality',  # ROE
            'volatility',  # Low vol factor
        ]

    def fit(
        self,
        returns: pd.DataFrame,
        macro_data: pd.DataFrame,
        style_characteristics: pd.DataFrame,
    ) -> "MLFactorModel":
        """
        Fit factor model.

        Args:
            returns: Security returns (columns = securities)
            macro_data: Macro factor returns
            style_characteristics: Security style characteristics
        """
        # Extract statistical factors via PCA
        scaled_returns = self.scaler.fit_transform(returns)
        self.pca.fit(scaled_returns)
        statistical_factors = self.pca.transform(scaled_returns)

        # Combine all factors
        all_factors = self._combine_factors(
            statistical_factors,
            macro_data,
            style_characteristics,
            returns.index,
        )

        # Fit factor model for each security
        for symbol in returns.columns:
            model = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=1000)
            model.fit(all_factors, returns[symbol])
            self.security_models[symbol] = model

        return self

    def get_factor_exposures(
        self,
        symbol: str,
        current_factors: pd.DataFrame,
    ) -> FactorExposure:
        """Get current factor exposures for a security."""
        if symbol not in self.security_models:
            raise ValueError(f"No model fitted for {symbol}")

        model = self.security_models[symbol]

        # Get factor loadings
        factor_names = self._get_factor_names()
        loadings = dict(zip(factor_names, model.coef_))

        # Calculate residual risk
        predictions = model.predict(current_factors)
        # ... calculate residual

        return FactorExposure(
            symbol=symbol,
            factor_loadings=loadings,
            residual_risk=0.0,  # Calculate from residuals
            r_squared=model.score(current_factors, [0]),  # Placeholder
            factor_contribution={},  # Calculate contribution
        )

    def decompose_portfolio_risk(
        self,
        weights: Dict[str, float],
        current_factors: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Decompose portfolio risk into factor contributions.

        Returns:
            Dict mapping factor name to risk contribution
        """
        # Get weighted factor exposures
        portfolio_exposures = {}
        factor_names = self._get_factor_names()

        for factor in factor_names:
            portfolio_exposures[factor] = sum(
                weights.get(symbol, 0) *
                self.security_models[symbol].coef_[factor_names.index(factor)]
                for symbol in weights
                if symbol in self.security_models
            )

        # Calculate factor covariance contribution
        # ... implementation details

        return portfolio_exposures

    def _combine_factors(
        self,
        statistical_factors: np.ndarray,
        macro_data: pd.DataFrame,
        style_characteristics: pd.DataFrame,
        index: pd.Index,
    ) -> pd.DataFrame:
        """Combine all factor sources."""
        all_factors = pd.DataFrame(index=index)

        # Add statistical factors
        for i in range(self.n_statistical_factors):
            all_factors[f'stat_factor_{i+1}'] = statistical_factors[:, i]

        # Add macro factors
        if self.use_macro_factors:
            for col in self.macro_factors:
                if col in macro_data.columns:
                    all_factors[col] = macro_data[col].values[:len(index)]

        # Style factors would be cross-sectional

        return all_factors.fillna(0)

    def _get_factor_names(self) -> List[str]:
        """Get list of all factor names."""
        names = [f'stat_factor_{i+1}' for i in range(self.n_statistical_factors)]

        if self.use_macro_factors:
            names.extend(self.macro_factors)

        if self.use_style_factors:
            names.extend(self.style_factors)

        return names
```

---

## Feature 7: Portfolio Optimization

### Overview
ML-enhanced portfolio optimization with robust covariance estimation and Black-Litterman views.

### Architecture

```python
# stanley/ml/optimization/portfolio_optimizer.py

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf, GraphicalLasso


@dataclass
class OptimizationResult:
    """Portfolio optimization result."""

    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float

    # Risk decomposition
    factor_risk: Dict[str, float]
    idiosyncratic_risk: float

    # Constraints status
    constraints_satisfied: bool
    binding_constraints: List[str]


class MLPortfolioOptimizer:
    """
    ML-enhanced portfolio optimization.

    Features:
    - Robust covariance estimation (Ledoit-Wolf, Graphical Lasso)
    - Black-Litterman integration with ML views
    - Transaction cost optimization
    - Regime-aware optimization
    """

    def __init__(
        self,
        covariance_method: str = "ledoit_wolf",
        risk_free_rate: float = 0.04,
    ):
        self.covariance_method = covariance_method
        self.risk_free_rate = risk_free_rate

    def estimate_covariance(
        self,
        returns: pd.DataFrame,
        method: str = None,
    ) -> pd.DataFrame:
        """
        Estimate robust covariance matrix.

        Args:
            returns: Historical returns
            method: Estimation method (ledoit_wolf, graphical_lasso, sample)
        """
        method = method or self.covariance_method

        if method == "ledoit_wolf":
            estimator = LedoitWolf()
            estimator.fit(returns)
            cov = estimator.covariance_
        elif method == "graphical_lasso":
            estimator = GraphicalLasso(alpha=0.01)
            estimator.fit(returns)
            cov = estimator.covariance_
        else:
            cov = returns.cov().values

        return pd.DataFrame(
            cov * 252,  # Annualize
            index=returns.columns,
            columns=returns.columns,
        )

    def optimize(
        self,
        expected_returns: pd.Series,
        covariance: pd.DataFrame,
        constraints: Optional[Dict] = None,
        objective: str = "max_sharpe",
    ) -> OptimizationResult:
        """
        Optimize portfolio weights.

        Args:
            expected_returns: Expected returns by asset
            covariance: Covariance matrix
            constraints: Optional constraints dict
            objective: Optimization objective
        """
        n_assets = len(expected_returns)
        symbols = list(expected_returns.index)

        # Default constraints
        bounds = [(0.0, 0.2) for _ in range(n_assets)]  # Max 20% per asset

        # Sum to 1 constraint
        constraints_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        ]

        # Add custom constraints
        if constraints:
            # Sector constraints, etc.
            pass

        # Initial weights
        w0 = np.ones(n_assets) / n_assets

        # Objective function
        if objective == "max_sharpe":
            def neg_sharpe(w):
                ret = np.dot(w, expected_returns)
                vol = np.sqrt(np.dot(w.T, np.dot(covariance, w)))
                return -(ret - self.risk_free_rate) / vol
            obj_func = neg_sharpe
        elif objective == "min_variance":
            def portfolio_var(w):
                return np.dot(w.T, np.dot(covariance, w))
            obj_func = portfolio_var
        else:
            raise ValueError(f"Unknown objective: {objective}")

        # Optimize
        result = minimize(
            obj_func,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list,
        )

        optimal_weights = result.x

        # Calculate metrics
        exp_ret = np.dot(optimal_weights, expected_returns)
        exp_vol = np.sqrt(np.dot(optimal_weights.T, np.dot(covariance, optimal_weights)))
        sharpe = (exp_ret - self.risk_free_rate) / exp_vol

        return OptimizationResult(
            weights=dict(zip(symbols, optimal_weights)),
            expected_return=float(exp_ret),
            expected_volatility=float(exp_vol),
            sharpe_ratio=float(sharpe),
            factor_risk={},
            idiosyncratic_risk=0.0,
            constraints_satisfied=result.success,
            binding_constraints=[],
        )

    def black_litterman(
        self,
        market_weights: pd.Series,
        covariance: pd.DataFrame,
        views: Dict[str, float],
        view_confidence: Dict[str, float],
        tau: float = 0.05,
    ) -> pd.Series:
        """
        Black-Litterman expected returns with ML views.

        Args:
            market_weights: Market cap weights
            covariance: Covariance matrix
            views: Absolute or relative views from ML models
            view_confidence: Confidence in each view
            tau: Uncertainty scaling parameter
        """
        # Calculate equilibrium returns
        risk_aversion = 2.5
        pi = risk_aversion * covariance.dot(market_weights)

        # Construct P matrix (view matrix)
        P = np.zeros((len(views), len(market_weights)))
        Q = np.zeros(len(views))
        omega_diag = np.zeros(len(views))

        symbols = list(market_weights.index)

        for i, (asset, view_return) in enumerate(views.items()):
            if asset in symbols:
                P[i, symbols.index(asset)] = 1
                Q[i] = view_return
                omega_diag[i] = (1 - view_confidence.get(asset, 0.5)) * 0.1

        Omega = np.diag(omega_diag)

        # Black-Litterman formula
        tau_sigma = tau * covariance

        # Posterior expected returns
        M_inverse = np.linalg.inv(
            np.linalg.inv(tau_sigma) + P.T @ np.linalg.inv(Omega) @ P
        )

        bl_returns = M_inverse @ (
            np.linalg.inv(tau_sigma) @ pi +
            P.T @ np.linalg.inv(Omega) @ Q
        )

        return pd.Series(bl_returns, index=market_weights.index)
```

---

## Feature 8: Signal Generation with ML

### Overview
Enhance the existing `signals/signal_generator.py` with ML-based signal generation and combination.

### Architecture

```python
# stanley/ml/signals/ml_signal_generator.py

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


@dataclass
class MLSignal:
    """ML-generated trading signal."""

    symbol: str
    signal_score: float  # -1 to 1
    direction: str  # "long", "short", "neutral"
    confidence: float
    probability_up: float
    probability_down: float

    # Feature attributions
    feature_contributions: Dict[str, float]

    # Model metadata
    model_version: str
    ensemble_agreement: float


class MLSignalGenerator:
    """
    ML-enhanced signal generation combining multiple models.

    Uses:
    - Stacking ensemble for robust predictions
    - Feature importance for explainability
    - Calibrated probabilities
    """

    def __init__(
        self,
        base_models: List[str] = None,
        meta_model: str = "logistic",
    ):
        base_models = base_models or ["xgb", "gbm", "rf"]

        # Build base estimators
        estimators = []
        if "xgb" in base_models:
            estimators.append(('xgb', XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            )))
        if "gbm" in base_models:
            estimators.append(('gbm', GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            )))

        # Meta-learner
        if meta_model == "logistic":
            final_estimator = LogisticRegression(max_iter=1000)
        else:
            final_estimator = XGBClassifier(n_estimators=50)

        self.model = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5,
            stack_method='predict_proba',
        )

        self.feature_names: List[str] = []
        self._is_fitted = False

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.Series,  # -1, 0, 1 for down, neutral, up
    ) -> "MLSignalGenerator":
        """
        Train signal generation model.

        Args:
            features: Feature matrix with columns:
                - Technical features (momentum, volatility, etc.)
                - Fundamental features (valuation, growth, etc.)
                - Alternative data (sentiment, flows, etc.)
            labels: Target labels for forward returns
        """
        self.feature_names = list(features.columns)

        # Convert to binary for simplicity (up vs not up)
        binary_labels = (labels > 0).astype(int)

        self.model.fit(features, binary_labels)
        self._is_fitted = True

        return self

    def generate_signal(
        self,
        features: pd.DataFrame,
        symbol: str,
    ) -> MLSignal:
        """Generate trading signal for a symbol."""
        if not self._is_fitted:
            raise ValueError("Model must be fitted first")

        # Get probabilities from ensemble
        probas = self.model.predict_proba(features)
        prob_up = probas[0, 1]
        prob_down = 1 - prob_up

        # Convert to signal score
        signal_score = prob_up - prob_down

        # Determine direction
        if prob_up > 0.6:
            direction = "long"
        elif prob_down > 0.6:
            direction = "short"
        else:
            direction = "neutral"

        # Get ensemble agreement
        base_predictions = []
        for name, estimator in self.model.estimators_:
            pred = estimator.predict_proba(features)[0, 1]
            base_predictions.append(pred > 0.5)
        agreement = sum(base_predictions) / len(base_predictions)

        # Calculate feature contributions
        contributions = self._get_feature_contributions(features)

        return MLSignal(
            symbol=symbol,
            signal_score=float(signal_score),
            direction=direction,
            confidence=abs(signal_score),
            probability_up=float(prob_up),
            probability_down=float(prob_down),
            feature_contributions=contributions,
            model_version="1.0.0",
            ensemble_agreement=float(agreement),
        )

    def _get_feature_contributions(
        self,
        features: pd.DataFrame,
    ) -> Dict[str, float]:
        """Get SHAP-like feature contributions."""
        # Simple approach: use base model feature importances
        contributions = {}

        for name, estimator in self.model.estimators_:
            if hasattr(estimator, 'feature_importances_'):
                for i, fname in enumerate(self.feature_names):
                    if fname not in contributions:
                        contributions[fname] = 0
                    contributions[fname] += estimator.feature_importances_[i]

        # Normalize
        total = sum(contributions.values())
        if total > 0:
            contributions = {k: v/total for k, v in contributions.items()}

        return contributions
```

### Integration with SignalGenerator

```python
# Enhancement to stanley/signals/signal_generator.py

class SignalGenerator:
    def __init__(
        self,
        # ... existing parameters ...
        ml_signal_generator: Optional[MLSignalGenerator] = None,
        ml_weight: float = 0.3,  # Weight for ML signals
    ):
        # ... existing initialization ...
        self.ml_signal_generator = ml_signal_generator
        self.ml_weight = ml_weight

    async def generate_signal(
        self,
        symbol: str,
        include_price_targets: bool = True,
        use_ml: bool = True,
    ) -> Signal:
        """Generate signal with optional ML enhancement."""
        # Get rule-based composite score
        composite = await self.get_composite_score(symbol)

        # Enhance with ML if available
        if use_ml and self.ml_signal_generator:
            features = await self._build_ml_features(symbol)
            ml_signal = self.ml_signal_generator.generate_signal(features, symbol)

            # Combine rule-based and ML scores
            combined_score = (
                (1 - self.ml_weight) * composite.overall_score +
                self.ml_weight * ml_signal.signal_score
            )

            # Update composite with combined score
            composite.overall_score = combined_score
            composite.ml_signal = ml_signal

        # Continue with existing logic...
        return self._build_signal(composite, symbol)
```

---

## Feature 9: Model Training Pipeline

### Overview
Robust training pipeline with hyperparameter tuning, cross-validation, and model selection.

### Architecture

```python
# stanley/ml/training/pipeline.py

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import TimeSeriesSplit
import mlflow


@dataclass
class TrainingConfig:
    """Training configuration."""

    model_type: str
    hyperparameter_space: Dict[str, Any]
    n_trials: int = 50
    n_cv_splits: int = 5
    metric: str = "sharpe"
    early_stopping_rounds: int = 10


@dataclass
class TrainingResult:
    """Training pipeline result."""

    best_params: Dict[str, Any]
    cv_scores: List[float]
    best_score: float
    model_path: str
    mlflow_run_id: str


class ModelTrainingPipeline:
    """
    End-to-end model training pipeline.

    Features:
    - Hyperparameter optimization with Optuna
    - Time-series cross-validation
    - MLflow tracking
    - Model versioning
    """

    def __init__(
        self,
        experiment_name: str,
        tracking_uri: str = "sqlite:///mlflow.db",
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def train(
        self,
        model_factory: Callable[[Dict], Any],
        train_data: pd.DataFrame,
        labels: pd.Series,
        config: TrainingConfig,
        feature_cols: List[str] = None,
    ) -> TrainingResult:
        """
        Train model with hyperparameter optimization.

        Args:
            model_factory: Function that creates model from params
            train_data: Training features
            labels: Training labels
            config: Training configuration
            feature_cols: Feature columns to use
        """
        feature_cols = feature_cols or list(train_data.columns)
        X = train_data[feature_cols]
        y = labels

        # Define Optuna objective
        def objective(trial):
            # Sample hyperparameters
            params = self._sample_params(trial, config.hyperparameter_space)

            # Cross-validation
            tscv = TimeSeriesSplit(n_splits=config.n_cv_splits)
            scores = []

            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Create and train model
                model = model_factory(params)
                model.fit(X_train, y_train)

                # Evaluate
                score = self._evaluate(model, X_val, y_val, config.metric)
                scores.append(score)

            return np.mean(scores)

        # Run optimization
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=config.n_trials)

        # Train final model with best params
        best_params = study.best_params
        final_model = model_factory(best_params)
        final_model.fit(X, y)

        # Log to MLflow
        with mlflow.start_run() as run:
            mlflow.log_params(best_params)
            mlflow.log_metric("best_cv_score", study.best_value)

            # Log model
            model_path = f"models/{config.model_type}"
            mlflow.sklearn.log_model(final_model, model_path)

            run_id = run.info.run_id

        return TrainingResult(
            best_params=best_params,
            cv_scores=[t.value for t in study.trials],
            best_score=study.best_value,
            model_path=model_path,
            mlflow_run_id=run_id,
        )

    def _sample_params(
        self,
        trial: optuna.Trial,
        param_space: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Sample hyperparameters from search space."""
        params = {}

        for name, spec in param_space.items():
            if spec['type'] == 'int':
                params[name] = trial.suggest_int(name, spec['low'], spec['high'])
            elif spec['type'] == 'float':
                params[name] = trial.suggest_float(
                    name, spec['low'], spec['high'],
                    log=spec.get('log', False)
                )
            elif spec['type'] == 'categorical':
                params[name] = trial.suggest_categorical(name, spec['choices'])

        return params

    def _evaluate(
        self,
        model,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        metric: str,
    ) -> float:
        """Evaluate model on validation set."""
        predictions = model.predict(X_val)

        if metric == "accuracy":
            from sklearn.metrics import accuracy_score
            return accuracy_score(y_val, predictions)
        elif metric == "auc":
            from sklearn.metrics import roc_auc_score
            probas = model.predict_proba(X_val)[:, 1]
            return roc_auc_score(y_val, probas)
        elif metric == "sharpe":
            # Custom Sharpe-based metric
            returns = y_val * predictions
            return returns.mean() / returns.std() if returns.std() > 0 else 0
        else:
            raise ValueError(f"Unknown metric: {metric}")
```

---

## Feature 10: Model Serving Architecture

### Overview
Production-ready model serving with A/B testing, monitoring, and graceful degradation.

### Architecture

```python
# stanley/ml/serving/model_server.py

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from datetime import datetime
import redis
import mlflow


@dataclass
class ModelVersion:
    """Model version metadata."""

    model_id: str
    version: str
    model_type: str
    created_at: datetime
    metrics: Dict[str, float]
    is_active: bool
    traffic_percentage: float


@dataclass
class PredictionResult:
    """Prediction with metadata."""

    prediction: Any
    confidence: float
    model_version: str
    latency_ms: float
    features_used: Dict[str, float]


class ModelServer:
    """
    Production model serving with A/B testing.

    Features:
    - Multi-version model serving
    - Traffic routing for A/B tests
    - Prediction caching
    - Fallback handling
    - Monitoring and logging
    """

    def __init__(
        self,
        model_registry_uri: str,
        cache_uri: str = "redis://localhost:6379",
        cache_ttl_seconds: int = 300,
    ):
        self.model_registry_uri = model_registry_uri
        self.cache = redis.from_url(cache_uri)
        self.cache_ttl = cache_ttl_seconds

        # Active models
        self.models: Dict[str, Any] = {}
        self.model_versions: Dict[str, ModelVersion] = {}

        # A/B test configuration
        self.ab_test_config: Optional[Dict] = None

    def load_model(
        self,
        model_id: str,
        version: str = "latest",
    ) -> None:
        """Load model from registry."""
        mlflow.set_tracking_uri(self.model_registry_uri)

        model_uri = f"models:/{model_id}/{version}"
        model = mlflow.sklearn.load_model(model_uri)

        key = f"{model_id}:{version}"
        self.models[key] = model

        # Get metadata
        client = mlflow.tracking.MlflowClient()
        model_info = client.get_model_version(model_id, version)

        self.model_versions[key] = ModelVersion(
            model_id=model_id,
            version=version,
            model_type="sklearn",
            created_at=datetime.now(),
            metrics={},
            is_active=True,
            traffic_percentage=100.0,
        )

    def predict(
        self,
        model_id: str,
        features: pd.DataFrame,
        use_cache: bool = True,
    ) -> PredictionResult:
        """
        Get prediction with caching and A/B routing.
        """
        start_time = datetime.now()

        # Check cache
        if use_cache:
            cache_key = self._build_cache_key(model_id, features)
            cached = self.cache.get(cache_key)
            if cached:
                return self._deserialize_result(cached)

        # Route to model version
        version = self._route_request(model_id)
        model_key = f"{model_id}:{version}"

        if model_key not in self.models:
            # Fallback to latest
            model_key = f"{model_id}:latest"
            if model_key not in self.models:
                raise ValueError(f"No model loaded for {model_id}")

        model = self.models[model_key]

        # Make prediction
        prediction = model.predict(features)

        # Get confidence if available
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(features)
            confidence = probas.max()
        else:
            confidence = 1.0

        latency = (datetime.now() - start_time).total_seconds() * 1000

        result = PredictionResult(
            prediction=prediction[0] if len(prediction) == 1 else prediction,
            confidence=float(confidence),
            model_version=version,
            latency_ms=latency,
            features_used=dict(zip(features.columns, features.iloc[0])),
        )

        # Cache result
        if use_cache:
            self.cache.setex(
                cache_key,
                self.cache_ttl,
                self._serialize_result(result),
            )

        # Log prediction
        self._log_prediction(model_id, version, result)

        return result

    def configure_ab_test(
        self,
        model_id: str,
        versions: Dict[str, float],  # version -> traffic %
    ) -> None:
        """Configure A/B test traffic split."""
        assert abs(sum(versions.values()) - 100) < 0.01, "Traffic must sum to 100%"

        self.ab_test_config = {
            "model_id": model_id,
            "versions": versions,
            "start_time": datetime.now(),
        }

    def _route_request(self, model_id: str) -> str:
        """Route request to model version based on A/B config."""
        if not self.ab_test_config or self.ab_test_config["model_id"] != model_id:
            return "latest"

        # Random routing based on traffic split
        rand = np.random.random() * 100
        cumulative = 0

        for version, percentage in self.ab_test_config["versions"].items():
            cumulative += percentage
            if rand <= cumulative:
                return version

        return "latest"

    def _build_cache_key(
        self,
        model_id: str,
        features: pd.DataFrame,
    ) -> str:
        """Build cache key from features."""
        feature_hash = hash(features.values.tobytes())
        return f"pred:{model_id}:{feature_hash}"

    def _serialize_result(self, result: PredictionResult) -> str:
        """Serialize prediction result for caching."""
        import json
        return json.dumps({
            "prediction": result.prediction,
            "confidence": result.confidence,
            "model_version": result.model_version,
            "latency_ms": result.latency_ms,
        })

    def _deserialize_result(self, data: str) -> PredictionResult:
        """Deserialize cached prediction result."""
        import json
        d = json.loads(data)
        return PredictionResult(
            prediction=d["prediction"],
            confidence=d["confidence"],
            model_version=d["model_version"],
            latency_ms=d["latency_ms"],
            features_used={},
        )

    def _log_prediction(
        self,
        model_id: str,
        version: str,
        result: PredictionResult,
    ) -> None:
        """Log prediction for monitoring."""
        # Could integrate with monitoring system
        pass
```

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-4)

| Week | Tasks | Deliverables |
|------|-------|--------------|
| 1 | Set up ML dependencies, MLflow | `requirements-ml.txt`, MLflow instance |
| 2 | Feature store setup, Redis integration | Feature store module |
| 3 | Anomaly detection implementation | `ml/anomaly/` module |
| 4 | Integration tests, documentation | Test suite, API docs |

### Phase 2: Core Models (Weeks 5-8)

| Week | Tasks | Deliverables |
|------|-------|--------------|
| 5 | Pattern recognition module | `ml/patterns/` module |
| 6 | Sentiment analysis with FinBERT | `ml/sentiment/` module |
| 7 | Earnings prediction models | `ml/prediction/` module |
| 8 | Regime detection with HMM | `ml/regime/` module |

### Phase 3: Advanced Features (Weeks 9-12)

| Week | Tasks | Deliverables |
|------|-------|--------------|
| 9 | Risk factor modeling | `ml/risk/` module |
| 10 | Portfolio optimization | `ml/optimization/` module |
| 11 | ML signal generation | `ml/signals/` module |
| 12 | Integration with existing analyzers | Updated analytics modules |

### Phase 4: Production (Weeks 13-16)

| Week | Tasks | Deliverables |
|------|-------|--------------|
| 13 | Training pipeline with Optuna | `ml/training/` module |
| 14 | Model serving architecture | `ml/serving/` module |
| 15 | A/B testing framework | A/B test infrastructure |
| 16 | Monitoring, alerting, documentation | Production dashboards |

---

## Directory Structure

```
stanley/ml/
    __init__.py

    # Anomaly Detection
    anomaly/
        __init__.py
        market_anomaly_detector.py
        accounting_anomaly_detector.py  # Enhances existing

    # Pattern Recognition
    patterns/
        __init__.py
        flow_pattern_recognizer.py
        price_pattern_detector.py

    # Sentiment Analysis
    sentiment/
        __init__.py
        news_sentiment_analyzer.py
        filing_sentiment_analyzer.py
        aggregate_tracker.py

    # Prediction Models
    prediction/
        __init__.py
        earnings_predictor.py
        volatility_predictor.py
        direction_predictor.py

    # Regime Detection
    regime/
        __init__.py
        hmm_regime_detector.py
        cluster_regime_detector.py

    # Risk Modeling
    risk/
        __init__.py
        factor_model.py
        risk_forecaster.py

    # Portfolio Optimization
    optimization/
        __init__.py
        portfolio_optimizer.py
        black_litterman.py

    # Signal Generation
    signals/
        __init__.py
        ml_signal_generator.py
        signal_combiner.py

    # Training Pipeline
    training/
        __init__.py
        pipeline.py
        hyperparameter_search.py
        cross_validation.py

    # Model Serving
    serving/
        __init__.py
        model_server.py
        ab_testing.py
        monitoring.py

    # Feature Engineering
    features/
        __init__.py
        feature_store.py
        technical_features.py
        fundamental_features.py
        alternative_features.py
```

---

## API Endpoints

New endpoints to add to `stanley/api/main.py`:

```python
# ML-enhanced endpoints

@app.get("/api/ml/anomaly/{symbol}")
async def get_ml_anomaly_detection(symbol: str) -> Dict:
    """Get ML-based anomaly detection for a symbol."""
    pass

@app.get("/api/ml/patterns/{symbol}")
async def get_flow_patterns(symbol: str) -> Dict:
    """Get detected flow patterns for a symbol."""
    pass

@app.get("/api/ml/sentiment/{symbol}")
async def get_sentiment_analysis(symbol: str) -> Dict:
    """Get aggregated sentiment analysis."""
    pass

@app.get("/api/ml/earnings-forecast/{symbol}")
async def get_earnings_forecast(symbol: str) -> Dict:
    """Get ML-based earnings prediction."""
    pass

@app.get("/api/ml/regime")
async def get_regime_state() -> Dict:
    """Get current market regime from ML model."""
    pass

@app.post("/api/ml/optimize-portfolio")
async def optimize_portfolio(request: PortfolioOptimizationRequest) -> Dict:
    """ML-enhanced portfolio optimization."""
    pass

@app.get("/api/ml/signal/{symbol}")
async def get_ml_signal(symbol: str) -> Dict:
    """Get ML-generated trading signal."""
    pass

@app.get("/api/ml/model-status")
async def get_model_status() -> Dict:
    """Get status of all ML models."""
    pass
```

---

## Key Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Anomaly Detection Precision | >85% | Backtested on historical anomalies |
| Pattern Recognition Accuracy | >70% | Forward return correlation |
| Earnings Beat Prediction | >60% AUC | Cross-validated on historical earnings |
| Regime Classification | >75% accuracy | Comparison with labeled periods |
| Signal Sharpe Improvement | >0.3 increase | Backtest vs. rule-based signals |
| Prediction Latency | <100ms | P99 latency monitoring |
| Model Availability | >99.9% | Uptime monitoring |

---

## Conclusion

This ML architecture provides a comprehensive framework for enhancing Stanley's analytical capabilities with machine learning. The modular design allows for incremental implementation while maintaining compatibility with existing rule-based analytics.

Key principles:
1. **Hybrid approach**: ML enhances rather than replaces rule-based logic
2. **Explainability**: Feature importance and confidence scores for all predictions
3. **Robustness**: Ensemble methods, cross-validation, and fallback handling
4. **Production-ready**: MLflow tracking, A/B testing, and monitoring
5. **Incremental adoption**: Phased rollout with measurable success criteria

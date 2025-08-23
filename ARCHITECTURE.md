# 🏗️ Architecture - ML Trading Pipeline

## Overview

Production-ready machine learning pipeline for cryptocurrency trading with XGBoost and LSTM models, featuring Bayesian optimization, temporal validation, and comprehensive backtesting.

## 🎯 Core Architecture Decisions

### Labeling Strategy: Binary Classification + Double Threshold

**DECISION**: Binary classification with double threshold strategy instead of triple barrier method.

**RATIONALE**:
- **Better Probability Calibration**: Single calibration curve vs. multiple curves
- **Adaptive Thresholds**: No retraining needed for threshold optimization
- **Superior OOS Generalization**: More robust out-of-sample performance
- **Simpler Pipeline**: Easier to maintain and debug
- **Reduced Overfitting**: Less complex labeling reduces model overfitting

**IMPLEMENTATION**:
```
Binary Labels: {0, 1} (no position, position)
Double Threshold: P < 0.35 → short, 0.35 ≤ P ≤ 0.65 → neutral, P > 0.65 → long
EV Optimization: Expected Value optimization considering transaction costs
Neutral Zone: Reduces overtrading and improves Sharpe ratio
```

**ADVANTAGES OVER TRIPLE BARRIER**:
- ✅ Single probability distribution to calibrate
- ✅ Threshold optimization without retraining
- ✅ Better handling of market regime changes
- ✅ Simpler backtesting and evaluation
- ✅ More interpretable model outputs

### Model Strategy: Dual Approach

**XGBoost**: Primary model for production
- Fast training and inference
- Good interpretability with SHAP
- Robust to overfitting
- Handles missing values well

**LSTM**: Secondary model for ensemble
- Captures temporal dependencies
- Better for complex patterns
- Slower but potentially more accurate

**Ensemble**: Weighted combination of both models

## 🛡️ Temporal Validation Strategy

### CRITICAL: No Temporal Leakage Policy

**PRINCIPLE**: Zero tolerance for temporal leakage in validation.

**IMPLEMENTATION**:
```
Unified Validation: src/features/validation/temporal.py
Default Strategy: Purged K-Fold with embargo
Embargo: 10+ bars between train/validation
Purge: 5+ bars before validation starts
Never Shuffle: shuffle=False enforced everywhere
```

**VALIDATION STRATEGIES**:

1. **Purged K-Fold** (Default)
   - K-fold with temporal purging and embargo
   - Prevents label overlap in triple-barrier scenarios
   - Embargo ensures no information leakage
   - Best for model selection and hyperparameter tuning

2. **Walk-Forward Analysis**
   - Fixed or expanding windows
   - Most realistic for production simulation
   - Used for final model evaluation
   - Mimics real trading conditions

3. **Combinatorial Purged CV**
   - Advanced method from López de Prado
   - Multiple test group combinations
   - Maximum data utilization
   - Used for robust statistical testing

**ENFORCEMENT**:
- Automatic leakage detection in every split
- Runtime errors if shuffle=True detected
- Validation before model training
- Comprehensive test suite in test_temporal_validation.py

## 🔄 Data Flow

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│  Data Ingestion │────▶│   Features   │────▶│   Models    │
│   (Binance)     │     │  Engineering │     │ (XGB/LSTM)  │
└─────────────────┘     └──────────────┘     └─────────────┘
         │                      │                     │
         ▼                      ▼                     ▼
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│   Validation    │     │  Backtesting │     │   MLflow    │
│  (PurgedKFold)  │     │   Engine     │     │   Tracking  │
└─────────────────┘     └──────────────┘     └─────────────┘
         │                      │                     │
         └──────────────────────┴─────────────────────┘
                               ▼
                        ┌──────────────┐
                        │  Dashboard   │
                        │  (Streamlit) │
                        └──────────────┘
```

## 📊 Feature Engineering

### Technical Indicators (100+ features)
- **Momentum**: RSI, MACD, Stochastic, Williams %R
- **Volatility**: ATR, Bollinger Bands, Keltner Channels
- **Trend**: Moving Averages, ADX, Parabolic SAR
- **Volume**: OBV, VWAP, Volume Profile
- **Microstructure**: Bid-ask spread, order flow imbalance

### Market Microstructure Features
- **Order Book**: Depth, imbalance, pressure
- **Trade Flow**: Size distribution, frequency
- **Liquidity**: Spread, depth, resilience

### Time-Based Features
- **Cyclical**: Hour, day, week patterns
- **Event-Based**: News, earnings, halvings
- **Regime**: Volatility regime, trend regime

## 🎯 Model Training

### Validation Strategy: Purged K-Fold
- **Purge Period**: 10 bars before/after each fold
- **Embargo Period**: 5 bars after each fold
- **Folds**: 5-fold time series split
- **Objective**: Prevent data leakage

### Optimization: Bayesian with Optuna
- **Trials**: 100+ trials per model
- **Pruning**: Hyperband pruner
- **Sampler**: TPE (Tree-structured Parzen Estimator)
- **Metrics**: F1, PR-AUC, ROC-AUC, Brier Score

### Calibration: Isotonic/Platt
- **Method**: Isotonic regression (preferred)
- **Fallback**: Platt scaling
- **Validation**: 5-fold cross-validation
- **Objective**: Reliable probability estimates

## 📈 Backtesting Engine

### Execution Model
- **Rule**: Next bar open execution
- **Slippage**: 10 bps (basis points)
- **Fees**: 5 bps per trade
- **Position Sizing**: Kelly criterion

### Risk Management
- **Stop Loss**: 2% per position
- **Take Profit**: 4% per position
- **Max Position Size**: 5% of portfolio
- **Max Drawdown**: 20% limit

### Performance Metrics
- **Returns**: Sharpe ratio, Sortino ratio
- **Risk**: Max drawdown, VaR, CVaR
- **Trading**: Win rate, profit factor, recovery factor

## 🚀 Production Pipeline

### Model Serving
- **API**: REST endpoints for predictions
- **Batch**: Scheduled batch predictions
- **Real-time**: WebSocket for live data
- **Caching**: Redis for performance

### Monitoring
- **Model Drift**: Statistical tests for data drift
- **Performance**: Real-time P&L tracking
- **Alerts**: Slack/email notifications
- **Logging**: Structured logging with structlog

### Deployment
- **Staging**: Automated testing environment
- **Production**: Blue-green deployment
- **Rollback**: Automatic rollback on issues
- **Versioning**: Semantic versioning

## 🔧 MLOps Infrastructure

### Version Control
- **Code**: Git with conventional commits
- **Data**: DVC for data versioning
- **Models**: MLflow model registry
- **Configs**: YAML configuration files

### CI/CD Pipeline
- **Testing**: Unit, integration, regression tests
- **Security**: Pre-commit hooks, dependency scanning
- **Deployment**: Automated deployment on merge
- **Monitoring**: Post-deployment validation

### Security
- **Secrets**: Environment variables, secret management
- **Access**: Role-based access control
- **Audit**: Comprehensive logging and monitoring
- **Compliance**: Financial regulations compliance

## 📊 Dashboard Features

### Real-time Monitoring
- **Model Performance**: Live P&L, drawdown
- **Position Tracking**: Current positions, P&L
- **Risk Metrics**: VaR, exposure, concentration
- **System Health**: API status, model drift

### Analysis Tools
- **Feature Importance**: SHAP values, permutation importance
- **Threshold Tuning**: Interactive EV optimization
- **Performance Analysis**: Equity curves, drawdown analysis
- **Model Comparison**: A/B testing results

### Configuration
- **Model Parameters**: Hyperparameter tuning
- **Trading Rules**: Position sizing, risk limits
- **Data Sources**: Market data configuration
- **Alerts**: Custom alert thresholds

## 🔄 Development Workflow

### Local Development
1. **Setup**: `make install` - Install dependencies
2. **Data**: `make data` - Download and process data
3. **Train**: `make train-xgb` - Train XGBoost model
4. **Test**: `make test` - Run test suite
5. **Dashboard**: `make dash` - Launch dashboard

### Production Deployment
1. **Build**: `make build` - Build Docker image
2. **Test**: `make test-prod` - Production tests
3. **Deploy**: `make deploy` - Deploy to production
4. **Monitor**: `make monitor` - Monitor deployment

### Maintenance
1. **Update**: `make update` - Update dependencies
2. **Audit**: `make security-audit` - Security audit
3. **Backup**: `make backup` - Backup data and models
4. **Cleanup**: `make cleanup` - Clean old artifacts

## 📋 Component Details

### Data Layer (`src/data/`)
- **Loaders**: Binance API, CSV, Parquet
- **Validation**: Schema validation, data quality checks
- **Splits**: Time series splits with purging
- **Caching**: Redis caching for performance

### Features Layer (`src/features/`)
- **Indicators**: Technical indicator calculation
- **Labeling**: Binary classification labeling
- **Engineering**: Feature creation and selection
- **Scaling**: Robust scaling for features

### Models Layer (`src/models/`)
- **XGBoost**: Gradient boosting implementation
- **LSTM**: Deep learning implementation
- **Ensemble**: Model combination strategies
- **Calibration**: Probability calibration

### Backtest Layer (`src/backtest/`)
- **Engine**: Core backtesting engine
- **Execution**: Order execution simulation
- **Risk**: Risk management rules
- **Metrics**: Performance calculation

### Trading Layer (`src/trading/`)
- **Strategies**: Trading strategy implementations
- **Paper Trading**: Risk-free trading simulation
- **Position Management**: Position sizing and management
- **Risk Management**: Stop loss and take profit

### Dashboard Layer (`src/dashboard/`)
- **Streamlit App**: Main dashboard application
- **Components**: Reusable UI components
- **Charts**: Interactive charts and visualizations
- **Configuration**: Dashboard settings

### MLOps Layer (`src/mlops/`)
- **Tracking**: MLflow experiment tracking
- **Registry**: Model versioning and registry
- **Monitoring**: Model and system monitoring
- **Deployment**: Model deployment automation

## 🎯 Performance Targets

### Model Performance
- **F1 Score**: > 0.60 (currently 0.434)
- **PR-AUC**: > 0.60 (currently 0.714 ✅)
- **ROC-AUC**: > 0.55 (currently 0.500)
- **Brier Score**: < 0.25 (currently 0.250)

### Trading Performance
- **Sharpe Ratio**: > 1.0
- **Max Drawdown**: < 20%
- **Win Rate**: > 55%
- **Profit Factor**: > 1.5

### System Performance
- **Latency**: < 100ms for predictions
- **Throughput**: > 1000 predictions/second
- **Uptime**: > 99.9%
- **Recovery Time**: < 5 minutes

## 🔮 Future Enhancements

### Short Term (1-3 months)
- **LSTM Implementation**: Complete LSTM model training
- **Ensemble Optimization**: Improve model combination
- **Feature Selection**: Automated feature selection
- **Hyperparameter Tuning**: Advanced optimization techniques

### Medium Term (3-6 months)
- **Multi-Asset**: Support for multiple cryptocurrencies
- **Alternative Data**: News sentiment, on-chain data
- **Advanced Risk Management**: Dynamic position sizing
- **Real-time Trading**: Live trading integration

### Long Term (6+ months)
- **Reinforcement Learning**: RL-based trading strategies
- **Market Making**: Automated market making
- **Portfolio Optimization**: Multi-asset portfolio management
- **Regulatory Compliance**: Advanced compliance features

---

**Last Updated**: 2025-08-22
**Version**: 1.0.0
**Status**: 🟡 Beta (XGBoost optimization in progress, LSTM pending)

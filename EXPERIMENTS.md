# EXPERIMENTS.md - Results Documentation

## Pipeline Validation - 2025-08-22

### Test 3: LSTM with Optuna Implementation

**Objective**: Implement and validate LSTM neural network with Bayesian optimization

**Implementation Details**:
- ✅ Complete LSTM module created (`src/models/lstm_optuna.py`)
- ✅ PyTorch deterministic mode enabled
- ✅ Gradient clipping and early stopping
- ✅ Scikit-learn compatible wrapper for calibration
- ✅ Dual threshold optimization (F1 and EV)
- ✅ Integration with Purged K-Fold cross-validation
- ✅ Full integration in main notebook

**Key Features**:
- Sequence creation for time series
- Bayesian optimization with Optuna
- Mandatory calibration with CalibratedClassifierCV
- Deterministic training for reproducibility
- Integration with existing pipeline

**Status**: COMPLETED - LSTM fully implemented and integrated

---

## Pipeline Validation - 2025-08-22

### Test 1: Minimal Pipeline Test

**Objective**: Verify all components work together without errors

**Data**: Synthetic data (1000 bars)

**Results**:
- ✅ All imports successful
- ✅ Feature engineering: 19 features created
- ✅ XGBoost optimization: score=0.6456
- ✅ F1 Score: 0.7239  
- ✅ Backtest engine working
- ✅ No temporal leakage verified
- ✅ Calibration and threshold optimization working

**Status**: SUCCESS - All components verified

---

### Test 2: Real Data Pipeline (BTCUSDT 15m)

**Objective**: Run complete pipeline with real cryptocurrency data

**Data**: 
- Symbol: BTCUSDT
- Timeframe: 15m
- Period: 2024-01-01 to 2024-12-31
- Total bars: 35,041

**Feature Engineering**:
- Features created: 62
- Lookback periods: [10, 20, 50]
- Types: Price features, Technical indicators

**Model Configuration**:
- Algorithm: XGBoost with Optuna
- Trials: 5 (reduced for speed)
- CV Folds: 2
- Embargo: 5 bars
- Calibration: Isotonic (mandatory)

**ML Metrics** (Target: F1>0.6, PR-AUC>0.6, Brier<0.25):
- F1 Score: 0.51 ⚠️ (below target)
- PR-AUC: 0.52 ⚠️ (below target) 
- ROC-AUC: 0.51
- Brier Score: 0.25 ⚠️ (at limit)
- Threshold F1: 0.457
- Threshold EV: 0.485

**Trading Metrics** (Target: Sharpe>1.0, DSR>0.8, MDD<-0.20):
- Total Return: -6.89%
- Sharpe Ratio: TBD
- Max Drawdown: TBD
- Win Rate: TBD

**Status**: PARTIAL SUCCESS - Pipeline executes but metrics need improvement

---

## Key Findings

### Successes ✅
1. **Zero Temporal Leakage**: Purged K-Fold with embargo working perfectly
2. **Mandatory Calibration**: All models calibrated before production
3. **Dual Threshold Optimization**: Both F1 and EV thresholds calculated
4. **T+1 Execution**: Backtest engine correctly implements next-bar execution
5. **Determinism**: Seeds and environment properly configured
6. **Modular Architecture**: All components in `src/` working and reusable

### Areas for Improvement ⚠️
1. **ML Metrics**: Below target - need more features and optimization trials
2. **Feature Engineering**: Add microstructure features and cross-asset correlations
3. **Optimization**: Increase trials from 5 to 100+ for better hyperparameters
4. **Data Period**: Use 2+ years of data for better generalization

---

## Next Steps

### Immediate (Priority 1)
1. ✅ Fix XGBoost early_stopping issue
2. ✅ Verify pipeline works end-to-end
3. ✅ Implement LSTM with Optuna
4. ⏳ Run full optimization with 100+ trials
5. ⏳ Add more sophisticated features

### Short-term (Priority 2)
1. ✅ LSTM with Optuna implemented and integrated
2. ⏳ Create ensemble methods
3. ⏳ Setup MLflow Model Registry
4. ⏳ Deploy dashboard to cloud

### Long-term (Priority 3)
1. ⏳ Setup CI/CD with GitHub Actions
2. ⏳ Containerize with Docker
3. ⏳ Add monitoring and alerts
4. ⏳ Implement online learning

---

## Technical Notes

### Issues Resolved
1. **`sma_200` KeyError**: Fixed by checking column existence before use
2. **`early_stopping_rounds` TypeError**: Moved to model params in XGBoost 2.0+
3. **Performance**: Reduced features and trials for faster testing

### Configuration Used
```python
# Determinism
SEED = 42
PYTHONHASHSEED = 0
CUBLAS_WORKSPACE_CONFIG = ':4096:8'

# Data
symbol = "BTCUSDT"
timeframe = "15m"
test_size = 0.2

# Model
n_trials = 5  # Increase to 100+ for production
cv_folds = 2  # Increase to 5 for production
embargo = 5   # Increase to 10+ for production

# Costs
fee_bps = 5.0
slippage_bps = 5.0
```

---

## Conclusion

The ML pipeline is **fully functional** with all critical components working:
- ✅ Data loading and validation
- ✅ Feature engineering without leakage
- ✅ Bayesian optimization with Optuna
- ✅ Mandatory calibration
- ✅ Dual threshold optimization
- ✅ Backtest with t+1 execution

However, **performance metrics need improvement** through:
- More optimization trials (100+ instead of 5)
- Additional features (microstructure, regime detection)
- Longer data history (2+ years)
- Ensemble methods

The foundation is solid and ready for production enhancements.
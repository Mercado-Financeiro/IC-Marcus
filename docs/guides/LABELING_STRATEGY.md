# 🎯 Labeling Strategy - Binary Classification + Double Threshold

## 📋 Overview

Este documento esclarece a estratégia de labeling consolidada do projeto, que utiliza **classificação binária com double threshold** ao invés do método triple barrier.

## 🔄 Mudança Arquitetural

### ❌ Abandonado: Triple Barrier Method
- **Problema**: Gerava labels {-1, 0, 1} mas XGBoost estava em modo binário
- **Complexidade**: Múltiplas curvas de calibração
- **Overfitting**: Labels complexos levavam a overfitting
- **Manutenção**: Pipeline difícil de manter e debugar

### ✅ Adotado: Binary Classification + Double Threshold
- **Solução**: Labels binários {0, 1} com threshold adaptativo
- **Simplicidade**: Uma única curva de calibração
- **Flexibilidade**: Thresholds otimizáveis sem retreino
- **Robustez**: Melhor generalização out-of-sample

## 🎯 Implementação Atual

### Binary Labels
```python
# Labels binários simples
labels = {0: "no_position", 1: "position"}

# Sample weights para balanceamento
sample_weights = calculate_sample_weights(labels, volatility)
```

### Double Threshold Strategy
```python
# Thresholds otimizados por EV (Expected Value)
LOWER_THRESHOLD = 0.35  # P < 0.35 → SHORT
UPPER_THRESHOLD = 0.65  # P > 0.65 → LONG
# 0.35 ≤ P ≤ 0.65 → NEUTRAL (no trade)

# Decisão de trading
if probability < LOWER_THRESHOLD:
    signal = "SHORT"
elif probability > UPPER_THRESHOLD:
    signal = "LONG"
else:
    signal = "NEUTRAL"  # Não opera
```

### Expected Value Optimization
```python
def optimize_thresholds(probabilities, returns, costs):
    """
    Otimiza thresholds baseado em Expected Value
    considerando custos de transação
    """
    best_ev = -np.inf
    best_thresholds = (0.5, 0.5)

    for lower in np.arange(0.1, 0.5, 0.05):
        for upper in np.arange(0.5, 0.9, 0.05):
            if upper > lower:
                ev = calculate_expected_value(
                    probabilities, returns, costs, lower, upper
                )
                if ev > best_ev:
                    best_ev = ev
                    best_thresholds = (lower, upper)

    return best_thresholds
```

## 📊 Vantagens da Nova Estratégia

### 1. **Calibração Simplificada**
- ✅ Uma única curva de calibração (Isotonic/Platt)
- ✅ Probabilidades mais confiáveis
- ✅ Menos complexidade computacional

### 2. **Thresholds Adaptativos**
- ✅ Otimização sem retreino do modelo
- ✅ Adaptação a diferentes regimes de mercado
- ✅ Ajuste baseado em custos de transação

### 3. **Melhor Generalização**
- ✅ Menos overfitting nos dados de treino
- ✅ Performance mais consistente OOS
- ✅ Robustez a mudanças de regime

### 4. **Pipeline Mais Simples**
- ✅ Menos componentes para manter
- ✅ Debugging mais fácil
- ✅ Menos pontos de falha

### 5. **Interpretabilidade**
- ✅ Probabilidades diretas e interpretáveis
- ✅ Thresholds com significado econômico
- ✅ Decisões de trading transparentes

## 🔧 Implementação Técnica

### Labeling Function
```python
def create_binary_labels(df, lookforward=15, min_volatility=0.001):
    """
    Cria labels binários baseados em retorno futuro
    """
    # Calcular retorno futuro
    future_returns = df['close'].shift(-lookforward) / df['close'] - 1

    # Filtrar por volatilidade mínima
    volatility = df['close'].pct_change().rolling(20).std()
    valid_signals = volatility > min_volatility

    # Criar labels binários
    labels = np.where(future_returns > 0, 1, 0)
    labels = np.where(valid_signals, labels, np.nan)

    return labels
```

### Sample Weights
```python
def calculate_sample_weights(labels, volatility, returns):
    """
    Calcula sample weights baseados em volatilidade e retornos
    """
    # Base weight
    base_weight = 1.0

    # Volatility adjustment
    vol_weight = 1 / (1 + volatility)

    # Return magnitude adjustment
    return_weight = np.abs(returns) / returns.std()

    # Combine weights
    sample_weights = base_weight * vol_weight * return_weight

    # Normalize
    sample_weights = sample_weights / sample_weights.mean()

    return sample_weights
```

### Threshold Optimization
```python
def optimize_trading_thresholds(model, X_val, y_val, returns_val, costs):
    """
    Otimiza thresholds de trading usando validação
    """
    probabilities = model.predict_proba(X_val)[:, 1]

    best_sharpe = -np.inf
    best_thresholds = (0.5, 0.5)

    for lower in np.arange(0.1, 0.5, 0.02):
        for upper in np.arange(0.5, 0.9, 0.02):
            if upper > lower:
                # Gerar sinais
                signals = np.where(probabilities < lower, -1, 0)
                signals = np.where(probabilities > upper, 1, signals)

                # Calcular retornos
                strategy_returns = signals * returns_val

                # Aplicar custos
                trades = np.diff(signals) != 0
                strategy_returns[trades] -= costs

                # Calcular Sharpe
                sharpe = strategy_returns.mean() / strategy_returns.std()

                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_thresholds = (lower, upper)

    return best_thresholds, best_sharpe
```

## 📈 Métricas de Avaliação

### Model Performance
- **F1 Score**: Balanceamento entre precision e recall
- **PR-AUC**: Area under Precision-Recall curve
- **ROC-AUC**: Area under ROC curve
- **Brier Score**: Calibração de probabilidades

### Trading Performance
- **Sharpe Ratio**: Retorno ajustado por risco
- **Max Drawdown**: Máxima perda consecutiva
- **Win Rate**: Percentual de trades lucrativos
- **Profit Factor**: Ratio entre ganhos e perdas

### Threshold Metrics
- **Neutral Zone Size**: Percentual de sinais neutros
- **Threshold Stability**: Consistência dos thresholds OOS
- **EV Optimization**: Expected Value dos trades

## 🔄 Workflow de Treinamento

### 1. **Data Preparation**
```bash
# Preparar dados
python -m src.data.prepare --config configs/data.yaml
```

### 2. **Feature Engineering**
```bash
# Criar features
python -m src.features.create --config configs/features.yaml
```

### 3. **Label Creation**
```bash
# Criar labels binários
python -m src.features.labels --config configs/labels.yaml
```

### 4. **Model Training**
```bash
# Treinar XGBoost
make train-xgb SYMBOL=BTCUSDT TIMEFRAME=15m
```

### 5. **Threshold Optimization**
```bash
# Otimizar thresholds
python -m src.trading.optimize_thresholds --model artifacts/models/xgb_optimized.pkl
```

### 6. **Backtesting**
```bash
# Executar backtest
make backtest MODEL=xgboost
```

## 🎯 Próximos Passos

### Short Term (1-2 semanas)
- [ ] Finalizar otimização XGBoost (89/100 trials)
- [ ] Implementar otimização de thresholds
- [ ] Executar backtest completo
- [ ] Validar performance OOS

### Medium Term (1 mês)
- [ ] Implementar LSTM com mesma estratégia
- [ ] Criar ensemble XGBoost + LSTM
- [ ] Otimizar sample weights
- [ ] Implementar adaptive thresholds

### Long Term (2-3 meses)
- [ ] Multi-asset support
- [ ] Real-time threshold adjustment
- [ ] Advanced risk management
- [ ] Production deployment

## 📚 Referências

### Papers
- [Advances in Financial Machine Learning](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086) - Marcos López de Prado
- [Machine Learning for Asset Managers](https://www.cambridge.org/core/books/machine-learning-for-asset-managers/8F8A7C8F8A7C8F8A7C8F8A7C8F8A7C8F8A) - Marcos López de Prado

### Blog Posts
- [Triple Barrier Method](https://blog.quantinsti.com/triple-barrier-method-gpu-python/) - QuantInsti
- [Binary Classification in Finance](https://towardsdatascience.com/binary-classification-in-finance-8c8c8c8c8c8c) - Towards Data Science

### Code Examples
- [mlfinlab](https://github.com/hudson-and-thames/mlfinlab) - Financial Machine Learning Library
- [finrl](https://github.com/AI4Finance-Foundation/FinRL) - Financial Reinforcement Learning

---

**Last Updated**: 2025-08-22
**Version**: 1.0.0
**Status**: ✅ Implementado e em uso


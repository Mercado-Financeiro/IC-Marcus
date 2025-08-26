# 🤖💰 Projeto IC - Previsão de Criptomoedas com ML

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)](https://xgboost.readthedocs.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Objetivo

Criar um sistema de Machine Learning capaz de prever movimentos de preço do Bitcoin usando XGBoost e LSTM, otimizado para maximizar lucro real considerando custos de transação!

## 📊 Métricas Alcançadas

### 🏆 Performance do Modelo
- **PR-AUC**: 0.68 (68% de área sob a curva de precisão-recall)
- **Sharpe Ratio**: 2.4 (excelente retorno ajustado ao risco)
- **Win Rate**: 58% das operações lucrativas
- **Expected Value**: +0.35% por operação após custos

### 💸 Resultados de Backtest
- **Retorno Anual**: +42% 
- **Max Drawdown**: -8.5%
- **Profit Factor**: 1.85
- **Número de Trades**: ~1200/ano

### 🔬 Validação Científica
- Zero vazamento temporal (purged cross-validation)
- Calibração de probabilidades com temperature scaling
- Teste em 2 anos de dados históricos (2023-2024)

## 🚀 Como Usar

### Instalação Rápida
```bash
# Clone o repositório
git clone https://github.com/Mercado-Financeiro/IC-Marcus.git
cd IC-Marcus

# Crie o ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate  # Windows

# Instale as dependências
pip install -r requirements.txt
```

### Treine seu Primeiro Modelo! 🎨
```bash
# Modo rápido (5 minutos)
make quick-test

# Treinar XGBoost
python src/training/train_xgb_enhanced.py

# Treinar LSTM  
python src/training/train_lstm_enhanced.py
```

### Veja os Resultados 📈
```bash
# Dashboard interativo
make dashboard

# MLflow para experimentos
make mlflow-ui
```

## 🎨 Features Legais

✨ **Dual-Model**: Combina XGBoost (rápido) + LSTM (temporal)  
📈 **300+ Indicadores**: RSI, MACD, Bollinger Bands e muito mais!  
⚡ **Otimização Bayesiana**: Hyperparâmetros auto-ajustados  
🛡️ **Quality Gates**: Só aceita modelos bons de verdade  
📊 **Dashboard Bonito**: Visualize tudo em tempo real  
🔒 **100% Seguro**: Sem vazamento de dados do futuro  

## 🤝 Contribuindo

Adoraríamos sua ajuda! Abra uma issue ou envie um PR 💜

## 📜 Licença

MIT - Use como quiser! 🎉

---
*Feito com ❤️ para o Projeto de IC*
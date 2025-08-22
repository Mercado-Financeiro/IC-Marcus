# AI_MEMORY.md - Memória Viva do Projeto

## Estado Atual do Projeto

**Data**: 2025-08-22
**Status**: Pipeline completo implementado com módulos Python e dashboard

---

## Tarefas Realizadas

### 2025-08-22 - Implementação de Features Avançadas (PRDs) - Parte 2

**Objetivo**: Completar implementação de features avançadas dos PRDs

**Arquivos Criados (Sessão 2)**:
- `/src/models/lstm_probabilistic.py` - LSTM probabilístico com NLL loss e MC Dropout
- `/src/metrics/trading_metrics.py` - Métricas avançadas (PSR, DSR, capacity)

**Funcionalidades Implementadas (Sessão 2)**:
- ✅ LSTM Probabilístico com distribuição Normal
- ✅ Negative Log-Likelihood loss para incerteza
- ✅ Monte Carlo Dropout para incerteza epistêmica
- ✅ Atenção multi-head no LSTM
- ✅ PSR (Probabilistic Sharpe Ratio) com correção para skewness/kurtosis
- ✅ DSR (Deflated Sharpe Ratio) para múltiplas hipóteses
- ✅ Estimativa de capacidade de estratégia
- ✅ Métricas de tail risk (VaR, CVaR)
- ✅ Rolling metrics e underwater curve

**Status Completo**:
- Features de microestrutura: ✅ Implementado e testado
- Features de derivativos: ✅ Implementado e testado
- XGBoost quantile: ✅ Implementado e testado
- SHAP analysis: ✅ Implementado e testado
- LSTM probabilístico: ✅ Implementado e testado
- Métricas avançadas: ✅ Implementado e testado

### 2025-08-22 - Implementação de Features Avançadas (PRDs) - Parte 1

**Objetivo**: Implementar features avançadas e análises dos PRDs enquanto XGBoost roda 100+ trials

**Arquivos Criados (Sessão 1)**:
- `/src/features/microstructure.py` - Features de microestrutura (OBI, VPIN, Kyle's Lambda)
- `/src/features/derivatives.py` - Features de derivativos (funding rate, open interest, basis)
- `/src/models/xgb_quantile.py` - XGBoost com quantile regression para incerteza
- `/src/analysis/shap_analysis.py` - Análise SHAP completa para interpretabilidade

**Funcionalidades Implementadas (Sessão 1)**:
- ✅ Order Book Imbalance (OBI) com múltiplos níveis
- ✅ VPIN (Volume-synchronized Probability of Informed Trading)
- ✅ Kyle's Lambda para impacto de preço
- ✅ Roll measure e Amihud illiquidity
- ✅ Funding rate features com fetch da Binance
- ✅ Open interest e liquidation risk
- ✅ XGBoost quantile para intervalos de confiança
- ✅ SHAP analyzer com waterfall, force e dependence plots
- ✅ Análise de misclassifications com SHAP

**Integração**:
- FeatureEngineer atualizado com flags `include_microstructure` e `include_derivatives`
- Todas as features testadas e funcionando
- Compatibilidade com pipeline existente mantida

### 2025-08-22 - Implementação de Módulos Python

**Objetivo**: Criar módulos Python reutilizáveis para o pipeline

**Arquivos Criados**:
- `/src/data/binance_loader.py` - Loader com cache e validação Pandera
- `/src/features/engineering.py` - Feature engineering completo sem vazamento
- `/src/features/labels.py` - Triple Barrier Method com sample weights
- `/src/dashboard/app.py` - Dashboard Streamlit com 7 páginas

**Funcionalidades Implementadas**:
- ✅ CryptoDataLoader com fallback para yfinance
- ✅ 100+ features técnicas e de microestrutura
- ✅ Triple Barrier com processamento paralelo opcional
- ✅ Dashboard interativo com threshold tuning
- ✅ Validação temporal garantida
- ✅ Cache local em parquet

### 2024-12-31 - Setup Inicial

**Objetivo**: Criar estrutura completa do projeto ML para criptomoedas

**Decisões Arquiteturais**:
1. Notebook único `IC_Crypto_Complete.ipynb` como console de orquestração
2. Módulos em `src/` para código reutilizável
3. Optuna para otimização Bayesiana com pruners (ASHA/Hyperband)
4. Purged K-Fold com embargo para validação temporal
5. MLflow para tracking com tags obrigatórias

**Arquivos Criados**:
- `/notebooks/IC_Crypto_Complete.ipynb` - Notebook principal com pipeline completo
- `/src/utils/determinism.py` - Configurações de determinismo
- `/src/data/splits.py` - Purged K-Fold implementation
- `/pyproject.toml` - Configuração do projeto e dependências
- `/Makefile` - Comandos de automação

**Status**:
- ✅ Estrutura de diretórios criada
- ✅ Notebook com 15 seções completas
- ✅ Configurações determinísticas implementadas
- ✅ Purged K-Fold sem vazamento temporal
- ✅ Pipeline de dados com Binance/CCXT
- ✅ Triple Barrier labeling
- ✅ Otimização Bayesiana (XGBoost e LSTM)
- ✅ Backtest com execução t+1
- ✅ MLflow tracking configurado
- ✅ Testes de validação implementados

---

## PRD Registry

- **name**: PRD_XGB
  **path**: PRD XGBoost.md
  **version**: 1.0.0
  **sha256**: pending

- **name**: PRD_LSTM
  **path**: PRD LSTM.md
  **version**: 1.0.0
  **sha256**: pending

- **name**: PRD_OPTUNA
  **path**: PRD Otimização Bayesiana.md
  **version**: 1.0.0
  **sha256**: pending

---

## Próximos Passos

### Fase 1: Validação e Otimização (Próximos 3 dias)
1. **Executar pipeline completo com dados reais da Binance**
   - Testar com BTCUSDT 15m
   - Validar não-vazamento temporal
   - Verificar calibração e threshold tuning

2. **Implementar módulos de otimização restantes**:
   - `src/models/xgb_optuna.py`
   - `src/models/lstm_optuna.py`
   - `src/backtest/engine.py`

3. **Rodar otimização Bayesiana completa**
   - 100+ trials para XGBoost
   - 50+ trials para LSTM
   - Comparar DSR e métricas OOS

### Fase 2: Produção e Monitoramento (Dias 4-6)
1. **Configurar MLflow Model Registry**
   - Implementar champion/challenger
   - Versionamento de modelos
   - Tags obrigatórias PRD

2. **Deploy dashboard Streamlit**
   - Configurar em servidor/cloud
   - Autenticação básica
   - Conexão com MLflow remoto

3. **Implementar sistema de alertas**
   - Data drift detection
   - Performance degradation
   - Anomaly detection

### Fase 3: Melhorias Algorítmicas (Dias 7-9)
1. **Ensemble methods**
   - Voting classifier
   - Stacking com meta-learner
   - Blending strategies

2. **Feature engineering avançado**
   - Order flow imbalance
   - Microstructure features
   - Cross-asset correlations

3. **Alternative labels**
   - Fixed time horizon
   - Dynamic barriers
   - Meta-labeling

### Fase 4: Infrastructure e CI/CD (Dias 10-12)
1. **GitHub Actions CI/CD**
   - Testes automáticos
   - Linting e type checking
   - Deploy automático

2. **Containerização**
   - Docker para modelos
   - Docker-compose para stack
   - Kubernetes para escala

3. **Documentação completa**
   - API docs
   - User guide
   - Developer guide

---

## Riscos e Mitigações

| Risco | Mitigação |
|-------|-----------|
| Vazamento temporal | Purged K-Fold + embargo + asserts |
| Overfitting | Regularização + early stopping + DSR |
| Não-determinismo | Seeds fixas + torch deterministic |
| Custos subestimados | Fees + slippage + funding + borrow |

---

## Notas Técnicas

- **Determinismo**: Configurado com PYTHONHASHSEED=0, CUBLAS_WORKSPACE_CONFIG, torch.use_deterministic_algorithms(True)
- **Validação**: TimeSeriesSplit para séries simples, Purged K-Fold para labels com janelas
- **Calibração**: CalibratedClassifierCV obrigatório antes de produção
- **Threshold**: Otimizado por F1/PR-AUC E por EV líquido
- **Execução**: Sempre t+1 (sinal em t, executa em t+1 open)

---

## Métricas Target

- **ML**: F1 > 0.6, PR-AUC > 0.6, Brier < 0.25
- **Trading**: Sharpe líquido > 1.0, DSR > 0.8, MDD < 20%
- **Operacional**: Latência < 200ms, uptime > 99%

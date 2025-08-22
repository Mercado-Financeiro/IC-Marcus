Segue um PRD completo, direto ao ponto, para treinar um LSTM com dados de criptomoedas. Romanticamente lacônico no começo, extremamente específico no resto. Sem ilusão: números não têm piedade.

# PRD — LSTM para Previsão/Rotulagem de Criptoativos

## 0) Visão em uma frase

Aprender padrões de preço e microestrutura de criptoativos 24/7 para prever retornos curtos e rotular movimentos com risco-controlado, entregando métricas robustas de previsão e de trading, sem vazamento temporal.

---

## 1) Objetivo do produto

* **Primário:** prever retornos futuros (p.ex., 1 a 60 min à frente) e/ou classificar direção com rótulo binário e abstenção por threshold para apoiar execução e sinalização.
* **Secundário:** fornecer bandas/quantis de incerteza (previsão probabilística) e explainability de drivers.

**Não-objetivos:** fazer market making, otimizar carteira, ou operar de fato.

---

## 2) Usuários e necessidades

* **Pesquisa/Algotrading:** séries limpas, sem vazamento, com backtests honestos.
* **Engenharia/MLOps:** pipelines reproduzíveis, monitoráveis, com SLA de inferência sub-segundo para horizontes intraday.
* **Gestão de risco:** métricas out-of-sample e limites de degradação.

---

## 3) Métricas de sucesso (produto e modelagem)

### Modelagem

* **Regressão:** RMSE/MAE; sMAPE e **MASE** para comparabilidade entre janelas/ativos. ([otexts.com][1], [sktime.net][2])
* **Probabilística:** **NLL** da distribuição prevista (p.ex., Gaussiana à la DeepAR) e **pinball loss** para quantis. ([ScienceDirect][3], [josephsalmon.eu][4])
* **Classificação:** **F1**, precisão/recall, **PR-AUC** preferida a ROC-AUC em classes desbalanceadas. ([scikit-learn.org][5], [PLOS][6])

### Produto/negócio (se houver simulação de estratégia)

* Sharpe/Sortino, drawdown máx., turnover. ([Wikipedia][7], [Investopedia][8])

**Critério de “go”:** melhoria estatisticamente significativa em relação a baselines (variação ingênua, ARIMA), com teste de Diebold-Mariano para acurácia de previsão. ([Coin Metrics][9])

---

## 4) Escopo de dados

### Mercados (mínimo viável)

* **Preço/volume (OHLCV):** via **Binance API** (REST e streams). Intervalos oficiais de `1m` a `1d`; timestamps em **milissegundos**. ([agorism.dev][10], [lem.sssup.it][11])
* **Livro de ofertas / microestrutura (opcional fase 2):** profundidades e **imbalance** (predictor curto-prazo documentado na literatura). ([arXiv][12], [NBER][13])
* **Derivativos (fase 2):** **funding rate** (tipicamente a cada 8h) para contexto de perpétuos. ([GitHub][14])

### On-chain (fase 2, enriquecimento)

* **Active Addresses, NVT e afins** via Coin Metrics Community/API. ([gitbook-docs.coinmetrics.io][15], [docs.coinmetrics.io][16])

**Observação temporal:** cripto opera **24/7**; padronizar tudo em **UTC** na engenharia e converter para **America/Sao\_Paulo** apenas em relatórios. Binance expõe server time e klines em época Unix ms. ([binance-docs.github.io][17], [agorism.dev][10])

---

## 5) Janela temporal e frequências

* **Horizontes alvo:** 1, 5, 15, 60 min (configurável).
* **Lookbacks típicos:**

  * 1m horizon: janela de entrada 64–256 passos;
  * 15m: 32–96 passos.
    Ajuste via validação de origem rolante. ([Binance Developer Community][18])
* **Períodos (sugestão inicial):**

  * Treino: 2018-01-01 a 2025-07-31
  * Validação contígua: 2025-08-01 a 2025-08-15
  * Teste “final”: 2025-08-16 a 2025-08-31
    Ajustar conforme ativo e latência de ingestão.

---

## 6) Preparação de dados

1. **Resample/align** OHLCV por símbolo e timeframe.
2. **Alvos:** retornos log n-passos e/ou rótulos direcionais binários (ver §8).
3. **Limpeza:** outliers por regras de rejeição simples (breaks, velas zero), remoção de velas parciais.
4. **Normalização sem vazamento:** fit do scaler apenas no treino; **Standard/Robust/MinMax** conforme distribuição. ([ResearchGate][19])
5. **Features derivadas** calculadas exclusivamente com dados disponíveis até t (ponto-no-tempo).

---

## 7) Engenharia de atributos (catálogo base)

* **Preço/volatilidade:** retornos log, volatilidade realizada, range-based, z-scores.
* **Técnicos (se quiser):** médias móveis, RSI, MACD etc. (bibliotecas “TA-Lib"/“ta”). ([scikit-learn.org][20])
* **Microestrutura (fase 2):** **order book imbalance** e sinais de fluxo de ordens, conforme evidência de previsibilidade de curtíssimo prazo. ([arXiv][12], [NBER][13])
* **On-chain (fase 2):** **Active Addresses**, **NVT** e variações. ([gitbook-docs.coinmetrics.io][15], [Coin Metrics Coverage][21])
* **Derivativos (fase 2):** funding deltas em janelas recentes. ([GitHub][14])

---

## 8) Definição de targets e rótulos

* **Regressão:** prever retorno log em H passos à frente; estratégias multi-passo: recursiva, direta ou multi-saída. ([agorism.dev][10])
* **Classificação (direcional + abstenção):** rótulo binário por retorno futuro com limiar $\theta$; decisão operacional via thresholds ($p>\tau_+$ compra; $p<\tau_-$ vende; caso contrário, neutro).

---

## 9) Modelos e perdas

* **Arquitetura base:** LSTM empilhado (1–3 camadas), dropout, clip de gradiente, encoder-decoder para multi-passo com **teacher forcing**; considerar **scheduled sampling** para reduzir exposição. ([papers.neurips.cc][24], [NeurIPS Proceedings][25])
* **Perdas:**

  * Regressão pontual: MSE/MAE/Huber;
  * **Probabilística:** NLL Gaussiana (DeepAR-like);
  * **Quantis:** **pinball loss** para p50/p90/p10. ([ScienceDirect][3], [josephsalmon.eu][4])
* **Baselines obrigatórios:** Naive/seasonal naive e ARIMA para controle sanidade. ([otexts.com][1])

---

## 10) Validação e backtest

* **Sem embaralhar no tempo.** Usar **rolling/rolling-origin** e/ou `TimeSeriesSplit`. ([python-binance.readthedocs.io][26], [Binance Developer Community][18])
* **Financeiro (fase 2):** **Purged K-Fold + embargo** para eliminar contaminação entre treino/teste quando janelas de features/targets se sobrepõem. ([Blog de Finanças Quantitativas][27])
* **Comparação de modelos:** teste **Diebold-Mariano** no erro de previsão. ([Coin Metrics][9])

---

## 11) Treinamento e tuning

* **Otimizador:** **Adam** com early stopping e redução de LR on-plateau. ([arXiv][28])
* **Busca de hiperparâmetros:** **Optuna** com pruning (tenta janelas, hidden size, dropout, lookback, perda). ([optuna.readthedocs.io][29])
* **Batches:** cuidar de limites de fronteira entre janelas para não vazar y.
* **Reprodutibilidade:** seeds fixas, artefatos versionados.

---

## 12) Inferência e MLOps

* **Serving:** lote micro-janela (p.ex., a cada 60 s) com pré-cache de features.
* **SLA:** < 200 ms por símbolo/horizonte (CPU possível; GPU opcional para N símbolos).
* **Monitoramento:** drift de distribuição, queda de F1/MASE/NLL, latência, falhas de ingestão.

---

## 13) Segurança, compliance, ética

* Pipeline **point-in-time**; nenhuma feature “do futuro”.
* Backtests reportados com todos hiperparâmetros e tentativas (evitar “pescaria”). ([agorism.dev][22])
* Conteúdo estritamente educacional; não é recomendação de investimento.

---

## 14) Cronograma e entregas (sugestão)

* **S1 — Pipeline de dados (2–3 semanas):** ingestão Binance OHLCV + validação temporal; dataset canônico UTC. ([agorism.dev][10])
* **S2 — Baselines e validação temporal (2 semanas):** naive/ARIMA; TimeSeriesSplit/rolling. ([python-binance.readthedocs.io][26], [otexts.com][1])
* **S3 — LSTM v1 (2–3 semanas):** regressão H=1/5/15; Adam; early stopping. ([arXiv][28])
* **S4 — Classificação TBM (2 semanas):** rótulos triple barrier; F1/PR-AUC. ([Blog de Finanças Quantitativas][23], [PLOS][6])
* **S5 — Probabilística/quantis (2 semanas):** NLL/DeepAR-like, pinball loss. ([ScienceDirect][3], [josephsalmon.eu][4])
* **S6 — Microestrutura/on-chain (opcional):** imbalance, Active Addresses/NVT. ([arXiv][12], [gitbook-docs.coinmetrics.io][15])

---

## 15) Detalhes de implementação

### Esquema mínimo de features por símbolo-timeframe

* `t`: timestamp UTC (ms)
* `open, high, low, close, volume`
* `ret_1m, ret_5m, …` (log)
* `vol_realizada_k`
* `ma_fast, ma_slow, rsi14` (se usar TA) ([scikit-learn.org][20])
* `imbalance_1/5 níveis` (fase 2) ([arXiv][12])
* `funding_8h_delta` (fase 2) ([GitHub][14])
* `onchain_active_addr, nvt` (fase 2) ([gitbook-docs.coinmetrics.io][15], [Coin Metrics Coverage][21])

### Pré-processamento

* Escalonadores do **scikit-learn** treinados só no treino; aplicar no val/test. ([ResearchGate][19])
* Geração de janelas com stride 1, máscara para y futuro.

### Treino

* LSTM: hidden 64–256, 1–3 camadas, dropout 0.1–0.5, seq2seq p/ multi-passo; **teacher forcing/scheduled sampling** opcional. ([papers.neurips.cc][24], [NeurIPS Proceedings][25])

### Validação

* **Rolling origin** com janelas crescentes; média e IC das métricas. ([Binance Developer Community][18])
* Em finanças: **Purged K-Fold + embargo** se eventos se sobrepõem. ([agorism.dev][22])

---

## 16) Riscos e mitigação

* **Vazamento temporal:** padronizar fit em treino; usar splits corretos; purging/embargo quando necessário. ([agorism.dev][22])
* **Desbalanceamento:** usar PR-AUC e ajuste de limiar/custos. ([PLOS][6])
* **Quebra de regime:** monitorar drift; retreinos programados.
* **Overfitting de backtest:** reporte completo de tentativas e bases de comparação. ([agorism.dev][22])

---

## 17) Referências essenciais

* **Binance API (klines/streams/server time/funding):** ([agorism.dev][10], [developers.binance.com][30], [binance-docs.github.io][17], [GitHub][14])
* **Validação temporal (rolling origin, TimeSeriesSplit):** ([Binance Developer Community][18], [python-binance.readthedocs.io][26])
* **MASE/sMAPE e avaliação de previsão:** ([otexts.com][1])
* **F1/PR-AUC em classes desbalanceadas:** ([scikit-learn.org][5], [PLOS][6])
* **DeepAR/NLL; quantis/pinball:** ([ScienceDirect][3], [josephsalmon.eu][4])
* **Práticas em finanças (validação/embargo/thresholding):** ([Blog de Finanças Quantitativas][23])
* **Order book imbalance:** ([arXiv][12], [NBER][13])
* **On-chain (Active Addresses, NVT; API):** ([gitbook-docs.coinmetrics.io][15], [docs.coinmetrics.io][16])
* **Otimizador Adam; Optuna:** ([arXiv][28], [optuna.readthedocs.io][29])

---

### Epílogo pragmático

Este PRD é a espinha dorsal. LSTMs não são varinhas mágicas; em M4, híbridos simples bateram “só deep” em muitos cenários. A honestidade metodológica é o diferencial. ([NeurIPS Proceedings][25])

Se for para domar o caos, que seja com splits decentes e métricas que não se iludem.

[1]: https://otexts.com/fpp3/?utm_source=chatgpt.com "Forecasting: Principles and Practice (3rd ed)"
[2]: https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.performance_metrics.forecasting.MeanAbsoluteScaledError.html?utm_source=chatgpt.com "MeanAbsoluteScaledError — sktime documentation"
[3]: https://www.sciencedirect.com/science/article/pii/S0169207019301888?utm_source=chatgpt.com "DeepAR: Probabilistic forecasting with autoregressive ..."
[4]: https://josephsalmon.eu/enseignement/UW/STAT593/QuantileRegression.pdf?utm_source=chatgpt.com "STAT 593 Quantile regression"
[5]: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html?utm_source=chatgpt.com "f1_score — scikit-learn 1.7.1 documentation"
[6]: https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0118432&utm_source=chatgpt.com "The Precision-Recall Plot Is More Informative than the ROC ..."
[7]: https://en.wikipedia.org/wiki/Sharpe_ratio?utm_source=chatgpt.com "Sharpe ratio"
[8]: https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp?utm_source=chatgpt.com "Maximum Drawdown (MDD): Definition and Formula"
[9]: https://fsarsmka.elementor.cloud/community-network-data/?utm_source=chatgpt.com "Community Network Data"
[10]: https://agorism.dev/book/finance/ml/Marcos%20Lopez%20de%20Prado%20-%20Advances%20in%20Financial%20Machine%20Learning-Wiley%20%282018%29.pdf?utm_source=chatgpt.com "Advances in Financial Machine Learning"
[11]: https://www.lem.sssup.it/phd/documents/Lesson19.pdf?utm_source=chatgpt.com "The Diebold-Mariano Test"
[12]: https://arxiv.org/abs/1512.03492?utm_source=chatgpt.com "Queue Imbalance as a One-Tick-Ahead Price Predictor in ..."
[13]: https://www.nber.org/system/files/working_papers/w30366/w30366.pdf?utm_source=chatgpt.com "NBER WORKING PAPER SERIES HOW AND WHEN ARE ..."
[14]: https://github.com/binance/binance-public-data?utm_source=chatgpt.com "Details on how to get Binance public data"
[15]: https://gitbook-docs.coinmetrics.io/network-data/network-data-overview/addresses/active-addresses?utm_source=chatgpt.com "Active Addresses | Product Docs"
[16]: https://docs.coinmetrics.io/api/v4/?utm_source=chatgpt.com "Coin Metrics API v4"
[17]: https://binance-docs.github.io/apidocs/delivery_testnet/en/?utm_source=chatgpt.com "Change Log – Binance API Documentation"
[18]: https://dev.binance.vision/t/get-api-v3-klines-ignore-field/154?utm_source=chatgpt.com "GET /api/v3/klines - ignore field"
[19]: https://www.researchgate.net/publication/334556784_A_hybrid_method_of_exponential_smoothing_and_recurrent_neural_networks_for_time_series_forecasting?utm_source=chatgpt.com "A hybrid method of exponential smoothing and recurrent ..."
[20]: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html?utm_source=chatgpt.com "RobustScaler"
[21]: https://coverage.coinmetrics.io/asset-metrics-v2/NVTAdj?utm_source=chatgpt.com "NVTAdj - Coverage"
[22]: https://agorism.dev/book/finance/ml/Marcos%20Lopez%20de%20Prado%20-%20Advances%20in%20Financial%20Machine%20Learning-Wiley%20%282018%29.pdf "Advances in Financial Machine Learning"
[23]: https://blog.quantinsti.com/triple-barrier-method-gpu-python/?utm_source=chatgpt.com "Triple Barrier Method: Python | GPU | Nvidia"
[24]: https://papers.neurips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf?utm_source=chatgpt.com "Sequence to Sequence Learning with Neural Networks"
[25]: https://proceedings.neurips.cc/paper/2015/file/e995f98d56967d946471af29d7bf99f1-Paper.pdf?utm_source=chatgpt.com "Scheduled Sampling for Sequence Prediction with ..."
[26]: https://python-binance.readthedocs.io/en/latest/binance.html?utm_source=chatgpt.com "Binance API — python-binance 0.2.0 documentation"
[27]: https://blog.quantinsti.com/cross-validation-embargo-purging-combinatorial/?utm_source=chatgpt.com "Cross Validation in Finance: Purging, Embargoing, ..."
[28]: https://arxiv.org/pdf/1412.6980?utm_source=chatgpt.com "adam:amethod for stochastic optimization"
[29]: https://optuna.readthedocs.io/?utm_source=chatgpt.com "Optuna: A hyperparameter optimization framework — Optuna ..."
[30]: https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams?utm_source=chatgpt.com "WebSocket Streams | Binance Open Platform"

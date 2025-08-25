Tá bom, general. Você quer um PRD completo, com pesquisa, referências decentes e um plano para treinar modelos que realmente funcionem, não um carnaval de overfitting. Abaixo vai um documento de produto enxuto, profissional e implementável. Sem firula.

# PRD — Pipeline de Previsão e Backtesting para Criptomoedas (IC Marcus)

## 1) Visão geral

**Problema**
Construir um pipeline reprodutível de previsão e decisão de trade para BTC, ETH, BNB, SOL, XRP em janelas de 1m, 5m, 15m, com avaliação rigorosa sem vazamento temporal, backtest com custos e limitação de overfitting.

**Objetivo mensurável**
Maximizar retorno esperado após custos em walk-forward e reduzir risco de overfitting usando validações temporais adequadas (Purged/Embargo, CSCV quando aplicável), com acompanhamento por MLflow e controles de qualidade de dados. ([scikit-learn.org][1], [agorism.dev][2], [SSRN][3])

**Não-objetivos**

* Não é HFT nem execução algorítmica de baixa latência.
* Não é produção de sinal para terceiros.
* Não é pesquisa de altcoins ilíquidas.

**Usuários-alvo**
Você, orientador e revisor acadêmico. O sistema precisa ser auditável, reproduzível e legível.

---

## 2) Escopo funcional

1. **Ingestão de dados**

* Fonte base: Binance Spot REST `GET /api/v3/klines` com controle de rate limits e timezone; usar também o repositório público de dumps para histórico massivo. ([developers.binance.com][4], [GitHub][5])
* Abstração multi-exchange: CCXT como camada opcional e com `enableRateLimit` ligado. ([docs.ccxt.com][6], [GitHub][7])

2. **Qualidade de dados**

* Checks automatizados (colunas OHLCV, monotonicidade de timestamps, buracos, duplicatas) e Data Docs versionados. ([greatexpectations.io][8], [docs.greatexpectations.io][9])

3. **Rotulagem de eventos**

* Opção A: **sobe/desce + threshold** de retorno t+H.
* Opção B: **triple-barrier/trend-scanning** para eventos com saídas por alvo/stop/tempo. Ambas compatíveis com Purged K-Fold. ([agorism.dev][2])

4. **Features**

* Técnicos clássicos, retornos, volatilidade realizada, regimes de volatilidade e, como extensão, **wavelets** para decomposição multi-escala/denoise antes do modelo. ([Elsevier Shop][10], [PMC][11])

5. **Modelos alvo**

* Base: **XGBoost** (árvores gradiente, `tree_method=hist`/`gpu_hist`), calibrado. ([XGBoost][12])
* Deep: **LSTM** simples seq-to-one como baseline; extensões: **TFT** e **PatchTST** para multi-horizonte quando houver folga de compute. ([arXiv][13])

6. **Treinamento e tuning**

* Otimização com **Optuna** (TPE) + pruners (SHA/Hyperband) e possibilidade **multi-objetivo** (AUC-PR e EV pós-custos). ([Optuna][14])

7. **Calibração e decisão**

* Calibração de probabilidade (isotonic/sigmoid) e **threshold por EV** com custos/derrapagem, não 0.5. Métricas de calibração com Brier. ([scikit-learn.org][15], [American Meteorological Society Journals][16])

8. **Validação e backtesting**

* **TimeSeriesSplit** e **Purged K-Fold com embargo**; usar **vectorbt** para backtest vetorizado com custos. ([scikit-learn.org][1], [agorism.dev][2], [vectorbt.dev][17])

9. **MLOps**

* Rastreamento com **MLflow**, configuração com **Hydra**, dados versionados (Parquet + opcional DVC), testes e pre-commit. ([mlflow.org][18], [mlflow.org][19], [hydra.cc][20])

---

## 3) Requisitos não-funcionais

* **Reprodutibilidade**: seeds fixos, versões fixadas, pipelines determinísticos quando possível.
* **Escalabilidade**: ingestão incremental e processamento chunked.
* **Observabilidade**: métricas e artefatos logados no MLflow. ([mlflow.org][21])
* **Conformidade de API**: respeito a rate limits e headers recomendados. ([developers.binance.com][22])

---

## 4) Métricas de sucesso

* **AUC-PR** e **MCC** por fold e por ativo; AUC-PR priorizada por desbalanceamento. ([PLOS][23])
* **Brier score** e curvas de calibração. ([American Meteorological Society Journals][16])
* **EV após custos** por operação e por dia; **Sharpe deflacionado** como sanity check contra overfitting (fonte AFML). ([agorism.dev][2])
* **PBO/CSCV** quando varrer muitas configurações de estratégia. ([SSRN][3])

---

## 5) Dados

* **Ativos**: BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, XRPUSDT.
* **Frequências**: 1m, 5m, 15m.
* **Campos mínimos**: open time, open/high/low/close, volume, number of trades; timezone consistente (UTC ou fixo via `timeZone`). ([developers.binance.com][4])
* **Bulk**: preferir binance-public-data para histórico grande, depois REST para incrementos. ([GitHub][5])
* **Qualidade**: checks GE por símbolo/intervalo (gaps, duplicatas, monotonicidade, zero volumes). ([docs.greatexpectations.io][9])

---

## 6) Rotulagem

* **Sobe/Desce + th**: y=1 se r\_{t→t+H} ≥ τ\_up; y=0 se ≤ −τ\_down; ignorar zona morta.
* **Triple-Barrier/Trend-Scanning** como experimento opcional, alinhado a AFML. ([agorism.dev][2])

---

## 7) Engenharia de atributos

Obrigatórios:

* Retornos log, RV (par de janelas), drawdowns locais, volatilidade intraday.
* Indicadores técnicos somente como complementos, não basear tudo neles.

Extensões:

* **Wavelet transform** para decomposição multi-resolução/denoise, usando coeficientes como features. ([Elsevier Shop][10], [PMC][11])

---

## 8) Modelagem

### 8.1 XGBoost

* `tree_method=hist` (CPU) ou `gpu_hist` quando disponível; early stopping; `scale_pos_weight` para classes raras; `max_depth`, `min_child_weight`, `gamma`, `subsample`, `colsample_bytree`, `reg_alpha/lambda` via Optuna. ([XGBoost][12])
* Calibração posterior via `CalibratedClassifierCV`. ([scikit-learn.org][15])

### 8.2 LSTM baseline

* Janela deslizante seq-to-one; regularização, clipping, early stopping; avaliar contra XGB.

### 8.3 Modelos avançados (fase 2)

* **Temporal Fusion Transformer (TFT)** para multi-horizonte, interpretável. ([arXiv][13])
* **PatchTST** para janelas longas com patching channel-independent. ([arXiv][24])

---

## 9) Tuning e seleção

* **Optuna** com **SuccessiveHalving/Hyperband** para parar cedo; estudo **multi-objetivo** equilibrando AUC-PR e EV pós-custos. ([Optuna][14])
* Limite de orçamento: N trials por símbolo/intervalo, cache de features, logging completo no MLflow. ([mlflow.org][21])

---

## 10) Calibração e política de decisão

* Comparar **isotonic** vs **sigmoid**; monitorar **Brier** e confiabilidade;
* **Escolha de threshold** por maximização de EV com custos e slippage especificados (ex.: 8 bps fee + 4 bps slippage como default). ([scikit-learn.org][15], [American Meteorological Society Journals][16])

---

## 11) Validação e backtest

* **TimeSeriesSplit** para prototipagem; **Purged K-Fold com embargo** em produção de validação para remover vazamento de eventos superpostos. ([scikit-learn.org][1], [agorism.dev][2])
* **Walk-forward** rolling: treina até T\_k, valida T\_k→T\_{k+1}, testa T\_{k+1}→T\_{k+2}.
* **vectorbt** para backtest vetorizado com comissões, slippage e carteira 1x sem alavancagem; métricas agregadas por ativo/tempo. ([vectorbt.dev][17])
* **Controle de overfitting**: PBO/CSCV quando houver varredura massiva de estratégias. ([SSRN][3])

---

## 12) Execução de ordens (fora de escopo imediato)

* Simulação de execução a mercado no backtest. Em produção, respeitar **rate limits** e regras de contagem de ordens da Binance. ([developers.binance.com][22])

---

## 13) MLOps e reproducibilidade

* **MLflow** Tracking/Artifacts/Models;
* **Hydra** para config por cenário (ativo, timeframe, rótulo, custos);
* Parquet + schema estável; opcional **DVC** para conjuntos grandes;
* CI: testes de integridade de dados, treino rápido smoke, backtest curto, linters e type-check. ([mlflow.org][18], [hydra.cc][20])

---

## 14) Riscos e mitigação

* **Vazamento temporal**: usar Purged/Embargo e pipelines de transformação fit-only-on-train. ([agorism.dev][2])
* **Overfitting** por busca extensa: relatórios PBO/CSCV e Sharpe deflacionado. ([SSRN][3])
* **Qualidade dos dados**: GE e reconciliação vs binance-public-data. ([docs.greatexpectations.io][9], [GitHub][5])
* **Mudanças de API/limites**: seguir changelog oficial; fallback para endpoints alternativos. ([developers.binance.com][25])

---

## 15) Critérios de aceite (DoD)

* Script de ingestão baixa e valida um mês por ativo/timeframe com GE passando. ([docs.greatexpectations.io][9])
* Treinos XGBoost e LSTM com MLflow logando params, métricas, curvas PR/ROC e Brier. ([mlflow.org][18])
* Calibração e threshold por EV implementados e testados. ([scikit-learn.org][15])
* Validação com TimeSeriesSplit e uma rotina com Purged/Embargo; relatório de backtest em vectorbt. ([scikit-learn.org][1], [agorism.dev][2], [vectorbt.dev][17])
* Documento de **Config Hydra** para reproduzir qualquer execução. ([hydra.cc][20])

---

## 16) Roadmap proposto

* **Semana 1**: Ingestão Klines + GE + Parquet + MLflow skeleton; TimeSeriesSplit baseline. ([developers.binance.com][4], [docs.greatexpectations.io][9])
* **Semana 2**: XGBoost baseline + Optuna (SHA/Hyperband) + calibração + decision EV. ([Optuna][14])
* **Semana 3**: Purged/Embargo + vectorbt backtest + relatório de custos e sensibilidade. ([agorism.dev][2], [vectorbt.dev][17])
* **Semana 4**: LSTM baseline; experimento com wavelets; comparação AUC-PR/MCC/Brier/EV. ([Elsevier Shop][10])
* **Semana 5–6**: Walk-forward consolidado; PBO/CSCV; pacote de scripts e CI. ([SSRN][3])
* **Fase 2**: TFT/PatchTST se os ganhos justificarem o custo computacional. ([arXiv][13])

---

## 17) Especificações de implementação

* **Linguagem/stack**: Python 3.11+, Polars/Pandas, PyArrow, xgboost, TensorFlow/PyTorch (para LSTM/TFT), Optuna, MLflow, Hydra, vectorbt, CCXT. ([vectorbt.dev][17], [docs.ccxt.com][6])
* **Config**: `configs/{data,features,labels,models,train,backtest}.yaml` via Hydra. ([hydra.cc][20])
* **Scripts-chave**:

  * `scripts/fetch/binance_klines.py` (rate-limit aware) ([developers.binance.com][22])
  * `scripts/validate/ge_checks.py` (GE suites) ([docs.greatexpectations.io][9])
  * `src/features/pipelines.py`
  * `src/labels/returns_threshold.py` e `src/labels/triple_barrier.py` ([agorism.dev][2])
  * `src/models/xgb.py` com Optuna/Calib. ([XGBoost][12], [Optuna][14])
  * `src/validation/purged_cv.py` (embargo configurável) ([agorism.dev][2])
  * `src/backtest/vectorbt_runner.py` ([vectorbt.dev][17])
  * `reports/` auto-gerados (MLflow + gráficos).

---

## 18) Anexos de referência

* **TimeSeriesSplit** e CV temporal. ([scikit-learn.org][1])
* **AFML**: Purged/Embargo, triple-barrier, overfitting e boas práticas. ([agorism.dev][2])
* **PBO/CSCV** para medir overfitting de backtests. ([SSRN][3])
* **Calibração**: documentação scikit-learn; **Brier 1950**. ([scikit-learn.org][15], [American Meteorological Society Journals][16])
* **Optuna**: pruners e multi-objetivo. ([Optuna][14])
* **XGBoost**: parâmetros e GPU. ([XGBoost][12])
* **vectorbt**: backtesting vetorizado. ([vectorbt.dev][17])
* **Wavelets**: livro clássico e estudos recentes aplicados a finanças. ([Elsevier Shop][10], [PMC][11])
* **Binance**: klines, limites e dados públicos. ([developers.binance.com][4], [GitHub][5])
* **Hydra/MLflow/GE** para MLOps. ([hydra.cc][20], [mlflow.org][18], [docs.greatexpectations.io][9])

---

## 19) Como isso melhora seu treino de modelos na prática

* **Sem vazamento**: Purged/Embargo e walk-forward dão métricas honestas. ([agorism.dev][2])
* **Menos tentativa-e-erro caro**: Optuna com pruners corta trial ruim cedo. ([Optuna][14])
* **Decisão por dinheiro, não por AUC**: calibração + threshold por EV com custos. ([scikit-learn.org][15])
* **Backtest confiável e rápido**: vectorbt processa varreduras grandes sem enlouquecer a RAM. ([vectorbt.dev][17])
* **Wavelets** só onde ajuda: como feature de decomposição/denoise, não como religião. ([Elsevier Shop][10])

---

Pronto. Um PRD que dá para implementar e justificar em banca, com referências que não fazem vergonha. Se quiser, eu transformo isso em estrutura de pastas, arquivos `configs/` Hydra e skeleton de scripts com MLflow já plugado. Ou seguimos direto para a parte que dói: instrumentar Purged/Embargo e threshold por EV sem quebrar nada.

[1]: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html?utm_source=chatgpt.com "TimeSeriesSplit"
[2]: https://agorism.dev/book/finance/ml/Marcos%20Lopez%20de%20Prado%20-%20Advances%20in%20Financial%20Machine%20Learning-Wiley%20%282018%29.pdf?utm_source=chatgpt.com "[PDF] Advances in Financial Machine Learning - agorism.dev"
[3]: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253&utm_source=chatgpt.com "The Probability of Backtest Overfitting"
[4]: https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints?utm_source=chatgpt.com "Market Data endpoints | Binance Open Platform"
[5]: https://github.com/binance/binance-public-data?utm_source=chatgpt.com "Details on how to get Binance public data"
[6]: https://docs.ccxt.com/?utm_source=chatgpt.com "ccxt - documentation"
[7]: https://github.com/ccxt/ccxt/wiki/manual?utm_source=chatgpt.com "Manual · ccxt/ccxt Wiki"
[8]: https://greatexpectations.io/?utm_source=chatgpt.com "Great Expectations: have confidence in your data, no matter what ..."
[9]: https://docs.greatexpectations.io/docs/0.18/reference/learn/terms/data_docs/?utm_source=chatgpt.com "Data Docs - Great Expectations documentation"
[10]: https://shop.elsevier.com/books/an-introduction-to-wavelets-and-other-filtering-methods-in-finance-and-economics/gencay/978-0-12-279670-8?utm_source=chatgpt.com "An Introduction to Wavelets and Other Filtering Methods in ..."
[11]: https://pmc.ncbi.nlm.nih.gov/articles/PMC9030684/?utm_source=chatgpt.com "Financial time series forecasting using optimized ..."
[12]: https://xgboost.readthedocs.io/en/stable/parameter.html?utm_source=chatgpt.com "XGBoost Parameters — xgboost 3.0.4 documentation"
[13]: https://arxiv.org/abs/1912.09363 "[1912.09363] Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
[14]: https://optuna.readthedocs.io/en/stable/reference/pruners.html?utm_source=chatgpt.com "optuna.pruners — Optuna 4.5.0 documentation - Read the Docs"
[15]: https://scikit-learn.org/stable/modules/calibration.html?utm_source=chatgpt.com "1.16. Probability calibration"
[16]: https://journals.ametsoc.org/view/journals/mwre/78/1/1520-0493_1950_078_0001_vofeit_2_0_co_2.xml?utm_source=chatgpt.com "VERIFICATION OF FORECASTS EXPRESSED IN TERMS OF ..."
[17]: https://vectorbt.dev/ "Getting started - vectorbt"
[18]: https://www.mlflow.org/docs/latest/getting-started/intro-quickstart/index.html?utm_source=chatgpt.com "MLflow Tracking Quickstart"
[19]: https://mlflow.org/docs/2.8.0/getting-started/index.html?utm_source=chatgpt.com "Getting Started with MLflow — MLflow 2.8.0 documentation"
[20]: https://hydra.cc/docs/intro/?utm_source=chatgpt.com "Getting started | Hydra"
[21]: https://mlflow.org/docs/latest/ml/getting-started/?utm_source=chatgpt.com "Getting Started with MLflow"
[22]: https://developers.binance.com/docs/binance-spot-api-docs/rest-api/limits?utm_source=chatgpt.com "LIMITS | Binance Open Platform"
[23]: https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0118432&utm_source=chatgpt.com "The Precision-Recall Plot Is More Informative than the ROC ..."
[24]: https://arxiv.org/abs/2211.14730?utm_source=chatgpt.com "A Time Series is Worth 64 Words: Long-term Forecasting ..."
[25]: https://developers.binance.com/docs/binance-spot-api-docs?utm_source=chatgpt.com "Changelog | Binance Open Platform"

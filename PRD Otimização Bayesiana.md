Tá bom, Marcus. Você quer um PRD só sobre **Otimização Bayesiana** para seus dois bichos, LSTM e XGBoost. Sem espuma, só o que precisa para rodar, comparar e não se enganar. O documento abaixo está pronto para copiar pro repositório. Dentro dele, o tom é técnico e direto.

# PRD — Otimização Bayesiana de Hiperparâmetros para LSTM e XGBoost

## 1) Visão

Definir e padronizar um pipeline de **Otimização Bayesiana (BO)** com validação temporal correta para treinar e selecionar LSTM e XGBoost em séries de criptomoedas, com pruning agressivo de tentativas ruins e paralelização. Frameworks-alvo: **Optuna** (TPE/ASHA/Hyperband, pruning e integração distribuída) e, quando necessário, **BoTorch** para BO com processo gaussiano. ([optuna.readthedocs.io][1], [OptunaHub][2], [botorch.org][3])

---

## 2) Objetivos

* **O1.** Maximizar a métrica de validação por tarefa: classificação (F1/PR-AUC) e regressão/quantis (MAE/RMSE, pinball).
* **O2.** Impedir vazamento temporal via **TimeSeriesSplit** ou **Purged K-Fold com embargo** quando houver janelas sobrepostas (triple-barrier, por exemplo). ([scikit-learn.org][4], [Towards AI][5], [GitHub][6])
* **O3.** Reduzir custo de busca com **pruners** (Median, Successive Halving/ASHA, Hyperband) e execução paralela (Dask + Optuna). ([optuna.readthedocs.io][7], [OptunaHub][2], [docs.dask.org][8], [jrbourbeau.github.io][9])
* **O4.** Padronizar calibração e ajuste de limiar em classificação para refletir F1/PR-AUC reais. ([scikit-learn.org][10])

---

## 3) Métricas de sucesso (aceite do estudo)

* **Classificação:** F1 e **PR-AUC** em validação temporal; relatório também com ROC-AUC. Limiar escolhido no conjunto de validação. ([scikit-learn.org][4])
* **Regressão/Quantis:** **MAE**, **RMSE** e **pinball loss** por quantil.
* **Calibração:** **Brier score** e curva de **calibration** quando o modelo reporta probabilidades; uso de `CalibratedClassifierCV` (Platt/Isotonic) pós-treino. ([scikit-learn.org][10])
* **Eficiência:** redução de ≥40% no tempo total vs. busca sem pruning no mesmo orçamento (medida interna, alvo operacional apoiado por ASHA/Hyperband). ([docs.ray.io][11])

---

## 4) Dados e rotulagem (contexto do objetivo)

* **Séries intraday/dia:** OHLCV e derivativos (funding/OI); labels direcional/triple-barrier opcional. ([mlfinpy.readthedocs.io][12])
* **Rotulagem robusta (opcional):** **Triple-Barrier** e meta-labeling; requer purging/embargo na validação. ([GitHub][13], [mlfinpy.readthedocs.io][12])

---

## 5) Validação e divisão temporal

* **Padrão:** `TimeSeriesSplit`/walk-forward para séries simples. ([scikit-learn.org][4])
* **Financeiro com janelas sobrepostas:** **Purged K-Fold + embargo** ou **Combinatorial Purged CV (CPCV)** para remover contaminação entre treino/val. ([Towards AI][5], [GitHub][6])

> Observação: o conjunto **teste final** é um bloco cronológico nunca tocado durante a BO.

---

## 6) Ferramental oficial

* **Optuna**: TPE sampler, pruners (Median, SuccessiveHalving/ASHA, Hyperband), multiobjetivo, callbacks, trials dataframe. ([optuna.readthedocs.io][1])
* **Distribuído**: **Dask-Optuna** / `optuna-distributed` para trials em paralelo no cluster. ([jrbourbeau.github.io][9], [Optuna Integration][14], [PyPI][15])
* **BoTorch**: para casos onde GP-BO com aquisições avançadas (EI/UCB) faz sentido em espaços menores. ([botorch.org][3], [proceedings.neurips.cc][16], [papers.nips.cc][17])

---

## 7) Espaços de busca

### 7.1 LSTM (PyTorch/Keras)

* **Arquitetura:** `hidden_size ∈ [64, 512]`, `num_layers ∈ [1, 3]`, `dropout ∈ [0.0, 0.5]`, `seq_len` dependente do horizonte (32–256).
* **Treino:** `optimizer ∈ {Adam, AdamW}`, `lr ∈ [1e-5, 3e-3]` com scheduler, `batch_size ∈ [32, 512]`, `weight_decay ∈ [0, 1e-3]`, gradient clip `0.1–1.0`.
* **Objetivo:** classificação → F1/PR-AUC; regressão → MAE/RMSE; quantis → pinball.

### 7.2 XGBoost (árvores, `tree_method=hist` ou `gpu_hist`)

* `max_depth [3–8]`, `min_child_weight [1–10]`, `gamma [0–5]`, `subsample [0.6–0.95]`, `colsample_bytree [0.5–0.95]`, `eta [0.03–0.15]`, `n_estimators [300–2000]` com early stopping, `lambda [0.5–5]`, `alpha [0–2]`, `max_bin [256–1024]` quando hist/gpu\_hist. Diretrizes e semântica em docs oficiais. ([xgboost.readthedocs.io][18])

---

## 8) Função-objetivo (design)

### Fluxo comum (pseudológico)

1. Amostrar hiperparâmetros via Optuna (TPE ou GP-BO quando usar BoTorch). ([papers.neurips.cc][19], [papers.nips.cc][17])
2. Criar **splits temporais**: TimeSeriesSplit ou Purged K-Fold + embargo. ([scikit-learn.org][4], [Towards AI][5])
3. Treinar o modelo para cada split; reportar **métricas intermediárias** ao pruner (por época para LSTM, por número de árvores para XGBoost). ([optuna.readthedocs.io][7])
4. **Classificação:** calibrar (Platt/Isotonic) e **otimizar limiar** no fold para a métrica-alvo; agregar por média ou mediana robusta. ([scikit-learn.org][20])
5. Retornar a métrica final do estudo (maximizar F1/PR-AUC ou minimizar MAE/RMSE/pinball).

### Samplers e pruners

* **Sampler (default):** **TPE** para espaços grandes/condicionais; GP-BO quando espaço é pequeno e suave. ([papers.neurips.cc][19], [Semantic Scholar PDFs][21], [papers.nips.cc][17])
* **Pruners:** **Median**, **SuccessiveHalving/ASHA** e **Hyperband** conforme recurso “épocas/árvores” disponível. ([optuna.readthedocs.io][7], [OptunaHub][2])

---

## 9) Execução em paralelo

* **Local/única máquina:** `n_jobs > 1` com storage SQLite ou RDBMS.
* **Cluster:** **Dask-Optuna** / `optuna-distributed` para trials concorrentes; manter versão de ambiente idêntica no cluster. ([jrbourbeau.github.io][9], [PyPI][15])

---

## 10) Pós-seleção

* **Re-treino** com hiperparâmetros vencedores no bloco treino+val.
* **Teste final** em bloco cronológico “virgem”.
* **Relato de calibração** (curva e Brier), matriz de confusão por limiar e curvas PR/ROC para classificação. ([scikit-learn.org][10])

---

## 11) Telemetria e reprodutibilidade

* Logar: seed, versão dos dados, commit, sampler/pruner, número de trials, tempo/trial, métricas por fold.
* Exportar: `study.trials_dataframe()` e artefatos. Optuna já fornece API para isso. ([optuna.readthedocs.io][1])

---

## 12) Critérios de aceitação (por modelo)

### LSTM

* **CA-1:** função-objetivo com TimeSeriesSplit/rolling reporta métricas por época e habilita **MedianPruner**/ASHA. ([optuna.readthedocs.io][7], [OptunaHub][2])
* **CA-2:** em classificação, pipeline inclui **calibração** e busca de **limiar** no fold. ([scikit-learn.org][20])
* **CA-3:** melhoria ≥ X% vs. baseline de grid aleatório no mesmo orçamento (definir X internamente).

### XGBoost

* **CA-4:** estudo com `tree_method=hist/gpu_hist`, early stopping e pruner habilitado. Docs de parâmetros referenciados na execução. ([xgboost.readthedocs.io][18])
* **CA-5:** relatório com PR-AUC, F1 e curva PR por fold; em regressão, MAE/RMSE; se quantílico, pinball loss.

---

## 13) Riscos e mitigação

* **Vazamento temporal:** usar Purged K-Fold + embargo quando labels dependem de janelas; testes unitários para checar sobreposição. ([Towards AI][5])
* **Overfitting ao val:** multi-fold e **teste final** isolado.
* **Custo computacional:** pruners (ASHA/Hyperband), paralelização Dask; limitar `n_trials` por horizonte. ([optuna.readthedocs.io][22], [docs.ray.io][11], [jrbourbeau.github.io][9])
* **Probabilidade mal calibrada:** aplicar **Platt** ou **isotonic** e medir ECE/Brier. ([scikit-learn.org][10])

---

## 14) Roadmap (datas referência, America/Sao\_Paulo)

* **S1 (até 2025-09-05):** implementação das `objective` para LSTM e XGB, TimeSeriesSplit e PurgedKFold+embargo; MedianPruner habilitado. ([optuna.readthedocs.io][7], [scikit-learn.org][4])
* **S2 (até 2025-09-19):** ASHA/Hyperband, logging completo de trials, calibração + thresholding; bloco de teste final. ([OptunaHub][2], [optuna.readthedocs.io][22])
* **S3 (até 2025-10-03):** Dask-Optuna/optuna-distributed em cluster, estudo multiativo por horizonte. ([jrbourbeau.github.io][9], [PyPI][15])
* **S4 (até 2025-10-17):** relatório comparativo (tempo, trials, métricas) e consolidação dos hiperparâmetros “estáveis”.

---

## 15) Entregáveis

* Código das `objective` (LSTM/XGB) com Optuna, pruners e validação temporal.
* Scripts de execução local e em cluster (Dask). ([docs.dask.org][8])
* Relatório de resultados: métricas por fold, curvas PR/ROC, calibração, tabela de hiperparâmetros por horizonte/ativo.
* Artefatos: `study.trials_dataframe()`, parâmetros vencedores, seeds e versões.

---

## 16) Apêndice — Referências essenciais

* **Optuna**: pruners (Median/Hyperband), tutoriais de algoritmos eficientes. ([optuna.readthedocs.io][7])
* **ASHA vs Hyperband**: recomendações de uso em schedulers (referência de Ray Tune, aplicável ao conceito). ([docs.ray.io][11])
* **Dask + Optuna**: docs oficiais e integração distribuída. ([docs.dask.org][8], [jrbourbeau.github.io][9])
* **BoTorch/GP-BO**: introdução, otimização de função de aquisição. ([botorch.org][3])
* **TimeSeriesSplit** (scikit-learn). ([scikit-learn.org][4])
* **Purged K-Fold/Embargo** e **CPCV** (implementações e guias). ([GitHub][6], [Towards AI][5])
* **Triple-Barrier** (conceito e libs abertas). ([mlfinpy.readthedocs.io][12])
* **XGBoost parameters & tuning notes**. ([xgboost.readthedocs.io][18])
* **Calibração de probabilidades** (scikit-learn). ([scikit-learn.org][10])
* **BO clássica**: TPE (Bergstra) e GP-EI (Snoek). ([papers.neurips.cc][19], [papers.nips.cc][17])

---

[1]: https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html?utm_source=chatgpt.com "Efficient Optimization Algorithms — Optuna 4.5.0 documentation"
[2]: https://hub.optuna.org/pruners/successive_halving/?utm_source=chatgpt.com "Successive Halving Pruner"
[3]: https://botorch.org/docs/introduction?utm_source=chatgpt.com "Introduction"
[4]: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html?utm_source=chatgpt.com "TimeSeriesSplit"
[5]: https://pub.towardsai.net/the-combinatorial-purged-cross-validation-method-363eb378a9c5?utm_source=chatgpt.com "The Combinatorial Purged Cross-Validation method"
[6]: https://github.com/hudson-and-thames/mlfinlab/blob/master/mlfinlab/cross_validation/combinatorial.py?utm_source=chatgpt.com "mlfinlab/mlfinlab/cross_validation/combinatorial.py at master"
[7]: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.MedianPruner.html?utm_source=chatgpt.com "optuna.pruners.MedianPruner - Read the Docs"
[8]: https://docs.dask.org/en/stable/ml.html?utm_source=chatgpt.com "Machine Learning - Optuna"
[9]: https://jrbourbeau.github.io/dask-optuna/?utm_source=chatgpt.com "Dask-Optuna documentation"
[10]: https://scikit-learn.org/stable/modules/calibration.html?utm_source=chatgpt.com "1.16. Probability calibration — scikit-learn 1.7.0 documentation"
[11]: https://docs.ray.io/en/latest/tune/api/schedulers.html?utm_source=chatgpt.com "Tune Trial Schedulers (tune.schedulers) — Ray 2.48.0"
[12]: https://mlfinpy.readthedocs.io/en/latest/Labelling.html?utm_source=chatgpt.com "Data Labelling - Mlfin.py"
[13]: https://github.com/hudson-and-thames/mlfinlab/blob/master/mlfinlab/labeling/labeling.py?utm_source=chatgpt.com "mlfinlab/mlfinlab/labeling/labeling.py at master"
[14]: https://optuna-integration.readthedocs.io/en/stable/reference/generated/optuna_integration.DaskStorage.html?utm_source=chatgpt.com "optuna_integration.DaskStorage - Optuna-Integration"
[15]: https://pypi.org/project/optuna-distributed/?utm_source=chatgpt.com "optuna-distributed"
[16]: https://proceedings.neurips.cc/paper/2012/file/05311655a15b75fab86956663e1819cd-Paper.pdf?utm_source=chatgpt.com "Practical Bayesian Optimization of Machine Learning ..."
[17]: https://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms?utm_source=chatgpt.com "Practical Bayesian Optimization of Machine Learning ..."
[18]: https://xgboost.readthedocs.io/en/stable/parameter.html?utm_source=chatgpt.com "XGBoost Parameters — xgboost 3.0.4 documentation"
[19]: https://papers.neurips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf?utm_source=chatgpt.com "Algorithms for Hyper-Parameter Optimization"
[20]: https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html?utm_source=chatgpt.com "CalibratedClassifierCV — scikit-learn 1.7.0 documentation"
[21]: https://pdfs.semanticscholar.org/d4f4/9717c9adb46137f49606ebbdf17e3598b5a5.pdf?utm_source=chatgpt.com "Hyperopt: A Python Library for Optimizing the ..."
[22]: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.HyperbandPruner.html?utm_source=chatgpt.com "optuna.pruners.HyperbandPruner - Read the Docs"

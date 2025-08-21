Belo plano. Otimização Bayesiana nos dois modelos é a vacina contra “tuning por fé”. Abaixo vai um blueprint enxuto, cruelmente pragmático, para você treinar LSTM e XGBoost com BO sem vazar um mísero milissegundo para o futuro.

## Arquitetura de BO (para ambos)

1. **Ferramental.** Use Optuna como espinha dorsal: TPE por padrão, com *pruners* para cortar tentativa ruim cedo (Median, SHA/ASHA). Se precisar de BO com GPs, BoTorch entra como motor de aquisição. ([Optuna][1], [optuna.readthedocs.io][2], [OptunaHub][3], [botorch.org][4])
2. **Objetivo correto.** A *objective* deve computar a métrica no **val de tempo**: walk-forward ou K-Fold **purgado com embargo**. Nada de K-Fold aleatório. Para rótulos tipo triple-barrier, purgue sobreposições de eventos. ([Cross Validated][5], [Kaggle][6], [quantresearch.org][7], [GitHub][8])
3. **Orçamento e *pruning*.** Rode 200–600 *trials* por horizonte/ativo como baseline; habilite `MedianPruner` ou `Hyperband/ASHA` para encurtar 40–70% do tempo mantendo qualidade. Use *report* de métricas parciais por época/árvore. ([optuna.readthedocs.io][9])
4. **Multiobjetivo quando fizer sentido.** Ex.: maximizar F1 e minimizar latência/overfit; Optuna tem *multi-objective* nativo e samplers como MOEA/D/PLMBO. ([optuna.readthedocs.io][10], [OptunaHub][11])
5. **Paralelização.** Distribua *trials* com Dask quando o dataset ficar parrudo; Optuna integra suave. ([Coiled][12])
6. **Escolha do BO.** TPE é robusto em espaços grandes/condicionais; GP-BO brilha em espaços menores e alvos suaves. Referências clássicas: Bergstra (random/TPE) e Snoek (GP-BO). ([JMLR][13], [Proceedings of Machine Learning Research][14], [NeurIPS Proceedings][15])

## Métricas-alvo por tarefa

* **Classificação (direção / triple-barrier):** otimize **PR-AUC** ou **F1**; dentro da *objective*, ajuste o **limiar** no conjunto de validação, senão você otimiza o modelo para um corte errado. ([Kaggle][6])
* **Regressão/quantis (retorno/vol):** **MAE/RMSE** e **pinball loss** para quantis; reporte **MASE** para comparabilidade entre séries. ([arXiv][16])

## Espaço de busca recomendado

### LSTM (PyTorch)

* **Arquitetura:** `hidden_size` 64–512; `num_layers` 1–3; `dropout` 0–0.5; `seq_len` 32–256 por horizonte.
* **Treino:** `optimizer ∈ {Adam, AdamW}`, `lr` 1e-5–3e-3 com *scheduler*; `batch_size` 32–512; `weight_decay` 0–1e-3; *gradient clip* 0.1–1.0.
* **Objetivo:** escolha conforme tarefa; relate métricas por janela do **walk-forward**. Use *pruning* por época via `trial.report(...)`. Tutoriais Optuna+PyTorch dão o esqueleto. ([optuna.readthedocs.io][2], [Medium][17], [GeeksforGeeks][18])

### XGBoost

* **Árvore:** `max_depth` 3–8; `min_child_weight` 1–10; `gamma` 0–5; `subsample` 0.6–0.95; `colsample_bytree` 0.5–0.95; `eta` 0.03–0.15; `n_estimators` 300–2000 com *early stopping*; `lambda` 0.5–5; `alpha` 0–2; `max_bin` 256–1024 se `hist/gpu_hist`.
* **Classe desbalanceada:** `scale_pos_weight ≈ Nneg/Npos` como *start*, mas faça **threshold tuning** no val.
* **Integração Optuna:** há exemplo oficial de *study* com XGBoost para copiar e adaptar. Param docs aqui. ([GitHub][19], [xgboost.readthedocs.io][20], [Analytics Vidhya][21])

## Pipeline da *objective* (esqueleto conceitual)

1. Receba `trial`; amostre hiperparâmetros.
2. **Crie splits temporais**: expanding/rolling ou purged+embargo, de acordo com seu rótulo.
3. Treine o modelo com *callbacks* para `trial.report(...)` e *pruner*.
4. No **val**: ajuste limiar (classificação) ou avalie pinball/MASE.
5. Retorne a métrica de interesse.
   Exemplos de integração e de *pruners* estão nos tutoriais oficiais do Optuna. ([GitHub][19], [optuna.readthedocs.io][2])

## Pré-treino “generalista” e *fine-tune*

* Rode um **estudo-mãe** multiativo por horizonte para aprender “políticas” de hiperparâmetros estáveis.
* **Reaproveite *trials* vencedores** como *priors* iniciais em estudos específicos por par/mercado (warm-start do espaço).
* Quando custo computacional apertar, use **ASHA/Hyperband** com recurso “épocas” (DL) ou “n\_estimators” (XGB). ([optuna.readthedocs.io][22], [OptunaHub][3])

## Anti-armadilhas essenciais

* **Sem vazamento temporal.** Purge/embargo para labels com janelas sobrepostas; nada de *shuffle*. ([Cross Validated][5], [Kaggle][6])
* **Calibração e custo.** Após escolher hiperparâmetros, calibre probabilidade (Platt/Isotonic) e avalie custo de transação em *forward test*. ([xgboost.readthedocs.io][20])
* **Rastreabilidade.** Salve `study.trials_dataframe()`, versões de dados e sementes. Você vai agradecer quando os deuses do mercado te perguntarem “como chegou nisso?”. ([optuna.readthedocs.io][23])

Se quiser poesia: BO é o arqueiro; seus *splits* no tempo são o vento. Sem medir o vento, a flecha vira oração. Com isso aqui, você para de rezar para a loss e começa a caçar.

[1]: https://optuna.org/?utm_source=chatgpt.com "Optuna - A hyperparameter optimization framework"
[2]: https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html?utm_source=chatgpt.com "Efficient Optimization Algorithms — Optuna 4.4.0 documentation"
[3]: https://hub.optuna.org/pruners/successive_halving/?utm_source=chatgpt.com "Successive Halving Pruner"
[4]: https://botorch.org/docs/introduction?utm_source=chatgpt.com "Introduction"
[5]: https://stats.stackexchange.com/questions/638157/are-purging-and-embargo-better-than-timeseriessplit?utm_source=chatgpt.com "Are Purging and Embargo better than TimeSeriesSplit?"
[6]: https://www.kaggle.com/competitions/optiver-trading-at-the-close/discussion/453128?utm_source=chatgpt.com "Enhancing Time Series Cross-Validation with Purging and ..."
[7]: https://www.quantresearch.org/Innovations.htm?utm_source=chatgpt.com "CPCV"
[8]: https://github.com/hudson-and-thames/mlfinlab/blob/master/mlfinlab/labeling/labeling.py?utm_source=chatgpt.com "mlfinlab/mlfinlab/labeling/labeling.py at master"
[9]: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.pruners.MedianPruner.html?utm_source=chatgpt.com "optuna.pruners.MedianPruner - Read the Docs"
[10]: https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/002_multi_objective.html?utm_source=chatgpt.com "Multi-objective Optimization with Optuna - Read the Docs"
[11]: https://hub.optuna.org/samplers/moead/?utm_source=chatgpt.com "MOEA/D sampler"
[12]: https://docs.coiled.io/examples/hpo.html?utm_source=chatgpt.com "Hyperparameter Optimization with XGBoost"
[13]: https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf?utm_source=chatgpt.com "Random Search for Hyper-Parameter Optimization"
[14]: https://proceedings.mlr.press/v28/bergstra13.pdf?utm_source=chatgpt.com "Making a Science of Model Search: Hyperparameter ..."
[15]: https://proceedings.neurips.cc/paper/2012/file/05311655a15b75fab86956663e1819cd-Paper.pdf?utm_source=chatgpt.com "Practical Bayesian Optimization of Machine Learning ..."
[16]: https://arxiv.org/pdf/2201.06433?utm_source=chatgpt.com "A Comparative study of Hyper-Parameter Optimization Tools"
[17]: https://medium.com/swlh/optuna-hyperparameter-optimization-in-pytorch-9ab5a5a39e77?utm_source=chatgpt.com "Optuna: Hyperparameter Optimization in PyTorch"
[18]: https://www.geeksforgeeks.org/deep-learning/hyperparameter-tuning-with-optuna-in-pytorch/?utm_source=chatgpt.com "Hyperparameter tuning with Optuna in PyTorch"
[19]: https://github.com/optuna/optuna-examples/blob/main/xgboost/xgboost_integration.py?utm_source=chatgpt.com "optuna-examples/xgboost/xgboost_integration.py at main"
[20]: https://xgboost.readthedocs.io/en/stable/parameter.html?utm_source=chatgpt.com "XGBoost Parameters — xgboost 3.0.4 documentation"
[21]: https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/?utm_source=chatgpt.com "XGBoost Parameters Tuning: A Complete Guide with ..."
[22]: https://optuna.readthedocs.io/en/stable/_modules/optuna/pruners/_hyperband.html?utm_source=chatgpt.com "Source code for optuna.pruners._hyperband"
[23]: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html?utm_source=chatgpt.com "optuna.study. - trials - Read the Docs"

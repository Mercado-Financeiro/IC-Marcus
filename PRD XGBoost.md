# PRD — Treinamento completo de XGBoost com dados de criptomoedas

## 1) Visão geral

**Propósito.** Definir, com rigor de pesquisa aplicada, o pipeline de treinamento e validação de modelos XGBoost para previsão em cripto (direção de retorno, magnitude/volatilidade e previsão quantílica), replicando práticas que se mostraram eficazes na literatura e em competições, com ênfase em evitar vazamento temporal e superavaliação de resultados.

**Benefícios esperados.** Sinais mais estáveis, interpretáveis (via SHAP), com métricas estatísticas e econômicas robustas após custos de transação, e operação contínua em mercados 24/7.

**Usuários-alvo.** Pesquisadores/engenheiros de ML quant, times de trading sistemático e ciência de dados.

---

## 2) Objetivos e resultados

* **O1. Classificação direcional** (janela de 5–60 min e 1–24 h): prever se o retorno futuro $r_{t+H}$ excede um limiar $\theta$ (positivo/negativo), com rótulo binário direcional e controle via threshold/abstenção.
* **O2. Regressão de retorno/volatilidade:** estimar $r_{t+H}$ ou $\sigma_{t+H}$ (volatilidade realizada), com previsão pontual (MAE/RMSE) e **previsões quantílicas** (p10, p50, p90) para envelopes de risco.
* **O3. Métricas econômicas**: Sharpe/Sortino após custos, *max drawdown*, *turnover*, *capacity*, significância (PSR/DSR).
* **O4. Interpretabilidade**: explicações nível feature e interação com TreeSHAP; validação de estabilidade ao longo do tempo.

**Resultados de aceite (resumo).** Ver §12.

---

## 3) Escopo / Não-escopo

**Escopo:** BTC, ETH e majors (mcap top 10), horizontes **H** ∈ {5m, 15m, 1h, 4h, 1d}. Dados: preço, volume, livro de ofertas (se disponível), derivativos (funding, OI, basis), *on-chain* (endereços ativos, taxas, NVT), sentimento (opcional).

**Não-escopo (fase 1):** CEX microestrutura proprietária de baixa latência, *options greeks* detalhados, execução algorítmica e *slippage* dinâmico por exchange; estratégias de portfólio cross-asset.

---

## 4) Métricas de sucesso

**Estatísticas (validação):**

* Classificação: F1, F\_β (β=0,5 e 2), **PR-AUC**, ROC-AUC, *Brier score* e **calibração** (ECE).
* Regressão: **MAE**, **RMSE**, **MASE**; para quantis: **Pinball loss** por quantil e *coverage* nominal.

**Econômicas (backtest com custos realistas):** Sharpe e Sortino anualizados, *max drawdown*, *hit rate*, *turnover* e *P\&L* líquido; **DSR** para significância; estabilidade de alfas por regime.

---

## 5) Dados

**Frequência e horizonte.** Série contínua em UTC, *candles* 1min/5min/15min/1h/4h/1d. Janelas de previsão **H** conforme §3.

**Fontes sugeridas:**

* **Spot**: OHLCV agregados multiexchange (ex.: CCXT/Kaiko/CoinMetrics).
* **Derivativos**: funding rate, *open interest*, *basis* (CME/perp).
* **Livro de ofertas**: profundidades 1–10 níveis para **OFI/queuing imbalance** (se disponível).
* **On-chain**: endereços ativos, NVT, *fees*, *hashrate* (para BTC), *supply growth*.
* **Sentimento** (opcional): *news/tweets* indexados.

**Qualidade.** Deduplicação cross-exchange, *outliers* por *winsorizing* em janelas móveis; *forward-fill* apenas para *micro-gaps* < 2× passo; alinhamento estrito de *timestamps*; normalização por *z-score* em janela móvel quando apropriado.

**Retenção temporal.** Treinamento inicial: últimos 3–5 anos (ou tudo que existir) para horizontes 1h–1d; para 5–15min, últimos 12–24 meses de granularidade alta.

---

## 6) Engenharia de *features* (o que replicar da literatura)

1. **Lags e janelas móveis**: $r_{t-k}$, $\sum r$, $\text{EMA/SMA}$, RSI, *stochastics*, *Bollinger*; retornos log e *z-scores* em janelas multi-escala.
2. **Volatilidade realizada**: RV, *bipower variation*, *realized quarticity*; *Garman–Klass*, *Parkinson* como *proxies*.
3. **Microestrutura (se LOB disponível)**: *order flow imbalance* (OFI), *queue imbalance*, *spread*, *depth imbalance*, taxa de cancelamentos.
4. **Derivativos**: **funding rate**, variações de **open interest**, *basis* spot-futuro; inclua dummies por *roll* e horário de ajuste.
5. **On-chain**: endereços ativos, *transactions per second*, *fees/KB*, NVT/NVR, *miner revenue*, variações de *supply* (emissões/queimas), distância ao halving.
6. **Calendário/regimes**: hora do dia, dia da semana, *halving windows*, volatilidade de macroeventos; *state features* por *vol regime* (baixa/média/alta) e *trend filter* (HT/ADX).
7. **Transformações**: **diferenciação fracionária** para estacionaridade preservando memória; *winsorize* 1–2%, *clipping* por MAD; *target encoding* para *categoricals*.

> **Padrões de implementação**: toda *feature* em **janelas causais**; sem *peeking*. Para cada *feature*, registre janela, atraso e função; guarde metadados em catálogo.

---

## 7) Definição de rótulos (labeling)

* **Direcional binário**: $\operatorname{sign}(r_{t\rightarrow t+H} - \theta) \in \{+1,-1\}$; mapear para {1,0}. $\theta$ pequeno > 0 para evitar *micro-noise*.
* **Abstenção por threshold**: operar apenas quando $p(y=1) > \tau_+$ ou $< \tau_-$; zona morta $[\tau_-,\tau_+]$ substitui o neutro.
* **Meta-labeling (opcional)**: classificar quando *entry rule* externa tem *edge*.
* **Regressão**: alvo $r_{t+H}$ ou $\sigma_{t+H}$; **quantis** τ ∈ {0.1, 0.5, 0.9}.

**Pesos de amostra:** decaimento temporal e ajuste por **unicidade** quando eventos se sobrepõem (sinais independentes recebem maior peso).

---

## 8) Particionamento e validação

* **Walk-forward purgado com embargo**: blocos sequenciais; **purge** para remover eventos sobrepostos entre treino/val; **embargo** de \~H para evitar contágio.
* **Esquemas**: *expanding window* para horizontes longos; *rolling window* para intradiário.
* **Validação final**: *forward test* fixo não tocado (últimos 3–6 meses intradiário; 6–12 meses diário).

---

## 9) Modelagem XGBoost

**Tarefas e objetivos**

* Classificação: `binary:logistic` com **`scale_pos_weight`**; *threshold moving* para F1/PR-AUC; calibração posterior (Platt/Isotonic).
* Regressão: `reg:squarederror` e `reg:absoluteerror`; para **quantis**, objetivo pinball via *custom objective*.

**Hiperparâmetros iniciais (intervalos sugeridos)**

* `tree_method`: `hist` ou `gpu_hist`; `device`: `cuda:0` quando disponível; `max_bin`: 256–1024.
* `n_estimators`: 300–2000 com **early stopping**.
* `learning_rate`: 0.03–0.15.
* `max_depth`: 3–8; `min_child_weight`: 1–10.
* `subsample`: 0.6–0.95; `colsample_bytree`: 0.5–0.95.
* `gamma`: 0–5; `lambda` (L2): 0.5–5; `alpha` (L1): 0–2.
* `max_delta_step`: até 1 para classes muito desbalanceadas.
* **Restrições opcionais**: `monotone_constraints` e `interaction_constraints` quando há conhecimento de domínio (ex.: risco ↑ não pode reduzir previsão de volatilidade).

**Desbalanceamento**

* `scale_pos_weight ≈ N_neg/N_pos` como *baseline*; ajuste fino pelo *threshold* ótimo ao métrico-alvo (F1, Fβ, custo-ponderado).

**Calibração**

* `CalibratedClassifierCV` com `method="sigmoid"` (Platt) e `isotonic`; escolher por menor ECE/Brier na validação.

---

## 10) Pipeline de treinamento

1. **Ingestão e *cleaning***: consolidar fontes; checagens de integridade; *resampling*; UTC; preenchimentos curtos.
2. **Feature store**: gerar lags/janelas; microestrutura/derivativos/on-chain; diferenciação fracionária; salvar artefatos com *hash* e datas.
3. **Labeling**: rótulo direcional por retorno futuro com limiar $\theta$; pesos de unicidade e decaimento temporal.
4. **Split**: walk-forward purgado + embargo.
5. **Busca de *hparams***: *Bayesian/Optuna* com orçamentos diferentes por horizonte; *early stopping* e *pruning* por *overfitting*.
6. **Treino**: XGBoost com *callbacks* de log; checagem de importância (gain, cover) para poda de *features* redundantes.
7. **Calibração e *thresholding***: calibrar probabilidades; varredura de *threshold* maximizando métrica-alvo em validação.
8. **Explainability**: TreeSHAP global e por regime; monitorar deriva de sinais relevantes.
9. **Backtest**: tradução de probabilidades/quantis em regras simples de posição; custos, *slippage*, latência; métricas econômicas; PSR/DSR.
10. **Empacotamento**: *model card*, *artifacts* (modelo, *hparams*, versões, SHAP), *schema* de dados, testes unitários.

---

## 11) Backtest e regras de decisão

**Mapeamentos possíveis**

* Classificação: entrar comprado se $p(y=1) > \tau_+$, vendido se $< \tau_-$; zona morta $[\tau_-, \tau_+]$; redimensionar posição por *confidence* ou *Kelly capped*.
* Quantis: comprar quando q0.9 − q0.1 > $\kappa$ e q0.5 > 0; *stop* por quantil inferior e *take-profit* por quantil superior.
* Volatilidade: dimensionar alavancagem/stop por previsão de $\sigma_{t+H}$.

**Custos e restrições**

* Custos de *taker/maker*, *gas*, *funding*; *slippage* por impacto linear em função do *ADV*.
* Restrições: *max leverage*, *max drawdown* alvo, limites de exposição por ativo e por regime.

**Métricas econômicas**

* Sharpe/Sortino líquidos; *Calmar*; *hit rate*; *turnover*; *capacity* (impacto vs. *ADV*); **DSR**.

---

## 12) Critérios de aceite

* **Sem vazamento**: auditoria de janelas e atrasos; *unit tests* que provem causalidade das *features*.
* **Validação**: PR-AUC ou F1 acima do *baseline* de heurísticas; MASE < 1 em regressão; cobertura quantílica dentro de ±2 p.p.
* **Econômico**: Sharpe líquido > 1,0 no *forward test* com custos conservadores; DSR > 0,8 por horizonte principal.
* **Estabilidade**: *feature* importances (SHAP) consistentes em 3+ blocos; *drift* controlado.

---

## 13) Riscos e mitigação

* **Mudança de regime**: janelas adaptativas; reamostragem de *features* e re-treino semanal/mensal por horizonte.
* **Dados ruidosos**: validação cruzada purgada; *robust features*; *winsorizing*; *bagging* das previsões ao longo de sub-janelas.
* **Sobreajuste**: regularização agressiva; *early stopping*; PSR/DSR; *dropout* via `subsample` e `colsample_bytree` < 1.
* **Imbalance extremo**: `scale_pos_weight`, *thresholding*, otimização de métrica adequada; *focal loss* opcional.

---

## 14) Entregáveis

* Código de *data prep*, *feature store*, *labeling*, *split purgado*, treino XGBoost, calibração, SHAP, backtest.
* *Model card* com versões, *hparams*, gráficos de calibração, PR/ROC, importância SHAP e estabilidade.
* Relatório de backtest com custos e PSR/DSR.

---

## 15) Especificações técnicas

**Linguagem/stack.** Python 3.10+, XGBoost 2.x/3.x (`hist`/`gpu_hist`), pandas/polars, NumPy, scikit-learn, Optuna, SHAP (TreeExplainer), mlfinlab (utilidades de validação), *pyfolio*/*empyrical* para métricas.

**Escalabilidade.** Para dados grandes, `QuantileDMatrix` e *external memory*; `gpu_hist` quando GPU disponível. *Pipelines* orquestrados (Airflow/Prefect) e *feature store* versionada.

**Monitoramento.** *Drift* de distribuição, ECE, *feature importances* por janela, *canary backtests*, alarmes de latência e *staleness* de dados.

---

## 16) Pseudocódigo (alto nível)

```python
# 1) Load & clean
df = load_sources(...).pipe(clean_align_to_utc)

# 2) Feature engineering
X = make_features(df, lags=[1,3,5,10,20], windows=[5,15,60,240,1440],
                  vol=['rv','bv','gk','parkinson'],
                  microstructure=['ofi','qi','spread','depth'],
                  derivatives=['funding','oi','basis'],
                  onchain=['active_addr','nvt','fees'],
                  transforms=['fracdiff','winsorize'])

# 3) Labeling (direcional com threshold)
ret_fut = compute_future_return(df, horizon=H)
y = (ret_fut > theta).astype(int)
w = time_decay_weights(df.index)  # opcional

# 4) Purged walk-forward splits
for trn, val in purged_cv_blocks(X.index, H, embargo=H):
    dtrn = DMatrix(X.iloc[trn], y.iloc[trn], weight=w.iloc[trn])
    dval = DMatrix(X.iloc[val], y.iloc[val])

    # 5) Train
    params = {...}
    bst = xgb.train(params, dtrn, num_boost_round=2000,
                    evals=[(dtrn,'trn'),(dval,'val')], early_stopping_rounds=100)

    # 6) Calibrate & threshold
    p_val = sigmoid(bst.predict(dval))
    p_cal = calibrate(p_val, y.iloc[val], method='platt')
    tau = argmax_f1_threshold(p_cal, y.iloc[val])

    # 7) Explain
    shap_vals = shap.TreeExplainer(bst).shap_values(X.iloc[val].sample(2000))

# 8) Backtest
signals = decision_rules(p_cal, tau)
metrics = backtest(signals, costs=costs_model, slippage=slippage_model)
```

---

## 17) Apêndice A — Busca de hiperparâmetros (grades sugeridas)

* **Curto prazo (5–15min)**: `max_depth ∈ {3,4,5}`, `min_child_weight ∈ {1,3,5}`, `subsample ∈ {0.6,0.8,0.9}`, `colsample_bytree ∈ {0.6,0.8,0.9}`, `gamma ∈ {0,1,3}`, `eta ∈ {0.05,0.1}`, `lambda ∈ {1,3}`, `alpha ∈ {0,0.5}`.
* **Médio/Longo (1h–1d)**: `max_depth ∈ {4,6,8}`, `min_child_weight ∈ {1,5,10}`, `subsample ∈ {0.6,0.75,0.9}`, `colsample_bytree ∈ {0.5,0.7,0.9}`, `gamma ∈ {0,2,5}`, `eta ∈ {0.03,0.07,0.1}`, `lambda ∈ {1,5}`, `alpha ∈ {0,1}`.

---

## 18) Apêndice B — Objetivo quantílico (pinball) para XGBoost

```python
def quantile_objective(q):
    def _obj(y_pred, dtrain):
        y = dtrain.get_label()
        e = y_pred - y
        grad = np.where(e < 0, q-1, q)
        hess = np.full_like(grad, 1e-6)
        return grad, hess
    return _obj
```

---

## 19) Apêndice C — Regras de controle de qualidade

* Testes unitários para evitar *leakage* de janelas.
* *Checklists* para causalidade das *features*.
* Reprodutibilidade: *seeds* fixos, *hash* de datasets e artefatos.
* *Model cards* com riscos e limitações.

---

## 20) Roadmap

* **S1 (2 semanas):** ingestão, *feature store* e labeling direcional; *purged CV*; *baseline* XGB classificação (H=1h).
* **S2 (2 semanas):** regressão e quantis; calibração; backtest e custos; relatório.
* **S3 (2 semanas):** LOB/derivativos/on-chain; SHAP e estabilidade; *hardening* e *model card*.

---

### Resumo do que “deu certo” para replicar

1. **Lags + janelas** para converter série em tabular e **walk-forward**; 2) **OFI/imbalance** e realizadas de alta frequência quando disponíveis; 3) **derivativos** (funding/OI/basis) e **on-chain** como *alpha* adicionais; 4) **purged CV + embargo**, **calibração** de probabilidades e *threshold tuning* orientado a F1/PR-AUC/EV; 5) **TreeSHAP** para seleção/estabilidade de *features* e controle de deriva; 6) **PSR/DSR** e custos no backtest para separar sorte de *edge* real.

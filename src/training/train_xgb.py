#!/usr/bin/env python3
"""Script de treinamento do XGBoost com Optuna.

Adições:
- Flags para reduzir trials (experimentos rápidos)
- Opções de PCA/SelectKBest para reduzir dimensionalidade
- Log de feature importance em MLflow (feature_importance.csv)
"""

import sys
import os
sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd
from datetime import datetime
import mlflow
import joblib
from pathlib import Path

import argparse
from src.models.xgb_optuna import XGBoostOptuna
from src.data.binance_loader import CryptoDataLoader
from src.features.engineering import FeatureEngineer
from src.features.selection import apply_pca, apply_select_kbest
from src.features.ga_selection import run_ga_feature_selection, GAConfig

def create_simple_labels(df: pd.DataFrame, horizon: int = 5, threshold: float = 0.002) -> pd.Series:
    """Cria labels simples baseados em retorno futuro."""
    # Calcular retorno futuro
    future_returns = df['close'].pct_change(horizon).shift(-horizon)
    
    # Criar labels binários
    labels = (future_returns > threshold).astype(int)
    
    # Remover NaN
    labels = labels.dropna()
    
    return labels

def main():
    print("=" * 80)
    print("TREINAMENTO XGBOOST COM OTUNA")
    print("=" * 80)
    print(f"Início: {datetime.now()}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10, help="Número de trials Optuna (padrão=10)")
    parser.add_argument("--pca", type=int, default=0, help="Aplicar PCA com n componentes (0=desliga)")
    parser.add_argument("--select-k", dest="select_k", type=int, default=0, help="Selecionar K melhores features (0=desliga)")
    parser.add_argument("--score-func", choices=["f_classif", "mutual_info"], default="f_classif")
    parser.add_argument("--cv", type=int, default=3, help="Folds de cross-validation (padrão=3)")
    # GA feature selection (opcional)
    parser.add_argument("--ga-select", action="store_true", help="Usar GA para seleção de features antes do Optuna")
    parser.add_argument("--ga-pop", type=int, default=20, help="Tamanho da população GA (padrão=20)")
    parser.add_argument("--ga-gen", type=int, default=5, help="Gerações GA (padrão=5)")
    parser.add_argument("--ga-mutate", type=float, default=0.1, help="Probabilidade de mutação (padrão=0.1)")
    parser.add_argument("--ga-cx", type=float, default=0.75, help="Probabilidade de crossover (padrão=0.75)")
    parser.add_argument("--refresh", action="store_true", help="Força refetch na Binance ignorando cache")
    parser.add_argument("--label-horizon", type=int, default=5, help="Horizonte do label em barras (padrão=5)")
    parser.add_argument("--label-th", type=float, default=0.002, help="Threshold de retorno para label (padrão=0.002)")
    parser.add_argument("--start", type=str, default='2024-01-01', help="Data inicial (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default='2024-08-23', help="Data final (YYYY-MM-DD)")
    args = parser.parse_args()

    # 1. Carregar dados
    print("\n1. Carregando dados...")
    loader = CryptoDataLoader(use_cache=not args.refresh)
    symbol = 'BTCUSDT'
    timeframe = '15m'
    df = loader.fetch_ohlcv(
        symbol=symbol,
        timeframe=timeframe,
        start_date=args.start,
        end_date=args.end
    )
    # Se refresh, opcionalmente salvar no cache para rodadas seguintes
    if args.refresh:
        try:
            cache_path = loader._get_cache_path(symbol, timeframe)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path)
            print(f"   ✓ Cache atualizado em: {cache_path}")
        except Exception as e:
            print(f"   ⚠ Falha ao atualizar cache: {e}")
    print(f"   ✓ Dados carregados: {len(df)} barras")
    
    # 2. Gerar features
    print("\n2. Gerando features...")
    engineer = FeatureEngineer(scaler_type="minmax")
    features_df = engineer.create_all_features(df)
    print(f"   ✓ Features criadas: {features_df.shape[1]} features")
    
    # 3. Criar labels simples
    print("\n3. Criando labels...")
    labels = create_simple_labels(df, horizon=args.label_horizon, threshold=args.label_th)
    print(f"   ✓ Labels criados: {len(labels)} amostras")
    
    # 4. Alinhar features e labels
    common_index = features_df.index.intersection(labels.index)
    X = features_df.loc[common_index]
    y = labels.loc[common_index]
    
    # Remover NaN e infinitos
    X = X.replace([np.inf, -np.inf], np.nan)
    mask = ~X.isna().any(axis=1) & ~y.isna()
    X = X[mask]  # Manter como DataFrame
    y = y[mask]  # Manter como Series
    
    print(f"   ✓ Dados finais: {len(y)} amostras ({y.mean():.2%} positivos)")
    
    # 5. Configurar MLflow
    print("\n4. Configurando MLflow...")
    mlflow.set_tracking_uri('artifacts/mlruns')
    mlflow.set_experiment('xgboost_optimization')
    
    # 6. Treinar modelo
    print("\n5. Iniciando otimização Bayesiana...")
    print(f"   - Trials: {args.trials}")
    print(f"   - CV Folds: {args.cv}")
    print("   - Metric: F1-Score")
    
    with mlflow.start_run(run_name=f'xgb_optuna_{datetime.now():%Y%m%d_%H%M%S}'):
        # Log parâmetros
        mlflow.log_param('symbol', 'BTCUSDT')
        mlflow.log_param('timeframe', '15m')
        mlflow.log_param('n_samples', len(y))
        mlflow.log_param('n_features', X.shape[1])
        mlflow.log_param('positive_rate', y.mean())

        # 4.1 Seleção de features: GA opcional (fora do CV, mas avaliado com CV no fitness sobre janela de treino)
        selection_info = None
        selection_method = 'none'
        selection_params = {}
        X_reduced = X
        if args.ga_select:
            print("   - GA Selection ativado (exploratória)")
            # Split temporal básico: usar 80% para GA e 20% holdout para posterior
            split_idx = int(len(X) * 0.8)
            X_tr_ga, y_tr_ga = X.iloc[:split_idx], y.iloc[:split_idx]
            ga_cfg = GAConfig(
                population_size=args.ga_pop,
                n_generations=args.ga_gen,
                mutation_prob=args.ga_mutate,
                crossover_prob=args.ga_cx,
                n_splits=args.cv,
                embargo=10,
                random_state=42,
            )
            res = run_ga_feature_selection(X_tr_ga, y_tr_ga, ga_cfg)
            selected_cols = res.selected_features
            X_reduced = X.loc[:, selected_cols]
            selection_info = res
            mlflow.log_param('ga_selected_features', len(selected_cols))
            out_sel = Path('selected_features.csv')
            pd.DataFrame({'feature': selected_cols}).to_csv(out_sel, index=False)
            mlflow.log_artifact(str(out_sel))
            print(f"   - GA selecionou {len(selected_cols)} features")
        if args.select_k and args.select_k > 0:
            selection_method = 'select_kbest'
            selection_params = {'k': args.select_k, 'score_func': args.score_func}
            mlflow.log_param('feature_selection', f'cv_select_kbest_{args.score_func}')
            mlflow.log_param('feature_selection_k', args.select_k)
            print(f"   - SelectKBest (per-fold): {args.select_k} features")
        elif args.pca and args.pca > 0 and not args.ga_select:
            selection_method = 'pca'
            selection_params = {'n_components': args.pca}
            mlflow.log_param('feature_selection', 'cv_pca')
            mlflow.log_param('feature_selection_k', args.pca)
            print(f"   - PCA (per-fold): {args.pca} componentes")

        # Quando GA não é utilizado, manter seleção por fold dentro do Optuna
        if not args.ga_select:
            X_reduced = X
        
        # Criar e treinar modelo
        model = XGBoostOptuna(
            n_trials=args.trials,
            cv_folds=args.cv,
            seed=42,
            use_mlflow=True,
            selection_method=selection_method,
            selection_params=selection_params,
        )
        
        print("\n" + "=" * 40)
        study, best_model = model.optimize(X_reduced, y)
        print("=" * 40)
        
        # 7. Obter métricas básicas do wrapper (sem get_best_metrics)
        print("\n6. Métricas finais (resumo):")
        summary = {
            'best_score': getattr(model, 'best_score', None),
            'threshold_f1': getattr(model, 'threshold_f1', None),
            'threshold_profit': getattr(model, 'threshold_profit', None),
            'threshold_ev': getattr(model, 'threshold_ev', None),
            'n_trials': getattr(study, 'n_trials', len(study.trials) if study else None),
        }
        for k, v in summary.items():
            if isinstance(v, (int, float)):
                print(f"   {k}: {v:.4f}")
                mlflow.log_metric(k, v)
        
        # 8. Salvar modelo
        print("\n7. Salvando modelo...")
        model_dir = Path('artifacts/models')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / f'xgboost_optuna_{datetime.now():%Y%m%d_%H%M%S}.pkl'
        joblib.dump(model, model_path)
        mlflow.log_artifact(str(model_path))

        print(f"   ✓ Modelo salvo em: {model_path}")
        
        # Log best params
        if hasattr(model, 'best_params_'):
            for k, v in model.best_params_.items():
                mlflow.log_param(f'best_{k}', v)

        # 9. Log de feature importance (se disponível)
        try:
            # Para o nosso wrapper, usamos o modelo interno após fit_final_model
            if hasattr(model, 'best_model') and model.best_model is not None:
                # Feature names may be unavailable due to per-fold transforms; log only raw importances
                feat_names = [f"f_{i}" for i in range(model.best_model.n_features_in_)]

                importances = getattr(model.best_model, 'feature_importances_', None)
                if importances is not None:
                    fi = pd.DataFrame({
                        'feature': feat_names,
                        'importance': importances
                    }).sort_values('importance', ascending=False)

                    # Normaliza para %
                    s = fi['importance'].sum() or 1.0
                    fi['importance_pct'] = fi['importance'] / s

                    out_path = Path('feature_importance.csv')
                    fi.to_csv(out_path, index=False)
                    mlflow.log_artifact(str(out_path))
                    print("   ✓ Feature importance logado em MLflow")
        except Exception as e:
            print(f"   ⚠ Falha ao logar feature importance: {e}")
    
    print("\n" + "=" * 80)
    print("TREINAMENTO CONCLUÍDO COM SUCESSO!")
    print(f"Fim: {datetime.now()}")
    print("=" * 80)

if __name__ == "__main__":
    main()

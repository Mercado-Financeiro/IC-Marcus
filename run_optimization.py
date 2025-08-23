#!/usr/bin/env python3
"""
Script principal para executar otimização completa do pipeline ML.

Uso:
    python run_optimization.py --config configs/xgb.yaml --symbol BTCUSDT
    python run_optimization.py --config configs/lstm.yaml --symbol ETHUSDT
    python run_optimization.py --model both --symbol BTCUSDT
    python run_optimization.py --quick  # Teste rápido com configs fast_mode
"""

import os
import sys
import warnings
import argparse
import time
import pickle
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Configurar ambiente
sys.path.append(str(Path(__file__).parent))
warnings.filterwarnings('ignore')

# Determinismo
os.environ['PYTHONHASHSEED'] = '0'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve, auc, roc_auc_score, brier_score_loss
import mlflow

# Importar módulos do projeto
from src.data.binance_loader import BinanceDataLoader
from src.data.splits import PurgedKFold
from src.features.engineering import FeatureEngineer
from src.models.xgb_optuna import XGBoostOptuna
from src.models.lstm_optuna import LSTMOptuna
from src.backtest.engine import BacktestEngine

# Seed global
SEED = 42
np.random.seed(SEED)


def load_config(config_path: str) -> Dict[str, Any]:
    """Carrega configuração YAML."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_all_configs() -> Dict[str, Any]:
    """Carrega todas as configurações."""
    configs = {}
    config_files = ['data.yaml', 'xgb.yaml', 'lstm.yaml', 'backtest.yaml', 'optuna.yaml']

    for config_file in config_files:
        config_path = f"configs/{config_file}"
        if Path(config_path).exists():
            configs[config_file.replace('.yaml', '')] = load_config(config_path)
        else:
            print(f"⚠️ Config não encontrado: {config_path}")

    return configs


class OptimizationPipeline:
    """Pipeline completo de otimização para modelos ML usando configurações YAML."""

    def __init__(self, configs: Dict[str, Any], quick_mode=False):
        """
        Args:
            configs: Dicionário com todas as configurações YAML
            quick_mode: Se True, usa configurações fast_mode dos YAMLs
        """
        self.configs = configs
        self.quick_mode = quick_mode
        self.loader = BinanceDataLoader()

        # MLflow
        data_config = configs.get('data', {})
        mlflow.set_tracking_uri("artifacts/mlruns")
        mlflow.set_experiment("crypto_ml_production")

        # Parâmetros padrão derivados das configs
        # Embargo: preferir de xgb.cv, senão de data.split.purged_kfold
        self.embargo = (
            configs.get('xgb', {}).get('cv', {}).get('embargo')
            or configs.get('split', {}).get('purged_kfold', {}).get('embargo')
            or 10
        )
        # Trials LSTM a partir de lstm.optuna
        self.lstm_trials = configs.get('lstm', {}).get('optuna', {}).get('n_trials', 50)

        # Aplicar fast_mode se necessário
        if quick_mode:
            self._apply_fast_mode()

    def _apply_fast_mode(self):
        """Aplica configurações fast_mode de todos os configs."""
        if 'xgb' in self.configs and 'fast_mode' in self.configs['xgb']:
            fast_config = self.configs['xgb']['fast_mode']
            if fast_config.get('enabled', False):
                self.configs['xgb']['optuna']['n_trials'] = fast_config.get('n_trials', 5)
                self.configs['xgb']['xgb']['n_estimators'] = fast_config.get('n_estimators', 50)

        if 'lstm' in self.configs and 'fast_mode' in self.configs['lstm']:
            fast_config = self.configs['lstm']['fast_mode']
            if fast_config.get('enabled', False):
                self.configs['lstm']['optuna']['n_trials'] = fast_config.get('n_trials', 3)
                self.configs['lstm']['training']['epochs'] = fast_config.get('epochs', 10)

    def prepare_data(self, symbol="BTCUSDT", timeframe="15m"):
        """Prepara dados com feature engineering completo usando configurações YAML."""

        data_config = self.configs.get('data', {})

        # Datas
        start_date = data_config.get('data', {}).get('start_date', '2020-01-01')
        end_date = data_config.get('data', {}).get('end_date', '2024-12-31')

        print(f"\n📊 Carregando dados {symbol} {timeframe}...")
        df = self.loader.fetch_ohlcv(symbol, timeframe, start_date, end_date)
        df = self.loader.validate_data(df)
        print(f"✅ {len(df)} barras carregadas")

        # Feature Engineering
        print("\n🔧 Criando features...")
        features_config = data_config.get('features', {})

        if self.quick_mode:
            # Features simplificadas para teste rápido
            df['returns'] = df['close'].pct_change()
            df['sma_20'] = df['close'].rolling(20).mean()
            df['rsi'] = self._calculate_rsi(df['close'])
            feature_cols = ['returns', 'sma_20', 'rsi']
        else:
            # Features completas usando configuração
            lookback_periods = features_config.get('lookback_periods', [5, 10, 20, 50, 100])
            feature_eng = FeatureEngineer(lookback_periods=lookback_periods)

            # Aplicar features conforme configuração
            df = feature_eng.create_price_features(df)
            df = feature_eng.create_technical_indicators(df)

            # Features avançadas se habilitadas
            if features_config.get('microstructure', {}).get('enabled', False):
                df = feature_eng.create_microstructure_features(df)

            feature_cols = [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume', 'label']]

        # Labeling direcional simples para criptomoedas (subir/descer)
        # Removido Triple Barrier - usando apenas direção do preço
        labels_config = data_config.get('labels', {})
        
        # Configuração de labeling direcional
        horizon_minutes = labels_config.get('horizon_minutes', 15)
        min_return_threshold = labels_config.get('min_return_threshold', 0.0)  # Threshold mínimo
        
        # Calcular retorno futuro baseado no horizonte
        if horizon_minutes == 15:  # 1 barra de 15min
            future_returns = df['returns'].shift(-1)
        elif horizon_minutes == 30:  # 2 barras
            future_returns = (df['close'].shift(-2) / df['close'] - 1)
        elif horizon_minutes == 60:  # 4 barras
            future_returns = (df['close'].shift(-4) / df['close'] - 1)
        elif horizon_minutes == 240:  # 16 barras (4h)
            future_returns = (df['close'].shift(-16) / df['close'] - 1)
        else:
            # Calcular dinamicamente baseado no timeframe
            bars_ahead = int(horizon_minutes / 15)  # Assumindo timeframe de 15min
            future_returns = (df['close'].shift(-bars_ahead) / df['close'] - 1)
        
        # Criar labels binários (subir=1, descer=0)
        df['label'] = (future_returns > min_return_threshold).astype(int)
        
        # Sample weights baseados na volatilidade (opcional)
        if labels_config.get('use_volatility_weights', False):
            # Usar ATR para ponderar amostras mais voláteis
            if 'atr_14' not in df.columns:
                df['atr_14'] = ta.volatility.AverageTrueRange(
                    df['high'], df['low'], df['close'], window=14
                ).average_true_range()
            
            # Normalizar ATR para sample weights
            atr_normalized = df['atr_14'] / df['atr_14'].mean()
            sample_weights = np.clip(atr_normalized, 0.5, 2.0)  # Limitar entre 0.5 e 2.0
        else:
            sample_weights = None

        # Análise da distribuição de labels
        label_distribution = df['label'].value_counts().to_dict()
        total_samples = len(df)
        up_count = label_distribution.get(1, 0)
        down_count = label_distribution.get(0, 0)
        
        print(f"\n📊 Labeling Direcional (Subir/Descer):")
        print(f"  • Horizonte: {horizon_minutes} minutos")
        print(f"  • Threshold mínimo: {min_return_threshold:.4f}")
        print(f"  • Subir (1): {up_count} ({up_count/total_samples*100:.1f}%)")
        print(f"  • Descer (0): {down_count} ({down_count/total_samples*100:.1f}%)")
        print(f"  • Total: {total_samples} amostras")

        # Limpar NaNs após criação de labels e filtragem
        df = df.dropna()
        print(f"✅ Features: {len(feature_cols)} | Binary Labels: {df['label'].value_counts().to_dict()}")

        # Preparar X, y
        X = df[feature_cols]
        y = df['label']

        # Alinhar sample_weights com X e y
        if sample_weights is not None:
            # Converter para Series com mesmo índice e alinhar após dropna/filtragem
            if not isinstance(sample_weights, pd.Series):
                sample_weights = pd.Series(sample_weights, index=df.index)
            sample_weights = sample_weights.loc[df.index]

        # Split temporal usando configuração
        split_config = data_config.get('split', {})
        test_size_pct = split_config.get('test_size', 0.2)
        test_size = int(len(X) * test_size_pct)

        X_train = X.iloc[:-test_size]
        X_test = X.iloc[-test_size:]
        y_train = y.iloc[:-test_size]
        y_test = y.iloc[-test_size:]

        if sample_weights is not None:
            # Handle both Series and numpy array
            if isinstance(sample_weights, pd.Series):
                weights_train = sample_weights.iloc[:-test_size]
            else:
                # Convert numpy array to Series with same index as X
                sample_weights = pd.Series(sample_weights, index=X.index)
                weights_train = sample_weights.iloc[:-test_size]
        else:
            weights_train = None

        print(f"\n✅ Train: {len(X_train)} | Test: {len(X_test)}")

        # Verificar não-vazamento
        assert X_train.index.max() < X_test.index.min(), "❌ Vazamento temporal!"
        print("🔒 Sem vazamento temporal verificado")

        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'weights_train': weights_train,
            'df': df,
            'feature_cols': feature_cols
        }

    def optimize_xgboost(self, data):
        """Otimização Bayesiana para XGBoost usando configurações YAML."""

        xgb_config = self.configs.get('xgb', {})
        optuna_config = xgb_config.get('optuna', {})

        n_trials = optuna_config.get('n_trials', 100)
        pruner_type = optuna_config.get('pruner', {}).get('type', 'hyperband')

        print(f"\n{'='*60}")
        print(f"🎯 XGBOOST - {n_trials} trials")
        print(f"{'='*60}")

        start_time = time.time()

        # Configurações CV
        cv_config = xgb_config.get('cv', {})
        cv_folds = cv_config.get('n_splits', 5)
        embargo = cv_config.get('embargo', 10)

        # Criar otimizador
        xgb_opt = XGBoostOptuna(
            n_trials=n_trials,
            cv_folds=cv_folds,
            embargo=embargo,
            pruner_type=pruner_type,
            use_mlflow=not self.quick_mode,
            seed=SEED
        )

        # Otimizar
        study, model = xgb_opt.optimize(
            data['X_train'],
            data['y_train'],
            sample_weights=data['weights_train']
        )

        elapsed = (time.time() - start_time) / 60
        print(f"\n✅ Otimização completa em {elapsed:.1f} minutos")
        print(f"📊 Melhor score: {study.best_value:.4f}")

        # Avaliar
        y_pred_proba = xgb_opt.predict_proba(data['X_test'])
        y_pred = xgb_opt.predict(data['X_test'])

        metrics = self._calculate_metrics(data['y_test'], y_pred, y_pred_proba)
        metrics['threshold_f1'] = xgb_opt.threshold_f1
        metrics['threshold_ev'] = xgb_opt.threshold_ev

        # Backtest com thresholds otimizados
        backtest_metrics = self._run_backtest(data['df'], data['X_test'], y_pred, y_pred_proba)

        # Salvar modelo
        if not self.quick_mode:
            model_path = "artifacts/models/xgboost_optimized.pkl"
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': xgb_opt.best_model,
                    'calibrator': xgb_opt.calibrator,
                    'params': xgb_opt.best_params,
                    'thresholds': {
                        'f1': xgb_opt.threshold_f1,
                        'ev': xgb_opt.threshold_ev
                    }
                }, f)
            print(f"💾 Modelo salvo: {model_path}")

        return {
            'ml_metrics': metrics,
            'trading_metrics': backtest_metrics,
            'best_params': xgb_opt.best_params
        }

    def optimize_lstm(self, data):
        """Otimização Bayesiana para LSTM."""

        print(f"\n{'='*60}")
        print(f"🧠 LSTM - {self.lstm_trials} trials")
        print(f"{'='*60}")

        start_time = time.time()

        # Preparar dados para LSTM (features normalizadas)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(data['X_train']),
            index=data['X_train'].index,
            columns=data['X_train'].columns
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(data['X_test']),
            index=data['X_test'].index,
            columns=data['X_test'].columns
        )

        # Criar otimizador
        lstm_opt = LSTMOptuna(
            n_trials=self.lstm_trials,
            cv_folds=3,  # Menos folds para LSTM (computacionalmente caro)
            embargo=self.embargo,
            pruner_type='hyperband' if not self.quick_mode else 'median',
            early_stopping_patience=10,
            seed=SEED
        )

        # Otimizar
        study = lstm_opt.optimize(X_train_scaled, data['y_train'])

        elapsed = (time.time() - start_time) / 60
        print(f"\n✅ Otimização completa em {elapsed:.1f} minutos")
        print(f"📊 Melhor score: {study.best_value:.4f}")

        # Treinar modelo final
        lstm_opt.fit_final_model(X_train_scaled, data['y_train'])

        # Avaliar
        y_pred_proba = lstm_opt.predict_proba(X_test_scaled)
        y_pred = lstm_opt.predict(X_test_scaled)

        metrics = self._calculate_metrics(data['y_test'], y_pred, y_pred_proba)
        metrics['threshold_f1'] = lstm_opt.threshold_f1
        metrics['threshold_ev'] = lstm_opt.threshold_ev

        # Backtest com thresholds otimizados
        backtest_metrics = self._run_backtest(data['df'], data['X_test'], y_pred, y_pred_proba)

        # Salvar modelo
        if not self.quick_mode:
            model_path = "artifacts/models/lstm_optimized.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': lstm_opt.best_model,
                    'calibrator': lstm_opt.calibrator,
                    'params': lstm_opt.best_params,
                    'scaler': scaler,
                    'thresholds': {
                        'f1': lstm_opt.threshold_f1,
                        'ev': lstm_opt.threshold_ev
                    }
                }, f)
            print(f"💾 Modelo salvo: {model_path}")

        return {
            'ml_metrics': metrics,
            'trading_metrics': backtest_metrics,
            'best_params': lstm_opt.best_params
        }

    def _calculate_metrics(self, y_true, y_pred, y_proba):
        """Calcula métricas de ML."""
        precision, recall, _ = precision_recall_curve(y_true, y_proba)

        metrics = {
            'f1_score': f1_score(y_true, y_pred),
            'pr_auc': auc(recall, precision),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'brier_score': brier_score_loss(y_true, y_proba)
        }

        print("\n📈 Métricas ML:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        return metrics

    def _run_backtest(self, df, X_test, y_pred, y_pred_proba=None):
        """Executa backtest com execução t+1 e thresholds otimizados."""

        print("\n💰 Executando backtest...")
        bt_df = df.loc[X_test.index]

        bt = BacktestEngine(
            initial_capital=100000,
            fee_bps=5,
            slippage_bps=5
        )

        # Se temos probabilidades, otimizar thresholds
        if y_pred_proba is not None:
            print("🎯 Otimizando thresholds para maximizar EV...")

            # Otimizar thresholds
            optimization_result = bt.optimize_thresholds_for_ev(
                bt_df,
                y_pred_proba[:, 1] if len(y_pred_proba.shape) > 1 else y_pred_proba,
                threshold_range=(0.25, 0.75),
                step=0.05,
                min_gap=0.2
            )

            optimal_thresholds = optimization_result['thresholds']
            print(f"  • Long Threshold: {optimal_thresholds['long']:.3f}")
            print(f"  • Short Threshold: {optimal_thresholds['short']:.3f}")

            # Gerar sinais com thresholds otimizados
            probas = y_pred_proba[:, 1] if len(y_pred_proba.shape) > 1 else y_pred_proba
            signals = bt.generate_signals_with_thresholds(
                probas,
                optimal_thresholds['long'],
                optimal_thresholds['short'],
                mode='double'
            )
            signals = pd.Series(signals, index=X_test.index)

            # Usar métricas da otimização
            metrics = optimization_result['metrics']
            print(f"  • Abstention Rate: {metrics.get('abstention_rate', 0):.1%}")
        else:
            # Usar sinais binários simples
            signals = pd.Series(y_pred * 2 - 1, index=X_test.index)  # Converter 0/1 para -1/1
            results = bt.run_backtest(bt_df, signals)
            metrics = bt.calculate_metrics(results)

        print("📊 Métricas Trading:")
        for k in ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'expected_value']:
            if k in metrics:
                print(f"  {k}: {metrics[k]:.4f}")

        return metrics

    def _calculate_rsi(self, prices, period=14):
        """Calcula RSI simples."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi


def main():
    """Função principal."""

    parser = argparse.ArgumentParser(description='Otimização de modelos ML para trading com configurações YAML')
    parser.add_argument('--config', type=str, default=None,
                       help='Caminho para arquivo de configuração específico (ex: configs/xgb.yaml)')
    parser.add_argument('--model', choices=['xgboost', 'lstm', 'both'],
                       default='xgboost', help='Modelo para otimizar')
    parser.add_argument('--quick', action='store_true',
                       help='Modo rápido usando fast_mode das configurações')
    parser.add_argument('--symbol', default='BTCUSDT',
                       help='Símbolo para treinar')
    parser.add_argument('--timeframe', default='15m',
                       help='Timeframe dos dados')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("🚀 PIPELINE DE OTIMIZAÇÃO ML COM CONFIGS YAML")
    print("="*60)
    print(f"Modelo: {args.model}")
    print(f"Modo: {'QUICK TEST' if args.quick else 'PRODUÇÃO'}")
    print(f"Symbol: {args.symbol} | Timeframe: {args.timeframe}")
    print(f"Config: {args.config or 'ALL CONFIGS'}")
    print(f"Início: {datetime.now()}")
    print("="*60)

    # Carregar configurações
    if args.config:
        # Carregar configuração específica
        if not Path(args.config).exists():
            print(f"❌ Arquivo de configuração não encontrado: {args.config}")
            return 1

        config_data = load_config(args.config)
        # Ainda precisamos das outras configs base
        all_configs = load_all_configs()

        # Override com config específico
        config_name = Path(args.config).stem
        all_configs[config_name] = config_data
        configs = all_configs
    else:
        # Carregar todas as configurações
        configs = load_all_configs()

    # Verificar se configs foram carregadas
    if not configs:
        print("❌ Nenhuma configuração encontrada! Certifique-se que configs/ existe")
        return 1

    print(f"✅ Configurações carregadas: {list(configs.keys())}")

    # Criar pipeline
    pipeline = OptimizationPipeline(configs, quick_mode=args.quick)

    # Preparar dados
    data = pipeline.prepare_data(args.symbol, args.timeframe)

    results = {}

    # Otimizar modelos
    try:
        if args.model in ['xgboost', 'both']:
            if 'xgb' not in configs:
                print("⚠️ Configuração XGBoost não encontrada, pulando...")
            else:
                results['xgboost'] = pipeline.optimize_xgboost(data)

        if args.model in ['lstm', 'both']:
            if 'lstm' not in configs:
                print("⚠️ Configuração LSTM não encontrada, pulando...")
            else:
                results['lstm'] = pipeline.optimize_lstm(data)

        # Resumo final
        print("\n" + "="*60)
        print("✅ OTIMIZAÇÃO COMPLETA!")
        print("="*60)

        for model_name, model_results in results.items():
            print(f"\n📊 {model_name.upper()}:")
            print(f"  F1 Score: {model_results['ml_metrics']['f1_score']:.4f}")
            print(f"  Sharpe Ratio: {model_results['trading_metrics'].get('sharpe_ratio', 0):.4f}")
            print(f"  Total Return: {model_results['trading_metrics'].get('total_return', 0):.2%}")

        # Verificar se atende aos targets das configurações
        for model_name, model_results in results.items():
            config_key = 'xgb' if model_name == 'xgboost' else 'lstm'
            targets = configs.get(config_key, {}).get('targets', {})

            ml_ok = (model_results['ml_metrics']['f1_score'] > targets.get('f1_score', 0.6) and
                    model_results['ml_metrics']['brier_score'] < targets.get('brier_score_max', 0.25))
            trade_ok = model_results['trading_metrics'].get('sharpe_ratio', 0) > 1.0

            if ml_ok and trade_ok:
                print(f"\n🏆 {model_name.upper()} está PRONTO PARA PRODUÇÃO!")
            else:
                print(f"\n⚠️ {model_name.upper()} precisa de mais otimização")
                print(f"   ML targets: F1>{targets.get('f1_score', 0.6)}, Brier<{targets.get('brier_score_max', 0.25)}")
                print(f"   Trading target: Sharpe>1.0")

        print(f"\nFim: {datetime.now()}")
        print("\n💡 Para ver resultados detalhados: mlflow ui")

    except KeyboardInterrupt:
        print("\n\n⚠️ Otimização interrompida pelo usuário")
    except Exception as e:
        print(f"\n❌ Erro: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

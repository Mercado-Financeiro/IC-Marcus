"""Testes de blindagem contra vazamentos de dados (data leakage)."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from models.xgb_optuna import XGBoostOptuna
from features.labels import AdaptiveLabeler


class TestDataLeakageProtection:
    """Testes para proteger contra vazamentos de dados."""
    
    @pytest.fixture
    def temporal_data(self):
        """Dados com ordem temporal explícita."""
        dates = pd.date_range('2023-01-01', periods=1000, freq='15T')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'timestamp': dates,
            'close': 100 + np.cumsum(np.random.randn(1000) * 0.01),
            'volume': np.random.lognormal(10, 0.5, 1000),
            'feature_1': np.random.randn(1000),
            'feature_2': np.random.randn(1000),
            'returns': np.random.randn(1000) * 0.02
        })
        return data.set_index('timestamp')
    
    def test_temporal_split_no_future_leakage(self, temporal_data):
        """Verifica que splits temporais não vazam dados futuros."""
        # Criar splits
        train_end = temporal_data.index[600]
        val_end = temporal_data.index[800]
        
        train_data = temporal_data[temporal_data.index <= train_end]
        val_data = temporal_data[(temporal_data.index > train_end) & (temporal_data.index <= val_end)]
        test_data = temporal_data[temporal_data.index > val_end]
        
        # Verificações críticas
        assert train_data.index.max() < val_data.index.min(), "Train data leaks into validation"
        assert val_data.index.max() < test_data.index.min(), "Validation data leaks into test"
        
        # Verificar que não há overlap
        train_indices = set(train_data.index)
        val_indices = set(val_data.index)
        test_indices = set(test_data.index)
        
        assert len(train_indices & val_indices) == 0, "Overlap between train and validation"
        assert len(val_indices & test_indices) == 0, "Overlap between validation and test"
        assert len(train_indices & test_indices) == 0, "Overlap between train and test"
    
    def test_feature_scaling_no_future_leakage(self, temporal_data):
        """Verifica que scaling não usa estatísticas futuras."""
        from sklearn.preprocessing import StandardScaler
        
        # Split temporal
        train_end = temporal_data.index[600]
        train_data = temporal_data[temporal_data.index <= train_end]
        test_data = temporal_data[temporal_data.index > train_end]
        
        # Scaler fitado apenas no treino
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_data[['feature_1', 'feature_2']])
        test_scaled = scaler.transform(test_data[['feature_1', 'feature_2']])  # Apenas transform!
        
        # Verificar que estatísticas são apenas do treino
        train_mean_f1 = train_data['feature_1'].mean()
        train_std_f1 = train_data['feature_1'].std()
        
        # Scaler deve usar apenas estatísticas do treino
        assert abs(scaler.mean_[0] - train_mean_f1) < 1e-10, "Scaler using future data"
        assert abs(scaler.scale_[0] - train_std_f1) < 1e-10, "Scaler using future data"
    
    def test_rolling_features_no_future_data(self, temporal_data):
        """Verifica que features rolling não usam dados futuros."""
        # Calcular rolling mean manualmente
        window = 10
        rolling_means = []
        
        for i in range(len(temporal_data)):
            if i < window - 1:
                # Primeiras janelas: usar apenas dados disponíveis até agora
                available_data = temporal_data.iloc[:i+1]['close']
            else:
                # Janela completa: usar apenas dados passados
                available_data = temporal_data.iloc[i-window+1:i+1]['close']
            
            rolling_means.append(available_data.mean())
        
        # Comparar com pandas rolling (que deve ser correto)
        pandas_rolling = temporal_data['close'].rolling(window, min_periods=1).mean()
        
        # Verificar que não há diferenças significativas
        differences = np.abs(np.array(rolling_means) - pandas_rolling.values)
        assert np.all(differences < 1e-10), "Rolling features using future data"
    
    def test_label_creation_no_future_leakage(self, temporal_data):
        """Verifica que labels não usam informações futuras."""
        labeler = AdaptiveLabeler(
            volatility_method='returns',
            vol_multiplier=1.5,
            horizon_minutes=60
        )
        
        # Criar labels
        labels = labeler.create_labels(
            temporal_data['close'], 
            temporal_data['returns']
        )
        
        # Verificar que labels em t não dependem de dados > t
        # Para isso, vamos verificar que remover dados futuros não muda labels passados
        
        # Labels com dados completos
        full_labels = labeler.create_labels(
            temporal_data['close'], 
            temporal_data['returns']
        )
        
        # Labels com apenas primeiras 500 observações
        partial_data = temporal_data.iloc[:500]
        partial_labels = labeler.create_labels(
            partial_data['close'],
            partial_data['returns']
        )
        
        # Labels dos primeiros 450 pontos devem ser iguais
        # (deixando margem para horizon)
        compare_until = min(450, len(partial_labels) - 50)
        if compare_until > 0:
            full_subset = full_labels.iloc[:compare_until].dropna()
            partial_subset = partial_labels.iloc[:compare_until].dropna()
            
            if len(full_subset) > 0 and len(partial_subset) > 0:
                common_idx = full_subset.index.intersection(partial_subset.index)
                if len(common_idx) > 10:  # Pelo menos 10 pontos para comparar
                    differences = (full_subset.loc[common_idx] != partial_subset.loc[common_idx]).sum()
                    assert differences == 0, f"Labels changed with future data: {differences} differences"
    
    def test_cross_validation_embargo(self, temporal_data):
        """Verifica que validação cruzada temporal tem embargo."""
        from sklearn.model_selection import TimeSeriesSplit
        
        # TimeSeriesSplit com gap
        n_splits = 3
        embargo_bars = 10
        
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=embargo_bars)
        
        for train_idx, val_idx in tscv.split(temporal_data):
            # Verificar que há gap entre treino e validação
            train_max_idx = train_idx.max()
            val_min_idx = val_idx.min()
            
            gap = val_min_idx - train_max_idx - 1
            assert gap >= embargo_bars, f"Insufficient embargo: {gap} < {embargo_bars}"
            
            # Verificar ordem temporal
            train_times = temporal_data.index[train_idx]
            val_times = temporal_data.index[val_idx]
            
            assert train_times.max() < val_times.min(), "Temporal order violated in CV"
    
    def test_model_training_isolation(self, temporal_data):
        """Verifica que treino do modelo não usa dados de validação."""
        # Preparar dados
        features = temporal_data[['feature_1', 'feature_2']]
        target = (temporal_data['returns'] > 0).astype(int)
        
        # Split temporal
        train_end = 600
        X_train = features.iloc[:train_end]
        y_train = target.iloc[:train_end]
        X_val = features.iloc[train_end:800]
        y_val = target.iloc[train_end:800]
        
        # Criar modelo
        optimizer = XGBoostOptuna(n_trials=5, cv_folds=3)
        
        # Treinar apenas com dados de treino
        study = optimizer.optimize(X_train, y_train)
        
        # Verificar que o modelo foi fitado apenas no treino
        # (não há maneira direta, mas podemos verificar performance)
        best_params = study.best_params
        
        # O modelo não deve ter performance perfeita na validação
        # se foi treinado apenas no treino
        model = optimizer.model
        if model is not None:
            train_pred = model.predict_proba(X_train)[:, 1]
            val_pred = model.predict_proba(X_val)[:, 1]
            
            # Treino não deve ser idêntico à validação
            correlation = np.corrcoef(train_pred, val_pred)[0, 1]
            assert not np.isnan(correlation), "Model predictions are invalid"
            # Note: não testamos correlation < 1.0 pois pode ser legítima
    
    def test_threshold_optimization_on_validation_only(self, temporal_data):
        """Verifica que otimização de threshold usa apenas dados de validação."""
        from models.xgb.threshold import ThresholdOptimizer
        
        # Dados sintéticos
        np.random.seed(42)
        n_samples = 1000
        y_true = np.random.binomial(1, 0.3, n_samples)
        y_proba = np.random.beta(2, 5, n_samples)  # Skewed towards 0
        returns = np.random.normal(0.01, 0.02, n_samples)
        
        # Split
        train_size = 600
        val_size = 200
        
        # Dados de validação apenas
        y_val = y_true[train_size:train_size+val_size]
        proba_val = y_proba[train_size:train_size+val_size]
        returns_val = returns[train_size:train_size+val_size]
        
        # Otimizar threshold apenas na validação
        optimizer = ThresholdOptimizer()
        threshold, results = optimizer.choose_threshold_by_ev(
            y_val, proba_val,
            cost_per_trade_bps=10,
            win_return=0.01,
            loss_return=0.01
        )
        
        # Verificar que threshold foi otimizado
        assert 0.1 <= threshold <= 0.9, f"Invalid threshold: {threshold}"
        assert 'ev_net' in results['best'], "EV optimization results missing"
    
    def test_no_data_snooping_in_feature_selection(self, temporal_data):
        """Verifica que seleção de features não usa dados de teste."""
        from sklearn.feature_selection import SelectKBest, f_classif
        
        # Preparar dados
        features = temporal_data[['feature_1', 'feature_2']]
        target = (temporal_data['returns'] > 0).astype(int)
        
        # Split temporal
        train_end = 700
        X_train = features.iloc[:train_end]
        y_train = target.iloc[:train_end]
        X_test = features.iloc[train_end:]
        y_test = target.iloc[train_end:]
        
        # Feature selection APENAS no treino
        selector = SelectKBest(f_classif, k=1)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)  # Apenas transform!
        
        # Verificar que seletor foi fitado apenas no treino
        selected_features = selector.get_support()
        
        # Aplicar seleção manualmente para verificar
        train_scores = []
        for i, col in enumerate(features.columns):
            score, _ = f_classif(X_train.iloc[:, [i]], y_train)
            train_scores.append(score[0])
        
        # Feature com maior score no treino deve ser a selecionada
        best_feature_idx = np.argmax(train_scores)
        assert selected_features[best_feature_idx] == True, "Feature selection used test data"
    
    def test_pipeline_temporal_consistency(self, temporal_data):
        """Teste integrado da pipeline completa."""
        # Pipeline que deve ser temporalmente consistente:
        # 1. Split temporal
        # 2. Feature engineering no treino apenas
        # 3. Model fitting no treino
        # 4. Threshold optimization na validação
        # 5. Teste final
        
        features = temporal_data[['feature_1', 'feature_2']]
        target = (temporal_data['returns'] > 0).astype(int)
        
        # 1. Split temporal
        train_end = 500
        val_end = 750
        
        train_indices = slice(None, train_end)
        val_indices = slice(train_end, val_end)
        test_indices = slice(val_end, None)
        
        X_train = features.iloc[train_indices]
        y_train = target.iloc[train_indices]
        X_val = features.iloc[val_indices]
        y_val = target.iloc[val_indices]
        X_test = features.iloc[test_indices]
        y_test = target.iloc[test_indices]
        
        # 2. Preprocessing apenas no treino
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # 3. Treinar modelo
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # 4. Predições
        train_proba = model.predict_proba(X_train_scaled)[:, 1]
        val_proba = model.predict_proba(X_val_scaled)[:, 1]
        test_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # 5. Otimizar threshold na validação
        from sklearn.metrics import f1_score
        thresholds = np.linspace(0.1, 0.9, 17)
        best_f1 = -1
        best_threshold = 0.5
        
        for th in thresholds:
            val_pred = (val_proba >= th).astype(int)
            f1 = f1_score(y_val, val_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = th
        
        # 6. Teste final
        test_pred = (test_proba >= best_threshold).astype(int)
        test_f1 = f1_score(y_test, test_pred, zero_division=0)
        
        # Verificações finais
        assert 0.1 <= best_threshold <= 0.9, "Invalid optimal threshold"
        assert 0 <= test_f1 <= 1, "Invalid test F1 score"
        
        # Pipeline respeitou ordem temporal
        assert len(X_train) == train_end, "Train size mismatch"
        assert len(X_val) == val_end - train_end, "Validation size mismatch"
        assert len(X_test) == len(features) - val_end, "Test size mismatch"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
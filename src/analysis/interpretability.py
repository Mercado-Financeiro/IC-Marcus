"""Sistema de interpretabilidade avançada para modelos de ML."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import shap
from sklearn.inspection import permutation_importance, partial_dependence
import structlog
from dataclasses import dataclass
import warnings
import json

log = structlog.get_logger()


@dataclass
class InterpretabilityConfig:
    """Configuração para análise de interpretabilidade."""
    
    n_samples_shap: int = 1000
    n_repeats_permutation: int = 10
    pdp_features: Optional[List[str]] = None
    pdp_grid_resolution: int = 20
    ice_n_samples: int = 100
    interaction_features: Optional[List[Tuple[str, str]]] = None
    temporal_stability_window: int = 30
    save_plots: bool = True
    plot_dir: str = "artifacts/interpretability"


class ModelInterpretability:
    """Análise de interpretabilidade para modelos de ML."""
    
    def __init__(self, config: Optional[InterpretabilityConfig] = None):
        """Inicializa o analisador de interpretabilidade.
        
        Args:
            config: Configuração para análise
        """
        self.config = config or InterpretabilityConfig()
        self.shap_values = None
        self.shap_explainer = None
        self.feature_importance_history = []
        
    def calculate_shap_values(self,
                            model: Any,
                            X: pd.DataFrame,
                            model_type: str = 'tree') -> np.ndarray:
        """Calcula SHAP values para o modelo.
        
        Args:
            model: Modelo treinado
            X: Features DataFrame
            model_type: Tipo do modelo ('tree', 'linear', 'kernel', 'deep')
            
        Returns:
            Array com SHAP values
        """
        log.info("calculating_shap_values",
                model_type=model_type,
                n_samples=len(X))
        
        # Limitar número de amostras se necessário
        if len(X) > self.config.n_samples_shap:
            X_sample = X.sample(n=self.config.n_samples_shap, random_state=42)
        else:
            X_sample = X
        
        # Criar explainer apropriado
        if model_type == 'tree':
            # Para XGBoost, LightGBM, etc.
            self.shap_explainer = shap.TreeExplainer(model)
            self.shap_values = self.shap_explainer.shap_values(X_sample)
            
        elif model_type == 'linear':
            # Para modelos lineares
            self.shap_explainer = shap.LinearExplainer(model, X_sample)
            self.shap_values = self.shap_explainer.shap_values(X_sample)
            
        elif model_type == 'kernel':
            # KernelExplainer para modelos black-box
            # Usar subset menor para background
            background = shap.sample(X, min(100, len(X)))
            self.shap_explainer = shap.KernelExplainer(model.predict, background)
            self.shap_values = self.shap_explainer.shap_values(X_sample)
            
        elif model_type == 'deep':
            # Para redes neurais
            background = X.iloc[:100]
            self.shap_explainer = shap.DeepExplainer(model, background)
            self.shap_values = self.shap_explainer.shap_values(X_sample.values)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Para modelos de classificação multi-classe
        if isinstance(self.shap_values, list):
            # Pegar valores para classe positiva (assumindo binário)
            self.shap_values = self.shap_values[1] if len(self.shap_values) == 2 else self.shap_values[0]
        
        log.info("shap_values_calculated", shape=self.shap_values.shape)
        
        return self.shap_values
    
    def get_feature_importance_shap(self, 
                                  X: pd.DataFrame,
                                  normalize: bool = True) -> pd.DataFrame:
        """Obtém importância das features baseada em SHAP.
        
        Args:
            X: Features DataFrame
            normalize: Se True, normaliza importâncias para somar 1
            
        Returns:
            DataFrame com importâncias
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Run calculate_shap_values first.")
        
        # Importância média absoluta
        importance = np.abs(self.shap_values).mean(axis=0)
        
        if normalize:
            importance = importance / importance.sum()
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': importance,
            'importance_std': np.abs(self.shap_values).std(axis=0)
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def calculate_permutation_importance(self,
                                        model: Any,
                                        X: pd.DataFrame,
                                        y: pd.Series,
                                        scoring: str = 'neg_mean_squared_error') -> pd.DataFrame:
        """Calcula importância por permutação.
        
        Args:
            model: Modelo treinado
            X: Features DataFrame
            y: Target Series
            scoring: Métrica para scoring
            
        Returns:
            DataFrame com importâncias
        """
        log.info("calculating_permutation_importance",
                n_features=X.shape[1],
                n_repeats=self.config.n_repeats_permutation)
        
        result = permutation_importance(
            model, X, y,
            n_repeats=self.config.n_repeats_permutation,
            random_state=42,
            scoring=scoring
        )
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': result.importances_mean,
            'importance_std': result.importances_std
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def calculate_pdp(self,
                     model: Any,
                     X: pd.DataFrame,
                     features: Optional[List[str]] = None) -> Dict:
        """Calcula Partial Dependence Plots.
        
        Args:
            model: Modelo treinado
            X: Features DataFrame
            features: Lista de features (None = todas)
            
        Returns:
            Dict com dados PDP
        """
        if features is None:
            features = self.config.pdp_features or list(X.columns)[:5]  # Top 5 por padrão
        
        log.info("calculating_pdp", features=features)
        
        pdp_results = {}
        
        for feature in features:
            if feature not in X.columns:
                continue
            
            feature_idx = list(X.columns).index(feature)
            
            # Calcular PDP
            pdp_result = partial_dependence(
                model, X, [feature_idx],
                grid_resolution=self.config.pdp_grid_resolution
            )
            
            pdp_results[feature] = {
                'values': pdp_result['values'][0].tolist(),
                'average': pdp_result['average'][0].tolist(),
                'grid': pdp_result['grid'][0].tolist()
            }
        
        return pdp_results
    
    def calculate_ice(self,
                     model: Any,
                     X: pd.DataFrame,
                     feature: str,
                     n_samples: Optional[int] = None) -> Dict:
        """Calcula Individual Conditional Expectation curves.
        
        Args:
            model: Modelo treinado
            X: Features DataFrame
            feature: Feature para análise
            n_samples: Número de amostras ICE
            
        Returns:
            Dict com curvas ICE
        """
        if n_samples is None:
            n_samples = min(self.config.ice_n_samples, len(X))
        
        log.info("calculating_ice", feature=feature, n_samples=n_samples)
        
        # Amostrar dados
        X_sample = X.sample(n=n_samples, random_state=42)
        
        # Grid de valores para a feature
        feature_values = np.linspace(
            X[feature].min(),
            X[feature].max(),
            self.config.pdp_grid_resolution
        )
        
        # Calcular predições para cada valor no grid
        ice_curves = []
        
        for idx, row in X_sample.iterrows():
            predictions = []
            
            for value in feature_values:
                # Criar cópia e modificar feature
                X_mod = row.copy()
                X_mod[feature] = value
                
                # Predizer
                pred = model.predict(X_mod.values.reshape(1, -1))[0]
                predictions.append(float(pred))
            
            ice_curves.append(predictions)
        
        return {
            'feature': feature,
            'grid': feature_values.tolist(),
            'curves': ice_curves,
            'n_samples': n_samples
        }
    
    def calculate_feature_interactions(self,
                                     model: Any,
                                     X: pd.DataFrame,
                                     feature_pairs: Optional[List[Tuple[str, str]]] = None) -> pd.DataFrame:
        """Calcula interações entre features usando SHAP.
        
        Args:
            model: Modelo treinado
            X: Features DataFrame
            feature_pairs: Pares de features para analisar
            
        Returns:
            DataFrame com scores de interação
        """
        if self.shap_values is None:
            self.calculate_shap_values(model, X)
        
        log.info("calculating_feature_interactions")
        
        # Se não especificado, usar top features
        if feature_pairs is None:
            importance_df = self.get_feature_importance_shap(X)
            top_features = importance_df.head(5)['feature'].tolist()
            
            from itertools import combinations
            feature_pairs = list(combinations(top_features, 2))
        
        interaction_scores = []
        
        for feat1, feat2 in feature_pairs:
            if feat1 not in X.columns or feat2 not in X.columns:
                continue
            
            idx1 = list(X.columns).index(feat1)
            idx2 = list(X.columns).index(feat2)
            
            # Calcular interação como correlação dos SHAP values
            interaction = np.corrcoef(
                self.shap_values[:, idx1],
                self.shap_values[:, idx2]
            )[0, 1]
            
            interaction_scores.append({
                'feature1': feat1,
                'feature2': feat2,
                'interaction_score': abs(interaction)
            })
        
        interaction_df = pd.DataFrame(interaction_scores)
        interaction_df = interaction_df.sort_values('interaction_score', ascending=False)
        
        return interaction_df
    
    def analyze_temporal_stability(self,
                                  model: Any,
                                  X: pd.DataFrame,
                                  timestamps: pd.Series,
                                  window_size: Optional[int] = None) -> pd.DataFrame:
        """Analisa estabilidade temporal das importâncias.
        
        Args:
            model: Modelo treinado
            X: Features DataFrame
            timestamps: Series com timestamps
            window_size: Tamanho da janela temporal
            
        Returns:
            DataFrame com análise de estabilidade
        """
        if window_size is None:
            window_size = self.config.temporal_stability_window
        
        log.info("analyzing_temporal_stability", window_size=window_size)
        
        # Dividir dados em janelas temporais
        unique_periods = pd.to_datetime(timestamps).dt.to_period('D').unique()
        
        if len(unique_periods) < 2:
            log.warning("Insufficient temporal variation for stability analysis")
            return pd.DataFrame()
        
        stability_results = []
        
        # Calcular importâncias para cada janela
        for i in range(0, len(unique_periods) - window_size + 1, window_size // 2):
            period_start = unique_periods[i]
            period_end = unique_periods[min(i + window_size, len(unique_periods) - 1)]
            
            # Filtrar dados da janela
            mask = (pd.to_datetime(timestamps).dt.to_period('D') >= period_start) & \
                   (pd.to_datetime(timestamps).dt.to_period('D') <= period_end)
            
            if mask.sum() < 10:  # Mínimo de amostras
                continue
            
            X_window = X[mask]
            
            # Calcular SHAP values para janela
            try:
                shap_values_window = self.calculate_shap_values(model, X_window)
                importance_window = np.abs(shap_values_window).mean(axis=0)
                
                # Normalizar
                importance_window = importance_window / importance_window.sum()
                
                # Armazenar
                for feat_idx, feat_name in enumerate(X.columns):
                    stability_results.append({
                        'feature': feat_name,
                        'period_start': str(period_start),
                        'period_end': str(period_end),
                        'importance': importance_window[feat_idx]
                    })
                
                # Adicionar ao histórico
                self.feature_importance_history.append({
                    'period': f"{period_start} to {period_end}",
                    'importances': dict(zip(X.columns, importance_window))
                })
                
            except Exception as e:
                log.warning(f"Error calculating importance for period {period_start}: {e}")
                continue
        
        if not stability_results:
            return pd.DataFrame()
        
        stability_df = pd.DataFrame(stability_results)
        
        # Calcular estatísticas de estabilidade
        stability_stats = stability_df.groupby('feature')['importance'].agg([
            'mean', 'std', 'min', 'max'
        ]).reset_index()
        
        stability_stats['cv'] = stability_stats['std'] / (stability_stats['mean'] + 1e-10)
        stability_stats['stability_score'] = 1 / (1 + stability_stats['cv'])
        
        return stability_stats.sort_values('stability_score', ascending=False)
    
    def create_interpretation_report(self,
                                    model: Any,
                                    X: pd.DataFrame,
                                    y: pd.Series,
                                    model_type: str = 'tree') -> Dict:
        """Cria relatório completo de interpretabilidade.
        
        Args:
            model: Modelo treinado
            X: Features DataFrame
            y: Target Series
            model_type: Tipo do modelo
            
        Returns:
            Dict com relatório completo
        """
        log.info("creating_interpretation_report")
        
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'model_type': model_type,
            'n_features': X.shape[1],
            'n_samples': X.shape[0]
        }
        
        # 1. SHAP values e importâncias
        try:
            self.calculate_shap_values(model, X, model_type)
            shap_importance = self.get_feature_importance_shap(X)
            report['shap_importance'] = shap_importance.to_dict('records')
            
            # SHAP summary stats
            report['shap_stats'] = {
                'mean_abs_shap': float(np.abs(self.shap_values).mean()),
                'std_shap': float(self.shap_values.std()),
                'max_abs_shap': float(np.abs(self.shap_values).max())
            }
        except Exception as e:
            log.warning(f"SHAP calculation failed: {e}")
            report['shap_importance'] = None
        
        # 2. Permutation importance
        try:
            perm_importance = self.calculate_permutation_importance(model, X, y)
            report['permutation_importance'] = perm_importance.to_dict('records')
        except Exception as e:
            log.warning(f"Permutation importance failed: {e}")
            report['permutation_importance'] = None
        
        # 3. PDP para top features
        try:
            top_features = shap_importance.head(5)['feature'].tolist() if 'shap_importance' in report else X.columns[:5]
            pdp_results = self.calculate_pdp(model, X, top_features)
            report['pdp'] = pdp_results
        except Exception as e:
            log.warning(f"PDP calculation failed: {e}")
            report['pdp'] = None
        
        # 4. Feature interactions
        try:
            interactions = self.calculate_feature_interactions(model, X)
            report['feature_interactions'] = interactions.head(10).to_dict('records')
        except Exception as e:
            log.warning(f"Interaction calculation failed: {e}")
            report['feature_interactions'] = None
        
        # 5. Model-specific interpretations
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            report['model_feature_importance'] = dict(zip(
                X.columns,
                model.feature_importances_.tolist()
            ))
        
        if hasattr(model, 'coef_'):
            # Linear models
            if len(model.coef_.shape) == 1:
                report['model_coefficients'] = dict(zip(
                    X.columns,
                    model.coef_.tolist()
                ))
            else:
                report['model_coefficients'] = dict(zip(
                    X.columns,
                    model.coef_[0].tolist()
                ))
        
        return report
    
    def plot_interpretability(self,
                            X: pd.DataFrame,
                            save_path: Optional[str] = None) -> None:
        """Cria visualizações de interpretabilidade.
        
        Args:
            X: Features DataFrame
            save_path: Caminho para salvar plots
        """
        try:
            import matplotlib.pyplot as plt
            
            if self.shap_values is None:
                log.warning("No SHAP values to plot")
                return
            
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            
            # 1. SHAP Summary Plot
            ax = axes[0, 0]
            shap.summary_plot(self.shap_values, X, plot_type="bar", show=False)
            plt.sca(ax)
            ax.set_title("SHAP Feature Importance")
            
            # 2. SHAP Beeswarm Plot
            ax = axes[0, 1]
            shap.summary_plot(self.shap_values, X, show=False)
            plt.sca(ax)
            ax.set_title("SHAP Beeswarm Plot")
            
            # 3. Top Feature Dependence
            importance_df = self.get_feature_importance_shap(X)
            top_feature = importance_df.iloc[0]['feature']
            
            ax = axes[0, 2]
            shap.dependence_plot(
                top_feature,
                self.shap_values,
                X,
                ax=ax,
                show=False
            )
            ax.set_title(f"SHAP Dependence: {top_feature}")
            
            # 4. Feature Importance Comparison
            if hasattr(self, 'feature_importance_history') and self.feature_importance_history:
                ax = axes[1, 0]
                
                # Plotar evolução temporal das top features
                top_5_features = importance_df.head(5)['feature'].tolist()
                
                for feature in top_5_features:
                    importances = []
                    periods = []
                    
                    for hist in self.feature_importance_history:
                        if feature in hist['importances']:
                            importances.append(hist['importances'][feature])
                            periods.append(hist['period'][:10])  # Primeiros 10 chars
                    
                    if importances:
                        ax.plot(periods, importances, marker='o', label=feature[:20])
                
                ax.set_xlabel("Period")
                ax.set_ylabel("Importance")
                ax.set_title("Feature Importance Over Time")
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                ax.grid(True, alpha=0.3)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            # 5. SHAP Waterfall for first prediction
            ax = axes[1, 1]
            shap.waterfall_plot(
                shap.Explanation(
                    values=self.shap_values[0],
                    base_values=self.shap_explainer.expected_value if self.shap_explainer else 0,
                    data=X.iloc[0],
                    feature_names=X.columns.tolist()
                ),
                max_display=10,
                show=False
            )
            ax.set_title("SHAP Waterfall (First Sample)")
            
            # 6. Force plot for first prediction
            ax = axes[1, 2]
            shap.force_plot(
                self.shap_explainer.expected_value if self.shap_explainer else 0,
                self.shap_values[0],
                X.iloc[0],
                show=False,
                matplotlib=True
            )
            ax.set_title("SHAP Force Plot (First Sample)")
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                log.info(f"Interpretability plots saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            log.warning(f"Plotting failed: {e}")
    
    def export_report(self, report: Dict, filepath: str) -> None:
        """Exporta relatório de interpretabilidade.
        
        Args:
            report: Relatório para exportar
            filepath: Caminho do arquivo
        """
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        log.info(f"Interpretability report exported to {filepath}")
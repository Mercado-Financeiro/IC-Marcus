"""
MLflow Model Registry com padrão Champion/Challenger.
Implementa versionamento semântico e promoção automática de modelos.
"""

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from typing import Optional, Dict, List, Any
import logging
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np


class ModelRegistry:
    """
    Gerencia o registro de modelos MLflow com padrão Champion/Challenger.
    
    Features:
    - Versionamento semântico automático
    - Promoção baseada em métricas
    - Rollback automático
    - Comparação de modelos
    - Tracking de performance
    """
    
    def __init__(self, tracking_uri: str = "artifacts/mlruns"):
        """
        Args:
            tracking_uri: URI do MLflow tracking server
        """
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        self.logger = logging.getLogger(__name__)
        
        # Métricas para comparação de modelos
        self.comparison_metrics = [
            "f1_score", "pr_auc", "roc_auc", "brier_score",
            "sharpe_ratio", "sortino_ratio", "max_drawdown", "total_return"
        ]
        
        # Thresholds para promoção
        self.promotion_thresholds = {
            "f1_score": 0.6,
            "pr_auc": 0.6,
            "brier_score": 0.25,  # Menor é melhor
            "sharpe_ratio": 1.0,
            "max_drawdown": -0.20  # Menor é melhor (menos negativo)
        }
    
    def register_model(
        self, 
        model_uri: str, 
        model_name: str, 
        tags: Dict[str, str] = None,
        description: str = None
    ) -> ModelVersion:
        """
        Registra um novo modelo no Model Registry.
        
        Args:
            model_uri: URI do modelo no MLflow
            model_name: Nome do modelo
            tags: Tags do modelo
            description: Descrição do modelo
            
        Returns:
            ModelVersion: Versão do modelo registrada
        """
        self.logger.info(f"Registrando modelo {model_name} com URI {model_uri}")
        
        # Registrar modelo
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
            tags=tags
        )
        
        # Adicionar descrição se fornecida
        if description:
            self.client.update_model_version(
                name=model_name,
                version=model_version.version,
                description=description
            )
        
        # Adicionar tags padrão
        default_tags = {
            "created_at": datetime.now().isoformat(),
            "stage": "None",
            "model_type": tags.get("model_type", "unknown") if tags else "unknown"
        }
        
        if tags:
            default_tags.update(tags)
        
        for key, value in default_tags.items():
            self.client.set_model_version_tag(
                name=model_name,
                version=model_version.version,
                key=key,
                value=str(value)
            )
        
        self.logger.info(f"Modelo registrado: {model_name} v{model_version.version}")
        return model_version
    
    def get_champion_model(self, model_name: str) -> Optional[ModelVersion]:
        """
        Retorna o modelo Champion atual (Production).
        
        Args:
            model_name: Nome do modelo
            
        Returns:
            ModelVersion ou None se não houver champion
        """
        try:
            versions = self.client.get_latest_versions(
                name=model_name, 
                stages=["Production"]
            )
            return versions[0] if versions else None
        except Exception as e:
            self.logger.warning(f"Erro ao buscar champion: {e}")
            return None
    
    def get_challenger_model(self, model_name: str) -> Optional[ModelVersion]:
        """
        Retorna o modelo Challenger atual (Staging).
        
        Args:
            model_name: Nome do modelo
            
        Returns:
            ModelVersion ou None se não houver challenger
        """
        try:
            versions = self.client.get_latest_versions(
                name=model_name, 
                stages=["Staging"]
            )
            return versions[0] if versions else None
        except Exception as e:
            self.logger.warning(f"Erro ao buscar challenger: {e}")
            return None
    
    def promote_to_staging(self, model_name: str, version: str) -> bool:
        """
        Promove um modelo para Staging (Challenger).
        
        Args:
            model_name: Nome do modelo
            version: Versão do modelo
            
        Returns:
            bool: True se promoção foi bem-sucedida
        """
        try:
            # Primeiro, arquivar challenger atual se existir
            current_challenger = self.get_challenger_model(model_name)
            if current_challenger:
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=current_challenger.version,
                    stage="Archived",
                    description="Arquivado: novo challenger promovido"
                )
            
            # Promover nova versão
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Staging",
                description="Promovido para Challenger (Staging)"
            )
            
            # Adicionar tag
            self.client.set_model_version_tag(
                name=model_name,
                version=version,
                key="promoted_to_staging_at",
                value=datetime.now().isoformat()
            )
            
            self.logger.info(f"Modelo {model_name} v{version} promovido para Staging")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao promover para staging: {e}")
            return False
    
    def promote_to_production(self, model_name: str, version: str, metrics: Dict[str, float] = None) -> bool:
        """
        Promove um modelo para Production (Champion).
        
        Args:
            model_name: Nome do modelo
            version: Versão do modelo
            metrics: Métricas do modelo para validação
            
        Returns:
            bool: True se promoção foi bem-sucedida
        """
        try:
            # Validar métricas se fornecidas
            if metrics and not self._validate_promotion_metrics(metrics):
                self.logger.warning("Métricas não atendem aos thresholds mínimos")
                return False
            
            # Arquivar champion atual se existir
            current_champion = self.get_champion_model(model_name)
            if current_champion:
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=current_champion.version,
                    stage="Archived",
                    description="Arquivado: novo champion promovido"
                )
            
            # Promover nova versão
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production",
                description="Promovido para Champion (Production)"
            )
            
            # Adicionar tags
            self.client.set_model_version_tag(
                name=model_name,
                version=version,
                key="promoted_to_production_at",
                value=datetime.now().isoformat()
            )
            
            if metrics:
                self.client.set_model_version_tag(
                    name=model_name,
                    version=version,
                    key="promotion_metrics",
                    value=json.dumps(metrics)
                )
            
            self.logger.info(f"Modelo {model_name} v{version} promovido para Production")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao promover para production: {e}")
            return False
    
    def compare_models(self, model_name: str, champion_run_id: str, challenger_run_id: str) -> Dict[str, Any]:
        """
        Compara métricas entre Champion e Challenger.
        
        Args:
            model_name: Nome do modelo
            champion_run_id: Run ID do champion
            challenger_run_id: Run ID do challenger
            
        Returns:
            Dict com comparação detalhada
        """
        try:
            # Buscar runs
            champion_run = self.client.get_run(champion_run_id)
            challenger_run = self.client.get_run(challenger_run_id)
            
            champion_metrics = champion_run.data.metrics
            challenger_metrics = challenger_run.data.metrics
            
            comparison = {
                "model_name": model_name,
                "champion": {
                    "run_id": champion_run_id,
                    "version": champion_run.data.tags.get("mlflow.parentRunId", "unknown"),
                    "metrics": champion_metrics
                },
                "challenger": {
                    "run_id": challenger_run_id,
                    "version": challenger_run.data.tags.get("mlflow.parentRunId", "unknown"),
                    "metrics": challenger_metrics
                },
                "comparison": {},
                "recommendation": None
            }
            
            # Comparar métricas
            better_count = 0
            total_metrics = 0
            
            for metric in self.comparison_metrics:
                if metric in champion_metrics and metric in challenger_metrics:
                    champion_value = champion_metrics[metric]
                    challenger_value = challenger_metrics[metric]
                    
                    # Determinar se challenger é melhor
                    is_better = self._is_metric_better(metric, challenger_value, champion_value)
                    
                    comparison["comparison"][metric] = {
                        "champion": champion_value,
                        "challenger": challenger_value,
                        "improvement": challenger_value - champion_value,
                        "improvement_pct": ((challenger_value - champion_value) / champion_value) * 100 if champion_value != 0 else 0,
                        "challenger_better": is_better
                    }
                    
                    if is_better:
                        better_count += 1
                    total_metrics += 1
            
            # Recomendação
            if total_metrics > 0:
                improvement_ratio = better_count / total_metrics
                
                if improvement_ratio >= 0.7:  # 70% das métricas melhoraram
                    comparison["recommendation"] = "PROMOTE"
                elif improvement_ratio >= 0.5:  # 50% das métricas melhoraram
                    comparison["recommendation"] = "EVALUATE"
                else:
                    comparison["recommendation"] = "REJECT"
            else:
                comparison["recommendation"] = "INSUFFICIENT_DATA"
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Erro ao comparar modelos: {e}")
            return {"error": str(e)}
    
    def auto_promote_challenger(self, model_name: str) -> Dict[str, Any]:
        """
        Avalia automaticamente se o Challenger deve ser promovido.
        
        Args:
            model_name: Nome do modelo
            
        Returns:
            Dict com resultado da avaliação e ações tomadas
        """
        result = {
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "action_taken": None,
            "reason": None,
            "comparison": None
        }
        
        try:
            champion = self.get_champion_model(model_name)
            challenger = self.get_challenger_model(model_name)
            
            if not challenger:
                result["action_taken"] = "NO_ACTION"
                result["reason"] = "No challenger available"
                return result
            
            if not champion:
                # Não há champion, promover challenger automaticamente
                if self.promote_to_production(model_name, challenger.version):
                    result["action_taken"] = "PROMOTED_TO_PRODUCTION"
                    result["reason"] = "No champion exists, promoted challenger"
                else:
                    result["action_taken"] = "PROMOTION_FAILED"
                    result["reason"] = "Failed to promote challenger to production"
                return result
            
            # Comparar modelos
            champion_run_id = champion.run_id
            challenger_run_id = challenger.run_id
            
            comparison = self.compare_models(model_name, champion_run_id, challenger_run_id)
            result["comparison"] = comparison
            
            if comparison.get("recommendation") == "PROMOTE":
                if self.promote_to_production(model_name, challenger.version):
                    result["action_taken"] = "PROMOTED_TO_PRODUCTION"
                    result["reason"] = "Challenger outperformed champion"
                else:
                    result["action_taken"] = "PROMOTION_FAILED"
                    result["reason"] = "Failed to promote challenger"
            elif comparison.get("recommendation") == "EVALUATE":
                result["action_taken"] = "NEEDS_MANUAL_REVIEW"
                result["reason"] = "Mixed results, manual evaluation required"
            else:
                result["action_taken"] = "NO_PROMOTION"
                result["reason"] = "Challenger did not outperform champion"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Erro na promoção automática: {e}")
            result["action_taken"] = "ERROR"
            result["reason"] = str(e)
            return result
    
    def rollback_to_previous_champion(self, model_name: str) -> bool:
        """
        Executa rollback para o champion anterior em caso de problemas.
        
        Args:
            model_name: Nome do modelo
            
        Returns:
            bool: True se rollback foi bem-sucedido
        """
        try:
            # Buscar versões arquivadas (champions anteriores)
            versions = self.client.search_model_versions(f"name='{model_name}'")
            
            # Filtrar champions anteriores arquivados
            archived_champions = [
                v for v in versions 
                if v.current_stage == "Archived" and 
                v.tags.get("promoted_to_production_at") is not None
            ]
            
            if not archived_champions:
                self.logger.warning("Nenhum champion anterior encontrado para rollback")
                return False
            
            # Pegar o mais recente (último champion)
            last_champion = max(archived_champions, 
                              key=lambda v: v.tags.get("promoted_to_production_at", ""))
            
            # Arquivar champion atual
            current_champion = self.get_champion_model(model_name)
            if current_champion:
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=current_champion.version,
                    stage="Archived",
                    description="Arquivado: rollback executado"
                )
            
            # Promover champion anterior
            self.client.transition_model_version_stage(
                name=model_name,
                version=last_champion.version,
                stage="Production",
                description="Rollback: promovido novamente após falha"
            )
            
            # Adicionar tag de rollback
            self.client.set_model_version_tag(
                name=model_name,
                version=last_champion.version,
                key="rolled_back_at",
                value=datetime.now().isoformat()
            )
            
            self.logger.info(f"Rollback executado: {model_name} v{last_champion.version}")
            return True
            
        except Exception as e:
            self.logger.error(f"Erro no rollback: {e}")
            return False
    
    def get_model_performance_history(self, model_name: str, limit: int = 10) -> List[Dict]:
        """
        Retorna histórico de performance dos modelos.
        
        Args:
            model_name: Nome do modelo
            limit: Número máximo de versões a retornar
            
        Returns:
            Lista com histórico de performance
        """
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            versions = sorted(versions, key=lambda v: v.creation_timestamp, reverse=True)
            
            history = []
            
            for version in versions[:limit]:
                if version.run_id:
                    try:
                        run = self.client.get_run(version.run_id)
                        metrics = run.data.metrics
                        
                        history.append({
                            "version": version.version,
                            "stage": version.current_stage,
                            "created_at": datetime.fromtimestamp(version.creation_timestamp / 1000).isoformat(),
                            "run_id": version.run_id,
                            "metrics": metrics,
                            "tags": version.tags
                        })
                    except Exception as e:
                        self.logger.warning(f"Erro ao buscar run {version.run_id}: {e}")
                        continue
            
            return history
            
        except Exception as e:
            self.logger.error(f"Erro ao buscar histórico: {e}")
            return []
    
    def _validate_promotion_metrics(self, metrics: Dict[str, float]) -> bool:
        """
        Valida se métricas atendem aos thresholds de promoção.
        
        Args:
            metrics: Métricas do modelo
            
        Returns:
            bool: True se métricas são válidas para promoção
        """
        for metric, threshold in self.promotion_thresholds.items():
            if metric not in metrics:
                continue
                
            value = metrics[metric]
            
            # Para métricas onde menor é melhor
            if metric in ["brier_score", "max_drawdown"]:
                if value > threshold:  # Valor é pior que threshold
                    return False
            else:
                # Para métricas onde maior é melhor
                if value < threshold:  # Valor é pior que threshold
                    return False
        
        return True
    
    def _is_metric_better(self, metric_name: str, new_value: float, current_value: float) -> bool:
        """
        Determina se um novo valor de métrica é melhor que o atual.
        
        Args:
            metric_name: Nome da métrica
            new_value: Novo valor
            current_value: Valor atual
            
        Returns:
            bool: True se novo valor é melhor
        """
        # Métricas onde menor é melhor
        if metric_name in ["brier_score", "max_drawdown"]:
            return new_value < current_value
        else:
            # Métricas onde maior é melhor
            return new_value > current_value


# Função de conveniência para uso global
_registry_instance = None

def get_model_registry(tracking_uri: str = "artifacts/mlruns") -> ModelRegistry:
    """
    Retorna instância global do ModelRegistry (singleton).
    
    Args:
        tracking_uri: URI do MLflow tracking server
        
    Returns:
        ModelRegistry: Instância do registry
    """
    global _registry_instance
    
    if _registry_instance is None:
        _registry_instance = ModelRegistry(tracking_uri)
    
    return _registry_instance
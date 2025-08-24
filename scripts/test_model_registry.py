#!/usr/bin/env python3
"""
Script para testar e demonstrar o MLflow Model Registry com Champion/Challenger.
"""

import sys
import os
from pathlib import Path

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent.parent))

import json
import pandas as pd
from datetime import datetime
from src.mlops.model_registry import ModelRegistry


def demo_model_registry():
    """Demonstra funcionalidades do Model Registry."""
    
    print("🏆 DEMONSTRAÇÃO DO MODEL REGISTRY")
    print("=" * 60)
    
    # Criar instância do registry
    registry = ModelRegistry()
    
    # Simular registro de modelos com diferentes métricas
    print("\n1. Simulando registro de modelos...")
    
    models_data = [
        {
            "name": "crypto_xgboost_BTCUSDT",
            "metrics": {
                "f1_score": 0.62,
                "pr_auc": 0.58, 
                "roc_auc": 0.59,
                "brier_score": 0.24,
                "sharpe_ratio": 1.2,
                "max_drawdown": -0.15,
                "total_return": 0.18
            },
            "tags": {
                "model_type": "xgboost",
                "symbol": "BTCUSDT",
                "timeframe": "15m"
            }
        },
        {
            "name": "crypto_lstm_BTCUSDT", 
            "metrics": {
                "f1_score": 0.58,
                "pr_auc": 0.61,
                "roc_auc": 0.60,
                "brier_score": 0.26,
                "sharpe_ratio": 0.9,
                "max_drawdown": -0.22,
                "total_return": 0.12
            },
            "tags": {
                "model_type": "lstm",
                "symbol": "BTCUSDT", 
                "timeframe": "15m"
            }
        }
    ]
    
    # Para demonstração, simular que temos modelos registrados
    print("✅ Modelos simulados:")
    for model in models_data:
        print(f"  - {model['name']}: F1={model['metrics']['f1_score']:.3f}, Sharpe={model['metrics']['sharpe_ratio']:.2f}")
    
    print("\n2. Testando comparação de modelos...")
    
    # Simular comparação
    champion_metrics = models_data[0]["metrics"]
    challenger_metrics = {
        "f1_score": 0.65,      # Melhor
        "pr_auc": 0.63,        # Melhor
        "roc_auc": 0.61,       # Melhor
        "brier_score": 0.22,   # Melhor (menor)
        "sharpe_ratio": 1.4,   # Melhor  
        "max_drawdown": -0.12, # Melhor (menos negativo)
        "total_return": 0.22   # Melhor
    }
    
    # Simular comparação manual
    comparison_result = simulate_comparison(champion_metrics, challenger_metrics)
    
    print("📊 Comparação Champion vs Challenger:")
    print(f"  Champion F1: {champion_metrics['f1_score']:.3f}")
    print(f"  Challenger F1: {challenger_metrics['f1_score']:.3f}")
    print(f"  Improvement: +{(challenger_metrics['f1_score'] - champion_metrics['f1_score']):.3f}")
    
    print(f"\n  Champion Sharpe: {champion_metrics['sharpe_ratio']:.2f}")
    print(f"  Challenger Sharpe: {challenger_metrics['sharpe_ratio']:.2f}") 
    print(f"  Improvement: +{(challenger_metrics['sharpe_ratio'] - champion_metrics['sharpe_ratio']):.2f}")
    
    print(f"\n📈 Recomendação: {comparison_result['recommendation']}")
    
    print("\n3. Testando thresholds de promoção...")
    
    # Testar validação de métricas
    valid_metrics = registry._validate_promotion_metrics(challenger_metrics)
    print(f"✅ Métricas atendem aos thresholds: {valid_metrics}")
    
    if valid_metrics:
        print("🎉 Challenger qualificado para promoção!")
    else:
        print("⚠️ Challenger não atende aos critérios mínimos")
    
    print("\n4. Demonstração de Model Registry workflow...")
    
    workflow_steps = [
        "1. Novo modelo treinado com Optuna",
        "2. Modelo registrado no MLflow Registry", 
        "3. Modelo promovido para Staging (Challenger)",
        "4. Comparação automática com Champion",
        "5. Promoção automática se métricas melhores",
        "6. Rollback disponível se problemas detectados"
    ]
    
    for step in workflow_steps:
        print(f"  {step}")
    
    print("\n5. Exemplo de configuração em produção...")
    
    production_config = {
        "auto_promotion_enabled": True,
        "promotion_schedule": "daily",
        "minimum_trials": 50,
        "metrics_improvement_threshold": 0.05,
        "rollback_on_alerts": True,
        "notification_channels": ["slack", "email"]
    }
    
    print("⚙️ Configuração de produção sugerida:")
    for key, value in production_config.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("✅ DEMONSTRAÇÃO COMPLETA!")
    print("🔧 Para usar em produção:")
    print("  1. Configure MLflow server remoto")
    print("  2. Implemente monitoramento de métricas")
    print("  3. Configure alertas automáticos") 
    print("  4. Teste workflows de rollback")
    
    return True


def simulate_comparison(champion_metrics, challenger_metrics):
    """Simula comparação entre champion e challenger."""
    
    better_count = 0
    total_metrics = 0
    
    comparison = {}
    
    for metric in champion_metrics.keys():
        if metric in challenger_metrics:
            champion_value = champion_metrics[metric]
            challenger_value = challenger_metrics[metric]
            
            # Determinar se challenger é melhor
            if metric in ["brier_score", "max_drawdown"]:
                is_better = challenger_value < champion_value
            else:
                is_better = challenger_value > champion_value
            
            comparison[metric] = {
                "champion": champion_value,
                "challenger": challenger_value, 
                "better": is_better
            }
            
            if is_better:
                better_count += 1
            total_metrics += 1
    
    improvement_ratio = better_count / total_metrics if total_metrics > 0 else 0
    
    if improvement_ratio >= 0.7:
        recommendation = "PROMOTE"
    elif improvement_ratio >= 0.5:
        recommendation = "EVALUATE"
    else:
        recommendation = "REJECT"
    
    return {
        "comparison": comparison,
        "improvement_ratio": improvement_ratio,
        "recommendation": recommendation
    }


def main():
    """Função principal."""
    
    try:
        demo_model_registry()
        return 0
    except Exception as e:
        print(f"❌ Erro na demonstração: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
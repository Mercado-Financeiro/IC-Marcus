#!/usr/bin/env python3
"""
Script to compare multiple trained models and generate comprehensive report.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
import argparse
from typing import Dict, List, Tuple

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# ML metrics
from sklearn.metrics import (
    f1_score, precision_recall_curve, auc, roc_auc_score,
    brier_score_loss, confusion_matrix, classification_report
)


class ModelComparator:
    """Compare multiple ML models and generate reports."""
    
    def __init__(self, output_dir: str = "artifacts/reports"):
        """Initialize model comparator.
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.results = {}
        
    def load_model(self, name: str, model_path: str):
        """Load a trained model.
        
        Args:
            name: Model identifier
            model_path: Path to model pickle file
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models[name] = {
            'model': model_data.get('model'),
            'calibrator': model_data.get('calibrator'),
            'thresholds': model_data.get('thresholds', {}),
            'params': model_data.get('params', {}),
            'path': model_path
        }
        
        print(f"✅ Loaded {name} from {model_path}")
    
    def evaluate_model(self, name: str, X_test, y_test, y_pred_proba=None):
        """Evaluate a single model.
        
        Args:
            name: Model identifier
            X_test: Test features
            y_test: True labels
            y_pred_proba: Optional pre-computed probabilities
        """
        if name not in self.models:
            raise ValueError(f"Model {name} not loaded")
        
        model_data = self.models[name]
        
        # Get predictions
        if y_pred_proba is None:
            if model_data['calibrator']:
                y_pred_proba = model_data['calibrator'].predict_proba(X_test)[:, 1]
            else:
                y_pred_proba = model_data['model'].predict_proba(X_test)[:, 1]
        
        # Apply threshold
        threshold = model_data['thresholds'].get('f1', 0.5)
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        
        metrics = {
            'f1_score': f1_score(y_test, y_pred),
            'pr_auc': auc(recall, precision),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'brier_score': brier_score_loss(y_test, y_pred_proba),
            'threshold': threshold
        }
        
        # Store results
        self.results[name] = {
            'metrics': metrics,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        return metrics
    
    def compare_metrics(self) -> pd.DataFrame:
        """Compare metrics across all models.
        
        Returns:
            DataFrame with comparison results
        """
        comparison = []
        
        for name, result in self.results.items():
            row = {'model': name}
            row.update(result['metrics'])
            comparison.append(row)
        
        df = pd.DataFrame(comparison)
        df = df.set_index('model')
        
        # Add ranking
        for col in df.columns:
            if col == 'brier_score':
                # Lower is better
                df[f'{col}_rank'] = df[col].rank(ascending=True)
            else:
                # Higher is better
                df[f'{col}_rank'] = df[col].rank(ascending=False)
        
        return df
    
    def create_visualizations(self) -> go.Figure:
        """Create comparison visualizations.
        
        Returns:
            Plotly figure with subplots
        """
        n_models = len(self.results)
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'ML Metrics Comparison',
                'Confusion Matrices',
                'Probability Distributions',
                'Model Rankings'
            ],
            specs=[
                [{'type': 'bar'}, {'type': 'heatmap'}],
                [{'type': 'histogram'}, {'type': 'scatter'}]
            ]
        )
        
        # 1. ML Metrics comparison
        metrics_df = self.compare_metrics()
        metric_cols = ['f1_score', 'pr_auc', 'roc_auc', 'brier_score']
        
        for i, model in enumerate(metrics_df.index):
            values = [metrics_df.loc[model, col] for col in metric_cols]
            fig.add_trace(
                go.Bar(name=model, x=metric_cols, y=values),
                row=1, col=1
            )
        
        # 2. Confusion matrices (simplified view)
        for i, (name, result) in enumerate(self.results.items()):
            cm = result['confusion_matrix']
            if i == 0:  # Show first model's confusion matrix
                fig.add_trace(
                    go.Heatmap(
                        z=cm,
                        text=cm,
                        texttemplate="%{text}",
                        colorscale='Blues',
                        showscale=True
                    ),
                    row=1, col=2
                )
        
        # 3. Probability distributions
        for name, result in self.results.items():
            fig.add_trace(
                go.Histogram(
                    x=result['y_pred_proba'],
                    name=name,
                    opacity=0.5,
                    nbinsx=50
                ),
                row=2, col=1
            )
        
        # 4. Model rankings scatter
        rankings = []
        for col in metric_cols:
            if f'{col}_rank' in metrics_df.columns:
                for model in metrics_df.index:
                    rankings.append({
                        'model': model,
                        'metric': col,
                        'rank': metrics_df.loc[model, f'{col}_rank']
                    })
        
        rankings_df = pd.DataFrame(rankings)
        for model in rankings_df['model'].unique():
            model_ranks = rankings_df[rankings_df['model'] == model]
            fig.add_trace(
                go.Scatter(
                    x=model_ranks['metric'],
                    y=model_ranks['rank'],
                    mode='lines+markers',
                    name=model,
                    line=dict(width=2),
                    marker=dict(size=10)
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Model Comparison Dashboard",
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Metric", row=1, col=1)
        fig.update_xaxes(title_text="Predicted/Actual", row=1, col=2)
        fig.update_xaxes(title_text="Probability", row=2, col=1)
        fig.update_xaxes(title_text="Metric", row=2, col=2)
        
        fig.update_yaxes(title_text="Score", row=1, col=1)
        fig.update_yaxes(title_text="Predicted/Actual", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_yaxes(title_text="Rank (1=best)", row=2, col=2)
        
        return fig
    
    def generate_report(self, format: str = 'html') -> str:
        """Generate comprehensive comparison report.
        
        Args:
            format: Output format ('html', 'pdf', 'markdown')
            
        Returns:
            Path to generated report
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'html':
            # Create HTML report
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Model Comparison Report - {timestamp}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    h1 {{ color: #333; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                    th {{ background-color: #4CAF50; color: white; }}
                    .winner {{ background-color: #d4edda; font-weight: bold; }}
                    .metric-card {{ 
                        border: 1px solid #ddd; 
                        padding: 20px; 
                        margin: 20px 0;
                        border-radius: 5px;
                    }}
                </style>
            </head>
            <body>
                <h1>🚀 Model Comparison Report</h1>
                <p>Generated: {timestamp}</p>
                
                <div class="metric-card">
                    <h2>📊 Summary</h2>
                    <p>Models compared: {', '.join(self.models.keys())}</p>
                </div>
            """
            
            # Add metrics table
            metrics_df = self.compare_metrics()
            html_content += """
                <div class="metric-card">
                    <h2>📈 Metrics Comparison</h2>
                    <table>
            """
            
            # Table header
            html_content += "<tr><th>Model</th>"
            for col in ['f1_score', 'pr_auc', 'roc_auc', 'brier_score']:
                html_content += f"<th>{col}</th>"
            html_content += "</tr>"
            
            # Table rows
            for model in metrics_df.index:
                html_content += f"<tr><td><strong>{model}</strong></td>"
                for col in ['f1_score', 'pr_auc', 'roc_auc', 'brier_score']:
                    value = metrics_df.loc[model, col]
                    rank = metrics_df.loc[model, f'{col}_rank']
                    css_class = 'winner' if rank == 1 else ''
                    html_content += f'<td class="{css_class}">{value:.4f}</td>'
                html_content += "</tr>"
            
            html_content += """
                    </table>
                </div>
            """
            
            # Add winner section
            overall_ranks = metrics_df[[c for c in metrics_df.columns if '_rank' in c]].mean(axis=1)
            best_model = overall_ranks.idxmin()
            
            html_content += f"""
                <div class="metric-card">
                    <h2>🏆 Overall Winner</h2>
                    <p style="font-size: 24px; color: #4CAF50;">
                        <strong>{best_model}</strong>
                    </p>
                    <p>Average rank: {overall_ranks[best_model]:.2f}</p>
                </div>
            """
            
            # Add visualizations
            fig = self.create_visualizations()
            plot_html = pio.to_html(fig, include_plotlyjs='cdn', div_id="plot")
            
            html_content += f"""
                <div class="metric-card">
                    <h2>📊 Visualizations</h2>
                    {plot_html}
                </div>
            """
            
            # Close HTML
            html_content += """
            </body>
            </html>
            """
            
            # Save report
            report_path = self.output_dir / f"comparison_report_{timestamp}.html"
            with open(report_path, 'w') as f:
                f.write(html_content)
            
            print(f"✅ Report saved to: {report_path}")
            return str(report_path)
        
        elif format == 'markdown':
            # Create Markdown report
            md_content = f"""# Model Comparison Report

Generated: {timestamp}

## 📊 Summary

Models compared: {', '.join(self.models.keys())}

## 📈 Metrics Comparison

"""
            metrics_df = self.compare_metrics()
            md_content += metrics_df.to_markdown()
            
            # Winner
            overall_ranks = metrics_df[[c for c in metrics_df.columns if '_rank' in c]].mean(axis=1)
            best_model = overall_ranks.idxmin()
            
            md_content += f"""

## 🏆 Overall Winner

**{best_model}** (Average rank: {overall_ranks[best_model]:.2f})

## 📊 Detailed Results

"""
            for name, result in self.results.items():
                md_content += f"""
### {name}

- F1 Score: {result['metrics']['f1_score']:.4f}
- PR-AUC: {result['metrics']['pr_auc']:.4f}
- ROC-AUC: {result['metrics']['roc_auc']:.4f}
- Brier Score: {result['metrics']['brier_score']:.4f}
- Threshold: {result['metrics']['threshold']:.3f}

"""
            
            report_path = self.output_dir / f"comparison_report_{timestamp}.md"
            with open(report_path, 'w') as f:
                f.write(md_content)
            
            print(f"✅ Report saved to: {report_path}")
            return str(report_path)
        
        else:
            raise ValueError(f"Format {format} not supported")
    
    def suggest_ensemble(self) -> Dict:
        """Suggest ensemble strategy based on model performances.
        
        Returns:
            Ensemble configuration
        """
        metrics_df = self.compare_metrics()
        
        # Calculate diversity (correlation between predictions)
        predictions = {}
        for name, result in self.results.items():
            predictions[name] = result['y_pred_proba']
        
        pred_df = pd.DataFrame(predictions)
        correlation = pred_df.corr()
        
        # Suggest weights based on performance
        weights = {}
        for model in metrics_df.index:
            # Weight based on F1 and PR-AUC
            score = (metrics_df.loc[model, 'f1_score'] + 
                    metrics_df.loc[model, 'pr_auc']) / 2
            weights[model] = score
        
        # Normalize weights
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        
        suggestion = {
            'method': 'weighted_average',
            'weights': weights,
            'diversity_score': 1 - correlation.values.mean(),
            'recommendation': ''
        }
        
        if suggestion['diversity_score'] > 0.3:
            suggestion['recommendation'] = "High diversity - ensemble recommended"
        else:
            suggestion['recommendation'] = "Low diversity - single best model may be better"
        
        return suggestion


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description='Compare ML Models')
    parser.add_argument('--models', nargs='+', required=True,
                       help='Model files to compare (name:path pairs)')
    parser.add_argument('--test-data', required=True,
                       help='Path to test data pickle')
    parser.add_argument('--format', default='html',
                       choices=['html', 'markdown'],
                       help='Report format')
    parser.add_argument('--output', default='artifacts/reports',
                       help='Output directory')
    
    args = parser.parse_args()
    
    # Create comparator
    comparator = ModelComparator(args.output)
    
    # Load models
    for model_spec in args.models:
        if ':' in model_spec:
            name, path = model_spec.split(':')
        else:
            name = Path(model_spec).stem
            path = model_spec
        comparator.load_model(name, path)
    
    # Load test data
    with open(args.test_data, 'rb') as f:
        test_data = pickle.load(f)
    
    X_test = test_data['X_test']
    y_test = test_data['y_test']
    
    # Evaluate all models
    print("\n📊 Evaluating models...")
    for name in comparator.models.keys():
        metrics = comparator.evaluate_model(name, X_test, y_test)
        print(f"\n{name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
    
    # Generate comparison
    print("\n📈 Comparison Results:")
    comparison_df = comparator.compare_metrics()
    print(comparison_df)
    
    # Suggest ensemble
    print("\n🤝 Ensemble Suggestion:")
    ensemble = comparator.suggest_ensemble()
    print(f"  Method: {ensemble['method']}")
    print(f"  Diversity: {ensemble['diversity_score']:.3f}")
    print(f"  Weights: {ensemble['weights']}")
    print(f"  💡 {ensemble['recommendation']}")
    
    # Generate report
    report_path = comparator.generate_report(args.format)
    print(f"\n✅ Comparison complete! Report: {report_path}")


if __name__ == "__main__":
    main()
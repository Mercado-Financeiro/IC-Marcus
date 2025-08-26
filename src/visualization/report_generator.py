"""
Comprehensive model evaluation report generator.
Generates all visualizations and creates an HTML report.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import visualization modules
from src.visualization.model_plots import (
    plot_pr_curve_with_baseline,
    plot_pr_auc_distribution,
    plot_roc_curve,
    plot_reliability_diagram,
    plot_brier_score_comparison,
    plot_calibrator_comparison,
    plot_ev_curve,
    plot_confusion_matrix_heatmap,
    plot_learning_curves,
    plot_mc_dropout_uncertainty
)

from src.visualization.temporal_plots import (
    plot_split_timeline,
    plot_walkforward_metrics,
    plot_temporal_stability
)

from src.visualization.optuna_plots import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_contour,
    generate_pruning_report
)

from src.visualization.backtest_plots import (
    plot_equity_curve_with_bands,
    plot_drawdown_curve,
    plot_returns_distribution,
    plot_qq_plot,
    plot_sharpe_comparison
)

# Import model utilities
from src.models.metrics.pr_auc import calculate_pr_auc
from src.models.metrics.quality_gates import QualityGates
from src.models.calibration.beta import BetaCalibration
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression


class ModelReportGenerator:
    """Generate comprehensive model evaluation report with all visualizations."""
    
    def __init__(self, output_dir: str = "src/visualization/figures"):
        """
        Initialize report generator.
        
        Args:
            output_dir: Directory to save figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.dirs = {
            'discrimination': self.output_dir / 'discrimination',
            'calibration': self.output_dir / 'calibration',
            'threshold': self.output_dir / 'threshold',
            'temporal': self.output_dir / 'temporal',
            'optuna': self.output_dir / 'optuna',
            'lstm': self.output_dir / 'lstm',
            'backtest': self.output_dir / 'backtest'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
    
    def load_model_data(
        self,
        model_path: Optional[str] = None,
        predictions_path: Optional[str] = None,
        optuna_db_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load model data from files.
        
        Args:
            model_path: Path to saved model
            predictions_path: Path to predictions file
            optuna_db_path: Path to Optuna database
            
        Returns:
            Dictionary with loaded data
        """
        data = {}
        
        # Load model
        if model_path and os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                data['model'] = pickle.load(f)
            print(f"✓ Loaded model from {model_path}")
        
        # Load predictions
        if predictions_path and os.path.exists(predictions_path):
            with open(predictions_path, 'rb') as f:
                predictions = pickle.load(f)
                data.update(predictions)
            print(f"✓ Loaded predictions from {predictions_path}")
        
        # Load Optuna study
        if optuna_db_path and os.path.exists(optuna_db_path):
            import optuna
            study = optuna.load_study(
                study_name="optimization_study",
                storage=f"sqlite:///{optuna_db_path}"
            )
            data['study'] = study
            print(f"✓ Loaded Optuna study from {optuna_db_path}")
        
        return data
    
    def generate_discrimination_plots(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        cv_results: Optional[Dict[str, List]] = None
    ) -> Dict[str, Any]:
        """Generate discrimination plots."""
        print("\n" + "="*60)
        print("GENERATING DISCRIMINATION PLOTS")
        print("="*60)
        
        results = {}
        
        # 1. PR Curve with baseline
        print("1. Creating PR curve with baseline...")
        pr_results = plot_pr_curve_with_baseline(
            y_true, y_proba,
            title="Precision-Recall Curve with Prevalence Baseline",
            save_path=self.dirs['discrimination'] / 'pr_curve.png'
        )
        results['pr_curve'] = pr_results
        
        # 2. PR-AUC distribution (if CV results available)
        if cv_results and 'pr_auc' in cv_results:
            print("2. Creating PR-AUC distribution plot...")
            prevalence = y_true.mean()
            dist_results = plot_pr_auc_distribution(
                cv_results['pr_auc'],
                baseline=prevalence,
                title="PR-AUC Distribution Across CV Folds",
                save_path=self.dirs['discrimination'] / 'pr_auc_distribution.png'
            )
            results['pr_auc_distribution'] = dist_results
        
        # 3. ROC Curve
        print("3. Creating ROC curve...")
        roc_results = plot_roc_curve(
            y_true, y_proba,
            title="ROC Curve (Secondary Metric)",
            save_path=self.dirs['discrimination'] / 'roc_curve.png'
        )
        results['roc_curve'] = roc_results
        
        return results
    
    def generate_calibration_plots(
        self,
        y_true: np.ndarray,
        y_proba_uncalibrated: np.ndarray,
        y_proba_calibrated: Optional[np.ndarray] = None,
        calibration_methods: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """Generate calibration plots."""
        print("\n" + "="*60)
        print("GENERATING CALIBRATION PLOTS")
        print("="*60)
        
        results = {}
        
        # 1. Reliability diagram
        print("1. Creating reliability diagram...")
        rel_results = plot_reliability_diagram(
            y_true, y_proba_uncalibrated, y_proba_calibrated,
            title="Calibration Reliability Diagram",
            save_path=self.dirs['calibration'] / 'reliability_diagram.png'
        )
        results['reliability'] = rel_results
        
        # 2. Brier score comparison
        print("2. Creating Brier score comparison...")
        predictions = {'Uncalibrated': y_proba_uncalibrated}
        if y_proba_calibrated is not None:
            predictions['Calibrated'] = y_proba_calibrated
        
        brier_results = plot_brier_score_comparison(
            y_true, predictions,
            title="Brier Score vs Baseline",
            save_path=self.dirs['calibration'] / 'brier_comparison.png'
        )
        results['brier'] = brier_results
        
        # 3. Calibration method comparison
        if calibration_methods:
            print("3. Creating calibration method comparison...")
            cal_results = plot_calibrator_comparison(
                y_true, calibration_methods,
                title="Calibration Method Comparison",
                save_path=self.dirs['calibration'] / 'calibration_comparison.png'
            )
            results['calibration_comparison'] = cal_results
        
        return results
    
    def generate_threshold_plots(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        costs: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Generate threshold analysis plots."""
        print("\n" + "="*60)
        print("GENERATING THRESHOLD ANALYSIS PLOTS")
        print("="*60)
        
        results = {}
        
        # 1. EV curve
        print("1. Creating Expected Value curve...")
        ev_results = plot_ev_curve(
            y_true, y_proba, costs,
            title="Expected Value vs Threshold",
            save_path=self.dirs['threshold'] / 'ev_curve.png'
        )
        results['ev_curve'] = ev_results
        
        # 2. Confusion matrix at optimal threshold
        print("2. Creating confusion matrix...")
        optimal_threshold = ev_results['optimal_threshold']
        cm_results = plot_confusion_matrix_heatmap(
            y_true, None, threshold=optimal_threshold, y_proba=y_proba,
            title=f"Confusion Matrix at Optimal Threshold",
            save_path=self.dirs['threshold'] / 'confusion_matrix.png'
        )
        results['confusion_matrix'] = cm_results
        
        return results
    
    def generate_temporal_plots(
        self,
        dates: pd.DatetimeIndex,
        splits: List[Dict],
        metrics_df: pd.DataFrame,
        predictions_over_time: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """Generate temporal validation plots."""
        print("\n" + "="*60)
        print("GENERATING TEMPORAL VALIDATION PLOTS")
        print("="*60)
        
        results = {}
        
        # 1. Split timeline
        print("1. Creating split timeline visualization...")
        timeline_results = plot_split_timeline(
            dates, splits,
            purge_gap=5, embargo_pct=0.1,
            title="Walk-Forward Validation Timeline",
            save_path=self.dirs['temporal'] / 'split_timeline.png'
        )
        results['timeline'] = timeline_results
        
        # 2. Walk-forward metrics
        print("2. Creating walk-forward metrics heatmap...")
        metrics_results = plot_walkforward_metrics(
            metrics_df,
            metrics=['pr_auc', 'mcc', 'sharpe'],
            title="Walk-Forward Performance Metrics",
            save_path=self.dirs['temporal'] / 'walkforward_metrics.png'
        )
        results['metrics'] = metrics_results
        
        # 3. Temporal stability
        if predictions_over_time:
            print("3. Creating temporal stability analysis...")
            stability_results = plot_temporal_stability(
                predictions_over_time, dates,
                title="Prediction Stability Analysis",
                save_path=self.dirs['temporal'] / 'temporal_stability.png'
            )
            results['stability'] = stability_results
        
        return results
    
    def generate_optuna_plots(self, study) -> Dict[str, Any]:
        """Generate Optuna optimization plots."""
        print("\n" + "="*60)
        print("GENERATING OPTUNA OPTIMIZATION PLOTS")
        print("="*60)
        
        results = {}
        
        # 1. Optimization history
        print("1. Creating optimization history...")
        hist_results = plot_optimization_history(
            study,
            title="Bayesian Optimization History",
            save_path=self.dirs['optuna'] / 'optimization_history.png'
        )
        results['history'] = hist_results
        
        # 2. Parameter importances
        print("2. Creating parameter importance plot...")
        imp_results = plot_param_importances(
            study,
            title="Hyperparameter Importances (fANOVA)",
            save_path=self.dirs['optuna'] / 'param_importances.png'
        )
        results['importances'] = imp_results
        
        # 3. Parallel coordinates
        print("3. Creating parallel coordinate plot...")
        parallel_results = plot_parallel_coordinate(
            study,
            title="Hyperparameter Parallel Coordinates",
            save_path=self.dirs['optuna'] / 'parallel_coordinates.html'
        )
        results['parallel'] = parallel_results
        
        # 4. Contour plot
        print("4. Creating contour plot...")
        contour_results = plot_contour(
            study,
            title="Hyperparameter Interaction Contour",
            save_path=self.dirs['optuna'] / 'contour.html'
        )
        results['contour'] = contour_results
        
        # 5. Pruning report
        print("5. Generating pruning analysis report...")
        pruning_results = generate_pruning_report(
            study,
            title="Pruning Effectiveness Analysis",
            save_path=self.dirs['optuna'] / 'pruning_report.png'
        )
        results['pruning'] = pruning_results
        
        return results
    
    def generate_lstm_plots(
        self,
        train_history: Optional[Dict[str, List]] = None,
        mc_predictions: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Generate LSTM-specific plots."""
        print("\n" + "="*60)
        print("GENERATING LSTM-SPECIFIC PLOTS")
        print("="*60)
        
        results = {}
        
        # 1. Learning curves
        if train_history:
            print("1. Creating learning curves...")
            learning_results = plot_learning_curves(
                train_history.get('loss', []),
                train_history.get('val_loss', []),
                train_history.get('pr_auc', []),
                train_history.get('val_pr_auc', []),
                metric_name="PR-AUC",
                title="LSTM Training Curves",
                save_path=self.dirs['lstm'] / 'learning_curves.png'
            )
            results['learning'] = learning_results
        
        # 2. MC Dropout uncertainty
        if mc_predictions is not None:
            print("2. Creating MC Dropout uncertainty analysis...")
            mc_results = plot_mc_dropout_uncertainty(
                mc_predictions,
                title="MC Dropout Uncertainty Analysis",
                save_path=self.dirs['lstm'] / 'mc_dropout.png'
            )
            results['mc_dropout'] = mc_results
        
        return results
    
    def generate_backtest_plots(
        self,
        returns: np.ndarray,
        equity_curve: Optional[np.ndarray] = None,
        n_strategies_tested: int = 1
    ) -> Dict[str, Any]:
        """Generate backtest performance plots."""
        print("\n" + "="*60)
        print("GENERATING BACKTEST PLOTS")
        print("="*60)
        
        results = {}
        
        # 1. Equity curve with bootstrap bands
        print("1. Creating equity curve with confidence bands...")
        equity_results = plot_equity_curve_with_bands(
            returns,
            title="Equity Curve with Bootstrap Confidence Bands",
            save_path=self.dirs['backtest'] / 'equity_curve.png'
        )
        results['equity'] = equity_results
        
        # 2. Drawdown analysis
        if equity_curve is None:
            equity_curve = 10000 * (1 + returns).cumprod()
        
        print("2. Creating drawdown analysis...")
        dd_results = plot_drawdown_curve(
            equity_curve,
            title="Drawdown Analysis",
            save_path=self.dirs['backtest'] / 'drawdown.png'
        )
        results['drawdown'] = dd_results
        
        # 3. Returns distribution
        print("3. Creating returns distribution analysis...")
        dist_results = plot_returns_distribution(
            returns,
            title="Returns Distribution Analysis",
            save_path=self.dirs['backtest'] / 'returns_distribution.png'
        )
        results['distribution'] = dist_results
        
        # 4. Q-Q plot
        print("4. Creating Q-Q plot...")
        qq_results = plot_qq_plot(
            returns,
            dist='norm',
            title="Returns Q-Q Plot",
            save_path=self.dirs['backtest'] / 'qq_plot.png'
        )
        results['qq'] = qq_results
        
        # 5. Sharpe vs Deflated Sharpe
        print("5. Creating Sharpe ratio comparison...")
        sharpe_results = plot_sharpe_comparison(
            returns,
            n_strategies_tested=n_strategies_tested,
            title="Sharpe vs Deflated Sharpe Ratio",
            save_path=self.dirs['backtest'] / 'sharpe_comparison.png'
        )
        results['sharpe'] = sharpe_results
        
        return results
    
    def generate_html_report(self) -> str:
        """Generate HTML report with all results."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Model Evaluation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        h1 { color: #333; border-bottom: 3px solid #007bff; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 30px; border-bottom: 1px solid #ddd; }
        .section { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric { display: inline-block; margin: 10px; padding: 10px; background: #f8f9fa; border-radius: 5px; }
        .metric-label { font-weight: bold; color: #666; }
        .metric-value { color: #007bff; font-size: 1.2em; }
        img { max-width: 100%; height: auto; margin: 10px 0; border: 1px solid #ddd; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
        .status { padding: 5px 10px; border-radius: 3px; color: white; font-weight: bold; }
        .status.pass { background: #28a745; }
        .status.fail { background: #dc3545; }
        .status.warning { background: #ffc107; color: #333; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 10px; text-align: left; border: 1px solid #ddd; }
        th { background: #007bff; color: white; }
        tr:nth-child(even) { background: #f8f9fa; }
    </style>
</head>
<body>
    <h1>🎯 Model Evaluation Report</h1>
    <p>Generated: {datetime}</p>
    
    <div class="section">
        <h2>📊 Quality Gates Summary</h2>
        {quality_gates}
    </div>
    
    <div class="section">
        <h2>A. Model Discrimination</h2>
        <div class="grid">
            <img src="discrimination/pr_curve.png" alt="PR Curve">
            <img src="discrimination/roc_curve.png" alt="ROC Curve">
            <img src="discrimination/pr_auc_distribution.png" alt="PR-AUC Distribution">
        </div>
        {discrimination_metrics}
    </div>
    
    <div class="section">
        <h2>B. Calibration</h2>
        <div class="grid">
            <img src="calibration/reliability_diagram.png" alt="Reliability Diagram">
            <img src="calibration/brier_comparison.png" alt="Brier Score">
            <img src="calibration/calibration_comparison.png" alt="Calibration Methods">
        </div>
        {calibration_metrics}
    </div>
    
    <div class="section">
        <h2>C. Threshold Analysis</h2>
        <div class="grid">
            <img src="threshold/ev_curve.png" alt="Expected Value Curve">
            <img src="threshold/confusion_matrix.png" alt="Confusion Matrix">
        </div>
        {threshold_metrics}
    </div>
    
    <div class="section">
        <h2>D. Temporal Validation</h2>
        <img src="temporal/split_timeline.png" alt="Split Timeline">
        <div class="grid">
            <img src="temporal/walkforward_metrics.png" alt="Walk-Forward Metrics">
            <img src="temporal/temporal_stability.png" alt="Temporal Stability">
        </div>
        {temporal_metrics}
    </div>
    
    <div class="section">
        <h2>E. Hyperparameter Optimization</h2>
        <div class="grid">
            <img src="optuna/optimization_history.png" alt="Optimization History">
            <img src="optuna/param_importances.png" alt="Parameter Importances">
            <img src="optuna/pruning_report.png" alt="Pruning Report">
        </div>
        <p>Interactive plots: <a href="optuna/parallel_coordinates.html">Parallel Coordinates</a> | 
           <a href="optuna/contour.html">Contour Plot</a></p>
        {optuna_metrics}
    </div>
    
    <div class="section">
        <h2>F. LSTM Analysis</h2>
        <div class="grid">
            <img src="lstm/learning_curves.png" alt="Learning Curves">
            <img src="lstm/mc_dropout.png" alt="MC Dropout">
        </div>
        {lstm_metrics}
    </div>
    
    <div class="section">
        <h2>G. Backtest Performance</h2>
        <img src="backtest/equity_curve.png" alt="Equity Curve">
        <div class="grid">
            <img src="backtest/drawdown.png" alt="Drawdown">
            <img src="backtest/sharpe_comparison.png" alt="Sharpe Comparison">
        </div>
        <div class="grid">
            <img src="backtest/returns_distribution.png" alt="Returns Distribution">
            <img src="backtest/qq_plot.png" alt="Q-Q Plot">
        </div>
        {backtest_metrics}
    </div>
    
    <div class="section">
        <h2>📝 Conclusion</h2>
        {conclusion}
    </div>
</body>
</html>
        """
        
        # Format datetime
        html = html.replace('{datetime}', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        # Add metrics sections
        html = self._add_metrics_to_html(html)
        
        # Save HTML
        report_path = self.output_dir / 'report.html'
        with open(report_path, 'w') as f:
            f.write(html)
        
        print(f"\n✓ HTML report saved to: {report_path}")
        return str(report_path)
    
    def _add_metrics_to_html(self, html: str) -> str:
        """Add metrics to HTML template."""
        # This would be populated with actual metrics from self.results
        # For now, returning template
        placeholders = {
            '{quality_gates}': '<p>Quality gates evaluation pending...</p>',
            '{discrimination_metrics}': '<p>Metrics pending...</p>',
            '{calibration_metrics}': '<p>Metrics pending...</p>',
            '{threshold_metrics}': '<p>Metrics pending...</p>',
            '{temporal_metrics}': '<p>Metrics pending...</p>',
            '{optuna_metrics}': '<p>Metrics pending...</p>',
            '{lstm_metrics}': '<p>Metrics pending...</p>',
            '{backtest_metrics}': '<p>Metrics pending...</p>',
            '{conclusion}': '<p>Model evaluation complete. Review metrics above for detailed performance assessment.</p>'
        }
        
        for placeholder, value in placeholders.items():
            html = html.replace(placeholder, value)
        
        return html


def generate_full_report(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    returns: Optional[np.ndarray] = None,
    study: Optional[Any] = None,
    output_dir: str = "src/visualization/figures"
) -> str:
    """
    Generate complete model evaluation report.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        returns: Trading returns (optional)
        study: Optuna study object (optional)
        output_dir: Output directory for figures
        
    Returns:
        Path to generated HTML report
    """
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE MODEL EVALUATION REPORT")
    print("="*80)
    
    generator = ModelReportGenerator(output_dir)
    
    # Generate discrimination plots
    generator.results['discrimination'] = generator.generate_discrimination_plots(
        y_true, y_proba
    )
    
    # Generate calibration plots
    generator.results['calibration'] = generator.generate_calibration_plots(
        y_true, y_proba
    )
    
    # Generate threshold plots
    generator.results['threshold'] = generator.generate_threshold_plots(
        y_true, y_proba
    )
    
    # Generate Optuna plots if study provided
    if study:
        generator.results['optuna'] = generator.generate_optuna_plots(study)
    
    # Generate backtest plots if returns provided
    if returns is not None:
        generator.results['backtest'] = generator.generate_backtest_plots(returns)
    
    # Generate HTML report
    report_path = generator.generate_html_report()
    
    print("\n" + "="*80)
    print("✓ REPORT GENERATION COMPLETE")
    print(f"📊 View report at: {report_path}")
    print("="*80)
    
    return report_path


if __name__ == "__main__":
    # Example usage
    print("Model Report Generator")
    print("This module generates comprehensive model evaluation reports.")
    print("\nUsage:")
    print("  from src.visualization.report_generator import generate_full_report")
    print("  report_path = generate_full_report(y_true, y_proba, returns, study)")
    print("\nAll visualizations will be saved to src/visualization/figures/")
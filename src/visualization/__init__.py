"""
Model visualization suite for comprehensive evaluation.
"""

from .model_plots import (
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

from .temporal_plots import (
    plot_split_timeline,
    plot_walkforward_metrics,
    plot_temporal_stability
)

from .optuna_plots import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_contour,
    generate_pruning_report
)

from .backtest_plots import (
    plot_equity_curve_with_bands,
    plot_drawdown_curve,
    plot_returns_distribution,
    plot_qq_plot,
    plot_sharpe_comparison
)

from .report_generator import generate_full_report

__all__ = [
    'plot_pr_curve_with_baseline',
    'plot_pr_auc_distribution', 
    'plot_roc_curve',
    'plot_reliability_diagram',
    'plot_brier_score_comparison',
    'plot_calibrator_comparison',
    'plot_ev_curve',
    'plot_confusion_matrix_heatmap',
    'plot_learning_curves',
    'plot_mc_dropout_uncertainty',
    'plot_split_timeline',
    'plot_walkforward_metrics',
    'plot_temporal_stability',
    'plot_optimization_history',
    'plot_param_importances',
    'plot_parallel_coordinate',
    'plot_contour',
    'generate_pruning_report',
    'plot_equity_curve_with_bands',
    'plot_drawdown_curve',
    'plot_returns_distribution',
    'plot_qq_plot',
    'plot_sharpe_comparison',
    'generate_full_report'
]
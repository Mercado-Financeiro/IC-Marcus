"""
Performance gates checker - blocks CI/CD if metrics degrade.
Hard stops to prevent bad models from reaching production.
"""
import json
import argparse
import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_validation_results(results_path: str) -> dict:
    """Load validation results from JSON file."""
    results_path = Path(results_path)
    
    if not results_path.exists():
        logger.error(f"Validation results not found: {results_path}")
        return {}
    
    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
        logger.info(f"Loaded validation results from {results_path}")
        return results
    except Exception as e:
        logger.error(f"Failed to load validation results: {e}")
        return {}


def check_minimum_performance_gates(metrics: dict, model_name: str) -> dict:
    """Check if metrics meet minimum performance gates."""
    
    # Minimum acceptable performance thresholds
    min_thresholds = {
        'ev': 0.0,         # EV must be positive
        'mcc': 0.05,       # MCC must show some signal
        'auc_pr': 0.55,    # AUC-PR must beat random (0.5 + class imbalance)
        'brier_score': 0.4 # Brier score must be reasonable (< 0.5 for decent calibration)
    }
    
    # For brier_score, lower is better
    brier_check = metrics.get('brier_score', 1.0) < min_thresholds['brier_score']
    
    gates = {
        'model': model_name,
        'passed': True,
        'failed_gates': [],
        'details': {}
    }
    
    # Check each gate
    for metric, min_value in min_thresholds.items():
        current_value = metrics.get(metric, 0)
        
        if metric == 'brier_score':
            passed = current_value < min_value  # Lower is better
            comparison = f"{current_value:.4f} < {min_value:.4f}"
        else:
            passed = current_value >= min_value  # Higher is better
            comparison = f"{current_value:.4f} >= {min_value:.4f}"
        
        gates['details'][metric] = {
            'current': float(current_value),
            'threshold': float(min_value),
            'passed': passed,
            'comparison': comparison
        }
        
        if not passed:
            gates['passed'] = False
            gates['failed_gates'].append(metric)
            logger.error(f"GATE FAILED: {model_name} {metric} = {current_value:.4f} "
                        f"(required: {comparison})")
        else:
            logger.info(f"GATE PASSED: {model_name} {metric} = {current_value:.4f} "
                       f"(required: {comparison})")
    
    return gates


def check_regression_gates(comparison: dict, model_name: str) -> dict:
    """Check if performance regression gates pass."""
    
    gates = {
        'model': model_name,
        'passed': comparison.get('passed', False),
        'details': comparison.get('details', {})
    }
    
    if gates['passed']:
        logger.info(f"REGRESSION GATES PASSED: {model_name} - No significant degradation")
    else:
        logger.error(f"REGRESSION GATES FAILED: {model_name} - Performance degradation detected")
        
        # Log specific failures
        for metric, details in gates['details'].items():
            if not details.get('passed', True):
                logger.error(f"  {metric}: {details.get('degradation_pct', 0):.2f}% degradation "
                           f"exceeds {details.get('threshold_pct', 0):.1f}% threshold")
    
    return gates


def check_data_quality_gates(data_info: dict, model_name: str) -> dict:
    """Check if data quality meets requirements."""
    
    gates = {
        'model': model_name,
        'passed': True,
        'failed_gates': [],
        'details': {}
    }
    
    # Data quality requirements
    min_samples = 500
    max_samples = 50000  # Avoid overly large test sets
    min_features = 5
    max_features = 200
    min_target_rate = 0.05  # At least 5% positive rate
    max_target_rate = 0.95  # At most 95% positive rate
    
    checks = [
        ('n_samples', data_info.get('n_samples', 0), min_samples, max_samples),
        ('n_features', data_info.get('n_features', 0), min_features, max_features),
        ('target_rate', data_info.get('target_rate', 0), min_target_rate, max_target_rate)
    ]
    
    for check_name, current_value, min_val, max_val in checks:
        passed = min_val <= current_value <= max_val
        
        gates['details'][check_name] = {
            'current': float(current_value),
            'min_required': float(min_val),
            'max_allowed': float(max_val),
            'passed': passed
        }
        
        if not passed:
            gates['passed'] = False
            gates['failed_gates'].append(check_name)
            logger.error(f"DATA QUALITY GATE FAILED: {check_name} = {current_value} "
                        f"(required: {min_val} <= value <= {max_val})")
        else:
            logger.info(f"DATA QUALITY GATE PASSED: {check_name} = {current_value}")
    
    return gates


def generate_gate_report(results: dict) -> str:
    """Generate a comprehensive gate check report."""
    
    model_name = results.get('model', 'unknown')
    metrics = results.get('metrics', {})
    comparison = results.get('comparison', {})
    data_info = results.get('data_info', {})
    
    # Run all gate checks
    min_perf_gates = check_minimum_performance_gates(metrics, model_name)
    regression_gates = check_regression_gates(comparison, model_name)
    data_quality_gates = check_data_quality_gates(data_info, model_name)
    
    # Overall pass/fail
    overall_passed = (min_perf_gates['passed'] and 
                     regression_gates['passed'] and 
                     data_quality_gates['passed'])
    
    # Generate report
    report = f"""
🚨 PERFORMANCE GATE CHECK REPORT - {model_name.upper()}
{'=' * 60}

OVERALL STATUS: {'✅ PASSED' if overall_passed else '❌ FAILED'}

📊 MINIMUM PERFORMANCE GATES: {'✅ PASSED' if min_perf_gates['passed'] else '❌ FAILED'}
"""
    
    for metric, details in min_perf_gates['details'].items():
        status = '✅' if details['passed'] else '❌'
        report += f"  {status} {metric}: {details['comparison']}\n"
    
    report += f"""
📉 REGRESSION GATES: {'✅ PASSED' if regression_gates['passed'] else '❌ FAILED'}
"""
    
    if not regression_gates['passed']:
        for metric, details in regression_gates.get('details', {}).items():
            if not details.get('passed', True):
                report += f"  ❌ {metric}: {details.get('degradation_pct', 0):.2f}% degradation\n"
    else:
        report += "  ✅ No performance regression detected\n"
    
    report += f"""
📋 DATA QUALITY GATES: {'✅ PASSED' if data_quality_gates['passed'] else '❌ FAILED'}
"""
    
    for check, details in data_quality_gates['details'].items():
        status = '✅' if details['passed'] else '❌'
        report += f"  {status} {check}: {details['current']}\n"
    
    # Key metrics summary
    report += f"""
📈 KEY METRICS SUMMARY:
  • Expected Value (EV): {metrics.get('ev', 0):.4f}
  • Matthews Correlation Coefficient: {metrics.get('mcc', 0):.4f}
  • AUC-PR: {metrics.get('auc_pr', 0):.4f}
  • Brier Score: {metrics.get('brier_score', 0):.4f}

{'=' * 60}
"""
    
    return report, overall_passed


def main():
    parser = argparse.ArgumentParser(description='Check performance gates')
    parser.add_argument('--results', required=True, help='Path to validation results JSON')
    parser.add_argument('--output', help='Path to save gate report')
    parser.add_argument('--fail-fast', action='store_true', 
                       help='Exit immediately on first gate failure')
    
    args = parser.parse_args()
    
    # Load validation results
    results = load_validation_results(args.results)
    if not results:
        logger.error("No validation results found")
        sys.exit(1)
    
    # Generate gate report
    report, passed = generate_gate_report(results)
    
    # Print report
    print(report)
    
    # Save report if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
        logger.info(f"Gate report saved to {output_path}")
    
    # Exit with appropriate code
    if passed:
        logger.info("🎉 ALL GATES PASSED - Ready for deployment")
        sys.exit(0)
    else:
        logger.error("🚫 GATES FAILED - Blocking deployment")
        if args.fail_fast:
            logger.error("Fail-fast mode: Exiting immediately")
        sys.exit(1)


if __name__ == "__main__":
    main()
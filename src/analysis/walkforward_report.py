#!/usr/bin/env python
"""
Analyze Walk-Forward Results from Database
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from src.eval.walkforward_db import WalkForwardDB


def analyze_results():
    """Analyze and display walk-forward results"""
    
    print("=" * 60)
    print("WALK-FORWARD ANALYSIS RESULTS")
    print("=" * 60)
    
    with WalkForwardDB() as db:
        # Get all runs
        cursor = db.conn.cursor()
        cursor.execute("""
            SELECT id, timestamp, symbol, timeframe, start_date, end_date,
                   n_folds_total, n_folds_valid
            FROM walkforward_runs
            ORDER BY id DESC
            LIMIT 10
        """)
        
        runs = cursor.fetchall()
        
        if not runs:
            print("No runs found in database")
            return
            
        print(f"\nFound {len(runs)} recent runs:")
        print("-" * 40)
        
        for run in runs:
            run_id = run[0]
            timestamp = run[1]
            symbol = run[2]
            timeframe = run[3]
            start_date = run[4]
            end_date = run[5]
            n_folds_total = run[6]
            n_folds_valid = run[7]
            
            print(f"\nRun ID: {run_id}")
            print(f"  Timestamp: {timestamp}")
            print(f"  Symbol: {symbol} {timeframe}")
            print(f"  Period: {start_date} to {end_date}")
            print(f"  Folds: {n_folds_valid}/{n_folds_total} valid")
            
            # Get fold results for this run
            cursor.execute("""
                SELECT fold_num, auprc, mcc, sharpe_ratio, n_trades, 
                       action_rate, threshold_long, threshold_short,
                       error_message
                FROM fold_results
                WHERE run_id = ?
                ORDER BY fold_num
            """, (run_id,))
            
            folds = cursor.fetchall()
            
            if folds:
                print("\n  Fold Results:")
                for fold in folds:
                    fold_num = fold[0]
                    if fold[8]:  # error_message
                        print(f"    Fold {fold_num}: ERROR - {fold[8][:50]}")
                    else:
                        auprc = fold[1]
                        mcc = fold[2]
                        sharpe = fold[3]
                        n_trades = fold[4]
                        action_rate = fold[5]
                        tau_long = fold[6]
                        tau_short = fold[7]
                        
                        print(f"    Fold {fold_num}:")
                        print(f"      AUPRC={auprc:.3f}, MCC={mcc:.3f}, Sharpe={sharpe:.2f}")
                        print(f"      Trades={n_trades}, Action={action_rate:.1%}")
                        print(f"      Thresholds: Long={tau_long:.3f}, Short={tau_short:.3f}")
            
            # Get aggregate stats
            stats = db.get_aggregate_stats(run_id)
            if stats and stats.get('n_folds', 0) > 0:
                print("\n  Aggregate Statistics:")
                print(f"    AUPRC: {stats.get('avg_auprc', 0):.3f} ± {stats.get('std_auprc', 0):.3f}")
                print(f"    MCC: {stats.get('avg_mcc', 0):.3f} ± {stats.get('std_mcc', 0):.3f}")
                print(f"    Sharpe: {stats.get('avg_sharpe', 0):.2f} ± {stats.get('std_sharpe', 0):.2f}")
                print(f"    Total Trades: {stats.get('total_trades', 0)}")
                print(f"    Avg Action Rate: {stats.get('avg_action_rate', 0):.1%}")
        
        print("\n" + "=" * 60)
        
        # Export best run to CSV
        if runs:
            best_run_id = runs[0][0]  # Most recent run
            csv_path = f"artifacts/reports/walkforward_run_{best_run_id}.csv"
            Path("artifacts/reports").mkdir(parents=True, exist_ok=True)
            
            df = db.get_run_summary(best_run_id)
            if not df.empty:
                df.to_csv(csv_path, index=False)
                print(f"Exported results to: {csv_path}")


if __name__ == "__main__":
    analyze_results()
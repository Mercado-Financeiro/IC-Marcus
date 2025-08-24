"""
Walk-Forward Analysis with Database Storage
Stores results incrementally in SQLite database
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any


class WalkForwardDB:
    """Database manager for walk-forward results"""
    
    def __init__(self, db_path: str = "artifacts/walkforward_results.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self._create_tables()
    
    def _create_tables(self):
        """Create tables if they don't exist"""
        cursor = self.conn.cursor()
        
        # Main results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS walkforward_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT,
                timeframe TEXT,
                start_date TEXT,
                end_date TEXT,
                lookback INTEGER,
                horizon INTEGER,
                label_threshold REAL,
                val_frac REAL,
                test_size INTEGER,
                fee_bps REAL,
                slippage_bps REAL,
                n_folds_total INTEGER,
                n_folds_valid INTEGER,
                config_json TEXT
            )
        """)
        
        # Fold results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS fold_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                fold_num INTEGER,
                train_start INTEGER,
                train_end INTEGER,
                val_start INTEGER,
                val_end INTEGER,
                test_start INTEGER,
                test_end INTEGER,
                
                -- ML Metrics
                prevalence REAL,
                auprc REAL,
                auprc_norm REAL,
                mcc REAL,
                brier_score REAL,
                ece REAL,
                threshold REAL,  -- Legacy single threshold
                threshold_long REAL,  -- New: long entry threshold
                threshold_short REAL,  -- New: short entry threshold
                
                -- Trading Metrics
                n_trades INTEGER,
                turnover REAL,
                pnl_mean REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                win_rate REAL,
                
                -- Position Stats
                long_rate REAL,
                short_rate REAL,
                action_rate REAL,
                
                -- Probability Distribution
                prob_min REAL,
                prob_p25 REAL,
                prob_median REAL,
                prob_p75 REAL,
                prob_max REAL,
                
                -- Additional Info
                execution_time_seconds REAL,
                error_message TEXT,
                
                FOREIGN KEY(run_id) REFERENCES walkforward_runs(id)
            )
        """)
        
        # Feature importance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feature_importance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                fold_num INTEGER,
                feature_name TEXT,
                importance_value REAL,
                importance_rank INTEGER,
                
                FOREIGN KEY(run_id) REFERENCES walkforward_runs(id)
            )
        """)
        
        # Trade log table (optional, for detailed analysis)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS trade_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                fold_num INTEGER,
                timestamp TEXT,
                position INTEGER,  -- -1, 0, 1
                probability REAL,
                actual_label INTEGER,
                pnl REAL,
                cumulative_pnl REAL,
                
                FOREIGN KEY(run_id) REFERENCES walkforward_runs(id)
            )
        """)
        
        # Market data cache table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_data_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                timeframe TEXT,
                timestamp DATETIME,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                downloaded_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timeframe, timestamp)
            )
        """)
        
        # Features cache table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS features_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                timeframe TEXT,
                timestamp DATETIME,
                features_json TEXT,  -- JSON encoded features
                feature_version TEXT,  -- Version of feature engineering
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timeframe, timestamp, feature_version)
            )
        """)
        
        # Create indexes for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time 
            ON market_data_cache(symbol, timeframe, timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_features_symbol_time 
            ON features_cache(symbol, timeframe, timestamp)
        """)
        
        self.conn.commit()
    
    def create_run(self, config: Dict[str, Any]) -> int:
        """Create a new run and return its ID"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO walkforward_runs (
                symbol, timeframe, start_date, end_date,
                lookback, horizon, label_threshold,
                val_frac, test_size, fee_bps, slippage_bps,
                n_folds_total, n_folds_valid, config_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            config.get('symbol'),
            config.get('timeframe'),
            config.get('start_date'),
            config.get('end_date'),
            config.get('lookback'),
            config.get('horizon'),
            config.get('label_threshold'),
            config.get('val_frac'),
            config.get('test_size'),
            config.get('fee_bps'),
            config.get('slippage_bps'),
            config.get('n_folds_total', 0),
            config.get('n_folds_valid', 0),
            json.dumps(config)
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def save_fold_result(self, run_id: int, fold_num: int, results: Dict[str, Any]):
        """Save results for a single fold"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO fold_results (
                run_id, fold_num,
                train_start, train_end, val_start, val_end, test_start, test_end,
                prevalence, auprc, auprc_norm, mcc, brier_score, ece, threshold, threshold_long, threshold_short,
                n_trades, turnover, pnl_mean, sharpe_ratio, max_drawdown, win_rate,
                long_rate, short_rate, action_rate,
                prob_min, prob_p25, prob_median, prob_p75, prob_max,
                execution_time_seconds, error_message
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?
            )
        """, (
            run_id, fold_num,
            results.get('train_start'), results.get('train_end'),
            results.get('val_start'), results.get('val_end'),
            results.get('test_start'), results.get('test_end'),
            
            results.get('prevalence'),
            results.get('auprc'),
            results.get('auprc_norm'),
            results.get('mcc'),
            results.get('brier_score'),
            results.get('ece'),
            results.get('threshold'),  # Legacy
            results.get('threshold_long'),  # New
            results.get('threshold_short'),  # New
            
            results.get('n_trades'),
            results.get('turnover'),
            results.get('pnl_mean'),
            results.get('sharpe_ratio'),
            results.get('max_drawdown'),
            results.get('win_rate'),
            
            results.get('long_rate'),
            results.get('short_rate'),
            results.get('action_rate'),
            
            results.get('prob_min'),
            results.get('prob_p25'),
            results.get('prob_median'),
            results.get('prob_p75'),
            results.get('prob_max'),
            
            results.get('execution_time_seconds'),
            results.get('error_message')
        ))
        
        self.conn.commit()
    
    def save_feature_importance(self, run_id: int, fold_num: int, 
                               importance_dict: Dict[str, float]):
        """Save feature importance for a fold"""
        cursor = self.conn.cursor()
        
        # Sort by importance
        sorted_features = sorted(importance_dict.items(), 
                               key=lambda x: x[1], reverse=True)
        
        for rank, (feature, importance) in enumerate(sorted_features, 1):
            cursor.execute("""
                INSERT INTO feature_importance (
                    run_id, fold_num, feature_name, importance_value, importance_rank
                ) VALUES (?, ?, ?, ?, ?)
            """, (run_id, fold_num, feature, importance, rank))
        
        self.conn.commit()
    
    def update_run_stats(self, run_id: int, n_folds_valid: int):
        """Update run statistics"""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE walkforward_runs 
            SET n_folds_valid = ?
            WHERE id = ?
        """, (n_folds_valid, run_id))
        self.conn.commit()
    
    def get_run_summary(self, run_id: int) -> pd.DataFrame:
        """Get summary statistics for a run"""
        query = """
            SELECT 
                fold_num,
                auprc, auprc_norm, mcc, brier_score, ece,
                sharpe_ratio, n_trades, turnover, pnl_mean,
                action_rate, threshold
            FROM fold_results
            WHERE run_id = ? AND error_message IS NULL
            ORDER BY fold_num
        """
        
        df = pd.read_sql_query(query, self.conn, params=(run_id,))
        return df
    
    def get_aggregate_stats(self, run_id: int) -> Dict[str, float]:
        """Get aggregate statistics across all folds"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT 
                AVG(auprc) as avg_auprc,
                SQRT(AVG(auprc*auprc) - AVG(auprc)*AVG(auprc)) as std_auprc,
                AVG(mcc) as avg_mcc,
                SQRT(AVG(mcc*mcc) - AVG(mcc)*AVG(mcc)) as std_mcc,
                AVG(sharpe_ratio) as avg_sharpe,
                SQRT(AVG(sharpe_ratio*sharpe_ratio) - AVG(sharpe_ratio)*AVG(sharpe_ratio)) as std_sharpe,
                AVG(turnover) as avg_turnover,
                SUM(n_trades) as total_trades,
                AVG(action_rate) as avg_action_rate,
                COUNT(*) as n_folds
            FROM fold_results
            WHERE run_id = ? AND error_message IS NULL
        """, (run_id,))
        
        result = cursor.fetchone()
        cols = [desc[0] for desc in cursor.description]
        
        return dict(zip(cols, result)) if result else {}
    
    def export_to_csv(self, run_id: int, output_path: str):
        """Export run results to CSV"""
        df = self.get_run_summary(run_id)
        df.to_csv(output_path, index=False)
        print(f"Results exported to {output_path}")
    
    def close(self):
        """Close database connection"""
        self.conn.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Usage example
if __name__ == "__main__":
    with WalkForwardDB() as db:
        # Create a new run
        config = {
            'symbol': 'BTCUSDT',
            'timeframe': '15m',
            'start_date': '2024-03-15',
            'end_date': '2024-07-15',
            'lookback': 16,
            'horizon': 2,
            'label_threshold': 0.0005,
            'val_frac': 0.1,
            'test_size': 200,
            'fee_bps': 8,
            'slippage_bps': 4
        }
        
        run_id = db.create_run(config)
        print(f"Created run ID: {run_id}")
        
        # Example of saving fold results
        fold_result = {
            'train_start': 0,
            'train_end': 1000,
            'val_start': 1020,
            'val_end': 1200,
            'test_start': 1220,
            'test_end': 1420,
            'auprc': 0.35,
            'mcc': 0.15,
            'sharpe_ratio': 1.2,
            'n_trades': 45,
            'action_rate': 0.08
        }
        
        db.save_fold_result(run_id, 1, fold_result)
        print("Fold result saved")
        
        # Get summary
        stats = db.get_aggregate_stats(run_id)
        print(f"Aggregate stats: {stats}")
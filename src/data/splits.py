from __future__ import annotations

from typing import Iterator, Sequence, Tuple, List, Optional, Dict
import numpy as np
import logging

logger = logging.getLogger(__name__)


def _array_indices(n: int) -> List[int]:
    return list(range(n))


def purged_kfold_index(
    times: Sequence,
    n_splits: int = 5,
    embargo: int = 0,
) -> Iterator[Tuple[List[int], List[int]]]:
    """Yield (train_idx, val_idx) with simple purge/embargo semantics.

    - No shuffling. Assumes `times` is ordered.
    - Embargo excludes a window around each validation fold from the train set.
    - Implemented without external deps to run offline.
    """
    n = len(times)
    idx = _array_indices(n)
    # split contiguous folds
    fold_sizes = [n // n_splits + (1 if x < n % n_splits else 0) for x in range(n_splits)]
    current = 0
    folds: List[List[int]] = []
    for fs in fold_sizes:
        folds.append(idx[current : current + fs])
        current += fs
    for val in folds:
        val_start, val_end = val[0], val[-1]
        train = [i for i in idx if i <= max(val_start - embargo - 1, -1) or i >= min(val_end + embargo + 1, n)]
        yield train, val


def walk_forward_anchored(
    n_samples: int,
    n_splits: int = 5,
    min_train_size: int | None = None,
    embargo_size: int = 0,
) -> Iterator[Tuple[List[int], List[int]]]:
    """Anchored walk-forward with embargo: train is [0:i), val is [i+embargo:j).

    Parameters
    - n_samples: total sample count
    - n_splits: number of validation windows
    - min_train_size: optional minimal initial train size
    - embargo_size: number of samples to skip between train and validation (prevents leakage)
    """
    n = n_samples
    split_points = [int((k + 1) * n / (n_splits + 1)) for k in range(n_splits)]
    anchor = min_train_size or max(1, int(0.2 * n))
    for sp in split_points:
        train_end = max(anchor, sp)
        val_start = min(train_end + embargo_size, n - 1)
        val_end = min(n, val_start + max(1, (n - val_start) // (n_splits)))
        train = list(range(0, train_end))
        val = list(range(val_start, val_end))
        if val:
            yield train, val


def temporal_train_test_split(
    data: Sequence,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    embargo_bars: int = 40,
    time_col: Optional[str] = None,
) -> Dict[str, Tuple[List[int], slice]]:
    """Create temporal train/val/test split with proper embargo.
    
    This function implements best practices for time series splitting:
    - No shuffling (maintains temporal order)
    - Embargo periods between splits to prevent leakage
    - Configurable split ratios
    
    Args:
        data: Input data (DataFrame or array)
        train_ratio: Fraction of data for training (default 0.6)
        val_ratio: Fraction of data for validation (default 0.2)
        embargo_bars: Number of bars to skip between splits (default 40 = 10 hours for 15-min)
        time_col: Optional time column name for logging
        
    Returns:
        Dictionary with 'train', 'val', 'test' keys containing indices and slices
        
    Example:
        >>> splits = temporal_train_test_split(df, embargo_bars=40)
        >>> df_train = df.iloc[splits['train'][1]]
        >>> df_val = df.iloc[splits['val'][1]]
        >>> df_test = df.iloc[splits['test'][1]]
    """
    n = len(data)
    
    # Calculate split points
    train_end = int(n * train_ratio)
    val_start = train_end + embargo_bars
    val_end = val_start + int(n * val_ratio)
    test_start = val_end + embargo_bars
    
    # Ensure we have enough data for all splits
    if test_start >= n:
        logger.warning(f"Embargo too large ({embargo_bars} bars). Reducing to maintain test set.")
        # Adjust embargo to ensure test set has at least 10% of data
        max_embargo = int((n * (1 - train_ratio - val_ratio) - n * 0.1) / 2)
        embargo_bars = max(0, min(embargo_bars, max_embargo))
        val_start = train_end + embargo_bars
        val_end = val_start + int(n * val_ratio)
        test_start = val_end + embargo_bars
    
    # Create splits
    splits = {
        'train': (list(range(0, train_end)), slice(0, train_end)),
        'val': (list(range(val_start, val_end)), slice(val_start, val_end)),
        'test': (list(range(test_start, n)), slice(test_start, n))
    }
    
    # Log split information
    logger.info(f"Temporal split with {embargo_bars}-bar embargo:")
    logger.info(f"  Train: {train_end} samples (0:{train_end})")
    logger.info(f"  Val:   {val_end - val_start} samples ({val_start}:{val_end})")
    logger.info(f"  Test:  {n - test_start} samples ({test_start}:{n})")
    
    if embargo_bars > 0:
        if hasattr(data, 'index') and hasattr(data.index, 'freq'):
            # For pandas with datetime index
            try:
                # Try new pandas API first
                freq_delta = pd.Timedelta(data.index.freq) if data.index.freq else pd.Timedelta(minutes=15)
                freq_minutes = freq_delta.total_seconds() / 60
            except:
                # Fallback for older pandas or unknown freq
                freq_minutes = 15
            embargo_hours = embargo_bars * freq_minutes / 60
            logger.info(f"  Embargo: {embargo_bars} bars = {embargo_hours:.1f} hours")
        else:
            # Assume 15-min bars as default
            logger.info(f"  Embargo: {embargo_bars} bars (~{embargo_bars * 0.25:.1f} hours for 15-min data)")
    
    # Validate splits
    total_samples = len(splits['train'][0]) + len(splits['val'][0]) + len(splits['test'][0])
    total_possible = n - 2 * embargo_bars
    logger.info(f"  Total samples used: {total_samples}/{n} ({total_samples/n*100:.1f}%)")
    
    return splits


class PurgedKFold:
    """Simple Purged K-Fold splitter with embargo for time series.
    
    Implements López de Prado's purged cross-validation to prevent data leakage.
    Embargo creates a gap between train and validation sets to account for 
    prediction horizon and feature lookahead.

    Usage:
        cv = PurgedKFold(n_splits=5, embargo=40)  # 40 bars = 10 hours for 15-min data
        for train_idx, val_idx in cv.split(X, y): ...
    """

    def __init__(self, n_splits: int = 5, embargo: int = 0):
        if n_splits < 2:
            raise ValueError("n_splits must be >=2")
        self.n_splits = int(n_splits)
        self.embargo = int(embargo)
        if embargo > 0:
            print(f"PurgedKFold initialized with {embargo}-bar embargo to prevent temporal leakage")

    def split(self, X, y: Optional[Sequence] = None) -> Iterator[Tuple[List[int], List[int]]]:
        n = len(X)
        idx = _array_indices(n)
        # contiguous folds
        fold_sizes = [n // self.n_splits + (1 if x < n % self.n_splits else 0) for x in range(self.n_splits)]
        current = 0
        folds: List[List[int]] = []
        for fs in fold_sizes:
            folds.append(idx[current : current + fs])
            current += fs
        for val in folds:
            val_start, val_end = val[0], val[-1]
            train = [
                i for i in idx
                if i <= max(val_start - self.embargo - 1, -1) or i >= min(val_end + self.embargo + 1, n)
            ]
            yield train, val
    
    def get_n_splits(self) -> int:
        """Return number of splits."""
        return self.n_splits

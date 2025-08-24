from __future__ import annotations

from typing import Iterator, Sequence, Tuple, List, Optional


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
) -> Iterator[Tuple[List[int], List[int]]]:
    """Anchored walk-forward: train is [0:i), val is [i:j).

    Parameters
    - n_samples: total sample count
    - n_splits: number of validation windows
    - min_train_size: optional minimal initial train size
    """
    n = n_samples
    split_points = [int((k + 1) * n / (n_splits + 1)) for k in range(n_splits)]
    anchor = min_train_size or max(1, int(0.2 * n))
    for sp in split_points:
        train_end = max(anchor, sp)
        val_end = min(n, sp + max(1, (n - sp) // (n_splits)))
        train = list(range(0, train_end))
        val = list(range(train_end, val_end))
        if val:
            yield train, val


class PurgedKFold:
    """Simple Purged K-Fold splitter with embargo for time series.

    Usage:
        cv = PurgedKFold(n_splits=5, embargo=5)
        for train_idx, val_idx in cv.split(X, y): ...
    """

    def __init__(self, n_splits: int = 5, embargo: int = 0):
        if n_splits < 2:
            raise ValueError("n_splits must be >=2")
        self.n_splits = int(n_splits)
        self.embargo = int(embargo)

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

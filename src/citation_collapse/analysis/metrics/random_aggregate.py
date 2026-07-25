import os
import re
from collections.abc import Sequence

import numpy as np
from scipy import stats


def _get_run_index(indexed_entry):
    return indexed_entry[0]


def list_random_run_dirs(random_root: str) -> list[str]:
    if not os.path.isdir(random_root):
        return []
    pattern = re.compile(r"^run_(\d+)$")
    indexed: list[tuple[int, str]] = []
    for name in os.listdir(random_root):
        m = pattern.match(name)
        if not m:
            continue
        path = os.path.join(random_root, name)
        if os.path.isdir(path):
            indexed.append((int(m.group(1)), path))
    indexed.sort(key=_get_run_index)
    return [p for _, p in indexed]


def bonferroni_mean_ci(
    values: Sequence[float],
    m: int,
    family_alpha: float = 0.01,
) -> tuple[float, float, float]:
    """Mean of values and two-sided Bonferroni-adjusted CI for the mean (m simultaneous intervals, FWER <= family_alpha)."""
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    n = int(finite_values.size)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(finite_values))
    if n == 1 or m < 1:
        return mean, mean, mean
    df = n - 1
    t_crit = float(stats.t.ppf(1.0 - family_alpha / (2 * m), df))
    se = float(np.std(finite_values, ddof=1) / np.sqrt(n))
    half = t_crit * se
    return mean, mean - half, mean + half

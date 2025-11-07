from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from .config import MomentumConfig


class MomentumSignal:
    """
    Compute cross-sectional momentum scores and trading signals.

    Expected input:
        - `prices`: pandas.DataFrame with DateTimeIndex (ascending), one column per symbol,
          containing close prices. The last row is "today" (as-of).
    """

    def __init__(self, config: MomentumConfig, weights: Tuple[float, float, float, float] | None = None):
        self.config = config
        self.weights: Tuple[float, float, float, float] = weights or (1.0, -0.5, 0.5, 0.25)

    def generate_signal(self, prices: pd.DataFrame) -> float:
        """Return the average composite momentum score across all symbols."""
        scores = self.scores_by_symbol(prices)
        return float(scores.mean())

    def scores_by_symbol(self, prices: pd.DataFrame) -> pd.Series:
        """Compute composite momentum score per symbol for the latest date."""
        comps = self._compute_components(prices)
        weights = pd.Series(self.weights, index=["M", "V", "R", "S"])
        return comps.dot(weights)

    def relative_signal(self, prices: pd.DataFrame) -> pd.Series:
        """Cross-sectional (relative) z-score of composite momentum."""
        scores = self.scores_by_symbol(prices)
        return (scores - scores.mean()) / scores.std(ddof=0)

    def absolute_signal(self, prices: pd.DataFrame) -> pd.Series:
        """Time-series (absolute) rolling z-score per symbol (latest value)."""
        W = int(self.config.momentum_window)
        rets = np.log(prices.replace(0.0, np.nan)).diff().dropna(how="all")
        wM, wV, wR, wS = self.weights
        M = rets.rolling(W).sum()
        V = rets.rolling(W).std(ddof=0)
        R = (rets > 0).rolling(W).mean()
        abs_r = rets.abs()
        denom = abs_r.rolling(W).sum()
        S = 1.0 - ((abs_r / denom) ** 2).rolling(W).sum()
        comp = wM * M + wV * V + wR * R + wS * S
        latest = comp.apply(lambda x: self._zscore_rolling(x).iloc[-1] if x.notna().any() else 0.0)
        return latest

    def _compute_components(self, prices: pd.DataFrame) -> pd.DataFrame:
        W = int(self.config.momentum_window)
        px = prices.replace(0.0, np.nan)
        rets = np.log(px).diff().dropna(how="all")
        window = rets.tail(W)
        n = window.notna().sum(axis=0).clip(lower=1)
        M = window.sum(axis=0, skipna=True)
        V = window.std(axis=0, ddof=0, skipna=True)
        R = (window > 0).sum(axis=0) / n
        abs_r = window.abs()
        denom = abs_r.sum(axis=0, skipna=True).replace(0.0, np.nan)
        weights = abs_r.div(denom, axis=1).fillna(0.0)
        HHI = (weights.pow(2)).sum(axis=0, skipna=True)
        S = 1.0 - HHI
        comps = pd.DataFrame({"M": M, "V": V, "R": R, "S": S})
        return comps.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _zscore_rolling(self, series: pd.Series) -> pd.Series:
        """Rolling z-score with window from config."""
        W = int(self.config.zscore_window)
        if W <= 1:
            mu = series.expanding(min_periods=1).mean()
            sd = series.expanding(min_periods=2).std(ddof=0).replace(0.0, np.nan)
            return (series - mu) / sd
        mu = series.rolling(window=W, min_periods=W).mean()
        sd = series.rolling(window=W, min_periods=W).std(ddof=0).replace(0.0, np.nan)
        return (series - mu) / sd
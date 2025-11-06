from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import MomentumConfig


class MomentumSignal:
    """
    Compute cross-sectional momentum scores and trading signals.

    Expected input:
        - `prices`: pandas.DataFrame with DateTimeIndex (ascending), one column per symbol,
          containing close prices. The last row is "today" (as-of).

    Outputs (from `generate_signal`):
        - Returns a single float: the aggregate momentum score across the cross-section.
    """

    def __init__(self, config: MomentumConfig, weights: Tuple[float, float, float, float] | None = None):
        self.config = config
        self.weights: Tuple[float, float, float, float] = weights or (1.0, -0.5, 0.5, 0.25)

    # -------- public API --------
    def generate_signal(self, prices: pd.DataFrame) -> float:
        """
        Compute and return the overall composite momentum score (float).

        Parameters
        ----------
        prices : pd.DataFrame
            Close prices, index sorted ascending. Columns are tickers.

        Returns
        -------
        float
            Aggregate momentum score across the cross-section.
        """
        self._validate_prices(prices)

        comps = self._compute_components(prices)
        weights = np.array(self.weights)
        comps_values = comps[["M", "V", "R", "S"]].values
        scores = comps_values @ weights
        total_score = float(np.mean(scores)) 

        return total_score

    # -------- internals --------
    def _validate_prices(self, prices: pd.DataFrame) -> None:
        if not isinstance(prices, pd.DataFrame):
            raise TypeError("prices must be a pandas DataFrame")
        if prices.shape[0] < self.config.momentum_window + 1:
            raise ValueError(
                f"need at least momentum_window+1 rows; got {prices.shape[0]}"
            )
        if not prices.index.is_monotonic_increasing:
            prices.sort_index(inplace=True)

    def _compute_components(self, prices: pd.DataFrame) -> pd.DataFrame:
        W = int(self.config.momentum_window)

        # Treat zero prices as missing to avoid log(0)
        px = prices.replace(0.0, np.nan)

        # daily log returns
        rets = np.log(px).diff()
        # only drop rows that are entirely NaN (not rows with just one NaN)
        rets = rets.dropna(how="all")

        # take the trailing window (may still have NaNs in some columns)
        window = rets.tail(W)

        # counts of valid observations per column in the window
        n = window.notna().sum(axis=0).clip(lower=1)

        # M: cumulative log return (NaNs ignored)
        M = window.sum(axis=0, skipna=True)

        # V: realized volatility (use ddof=0; NaNs ignored)
        V = window.std(axis=0, ddof=0, skipna=True)

        # R: fraction of up days using effective count per column
        R = (window > 0).sum(axis=0) / n

        # S: smoothness = 1 - HHI of abs daily contributions (NaNs ignored)
        abs_r = window.abs()
        denom = abs_r.sum(axis=0, skipna=True).replace(0.0, np.nan)
        weights = abs_r.div(denom, axis=1)  # columnwise normalization
        weights = weights.fillna(0.0)
        HHI = (weights.pow(2)).sum(axis=0, skipna=True)
        S = 1.0 - HHI

        comps = pd.DataFrame({"M": M, "V": V, "R": R, "S": S})
        comps = comps.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return comps

    def _zscore_rolling(self, series: pd.Series) -> pd.Series:
        """
        Time-series z-score using a rolling window from config.zscore_window.

        z_t = (x_t - mean_{t-W+1..t}) / std_{t-W+1..t}

        Parameters
        ----------
        series : pd.Series
            A single asset's time series (e.g., daily scores or returns).

        Returns
        -------
        pd.Series
            Rolling z-score with NaNs for the initial periods before the window fills.
        """
        W = int(self.config.zscore_window)
        if W <= 1:
            # fall back to zero-centered scaling if misconfigured
            mu = series.expanding(min_periods=1).mean()
            sd = series.expanding(min_periods=2).std(ddof=0).replace(0.0, np.nan)
            return (series - mu) / sd

        mu = series.rolling(window=W, min_periods=W).mean()
        sd = series.rolling(window=W, min_periods=W).std(ddof=0)
        sd = sd.replace(0.0, np.nan)  # avoid division by zero
        z = (series - mu) / sd
        return z
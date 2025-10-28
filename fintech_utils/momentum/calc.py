import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict

def _rolling_prod_1p(x: pd.Series) -> float:
    """Helper: product(1+x) - 1"""
    return float(np.prod(1.0 + x) - 1.0)

def compute_12_1_momentum(
    prices: pd.DataFrame,
    trading_days_per_month: int = 21,
    trading_days_per_year: int = 252,
) -> pd.DataFrame:
    """
    12-1 momentum: cumulative return over ~12 months, excluding the most recent ~1 month.
    prices: DataFrame of prices with DateTime index
    return: DataFrame of 12-1 momentum values
    """
    rets = prices.pct_change()
    window = trading_days_per_year - trading_days_per_month  # ~252 - 21 = 231
    rets_excl_recent = rets.shift(trading_days_per_month)
    mom_12_1 = rets_excl_recent.rolling(window=window, min_periods=window).apply(_rolling_prod_1p, raw=False)
    return mom_12_1

def compute_path_smoothness_metrics(
    prices: pd.DataFrame,
    trading_days_per_month: int = 21,
    trading_days_per_year: int = 252,
    top_k: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    Smoothness metrics on the same 12-1 window:
      - pos_ratio: fraction of up days
      - conc_topk_abs: share of total |daily return| concentrated in the largest k days (lower = smoother)
      - herfindahl_abs: Herfindahl index of |daily return| shares (lower = smoother)
      - vol_realized: realized volatility (std of daily returns)
    prices: DataFrame of prices with DateTime index
    return: dict of DataFrames for each metric
    """
    rets = prices.pct_change()
    window = trading_days_per_year - trading_days_per_month
    rets_excl_recent = rets.shift(trading_days_per_month)

    # Positive-day ratio
    pos_ratio = rets_excl_recent.rolling(window=window, min_periods=window).apply(
        lambda x: np.mean(x > 0), raw=False
    )

    # Concentration and Herfindahl on absolute daily returns
    def _conc_topk_abs(x: np.ndarray, k: int = top_k) -> float:
        ax = np.abs(x)
        s = ax.sum()
        if s == 0:
            return 0.0
        shares = np.sort(ax / s)[::-1]
        return float(np.sum(shares[:k]))

    def _herfindahl_abs(x: np.ndarray) -> float:
        ax = np.abs(x)
        s = ax.sum()
        if s == 0:
            return 0.0
        shares = ax / s
        return float(np.sum(shares ** 2))

    conc_topk_abs = rets_excl_recent.rolling(window=window, min_periods=window).apply(_conc_topk_abs, raw=True)
    herfindahl_abs = rets_excl_recent.rolling(window=window, min_periods=window).apply(_herfindahl_abs, raw=True)

    # Realized volatility (daily std)
    vol_realized = rets_excl_recent.rolling(window=window, min_periods=window).std()

    return {
        "pos_ratio": pos_ratio,
        "conc_topk_abs": conc_topk_abs,
        "herfindahl_abs": herfindahl_abs,
        "vol_realized": vol_realized,
    }

def compute_vol_adj_momentum(mom_12_1: pd.DataFrame, vol_realized: pd.DataFrame, eps: float = 1e-8) -> pd.DataFrame:
    """Volatility-adjusted momentum: (12-1) / realized vol.
    mom_12_1: DataFrame of 12-1 momentum values
    vol_realized: DataFrame of realized volatility values
    return: DataFrame of vol-adjusted momentum values
    """
    return mom_12_1 / (vol_realized + eps)
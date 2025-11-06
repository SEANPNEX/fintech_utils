
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

from ..findata.options import bsm, bsm_pnl
from itertools import product


# Lightweight IV proxy when IV history is unavailable
# Uses rolling realized volatility from prices (annualized) as a stand-in IV history.
# This keeps the scenario engine operational with price data only.

def _iv_proxy(prices: pd.DataFrame | Dict[str, pd.Series], window: int = 30) -> pd.DataFrame:
    px = _ensure_df(prices, "prices").ffill()
    r = px.pct_change()
    iv = r.rolling(window).std() * np.sqrt(252.0)
    # Fill reasonable defaults to avoid empty columns early in history
    iv = iv.replace([np.inf, -np.inf], np.nan)
    if iv.isna().all().all():
        # extreme fallback
        iv = px.copy() * 0.0 + 0.20
    else:
        # forward/backward fill within each column; fallback to column medians, then 20%
        iv = iv.fillna(method="bfill").fillna(iv.median()).fillna(0.20)
    return iv



# -----------------------------
# Data structures
# -----------------------------
@dataclass
class OptionRow:
    """Plain container for one option in the portfolio.

    S: spot price at t0
    X: strike
    T: time to expiry in years at t0
    r: risk-free rate (cont. comp.)
    q: dividend yield (cont. comp.)
    sigma: implied vol at t0
    option_type: "call" or "put"
    qty: signed quantity (positive = long, negative = short)
    """
    S: float
    X: float
    T: float
    r: float
    q: float
    sigma: float
    option_type: str
    qty: float


# -----------------------------
# Core P&L helpers
# -----------------------------

def option_pnl_full_reval(
    S0: float,
    X: float,
    T: float,
    r: float,
    q: float,
    sigma0: float,
    S1: float,
    sigma1: float,
    option_type: str = "call",
    h: float = 1.0 / 252.0,
) -> float:
    """Full revaluation P&L for one option leg over horizon h years.

    Prices option at t0 with (S0, sigma0, T) and at t1 with (S1, sigma1, T-h).
    Returns ΔV = V1 - V0 (positive if option value increased).
    """
    T1 = max(T - h, np.finfo(float).eps)
    V0 = bsm(S0, X, T, r, sigma0, option_type=option_type, q=q)
    V1 = bsm(S1, X, T1, r, sigma1, option_type=option_type, q=q)
    return float(V1 - V0)


def portfolio_losses_from_scenarios(
    positions: Iterable[OptionRow] | pd.DataFrame,
    scenarios: pd.DataFrame,
    h: float = 1.0 / 252.0,
) -> np.ndarray:
    """Compute portfolio loss L for each scenario via full revaluation.

    Parameters
    ----------
    positions : iterable of OptionRow or DataFrame
        If DataFrame, must have columns: [S, X, T, r, q, sigma, option_type, qty].
    scenarios : DataFrame
        Rows = scenarios (k). Columns can include multi-asset shocks:
        - If a single underlying: columns ['dS', 'dsigma'].
        - If multiple underlyings: use a MultiIndex or columns like 'dS:SYMBOL', 'dsigma:SYMBOL'.
        For each option we will pick the appropriate columns by the optional 'symbol' column in positions
        if present (e.g., positions['symbol'] = 'AAPL' implies using 'dS:AAPL', 'dsigma:AAPL').
    h : float
        Horizon in years (default 1/252).

    Returns
    -------
    np.ndarray
        Vector of losses L^(k) (positive = loss), one per scenario row.
    """
    # Convert positions to DataFrame if needed
    if not isinstance(positions, pd.DataFrame):
        pos_df = pd.DataFrame([p.__dict__ for p in positions])
    else:
        pos_df = positions.copy()

    # Optional symbol column indicates which scenario columns to use
    has_symbol = 'symbol' in pos_df.columns

    # Precompute t0 values for speed
    V0 = pos_df.apply(
        lambda row: bsm(
            row['S'], row['X'], row['T'], row['r'], row['sigma'],
            option_type=row['option_type'], q=row.get('q', 0.0)
        ), axis=1
    ).to_numpy()

    qty = pos_df['qty'].to_numpy(dtype=float)
    S0 = pos_df['S'].to_numpy(dtype=float)
    X = pos_df['X'].to_numpy(dtype=float)
    T = pos_df['T'].to_numpy(dtype=float)
    r = pos_df['r'].to_numpy(dtype=float)
    q = pos_df.get('q', pd.Series(0.0, index=pos_df.index)).to_numpy(dtype=float)
    sigma0 = pos_df['sigma'].to_numpy(dtype=float)
    types = pos_df['option_type'].to_numpy()

    losses: List[float] = []

    for _, sc in scenarios.iterrows():
        # Extract shocks per position
        if has_symbol:
            dS = np.array([sc.get(f'dS:{sym}', 0.0) for sym in pos_df['symbol']])
            dsig = np.array([sc.get(f'dsigma:{sym}', 0.0) for sym in pos_df['symbol']])
        else:
            dS = np.full(len(pos_df), sc.get('dS', 0.0))
            dsig = np.full(len(pos_df), sc.get('dsigma', 0.0))

        S1 = S0 + dS
        sigma1 = np.clip(sigma0 + dsig, 1e-6, None)
        T1 = np.maximum(T - h, np.finfo(float).eps)

        # Price at t1
        V1 = np.array([
            bsm(S1[i], X[i], T1[i], r[i], sigma1[i], option_type=types[i], q=q[i])
            for i in range(len(pos_df))
        ])

        dV = V1 - V0
        dP = np.sum(qty * dV)  # portfolio change in value
        L = -float(dP)         # define loss as negative P&L
        losses.append(L)

    return np.asarray(losses, dtype=float)


# -----------------------------
# Tail metrics (historical estimators)
# -----------------------------

def var_es_from_losses(losses: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    """Empirical VaR and ES at level alpha from a vector of losses.

    Returns positive numbers (tail loss).
    """
    if not isinstance(losses, np.ndarray):
        losses = np.asarray(losses, dtype=float)
    losses = np.sort(losses)  # ascending
    K = losses.shape[0]
    j = int(np.ceil(alpha * K))
    j = max(1, min(j, K))
    var_val = float(losses[j - 1])
    es_val = float(losses[j - 1 :].mean())
    return var_val, es_val



# -----------------------------
# Scenario Engine (automatic)
# -----------------------------


def _ensure_df(x: pd.DataFrame | Dict[str, pd.Series] | None, name: str) -> pd.DataFrame:
    """Normalize input to a DataFrame with sorted DateTimeIndex.
    Accepts a DataFrame (columns = symbols) or a dict of Series.
    """
    if x is None:
        raise ValueError(f"{name} history is required")
    if isinstance(x, dict):
        df = pd.DataFrame(x)
    else:
        df = x.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(f"{name} must have a DateTimeIndex")
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    return df


def build_scenarios_hr(
    prices: pd.DataFrame | Dict[str, pd.Series],
    iv: Optional[pd.DataFrame | Dict[str, pd.Series]],
    lookback_days: int = 750,
    symbols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Historical Replay scenarios (nonparametric).

    Returns a DataFrame whose rows are days and columns are shocks per symbol:
    dS:SYMBOL (absolute price change) and dsigma:SYMBOL (absolute IV change, decimal units).
    If iv is None, a rolling realized-vol proxy is used.
    """
    px = _ensure_df(prices, "prices").ffill()
    ivdf = _iv_proxy(prices) if iv is None else _ensure_df(iv, "iv").ffill()
    if symbols is None:
        symbols = [c for c in px.columns if c in ivdf.columns]
    symbols = list(symbols)

    # Align and compute daily diffs
    common = px.index.intersection(ivdf.index)
    px, ivdf = px.loc[common, symbols], ivdf.loc[common, symbols]
    dS = px.diff().dropna()
    dSig = ivdf.diff().dropna()

    # Limit to lookback window
    if lookback_days is not None and lookback_days > 0:
        dS = dS.iloc[-lookback_days:]
        dSig = dSig.iloc[-lookback_days:]

    # Build scenario table
    cols = {}
    for s in symbols:
        cols[f"dS:{s}"] = dS[s]
        cols[f"dsigma:{s}"] = dSig[s]
    sc = pd.DataFrame(cols, index=dS.index)
    sc = sc.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    sc.reset_index(drop=True, inplace=True)
    return sc


def _ewma_std(x: pd.Series, lam: float = 0.94) -> pd.Series:
    """EWMA standard deviation (RiskMetrics)."""
    v = x.pow(2).ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(v)


def build_scenarios_fhs(
    prices: pd.DataFrame | Dict[str, pd.Series],
    iv: Optional[pd.DataFrame | Dict[str, pd.Series]],
    lam_price: float = 0.94,
    lam_iv: float = 0.97,
    lookback_days: int = 750,
    symbols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Filtered Historical Simulation: rescale historical shocks to today's volatility level.

    For each symbol, scale historical dS_t by (sigma_today / sigma_t) using EWMA stdev; similarly for dsigma.
    If iv is None, a rolling realized-vol proxy is used.
    """
    px = _ensure_df(prices, "prices").ffill()
    ivdf = _iv_proxy(prices) if iv is None else _ensure_df(iv, "iv").ffill()
    if symbols is None:
        symbols = [c for c in px.columns if c in ivdf.columns]
    symbols = list(symbols)

    common = px.index.intersection(ivdf.index)
    px, ivdf = px.loc[common, symbols], ivdf.loc[common, symbols]
    dS = px.diff().dropna()
    dSig = ivdf.diff().dropna()

    if lookback_days is not None and lookback_days > 0:
        dS = dS.iloc[-lookback_days:]
        dSig = dSig.iloc[-lookback_days:]

    # EWMA vols
    vol_price = dS.apply(lambda s: _ewma_std(s, lam_price))
    vol_iv = dSig.apply(lambda s: _ewma_std(s, lam_iv))
    scale_price = vol_price.iloc[-1] / vol_price.replace(0.0, np.nan)
    scale_iv = vol_iv.iloc[-1] / vol_iv.replace(0.0, np.nan)

    dS_scaled = dS.multiply(scale_price, axis=1)
    dSig_scaled = dSig.multiply(scale_iv, axis=1)

    cols = {}
    for s in symbols:
        cols[f"dS:{s}"] = dS_scaled[s]
        cols[f"dsigma:{s}"] = dSig_scaled[s]
    sc = pd.DataFrame(cols, index=dS.index)
    sc = sc.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    sc.reset_index(drop=True, inplace=True)
    return sc


def build_scenarios_mc(
    prices: pd.DataFrame | Dict[str, pd.Series],
    iv: Optional[pd.DataFrame | Dict[str, pd.Series]],
    n_paths: int = 50000,
    lookback_days: int = 750,
    symbols: Iterable[str] | None = None,
    random_state: int | None = 42,
) -> pd.DataFrame:
    """Gaussian Monte Carlo scenarios preserving empirical covariance across symbols and between dS and dsigma.
    If iv is None, a rolling realized-vol proxy is used.
    """
    rng = np.random.default_rng(random_state)
    px = _ensure_df(prices, "prices").ffill()
    ivdf = _iv_proxy(prices) if iv is None else _ensure_df(iv, "iv").ffill()
    if symbols is None:
        symbols = [c for c in px.columns if c in ivdf.columns]
    symbols = list(symbols)

    common = px.index.intersection(ivdf.index)
    px, ivdf = px.loc[common, symbols], ivdf.loc[common, symbols]
    dS = px.diff().dropna()
    dSig = ivdf.diff().dropna()

    if lookback_days is not None and lookback_days > 0:
        dS = dS.iloc[-lookback_days:]
        dSig = dSig.iloc[-lookback_days:]

    # Build joint matrix [dS | dsigma]
    X = pd.concat([dS, dSig], axis=1, keys=["dS", "dsigma"]).dropna(how="any")
    Xm = X.values
    mu = np.zeros(Xm.shape[1])
    cov = np.cov(Xm, rowvar=False)
    # Ensure PSD
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        # Simple near-PSD fix: clip negative eigenvalues
        w, V = np.linalg.eigh(cov)
        w = np.clip(w, 1e-10, None)
        cov = (V * w) @ V.T
        L = np.linalg.cholesky(cov)

    Z = rng.standard_normal(size=(n_paths, Xm.shape[1]))
    draws = Z @ L.T  # mean zero

    # Build DataFrame with proper column names
    cols = []
    for s in symbols:
        cols.append(f"dS:{s}")
    for s in symbols:
        cols.append(f"dsigma:{s}")
    sc = pd.DataFrame(draws, columns=cols)
    return sc


def build_scenarios_stress(
    symbols: Iterable[str],
    s_steps: Iterable[float] | None = None,
    v_steps: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Deterministic stress lattice.

    s_steps are absolute price changes (e.g., -0.10*S is not used here; this is absolute, so pass -5.0 for a $5 drop).
    v_steps are absolute IV changes in decimal units (0.05 = +5 vol pts).
    The same shock is applied to all symbols per row for conservatism.
    """
    s_steps = list(s_steps) if s_steps is not None else [-0.10, -0.075, -0.05, -0.03, 0.0, 0.03, 0.05]
    v_steps = list(v_steps) if v_steps is not None else [0.10, 0.05, 0.00, -0.05]

    # Interpret s_steps as *relative* returns if magnitudes are small (|x|<1), convert to absolute dS later per symbol
    rel = all(abs(x) <= 1.0 for x in s_steps)

    rows = []
    for ds_rel, dv in product(s_steps, v_steps):
        row = {}
        for sym in symbols:
            keyS = f"dS:{sym}"
            keyV = f"dsigma:{sym}"
            row[keyS] = ds_rel  # will be converted by caller if needed
            row[keyV] = dv
        rows.append(row)
    sc = pd.DataFrame(rows)
    return sc


def stack_scenarios(*dfs: pd.DataFrame, weights: Iterable[float] | None = None) -> pd.DataFrame:
    """Concatenate multiple scenario tables; optional weights can be carried via a column 'w'."""
    out = []
    if weights is None:
        for d in dfs:
            if d is None or len(d) == 0:
                continue
            tmp = d.copy()
            out.append(tmp)
    else:
        for d, w in zip(dfs, weights):
            if d is None or len(d) == 0:
                continue
            tmp = d.copy()
            tmp["w"] = float(w)
            out.append(tmp)
    if not out:
        return pd.DataFrame()
    sc = pd.concat(out, axis=0, ignore_index=True)
    return sc.replace([np.inf, -np.inf], np.nan).dropna(how="any")


# -----------------------------
# Orchestration: compute ES from positions + histories
# -----------------------------

def compute_portfolio_es(
    positions: Iterable[OptionRow] | pd.DataFrame,
    prices: pd.DataFrame | Dict[str, pd.Series],
    iv: Optional[pd.DataFrame | Dict[str, pd.Series]],
    alpha: float = 0.05,
    h: float = 1.0 / 252.0,
    mode: str = "HR+FHS+MC+STRESS",
    n_paths_mc: int = 50000,
    lookback_days: int = 750,
    fhs_lambda_price: float = 0.94,
    fhs_lambda_iv: float = 0.97,
) -> Tuple[float, float, np.ndarray]:
    """End-to-end ES: builds scenarios automatically and computes VaR/ES via full revaluation.

    Returns (VaR, ES, losses).
    """
    # Determine universe symbols from positions if available
    if isinstance(positions, pd.DataFrame):
        syms = positions['symbol'].unique().tolist() if 'symbol' in positions.columns else None
    else:
        arr = [p.__dict__ for p in positions]
        syms = list({row.get('symbol') for row in arr if 'symbol' in row and row.get('symbol') is not None}) or None

    hr = fhs = mc = stress = None
    modes = set(m.strip().upper() for m in mode.split('+'))

    if 'HR' in modes:
        hr = build_scenarios_hr(prices, iv, lookback_days=lookback_days, symbols=syms)
    if 'FHS' in modes:
        fhs = build_scenarios_fhs(prices, iv, lam_price=fhs_lambda_price, lam_iv=fhs_lambda_iv, lookback_days=lookback_days, symbols=syms)
    if 'MC' in modes:
        mc = build_scenarios_mc(prices, iv, n_paths=n_paths_mc, lookback_days=lookback_days, symbols=syms)
    if 'STRESS' in modes:
        if syms is None:
            # try to infer from price columns
            px = _ensure_df(prices, "prices")
            syms = list(px.columns)
        stress = build_scenarios_stress(syms)

    sc = stack_scenarios(*(x for x in [hr, fhs, mc, stress] if x is not None))

    # If stress grid used relative returns for dS, convert to absolute using latest S for each symbol
    if len(sc) > 0:
        px = _ensure_df(prices, "prices").ffill()
        lastS = px.iloc[-1]
        for c in sc.columns:
            if c.startswith('dS:'):
                sym = c.split(':', 1)[1]
                if abs(sc[c]).max() <= 1.0:  # treat as relative return
                    sc[c] = sc[c] * float(lastS[sym])

    losses = portfolio_losses_from_scenarios(positions, sc, h=h)
    var_val, es_val = var_es_from_losses(losses, alpha=alpha)
    return var_val, es_val, losses



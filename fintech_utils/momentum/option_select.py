# -----------------------------
# Delta-to-strike inversion (BSM)
# -----------------------------
from scipy.stats import norm
from typing import Tuple, Iterable, Optional, Dict, Any
import numpy as np
import pandas as pd
from datetime import datetime, date

from .config import MomentumConfig


def bsm_strike_for_delta(
    S: float,
    sigma: float,
    T: float,
    r: float,
    q: float,
    option_type: str,
    delta_target: float,
) -> float:
    """Closed-form inversion: find strike K that yields the target Delta in BSM.

    For calls:  Delta = exp(-q T) * N(d1)      in (0, exp(-qT))
    For puts:   Delta = exp(-q T) * (N(d1) - 1) in (-exp(-qT), 0)

    Solve for d1, then K from d1 definition.
    """
    if T <= 0 or sigma <= 0 or S <= 0:
        raise ValueError("S, sigma, T must be positive")

    etq = np.exp(-q * T)
    if option_type == 'call':
        x = delta_target / etq
        x = np.clip(x, 1e-8, 1 - 1e-8)
        d1 = norm.ppf(x)
    elif option_type == 'put':
        x = delta_target / etq + 1.0  # since Delta_put = etq*(N(d1)-1)
        x = np.clip(x, 1e-8, 1 - 1e-8)
        d1 = norm.ppf(x)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    drift = (r - q + 0.5 * sigma**2) * T
    K = S * np.exp(-(sigma * np.sqrt(T)) * d1 + drift)
    return float(K)


def bsm_strike_band_for_delta(
    S: float,
    sigma: float,
    T: float,
    r: float,
    q: float,
    option_type: str,
    delta_band: Tuple[float, float],
) -> Tuple[float, float, float]:
    """Return (K_low, K_star, K_high) for a desired |Delta| band.

    For puts, pass negative deltas; for calls, positive deltas.
    """
    d_lo, d_hi = delta_band
    if option_type == 'put' and (d_hi > 0 or d_lo > 0):
        # ensure negatives for puts
        d_lo, d_hi = -abs(d_lo), -abs(d_hi)
    if option_type == 'call' and (d_hi < 0 or d_lo < 0):
        d_lo, d_hi = abs(d_lo), abs(d_hi)

    K_lo = bsm_strike_for_delta(S, sigma, T, r, q, option_type, d_lo)
    K_star = bsm_strike_for_delta(S, sigma, T, r, q, option_type, 0.5 * (d_lo + d_hi))
    K_hi = bsm_strike_for_delta(S, sigma, T, r, q, option_type, d_hi)
    return (K_lo, K_star, K_hi)


# -----------------------------
# Option selection with IV filter
# -----------------------------

class OptionSelector:
    """Select options to trade given delta band, DTM band, and an IV filter.

    Expected option chain schema (flexible, case-insensitive columns):
      - symbol (str)
      - expiry (datetime.date or str) or T (float, years)
      - dtm (int, days)   [optional; computed from expiry if not present]
      - option_type ("call"/"put" or "C"/"P")
      - strike (float)
      - iv (float, decimal)
      - bid, ask, mid [optional]
      - delta [optional]; if missing, we compute with BSM using (S, iv, T, r, q)

    IV history: pandas.Series indexed by date with current value at the end.
    """

    def __init__(self, config: MomentumConfig):
        self.cfg = config

    # ---------- public API ----------
    def select_contract(
        self,
        symbol: str,
        spot: float,
        chain: pd.DataFrame,
        iv_history: pd.Series,
        option_side: str,
        r: float,
        q: float,
        asof: Optional[pd.Timestamp] = None,
    ) -> Optional[Dict[str, Any]]:
        """Return the best contract dict or None if nothing passes filters.

        Parameters
        ----------
        symbol : str
            Underlying ticker.
        spot : float
            Current underlying price S.
        chain : pd.DataFrame
            Option chain for the symbol as of `asof`.
        iv_history : pd.Series
            Time series of (e.g.) ATM IV for the tenor close to target DTM.
        option_side : str
            "put" for long-momentum (we sell puts), "call" for short-momentum (we sell calls).
        r, q : float
            Risk-free rate, dividend yield (cont. comp.).
        asof : pd.Timestamp | None
            As-of date; needed only if `expiry` needs parsing for DTM.
        """
        ch = self._normalize_chain(chain)

        # 2) DTM window selection (nearest to target within band)
        ch = self._attach_time_columns(ch, asof)
        dtm_lo, dtm_hi = self.cfg.dtm_range
        target = int(round(np.mean(self.cfg.dtm_range))) if getattr(self.cfg, 'target_dtm', None) is None else int(self.cfg.target_dtm)
        eligible = ch[(ch['dtm'] >= dtm_lo) & (ch['dtm'] <= dtm_hi) & (ch['option_type'] == option_side.lower())]
        if eligible.empty:
            return None
        # pick expiry closest to target
        sel_expiry = eligible.iloc[(eligible['dtm'] - target).abs().argsort()].iloc[0]['expiry']
        slice_exp = eligible[eligible['expiry'] == sel_expiry].copy()

        # IV filter (history optional): use chain IV median for the chosen expiry if no history
        iv_now = None
        if 'iv' in slice_exp.columns and not slice_exp['iv'].dropna().empty:
            iv_now = float(slice_exp['iv'].median())
        if not self._passes_iv_filter(iv_history, iv_now):
            return None

        # 3) Delta band selection
        delta_lo, delta_hi = self.cfg.delta_range
        # ensure sign convention
        if option_side.lower() == 'put':
            d_band = (-abs(delta_hi), -abs(delta_lo))  # e.g., (-0.40, -0.25)
        else:
            d_band = (abs(delta_lo), abs(delta_hi))    # e.g., (0.25, 0.40)

        # compute delta if missing
        if 'delta' not in slice_exp.columns or slice_exp['delta'].isna().all():
            # need T (years) and iv per row
            if 'T' not in slice_exp.columns:
                # compute from dtm
                slice_exp['T'] = slice_exp['dtm'] / 365.0
            if 'iv' not in slice_exp.columns:
                raise ValueError("option chain must have 'iv' column or provide precomputed delta")
            deltas = self._bsm_delta_vec(spot, slice_exp['strike'].to_numpy(float), slice_exp['T'].to_numpy(float), r, q, slice_exp['iv'].to_numpy(float), option_side)
            slice_exp['delta'] = deltas

        # filter by delta band and liquidity gates if present
        band_mask = (slice_exp['delta'] >= min(d_band)) & (slice_exp['delta'] <= max(d_band))
        slice_exp = slice_exp[band_mask]
        if slice_exp.empty:
            # if no strikes inside band, choose closest to midpoint
            target_delta = 0.5 * sum(d_band)
            closest = (slice_exp['delta'] - target_delta).abs().argsort() if not slice_exp.empty else None
            if closest is None or len(closest) == 0:
                # fall back to computing target strike and selecting nearest available
                T_row = float(slice_exp['T'].iloc[0]) if 'T' in slice_exp.columns and len(slice_exp) > 0 else (target / 365.0)
                iv_star = float(slice_exp['iv'].median()) if 'iv' in slice_exp.columns and len(slice_exp) > 0 else float(iv_history.iloc[-1])
                K_target = bsm_strike_for_delta(spot, iv_star, T_row, r, q, option_side, target_delta)
                # choose nearest strike from full eligible set
                full = eligible[eligible['expiry'] == sel_expiry].copy()
                full['dist'] = (full['strike'] - K_target).abs()
                if full.empty:
                    return None
                row = full.iloc[full['dist'].argsort().iloc[0]].drop(labels=['dist'])
                return self._row_to_dict(symbol, row)

        # Liquidity filters if present
        liq_mask = pd.Series(True, index=slice_exp.index)
        for col, thr in [('oi', 1000), ('volume', 100), ('bid', None), ('ask', None)]:
            if col in slice_exp.columns:
                if col in ('oi', 'volume'):
                    liq_mask &= slice_exp[col].fillna(0) >= thr
        slice_exp = slice_exp[liq_mask]
        if slice_exp.empty:
            return None

        # choose by delta closeness to band midpoint and then tightest spread (if available)
        midpoint = 0.5 * sum(d_band)
        slice_exp['delta_dist'] = (slice_exp['delta'] - midpoint).abs()
        order_idx = slice_exp['delta_dist'].argsort(kind='mergesort')
        slice_exp = slice_exp.loc[order_idx]
        if 'bid' in slice_exp.columns and 'ask' in slice_exp.columns:
            slice_exp['spread'] = (slice_exp['ask'] - slice_exp['bid']).abs()
            slice_exp = slice_exp.sort_values(['delta_dist', 'spread'])

        best = slice_exp.iloc[0]
        return self._row_to_dict(symbol, best)

    # ---------- helpers ----------
    @staticmethod
    def _normalize_chain(chain: pd.DataFrame) -> pd.DataFrame:
        ch = chain.copy()
        # normalize column names
        ch.columns = [str(c).strip().lower() for c in ch.columns]
        # map alt names
        rename = {}
        if 'type' in ch.columns and 'option_type' not in ch.columns:
            rename['type'] = 'option_type'
        if 'strike_price' in ch.columns and 'strike' not in ch.columns:
            rename['strike_price'] = 'strike'
        ch = ch.rename(columns=rename)
        # enforce required columns
        req = {'option_type', 'strike'}
        missing = req - set(ch.columns)
        if missing:
            raise ValueError(f"option chain is missing required columns: {missing}")
        # normalize option_type
        ch['option_type'] = ch['option_type'].str.lower().map({'c': 'call', 'call': 'call', 'p': 'put', 'put': 'put'})
        return ch

    @staticmethod
    def _attach_time_columns(ch: pd.DataFrame, asof: Optional[pd.Timestamp]) -> pd.DataFrame:
        out = ch.copy()
        if 'dtm' not in out.columns:
            if 'expiry' in out.columns:
                # parse expiry to date
                if not np.issubdtype(out['expiry'].dtype, np.datetime64):
                    out['expiry'] = pd.to_datetime(out['expiry']).dt.tz_localize(None)
                base = pd.Timestamp(asof).normalize() if asof is not None else pd.Timestamp(datetime.utcnow().date())
                out['dtm'] = (out['expiry'].dt.normalize() - base).dt.days.clip(lower=0)
            elif 't' in out.columns:
                out['dtm'] = (out['t'] * 365.0).round().astype(int)
            else:
                raise ValueError("need either 'dtm', 'expiry', or 'T' in option chain to determine days to maturity")
        if 'expiry' not in out.columns:
            # synthesize expiry as asof + dtm
            base = pd.Timestamp(asof).normalize() if asof is not None else pd.Timestamp(datetime.utcnow().date())
            out['expiry'] = base + pd.to_timedelta(out['dtm'], unit='D')
        return out

    @staticmethod
    def _bsm_delta_vec(S: float, K: np.ndarray, T: np.ndarray, r: float, q: float, sigma: np.ndarray, option_type: str) -> np.ndarray:
        # N(d1) with d1 = [ln(S/K) + (r - q + 0.5 sigma^2) T] / (sigma sqrt(T))
        with np.errstate(divide='ignore', invalid='ignore'):
            d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        Nd1 = norm.cdf(d1)
        etq = np.exp(-q * T)
        if option_type == 'call':
            delta = etq * Nd1
        else:
            delta = etq * (Nd1 - 1.0)
        return delta

    def _passes_iv_filter(self, iv_hist: Optional[pd.Series], iv_now: Optional[float]) -> bool:
        """IV filter with two modes:
        - Percentile mode: require current IV percentile >= cfg.iv_pctile_min over iv_window.
        - Absolute mode: require iv_now >= cfg.iv_abs_min.
        If cfg.iv_filter_mode == 'auto', use percentile when history is usable, else absolute.
        """
        mode = str(getattr(self.cfg, 'iv_filter_mode', 'auto')).lower()
        iv_abs_min = float(getattr(self.cfg, 'iv_abs_min', 0.20))
        iv_pctile_min = float(getattr(self.cfg, 'iv_pctile_min', 0.5))
        W = int(getattr(self.cfg, 'iv_window', 252))

        # Decide mode
        use_percentile = False
        if mode == 'percentile':
            use_percentile = True
        elif mode == 'auto':
            if isinstance(iv_hist, pd.Series) and not iv_hist.dropna().empty and len(iv_hist.dropna()) >= max(30, W // 4):
                use_percentile = True

        if use_percentile:
            if not isinstance(iv_hist, pd.Series):
                return False
            hist = iv_hist.dropna()
            if hist.empty:
                return False
            hist = hist.iloc[-W:]
            now = float(hist.iloc[-1])
            pct = (hist <= now).mean()
            return pct >= iv_pctile_min
        else:
            # absolute threshold mode
            if iv_now is None or not np.isfinite(iv_now):
                return False
            return iv_now >= iv_abs_min

    @staticmethod
    def _row_to_dict(symbol: str, row: pd.Series) -> Dict[str, Any]:
        out = {
            'symbol': symbol,
            'option_type': row['option_type'],
            'strike': float(row['strike']),
            'expiry': pd.to_datetime(row['expiry']).date() if 'expiry' in row else None,
            'dtm': int(row['dtm']) if 'dtm' in row else None,
            'iv': float(row['iv']) if 'iv' in row and pd.notna(row['iv']) else None,
        }
        for k in ('bid', 'ask', 'mid', 'delta', 'oi', 'volume'):
            if k in row and pd.notna(row[k]):
                out[k] = float(row[k]) if k != 'option_type' else row[k]
        return out
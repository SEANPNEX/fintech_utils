from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import asyncio
from math import sqrt, exp
from scipy.stats import norm

from .config import MomentumConfig
from .risk import compute_portfolio_es
from .signal import MomentumSignal

# --------------------------- Canonical position schema ---------------------------
CANONICAL_POSITION_COLS = [
    'symbol','S','X','T','r','q','sigma','option_type','qty','dtm','multiplier'
]

def _order_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with canonical columns first (preserve extras), creating missing with sensible defaults."""
    defaults = {
        'q': 0.0,
        'multiplier': 1.0,
    }
    out = df.copy()
    for c in CANONICAL_POSITION_COLS:
        if c not in out.columns:
            out[c] = defaults.get(c, np.nan)
    # stable order: canonical first, then the rest
    other = [c for c in out.columns if c not in CANONICAL_POSITION_COLS]
    return out[CANONICAL_POSITION_COLS + other]

# -------- helpers used by position control --------

def _smooth_scale(es_val: float, es_target: float, gamma: float = 0.5, eps: float = 1e-9) -> float:
    """Smooth, monotone scaling in (0,1], decreasing in ES.

    w = min(1, (es_target / max(es_val, eps))**gamma)
    - If ES is well below target -> w ≈ 1
    - If ES equals target       -> w = 1
    - If ES exceeds target      -> w in (0,1)
    gamma in (0,1] controls curvature (0.5 = square-root).
    """
    if es_val is None or not np.isfinite(es_val) or es_val <= 0:
        return 1.0
    if es_target is None or not np.isfinite(es_target) or es_target <= 0:
        return 1.0
    w = (es_target / max(es_val, eps)) ** float(gamma)
    return float(max(0.0, min(1.0, w)))


def _discrete_direction(signal: float, neutral_band: float = 0.25, contrarian: bool = True) -> int:
    """Map a scalar signal to {-1,0,1} with a dead-zone (neutral band).

    contrarian=True implements long losers / short winners:
      dir = -sign(signal) outside the neutral band; else 0.
    """
    if signal is None or not np.isfinite(signal):
        return 0
    if abs(signal) < neutral_band:
        return 0
    sgn = -np.sign(signal) if contrarian else np.sign(signal)
    return int(np.clip(sgn, -1, 1))

# =============================================================
# BSM Greek helpers (closed-form)
# =============================================================
from math import log, sqrt, exp
from ..findata.options import bsm_greeks
from scipy.stats import norm


# =============================================================
# Monitor
# =============================================================

@dataclass
class Monitor:
    cfg: MomentumConfig

    # --------------------------- ES Nowcast ---------------------------
    def nowcast_es(
        self,
        positions: pd.DataFrame,
        prices: Union[pd.DataFrame, Dict[str, pd.Series]],
        iv: Union[pd.DataFrame, Dict[str, pd.Series]],
        alpha: Optional[float] = None,
        mode: str = "HR+FHS+MC",
        n_paths_mc: int = 30000,
    ) -> Dict[str, object]:
        """Compute VaR/ES for the current book via automatic scenarios.

        Parameters
        ----------
        positions : DataFrame
            Must contain columns: ['symbol','S','X','T','r','q','sigma','option_type','qty','dtm'].
        prices, iv : DataFrames or dicts of Series
            Market histories used by the scenario engine (see risk.compute_portfolio_es).
        alpha : float, optional
            ES/VaR confidence (default = cfg.es_alpha if provided else 0.05).
        mode : str
            Scenario modes combined by '+', default HR+FHS+MC.
        n_paths_mc : int
            Monte-Carlo path count for MC component.
        """
        a = float(alpha if alpha is not None else getattr(self.cfg, "es_alpha", 0.05))
        lookback_days = int(getattr(self.cfg, "iv_window", 252))
        VaR, ES, losses = compute_portfolio_es(
            positions=positions,
            prices=prices,
            iv=iv,
            alpha=a,
            mode=mode,
            n_paths_mc=n_paths_mc,
            lookback_days=lookback_days,
        )
        budget = float(getattr(self.cfg, "es_budget", np.nan))
        breach = bool(np.isfinite(budget) and ES > budget)
        return {
            "VaR": float(VaR),
            "ES": float(ES),
            "ES_budget": budget,
            "ES_breach": breach,
            "n_scenarios": int(len(losses)),
            "losses": losses,
        }

    # --------------------------- Greeks Watch ---------------------------
    def portfolio_greeks(self, positions: pd.DataFrame) -> Dict[str, float]:
        """Aggregate portfolio Greeks using BSM formulas.

        Expects columns: ['S','X','T','r','q','sigma','option_type','qty'].
        Returns Delta$, Gamma$, Vega$, Theta$ (scaled by S for $, except Theta which is raw per-year cash Greek).
        """
        req = {'S','X','T','r','q','sigma','option_type','qty'}
        miss = req - set(map(str, positions.columns))
        if miss:
            raise ValueError(f"positions missing columns: {miss}")

        totals = dict(Delta=0.0, Gamma=0.0, Vega=0.0, Theta=0.0, Delta_usd=0.0, Gamma_usd=0.0, Vega_usd=0.0)
        for _, row in positions.iterrows():
            S = float(row['S']); X = float(row['X']); T = max(float(row['T']), 1e-8)
            r = float(row['r']); q = float(row.get('q', 0.0)); sigma = max(float(row['sigma']), 1e-6)
            typ = str(row['option_type']).lower(); qty = float(row['qty'])
            greeks = bsm_greeks(S, X, T, r, q = q, sigma=sigma, option_type=typ)
            d = greeks.delta()
            g = greeks.gamma()
            v = greeks.vega()
            th = greeks.theta()
            if not np.isfinite(d):
                continue
            totals['Delta'] += qty * d
            totals['Gamma'] += qty * g
            totals['Vega']  += qty * v
            totals['Theta'] += qty * th
            # Dollar Greeks
            totals['Delta_usd'] += qty * d * S
            totals['Gamma_usd'] += qty * g * S * S
            totals['Vega_usd']  += qty * v
        return {k: float(v) for k, v in totals.items()}

    # --------------------------- Roll Rules ---------------------------
    def roll_list(self, positions: pd.DataFrame) -> pd.DataFrame:
        """Return positions that should be rolled based on DTM < cfg.roll_min_dtm."""
        if 'dtm' not in positions.columns:
            raise ValueError("positions must have 'dtm' column (days to maturity)")
        min_dtm = int(getattr(self.cfg, 'roll_min_dtm', 10))
        return positions.loc[positions['dtm'] < min_dtm].copy()

    # --------------------------- Exit Rules ---------------------------
    def exit_on_signal(
        self,
        positions: pd.DataFrame,
        signals: pd.Series,
    ) -> pd.DataFrame:
        """Suggest exits based on signal weakening.

        Parameters
        ----------
        positions : DataFrame with columns ['symbol','option_type', ...]
        signals : pd.Series mapping symbol -> z-like score (positive = bullish, negative = bearish)

        Thresholds are read from config: cfg.neutral_band (float).
        """
        nb = float(getattr(self.cfg, 'neutral_band', 0.25))
        if 'symbol' not in positions.columns:
            raise ValueError("positions must have 'symbol' column for signal-based exits")
        side = positions['option_type'].str.lower().map({'call': 'short', 'put': 'long'})
        df = positions.copy()
        df['signal'] = df['symbol'].map(signals)
        df['exit_flag'] = False
        # Long-momentum legs are implemented with PUT shorts; exit if signal > -nb (contrarian)
        df.loc[(df['option_type'].str.lower() == 'put') & (df['signal'] > -nb), 'exit_flag'] = True
        # Short-momentum legs are implemented with CALL shorts; exit if signal < nb (contrarian)
        df.loc[(df['option_type'].str.lower() == 'call') & (df['signal'] < nb), 'exit_flag'] = True
        return df[df['exit_flag']].copy()


    # --------------------------- Position Control ---------------------------
    def position_control(
        self,
        positions: pd.DataFrame,
        prices: Union[pd.DataFrame, Dict[str, pd.Series]],
        iv: Union[pd.DataFrame, Dict[str, pd.Series]],
    ) -> pd.DataFrame:
        """Compute **position differences** given ES budget and internally computed signals.

        Signals are computed internally via MomentumSignal based on provided prices.

        All thresholds and switches are taken from cfg: es_alpha, es_target, neutral_band, contrarian, scale_gamma, base_qty, per_symbol_base, contract_mult_col, round_to_int.
        """
        nb = float(getattr(self.cfg, 'neutral_band', 0.25))
        contrarian = bool(getattr(self.cfg, 'contrarian', True))
        gamma = float(getattr(self.cfg, 'scale_gamma', 0.5))
        base_qty = float(getattr(self.cfg, 'base_qty', 1.0))
        per_symbol_base = getattr(self.cfg, 'per_symbol_base', None)
        contract_mult_col = str(getattr(self.cfg, 'contract_mult_col', 'multiplier'))
        round_to_int = bool(getattr(self.cfg, 'round_to_int', True))
        a = float(getattr(self.cfg, 'es_alpha', 0.05))

        # Compute per-symbol signals using MomentumSignal only (no math here)
        ms = MomentumSignal(self.cfg)
        pos_syms = positions['symbol'].astype(str).unique().tolist()
        price_df = prices if isinstance(prices, pd.DataFrame) else pd.DataFrame(prices)

        # Compute per-symbol signals via signal.py (single source of truth)
        signals = ms.relative_signal(price_df[pos_syms].dropna(axis=1, how='all'))
        signals = signals.reindex(pos_syms).fillna(0.0)

        # 1) Exit suggestions based on contrarian hysteresis
        exits_df = self.exit_on_signal(positions, signals)
        exit_syms = set(exits_df['symbol']) if not exits_df.empty else set()

        # 2) ES nowcast for whole book and smooth scale vs ES target
        es_out = self.nowcast_es(positions, prices, iv, alpha=a)
        es_val = float(es_out.get('ES', np.nan))
        es_target = float(getattr(self.cfg, 'es_target', np.nan))
        scale = _smooth_scale(es_val, es_target, gamma=gamma)

        # 3) Build discrete directions per symbol
        dir_map: Dict[str, int] = {}
        for sym, sig in signals.items():
            dir_map[str(sym)] = _discrete_direction(float(sig), neutral_band=nb, contrarian=contrarian)

        # 4) Determine base size per symbol
        def base_for(sym: str) -> float:
            if per_symbol_base and sym in per_symbol_base:
                return float(per_symbol_base[sym])
            return float(base_qty)

        # 5) Assemble targets and differences
        rows = []
        for _, row in positions.iterrows():
            sym = str(row['symbol'])
            cur_qty = float(row.get('qty', 0.0))
            d_disc = int(dir_map.get(sym, 0))
            do_exit = 1 if sym in exit_syms else 0

            if do_exit:
                tgt_dir = 0
                sc = 0.0
            else:
                tgt_dir = d_disc
                sc = scale

            # target quantity in contracts
            tgt_qty = base_for(sym) * tgt_dir * sc
            if round_to_int:
                # keep sign, round magnitude to nearest integer
                sign = np.sign(tgt_qty)
                tgt_qty = float(sign * np.round(abs(tgt_qty)))

            d_qty = float(tgt_qty - cur_qty)

            # optional multiplier if present (reporting only)
            mult = float(row.get(contract_mult_col, 1.0)) if contract_mult_col in row else 1.0

            rows.append({
                'symbol': sym,
                'current_qty': cur_qty,
                'target_qty': tgt_qty,
                'delta_qty': d_qty,
                'dir_discrete': d_disc,
                'scale': float(sc),
                'exit_flag': do_exit,
                'ES': es_val,
                'ES_target': es_target,
                contract_mult_col: mult,
            })

        out = pd.DataFrame(rows)
        # ensure stable column order
        cols = ['symbol','current_qty','target_qty','delta_qty','dir_discrete','scale','exit_flag','ES','ES_target']
        if contract_mult_col in out.columns and contract_mult_col not in cols:
            cols.append(contract_mult_col)
        return out[cols]

    def rebalance(
        self,
        positions: pd.DataFrame,
        prices: Union[pd.DataFrame, Dict[str, pd.Series]],
        iv: Union[pd.DataFrame, Dict[str, pd.Series]],
    ) -> pd.DataFrame:
        """Return updated positions (same schema) with qty set to target quantities.

        This enables a read→control→write→read workflow. Any extra columns are preserved.
        """
        ctrl = self.position_control(positions, prices, iv)
        # Merge targets back into the original positions
        pos2 = positions.copy()
        pos2 = _order_canonical(pos2)
        pos2 = pos2.merge(ctrl[['symbol','target_qty']], on='symbol', how='left', suffixes=('', '_ctrl'))
        # If no target was produced for a symbol, keep its original qty
        pos2['qty'] = np.where(pos2['target_qty'].notna(), pos2['target_qty'], pos2['qty']).astype(float)
        pos2 = pos2.drop(columns=['target_qty'])
        # Ensure canonical ordering and types
        return _order_canonical(pos2)


    async def build_portfolio_from_scratch(
        self,
        prices: pd.DataFrame,
        percentile: float = 0.1,
        r: float = 0.0414,
    ) -> pd.DataFrame:
        """Construct a new portfolio from raw price data and config settings.

        Steps:
        1. Compute momentum signals asynchronously via MomentumSignal.
        2. Rank symbols and pick top/bottom percentiles.
        3. Infer hypothetical option parameters (sell options only):
           - Strike X from target delta (cfg.delta_range)
           - Maturity T from dtm_range
           - r defaults to 4.14% but can be overridden.
        4. Use risk module to compute ES and adjust target positions.

        Returns a DataFrame with: symbol, signal, S, X, T, sigma, option_type, ES, ES_target, target_qty.
        """
        

        ms = MomentumSignal(self.cfg)

        # Compute signals concurrently per symbol
        async def score_one(sym: str) -> tuple[str, float]:
            try:
                val = ms.generate_signal(prices[[sym]])
                return sym, float(val)
            except Exception:
                return sym, np.nan

        tasks = [score_one(c) for c in prices.columns if prices[c].notna().sum() > self.cfg.momentum_window]
        results = await asyncio.gather(*tasks)
        sig_series = pd.Series({s: v for s, v in results}).dropna()

        # Rank and select top/bottom percentiles
        n = len(sig_series)
        k = max(1, int(n * percentile))
        winners = sig_series.sort_values(ascending=False).head(k)
        losers = sig_series.sort_values(ascending=True).head(k)

        selected = pd.concat([winners, losers])
        selected.name = 'signal'

        # Infer options to sell (puts for losers, calls for winners)
        dtm_range = getattr(self.cfg, 'dtm_range', (30, 45))
        delta_range = getattr(self.cfg, 'delta_range', (0.25, 0.40))
        T_years = np.mean(dtm_range) / 365.0
        target_delta = np.mean(delta_range)

        rows = []
        for sym, sig in selected.items():
            S = float(prices[sym].iloc[-1])
            sigma = prices[sym].pct_change().std() * np.sqrt(252.0)
            opt_type = 'call' if sym in winners.index else 'put'
            td = target_delta if opt_type == 'call' else -target_delta
            z = norm.ppf(td if opt_type == 'call' else td + 1)
            ln_S_over_X = sigma * np.sqrt(T_years) * z - (r + 0.5 * sigma * sigma) * T_years
            X = S / np.exp(ln_S_over_X)

            rows.append({
                'symbol': sym,
                'signal': sig,
                'S': S,
                'X': X,
                'T': T_years,
                'r': r,
                'q': 0.0,
                'sigma': sigma,
                'option_type': opt_type,
                'qty': 0.0,
                'dtm': np.mean(dtm_range),
                'multiplier': 1.0,
            })

        positions = pd.DataFrame(rows)

        # Compute risk and scaled positions, then apply targets to get positions DataFrame
        ctrl = self.position_control(positions, prices, iv=None)
        # Attach signal for transparency, but preserve canonical schema
        positions['signal'] = positions['symbol'].map(selected)
        # Apply targets (qty := target_qty)
        reb = positions.merge(ctrl[['symbol','target_qty','ES','ES_target','scale']], on='symbol', how='left')
        reb['qty'] = reb['target_qty'].fillna(0.0).astype(float)
        reb = reb.drop(columns=['target_qty'])
        # Canonical ordering with extras kept
        reb = _order_canonical(reb)
        return reb
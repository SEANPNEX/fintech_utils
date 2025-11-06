from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .config import MomentumConfig
from .risk import compute_portfolio_es
from .signal import MomentumSignal

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
        long_exit: float = 0.3,
        short_exit: float = 0.3,
    ) -> pd.DataFrame:
        """Suggest exits based on signal weakening.

        Parameters
        ----------
        positions : DataFrame with columns ['symbol','option_type', ...]
        signals : pd.Series mapping symbol -> z-like score (positive = bullish, negative = bearish)
        long_exit, short_exit : float
            Hysteresis bands (contrarian): exit PUT shorts (long losers) if signal > -long_exit; exit CALL shorts (short winners) if signal < short_exit.
        """
        if 'symbol' not in positions.columns:
            raise ValueError("positions must have 'symbol' column for signal-based exits")
        side = positions['option_type'].str.lower().map({'call': 'short', 'put': 'long'})
        df = positions.copy()
        df['signal'] = df['symbol'].map(signals)
        df['exit_flag'] = False
        # Long-momentum legs are implemented with PUT shorts; exit if signal > -long_exit (contrarian)
        df.loc[(df['option_type'].str.lower() == 'put') & (df['signal'] > -long_exit), 'exit_flag'] = True
        # Short-momentum legs are implemented with CALL shorts; exit if signal < short_exit (contrarian)
        df.loc[(df['option_type'].str.lower() == 'call') & (df['signal'] < short_exit), 'exit_flag'] = True
        return df[df['exit_flag']].copy()


    # --------------------------- Position Control ---------------------------
    def position_control(
        self,
        positions: pd.DataFrame,
        prices: Union[pd.DataFrame, Dict[str, pd.Series]],
        iv: Union[pd.DataFrame, Dict[str, pd.Series]],
        *,
        alpha: Optional[float] = None,
        neutral_band: float = 0.25,
        contrarian: bool = True,
        gamma: float = 0.5,
        base_qty: float = 1.0,
        per_symbol_base: Optional[Dict[str, float]] = None,
        contract_mult_col: str = 'multiplier',
        round_to_int: bool = True,
    ) -> pd.DataFrame:
        """Compute **position differences** given ES budget and internally computed signals.

        Signals are computed internally via MomentumSignal based on provided prices.

        Returns a DataFrame with columns:
          symbol, current_qty, target_qty, delta_qty, dir_discrete, scale, exit_flag, ES, ES_target

        Notes
        -----
        - Contrarian mapping (default): long losers (PUT shorts), short winners (CALL shorts).
        - If an exit is triggered for a symbol, target is set to 0.
        - `base_qty` or `per_symbol_base` define the magnitude of a unit position per symbol.
        - If `contract_mult_col` exists in `positions`, it will be used only in reporting; quantities are integers in contracts.
        """
        # Compute per-symbol signals using MomentumSignal only (no math here)
        ms = MomentumSignal(self.cfg)
        pos_syms = positions['symbol'].astype(str).unique().tolist()
        price_df = prices if isinstance(prices, pd.DataFrame) else pd.DataFrame(prices)

        signals = None
        # Prefer a z-scored per-symbol method if it exists in signal.py
        if hasattr(ms, 'zscore_by_symbol') and callable(getattr(ms, 'zscore_by_symbol')):
            try:
                zs = ms.zscore_by_symbol(price_df[pos_syms].dropna(axis=1, how='all'))
                signals = zs.reindex(pos_syms).fillna(0.0)
            except Exception:
                signals = None
        # Else prefer raw per-symbol scores if available
        if signals is None and hasattr(ms, 'scores_by_symbol') and callable(getattr(ms, 'scores_by_symbol')):
            try:
                sc = ms.scores_by_symbol(price_df[pos_syms].dropna(axis=1, how='all'))
                signals = sc.reindex(pos_syms).fillna(0.0)
            except Exception:
                signals = None
        # Fallback: call generate_signal per symbol (still using signal.py API)
        if signals is None:
            sig_map: Dict[str, float] = {}
            for sym in pos_syms:
                if isinstance(price_df, pd.DataFrame) and sym in price_df.columns:
                    try:
                        sig_map[sym] = ms.generate_signal(price_df[[sym]])
                    except Exception:
                        sig_map[sym] = 0.0
                else:
                    sig_map[sym] = 0.0
            signals = pd.Series(sig_map)

        # 1) Exit suggestions based on contrarian hysteresis
        exits_df = self.exit_on_signal(positions, signals, long_exit=neutral_band, short_exit=neutral_band)
        exit_syms = set(exits_df['symbol']) if not exits_df.empty else set()

        # 2) ES nowcast for whole book and smooth scale vs ES target
        a = float(alpha if alpha is not None else getattr(self.cfg, "es_alpha", 0.05))
        es_out = self.nowcast_es(positions, prices, iv, alpha=a)
        es_val = float(es_out.get('ES', np.nan))
        es_target = float(getattr(self.cfg, 'es_target', np.nan))
        scale = _smooth_scale(es_val, es_target, gamma=gamma)

        # 3) Build discrete directions per symbol
        dir_map: Dict[str, int] = {}
        for sym, sig in signals.items():
            dir_map[str(sym)] = _discrete_direction(float(sig), neutral_band=neutral_band, contrarian=contrarian)

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
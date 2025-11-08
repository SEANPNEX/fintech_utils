import numpy as np
from scipy.stats import norm
from typing import List


def bsm(S, X, T, r, sigma, option_type="call", q=0.0):
    """
    Black-Scholes-Merton option pricing formula.
    """
    d1 = (np.log(S / X) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - X * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        return X * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

def bsm_iv(price, S, X, T, r, option_type="call", tol=1e-6, max_iterations=100, q=0.0):
    """
    Implied volatility calculation using the Black-Scholes-Merton model.
    """
    sigma = 0.2 
    for i in range(max_iterations):
        price_estimate = bsm(S, X, T, r, sigma, option_type, q=q)
        vega = S * np.exp(-q * T) * norm.pdf((np.log(S / X) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))) * np.sqrt(T)
        price_diff = price - price_estimate
        if abs(price_diff) < tol:
            return sigma
        sigma += price_diff / vega
    return sigma 

def bsm_american_approx(S, X, T, r, sigma, option_type="call"):
    """
    Approximate pricing for American options using the BSM model.
    """
    if option_type == "call":
        return bsm(S, X, T, r, sigma, option_type)
    elif option_type == "put":
        # Using a simple approximation for American put options
        bsm_price = bsm(S, X, T, r, sigma, option_type)
        early_exercise_premium = X * np.exp(-r * T) - S if S < X else 0
        return bsm_price + early_exercise_premium
    
class bsm_greeks:
    def __init__(self, S, X, T, r, sigma, option_type="call", q=0.0):
        self.S = S
        self.X = X
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type
        self.q = q
        self.d1 = (np.log(S / X) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        self.d2 = self.d1 - sigma * np.sqrt(T)

    def delta(self):
        if self.option_type == "call":
            return np.exp(-self.q * self.T) * norm.cdf(self.d1)
        elif self.option_type == "put":
            return np.exp(-self.q * self.T) * (norm.cdf(self.d1) - 1)

    def gamma(self):
        return np.exp(-self.q * self.T) * norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(self.T))

    def theta(self):
        if self.option_type == "call":
            return (-(self.S * np.exp(-self.q * self.T) * norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(self.T))
                    - self.r * self.X * np.exp(-self.r * self.T) * norm.cdf(self.d2)
                    + self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(self.d1))
        elif self.option_type == "put":
            return (-(self.S * np.exp(-self.q * self.T) * norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(self.T))
                    + self.r * self.X * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
                    - self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(-self.d1))

    def vega(self):
        return self.S * np.exp(-self.q * self.T) * norm.pdf(self.d1) * np.sqrt(self.T)

    def rho(self):
        if self.option_type == "call":
            return self.X * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2)
        elif self.option_type == "put":
            return -self.X * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
        
def bsm_pnl(S0, S1, X, T, r, sigma, option_type="call"):
    """
    Calculate the P&L of a BSM option position from time 0 to time 1.
    """
    option_0 = bsm(S0, X, T, r, sigma, option_type)
    next_T = max(T - (1 / 365), np.finfo(float).eps)
    option_1 = bsm(S1, X, next_T, r, sigma, option_type)  # Assuming 1 day has passed
    return option_1 - option_0


# --- Binomial Greeks via bump-and-reprice ---
class binomial_greeks:
    """
    Greeks via binomial tree (American/European, continuous dividend q)
    with improved numerical stability and consistent drift b = r - q.
    """
    def __init__(
        self,
        S, X, T, r, sigma, N=1000,
        option_type="call", q=0.0, american=False,
        rel_bump_S=5e-5, abs_bump_sigma=5e-5, abs_bump_r=1e-5
    ):
        self.S = float(S)
        self.X = float(X)
        self.T = float(T)
        self.r = float(r)
        self.q = float(q)
        self.sigma = float(sigma)
        self.N = int(N)
        self.option_type = option_type
        self.american = american
        # bump parameters
        self.rel_bump_S = rel_bump_S
        self.abs_bump_sigma = abs_bump_sigma
        self.abs_bump_r = abs_bump_r
    
    def _build_trees(self, S=None, X=None, T=None, r=None, sigma=None):
        S = self.S if S is None else S
        X = self.X if X is None else X
        T = self.T if T is None else T
        r = self.r if r is None else r
        sigma = self.sigma if sigma is None else sigma
        q = self.q
        N = self.N

        dt = T / N
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        b = r - q
        p = (np.exp(b * dt) - d) / (u - d)
        p = np.clip(p, 0.0, 1.0)
        df = np.exp(-r * dt)

        # Stock tree
        S_tree = np.zeros((N + 1, N + 1))
        S_tree[0, 0] = S
        for j in range(1, N + 1):
            S_tree[0:j, j] = S_tree[0:j, j - 1] * u
            S_tree[j, j] = S_tree[j - 1, j - 1] * d

        # Option tree with early exercise
        V = np.zeros_like(S_tree)
        z = 1 if self.option_type == "call" else -1
        V[:, N] = np.maximum(z * (S_tree[:, N] - X), 0.0)
        for j in range(N - 1, -1, -1):
            for i in range(j + 1):
                cont = df * (p * V[i, j + 1] + (1 - p) * V[i + 1, j + 1])
                if self.american:
                    intrinsic = max(z * (S_tree[i, j] - X), 0.0)
                    V[i, j] = max(cont, intrinsic)
                else:
                    V[i, j] = cont

        return dt, S_tree, V

    # --- internal helper: shared stock tree ---
    def _price(self, S=None, X=None, T=None, r=None, sigma=None):
        """Core binomial price using drift b = r - q."""
        S = self.S if S is None else S
        X = self.X if X is None else X
        T = self.T if T is None else T
        r = self.r if r is None else r
        sigma = self.sigma if sigma is None else sigma
        q = self.q
        N = self.N
        option_type = self.option_type
        american = self.american

        dt = T / N
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        b = r - q
        p = (np.exp(b * dt) - d) / (u - d)
        p = np.clip(p, 0, 1)
        df = np.exp(-r * dt)

        # build stock tree only once
        stock_tree = np.zeros((N + 1, N + 1))
        stock_tree[0, 0] = S
        for j in range(1, N + 1):
            stock_tree[0:j, j] = stock_tree[0:j, j - 1] * u
            stock_tree[j, j] = stock_tree[j - 1, j - 1] * d

        # option value tree
        option_tree = np.zeros_like(stock_tree)
        z = 1 if option_type == "call" else -1

        # terminal payoffs
        option_tree[:, N] = np.maximum(z * (stock_tree[:, N] - X), 0.0)

        # backward induction
        for j in range(N - 1, -1, -1):
            for i in range(j + 1):
                cont = df * (p * option_tree[i, j + 1] + (1 - p) * option_tree[i + 1, j + 1])
                if american:
                    intrinsic = max(z * (stock_tree[i, j] - X), 0.0)
                    option_tree[i, j] = max(cont, intrinsic)
                else:
                    option_tree[i, j] = cont
        return option_tree[0, 0]

    # --- Greek calculations ---
    def price(self):
        return self._price()

    def delta(self):
        dt, S_tree, V = self._build_trees()
        S_up = S_tree[0, 1]
        S_dn = S_tree[1, 1]
        return (V[0, 1] - V[1, 1]) / (S_up - S_dn)

    def gamma(self):
        dt, S_tree, V = self._build_trees()
        Vuu, Vud, Vdd = V[0, 2], V[1, 2], V[2, 2]
        S_uu, S_ud, S_dd = S_tree[0, 2], S_tree[1, 2], S_tree[2, 2]
        delta_up = (Vuu - Vud) / (S_uu - S_ud)
        delta_dn = (Vud - Vdd) / (S_ud - S_dd)
        hS = 0.5 * (S_uu - S_dd)
        return (delta_up - delta_dn) / hS

    def vega(self):
        eps = self.abs_bump_sigma
        Vu = self._price(sigma=self.sigma + eps)
        Vd = self._price(sigma=self.sigma - eps)
        return (Vu - Vd) / (2 * eps)

    def rho(self):
        """
        Rho for American option with continuous dividend yield.
        Computed by finite differencing the binomial tree prices.
        """
        eps = max(self.abs_bump_r, 1e-5)  # small absolute bump in rate
        Vu = self._price(r=self.r + eps)
        Vd = self._price(r=self.r - eps)
        rho_val = (Vu - Vd) / (2 * eps)
        return rho_val
    
    def theta(self, convention="market_neg"):
        """
        Stable node-based theta using two-step central difference.
        convention: 'math' (signed), 'market_neg' (-theta), 'abs' (absolute)
        """
        dt, S_tree, V = self._build_trees()
        theta_val = (V[1, 2] - V[0, 0]) / (2.0 * dt)
        if convention == "market_neg":
            return -theta_val
        elif convention == "abs":
            return abs(theta_val)
        return theta_val
    

def bt_american_discrete_div(call, S, X, T, r, div_amts, div_times_steps, sigma, N):
    """
    American option with DISCRETE CASH dividends using recursive split at each dividend date.
    div_times_steps: iterable of integer step indices k in [1, N] for ex-div dates.
    div_amts: same length, cash amounts paid at those steps.
    """
    # base cases: no dividends or first dividend beyond the grid -> standard American (no discrete divs)
    if len(div_amts) == 0 or len(div_times_steps) == 0:
        return _bt_american_no_discrete_div(call, S, X, T, r, sigma, N)
    k1 = int(div_times_steps[0])
    if k1 > N:
        return _bt_american_no_discrete_div(call, S, X, T, r, sigma, N)

    dt = T / N
    u  = np.exp(sigma * np.sqrt(dt))
    d  = 1.0 / u
    p  = (np.exp(r*dt) - d) / (u - d)
    p  = np.clip(p, 0.0, 1.0)
    df = np.exp(-r*dt)
    z  = +1 if call else -1

    # triangular indexing helpers
    def nnode(n):  # total nodes from level 0..n
        return (n+1)*(n+2)//2
    def idx(i, j):  # 0-based
        return ((j)*(j+1))//2 + i

    # storage up to first dividend level k1
    V = np.empty(nnode(k1))
    # backward through levels j = k1 .. 0
    for j in range(k1, -1, -1):
        for i in range(j, -1, -1):
            price = S * (u**i) * (d**(j - i))
            if j < k1:
                # normal backward induction before dividend date
                cont = df * (p * V[idx(i+1, j+1)] + (1 - p) * V[idx(i, j+1)])
                exer = max(0.0, z*(price - X))
                V[idx(i, j)] = max(exer, cont)
            else:
                # at the dividend date: choose between exercising now or holding through the jump
                S_after = max(price - float(div_amts[0]), 0.0)
                # recursive tail: remaining time, steps, and shifted dividends
                tail_T   = T - k1*dt
                tail_N   = N - k1
                tail_div_amts  = div_amts[1:]
                tail_div_steps = [k - k1 for k in div_times_steps[1:]]
                hold_val = bt_american_discrete_div(call, S_after, X, tail_T, r,
                                                    tail_div_amts, tail_div_steps, sigma, tail_N)
                exer_val = max(0.0, z*(price - X))
                V[idx(i, j)] = max(hold_val, exer_val)

    return float(V[0])


def _bt_american_no_discrete_div(call, S, X, T, r, sigma, N):
    """Standard CRR American with NO discrete dividends (early exercise allowed)."""
    dt = T / N
    u  = np.exp(sigma * np.sqrt(dt))
    d  = 1.0 / u
    p  = (np.exp(r*dt) - d) / (u - d)
    p  = np.clip(p, 0.0, 1.0)
    df = np.exp(-r*dt)
    z  = +1 if call else -1

    # terminal layer
    ST = S * (u ** np.arange(N, -1, -1)) * (d ** np.arange(0, N+1, 1))
    V  = np.maximum(z*(ST - X), 0.0)

    # backward induction
    for j in range(N-1, -1, -1):
        V = df * (p * V[:-1] + (1 - p) * V[1:])
        S_layer = S * (u ** np.arange(j, -1, -1)) * (d ** np.arange(0, j+1, 1))
        V = np.maximum(V, np.maximum(z*(S_layer - X), 0.0))
    return float(V[0])
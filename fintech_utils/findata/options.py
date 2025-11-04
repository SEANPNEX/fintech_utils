import numpy as np
from scipy.stats import norm
from typing import List

class BinomialTreeOption:
    def __init__(self, S, X, r, N, T, option_type="call", q=0.0, american=False, sigma=0.2):
        """
        Initialize the Binomial Tree for option pricing with fixed volatility.
        """
        self.S = S
        self.X = X
        self.T = T
        self.r = r
        self.q = q
        self.N = N
        self.option_type = option_type
        self.american = american
        self.sigma = sigma
        self.dt = self.T / N  # Time step size
        self._min_vol = 1e-8
        self._u_values: List[float] = []
        self._d_values: List[float] = []

    def build_stock_tree(self):
        """
        Build the binomial tree for stock prices with constant volatility.
        """
        stock_tree = np.zeros((self.N + 1, self.N + 1))
        stock_tree[0, 0] = self.S

        u = np.exp(self.sigma * np.sqrt(self.dt))
        d = np.exp(-self.sigma * np.sqrt(self.dt))
        self._u_values = [u] * self.N
        self._d_values = [d] * self.N

        for i in range(1, self.N + 1):
            prev_column = stock_tree[:i, i - 1]
            next_column = np.empty(i + 1)
            next_column[0] = prev_column[0] * u
            for j in range(1, i):
                next_column[j] = prev_column[j - 1] * d
            next_column[i] = prev_column[i - 1] * d
            stock_tree[: i + 1, i] = next_column

        return stock_tree

    def build_option_tree(self):
        """
        Build the option price tree using backward induction.
        """
        stock_tree = self.build_stock_tree()
        option_tree = np.zeros((self.N + 1, self.N + 1))

        if self.option_type == "call":
            option_tree[:, self.N] = np.maximum(stock_tree[:, self.N] - self.X, 0)
        elif self.option_type == "put":
            option_tree[:, self.N] = np.maximum(self.X - stock_tree[:, self.N], 0)

        for i in range(self.N - 1, -1, -1):
            u = self._u_values[i]
            d = self._d_values[i]
            if np.isclose(u, d):
                p = 0.5
            else:
                p = (np.exp((self.r - self.q) * self.dt) - d) / (u - d)
                p = np.clip(p, 0.0, 1.0)
            for j in range(i + 1):
                cont = np.exp(-self.r * self.dt) * (
                    p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1]
                )
                if self.american:
                    # Intrinsic value at node (j,i)
                    S_ji = stock_tree[j, i]
                    if self.option_type == "call":
                        intrinsic = max(S_ji - self.X, 0.0)
                    else:  # put
                        intrinsic = max(self.X - S_ji, 0.0)
                    option_tree[j, i] = max(cont, intrinsic)
                else:
                    option_tree[j, i] = cont

        return option_tree

    def price(self):
        """
        Compute the option price at time 0.
        """
        option_tree = self.build_option_tree()
        return option_tree[0, 0]

def compute_binomial_price(S, X, T, r, sigma, N, option_type="call", q=0.0, american=False):
    """
        Compute the binomial tree option price for given parameters
        S: underlying price
        X: strike price
        T: time to maturity (in years)
        r: risk-free rate
        sigma: volatility
        N: number of steps in the binomial tree
        option_type: type of the option ("call" or "put")
        return: option price
    """
    option = BinomialTreeOption(S, X, r, N, T, option_type=option_type, q=q, american=american, sigma=sigma)
    return option.price()

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
def binomial_greeks(
    S, X, T, r, sigma, N,
    option_type="call", q=0.0, american=False,
    rel_bump_S=1e-4, abs_bump_sigma=1e-4, abs_bump_r=1e-4
):
    """
    Compute Greeks for the binomial option pricer (supports American/European and dividend yield q)
    using bump-and-reprice finite differences.

    Parameters
    ----------
    S : float
        Spot price
    X : float
        Strike
    T : float
        Time to maturity in years
    r : float
        Risk-free rate (continuously compounded)
    sigma : float
        Volatility (per year)
    N : int
        Number of steps in the binomial tree
    option_type : {"call", "put"}
    q : float
        Continuous dividend yield
    american : bool
        If True, allows early exercise (American). Otherwise European.
    rel_bump_S : float
        Relative bump for S used in delta/gamma
    abs_bump_sigma : float
        Absolute bump for sigma used in vega
    abs_bump_r : float
        Absolute bump for r used in rho

    Returns
    -------
    dict with keys: price, delta, gamma, vega, rho, theta
    """
    # Base price
    V0 = compute_binomial_price(S, X, T, r, sigma, N, option_type=option_type, q=q, american=american)

    # --- Delta & Gamma (symmetric bumps on S) ---
    Su = S * (1.0 + rel_bump_S)
    Sd = S * (1.0 - rel_bump_S)
    Vu = compute_binomial_price(Su, X, T, r, sigma, N, option_type=option_type, q=q, american=american)
    Vd = compute_binomial_price(Sd, X, T, r, sigma, N, option_type=option_type, q=q, american=american)
    delta = (Vu - Vd) / (Su - Sd)
    # central second derivative w.r.t S
    gamma = (Vu - 2.0 * V0 + Vd) / ((S * rel_bump_S) ** 2)

    # --- Vega (symmetric bump on sigma) ---
    sig_u = max(sigma + abs_bump_sigma, 1e-12)
    sig_d = max(sigma - abs_bump_sigma, 1e-12)
    Vsig_u = compute_binomial_price(S, X, T, r, sig_u, N, option_type=option_type, q=q, american=american)
    Vsig_d = compute_binomial_price(S, X, T, r, sig_d, N, option_type=option_type, q=q, american=american)
    vega = (Vsig_u - Vsig_d) / (2.0 * abs_bump_sigma)

    # --- Rho (symmetric bump on r) ---
    ru = r + abs_bump_r
    rd = r - abs_bump_r
    Vru = compute_binomial_price(S, X, T, ru, sigma, N, option_type=option_type, q=q, american=american)
    Vrd = compute_binomial_price(S, X, T, rd, sigma, N, option_type=option_type, q=q, american=american)
    rho = (Vru - Vrd) / (2.0 * abs_bump_r)

    # --- Theta (backward difference in time) ---
    # Use one time step or one day, whichever is larger, to avoid too tiny dt
    dt = max(T / max(N, 1), 1.0 / 365.0)
    T_next = max(T - dt, np.finfo(float).eps)
    V_next = compute_binomial_price(S, X, T_next, r, sigma, N, option_type=option_type, q=q, american=american)
    theta = (V_next - V0) / (-dt)

    return {
        "price": V0,
        "delta": float(delta),
        "gamma": float(gamma),
        "vega": float(vega),
        "rho": float(rho),
        "theta": float(theta),
    }

class BinomialGreeks:
    """
    Greeks via binomial tree (supports American/European and dividend yield q)
    using bump-and-reprice finite differences.

    Usage mirrors `bsm_greeks`:
        g = BinomialGreeks(S, X, T, r, sigma, N, option_type="call", q=0.0, american=False)
        g.price(); g.delta(); g.gamma(); g.vega(); g.rho(); g.theta()
    """
    def __init__(
        self,
        S, X, T, r, sigma, N,
        option_type="call", q=0.0, american=False,
        rel_bump_S=1e-4, abs_bump_sigma=1e-4, abs_bump_r=1e-4
    ):
        self.S = float(S)
        self.X = float(X)
        self.T = float(T)
        self.r = float(r)
        self.sigma = float(sigma)
        self.N = int(N)
        self.option_type = option_type
        self.q = float(q)
        self.american = bool(american)
        # bump params
        self.rel_bump_S = float(rel_bump_S)
        self.abs_bump_sigma = float(abs_bump_sigma)
        self.abs_bump_r = float(abs_bump_r)

    # ---- helpers ----
    def _price(self, S=None, X=None, T=None, r=None, sigma=None):
        return compute_binomial_price(
            S if S is not None else self.S,
            X if X is not None else self.X,
            T if T is not None else self.T,
            r if r is not None else self.r,
            sigma if sigma is not None else self.sigma,
            self.N,
            option_type=self.option_type,
            q=self.q,
            american=self.american,
        )

    # ---- greek methods ----
    def price(self):
        return self._price()

    def delta(self):
        h = self.rel_bump_S
        Su = self.S * (1.0 + h)
        Sd = self.S * (1.0 - h)
        Vu = self._price(S=Su)
        Vd = self._price(S=Sd)
        return (Vu - Vd) / (Su - Sd)

    def gamma(self):
        h = self.rel_bump_S
        Su = self.S * (1.0 + h)
        Sd = self.S * (1.0 - h)
        Vu = self._price(S=Su)
        V0 = self._price()
        Vd = self._price(S=Sd)
        return (Vu - 2.0 * V0 + Vd) / ((self.S * h) ** 2)

    def vega(self):
        eps = self.abs_bump_sigma
        sig_u = max(self.sigma + eps, 1e-12)
        sig_d = max(self.sigma - eps, 1e-12)
        Vsig_u = self._price(sigma=sig_u)
        Vsig_d = self._price(sigma=sig_d)
        return (Vsig_u - Vsig_d) / (2.0 * eps)

    def rho(self):
        eps = self.abs_bump_r
        Vru = self._price(r=self.r + eps)
        Vrd = self._price(r=self.r - eps)
        return (Vru - Vrd) / (2.0 * eps)

    def theta(self):
        # Backward difference in time
        dt_tree = self.T / max(self.N, 1)
        dt = max(dt_tree, 1.0 / 365.0)
        T_next = max(self.T - dt, np.finfo(float).eps)
        V_next = self._price(T=T_next)
        V0 = self._price()
        return (V_next - V0) / (-dt)

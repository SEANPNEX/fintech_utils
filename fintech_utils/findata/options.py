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
        European-style rho (∂V/∂r) computed on a no-early-exercise tree,
        returned per +1% change in r (i.e., divide by 100).
        This matches common CSV/lecture conventions better than American rho.
        """
        S, X, T, r, q, sigma, N = self.S, self.X, self.T, self.r, self.q, self.sigma, self.N
        dt = T / N
        u = np.exp(sigma * np.sqrt(dt))
        d = 1.0 / u
        b = r - q
        p = (np.exp(b * dt) - d) / (u - d)
        p = np.clip(p, 0.0, 1.0)
        df = np.exp(-r * dt)

        # derivatives wrt r
        dp_dr  = (dt * np.exp(b * dt)) / (u - d)     # from p = (e^{b dt}-d)/(u-d)
        ddf_dr = -dt * df                             # from df = e^{-r dt}

        # stock tree
        S_tree = np.zeros((N+1, N+1))
        S_tree[0,0] = S
        for j in range(1, N+1):
            S_tree[0:j, j] = S_tree[0:j, j-1] * u
            S_tree[j,   j] = S_tree[j-1, j-1] * d

        # value & rho trees (European: no early exercise)
        V = np.zeros_like(S_tree)
        R = np.zeros_like(S_tree)  # dV/dr
        z = 1 if self.option_type == "call" else -1
        V[:, N] = np.maximum(z * (S_tree[:, N] - X), 0.0)
        # rho at maturity is 0
        for j in range(N-1, -1, -1):
            for i in range(j+1):
                cont = p*V[i, j+1] + (1-p)*V[i+1, j+1]
                V[i, j] = df * cont
                # derivative of continuation value
                R[i, j] = (
                    ddf_dr * cont +
                    df * (dp_dr * (V[i, j+1] - V[i+1, j+1]) + p*R[i, j+1] + (1-p)*R[i+1, j+1])
                )

        # return per 1% change in r (divide by 100)
        return R[0, 0] / 100.0
    
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
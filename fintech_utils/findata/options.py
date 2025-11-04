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

from ..linalg.mat import near_psd_cov, higham_cov, pca_cov
from scipy import stats
from scipy.stats import norm, t
import numpy as np


def monte_carlo_normal_var(mat, N = 100000):
    L = np.linalg.cholesky(mat)
    k = mat.shape[0]
    Z = np.random.normal(0, 1, (N, k))
    R = Z @ L.T
    return np.cov(R, rowvar=False)

def monte_carlo_normal_var_psd(mat, N = 100000):
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals[eigvals < 0] = 0
    L = eigvecs @ np.diag(np.sqrt(eigvals))
    k = mat.shape[0]
    Z = np.random.normal(0, 1, (N, k))
    R = Z @ L.T
    return np.cov(R, rowvar=False)

def monte_carlo_normal_var_near_psd(mat, N = 100000):
    mat_psd = near_psd_cov(mat)
    return monte_carlo_normal_var_psd(mat_psd, N)

def monte_carlo_normal_var_higham(mat, N = 100000):
    mat_psd = higham_cov(mat)
    return monte_carlo_normal_var_psd(mat_psd, N)


def monte_carlo_pca_var(mat, N = 100000, var_explained=0.99):
    mat_pca = pca_cov(mat, var_explained)
    return monte_carlo_normal_var_psd(mat_pca, N)


def var_normal(x, alpha=0.05):
    mu = np.mean(x)
    sigma = np.std(x, ddof=1)
    z_alpha = stats.norm.ppf(alpha)  
    var_signed = mu + sigma * z_alpha  
    var_abs = -var_signed              
    var_diff = -sigma * z_alpha        
    return var_abs, var_diff

def var_t(x, alpha=0.05):
    nu_hat, loc_hat, scale_hat = stats.t.fit(x)
    t_alpha = stats.t.ppf(alpha, df=nu_hat) 
    q_alpha = loc_hat + scale_hat * t_alpha  
    var_abs = -q_alpha                       
    var_diff = -scale_hat * t_alpha          
    return var_abs, var_diff

def es_normal(x, alpha=0.05):
    mu = np.mean(x)
    sigma = np.std(x, ddof=1)
    z_alpha = norm.ppf(1 - alpha)
    phi = norm.pdf(z_alpha)
    es_abs = mu - sigma * phi / alpha
    es_diff = - sigma * phi / alpha
    return -es_abs, -es_diff

def es_t(x, alpha=0.05):
    nu_hat, loc_hat, scale_hat = t.fit(x)
    t_alpha = t.ppf(alpha, df=nu_hat)
    phi = t.pdf(t_alpha, df=nu_hat)
    es_abs = loc_hat - scale_hat * (nu_hat + t_alpha**2) / (nu_hat - 1) * phi / alpha
    es_diff = - scale_hat * (nu_hat + t_alpha**2) / (nu_hat - 1) * phi / alpha
    return -es_abs, -es_diff

def portfolio_var_es_copula(returns, dists, holdings, prices, alpha=0.05, n_simu=1_000_000):
    """
        Get portfolio VaR and ES using a copula approach.
        Inputs:
            returns: DataFrame of asset returns, each column is an asset
            dists: list of distribution names for each asset, e.g. ['norm', 't']
            holdings: list of number of shares held for each asset
            prices: list of prices per share for each asset
            alpha: significance level for VaR/ES
            n_simu: number of simulations to run
        Outputs:
            VaR and ES in dollar terms
    """
    holdings_arr = np.asarray(holdings, dtype=float)
    prices_arr = np.asarray(prices, dtype=float)
    if holdings_arr.shape != prices_arr.shape:
        raise ValueError("holdings and prices must have the same shape")
    assets = returns.columns
    n_assets = len(assets)
    v0 = holdings_arr * prices_arr
    V0 = float(v0.sum())
    weights = v0 / V0

    # Fit marginals
    params = []
    uni = []
    for i, asset in enumerate(assets):
        if dists[i] == 'norm':
            mu, sigma = np.mean(returns[asset]), np.std(returns[asset], ddof=1)
            params.append((mu, sigma))
            uni.append(norm.cdf((returns[asset] - mu) / sigma))
        elif dists[i] == 't':
            nu, loc, scale = t.fit(returns[asset], floc=np.mean(returns[asset]))
            params.append((nu, loc, scale))
            uni.append(t.cdf((returns[asset] - loc) / scale, df=nu))
        else:
            raise ValueError(f"Unsupported distribution: {dists[i]}")

    # Estimate correlation (Gaussian copula)
    clip_eps = np.finfo(float).eps
    uni = np.clip(np.array(uni).T, clip_eps, 1.0 - clip_eps)  # (n_samples, n_assets)
    z = norm.ppf(uni)
    rho = np.corrcoef(z.T)
    # Simulate correlated normals
    Z = np.random.multivariate_normal(np.zeros(n_assets), rho, n_simu)
    U = np.clip(norm.cdf(Z), clip_eps, 1.0 - clip_eps)
    # Transform back to each marginal
    R_sim = np.zeros_like(U)
    for i in range(n_assets):
        if dists[i] == 'norm':
            mu, sigma = params[i]
            R_sim[:,i] = norm.ppf(U[:,i], loc=mu, scale=sigma)
        elif dists[i] == 't':
            nu, loc, scale = params[i]
            R_sim[:,i] = t.ppf(U[:,i], df=nu, loc=loc, scale=scale)
    # Compute portfolio returns
    R_port = R_sim @ weights
    # Compute VaR / ES
    q_alpha = np.quantile(R_port, alpha)
    var_pct = -q_alpha
    es_pct = -R_port[R_port <= q_alpha].mean()
    var_dollar = V0 * var_pct
    es_dollar = V0 * es_pct
    return {'VaR_dollar': var_dollar, 'ES_dollar': es_dollar, 'VaR_pct': var_pct, 'ES_pct': es_pct}

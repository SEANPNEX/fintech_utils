import pandas as pd
import numpy as np
import numpy.typing as npt
from math import lgamma, pi, log
from scipy.optimize import minimize
from scipy import stats
from scipy.stats import norm, t
# EW covariance matrix, lambda = 0.97

def fit_normal(data: npt.NDArray) -> tuple:
    """
        Given a series of data, return tuple of mean and variance
    """
    n = data.size
    mu = data.sum() / n
    var = ((data - mu) ** 2).sum() / (n - 1)
    return (mu, np.sqrt(var))

def log_t(x: float, mu: float, sigma: float, nu: float) -> float:
    """"
        The log PDF for T-distribution
    """
    return lgamma((nu + 1) / 2) - lgamma(nu / 2) - 0.5 * log(nu * pi) - log(sigma) - ((nu + 1) / 2) * np.log1p((((x - mu) / sigma) ** 2) / nu)

def neg_loglikelihood(theta, x: npt.NDArray):
    """"
        Give the negative ll for optimizer to minimize
    """
    mu, eta, tau = theta
    sigma = np.exp(eta)
    nu = np.exp(tau) + 2.0
    ll = log_t(x, mu, sigma, nu).sum()
    return -ll
      
      

def fit_T(x: npt.NDArray) -> tuple:
    """
        Given data, return tuple of mu, sigma, nu
    """
    mu0 = np.median(x)
    mad = np.median(np.abs(x - mu0))
    eta0 = np.log(mad)
    tau0 = np.log(10.0 - 2.0)  
    x0 = np.array([mu0, eta0, tau0])
    res = minimize(
        neg_loglikelihood, x0, args=(x,),
        method="L-BFGS-B"
    )
    mu_hat, eta_hat, tau_hat = res.x
    sigma_hat = np.exp(eta_hat)
    nu_hat = np.exp(tau_hat) + 2.0
    return (mu_hat, sigma_hat, nu_hat)

def student_t_logpdf(e, sigma, nu):
    sigma = np.maximum(sigma, 1e-12)
    nu = np.maximum(nu, 1.999999)  # ensure variance finite
    term1 = lgamma((nu + 1.0)/2.0) - lgamma(nu/2.0)
    term2 = -0.5*(np.log(nu) + np.log(np.pi)) - np.log(sigma)
    term3 = -((nu + 1.0)/2.0)*np.log1p((e/sigma)**2 / nu)
    return term1 + term2 + term3

def nll_t_regression(theta, X, y):
    n, p = X.shape 
    beta = theta[:p]
    log_sigma = theta[p]
    logit_nu = theta[p+1]  
    lo, hi = 2.0, 100.0
    nu = lo + (hi - lo)/(1.0 + np.exp(-logit_nu))
    sigma = np.exp(log_sigma)
    resid = y - X @ beta
    return -np.sum(student_t_logpdf(resid, sigma, nu))


def ew_cov(mat, lam=0.97):
    m, n = mat.shape
    # get the weight
    w = np.array([(1 - lam) * (lam ** (m - 1 - i)) for i in range(m)])
    w = w / sum(w)
    # get diag(w)
    W = np.diag(w)
    # calculate mu with mu = Rw
    mu = np.dot(mat.T, w)
    # calculate ew cov with EW = RWR^T - mumu^T
    ew_cov = mat.T @ W @ mat - np.outer(mu, mu)
    return ew_cov


# ew correlation matrix, lambda = 0.94
def ew_corr(mat, lam=0.94):
    cov = ew_cov(mat, lam)
    d = np.sqrt(np.diag(cov))
    corr = cov / np.outer(d, d)
    return corr

# near psd matrix
def near_psd_cov(A, epsilon = 0):
    # get correlation matrix
    d = np.sqrt(np.diag(A))
    corr = A / np.outer(d, d)
    eig_vals, eig_vecs = np.linalg.eig(corr)
    eig_vals[eig_vals < epsilon] = epsilon
    corr_psd = (eig_vecs @ np.diag(eig_vals) @ eig_vecs.T)
    corr_psd = corr_psd / np.outer(np.sqrt(np.diag(corr_psd)), np.sqrt(np.diag(corr_psd)))
    near_A = np.outer(d, d) * corr_psd
    return near_A

def near_psd_corr(A, epsilon = 0):
    eig_vals, eig_vecs = np.linalg.eig(A)
    eig_vals[eig_vals < epsilon] = epsilon
    corr_psd = (eig_vecs @ np.diag(eig_vals) @ eig_vecs.T)
    corr_psd = corr_psd / np.outer(np.sqrt(np.diag(corr_psd)), np.sqrt(np.diag(corr_psd)))
    return corr_psd

# Higham coverance matrix
def higham_corr(A, max_iter=100, tol=1e-6):
    n = A.shape[0]
    # symmetric
    A = (A + A.T) / 2
    X = A.copy()
    Y = np.zeros_like(A)
    for i in range(max_iter):
        R = X - Y
        eig_vals, eig_vecs = np.linalg.eig(R)
        eig_vals[eig_vals < 0] = 0
        X_new = eig_vecs @ np.diag(eig_vals) @ eig_vecs.T
        Y = X_new - R
        X_new[np.diag_indices(n)] = 1
        if np.linalg.norm(X_new - X, ord='fro') < tol:
            break
        X = X_new
    return X


def higham_cov(A, max_iter=100, tol=1e-6):
    # get correlation matrix
    d = np.sqrt(np.diag(A))
    corr = A / np.outer(d, d)
    corr_psd = higham_corr(corr, max_iter, tol)
    near_A = np.outer(d, d) * corr_psd
    return near_A

def pca_cov(mat, var_explained=0.99):
    eigvals, eigvecs = np.linalg.eigh(mat)
    total_var = np.sum(eigvals)
    sorted_indices = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]
    cum_var = np.cumsum(eigvals)
    num_components = np.searchsorted(cum_var, var_explained * total_var) + 1
    L = eigvecs[:, :num_components] @ np.diag(np.sqrt(eigvals[:num_components]))
    return L @ L.T

def arithm_ret(prices: npt.NDArray) -> npt.NDArray:
    """
        Given a series of prices, return a series of arithmetic returns
    """
    return prices[1:] / prices[:-1] - 1

def log_ret(prices: npt.NDArray) -> npt.NDArray:
    """
        Given a series of prices, return a series of log returns
    """
    return np.log(prices[1:] / prices[:-1])

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
    assets = returns.columns
    n_assets = len(assets)
    v0 = holdings * prices
    V0 = sum(v0)
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
    uni = np.array(uni).T # (n_samples, n_assets)
    z = norm.ppf(np.array(uni))
    rho = np.corrcoef(z.T)
    # Simulate correlated normals
    Z = np.random.multivariate_normal(np.zeros(n_assets), rho, n_simu)
    U = norm.cdf(Z)
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
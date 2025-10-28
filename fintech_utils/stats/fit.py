import numpy as np
import numpy.typing as npt
from math import lgamma, pi, log
from scipy.optimize import minimize


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

import numpy as np

# evaluate performance of momentum strategy

def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate the annualized Sharpe Ratio of a return series.
    returns: 1D array of periodic returns
    risk_free_rate: annual risk-free rate
    return: annualized Sharpe Ratio
    """
    excess_returns = returns - risk_free_rate / 252  # assuming daily returns
    mean_excess_return = np.mean(excess_returns)
    std_excess_return = np.std(excess_returns)
    sharpe_ratio = (mean_excess_return / std_excess_return) * np.sqrt(252)  # annualize
    return sharpe_ratio

def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate the annualized Sortino Ratio of a return series.
    returns: 1D array of periodic returns
    risk_free_rate: annual risk-free rate
    return: annualized Sortino Ratio
    """
    excess_returns = returns - risk_free_rate / 252  # assuming daily returns
    mean_excess_return = np.mean(excess_returns)
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = np.std(downside_returns)
    sortino_ratio = (mean_excess_return / downside_std) * np.sqrt(252)  # annualize
    return sortino_ratio

def max_drawdown(returns: np.ndarray) -> float:
    """
    Calculate the maximum drawdown of a return series.
    returns: 1D array of periodic returns
    return: maximum drawdown as a fraction
    """
    cumulative_returns = np.cumprod(1 + returns) - 1
    peak = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - peak) / peak
    max_drawdown = np.min(drawdowns)
    return max_drawdown

def calmar_ratio(returns: np.ndarray) -> float:
    """
    Calculate the Calmar Ratio of a return series.
    returns: 1D array of periodic returns
    return: Calmar Ratio
    """
    annual_return = (1 + np.mean(returns)) ** 252 - 1  # annualize
    max_dd = abs(max_drawdown(returns))
    calmar_ratio = annual_return / max_dd if max_dd != 0 else np.inf
    return calmar_ratio

def annualized_volatility(returns: np.ndarray) -> float:
    """
    Calculate the annualized volatility of a return series.
    returns: 1D array of periodic returns
    return: annualized volatility
    """
    daily_vol = np.std(returns)
    annual_vol = daily_vol * np.sqrt(252)  # annualize
    return annual_vol

def information_ratio(returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
    """
    Calculate the Information Ratio of a return series against a benchmark.
    returns: 1D array of periodic returns
    benchmark_returns: 1D array of benchmark periodic returns
    return: Information Ratio
    """
    active_returns = returns - benchmark_returns
    mean_active_return = np.mean(active_returns)
    std_active_return = np.std(active_returns)
    information_ratio = (mean_active_return / std_active_return) * np.sqrt(252)  # annualize
    return information_ratio

def omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
    """
    Calculate the Omega Ratio of a return series.
    returns: 1D array of periodic returns
    threshold: return threshold
    return: Omega Ratio
    """
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns < threshold]
    omega_ratio = np.sum(gains) / np.sum(losses) if np.sum(losses) != 0 else np.inf
    return omega_ratio

def tail_ratio(returns: np.ndarray, prob: float = 0.05) -> float:
    """
    Calculate the Tail Ratio of a return series.
    returns: 1D array of periodic returns
    prob: tail probability
    return: Tail Ratio
    """
    lower_quantile = np.quantile(returns, prob)
    upper_quantile = np.quantile(returns, 1 - prob)
    tail_ratio = abs(upper_quantile / lower_quantile) if lower_quantile != 0 else np.inf
    return tail_ratio

def cagr(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """
    Calculate the Compound Annual Growth Rate (CAGR) of a return series.
    returns: 1D array of periodic returns
    periods_per_year: number of return periods in a year
    return: CAGR
    """
    total_periods = len(returns)
    cumulative_return = np.prod(1 + returns) - 1
    cagr = (1 + cumulative_return) ** (periods_per_year / total_periods) - 1
    return cagr


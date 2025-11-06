import numpy as np

# evaluate performance of momentum strategy

def sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate the annualized Sharpe Ratio of a return series.
    returns: 1D array of periodic returns
    risk_free_rate: annual risk-free rate
    return: annualized Sharpe Ratio
    """
    if returns.size == 0:
        return 0.0
    excess_returns = returns - risk_free_rate / 252  # assuming daily returns
    mean_excess_return = np.mean(excess_returns)
    std_excess_return = np.std(excess_returns)
    if std_excess_return == 0:
        return 0.0
    return (mean_excess_return / std_excess_return) * np.sqrt(252)  # annualize

def sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
    """
    Calculate the annualized Sortino Ratio of a return series.
    returns: 1D array of periodic returns
    risk_free_rate: annual risk-free rate
    return: annualized Sortino Ratio
    """
    if returns.size == 0:
        return 0.0
    excess_returns = returns - risk_free_rate / 252  # assuming daily returns
    mean_excess_return = np.mean(excess_returns)
    downside_returns = excess_returns[excess_returns < 0]
    if downside_returns.size == 0:
        return np.inf
    downside_std = np.std(downside_returns)
    if downside_std == 0:
        return np.inf
    return (mean_excess_return / downside_std) * np.sqrt(252)  # annualize

def max_drawdown(returns: np.ndarray) -> float:
    """
    Calculate the maximum drawdown of a return series.
    returns: 1D array of periodic returns
    return: maximum drawdown as a fraction
    """
    if returns.size == 0:
        return 0.0
    wealth = np.concatenate(([1.0], np.cumprod(1 + returns, dtype=float)))
    peak = np.maximum.accumulate(wealth)
    drawdowns = wealth / peak - 1.0
    return float(np.min(drawdowns))

def calmar_ratio(returns: np.ndarray) -> float:
    """
    Calculate the Calmar Ratio of a return series.
    returns: 1D array of periodic returns
    return: Calmar Ratio
    """
    if returns.size == 0:
        return 0.0
    annual_return = cagr(returns)
    max_dd = abs(max_drawdown(returns))
    if max_dd == 0:
        return np.inf
    return annual_return / max_dd

def annualized_volatility(returns: np.ndarray) -> float:
    """
    Calculate the annualized volatility of a return series.
    returns: 1D array of periodic returns
    return: annualized volatility
    """
    if returns.size == 0:
        return 0.0
    daily_vol = np.std(returns)
    return daily_vol * np.sqrt(252)  # annualize

def information_ratio(returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
    """
    Calculate the Information Ratio of a return series against a benchmark.
    returns: 1D array of periodic returns
    benchmark_returns: 1D array of benchmark periodic returns
    return: Information Ratio
    """
    if returns.size == 0 or benchmark_returns.size == 0:
        return 0.0
    active_returns = returns - benchmark_returns
    mean_active_return = np.mean(active_returns)
    std_active_return = np.std(active_returns)
    if std_active_return == 0:
        return np.inf if mean_active_return > 0 else 0.0
    return (mean_active_return / std_active_return) * np.sqrt(252)  # annualize

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
    if total_periods == 0:
        return 0.0
    cumulative_return = np.prod(1 + returns) - 1
    cagr = (1 + cumulative_return) ** (periods_per_year / total_periods) - 1
    return cagr

def worst_drawdown(returns: np.ndarray) -> float:
    """
    Calculate the worst drawdown of a return series.
    returns: 1D array of periodic returns
    return: worst drawdown as a fraction
    """
    return max_drawdown(returns)

def worst_month_return(returns: np.ndarray) -> float:
    """
    Calculate the worst monthly return of a return series.
    returns: 1D array of periodic returns
    return: worst monthly return as a fraction
    """
    monthly_returns = []
    for i in range(0, len(returns), 21):  # assuming ~21 trading days per month
        month_return = np.prod(1 + returns[i:i+21]) - 1
        monthly_returns.append(month_return)
    if not monthly_returns:
        return 0.0
    return float(np.min(monthly_returns))

def best_month_return(returns: np.ndarray) -> float:
    """
    Calculate the best monthly return of a return series.
    returns: 1D array of periodic returns
    return: best monthly return as a fraction
    """
    monthly_returns = []
    for i in range(0, len(returns), 21):  # assuming ~21 trading days per month
        month_return = np.prod(1 + returns[i:i+21]) - 1
        monthly_returns.append(month_return)
    if not monthly_returns:
        return 0.0
    return float(np.max(monthly_returns))

def profitable_month_ratio(returns: np.ndarray) -> float:
    """
    Calculate the ratio of profitable months in a return series.
    returns: 1D array of periodic returns
    return: ratio of profitable months
    """
    if returns.size == 0:
        return 0.0
    profitable_months = 0
    total_months = 0
    for i in range(0, len(returns), 21):  # assuming ~21 trading days per month
        month_return = np.prod(1 + returns[i:i+21]) - 1
        if month_return > 0:
            profitable_months += 1
        total_months += 1
    ratio = profitable_months / total_months if total_months > 0 else 0.0
    return ratio

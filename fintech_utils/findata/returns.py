import numpy as np
import numpy.typing as npt


def arithm_ret(prices: npt.NDArray) -> npt.NDArray:
    """
        Given a series of prices, return a series of arithmetic returns
        prices: 1D array of prices
        return: 1D array of arithmetic returns
    """
    return prices[1:] / prices[:-1] - 1

def log_ret(prices: npt.NDArray) -> npt.NDArray:
    """
        Given a series of prices, return a series of log returns
        prices: 1D array of prices
        return: 1D array of log returns
    """
    return np.log(prices[1:] / prices[:-1])
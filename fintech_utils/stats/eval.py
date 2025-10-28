import sklearn.metrics as skm
import numpy as np

# r2 score
def r2_score(y_true, y_pred):
    """
        Calculate R-squared score
        y_true: true values
        y_pred: predicted values
        return: R-squared score
    """
    return skm.r2_score(y_true, y_pred)

# aic
def aic(n, rss, k):
    """
        Calculate Akaike Information Criterion (AIC)
        n: number of observations
        rss: residual sum of squares
        k: number of parameters
        return: AIC value
    """
    return n * np.log(rss / n) + 2 * k

# bic
def bic(n, rss, k):
    """
        Calculate Bayesian Information Criterion (BIC)
        n: number of observations
        rss: residual sum of squares
        k: number of parameters
        return: BIC value
    """
    return n * np.log(rss / n) + k * np.log(n)

# mse
def mse(y_true, y_pred):
    """
        Calculate Mean Squared Error (MSE)
        y_true: true values
        y_pred: predicted values
        return: MSE value
    """
    return skm.mean_squared_error(y_true, y_pred)

# rmse
def rmse(y_true, y_pred):
    """
        Calculate Root Mean Squared Error (RMSE)
        y_true: true values
        y_pred: predicted values
        return: RMSE value
    """
    return np.sqrt(skm.mean_squared_error(y_true, y_pred))

# mae
def mae(y_true, y_pred):
    """
        Calculate Mean Absolute Error (MAE)
        y_true: true values
        y_pred: predicted values
        return: MAE value
    """
    return skm.mean_absolute_error(y_true, y_pred)

# mape
def mape(y_true, y_pred):
    """
        Calculate Mean Absolute Percentage Error (MAPE)
        y_true: true values
        y_pred: predicted values
        return: MAPE value
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

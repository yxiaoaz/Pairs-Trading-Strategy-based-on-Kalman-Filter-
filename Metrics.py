#region imports
from AlgorithmImports import *
import numpy as np
import numpy as np
import pandas as pd
import sys
import collections, functools, operator

import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.metrics import silhouette_score
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.manifold import TSNE
#endregion
def zscore(series):
        """
        Returns the nromalized time series assuming a normal distribution
        """
        return (series-series.mean())/np.std(series)

def calculate_half_life(z_array):
    """
    This function calculates the half life parameter of a
    mean reversion series
    """
    z_lag = np.roll(z_array, 1)
    z_lag[0] = 0
    z_ret = z_array - z_lag
    z_ret[0] = 0

    # adds intercept terms to X variable for regression
    z_lag2 = sm.add_constant(z_lag)

    model = sm.OLS(z_ret[1:], z_lag2[1:])
    res = model.fit()

    halflife = -np.log(2) / res.params[1]

    return halflife

def hurst(ts):
    """
    Returns the Hurst Exponent of the time series vector ts.
    Series vector ts should be a price series.
    """
    # Create the range of lag values
    lags = range(2, 100)

    tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]

    # Use a linear fit to estimate the Hurst Exponent
    poly = np.polyfit(np.log(lags), np.log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    return poly[0] * 2.0

def variance_ratio(ts, lag=2):
 
    # make sure we are working with an array, convert if necessary
    ts = np.asarray(ts)

    # Apply the formula to calculate the test
    n = len(ts)
    mu = sum(ts[1:n] - ts[:n - 1]) / n
    m = (n - lag + 1) * (1 - lag / n)
    b = sum(np.square(ts[1:n] - ts[:n - 1] - mu)) / (n - 1)
    t = sum(np.square(ts[lag:n] - ts[:n - lag] - lag * mu)) / m
    return t / (lag * b)

def zero_crossings( x):

    x = x - x.mean()
    zero_crossings = sum(1 for i, _ in enumerate(x) if (i + 1 < len(x)) if ((x[i] * x[i + 1] < 0) or (x[i] == 0)))

    return zero_crossings


# Your New Python File

import numpy as np
import pandas as pd
from ._func import _Function, function_wrapper


def _protected_division(df1, df2):
    """Closure of division (df / df) for zero denominator."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(np.abs(df2) > 0.001, np.divide(df1, df2), 1.)
    return pd.DataFrame(result, index=df1.index, columns=df1.columns)


def _protected_sqrt(df):
    """Closure of square root for negative arguments."""
    result = np.sqrt(np.abs(df))
    return pd.DataFrame(result, index=df.index, columns=df.columns)


def _protected_log(df):
    """Closure of log for zero and negative arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(np.abs(df) > 0.001, np.log(np.abs(df)), 0.)
    return pd.DataFrame(result, index=df.index, columns=df.columns)


def _protected_inverse(df):
    """Closure of inverse for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(np.abs(df) > 0.001, 1. / df, 0.)
    return pd.DataFrame(result, index=df.index, columns=df.columns)


def _sigmoid(df):
    """Special case of logistic function to transform to probabilities."""
    with np.errstate(over='ignore', under='ignore'):
        result = 1 / (1 + np.exp(-df))
    return pd.DataFrame(result, index=df.index, columns=df.columns)


def rank(df1):
    return df1.rank(axis=1)


import numpy as np


def scale(df1):
    value = df1.div(np.abs(df1).sum(axis=1), axis=0)
    return value


def sign(df1):
    return np.sign(df1)


def make_delay(d):
    def delay(df1):
        return df1.shift(d)
    return function_wrapper(delay, f'delay{d}', 1)


def make_corr(d):
    def corr(df1, df2):
        return df1.rolling(d).corr(df2)
    return function_wrapper(corr, f'corr{d}', 2)


def make_cov(d):
    def cov(df1, df2):
        return df1.rolling(d).cov(df2)
    return function_wrapper(cov, f'cov{d}', 2)


def make_delta(d):
    def delta(df1):
        return df1.diff(d)
    return function_wrapper(delta, f'delta{d}', 1)


def make_signed_power(a):
    def signed_power(df1):
        return np.sign(df1) * (np.abs(df1) ** a)
    return function_wrapper(signed_power, f'signed_power{a}', 1)


def make_decay_linear(d):
    def decay_linear(df1):
        weights = np.arange(1, d + 1).astype(float)
        weights /= weights.sum()

        def weighted_avg(x):
            return np.dot(x[-len(weights):], weights) / weights.sum()

        return df1.rolling(window=d).apply(weighted_avg, raw=True)

    return function_wrapper(decay_linear, f'decay_linear{d}', 1)


def make_ts_min(d):
    def ts_min(df1):
        return df1.rolling(d).min()

    return function_wrapper(ts_min, f'ts_min{d}', 1)


def make_ts_max(d):
    def ts_max(df1):
        return df1.rolling(d).max()

    return function_wrapper(ts_max, f'ts_max{d}', 1)


def make_ts_argmin(d):
    def ts_argmin(df1):
        return df1.rolling(window=d).apply(lambda x: np.argmin(x.values) + 1, raw=False)

    return function_wrapper(ts_argmin, f'ts_argmin{d}', 1)


def make_ts_argmax(d):
    def ts_argmax(df1):
        return df1.rolling(window=d).apply(lambda x: np.argmax(x.values) + 1, raw=False)

    return function_wrapper(ts_argmax, f'ts_argmax{d}', 1)


def make_ts_rank(d):
    def ts_rank(df1):
        return df1.rolling(window=d).apply(lambda x: x.rank().iloc[-1], raw=False)

    return function_wrapper(ts_rank, f'ts_rank{d}', 1)


def make_sum(d):
    def sum_func(df1):
        return df1.rolling(window=d).sum()
    return function_wrapper(sum_func, f'sum{d}', 1)

def make_product(d):
    def product(df1):
        return df1.rolling(window=d).apply(np.prod, raw=True)
    return function_wrapper(product, f'prod{d}', 1)

def make_stddev(d):
    def stddev(df1):
        return df1.rolling(window=d).std()
    return function_wrapper(stddev, f'std{d}', 1)

def make_slope(d):
    def slope(df1, df2):
        return df1.rolling(d).cov(df2) / df1.rolling(d).var()
    return function_wrapper(slope, f'slope{d}', 2)

def make_intercept(d):
    def intercept(df1, df2):
        slope_val = df1.rolling(window=d).cov(df2) / df1.rolling(window=d).var()
        intercepts = df2.rolling(window=d).mean() - slope_val * df1.rolling(window=d).mean()
        return intercepts
    return function_wrapper(intercept, f'intercept{d}', 2)

def make_residual(d):
    def residual(df1, df2):
        slope_val = df1.rolling(window=d).cov(df2) / df1.rolling(window=d).var()
        intercept_val = df2.rolling(window=d).mean() - slope_val * df1.rolling(window=d).mean()
        predicted_y = slope_val * df1 + intercept_val
        residuals = df2 - predicted_y
        return residuals
    return function_wrapper(residual, f'residual{d}', 2)

add2 = _Function(function=np.add, name='add', arity=2)
sub2 = _Function(function=np.subtract, name='sub', arity=2)
mul2 = _Function(function=np.multiply, name='mul', arity=2)
div2 = _Function(function=_protected_division, name='div', arity=2)
sqrt1 = _Function(function=_protected_sqrt, name='sqrt', arity=1)
log1 = _Function(function=_protected_log, name='log', arity=1)
neg1 = _Function(function=np.negative, name='neg', arity=1)
inv1 = _Function(function=_protected_inverse, name='inv', arity=1)
abs1 = _Function(function=np.abs, name='abs', arity=1)
max2 = _Function(function=np.maximum, name='max', arity=2)
min2 = _Function(function=np.minimum, name='min', arity=2)
sig1 = _Function(function=_sigmoid, name='sig', arity=1)

function_map = {
    'add': add2, 'sub': sub2, 'mul': mul2, 'div': div2,
    'sqrt': sqrt1, 'log': log1, 'neg': neg1, 'inv': inv1,
    'abs': abs1, 'max': max2, 'min': min2, 'sig': sig1,
    'rank': function_wrapper(rank, 'rank', 1),
    'scale': function_wrapper(scale, 'scale', 1),
    'sign': function_wrapper(sign, 'sign', 1),
    'delay1': make_delay(1), 'delay5': make_delay(5), 'delay20': make_delay(20),
    # 'delay60': make_delay(60), 'delay120': make_delay(120), 'delay252': make_delay(252),
    'corr10': make_corr(10), 'corr20': make_corr(20), 'corr60': make_corr(60),
    # 'corr120': make_corr(120), 'corr252': make_corr(252),
    'cov10': make_cov(10), 'cov20': make_cov(20), 'cov60': make_cov(60),
    # 'cov120': make_cov(120), 'cov252': make_cov(252),
    'decay_linear5': make_decay_linear(5), 'decay_linear10': make_decay_linear(10),
    # 'decay_linear20': make_decay_linear(20), 'decay_linear60': make_decay_linear(60),
    'ts_min5': make_ts_min(5), 'ts_min10': make_ts_min(10), 'ts_min20': make_ts_min(20),
    # 'ts_min60': make_ts_min(60), 'ts_min120': make_ts_min(120), 'ts_min252': make_ts_min(252),
    'ts_max5': make_ts_max(5), 'ts_max10': make_ts_max(10), 'ts_max20': make_ts_max(20),
    # 'ts_max60': make_ts_max(60), 'ts_max120': make_ts_max(120), 'ts_max252': make_ts_max(252),
    'ts_argmin5': make_ts_argmin(5), 'ts_argmin10': make_ts_argmin(10), 'ts_argmin20': make_ts_argmin(20),
    # 'ts_argmin60': make_ts_argmin(60), 'ts_argmin120': make_ts_argmin(120), 'ts_argmin252': make_ts_argmin(252),
    'ts_argmax5': make_ts_argmax(5), 'ts_argmax10': make_ts_argmax(10), 'ts_argmax20': make_ts_argmax(20),
    # 'ts_argmax60': make_ts_argmax(60), 'ts_argmax120': make_ts_argmax(120), 'ts_argmax252': make_ts_argmax(252),
    'ts_rank5': make_ts_rank(5), 'ts_rank10': make_ts_rank(10), 'ts_rank20': make_ts_rank(20),
    # 'ts_rank60': make_ts_rank(60), 'ts_rank120': make_ts_rank(120), 'ts_rank252': make_ts_rank(252),
    'sum5': make_sum(5), 'sum10': make_sum(10), 'sum20': make_sum(20),
    # 'sum60': make_sum(60), 'sum120': make_sum(120), 'sum252': make_sum(252),
    # 'prod5': make_product(5), 'prod10': make_product(10), 'prod20': make_product(20),
    # 'prod60': make_product(60), 'prod120': make_product(120), 'prod252': make_product(252),
    'std5': make_stddev(5), 'std10': make_stddev(10), 'std20': make_stddev(20),
    # 'std60': make_stddev(60), 'std120': make_stddev(120), 'std252': make_stddev(252),
}

import numpy as np


class _Fitness(object):
    """A metric to measure the fitness of a program.

    This object is able to be called with NumPy vectorized arguments and return
    a resulting floating point score quantifying the quality of the program's
    representation of the true relationship.

    Parameters
    ----------
    function : callable
        A function with signature function(y, y_pred, sample_weight) that
        returns a floating point number. Where `y` is the input target y
        vector, `y_pred` is the predicted values from the genetic program, and
        sample_weight is the sample_weight vector.

    greater_is_better : bool
        Whether a higher value from `function` indicates a better fit. In
        general this would be False for metrics indicating the magnitude of
        the error, and True for metrics indicating the quality of fit.

    """

    def __init__(self, function, greater_is_better):
        self.function = function
        self.greater_is_better = greater_is_better
        self.sign = 1 if greater_is_better else -1

    def __call__(self, *args):
        return self.function(*args)


def _ic(y, y_pred):
    """Calculate the Pearson correlation coefficient."""
    with np.errstate(divide='ignore', invalid='ignore'):
        feature = y_pred.dropna(thresh=y_pred.shape[1]*0.5)
        feature = feature.sub(feature.mean(axis=1), 0).div(feature.std(axis=1), 0)
        y = y[y.index.isin(feature.index)]
        feat = feature[feature.index.isin(y.index)]
        corr = y.corrwith(feat, axis=1).fillna(0).mean()
    if np.isfinite(corr):
        return np.abs(corr)
    return 0.

def _ric(y, y_pred):
    """Calculate the weighted Spearman correlation coefficient."""
    # Check if any column in y or y_pred is constant
    with np.errstate(divide='ignore', invalid='ignore'):
        feature = y_pred.loc[(y_pred.apply(lambda x: x.nunique(), axis=1) > 1)]
        feature = feature.dropna(thresh=y_pred.shape[1]*0.5)
        feature = feature.sub(feature.mean(axis=1), axis=0).div(feature.std(axis=1), axis=0)
        y = y[y.index.isin(feature.index)]
        feat = feature[feature.index.isin(y.index)]
        corr = y.corrwith(feat, axis=1, method='spearman').fillna(0).mean()

    if np.isfinite(corr):
        return np.abs(corr)
    return 0.


ic = _Fitness(function=_ic,
              greater_is_better=True)
ric = _Fitness(function=_ric,
               greater_is_better=True)

_fitness_map = {'ic': ic,
                'ric': ric}

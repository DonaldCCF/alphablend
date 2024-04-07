import pandas as pd
from joblib import cpu_count
import numpy as np
import numbers

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.

    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def _get_n_jobs(n_jobs):
    """Get number of jobs for the computation.

    This function reimplements the logic of joblib to determine the actual
    number of jobs depending on the cpu count. If -1 all CPUs are used.
    If 1 is given, no parallel computing code is used at all, which is useful
    for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.
    Thus for n_jobs = -2, all CPUs but one are used.

    Parameters
    ----------
    n_jobs : int
        Number of jobs stated in joblib convention.

    Returns
    -------
    n_jobs : int
        The actual number of jobs as positive integer.

    """
    if n_jobs < 0:
        return max(cpu_count() + 1 + n_jobs, 1)
    elif n_jobs == 0:
        raise ValueError('Parameter n_jobs == 0 has no meaning.')
    else:
        return n_jobs


def _partition_estimators(n_estimators, n_jobs):
    """Private function used to partition estimators between jobs."""
    # Compute the number of jobs
    n_jobs = min(_get_n_jobs(n_jobs), n_estimators)

    # Partition estimators between jobs
    n_estimators_per_job = (n_estimators // n_jobs) * np.ones(n_jobs,
                                                              dtype=int)
    n_estimators_per_job[:n_estimators % n_jobs] += 1
    starts = np.cumsum(n_estimators_per_job)

    return n_jobs, n_estimators_per_job.tolist(), [0] + starts.tolist()


def _validate_data(X, y):
    """
    Ensures all values in the dictionary of DataFrames X and DataFrame Y are numeric,
    and that they all have the same columns and index. Raises an error if conditions are not met.

    Parameters:
    - X: dict of DataFrames
    - Y: DataFrame
    """
    # Ensure Y's values are numeric
    y = y.apply(pd.to_numeric, errors='coerce')

    for key in X:
        # Convert values to numeric
        X[key] = X[key].apply(pd.to_numeric, errors='coerce')

        # Check for mismatched columns or indices after reindexing
        if not X[key].columns.equals(y.columns) or not X[key].index.equals(y.index):
            raise ValueError(f"DataFrame {key} columns or index do not match Y after reindexing.")

    return X, y

def _calculate_correlation(dict_of_dfs):
    concatenated_df = pd.concat(
        [df.add_prefix(str(key) + '_') for key, df in dict_of_dfs.items()],
        axis=1
    )
    corr_matrix = concatenated_df.corr()
    result_corr_matrix = pd.DataFrame(index=dict_of_dfs.keys(), columns=dict_of_dfs.keys())
    for key1 in dict_of_dfs.keys():
        for key2 in dict_of_dfs.keys():
            if key1 == key2:
                result_corr_matrix.loc[key1, key2] = 1
            else:
                relevant_corr = corr_matrix.filter(like=key1 + '_', axis=0).filter(like=key2 + '_', axis=1)
                result_corr_matrix.loc[key1, key2] = relevant_corr.mean().mean()
    return result_corr_matrix

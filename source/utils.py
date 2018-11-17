import numpy as np

def reconstruct_1(ts, m, J):
    """
    Function which reconstructs z-vectors consequently without overlapping
    :param ts: The origin time series.
    :param m: The embedding dimensionality
    :param J: J is the delay parameter
    :return:
    """
    ts_length = ts.shape[0]
    n_samples = ts_length // (m + J)
    ts = ts[:n_samples * (m + J)]
    ts_reconstructed = ts.reshape((n_samples, m + J))
    if J > 0:
        ts_reconstructed = ts_reconstructed[:, :-J]
    return ts_reconstructed


def reconstruct_2(ts, m, J=None):
    """
    Function which reconstructs z-vectors with overlapping
    :param ts: The origin time series.
    :param m: The embedding dimensionality
    :param J: J is the delay parameter
    :return:
    """
    l = []
    if np.ndim(ts) == 1:
        # Reshape this array
        ts = ts.reshape(-1, 1)
        l.append(ts)
    for i in range(1, m+1):
        ts_sliced = ts[i:-(m-i+1)]
        l.append(ts_sliced)
    ts_reconstructed = np.concatenate(l, axis=1)
    return ts_reconstructed


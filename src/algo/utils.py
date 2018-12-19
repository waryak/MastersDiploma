import string
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
        l.append(ts[:-(m+1)])
    for i in range(1, m+1):
        ts_sliced = ts[i:-(m-i+1)]
        l.append(ts_sliced)
    ts_reconstructed = np.concatenate(l, axis=1)
    return ts_reconstructed


# TODO: REFACTOR THIS SHIT. WORKS OK BUT CODE IS NOT FOR HUMANS!!!!
def reconstruct_3(ts, m, template):
    """

    :param ts:
    :param m:
    :param template:
    :return:
    """
    template = template.astype(int)

    assert (m - 1) == len(template), "Discrepancy between"
    l = []
    if np.ndim(ts) == 1:
        # Reshape this array
        ts = ts.reshape(-1, 1)
        l.append(ts[:-sum(template)])
    for i in range(1, m):
        reconstruction_delay = sum(template[:i])
        start_slice = reconstruction_delay
        end_slice = sum(template) - reconstruction_delay
        if end_slice > 0:
            ts_sliced = ts[start_slice:-end_slice]
        else:
            ts_sliced = ts[start_slice:]
        l.append(ts_sliced)
    ts_reconstructed = np.concatenate(l, axis=1)
    return ts_reconstructed



# TODO: Bad solution - need smth else
def _int2base(x, base):
    """

    :param x:
    :param base:
    :return:
    """
    digs = string.digits + string.ascii_letters
    if x < 0:
        sign = -1
    elif x == 0:
        return digs[0]
    else:
        sign = 1

    x *= sign
    digits = []

    while x:
        digits.append(digs[int(x % base)])
        x = int(x / base)

    if sign < 0:
        digits.append('-')

    digits.reverse()

    return ''.join(digits)

def _base2int(x):
    digs = string.digits + string.ascii_letters
    return digs.find(x)

def generate_templates(n_dimension, max_distance, step, start_index, end_index):
    """

    :param n_dimension:
    :param max_distance:
    :param step:
    :param start_index:
    :param end_index:
    :return:
    """
    assert (end_index - start_index) % step == 0, "The template step %d does not fit indexes " %step
    shape = (int((end_index - start_index) / step), n_dimension)
    templates = np.ones(shape)
    for i, index in enumerate(range(start_index, end_index, step)):
        index = _int2base(x=index, base=max_distance)
        for j, number in enumerate(index[::-1]):
            number = _base2int(x=number)
            templates[i, -(j+1)] = number
    templates = templates.astype(int)
    return templates

def generate_arguments(templates, ts, m, n_lags):
    """

    :param templates:
    :param ts:
    :param m:
    :return:
    """
    arguments = []
    for template in templates:
        args = {}
        reconstructed_ts = reconstruct_3(ts=ts, m=m, template=template)
        name = "model"
        for delay in template:
            name = name + "_" + str(delay)
        args["z_vectors"] = reconstructed_ts
        args["model_name"] = name
        args["n_lags"] = n_lags
        arguments.append(args)
    return arguments




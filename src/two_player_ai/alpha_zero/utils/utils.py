import numba as nb
import numpy as np


@nb.jit(nopython=True, nogil=True, cache=True)
def normalize(array, tau=1):
    if tau == 1:
        return array / np.sum(array)
    else:
        tau_array = np.power(array, tau)
        return tau_array / np.sum(tau_array)


@nb.jit(nopython=True, nogil=True, cache=True)
def reshape(array, dims):
    return array.reshape(dims)

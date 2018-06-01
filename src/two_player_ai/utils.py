import numba as nb
import numpy as np
import time


@nb.jit(nopython=True, nogil=True, cache=True)
def normalize(array, tau=1):
    if np.any(array):
        if tau == 1:
            array = array / np.sum(array)
        else:
            tau_array = np.power(array, tau)
            array = tau_array / np.sum(tau_array)

    return array


@nb.jit(nopython=True, nogil=True, cache=True)
def flatten(array):
    return array.flatten()


@nb.jit(nopython=True, nogil=True, cache=True)
def reshape(array, dims):
    return array.reshape(dims)


class cached_property(object):
    def __init__(self, factory):
        self._attr_name = factory.__name__
        self._factory = factory

    def __get__(self, instance, owner):
        attr = self._factory(instance)
        setattr(instance, self._attr_name, attr)
        return attr


def benchmark(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print('%r (%r, %r) %2.2f sec' % (method.__name__, args, kw, te-ts))
        return result

    return timed

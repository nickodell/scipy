# cython: profile=False
# cython: linetrace=False
cimport cython
import numpy as np
cimport numpy as cnp

from scipy._lib.array_api_compat import numpy as numpy_namespace


cdef bint is_numpy_namespace(xp):
    return xp is numpy_namespace


cdef bint is_numpy_array(array):
    return isinstance(array, np.ndarray)

cdef asarray(xp, item):
    if is_numpy_namespace(xp):
        if is_numpy_array(item):
            # Already an array
            return item
        else:
            # A scalar. Dispatch this to numpy.
            return np.asarray(item)
    # A more complicated case, such as a non-numpy array
    # or non-numpy namespace
    return xp.asarray(item)

#@line_profiler.profile
def find_core_batch_shapes(xp, arrays: list, ndims: tuple):
    cdef list batch_shapes = []
    cdef list core_shapes = []
    cdef tuple shape
    cdef int ndim
    cdef object ndim_definition
    # Determine core and batch shapes
    N = len(arrays)
    if N != len(ndims):
        raise ValueError("lists not same size")
    for i in range(N):
        array = arrays[i]
        ndim_definition = ndims[i]
        array = None if array is None else asarray(xp, array)
        shape = () if array is None else array.shape

        if ndim_definition == "1|2":  # special case for `solve`, etc.
            ndim = 2 if array.ndim >= 2 else 1
        else:
            ndim = ndim_definition

        arrays[i] = array
        batch_shapes.append(shape[:-ndim] if ndim > 0 else shape)
        core_shapes.append(shape[-ndim:] if ndim > 0 else ())
    return batch_shapes, core_shapes

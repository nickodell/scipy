cimport cython
from libc cimport stdio, stdlib
from cpython cimport PyBytes_FromStringAndSize

import os
import tempfile

cpdef asarray(xp, array):
    return xp.asarray(array)

def find_core_batch_shapes(xp, arrays, ndims):
    # Determine core and batch shapes
    batch_shapes = []
    core_shapes = []
    for i, (array, ndim) in enumerate(zip(arrays, ndims)):
        array = None if array is None else asarray(xp, array)
        shape = () if array is None else array.shape

        if ndim == "1|2":  # special case for `solve`, etc.
            ndim = 2 if array.ndim >= 2 else 1

        arrays[i] = array
        batch_shapes.append(shape[:-ndim] if ndim > 0 else shape)
        core_shapes.append(shape[-ndim:] if ndim > 0 else ())
    return batch_shapes, core_shapes

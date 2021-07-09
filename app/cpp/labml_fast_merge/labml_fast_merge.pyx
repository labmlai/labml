# distutils: language = c++

import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def merge(np.ndarray[np.double_t, ndim=1] values,
          np.ndarray[np.double_t, ndim=1] last_step,
          np.ndarray[np.double_t, ndim=1] steps,
          double step_gap,
          double prev_last_step,
          int i,  # from_step
          ):
    cdef int j = i + 1
    cdef int length = values.shape[0]
    cdef double iw, jw
    while j < length:
        if last_step[j] - prev_last_step < step_gap or last_step[j] - last_step[j - 1] < 1e-3:  # merge
            iw = max(1., last_step[i] - prev_last_step)
            jw = max(1., last_step[j] - last_step[i])
            steps[i] = (steps[i] * iw + steps[j] * jw) / (iw + jw)
            values[i] = (values[i] * iw + values[j] * jw) / (iw + jw)
            last_step[i] = last_step[j]
            j += 1
        else:  # move to next
            prev_last_step = last_step[i]
            i += 1
            last_step[i] = last_step[j]
            steps[i] = steps[j]
            values[i] = values[j]
            j += 1

    return i + 1  # size after merging

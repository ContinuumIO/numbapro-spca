from __future__ import print_function, division
import numpy as np
from numbapro import cuda
import numba

def prefixsum(masks, indices, init=0, nelem=None):
    nelem = masks.size if nelem is None else nelem

    carry = init
    for i in range(nelem):
        indices[i] = carry
        if masks[i]:
            carry += 1

    indices[nelem] = carry
    return carry


def scatterprefix(indices, inarr, outarr, nelem=None):
    nelem = inarr.size if nelem is None else nelem
    assert indices.size > nelem
    for i in range(nelem):
        do_assign = False
        curidx = indices[i]
        rightidx = indices[i + 1]
        if curidx != rightidx:
            do_assign = True
        if do_assign:
            outarr[curidx] = inarr[i]


def mapfn(fn, arr, out, nelem=None):
    nelem = arr.size if nelem is None else nelem
    for i in range(nelem):
        out[i] = fn(arr[i])


def n_way_bucket(array, vmin, vmax, numbucket, buckets):
    indices = np.empty(shape=array.size + 1, dtype=np.intp)
    bucket_width = (vmax - vmin) / numbucket
    masks = np.empty(shape=array.size, dtype=np.bool)

    buckends = np.zeros(numbucket, dtype=np.intp)

    # Do bucketing
    for b in range(numbucket):
        lo = vmin + b * bucket_width
        hi = lo + bucket_width
        if b == numbucket - 1:
            functor = lambda x: x >= lo
        elif b == 0:
            functor = lambda x: x < hi
        else:
            functor = lambda x: hi > x >= lo
        mapfn(functor, array, masks)
        init = 0
        if b > 0:
            init = buckends[b - 1]
        end = prefixsum(masks, indices, init=init)
        buckends[b] = end
        scatterprefix(indices, array, buckets)

    return buckends


@cuda.autojit
def cuda_map_ge(arr, lo, res):
    i = cuda.grid(1)
    res[i] = arr[i] >= lo

@cuda.autojit
def cuda_map_lt(arr, hi, res):
    i = cuda.grid(1)
    res[i] = arr[i] < hi


@cuda.autojit
def cuda_map_within(arr, hi, lo, res):
    i = cuda.grid(1)
    res[i] = hi > arr[i] >= lo

@cuda.autojit
def cuda_prefixsum_base2(masks, indices, init, nelem):
    """
    Args
    ----
    nelem:
        Must be power of 2.

    Note
    ----
    Launch 2*nelem threads.  Support 1 block/grid.
    """
    sm = cuda.shared.array((1024,), dtype=numba.int64)
    tid = cuda.threadIdx.x

    # Preload
    if 2 * tid + 1 < nelem:
        sm[2 * tid] = masks[2 * tid]
        sm[2 * tid + 1] = masks[2 * tid + 1]

    # Up phase
    limit = nelem >> 1
    step = 1
    idx = tid * 2
    two_d = 1
    for d in range(3):
        offset = two_d - 1

        if tid < limit:
            sm[offset + idx + step] += sm[offset + idx]

        limit >>= 1
        idx <<= 1
        step <<= 1
        two_d <<= 1

    cuda.syncthreads()

    # Down phase

    if tid == 0:
        sm[nelem - 1] = 0


    cuda.syncthreads()

    # Writeback
    if 2 * tid + 1 < nelem:
        indices[2 * tid] = sm[2 * tid]
        indices[2 * tid + 1] = sm[2 * tid + 1]


def minmax(array):
    """
    Reduction
    """
    small = array[0]
    big = array[0]
    for i in range(1, array.size):
        if array[i] < small:
            small = array[i]
        elif array[i] > big:
            big = array[i]
    return small, big


def k_largest(array, k):
    limit = 8
    numbucket = array.size // k
    while numbucket > limit:
        numbucket = limit
        buckets = np.empty_like(array)
        vmin, vmax = minmax(array)
        buckends = n_way_bucket(array, vmin=vmin, vmax=vmax,
                                numbucket=numbucket, buckets=buckets)

        e = buckends[-1]
        for i in range(1, buckends.size):
            s = buckends[-i]
            if e - s >= k:
                break

        newset = buckets[s:e]
        tmparray = np.empty(newset.size, dtype=newset.dtype)
        tmparray[:] = newset
        array = tmparray
        numbucket = array.size // k

        del buckets

    if numbucket > 1:
        buckets = np.empty_like(array)

        vmin, vmax = minmax(array)
        buckends = n_way_bucket(array, vmin=vmin, vmax=vmax,
                                numbucket=numbucket, buckets=buckets)
        result = buckets[buckends[-2]:]
    else:
        result = array.copy()
    result.sort()
    return result[-k:]


def test_primitives():
    values = np.arange(10)
    masks = np.zeros(shape=values.size, dtype=np.bool)
    mapfn(lambda x: x % 2 == 0, values, masks)
    indices = np.zeros(shape=masks.size + 1, dtype=np.intp)
    nelem = prefixsum(masks, indices)

    print(indices)

    bucket = np.zeros(shape=nelem, dtype=values.dtype)
    scatterprefix(indices, values, bucket)
    print(bucket)


def test_k_bucket_largest():
    array = np.array(list(reversed(range(10000))), dtype=np.float64)
    print(array.size)
    k = 5
    got = k_largest(array, k)

    sortedarray = array.copy()
    sortedarray.sort()
    expect = sortedarray[-k:]

    print(got)
    print(expect)
    assert np.all(expect == got)


def test_prefixsum():
    values = np.arange(8)
    masks = np.ones(shape=values.size, dtype=np.int8)
    # mapfn(lambda x: x % 2 == 0, values, masks)
    indices = np.zeros(shape=masks.size + 1, dtype=np.intp)

    cuda_prefixsum_base2[1, values.size//2](masks, indices, 0, values.size)

    print(masks)
    print(indices)


def check_indices():
    for tid in range(8):
        limit = 4
        step = 1
        idx = tid * 2
        two_d = 1
        for d in range(4):
            offset = two_d - 1
            if tid < limit:
                print(tid, "write", offset + idx, offset + idx + step)
                #sm[offset + idx] += sm[offset + idx + step]

            limit //= 2
            idx *= 2
            step *= 2
            two_d *= 2


if __name__ == '__main__':
    # test_primitives()
    # test_k_bucket_largest()
    test_prefixsum()
    # check_indices()

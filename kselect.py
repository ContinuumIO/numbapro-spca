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
        curidx = indices[i]
        rightidx = indices[i + 1]
        if curidx != rightidx:
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
def cuda_prefixsum_base2(masks, indices, init, nelem, nround, nidx_out):
    """
    Prefix-sum building block.  Performs blocked prefixsum over blocks of
    1024 elements maximum.

    Args
    ----
    nelem:
        Number of element per bock. Must be power of 2 and <= 1024.
    nround:
        precomputed log2(nelem)

    Note
    ----
    Launch nelem/2 threads. Hardcoded to do 1024 element maximum due to
    shared memory limitation.
    """
    sm = cuda.shared.array((1024,), dtype=numba.int64)
    tid = cuda.threadIdx.x
    blkid = cuda.blockIdx.x
    block_offset = blkid * nelem

    # Preload
    if 2 * tid + 1 < nelem:
        sm[2 * tid] = masks[block_offset + 2 * tid]
        sm[2 * tid + 1] = masks[block_offset + 2 * tid + 1]

    # Up phase
    # This is a reduction.  Sum is stored in the last element.
    limit = nelem >> 1
    step = 1
    idx = tid * 2
    two_d = 1
    for d in range(nround):
        offset = two_d - 1

        if tid < limit:
            sm[offset + idx + step] += sm[offset + idx]

        limit >>= 1
        idx <<= 1
        step <<= 1
        two_d <<= 1

    # Down phase
    if tid == 0:
        # Write total of ones in mask
        nidx_out[blkid] = sm[nelem - 1]
        sm[nelem - 1] = init

    limit = 1
    step = nelem // 2
    two_d = nelem
    for d in range(nround):
        cuda.syncthreads()

        offset = two_d - 1
        idx = tid * two_d

        if tid < limit:
            storeidx = offset + idx
            swapidx = offset + idx - step
            stval = sm[storeidx]
            swval = sm[swapidx]
            sm[swapidx] = stval
            sm[storeidx] = stval + swval

        limit <<= 1
        step >>= 1
        two_d >>= 1

    cuda.syncthreads()

    # Writeback
    if 2 * tid + 1 < nelem:
        indices[block_offset + 2 * tid] = sm[2 * tid]
        indices[block_offset + 2 * tid + 1] = sm[2 * tid + 1]


@cuda.autojit
def cuda_prefixsum_fix(indices, blksums):
    tid = cuda.grid(1)
    blkid = cuda.blockIdx.x
    blksz = cuda.blockDim.x
    i = blksz + tid
    total = blksums[0]
    for j in range(1, blkid + 1):
        total += blksums[j]
    indices[i] += total


@numba.autojit
def prefixsum_fast(masks, indices, init, nelem):
    carry = init
    for i in range(nelem):
        indices[i] = carry
        if masks[i]:
            carry += 1

    indices[nelem] = carry
    return carry


def cuda_prefixsum(masks, indices):
    nelem = masks.size
    blksz = 1024
    nblk, remain = divmod(nelem, blksz)
    nround = np.log2(blksz)
    lastinit = 0

    # Uses the GPU to compute the prefixsum for blocks of 1024
    if nblk:
        stream = cuda.stream()
        blksums = cuda.device_array(shape=nblk, dtype=np.intp, stream=stream)
        cuda_prefixsum_base2[nblk, blksz // 2, stream](masks, indices, 0, blksz,
                                                       nround, blksums)
        blksums_host = blksums.copy_to_host(stream=stream)

        # Fix the count from the second block onwards
        if nblk > 1:
            cuda_prefixsum_fix[nblk - 1, blksz, stream](indices, blksums)

        stream.synchronize()
        lastinit = blksums_host.sum()

    # Uses the CPU to compute the remaining elements
    if remain:
        prefixsum_fast(masks[-remain:], indices[-remain - 1:], lastinit,
                       remain)
    else:
        indices[-1] = lastinit


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
    values = np.arange(1100)
    masks = np.ones(shape=values.size, dtype=np.int8)
    # mapfn(lambda x: x % 2 == 0, values, masks)
    indices = np.zeros(shape=masks.size + 1, dtype=np.intp)

    cuda_prefixsum(masks, indices)

    print(masks)
    print(indices)


if __name__ == '__main__':
    # test_primitives()
    # test_k_bucket_largest()
    test_prefixsum()

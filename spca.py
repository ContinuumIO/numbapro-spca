from __future__ import print_function, division
import sys
import numpy as np
import timeit
import math
from numbapro import cuda, float64, int16, int32
from numbapro.cudalib import curand, cublas


blas = cublas.Blas()

cached_input_file = "input.npy"


def generate_input():
    p = 1000
    n = 100
    X = np.random.randn(n, p)

    A = (1. / n) * X.T.dot(X)
    U, S, tmp = np.linalg.svd(A)
    A = U.dot(np.diag(S).dot(np.diag(1. / np.arange(1, p + 1)))).dot(
        np.conjugate(U.T))

    return A


def spca_unopt(Vd, epsilon=0.1, d=3, k=10):
    p = Vd.shape[0]
    numSamples = (4. / epsilon) ** d

    ##actual algorithm
    opt_x = np.zeros((p, 1))
    opt_v = -np.inf

    #GENERATE ALL RANDOM SAMPLES BEFORE
    C = np.random.randn(d, numSamples)

    for i in np.arange(1, numSamples + 1):

        #c = np.random.randn(d,1)
        #c = C[:,i-1]
        c = C[:, i - 1:i]
        c = c / np.linalg.norm(c)
        a = Vd.dot(c)

        #partial argsort in numpy?
        #if partial, kth largest is p-k th smallest
        #but need indices more than partial
        I = np.argsort(a, axis=0)
        val = np.linalg.norm(a[I[-k:]]) #index backwards to get k largest

        if val > opt_v:
            opt_v = val
            opt_x = np.zeros((p, 1))
            #print((opt_x[I[0:k]]).shape)
            #print((a[I[0:k]]/val).shape)
            opt_x[I[-k:]] = a[I[-k:], :] / val

    return opt_x


@cuda.jit("void(float64[:,:], int32)")
def norm_random_nums(C, d):
    i = cuda.grid(1)
    if i >= C.shape[1]:
        return

    c = C[:, i]
    sum = 0.0
    for j in range(d):
        cj = c[j]
        sum += cj * cj
    val = math.sqrt(sum)
    for j in range(d):
        c[j] /= val


@cuda.jit("void(float64[:,:], float64[:,:], float64[:, :])")
def batch_matmul(Vd, C, A):
    sampleIdx = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    sum = 0.0
    # Assume C.shape[0] is usually small
    for k in range(C.shape[0]):
        sum += Vd[tid, k] * C[k, sampleIdx]

    A[tid, sampleIdx] = sum


@cuda.jit("void(float64[::1], int32, int32)", device=True)
def swapf(ary, a, b):
    t = ary[a]
    ary[a] = ary[b]
    ary[b] = t


@cuda.jit("void(int16[::1], int32, int32)", device=True)
def swapi(ary, a, b):
    t = ary[a]
    ary[a] = ary[b]
    ary[b] = t


@cuda.jit("void(float64[:,:], int16[:,:], int16)")
def batch_k_selection(A, I, k):
    """QuickSelect
    """
    sampleIdx = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    # XXX: hardcoded array size for maximum capability
    values = cuda.shared.array(shape=1000, dtype=float64)
    indices = cuda.shared.array(shape=1000, dtype=int16)
    storeidx = cuda.shared.array(shape=1, dtype=int32)
    rightidx = cuda.shared.array(shape=1, dtype=int32)

    # Prefill cache
    values[tid] = A[tid, sampleIdx]
    indices[tid] = tid
    cuda.syncthreads()

    st = 0
    rst = 0
    n = A.shape[0]
    left = 0
    right = n - 1
    val = 0.0
    ind = 0
    while left < right:
    # for _ in range(1):
        st = -1
        rst = -1

        pivot = right  #(right + left + 1) // 2

        storeidx[0] = left
        rightidx[0] = 0

        pval = values[pivot]

        # Move pivot to the end
        # if tid == 0:
        #     print(7777777)
        #     print(left + 0)
        #     print(right + 0)
        #     print(pivot + 0)
        # swapf(values, right, pivot)
        # swapi(indices, right, pivot)
        cuda.syncthreads()

        # Compare
        if tid >= left and tid < right:
            val = values[tid]
            ind = indices[tid]
            if val < pval:
                st = cuda.atomic.add(storeidx, 0, 1)
            else:
                rst = cuda.atomic.add(rightidx, 0, 1)

        cuda.syncthreads()
        finalpivot = storeidx[0]

        if rst != -1:
            # Assign right partition index
            st = finalpivot + rst

        # Swap
        if st != -1 and st != tid:
            values[st] = val
            indices[st] = ind

        cuda.syncthreads()

        # Move pivot to final destination
        if tid == 0:
            swapf(values, finalpivot, right)
            swapi(indices, finalpivot, right)

        cuda.syncthreads()

        # Adjust range or done
        remain = n - finalpivot
        if remain == k:
            break
        elif remain > k:
            left = finalpivot + 1
        else:
            right = finalpivot - 1

    if tid < k:
        I[tid, sampleIdx] = indices[n - tid - 1]
        # I[tid, sampleIdx] = indices[tid]


@cuda.jit("void(float64[:,:], int16[:,:], float64[:])")
def batch_scatter_norm(A, I, aInorm):
    tid = cuda.grid(1)

    if tid >= I.shape[1]:
        return

    sum = 0.0
    for k in range(I.shape[0]):
        ind = I[k, tid]
        val = A[ind, tid]
        sum += val * val

    aInorm[tid] = math.sqrt(sum)


def calc_ncta1d(size, blksz):
    return size + (blksz - 1) // blksz


def gpu_slice(arr, col):
    """
    Missing feature in NumbaPro
    """
    from numbapro.cudadrv.driver import device_to_host, DeviceView

    strides = arr.strides
    s = col * strides[1]
    e = (col + 1) * strides[1]
    host = np.empty(shape=arr.shape[0], dtype=arr.dtype)
    view = DeviceView(arr.gpu_data, s, e)
    device_to_host(host, view, e - s)
    return host



def spca(Vd, epsilon=0.1, d=3, k=10):
    p = Vd.shape[0]
    initNumSamples = int((4. / epsilon) ** d)

    maxSize = 64000

    ##actual algorithm
    opt_x = np.zeros((p, 1))
    opt_v = -np.inf

    # Send Vd to GPU
    dVd = cuda.to_device(Vd)

    remaining = initNumSamples

    custr = cuda.stream()
    prng = curand.PRNG(stream=custr)

    while remaining:
        numSamples = min(remaining, maxSize)
        remaining -= numSamples

        # Prepare storage for vector A
        dA = cuda.device_array(shape=(Vd.shape[0], numSamples), order='F')
        dI = cuda.device_array(shape=(k, numSamples), dtype=np.int16, order='F')
        daInorm = cuda.device_array(shape=numSamples, dtype=np.float64)
        dC = cuda.device_array(shape=(d, numSamples), order='F')

        #GENERATE ALL RANDOM SAMPLES BEFORE
        # Also do normalization on the device
        prng.normal(dC.reshape(dC.size), mean=0, sigma=1)

        norm_random_nums[calc_ncta1d(dC.shape[1], 512), 512, custr](dC, d)
        #C = dC.copy_to_host()

        # Replaces: a = Vd.dot(c)
        # XXX: Vd.shape[0] must be within compute capability requirement
        # Note: this kernel can be easily scaled due to the use of num of samples
        #       as the ncta
        batch_matmul[numSamples, Vd.shape[0], custr](dVd, dC, dA)

        # Replaces: I = np.argsort(a, axis=0)
        # Note: the k-selection is dominanting the time
        batch_k_selection[numSamples, Vd.shape[0], custr](dA, dI, k)

        # Replaces: val = np.linalg.norm(a[I[-k:]])
        batch_scatter_norm[calc_ncta1d(numSamples, 512), 512, custr](dA, dI,
                                                                     daInorm)

        aInorm = daInorm.copy_to_host(stream=custr)

        custr.synchronize()

        for i in xrange(numSamples):
            val = aInorm[i]
            if val > opt_v:
                opt_v = val
                opt_x.fill(0)

                # Only copy what we need
                a = gpu_slice(dA, i).reshape(p, 1)
                Ik = gpu_slice(dI, i).reshape(k, 1)
                aIk = a[Ik]
                opt_x[Ik] = (aIk / val)

        # Free allocations
        del dA, dI, daInorm, dC

    return opt_x


def generate_input_file():
    A = generate_input()
    np.save(cached_input_file, A)


def check_result():
    dit = 3
    kit = 10
    A = np.load(cached_input_file)
    U, S, _ = np.linalg.svd(A)
    Vd = U[:, 0:dit].dot(np.diag(np.sqrt(S[0:dit])))
    cpu_opt_x = spca_unopt(Vd, d=dit, k=kit)
    gpu_opt_x = spca(Vd, d=dit, k=kit)

    Gopt = gpu_opt_x.T.dot(A.dot(gpu_opt_x))
    Copt = cpu_opt_x.T.dot(A.dot(cpu_opt_x))
    print("These should be close:", Copt, Gopt)


def benchmark():
    dit = 3
    kit = 10
    A = np.load(cached_input_file)
    U, S, _ = np.linalg.svd(A)
    Vd = U[:, 0:dit].dot(np.diag(np.sqrt(S[0:dit])))

    print(min(timeit.repeat(lambda: spca(Vd, d=dit, k=kit), repeat=3,
                            number=1)))
    # Best CPU time 7.05 seconds


def benchmarkLarge():
    A = np.load(cached_input_file)
    dmax = 5
    kmax = 50
    #SVD ONLY HERE
    p = A.shape[0]
    U, S, _ = np.linalg.svd(A)

    for dit in range(2, dmax + 1):
        Vd = U[:, 0:dit].dot(np.diag(np.sqrt(S[0:dit])))
        for kit in range(10, kmax + 1, 10):
            #eventually another loop for iterations
            #print(min(timeit.repeat(lambda: spca(Vd, d=dit, k=kit), repeat=3, number=1)))
            #print(min(timeit.repeat(lambda: spca_unopt(Vd, d=dit, k=kit), repeat=3, number=1)))

            t1 = timeit.default_timer()
            outGPU = spca(Vd, d=dit, k=kit)
            t2 = timeit.default_timer()
            outCPU = spca_unopt(Vd, d=dit, k=kit)
            t3 = timeit.default_timer()

            Gopt = outGPU.T.dot(A.dot(outGPU))
            Copt = outCPU.T.dot(A.dot(outCPU))

            print("%d, %d, %f, %f, %f, %f" %
                  (dit, kit, t2 - t1, t3 - t2, Gopt, Copt))


def main():
    if '--gen' in sys.argv:
        generate_input_file()
    elif '--benchL' in sys.argv:
        benchmarkLarge()
    elif '--bench' in sys.argv:
        benchmark()
    else:
        check_result()


def test_sorter():
    k = 3

    n = 10
    A = np.asfortranarray(np.random.rand(n, 1))
    # A = np.array([[0.31255729],
    #               [0.68038179],
    #               [0.1824953],
    #               [0.82793691],
    #               [0.05213435],
    #               [0.79801885],
    #               [0.4090768],
    #               [0.62787787],
    #               [0.03544625],
    #               [0.42592408]], dtype='float64')
    I = np.zeros(k, dtype='int16', order='F').reshape(k, 1)
    # I = np.zeros(n, dtype='int16', order='F').reshape(n, 1)
    print(A)

    expect = sorted(A.flatten().tolist())[-k:]
    batch_k_selection[1, n](A, I, k)

    print(I)
    print(A[I])
    got = A[I].flatten().tolist()

    print(expect)
    print(got)
    assert set(expect) == set(got)


if __name__ == '__main__':
    main()
    # test_sorter()

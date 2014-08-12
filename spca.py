from __future__ import print_function, division
import sys
import numpy as np
import timeit
import itertools
import math
from numbapro import cuda, float64, int16, int32
from numbapro.cudalib import curand, cublas, sorting

try:
    xrange
    zip = itertools.izip
except NameError:
    xrange = range

cached_input_file = "input.npy"


def generate_input():
    p = 5000
    n = 100
    X = np.random.randn(n, p)

    A = (1. / n) * X.T.dot(X)
    U, S, tmp = np.linalg.svd(A)
    A = U.dot(np.diag(S).dot(np.diag(1. / np.arange(1, p + 1)))).dot(
        np.conjugate(U.T))

    print(A.shape)
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
    tid = int32(cuda.threadIdx.x)
    ntid = int32(cuda.blockDim.x)

    remain = Vd.shape[0]

    offset = 0
    while tid < remain:
        j = tid + offset

        sum = 0.0
        for k in range(C.shape[0]):
            sum += Vd[j, k] * C[k, sampleIdx]

        A[j, sampleIdx] = sum

        remain -= ntid
        offset += ntid


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


@cuda.jit("void(float64[:,:], float64[:], int32)")
def batch_norm(A, aInorm, K):
    tid = cuda.grid(1)

    if tid >= A.shape[1]:
        return

    sum = 0.0
    for k in range(K):
        val = A[k, tid]
        sum += val * val

    aInorm[tid] = math.sqrt(sum)


def calc_ncta1d(size, blksz):
    return size + (blksz - 1) // blksz


def spca(Vd, epsilon=0.1, d=3, k=10):
    p = Vd.shape[0]
    initNumSamples = int((4. / epsilon) ** d)

    maxSize = 32000

    ##actual algorithm
    opt_x = np.zeros((p, 1))
    opt_v = -np.inf

    # Send Vd to GPU
    dVd = cuda.to_device(Vd)

    remaining = initNumSamples

    custr = cuda.stream()
    prng = curand.PRNG(stream=custr)

    rsstr1 = cuda.stream()
    rsstr2 = cuda.stream()
    sorter1 = sorting.Radixsort(dtype=np.float64, stream=rsstr1)
    sorter2 = sorting.Radixsort(dtype=np.float64, stream=rsstr2)

    while remaining:
        numSamples = min(remaining, maxSize)
        remaining -= numSamples

        # Prepare storage for vector A
        dA = cuda.device_array(shape=(Vd.shape[0], numSamples), order='F')
        dI = cuda.device_array(shape=(k, numSamples), dtype=np.int32,
                               order='F')
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
        batch_matmul[numSamples, 512, custr](dVd, dC, dA)

        # Replaces: I = np.argsort(a, axis=0)
        # Note: the k-selection is dominanting the time

        lastprogress = 0
        custr.synchronize()

        # Distribute the work to two streams

        async_dA1 = dA.bind(rsstr1)
        async_dI1 = dI.bind(rsstr1)
        async_dA2 = dA.bind(rsstr2)
        async_dI2 = dI.bind(rsstr2)
        selnext1 = sorter1.batch_argselect(dtype=dA.dtype,
                                           count=dA.shape[0],
                                           k=k, reverse=True)
        selnext2 = sorter2.batch_argselect(dtype=dA.dtype,
                                           count=dA.shape[0],
                                           k=k, reverse=True)

        def take2(it):
            it = iter(it)
            while True:
                yield next(it), next(it)

        def work(i, packed):
            (dA, dI, selnext, stream) = packed
            # print('read', i)
            subdA = dA[:, i]
            yield
            # print('compute', i)
            dIi = selnext(subdA)
            yield
            # print('write', i)
            dI[:, i].copy_to_device(dIi, stream=stream)
            yield

        def exhaust(*args):
            for _ in zip(*args):
                pass

        g1 = async_dA1, async_dI1, selnext1, rsstr1
        g2 = async_dA2, async_dI2, selnext2, rsstr2
        roundrobin = itertools.cycle([g1, g2])

        seq1 = (exhaust(work(*arg1), work(*arg2))
                for arg1, arg2
                in take2(zip(range(numSamples), roundrobin)))

        exhaust(seq1)

        # for i, (async_dA, async_dI, selnext, stream) \
        #         in zip(range(numSamples), roundrobin()):
        #     dIi = selnext(async_dA[:, i])
        #     async_dI[:, i].copy_to_device(dIi, stream=stream)
        #
        #     newprogress = i / numSamples
        #     if newprogress - lastprogress > 0.1:
        #         print('progress', newprogress)
        #         lastprogress = newprogress

        rsstr1.synchronize()
        rsstr2.synchronize()

        # Replaces: val = np.linalg.norm(a[I[-k:]])
        # batch_scatter_norm[calc_ncta1d(numSamples, 512), 512, custr](dA, dI,
        #                                                              daInorm)
        batch_norm[calc_ncta1d(numSamples, 512), 512, custr](dA, daInorm, k)

        aInorm = daInorm.copy_to_host(stream=custr)

        custr.synchronize()

        for i in xrange(numSamples):
            val = aInorm[i]
            if val > opt_v:
                opt_v = val
                opt_x.fill(0)

                # Only copy what we need
                Ik = dI[:, i].copy_to_host()
                aIk = dA[:k, i].copy_to_host().reshape(k, 1)
                opt_x[Ik] = (aIk / val)

        # Free allocations
        del dA, dI, daInorm, dC

    return opt_x


def generate_input_file():
    A = generate_input()
    np.save(cached_input_file, A)


def check_result():
    dit = 1
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

    print(min(timeit.repeat(lambda: spca(Vd, d=dit, k=kit), repeat=1,
                            number=1)))
    # Best CPU time 7.05 seconds


def benchmarkLarge():
    A = np.load(cached_input_file)
    dmax = 3
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


if __name__ == '__main__':
    main()

import sys
import numpy as np
import timeit
import math
from numbapro import cuda
from numbapro.cudalib import curand


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


def spca_unopt(A, epsilon=0.1, d=3, k=10):
    p = A.shape[0]

    U, S, _ = np.linalg.svd(A)

    #Vd = U[:,1:d+1].dot(np.diag(np.sqrt(S[1:d+1])))
    Vd = U[:, 0:d].dot(np.diag(np.sqrt(S[0:d])))
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

    return Vd, opt_x


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


def calc_ncta1d(size, blksz):
    return size + (blksz - 1) // blksz


def spca(A, epsilon=0.1, d=3, k=10):
    p = A.shape[0]

    U, S, _ = np.linalg.svd(A)

    #Vd = U[:,1:d+1].dot(np.diag(np.sqrt(S[1:d+1])))
    Vd = U[:, 0:d].dot(np.diag(np.sqrt(S[0:d])))
    numSamples = int((4. / epsilon) ** d)

    ##actual algorithm
    opt_x = np.zeros((p, 1))
    opt_v = -np.inf

    #GENERATE ALL RANDOM SAMPLES BEFORE
    # Also do normalization on the device
    dC = curand.normal(mean=0, sigma=1, size=(d * numSamples),
                       device=True).reshape(d, numSamples)
    norm_random_nums[calc_ncta1d(dC.shape[1], 512), 512](dC, d)
    C = dC.copy_to_host()

    for i in np.arange(1, numSamples + 1):
        c = C[:, i - 1:i]
        a = Vd.dot(c)

        #partial argsort in numpy?
        #if partial, kth largest is p-k th smallest
        #but need indices more than partial
        I = np.argsort(a, axis=0)
        Ik = I[-k:]     #index backwards to get k largest
        aIk = a[Ik]
        val = np.linalg.norm(aIk)

        if val > opt_v:
            opt_v = val
            opt_x.fill(0)
            opt_x[Ik] = (aIk / val)

    return Vd, opt_x


def generate_input_file():
    A = generate_input()
    np.save(cached_input_file, A)


def check_result():
    A = np.load(cached_input_file)
    Vd, opt_x = spca(A)

    r0 = np.linalg.norm(np.conjugate(Vd.T).dot(opt_x))
    r1 = np.conjugate(opt_x.T).dot(A.dot(opt_x))

    # Depend on input file
    xr0 = 1.11100453416
    xr1 = [[1.27317481]]
    print(r0, xr0)
    print(r1, xr1)


def benchmark():
    A = np.load(cached_input_file)
    print(min(timeit.repeat(lambda: spca(A), repeat=3, number=1)))
    # Best CPU time 7.05 seconds


def main():
    if '--gen' in sys.argv:
        generate_input_file()
    elif '--bench' in sys.argv:
        benchmark()
    else:
        check_result()


if __name__ == '__main__':
    main()

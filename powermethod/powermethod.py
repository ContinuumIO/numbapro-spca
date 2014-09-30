"""
References:
http://www.math.kent.edu/~reichel/courses/intr.num.comp.2/lecture21/evmeth.pdf


"""
import numpy as np


def rayleigh_quotient(A, x):
    top = A.dot(x).dot(x)
    return top / x.dot(x)


def normalize(x):
    return x / np.linalg.norm(x)#, ord=np.inf)


def powermethod(A, rtol=1e-20):
    N = A.shape[0]

    x0 = np.ones(N, dtype=A.dtype)

    # First iteration
    x1 = normalize(A.dot(x0))
    s1 = rayleigh_quotient(A, x1)

    # Second iteration
    x2 = normalize(A.dot(x1))
    s2 = rayleigh_quotient(A, x2)

    err = abs((s2 - s1) / s2)

    while err > rtol:
        s1 = s2
        x1 = x2

        x2 = normalize(A.dot(x1))
        s2 = rayleigh_quotient(A, x2)

        err = abs((s2 - s1) / s2)

    assert np.allclose(A.dot(x2), s2 * x2)
    return x2, s2


def main():
    # A = np.array([[4, 14, 0], [5, 13, 0], [1, 0, 2]])


    # This works for symmetric matrix
    # And seems to work for nonsymmetric matrix where all values are non-zero
    # and positive (?)
    # A = np.array([[4, 14, 1], [5, 13, 1], [1, 1, 2]])  # nonsymmetric and all>0
    A = np.array([[5, -2], [-2, 8]])  # symmetric

    S, U = np.linalg.eig(A)
    for i in range(S.size):
        print(S[i], U[:, i])

    print('=' * 80)

    # First

    x, s = powermethod(A)
    print('first', x, s)

    xcol = normalize(x.reshape(x.size, 1))
    assert np.allclose(A.dot(x), s * x)

    # Second

    B = A - s * xcol.dot(xcol.T)

    x, s = powermethod(B)
    print('second', x, s)

    assert np.allclose(A.dot(x), s * x)


    # p = 0
    # B = A - x.reshape(x.size, 1).dot(A[p].reshape(x.size, 1).T)
    # print(B)
    # x, s = powermethod(B)
    #
    # print('second', x, s)
    #
    # print(A.dot(x), s * x)
    # assert np.allclose(A.dot(x), s * x)


if __name__ == '__main__':
    main()

"""
TO READ
http://www.robots.ox.ac.uk/~sjrob/Teaching/EngComp/ecl4.pdf
"""

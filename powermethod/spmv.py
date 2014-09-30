import numpy as np
import scipy.sparse as ss


def spmv(A, x):
    n = len(x)
    assert A.shape[1] == n
    xrow = np.zeros_like(x)
    for i in range(n):
        colidx = A[i].nonzero()[1]  # col index of nonzero entries
        anz = A[i, colidx].todense().A.flatten()  # nz vals in row
        xnz = x[colidx]
        xrow[i] = np.sum(anz * xnz)
    return xrow


n = 10
A = ss.rand(n, n, density=0.33, format='csr', random_state=0)
print(A.todense())

x = np.arange(n, dtype=np.float32)
xrow = spmv(A, x)
assert np.allclose(xrow, A * x)


'''https://lucatrevisan.wordpress.com/tag/power-method/'''

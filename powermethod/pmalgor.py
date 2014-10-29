import numpy as np


class Shift(object):
    def __init__(self, *veclist):
        self.veclist = veclist

    def __call__(self, x):
        for v in self.veclist:
            x = self.deparallelize(x, v)
        return x

    def deparallelize(self, x, y):
        ny = y / np.linalg.norm(y)
        c = x.dot(y) * ny
        return x - c


class PowerMethod(object):
    def rayleigh_quotient(self, A, x):
        top = A.dot(x).dot(x)
        return top / x.dot(x)

    def normalize(self, x):
        return x / np.linalg.norm(x)#, ord=np.inf)

    def powermethod(self, A, rtol=1e-3, maxiter=20, shift=None,
                    geteigvalue=True, check=False):
        depar = Shift() if shift is None else Shift(*shift)
        ncol = A.shape[1]

        x0 = depar(np.ones(ncol, dtype=A.dtype))

        # First iteration
        x1 = self.normalize(depar(A.dot(x0)))

        # Second iteration
        x2 = self.normalize(depar(A.dot(x1)))
        if geteigvalue:
            s2 = self.rayleigh_quotient(A, x2)

        err = np.linalg.norm(x2 - x1)

        niter = 2
        while err > rtol and niter < maxiter:
            if geteigvalue:
                s1 = s2
            x1 = x2

            x2 = self.normalize(depar(A.dot(x1)))

            if geteigvalue:
                s2 = self.rayleigh_quotient(A, x2)

            err = np.linalg.norm(x2 - x1)
            niter += 1

        if check:
            assert geteigvalue
            lhs = A.dot(x2)
            rhs = s2 * x2
            diff = np.abs(lhs - rhs)
            maxdiff = np.max(diff)
            print("max diff", maxdiff)
            assert np.max(diff) < 0.01

        if geteigvalue:
            return x2, s2
        else:
            return x2, None

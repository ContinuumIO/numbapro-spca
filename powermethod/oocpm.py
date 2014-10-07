# Out-of-core SPMV
import tempfile
import os
import logging
from contextlib import contextmanager
from timeit import default_timer as timer
import numpy as np
import scipy.sparse as ss
import tables

logging.basicConfig(level=logging.INFO)

NDEFAULT = 1000

filters = tables.Filters(complib='blosc', complevel=5)


class CumulativeTime(object):
    def __init__(self, name):
        self.name = name
        self.duration = 0

    @contextmanager
    def time(self):
        ts = timer()
        yield
        te = timer()
        self.duration += te - ts

    def __repr__(self):
        return "~~~ {name} takes {duration}s".format(name=self.name,
                                                     duration=self.duration)


def create_matrix(n=NDEFAULT):
    h5file = tables.open_file("mat.h5", mode="w", title="matrix file")

    # Generate random data
    A = ss.rand(n, n, density=0.33, format='csr', random_state=0,
                dtype=np.float32)
    A = (A + A.T) / 2
    # print(A.todense())

    # Build containers
    ndata = A.getnnz()
    h5data = h5file.create_carray('/', "data", shape=(ndata,),
                                  atom=tables.Float32Atom(), filters=filters)
    h5indices = h5file.create_carray('/', "indices", shape=(ndata,),
                                     atom=tables.UInt64Atom(), filters=filters)
    h5indptr = h5file.create_carray('/', "indptr", shape=(n + 1,),
                                    atom=tables.UInt64Atom(), filters=filters)

    # Populate data
    h5data[...] = A.data
    h5indices[...] = A.indices
    h5indptr[...] = A.indptr

    print(h5data)
    print(h5indices)
    print(h5indptr)

    h5file.close()


def write_vector(name, n=NDEFAULT):
    h5file = tables.open_file(name, mode="w", title="vector file")
    h5data = h5file.create_carray('/', 'data', shape=(n,),
                                  atom=tables.Float32Atom(), filters=filters)
    return h5file, h5data


def read_vector(name):
    h5file = tables.open_file(name, mode="r")
    return h5file.root.data


def chunk(iterable, n=1):
    it = iter(iterable)
    while True:
        buf = []
        while len(buf) < n:
            try:
                v = next(it)
            except StopIteration:
                break
            else:
                buf.append(v)

        if not buf:
            raise StopIteration

        yield buf


class OocSparseRow(object):
    def __init__(self, data, indices, start, stop):
        self.data = data
        self.dtype = data.dtype
        self.indices = indices
        self.start = int(start)
        self.stop = int(stop)
        assert self.start == start
        assert self.stop == stop
        self.nnz = stop - start

    def __getitem__(self, col):
        for offset, colidx in enumerate(self.indices.iterrows(self.start,
                                                              self.stop)):
            if col == colidx:
                break
            elif col > colidx:
                return self.dtype.type(0)
        else:
            return self.dtype.type(0)

        return self.data[offset + self.start]

    def iternz(self):
        chunksize = 1000
        for cols, dats in self.iternz_chunks(chunksize):
            for icol, idat in zip(cols, dats):
                yield icol, idat

    def iternz_chunks(self, n=1000):
        base = self.start
        it = self.indices.iterrows(self.start, self.stop)
        for i, cols in enumerate(chunk(it, n=n)):
            size = len(cols)
            datbuf = self.data[base: base + size]
            base += size
            yield cols, datbuf


class OocSparseMat(object):
    def __init__(self, shape, data, indices, indptr):
        self.shape = shape
        self.dtype = data.dtype
        self.data = data
        self.indices = indices
        self.indptr = indptr

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.getrow(item)
        else:
            (row, col) = item
            return self.getrow(row)[col]

    def getrow(self, row):
        start = self.indptr[row]
        stop = self.indptr[row + 1]
        return OocSparseRow(self.data, self.indices, start, stop)

    def as_sparse_matrix(self):
        return ss.csr_matrix((self.data, self.indices, self.indptr))


def read_matrix():
    h5file = tables.open_file("mat.h5")
    h5data = h5file.root.data
    h5indices = h5file.root.indices
    h5indptr = h5file.root.indptr

    n = len(h5indptr) - 1
    mat = OocSparseMat((n, n), h5data, h5indices, h5indptr)
    return mat


time_chunked_spmv = CumulativeTime("chunked_spmv")
time_chunked_spmv_calc = CumulativeTime("chunked_spmv_calc")


def chunked_spmv(A, x, out):
    n = len(x)
    assert A.shape[1] == n, (A.shape, n)
    for i in range(n):
        srow = A[i]
        cum = out.dtype.type(0)
        with time_chunked_spmv.time():
            for colidx, aval in srow.iternz_chunks(60000):
                with time_chunked_spmv_calc.time():
                    cum += np.sum(aval * x[colidx])
            out[i] = cum
    return out


def vecdot(x, y):
    assert len(x) == len(y)
    return sum(x[i] * y[i] for i in range(len(x)))


def vecnorm(x):
    return np.sqrt(vecdot(x, x))


def rayleigh_quotient(A, x):
    try:
        _, tmppath = tempfile.mkstemp()
        tmpfile, temp = write_vector(tmppath)
        chunked_spmv(A, x, temp)
        top = vecdot(temp, x)
        return top / vecdot(x, x)
    finally:
        temp.close()
        tmpfile.close()
        os.remove(tmppath)


def normalize_inplace(x):
    nrm = vecnorm(x)
    for i in range(len(x)):
        x[i] /= nrm
    return x


def deparallelize_inplace(x, y):
    """
    y must be normalized
    """
    c = vecdot(x, y)
    for i in range(len(x)):
        x[i] -= c * y[i]
    return x


class Shift(object):
    def __init__(self, *veclist):
        self.veclist = veclist

    def __call__(self, x):
        for v in self.veclist:
            x = deparallelize_inplace(x, v)
        return x


class VecFactory(object):
    def __init__(self):
        self.files = []

    def next(self):
        fobj, vecobj = write_vector("vec{0}.h5".format(len(self.files)))
        self.files.append(fobj)
        return vecobj

    def close_all(self):
        for f in self.files:
            f.close()


class DoubleBuffer(object):
    def __init__(self, first, second):
        self.first = first
        self.second = second

    def swap(self):
        self.first, self.second = self.second, self.first


vecfac = VecFactory()


def powermethod(A, rtol=1e-10, shift=None, maxiter=10):
    logger = logging.getLogger("powermethod")
    logger.info("start")

    depar = Shift() if shift is None else Shift(*shift)

    buffer = DoubleBuffer(vecfac.next(), vecfac.next())
    buffer.first[...] = 1

    def iteration(A, x0, x1):
        x1 = normalize_inplace(depar(chunked_spmv(A, x0, x1)))
        s1 = rayleigh_quotient(A, x1)
        return x1, s1

    # First iteration
    niter = 1
    x1, s1 = iteration(A, buffer.first, buffer.second)
    buffer.swap()
    logger.info("#%d iteration | eigenvalue %s", niter, s1)

    # Second iteration
    niter += 1
    x2, s2 = iteration(A, buffer.first, buffer.second)
    buffer.swap()
    logger.info("#%d iteration | eigenvalue %s", niter, s1)

    err = abs((s2 - s1) / s2)

    while err > rtol and niter < maxiter:
        logger.info("#%d iteration | eigenvalue %s", niter, s2)
        s1 = s2
        x2, s2 = iteration(A, buffer.first, buffer.second)
        buffer.swap()
        err = abs((s2 - s1) / s2)
        niter += 1

    return x2, s2


def test():
    A = read_matrix()
    k = 2

    eigs = []

    time_poweriteration = CumulativeTime("power iteration")
    with time_poweriteration.time():
        for i in range(k):
            x, s = powermethod(A, shift=eigs)
            eigs.append(x)

            gold = A.as_sparse_matrix().dot(x)
            got = s * x

            maximum_difference = np.max(np.abs(gold - got))
            print(maximum_difference)

    print(time_chunked_spmv_calc)
    print(time_chunked_spmv)
    print(time_poweriteration)


if __name__ == '__main__':
    create_matrix()
    test()

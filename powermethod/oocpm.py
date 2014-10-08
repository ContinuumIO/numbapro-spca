# Out-of-core SPMV
import logging
from contextlib import contextmanager
from timeit import default_timer as timer
import numpy as np
import scipy.sparse as ss
import tables
# from numba import njit

logging.basicConfig(level=logging.INFO)

NDEFAULT = 5000

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
    # A = A + A.T
    # A /= 2
    A = A.dot(A.T)
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
time_chunked_spmv_io_chunk = CumulativeTime("chunked_spmv_io_chunk")
time_chunked_spmv_io_scatter = CumulativeTime("chunked_spmv_io_scatter")


class MatChunkCache(object):
    def __init__(self, mat):
        self.mat = mat
        self.rows = {}

    def getrow(self, row):
        if row not in self.rows:
            chunks = list(self.mat[row].iternz_chunks(10000))
            self.rows[row] = chunks

        return self.rows[row]


def vecdot(x, y):
    return np.dot(x, y)


def vecnorm(x):
    return np.sqrt(vecdot(x, x))


def normalize_inplace(x):
    nrm = vecnorm(x)
    x /= nrm
    return x


def deparallelize_inplace(x, y):
    """
    y must be normalized
    """
    c = vecdot(x, y)
    x -= c * y
    return x


class Shift(object):
    def __init__(self, *veclist):
        self.veclist = veclist

    def __call__(self, x):
        for v in self.veclist:
            x = deparallelize_inplace(x, v)
        return x


class DoubleBuffer(object):
    def __init__(self, first, second):
        self.first = first
        self.second = second

    def swap(self):
        self.first, self.second = self.second, self.first


class CacheSPMV(object):
    def __init__(self):
        self.cached = None

    def set(self, A, x, out):
        self.cached = A, x, out

    def get(self, A, x):
        if self.cached is not None:
            ca, cx, cout = self.cached
            if ca is A and cx is x:
                return cout

        else:
            return None


class PowerMethod(object):
    def __init__(self, A):
        self.A = A

        self.logger = logging.getLogger("powermethod")
        self.logger.info("start")
        self.chunkcache = MatChunkCache(A)


    def repeat(self, shift=None, rtol=1e-20, maxiter=20):
        self.niter = 0
        self.cachespmv = CacheSPMV()
        A = self.A
        buffer = DoubleBuffer(np.empty(A.shape[0], dtype=np.float32),
                              np.empty(A.shape[0], dtype=np.float32))
        buffer.first[...] = 1

        self.depar = Shift() if shift is None else Shift(*shift)

        # First iteration
        x1, s1 = self.iteration(A, buffer.first, buffer.second)
        buffer.swap()

        # Second iteration
        x2, s2 = self.iteration(A, buffer.first, buffer.second)
        buffer.swap()

        err = abs((s2 - s1) / s2)

        while err > rtol and self.niter < maxiter:
            s1 = s2
            x2, s2 = self.iteration(A, buffer.first, buffer.second)
            buffer.swap()
            err = abs((s2 - s1) / s2)

        self.logger.info("end")

        del self.cachespmv
        return x2, s2

    def iteration(self, A, x0, x1):
        self.niter += 1
        x1 = normalize_inplace(self.depar(self.chunked_spmv(A, x0, x1)))
        s1 = self.rayleigh_quotient(A, x1)
        self.logger.info("#%d iteration | eigenvalue %s", self.niter, s1)
        return x1, s1

    def rayleigh_quotient(self, A, x):
        temp = np.empty_like(x)
        self.chunked_spmv(A, x, temp)
        top = vecdot(temp, x)
        return top / vecdot(x, x)

    def chunked_spmv(self, A, x, out):
        chunkcache = self.chunkcache
        cachespmv = self.cachespmv

        cached = cachespmv.get(A, x)
        if cached is not None:
            out[:] = cached
            return out

        n = len(x)
        assert A.shape[1] == n, (A.shape, n)
        for i in range(n):
            cum = out.dtype.type(0)
            with time_chunked_spmv.time():
                with time_chunked_spmv_io_chunk.time():
                    chunks = chunkcache.getrow(i)

                with time_chunked_spmv_io_scatter.time():
                    ax = x[...]
                    grouped = [(ax[colidx], avals) for colidx, avals in chunks]

                with time_chunked_spmv_calc.time():
                    for scattered, aval in grouped:
                        cum += np.sum(aval * scattered)
                    out[i] = cum

        cachespmv.set(A, x, out)
        return out


def test():
    A = read_matrix()
    k = 2

    eigs = []

    time_poweriteration = CumulativeTime("power iteration")

    powermethod = PowerMethod(A)
    for i in range(k):
        with time_poweriteration.time():
            x, s = powermethod.repeat(maxiter=20, shift=eigs)
        eigs.append(x)

        if A.shape[0] < 1000:
            gold = A.as_sparse_matrix().dot(x)
            got = s * x

            diff = np.abs(gold - got)
            print('diff (mean, max)', np.mean(diff), np.max(diff))

    print(time_chunked_spmv_io_chunk)
    print(time_chunked_spmv_io_scatter)
    print(time_chunked_spmv_calc)
    print(time_chunked_spmv)
    print(time_poweriteration)


if __name__ == '__main__':
    create_matrix()
    test()

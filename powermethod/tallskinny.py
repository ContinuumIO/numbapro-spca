from __future__ import print_function, absolute_import
from timeit import default_timer as timer
from contextlib import contextmanager
from collections import namedtuple
import itertools
import sys
import os
import numpy as np
import scipy.sparse as ss
from multiprocessing import Process, Queue
from numbapro.cudalib.cusparse import Sparse
from numbapro import cuda
from utils import CumulativeTime

# node_count = 42889799
node_count = 42890000

RANDOMIZE = len(sys.argv) == 4
CHECK_ANSWER = False

if RANDOMIZE:
    WIDTH = int(sys.argv[1])
    RANDSIZE = int(sys.argv[2])
    # 'mt', 'gpu', 'cpu'
    STRATEGY = sys.argv[3]
else:
    assert len(sys.argv) == 3
    WIDTH = sys.argv[1]
    STRATEGY = sys.argv[2]
print('STRATEGY', STRATEGY)

# Number of processes for MT strategy
NPROC = 2

bm_spmv = CumulativeTime("SPMV")
bm_powermethod = CumulativeTime("PowerMethod")


def read_column(filename, dtype=np.float32):
    """Read a column file
    """
    shape = node_count, 1
    if os.path.isfile(filename):
        with open(filename) as fin:
            ii = np.array(list(map(int, fin)))
            data = np.ones_like(ii, dtype=dtype)
            jj = np.zeros_like(ii)
    else:
        data = np.empty(0, dtype=dtype)
        ii = np.empty(0, dtype=np.intp)
        jj = np.empty(0, dtype=np.intp)

    return ss.coo_matrix((data, (ii, jj)), shape=shape)


def create_matrix(files):
    """Build matrix by stacking column files
    """
    cols = [read_column(f) for f in files]
    empty_col = ss.coo_matrix(([], ([], [])),
                              shape=(node_count, node_count - len(cols)))
    cols.append(empty_col)
    return ss.csr_matrix(ss.hstack(cols))


def stack_matrix(files):
    cols = [read_column(f) for f in files]
    return ss.csr_matrix(ss.hstack(cols))


def rand_matrix(n):
    """
    Randomize symmetric matrix
    """
    ndata = int((n ** 2) * 0.33)
    if ndata % 2 != 0:
        ndata += 1

    data = np.ones(ndata, np.float32)

    row = np.zeros(ndata, dtype=np.intp)
    col = np.zeros(ndata, dtype=np.intp)

    row[:ndata // 2] = np.random.randint(0, n, size=ndata // 2)
    col[:ndata // 2] = np.random.randint(0, n, size=ndata // 2)
    row[ndata // 2:] = col[:ndata // 2]
    col[ndata // 2:] = row[:ndata // 2]

    mat = ss.csr_matrix((data, (row, col)), shape=(n, n))
    return mat


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

    def powermethod(self, A, rtol=1e-10, maxiter=20, shift=None):
        depar = Shift() if shift is None else Shift(*shift)
        ncol = A.shape[1]

        x0 = depar(np.ones(ncol, dtype=A.dtype))

        # First iteration
        x1 = self.normalize(depar(A.dot(x0)))
        s1 = self.rayleigh_quotient(A, x1)

        # Second iteration
        x2 = self.normalize(depar(A.dot(x1)))
        s2 = self.rayleigh_quotient(A, x2)

        err = abs((s2 - s1) / s2)

        niter = 2
        while err > rtol and niter < maxiter:
            s1 = s2
            x1 = x2

            x2 = self.normalize(depar(A.dot(x1)))
            s2 = self.rayleigh_quotient(A, x2)

            err = abs((s2 - s1) / s2)
            niter += 1

        if CHECK_ANSWER:
            lhs = A.dot(x2)
            rhs = s2 * x2
            diff = np.abs(lhs - rhs)
            maxdiff = np.max(diff)
            print("max diff", maxdiff)
            assert np.max(diff) < 0.01
        return x2, s2


def proc_initialize(inqueue, outqueue, gpunum):
    from numbapro import cuda

    cuda.select_device(gpunum)
    cp = ChildProcess(inqueue, outqueue)
    cp.run()


class ChildProcess(object):
    def __init__(self, inqueue, outqueue):
        self.inqueue = inqueue
        self.outqueue = outqueue
        self.cusp = Sparse()

    def run(self):
        cuda._profile_start()

        while True:
            task = self.inqueue.get()
            descr = task[0]
            if descr == 'stop':
                return
            elif descr == 'initvec':
                self.vector = cuda.to_device(task[1])
                self.dev_out = cuda.to_device(np.zeros_like(task[1]))

            elif descr == 'compute':

                matrix, start, stop = task[1:]

                m, n = matrix.shape
                nnz = matrix.getnnz()

                descr = self.cusp.matdescr()

                self.cusp.csrmv('N', m, n, nnz, 1, descr, matrix.data,
                                matrix.indptr, matrix.indices,
                                self.vector[start:stop], 1, self.dev_out)

            elif descr == 'return':
                self.outqueue.put(self.dev_out.copy_to_host())

            else:
                raise NotImplementedError(descr)


class BigSparseMatrix(object):
    def __init__(self, mat):
        self.mat = mat
        self.shape = mat.shape
        self.dtype = mat.dtype

        if STRATEGY == 'gpu':
            self.stream = cuda.stream()
            self.cusp = Sparse()
            self.cusp.stream = self.stream

        if STRATEGY == 'mt':
            self.numprocesses = NPROC
            self.outqueues = [Queue() for _ in range(self.numprocesses)]
            self.inqueues = [Queue() for _ in range(self.numprocesses)]
            self.processes = [Process(target=proc_initialize,
                                      args=(outq, inq, 0))
                              for outq, inq in
                              zip(self.outqueues, self.inqueues)]
            for p in self.processes:
                p.start()

    def __del__(self):
        if STRATEGY == 'mt':
            for q in self.outqueues:
                q.put(('stop',))
            for p in self.processes:
                p.join()

    def naive_dot(self, vector):
        assert vector.ndim == 1
        return self.mat.dot(vector)

    def cpu_dot(self, vector):
        with bm_spmv.time():
            assert vector.ndim == 1
            ncols = vector.size
            width = WIDTH
            acum = np.zeros_like(vector)
            # Compute the dot product as a sum of all tall-skinny submatrices,
            # which multiple columns of the matrix
            assert ncols // width * width == ncols, "not evenly divided"
            for i in range(ncols // width):
                start = i * width
                stop = (i + 1) * width
                submat = self.get_sub_matrix(start, stop)
                if submat.getnnz() > 0:
                    assert submat.shape == (ncols, width)
                    subvec = vector[start:stop]
                    acum += submat.dot(subvec)

            return acum

    def gpu_dot(self, vector):
        with bm_spmv.time():
            assert vector.ndim == 1
            ncols = vector.size
            width = WIDTH
            acum = np.zeros_like(vector)
            dev_acum = cuda.to_device(acum, stream=self.stream)
            dev_vector = cuda.to_device(vector, stream=self.stream)
            descr = self.cusp.matdescr()
            # Compute the dot product as a sum of all tall-skinny submatrices,
            # which multiple columns of the matrix
            assert ncols // width * width == ncols, "not evenly divided"
            for i in range(ncols // width):
                start = i * width
                stop = (i + 1) * width

                subvec = dev_vector[start:stop]

                submat = self.get_sub_matrix(start, stop)
                if submat.getnnz() > 0:
                    assert submat.shape == (ncols, width)

                    # CUDA Sparse MV
                    m, n = submat.shape
                    nnz = submat.getnnz()

                    dev_data = cuda.to_device(submat.data, stream=self.stream)
                    dev_indptr = cuda.to_device(submat.indptr,
                                                stream=self.stream)
                    dev_indices = cuda.to_device(submat.indices,
                                                 stream=self.stream)

                    self.cusp.csrmv('N', m, n, nnz, 1, descr, dev_data,
                                    dev_indptr, dev_indices, subvec, 1,
                                    dev_acum)

            return dev_acum.copy_to_host()

    def mt_gpu_dot(self, vector):
        for outq in self.outqueues:
            outq.put(('initvec', vector))

        with bm_spmv.time():
            assert vector.ndim == 1
            ncols = vector.size
            width = WIDTH
            acum = np.zeros_like(vector)
            # Compute the dot product as a sum of all tall-skinny submatrices,
            # which multiple columns of the matrix
            assert ncols // width * width == ncols, "not evenly divided"

            for i, outq in zip(range(ncols // width),
                               itertools.cycle(self.outqueues)):
                start = i * width
                stop = (i + 1) * width

                submat = self.get_sub_matrix(start, stop)
                if submat.getnnz() > 0:
                    assert submat.shape == (ncols, width)

                    outq.put(('compute', submat, start, stop))

            for outq in self.outqueues:
                outq.put(('return',))

            for inq in self.inqueues:
                acum += inq.get()

            return acum

    def cpu_dot_check(self, vector):
        got = self.cpu_dot(vector)
        exp = self.naive_dot(vector)
        maxdiff = np.max(np.abs(got - exp))
        assert maxdiff < 0.001, "maxdiff %f" % maxdiff
        return got

    def gpu_dot_check(self, vector):
        got = self.gpu_dot(vector)
        exp = self.naive_dot(vector)
        maxdiff = np.max(np.abs(got - exp))
        assert maxdiff < 0.001, "maxdiff %f" % maxdiff
        return got

    def mt_gpu_dot_check(self, vector):
        got = self.mt_gpu_dot(vector)
        exp = self.naive_dot(vector)
        maxdiff = np.max(np.abs(got - exp))
        assert maxdiff < 0.001, "maxdiff %f" % maxdiff
        return got

    def inmemory_get_sub_matrix(self, start, stop):
        return self.mat[:, start:stop]

    def infile_get_sub_matrix(self, start, stop):
        print('row {0} {1}%'.format(start, int(stop / node_count * 100)))
        fmt = "data/arc-{0:d}.dat"
        files = [fmt.format(n) for n in range(start, stop)]
        return stack_matrix(files)

    get_sub_matrix = inmemory_get_sub_matrix if RANDOMIZE else infile_get_sub_matrix

    if STRATEGY == 'mt':
        dot = mt_gpu_dot
    elif STRATEGY == 'gpu':
        dot = gpu_dot
    elif STRATEGY == 'cpu':
        dot = cpu_dot


def main():
    if RANDOMIZE:
        mat = rand_matrix(RANDSIZE)
    else:
        mat = namedtuple("dummy_matrix", ['dtype', 'shape'])(dtype=np.float32,
                                                             shape=(node_count,
                                                                    node_count))

    mat = BigSparseMatrix(mat)

    print('powermethod')
    with bm_powermethod.time():
        pm = PowerMethod()
        eigvec1, eigval1 = pm.powermethod(mat)
        print("first eigenvalue", eigval1)
        # eigvec2, eigval2 = pm.powermethod(mat, shift=[eigvec1])
        # print("second eigenvalue", eigval2)

    print(bm_spmv)
    print(bm_powermethod)


if __name__ == '__main__':
    main()

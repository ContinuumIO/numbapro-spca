# Out-of-core SPMV
import logging
from contextlib import contextmanager
from timeit import default_timer as timer
import threading
import numpy as np
import scipy.sparse as ss
import tables
from numbapro import cuda
import sys
import queue
from collections import defaultdict


logging.basicConfig(level=logging.INFO)

# NDEFAULT = 2000
# NDEFAULT = 43*1000000 # 43 million
NDEFAULT = 10000
CHUNKSIZE = 100000
MAXITERATION = 6

FILE = "mat.h5"
# FILE = 'arc1m.h5'
# FILE = 'pld-arc-50k.h5'

filters = tables.Filters(complib='blosc', complevel=5)


def WorkerClass():
    return Worker()


WORKER_COUNT = 1


class KeepStat(object):
    def __init__(self):
        self.max = 0
        self.min = float('+inf')
        self.avg = 0
        self.bins = {1000: 0, 5000: 0, 10000: 0, 50000: 0}
        self.total = 0

    def update(self, x):
        self.max = max(self.max, x)
        self.min = min(self.min, x)
        self.avg = (self.avg + x) / 2
        for k in sorted(self.bins.keys()):
            if x <= k:
                self.bins[k] += 1
                break
        self.total += 1

    def __str__(self):
        return str((self.max, self.min, self.avg, self.bins, self.total))


stat_colidx = KeepStat()
stat_avals = KeepStat()
#
#
# class Worker(object):
#     def __init__(self):
#         self.tasks = []
#         self.out = {}
#
#     # Can overide
#     def allocate(self):
#         """Called after adding task to finalize resource allocation
#         """
#         pass
#
#     # Can overide
#     def add(self, row, chunks):
#         self.tasks.append((row, chunks))
#
#     # Can overide
#     def start(self, x):
#         for row, chunks in self.tasks:
#             c = 0
#             for colidx, avals in chunks:
#                 # maintain stats
#                 stat_colidx.update(len(colidx))
#                 stat_avals.update(len(avals))
#
#                 with time_scattering.time():
#                     scattered = x[colidx]
#                 with time_spmv_calc.time():
#                     c += np.sum(avals * scattered)
#             self.out[row] = c
#
#     def join(self):
#         return self.out
#
#


# cuda.select_device(0)

#
# class Worker(object):
#     def __init__(self):
#         self.tasks = []
#         self.out = {}
#
#     # Can overide
#     def allocate(self):
#         """Called after adding task to finalize resource allocation
#         """
#         self.fast_avals = cuda.pinned_array(CHUNKSIZE, dtype=np.float32)
#         # self.fast_colidx = cuda.pinned_array(CHUNKSIZE, dtype=np.intp)
#         self.fast_x = cuda.pinned_array(CHUNKSIZE, dtype=np.float32)
#
#         self.stream = stream = cuda.stream()
#         self.dout = cuda.device_array(CHUNKSIZE, dtype=np.float32,
#                                       stream=stream).bind(stream=stream)
#
#         self.davals = cuda.device_array(CHUNKSIZE, dtype=np.float32,
#                                         stream=stream).bind(stream=stream)
#         self.dx = cuda.device_array(CHUNKSIZE, dtype=np.float32,
#                                     stream=stream).bind(stream=stream)
#         # self.dcolidx = cuda.device_array(CHUNKSIZE, dtype=np.intp,
#         #                                  stream=stream).bind(stream=stream)
#
#         self.jobqueue = queue.Queue()
#         self.resqueue = queue.Queue()
#         self.gpuworker = threading.Thread(target=self.gpu_thread,
#                                           args=(self.jobqueue, self.resqueue),
#                                           daemon=True)
#         self.gpuworker.start()
#
#     # Can overide
#     def add(self, row, chunks):
#         self.tasks.append((row, chunks))
#
#     # Can overide
#     def start(self, x):
#         jobcount = 0
#         # self.dx = cuda.to_device(x, stream=self.stream)
#         for row, chunks in self.tasks:
#             c = 0
#             for colidx, avals in chunks:
#                 # maintain stats
#                 stat_colidx.update(len(colidx))
#                 stat_avals.update(len(avals))
#
#                 if avals.size < 1000:
#                     # CPU
#                     with time_scattering.time():
#                         scattered = x[colidx]
#                     with time_spmv_calc.time():
#                         c += np.sum(avals * scattered)
#                 else:
#                     # GPU
#                     self.jobqueue.put((row, avals, colidx, x[colidx]))
#                     jobcount += 1
#
#             self.out[row] = c
#
#         # Wait for all job to complete
#         print('using gpu', jobcount)
#         for _ in range(jobcount):
#             row, cpartial = self.resqueue.get()
#             self.out[row] += cpartial
#
#         # Wait for jobs to complete
#         self.jobqueue.join()
#
#     def gpu_thread(self, inqueue, outqueue):
#         fast_avals = self.fast_avals
#         # fast_colidx = self.fast_colidx
#         stream = self.stream
#         dout = self.dout
#
#         fast_x = self.fast_x
#
#         while True:
#             task = inqueue.get()
#             if task is StopIteration:
#                 return
#             row, avals, colidx, x = task
#             # Get memory slice
#
#             tmp_avals = fast_avals[:avals.size]
#             tmp_avals[:] = avals
#
#             # tmp_colidx = fast_colidx[:len(colidx)]
#             # tmp_colidx[:] = colidx
#
#             tmp_x = fast_x[:x.size]
#             tmp_x[:] = x
#
#             davals = self.davals[:tmp_avals.size]
#             dx = self.dx[:tmp_x.size]
#             # dcolidx = self.dcolidx[:tmp_colidx.size]
#
#             # H->D
#             davals.copy_to_device(tmp_avals, stream=stream)
#             dx.copy_to_device(tmp_x, stream=stream)
#             # dcolidx.copy_to_device(tmp_colidx, stream=stream)
#
#             # Compute
#             # scatter_multiply_for = scatter_multiply.forall(
#             #     dcolidx.size, stream=stream)
#             # scatter_multiply_for(davals, self.dx, dcolidx, dout)
#
#             multiply_for = multiply.forall(dx.size, stream=stream)
#             multiply_for(davals, dx, dout)
#             sz = sum_reduce.device_partial_inplace(dout,
#                                                    size=dx.size,
#                                                    stream=stream)
#             cpartial = dout[:sz].copy_to_host().sum()
#             # Task Done
#             inqueue.task_done()
#             outqueue.put((row, cpartial))
#
#     def join(self):
#         return self.out


class CPUWork(object):
    def __init__(self, colidx, avals):
        self.colidx = np.array(colidx)
        self.avals = avals

    def __call__(self, x, dx):
        with time_scattering.time():
            scattered = x[self.colidx]
        with time_spmv_calc.time():
            return np.sum(self.avals * scattered)


from asynctodevice import AsyncToDevice, AsyncToHost

cuda.select_device(0)
async_to_device = AsyncToDevice()
async_to_host = AsyncToHost()


class GPUWork(object):
    def __init__(self, colidx, avals, stream):
        self.dcolidx = async_to_device.to_device(np.array(colidx),
                                                 stream=stream)
        self.davals = async_to_device.to_device(avals, stream=stream)

        self.dout = cuda.device_array(self.dcolidx.size, dtype=np.float32,
                                      stream=stream)
        self.stream = stream

    def __call__(self, x, dx):
        dcolidx = self.dcolidx
        davals = self.davals
        dout = self.dout
        stream = self.stream
        async_to_device.join()
        # dx = cuda.to_device(x, stream=stream)

        scatter_multiply_for = scatter_multiply.forall(
            dcolidx.size, stream=stream)
        scatter_multiply_for(davals, dx, dcolidx, dout)
        sz = sum_reduce.device_partial_inplace(dout,
                                               size=dcolidx.size,
                                               stream=stream)
        final = dout.bind(stream=stream)[:sz]
        # out = async_to_host.to_host(final, stream=stream)
        return final.copy_to_host().sum()


class Worker(object):
    def __init__(self):
        self.tasks = []
        self.gpu_count = 0
        self.stream = cuda.stream()

    # Can overide
    def allocate(self):
        print('gpu_count', self.gpu_count)

    # Can overide
    def add(self, row, chunks):
        THRESHOLD = 5000

        for colidx, avals in chunks:

            stat_colidx.update(len(colidx))
            stat_avals.update(len(avals))
            if len(colidx) >= THRESHOLD:
                work = GPUWork(colidx, avals, stream=self.stream)
                self.gpu_count += 1
            else:
                work = CPUWork(colidx, avals)

            self.tasks.append((row, work))

    # Can overide
    def start(self, x):
        dx = async_to_device.to_device(x, self.stream)
        pending = defaultdict(list)
        self.out = defaultdict(float)
        for row, work in self.tasks:
            res = work(x, dx)
            if isinstance(res, float):
                self.out[row] += res
            else:
                pending[row].append(res)

        async_to_host.join()
        for k, vl in pending.items():
            for v in vl:
                self.out[k] += v.sum()

    def join(self):
        return self.out


@cuda.autojit
def scatter_multiply(avals, x, colidx, out):
    i = cuda.grid(1)
    if i >= colidx.size:
        return
    scattered = x[colidx[i]]
    out[i] = scattered * avals[i]

#
# @cuda.autojit
# def multiply(avals, x, out):
#     i = cuda.grid(1)
#     if i >= x.size:
#         return
#     out[i] = x[i] * avals[i]


@cuda.reduce
def sum_reduce(a, b):
    return a + b


class GpuWorker(Worker):
    def __init__(self):
        self.tasks = []
        self.out = {}

    def add(self, row, chunks):
        dchunks = []
        for colidx, aval in chunks:
            dchunks.append((np.array(colidx), aval))

        self.tasks.append((row, dchunks))

        # Can overide

    def allocate(self):
        """Called after adding task to finalize resource allocation
        """
        self.stream = cuda.stream()

    # Can overide
    def start_runner(self, x):
        stream = self.stream
        logger = logging.getLogger(threading.current_thread().name)

        for row, chunks in self.tasks:
            if row % 5000 == 0:
                logger.info("row %d", row)
            c = 0
            for dcolidx, davals in chunks:
                dout = cuda.device_array(dcolidx.size, dtype=np.float32,
                                         stream=stream)
                assert dcolidx.size <= dout.size
                dx = cuda.to_device(x, stream=stream)
                scatter_multiply_for = scatter_multiply.forall(
                    dcolidx.size, stream=stream)
                scatter_multiply_for(davals, dx, dcolidx, dout)
                sz = sum_reduce.device_partial_inplace(dout,
                                                       size=dcolidx.size,
                                                       stream=stream)
                c += dout.bind(stream=stream)[:sz].copy_to_host().sum()
            self.out[row] = c

    def start(self, x):
        self.thread = threading.Thread(target=self.start_runner, args=(x,))
        self.thread.start()

    def join(self):
        self.thread.join()
        self.stream.synchronize()
        return self.out


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
    logger = logging.getLogger('create_matrix')
    h5file = tables.open_file("mat.h5", mode="w", title="matrix file")

    # Generate random data
    logger.info("build random matrix A")
    A = ss.rand(n, n, density=0.33, format='csr', random_state=0,
                dtype=np.float32)

    logger.info("compute symmetric matrix")
    A = A + A.T
    A /= 2

    # A = A.dot(A.T)
    # print(A.todense())

    # Build containers
    logger.info("build containers")
    ndata = A.getnnz()
    h5data = h5file.create_carray('/', "data", shape=(ndata,),
                                  atom=tables.Float32Atom(), filters=filters)
    h5indices = h5file.create_carray('/', "indices", shape=(ndata,),
                                     atom=tables.UInt64Atom(), filters=filters)
    h5indptr = h5file.create_carray('/', "indptr", shape=(n + 1,),
                                    atom=tables.UInt64Atom(), filters=filters)

    # Populate data
    logger.info("storing")
    h5data[...] = A.data
    h5indices[...] = A.indices
    h5indptr[...] = A.indptr

    print(h5data)
    print(h5indices)
    print(h5indptr)

    logger.info("done")
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
    h5file = tables.open_file(FILE)
    h5data = h5file.root.data
    h5indices = h5file.root.indices
    h5indptr = h5file.root.indptr

    n = len(h5indptr) - 1
    mat = OocSparseMat((n, n), h5data, h5indices, h5indptr)

    return mat


class MatChunkCache(object):
    def __init__(self, mat):
        self.mat = mat
        self.rows = {}

    def getrow(self, row):
        if False:
            if row not in self.rows:
                chunks = list(self.mat[row].iternz_chunks(CHUNKSIZE))
                self.rows[row] = chunks

            return self.rows[row]
        else:
            chunks = list(self.mat[row].iternz_chunks(CHUNKSIZE))
            return chunks


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


time_reading_chunks = CumulativeTime("reading_chunks")
time_scattering = CumulativeTime("scattering")
time_spmv_calc = CumulativeTime("spmv_calc")
time_spmv = CumulativeTime("spmv")


class PowerMethod(object):
    def __init__(self, A):
        self.A = A
        self.logger = logging.getLogger("powermethod")
        self.chunkcache = MatChunkCache(A)

        # init workers
        self.logger.info("initialize workers")

        self.workers = [WorkerClass() for i in range(WORKER_COUNT)]
        #
        # def schedule_workers():
        #     while True:
        #         for w in self.workers:
        #             yield w
        #
        # for row, worker in zip(range(self.A.shape[0]), schedule_workers()):
        #     if (row % int(A.shape[0] / 100)) == 0:
        #         self.logger.info("reading row %d", row)
        #     with time_reading_chunks.time():
        #         worker.add(row, self.chunkcache.getrow(row))

        for row in range(self.A.shape[0]):
            if A.shape[0] <= 100 or (row % int(A.shape[0] / 100)) == 0:
                self.logger.info("reading row %d (%d%%)", row,
                                 100 * row / A.shape[0])

            worker = self.workers[row % len(self.workers)]
            with time_reading_chunks.time():
                worker.add(row, self.chunkcache.getrow(row))

        for worker in self.workers:
            worker.allocate()

    def repeat(self, shift=None, rtol=1e-20, maxiter=20):
        self.logger.info("start")
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
        cachespmv = self.cachespmv

        cached = cachespmv.get(A, x)
        if cached is not None:
            out[:] = cached
            return out

        with time_spmv.time():
            for worker in self.workers:
                worker.start(x)

            for worker in self.workers:
                for row, val in worker.join().items():
                    out[row] = val

        cachespmv.set(A, x, out)
        return out


def test():
    A = read_matrix()
    k = 1

    eigs = []

    time_poweriteration = CumulativeTime("power iteration")

    with time_poweriteration.time():
        powermethod = PowerMethod(A)

    for i in range(k):
        with time_poweriteration.time():
            x, s = powermethod.repeat(maxiter=MAXITERATION, shift=eigs)
        eigs.append(x)

        if A.shape[0] <= 1000:
            logging.info("checking result")
            gold = A.as_sparse_matrix().dot(x)
            got = s * x

            diff = np.abs(gold - got)
            print('diff (mean, max)', np.mean(diff), np.max(diff))

    print(time_reading_chunks)
    print(time_scattering)
    print(time_spmv_calc)
    print(time_spmv)

    print(time_poweriteration)

    print('colidx', stat_colidx)
    print('avals', stat_avals)

    print("writing eigenvectors")
    for ek, ei in enumerate(eigs):
        ei.dump("eig{0}.dat".format(ek))


if __name__ == '__main__':
    import sys

    entry = sys.argv[1]
    print("entry", entry)
    globals()[entry]()

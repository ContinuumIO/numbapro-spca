from __future__ import print_function, division
import os
import zipfile
import logging
import numpy as np
from collections import deque
from utils import CumulativeTime
from pmalgor import PowerMethod
from numbapro import cuda

logger = logging.getLogger(__name__)

if True:
    # testing
    numfiles = 105 + 1  # last file ID + 1
    fileformat = 'data/{0:d}'
    indexfile = 'example_index'
    numnodes = 105 + 1
else:
    numfiles = 27343 + 1  # last file ID + 1
    fileformat = 'bigdata/{0:d}'
    indexfile = 'pld-index'
    numnodes = 42889799 + 1

CACHE_LIMIT = 10000


def avail_file_set():
    outset = set()
    for i in range(numfiles):
        path = fileformat.format(i)
        if os.path.isfile(path):
            outset.add(path)
    return outset


def read_file(path, retdata=True):
    if retdata:
        logger.info("reading file %s", path)
    else:
        logger.info("indexing file %s", path)
    with zipfile.ZipFile(path, 'r', compression=zipfile.ZIP_DEFLATED) as zipf:
        cols = {}
        names = zipf.namelist()
        for i, colname in enumerate(names):
            colid = int(colname)
            if retdata:
                with zipf.open(colname) as colfile:
                    nzarg = np.array([int(x) for x in colfile.readlines()])
                    cols[colid] = nzarg
            else:
                cols[colid] = ()

        return cols


def get_nz_cols(paths):
    nzcols = {}
    for path in paths:
        for col in read_file(path, retdata=False).keys():
            nzcols[col] = path
    return nzcols


class Column(object):
    def __init__(self, col, data):
        self.col = col
        if data is None:
            data = np.array([])
            assert data.size == 0
        self.data = data

    @property
    def matrix(self):
        return self.data


class CacheEntry(object):
    def __init__(self, rank, data):
        self.rank = rank
        self.data = data

    def use(self, col):
        return self.data[col]


class ColCache(object):
    def __init__(self, limit):
        self._limit = limit
        self._cache = {}

    def get(self, path, col):
        """Load column from file or from cache
        """
        if path in self._cache:
            return self._cache[path].use(col)
        else:
            if len(self._cache) >= self._limit:
                # Cache is full, evict the lowest ranking entry
                lfu = min((ent.rank, key) for key, ent in self._cache.items())
                del self._cache[lfu[1]]

            timing = CumulativeTime(path)
            with timing.time():
                cols = read_file(path, retdata=True)
            print(timing)
            self._cache[path] = ent = CacheEntry(rank=timing.duration,
                                                 data=cols)
            return ent.use(col)


class CuUpdate(object):
    def __init__(self):
        # 50-MB
        self._pinned_size = int(10 * 1e6)
        self._pinned_mems = []
        self._device_mems = []
        self._streams = []
        self._events = []

        for _ in range(4):
            pm = cuda.pinned_array(self._pinned_size, dtype=np.intp)
            stm = cuda.stream()
            dm = cuda.device_array_like(pm).bind(stream=stm)
            self._pinned_mems.append(pm)
            self._device_mems.append(dm)
            self._streams.append(stm)
            self._events.append(cuda.event())

        self._queue = deque(
            zip(self._streams, self._pinned_mems, self._device_mems,
                self._events))

    def cycle(self):
        out = self._queue.popleft()
        self._queue.append(out)
        return out

    def work(self, dev_accum, indices, value):
        # Chop it up into partition that fits in our pinned memory
        for start in range(0, indices.size, self._pinned_size):
            stop = min(start + self._pinned_size, indices.size)
            size = stop - start
            assert size < self._pinned_size
            self._work(dev_accum, indices[start:stop], value)

    def _work(self, dev_accum, indices, value):
        assert indices.size < self._pinned_size
        # Find stream that is ready
        while True:
            stm, pmem, dmem, evt = self.cycle()
            if evt.query():
                break

        # Real work
        size = indices.size
        ## Transfer to pinned
        pmem[:size] = indices
        ## Transfer to device memory
        dev_indices = dmem[:size]
        dev_indices.copy_to_device(pmem, stream=stm)
        # Record the H2D event
        evt.record(stream=stm)
        ## Call kernel
        cukernel = cuda_update.forall(size, stream=stm)
        cukernel(dev_accum, dev_indices, value)

    def synchronize(self):
        for stm in self._streams:
            stm.synchronize()


@cuda.autojit
def cuda_update(acum, idx, scal):
    tid = cuda.grid(1)
    if tid < idx.size:
        acum[idx[tid]] += scal


cu_update = CuUpdate()
bm_dot = CumulativeTime("dot")
bm_mod = CumulativeTime("modify")


class ColMatrix(object):
    def __init__(self):
        self.dtype = np.float32
        self._availfiles = avail_file_set()
        self._nzcols = get_nz_cols(self._availfiles)
        self.shape = (numnodes, numnodes)
        self._cache = ColCache(limit=CACHE_LIMIT)

    def getnzcols(self):
        return list(self._nzcols)

    def getcol(self, col):
        path = self._nzcols.get(col, None)
        if path is None:
            return Column(col, None).matrix
        else:
            return Column(col, self._cache.get(path, col)).matrix


class DotBase(object):
    pass


class CpuDot(DotBase):
    def dot(self, vector):
        assert vector.ndim == 1
        acum = np.zeros_like(vector)
        print("DOT".center(80, '='))

        zeroidxvals = 0
        with bm_dot.time():
            tavg = 0
            chunk = 1000
            for start in range(0, numnodes, chunk):
                stop = min(start + chunk, numnodes)
                bm_round = CumulativeTime("one round")
                logger.info("dot %d/%d (%.1f%%)", start, numnodes,
                            start / numnodes * 100)

                with bm_round.time():
                    for colid in range(start, stop):
                        minrate = 1 / numnodes
                        nzrows = self.getcol(colid)
                        scalar = vector[colid]

                        if nzrows.size > 0:
                            acum[nzrows] += (1 - minrate) * scalar

                        zeroidxvals += scalar * minrate

                tavg = (tavg + bm_round.duration) / 2
                hrs = (tavg * (numnodes - stop) / chunk) / (60 ** 2)
                days = hrs / 24
                logger.info("remaining %.2f hours %.2f days",
                            hrs, days)

            return acum + zeroidxvals


class GpuDot(DotBase):
    def dot(self, vector):
        assert vector.ndim == 1

        dev_acum = cuda.device_array_like(vector)
        cuda.driver.device_memset(dev_acum, 0,
                                  dev_acum.size * dev_acum.dtype.itemsize)

        print("DOT".center(80, '='))

        with bm_dot.time():
            tavg = 0
            chunk = 100000

            zeroidxvals = 0
            for start in range(0, numnodes, chunk):
                stop = min(start + chunk, numnodes)
                bm_round = CumulativeTime("one round")
                logger.info("dot %d/%d (%.1f%%)", start, numnodes,
                            start / numnodes * 100)

                with bm_round.time():
                    for colid in range(start, stop):
                        minrate = 1 / numnodes
                        nzrows = self.getcol(colid)
                        scalar = vector[colid]

                        if nzrows.size > 0:
                            incr = (1 - minrate) * scalar
                            cu_update.work(dev_acum, nzrows, incr)

                        # Always increment the zeroidxvals
                        zeroidxvals += scalar * minrate

                tavg = (tavg + bm_round.duration) / 2
                hrs = (tavg * (numnodes - stop) / chunk) / (60 ** 2)
                mins = hrs * 60
                days = hrs / 24
                logger.info("remaining %.2f mins %.2f hours %.2f days",
                            mins, hrs, days)

            cu_update.synchronize()
            return dev_acum.copy_to_host() + zeroidxvals


class GpuMatrix(ColMatrix, GpuDot):
    pass


if __name__ == '__main__':
    import topranking

    logging.basicConfig(level=logging.INFO)

    cm = GpuMatrix()
    pm = PowerMethod()

    try:
        eigvec, eigval = pm.powermethod(cm, maxiter=3, geteigvalue=False,
                                        check=False)

        print("Done".center(80, '='))
    except KeyboardInterrupt:
        print(bm_dot)
        print(bm_mod)
        raise
    else:
        print(bm_dot)
        print(bm_mod)

    for i in reversed(topranking.toprank(list(eigvec.argsort()[-10:]),
                                         filename=indexfile)):
        print(i)


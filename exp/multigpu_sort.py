from numbapro.cudalib.sorting import RadixSort
from numbapro import cuda, jit
import numpy as np
from timeit import default_timer as timer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

ngpus = len(cuda.gpus)
K = 1000
logger.info("# of GPUs: %s", ngpus)

np.random.seed(0)

cuda.select_device(0)
MAX = 75000000

with cuda.gpus[0]:
    stream0 = cuda.stream()
    sorter0 = RadixSort(maxcount=MAX, stream=stream0, dtype=np.float32)

with cuda.gpus[2]:
    stream1 = cuda.stream()
    sorter1 = RadixSort(maxcount=MAX, stream=stream1, dtype=np.float32)


def make_data(N):
    global A
    A = np.random.random_integers(low=0, high=N, size=N).astype(np.float32)
    logger.info('data size %s', A.data.nbytes / 2 ** 20)


class Work(object):
    def __init__(self, arg):
        A, sorter, stream = arg
        assert A.size < MAX
        self.stream = stream
        self.sorter = sorter
        self.A = A
        self.result = np.empty(K, dtype=A.dtype)

    def to_device(self):
        self.dA = cuda.to_device(self.A, stream=self.stream)

    def to_host(self):
        kchunk = self.dA.bind(self.stream)[-K:]
        kchunk.copy_to_host(self.result, stream=self.stream)

    def work(self):
        self.sorter.sort(self.dA)

    @classmethod
    def stages(cls):
        return [cls.to_device, cls.work, cls.to_host]


def schedule(gpus):
    for gid in gpus:
        with cuda.gpus[gid]:
            yield cuda.get_current_device()


def mgpu_schedule(workcls, args, gpus):
    workers = []
    for gpu, arg in zip(schedule(gpus), args):
        logger.info(gpu)
        workers.append(workcls(arg))

    for stage in workers[0].stages():
        for gpu, worker in zip(schedule(gpus), workers):
            stage(worker)

    return workers


@jit(nopython=True)
def mergesort(A, B, C):
    i = j = k = 0
    while i < A.size and j < B.size:
        if A[i] <= B[j]:
            C[k] = A[i]
            i += 1
        else:
            C[k] = B[j]
            j += 1
        k += 1

    while i < A.size:
        C[k] = A[i]
        i += 1
        k += 1

    while j < B.size:
        C[k] = B[j]
        j += 1
        k += 1


sig = "void(float32[::1], float32[::1], float32[::1])"
cmergesort = mergesort.compile(sig)


def multi_gpu():
    gpus = [0, 2]

    results = []
    half = A.size // 2
    ts = timer()
    args0 = A[:half], sorter0, stream0
    args1 = A[half:], sorter1, stream1
    workers = mgpu_schedule(Work, [args0, args1], gpus)

    for gpu, worker in zip(schedule(gpus), workers):
        worker.stream.synchronize()
        results.append(worker.result)

    output = np.empty(results[0].size + results[1].size,
                      dtype=results[0].dtype)
    cmergesort(results[0], results[1], output)
    te = timer()
    logger.info(output[-K:])
    return te - ts


def single_gpu():
    gpus = [0]
    results = []
    ts = timer()
    workers = mgpu_schedule(Work, [(A, sorter0, stream0)], gpus)
    for gpu, worker in zip(schedule(gpus), workers):
        worker.stream.synchronize()
        results.append(worker.result)
    te = timer()
    assert len(results) == 1
    logger.info(results[0])
    return te - ts


def main():
    step = 5000000
    N = step
    while N < MAX * 2:
        logging.info("N = %s", N)
        make_data(N)

        # Start
        try:
            t_one = single_gpu()
        except BaseException as e:
            logging.warning("single gpu failed %s", e)
            t_one = float('nan')
        logger.info("single gpu time: %s", t_one)
        t_two = multi_gpu()
        logger.info("multi gpu time: %s", t_two)
        print('\t'.join(map(str, [N, t_one, t_two])))

        N += step


if __name__ == '__main__':
    main()

from contextlib import contextmanager
from timeit import default_timer as timer
from numbapro.cudalib.cusparse import Sparse
from numbapro import cuda
import scipy.sparse as ss
import numpy as np
import sys


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


bm_gpu = CumulativeTime("gpu")
bm_cpu = CumulativeTime("cpu")

cusp = Sparse()

# random matrix
m, n = 42889799, int(sys.argv[1])
print(m, n)
print("randomize matrix")
mat = ss.rand(m, n, density=0.33, format='csr', dtype=np.float32,
              random_state=0)
print("randomize vector")
vec = np.random.rand(n).astype(np.float32)

for i in range(1):
    print("compute round", i)
    print("gpu")
    with bm_gpu.time():
        dev_out = cuda.device_array(m, dtype=np.float32)
        dscr = cusp.matdescr()

        cusp.csrmv('N', m, n, mat.getnnz(), 1, dscr, mat.data, mat.indptr,
                   mat.indices, vec, 0, dev_out)

        gpuout = dev_out.copy_to_host()

    print("cpu")
    with bm_cpu.time():
        cpuout = mat.dot(vec)

assert np.allclose(gpuout, cpuout)

print(bm_cpu)
print(bm_gpu)

"""
Usage:
    python plotter.py <filename>

Example:
    python plotter.py benchmark/bm1_gtx780ti_f64.txt

Data are expected to be in the form:
    d, k, gpu time, cpu time, gpu result, cpu result
"""
from __future__ import print_function, division, absolute_import
from collections import defaultdict
import sys
import numpy as np
from matplotlib import pyplot as plt


def plot(filename):
    dbin = defaultdict(list)
    with open(filename) as fin:
        lines = list(fin)

        # Sanitize input
        for line in lines:
            d, k, tgpu, tcpu, resgpu, rescpu = map(lambda s: s.strip(),
                                                   line.split(','))

            k = int(k)
            d = int(d)
            tgpu = float(tgpu)
            tcpu = float(tcpu)

            # Ignoring resgpu and rescpu for now
            del resgpu, rescpu

            dbin[d].append((k, tgpu, tcpu))

    ax = plt.subplot()

    for d, datapts in dbin.items():
        n = len(datapts)
        xarr = np.zeros(n, dtype='int')
        yspeedup = np.zeros(n, dtype='float')
        for i, (k, tgpu, tcpu) in enumerate(datapts):
            xarr[i] = k
            yspeedup[i] = tcpu / tgpu
        ax.plot(xarr, yspeedup, '-x', label=("d=%d" % d))
        # ygpu = np.zeros(n, dtype='float')
        # ycpu = np.zeros(n, dtype='float')
        #
        # for i, (k, tgpu, tcpu) in enumerate(datapts):
        #     xarr[i] = k
        #     ygpu[i] = tgpu
        #     ycpu[i] = tcpu

        # ax.plot(xarr, ygpu, label=("gpu d=%d" % d))
        # ax.plot(xarr, ycpu, label=("cpu d=%d" % d))

    ax.set_ylabel("Speedup over CPU")
    ax.set_xlabel("Solution Sparsity (parameter k)")
    ax.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    filename = sys.argv[1]
    print("file", filename)
    plot(filename=filename)


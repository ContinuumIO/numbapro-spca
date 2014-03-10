"""
Usage:
    python plotter.py <filename>

Example:
    python plotter.py benchmark/bm1_gtx780ti_f64.txt

Data are expected to be in the form:
    k, d, gpu time, cpu time, gpu result, cpu result
"""
from __future__ import print_function, division, absolute_import
from collections import defaultdict
import sys
import numpy as np
from matplotlib import pyplot as plt


def plot(filename):
    kbin = defaultdict(list)
    with open(filename) as fin:
        lines = list(fin)

        # Sanitize input
        for line in lines:
            k, d, tgpu, tcpu, resgpu, rescpu = map(lambda s: s.strip(),
                                                   line.split(','))

            k = int(k)
            d = int(d)
            tgpu = float(tgpu)
            tcpu = float(tcpu)

            # Ignoring resgpu and rescpu for now
            del resgpu, rescpu

            kbin[k].append((d, tgpu, tcpu))

    ax = plt.subplot()

    for k, datapts in kbin.items():
        n = len(datapts)
        xarr = np.zeros(n, dtype='int')
        yspeedup = np.zeros(n, dtype='float')
        for i, (d, tgpu, tcpu) in enumerate(datapts):
            xarr[i] = d
            yspeedup[i] = tcpu / tgpu
        ax.plot(xarr, yspeedup, label=("k=%d" % k))
        # ygpu = np.zeros(n, dtype='float')
        # ycpu = np.zeros(n, dtype='float')
        #
        # for i, (d, tgpu, tcpu) in enumerate(datapts):
        #     xarr[i] = d
        #     ygpu[i] = tgpu
        #     ycpu[i] = tcpu

        # ax.plot(xarr, ygpu, label=("gpu k=%d" % k))
        # ax.plot(xarr, ycpu, label=("cpu k=%d" % k))

    ax.set_ylabel("Speedup over CPU")
    ax.set_xlabel("Accuracy (parameter d)")
    ax.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    filename = sys.argv[1]
    print("file", filename)
    plot(filename=filename)


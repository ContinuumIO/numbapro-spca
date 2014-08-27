from __future__ import print_function, absolute_import, division
import math
from contextlib import contextmanager
import functools
import numpy as np
import scipy
import matplotlib.pyplot as plt
import networkx as nx
from timeit import default_timer as timer

import wdc_dataset as wdc
import progress

float_dtype = np.dtype(np.float32)


@contextmanager
def benchmark(name):
    print("--- {name}".format(name=name))
    ts = timer()
    yield
    te = timer()
    print("=== {name} takes {duration}s".format(name=name, duration=te - ts))


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


def compute_Vd(A, d):
    """Form Vd with dense SVD
    Args
    ----
    A: numpy array
        Adjacency matrix
    d: int
        rank
    """
    U, D, _ = np.linalg.svd(A)
    return U[:, :d].dot(np.diag(np.sqrt(D[:d])))


def compute_sparse_Vd(A, d):
    """Form Vd with sparse SVD

    Args
    ----
    A: scipy sparse array
        Adjacency matrix
    d: int
        rank
    """
    U, D, _ = scipy.sparse.linalg.svds(A, k=d)
    return U[:, :d].dot(np.diag(np.sqrt(D[:d])))


def spannogram_Dks_eps_psd(k, V, eps, delta):
    """
    Args
    ----
    k: int
        number of nodes in the output subgraph

    V: numpy array
        Rank-d approximation matrix

    eps, delta: float
        Error

    Returns
    -------

    A tuple of (metric, subgraph node index)

    Assumption
    -----------
    - Adjacency matrix (A) is large

    """
    n, d = V.shape

    supp_opt = 0
    metric_opt = 0

    Mopt = (1 + 4 / eps) ** d
    M = (math.log(delta) - math.log(Mopt)) / math.log(1 - 1 / Mopt)
    print(M)
    print('V.shape =', V.shape)

    t_dot = CumulativeTime("dot")
    t_argsort = CumulativeTime("argsort")
    t_norm = CumulativeTime("norm-scatter-square")

    count = int(round(M))
    for i in range(count):
        progress.render_progress(i / count, width=50)

        c = np.random.randn(d, 1)

        # Do matrix multiplication
        with t_dot.time():
            Vc = V.dot(c)

        # Do sort
        with t_argsort.time():
            indx = np.argsort(Vc, axis=0)

        # Get last k
        topk = indx[-k:]
        # Assume A is huge
        with t_norm.time():
            metric = np.linalg.norm(V[topk, :]) ** 2
        if metric > metric_opt:
            metric_opt = metric
            supp_opt = topk
    print()
    print(t_dot)
    print(t_argsort)
    print(t_norm)
    return metric_opt, supp_opt


class ParallelSpannogram(object):
    def __init__(self, V, k, threshold):
        self.Vcs = []
        self.threshold = threshold
        self.k = k
        self.V = V

    def append(self, Vc):
        self.Vcs.append(Vc)

    def process(self, metric_opt, supp_opt):
        if len(self.Vcs) >= self.threshold:
            return self.flush(metric_opt, supp_opt)
        else:
            return metric_opt, supp_opt

    def flush(self, metric_opt, supp_opt):
        k = self.k
        V = self.V
        for Vc in self.Vcs:
            # Do sort
            indx = np.argsort(Vc, axis=0)
            # Get last k
            topk = indx[-k:]
            # Assume A is huge
            metric = np.linalg.norm(V[topk, :]) ** 2
            if metric > metric_opt:
                metric_opt = metric
                supp_opt = topk

        # Clear all Vc
        self.Vcs.clear()
        return metric_opt, supp_opt


def parallel_spannogram_Dks_eps_psd(k, V, eps, delta):
    """
    Args
    ----
    k: int
        number of nodes in the output subgraph

    V: numpy array
        Rank-d approximation matrix

    eps, delta: float
        Error

    Returns
    -------

    A tuple of (metric, subgraph node index)

    Assumption
    -----------
    - Adjacency matrix (A) is large

    """
    n, d = V.shape

    supp_opt = 0
    metric_opt = 0

    Mopt = (1 + 4 / eps) ** d
    M = (math.log(delta) - math.log(Mopt)) / math.log(1 - 1 / Mopt)
    print(M)
    print('V.shape =', V.shape)

    t_dot = CumulativeTime("dot")
    t_process = CumulativeTime("process")

    count = int(round(M))
    parspan = ParallelSpannogram(V=V, k=k, threshold=1000)
    for i in range(count):
        progress.render_progress(i / count, width=50)

        c = np.random.randn(d, 1)

        # Do matrix multiplication
        with t_dot.time():
            Vc = V.dot(c)

        with t_process.time():
            parspan.append(Vc)
            metric_opt, supp_opt = parspan.process(metric_opt, supp_opt)

    with t_process.time():
        metric_opt, supp_opt = parspan.flush(metric_opt, supp_opt)

    print()
    print(t_dot)
    print(t_process)

    return metric_opt, supp_opt


def build_simple_test_graph():
    nodes = 'abcdefgh'
    edges = [
        'ab',
        'bc',
        'bh',
        'ch',
        'ce',
        'cd',
        'dh',
        'de',
        'ef',
        'fg',
        'gh',
    ]
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    return G


def _dks_pipeline(spannogram, G, d, k, eps=0.1, delta=0.1):
    # Draw original graph

    if G.number_of_edges() > 1000:
        layout = None
    else:
        plt.figure(0)
        print("creating layout")
        # Note: nx.spring_layout takes a long time for big graphs
        layout = nx.circular_layout(G)

        print("drawing original graph")
        nx.draw_networkx(G, pos=layout)

    # Get adjacency matrix
    with benchmark("getting adjacency matrix"):
        A = nx.adjacency_matrix(G).astype(np.float32)

    # Compute rank-d approximation
    with benchmark("computing sparse SVD"):
        Vd = compute_sparse_Vd(A, d)

    # Solve for densest subgraph
    with benchmark("computing spannogram"):
        metric, supp = spannogram(k, Vd, eps=eps, delta=delta)
    print('metric', metric)

    # Select nodes from result
    with benchmark("selecting subgraph"):
        selected = np.array(G.nodes())[supp.flatten()].tolist()
        Gk = G.subgraph(selected)

    # Draw subgraph
    plt.figure(1)

    with benchmark("drawing subgraph"):
        nx.draw_networkx(Gk, pos=layout)

    # Render
    with benchmark('rendering'):
        plt.show()

    return selected


dks_pipeline = functools.partial(_dks_pipeline, spannogram_Dks_eps_psd)
parallel_dks_pipeline = functools.partial(_dks_pipeline,
                                          parallel_spannogram_Dks_eps_psd)


def test():
    answer = 'bcdeh'
    G = build_simple_test_graph()
    print(G.number_of_nodes(), G.number_of_edges())
    print(G.nodes())
    sgnodes = dks_pipeline(G, d=2, k=5)
    assert set(sgnodes) == set(answer)


def test_wdc_sample():
    """
    Uses the sample subgraph
    """
    G = wdc.create_sample_graph()
    dks_pipeline(G, d=2, k=20)


def test_wdc_subsampled():
    G = wdc.create_graph("pld-index-10k.dat", "pld-arc-10k.dat")
    parallel_dks_pipeline(G, d=2, k=100)


if __name__ == '__main__':
    # test()
    # test_wdc_sample()
    test_wdc_subsampled()

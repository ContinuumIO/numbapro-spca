from __future__ import print_function, absolute_import, division
import math
import numpy as np
import scipy
import matplotlib.pyplot as plt
import networkx as nx

import wdc_dataset as wdc

float_dtype = np.dtype(np.float32)


def compute_Vd(A, d):
    U, D, _ = np.linalg.svd(A)
    return U[:, :d].dot(np.diag(np.sqrt(D[:d])))


def compute_sparse_Vd(A, d):
    U, D, _ = scipy.sparse.linalg.svds(A, k=d)
    return U[:, :d].dot(np.diag(np.sqrt(D[:d])))


def spannogram_Dks_eps_psd(k, V, eps, delta):
    """
    Assumption
    -----------
    - A is large

    """
    n, d = V.shape

    supp_opt = 0
    metric_opt = 0

    Mopt = (1 + 4 / eps) ** d
    M = (math.log(delta) - math.log(Mopt)) / math.log(1 - 1 / Mopt)
    print(M)
    for _ in range(int(round(M))):
        c = np.random.randn(d, 1)
        indx = np.argsort(V.dot(c), axis=0)
        # Get last k
        topk = indx[-k:]
        # Assume A is huge
        metric = np.linalg.norm(V[topk, :]) ** 2
        if metric > metric_opt:
            metric_opt = metric
            supp_opt = topk

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


def dks_pipeline(G, d, k, eps=0.1, delta=0.1):
    # Draw original graph
    plt.figure(0)
    layout = nx.spring_layout(G)
    nx.draw_networkx(G, pos=layout)

    # Get adjacency matrix
    A = nx.adjacency_matrix(G).astype(np.float32)

    # Compute rank-d approximation
    Vd = compute_sparse_Vd(A, d)

    # Solve for densest subgraph
    metric, supp = spannogram_Dks_eps_psd(k, Vd, eps=eps, delta=delta)
    print('metric', metric)

    # Select nodes from result
    selected = np.array(G.nodes())[supp.flatten()].tolist()
    Gk = G.subgraph(selected)

    # Draw subgraph
    plt.figure(1)
    nx.draw_networkx(Gk, pos=layout)

    # Render
    plt.show()

    return selected


def test():
    d = 2
    k = 5
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


if __name__ == '__main__':
    # test()
    test_wdc_sample()

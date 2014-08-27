from __future__ import print_function, absolute_import, division
import math
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

float_dtype = np.dtype(np.float32)


def compute_Vd(A, d):
    U, D, _ = np.linalg.svd(A)
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


def test():
    d = 2
    k = 5
    answer = 'bcdeh'
    G = build_simple_test_graph()
    print(G.number_of_nodes(), G.number_of_edges())
    print(G.nodes())

    plt.figure(0)
    layout = nx.spring_layout(G)
    nx.draw_networkx(G, pos=layout)

    A = nx.adjacency_matrix(G).todense().A

    Vd = compute_Vd(A, d)
    assert Vd.shape[1] == d
    metric, supp = spannogram_Dks_eps_psd(k, Vd, eps=0.1, delta=0.1)
    print(metric)
    print(supp)

    selected = np.array(G.nodes())[supp.flatten()].tolist()
    assert set(selected) == set(answer)
    Gk = G.subgraph(selected)
    print(Gk.nodes())

    plt.figure(1)
    nx.draw_networkx(Gk, pos=layout)

    plt.show()


if __name__ == '__main__':
    test()

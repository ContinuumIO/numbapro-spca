from scipy.sparse import dok_matrix
import numpy as np
import logging

N = 475719
file = open('pld-arc-1m.dat')
graph = dok_matrix((N, N), dtype=np.int8)

nodes = {}


def remap(x):
    # return x
    if x not in nodes:
        nodes[x] = len(nodes)
    return nodes[x]


def iter_arc(file):
    for i in iter(file):
        a, b = i.split()
        yield remap(int(a)), remap(int(b))

#build graph
for a, b in iter_arc(file):
    graph[a, b] = graph[b, a] = 1

import tables

filters = tables.Filters(complib='blosc', complevel=5)
def create_matrix(A):
    h5file = tables.open_file("pld-arc-1m.h5", mode="w", title="matrix file")

    # Build containers
    print("build containers")
    ndata = A.getnnz()
    n = A.shape[0]
    print(n, ndata)
    h5data = h5file.create_carray('/', "data", shape=(ndata,),
                                  atom=tables.Float32Atom(), filters=filters)
    h5indices = h5file.create_carray('/', "indices", shape=(ndata,),
                                     atom=tables.UInt64Atom(), filters=filters)
    h5indptr = h5file.create_carray('/', "indptr", shape=(n + 1,),
                                    atom=tables.UInt64Atom(), filters=filters)

    # Populate data
    print("storing")
    h5data[...] = A.data
    h5indices[...] = A.indices
    h5indptr[...] = A.indptr

    print(h5data)
    print(h5indices)
    print(h5indptr)

    print("done")
    h5file.close()


print("to_csr")
create_matrix(graph.tocsr())

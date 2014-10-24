from collections import defaultdict
import numpy as np
import tables
# from scipy.sparse import dok_matrix

filename = 'pld-arc-50k.dat'

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

# count columns
rows = defaultdict(int)

print("first pass")
with open(filename) as fin:
    for a, b in iter_arc(fin):
        rows[a] += 1
        rows[b] += 1

num_of_nodes = len(rows)

assert num_of_nodes == len(nodes)


# get prefix sum
row_offsets = {}
prefixsum = 0
for i in range(num_of_nodes):
    row_offsets[i] = prefixsum
    prefixsum += rows[i]

print('nodes', num_of_nodes)
print('arcs', prefixsum)
num_of_arc = prefixsum
del rows

filters = tables.Filters(complib='blosc', complevel=5)

# build file

h5file = tables.open_file("pld-arc-50k.h5", mode="w", title="matrix file")

# Build containers
print("build containers")

h5data = h5file.create_carray('/', "data", shape=(num_of_arc,),
                              atom=tables.Float32Atom(), filters=filters)
h5indices = h5file.create_carray('/', "indices", shape=(num_of_arc,),
                                 atom=tables.UInt64Atom(), filters=filters)
h5indptr = h5file.create_carray('/', "indptr", shape=(num_of_nodes + 1,),
                                atom=tables.UInt64Atom(), filters=filters)

# Populate data
print("populating")
for i in range(num_of_nodes):
    h5indptr[i] = row_offsets[i]
h5indptr[num_of_nodes] = num_of_arc
row_offsets[num_of_nodes] = num_of_arc

rowinds = defaultdict(int)

# graph = dok_matrix((num_of_nodes, num_of_nodes))

def storedata(a, b):
    base = row_offsets[a]
    offset = rowinds[a]
    rowinds[a] += 1
    idx = base + offset
    h5data[idx] = 1
    h5indices[idx] = b


print("getting data")
with open(filename) as fin:
    for i, (a, b) in enumerate(iter_arc(fin)):

        if i % 10000 == 0:
            print('i=', i)
        storedata(a, b)
        storedata(b, a)
        # graph[a, b] = graph[b, a] = 1

# ref = graph.tocsr()

# Make sure everything is in the right order
print("checking")
for i in range(num_of_nodes):
    if i % 10000 == 0:
        print('i=', i)
    start = row_offsets[i]
    stop = start + rowinds[i]
    assert row_offsets[i + 1] == stop
    indices = h5indices[start:stop]
    data = h5data[start:stop]
    ordering = np.argsort(indices)

    odata = data[ordering]
    oindices = indices[ordering]

    # assert np.all(ref.data[start:stop] == odata)
    # assert np.all(ref.indices[start:stop] == oindices)

    h5data[start:stop] = odata
    h5indices[start:stop] = oindices

# assert np.all(ref.data == h5data)
# assert np.all(ref.indices == h5indices)
# assert np.all(ref.indptr == h5indptr)

print(h5data)
print(h5indices)
print(h5indptr)

print("done")
h5file.close()



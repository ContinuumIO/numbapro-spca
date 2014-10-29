def toprank(indices, filename):
    iset = frozenset(indices)
    names = [None] * len(indices)
    with open(filename) as fin:
        for line in fin:
            name, idx = line.split()
            idx = int(idx)
            if idx in iset:
                names[indices.index(idx)] = name
    return names


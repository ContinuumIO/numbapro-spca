from __future__ import print_function, absolute_import, division
import sys
import os
import networkx as nx


def parse_index_streaming(findex):
    for line in findex:
        name, num = (s.strip() for s in line.split())
        yield int(num), name


def parse_index(findex):
    return dict(parse_index_streaming(findex))


def parse_arc(farc):
    conn = []
    for line in farc:
        a, b = (int(s.strip()) for s in line.split())
        conn.append((a, b))
    return conn


def create_graph(idxfile, arcfile):
    print("parsing indices")
    with open(idxfile) as findex:
        nodes = parse_index(findex)

    G = nx.Graph()
    G.add_nodes_from(nodes.values())

    print("parsing arcs")
    with open(arcfile) as farc:
        conn = parse_arc(farc)

    edges = ((nodes[a], nodes[b]) for a, b in conn)
    G.add_edges_from(edges)

    print("done building graph")
    return G


def create_sample_graph():
    return create_graph(indexfile="index.dat", arcfile="arc.dat")


def subsample_from_arc(arcfile, idxfile, outfile):
    nodes = set()
    with open(arcfile) as farc:
        conn = parse_arc(farc)
    for a, b in conn:
        nodes.add(a)
        nodes.add(b)

    # Filer the full index file to only contain the nodes that have edges
    with open(idxfile) as fidx:
        with open(outfile, 'w') as fout:
            for num, name in parse_index_streaming(fidx):
                if num in nodes:
                    print("{name}\t{num}".format(num=num, name=name),
                          file=fout)


if __name__ == '__main__':
    arcfile, idxfile, outfile = sys.argv[1:]
    print('arcfile', arcfile)
    print('idxfile', idxfile)
    print('outfile', outfile)
    if os.path.isfile(outfile):
        print('{outfile} already exist'.format(outfile=outfile))
        sys.exit(1)
    if input('ok? > ') == 'y':
        subsample_from_arc(arcfile, idxfile, outfile)
    else:
        print('abort')

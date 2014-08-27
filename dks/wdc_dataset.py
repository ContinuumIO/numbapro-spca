import networkx as nx


def parse_index(findex):
    data = {}
    for line in findex:
        name, num = (s.strip() for s in line.split())
        data[int(num)] = name
    return data


def parse_arc(farc):
    conn = []
    for line in farc:
        a, b = (int(s.strip()) for s in line.split())
        conn.append((a, b))
    return conn


def create_sample_graph():
    with open("index.dat") as findex:
        nodes = parse_index(findex)

    with open("arc.dat") as farc:
        conn = parse_arc(farc)

    G = nx.Graph()
    G.add_nodes_from(nodes.values())

    edges = ((nodes[a], nodes[b]) for a, b in conn)
    G.add_edges_from(edges)

    return G

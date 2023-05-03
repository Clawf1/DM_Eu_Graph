import json
from igraph import *


def minimum_vertex_cover(g):
    G = Graph()
    for t in g.keys():
        G.add_vertex(t)
    for i in g.keys():
        for j in g[i]:
            G.add_edge(i, j)
    cover = []
    st_set = G.largest_independent_vertex_sets()[0]
    for vertex in G.vs:
        if vertex.index in st_set:
            continue
        cover.append(G.vs[vertex.index]["name"])

    return cover


if __name__ == '__main__':
    with open('graph.json', 'r') as f:
        graph = json.load(f)
    min_coverage = minimum_vertex_cover(graph)
    print(len(min_coverage), min_coverage)

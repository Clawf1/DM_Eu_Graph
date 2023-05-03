import copy
from collections import deque, defaultdict
import networkx as nx
from heapq import heappush, heappop, heapify
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from random import randint
import math


def bfs(g, start):
    distances = {node: float('inf') for node in g}
    distances[start] = 0
    queue = deque([start])
    while queue:
        current = queue.popleft()
        for neighbor in g[current]:
            if distances[neighbor] == float('inf'):
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
    return distances


def get_radius_diameter(g):
    nodes = list(g.keys())
    min_diameter = float('inf')
    for node in nodes:
        distances = bfs(g, node)
        max_distance = max(distances.values())
        if max_distance == float('inf'):
            return 0, 0
        min_diameter = min(min_diameter, max_distance)
    radius = min_diameter
    max_diameter = 0
    for node in nodes:
        distances = bfs(g, node)
        max_distance = max(distances.values())
        max_diameter = max(max_diameter, max_distance)
    diameter = max_diameter
    return radius, diameter


def get_center(g):
    nodes = list(g.keys())
    center = []
    min_radius = float('inf')
    for node in nodes:
        distances = bfs(g, node)
        max_distance = max(distances.values())
        if max_distance == float('inf'):
            return []
        if max_distance < min_radius:
            min_radius = max_distance
            center = [node]
        elif max_distance == min_radius:
            center.append(node)
    return center


def min_max_degree(g):
    degrees = [len(neighbors) for neighbors in g.values()]
    min_degree = min(degrees)
    max_degree = max(degrees)
    return min_degree, max_degree


def min_edge_coverage(g):
    g_edges = set()
    uncovered_vertices = set(g.keys())
    # Iterate over all vertices
    for v in g.keys():
        # If v is not covered by any edge yet, add an edge to any uncovered neighbor
        if v in uncovered_vertices:
            neighbors = g[v]
            for u in neighbors:
                if u in uncovered_vertices:
                    g_edges.add((v, u))
                    uncovered_vertices.remove(v)
                    uncovered_vertices.remove(u)
                    break

    return g_edges


def greedy_coloring(g):
    colors = {}
    for node in g:
        used_colors = set(colors.get(neighbor) for neighbor in g[node] if neighbor in colors)
        available_colors = set(range(len(g))) - used_colors
        color = min(available_colors)
        colors[node] = color
    return colors


def get_minimum_vertex_coloring(g):
    coloring = greedy_coloring(g)
    return max(coloring.values()) + 1


def get_minimum_edge_coloring(g):
    delta = max(len(neighbors) for neighbors in g.values())
    upper_bound = delta + 1

    while True:
        colors = {}
        used_colors = set()
        for node in g:
            for neighbor in g[node]:
                edge = tuple(sorted((node, neighbor)))
                used = set(colors.get(other_edge) for other_edge in colors if set(edge) & set(other_edge))
                available = set(range(upper_bound)) - used_colors - used
                if not available:
                    break
                color = min(available)
                colors[edge] = color
                used_colors.add(color)
            else:
                continue
            break
        else:
            return upper_bound - 1
        upper_bound += 1


def bron_kerbosch_clique(R, P, X, g):
    if not P and not X:
        return R
    max_clique = []
    for v in P:
        neighbors = g[v]
        max_clique = max(max_clique, bron_kerbosch_clique(R + [v], [p for p in P if p in neighbors],
                                                          [p for p in X if p in neighbors], g), key=len)
        P.remove(v)
        X.append(v)
    return max_clique


def find_max_clique(g):
    nodes = list(g.keys())
    return bron_kerbosch_clique([], nodes, [], g)


def bron_kerbosch(g):
    def bron_kerbosch_internal(r, p, x):
        nonlocal max_clique

        if not p and not x:
            if len(r) > len(max_clique):
                max_clique = r
            return

        for v in p.copy():
            bron_kerbosch_internal(r.union([v]), p.intersection(g[v]), x.intersection(g[v]))
            p.remove(v)
            x.add(v)

    max_clique = set()
    bron_kerbosch_internal(set(), set(g.keys()), set())

    return max_clique


def dfs(g, left, right, dist, u):
    if u is not None:
        for v in g[u]:
            if dist[right[v]] == dist[u] + 1 and dfs(g, left, right, dist, right[v]):
                right[v] = u
                left[u] = v
                return True
        dist[u] = float('inf')
        return False
    return True


def min_vertex_cover(g):
    def maximal_matching():
        nonlocal matching

        for vert in g:
            if vert not in matching:
                for u in g[vert]:
                    if u not in matching:
                        matching[vert] = u
                        matching[u] = vert
                        break

        return set(matching.keys())

    matching = {}
    cover = set()
    vertices = set(g.keys())

    while vertices:
        v = maximal_matching().pop()
        cover.add(v)
        vertices.remove(v)
        vertices -= g[v]

    return cover


def capital_distance(country1, country2):
    geolocator = Nominatim(user_agent="my_app")
    try:
        capital1 = geolocator.geocode(f"capital of {country1}", timeout=10)
        capital2 = geolocator.geocode(f"capital of {country2}", timeout=10)
        distance_km = geodesic((capital1.latitude, capital1.longitude), (capital2.latitude, capital2.longitude)).km
        return distance_km
    except GeocoderTimedOut:
        print("Geocoder service timed out. Please try again later.")


def prim(g, d):
    visited = set()
    min_span_tree = set()
    start_node = next(iter(g.keys()))
    visited.add(start_node)

    heap = [(d(start_node, neighbor), start_node, neighbor) for neighbor in g[start_node]]
    heapify(heap)

    while heap:
        weight, node1, node2 = heappop(heap)
        if node2 not in visited:
            visited.add(node2)
            min_span_tree.add((node1, node2, weight))
            for neighbor in g[node2]:
                if neighbor not in visited:
                    heappush(heap, (d(node2, neighbor), node2, neighbor))

    return min_span_tree


graph = {
    'Portugal': ['Spain'],
    'Croatia': ['Montenegro', 'Bosnia & Herzegovina', 'Slovenia', 'Serbia', 'Hungary'],
    'Montenegro': ['Croatia', 'Kosovo', 'Albania', 'Bosnia & Herzegovina', 'Serbia'],
    'France': ['Andorra', 'Belgium', 'Luxembourg', 'Monaco', 'Italy', 'Switzerland', 'Spain', 'Germany'],
    'Andorra': ['France', 'Spain'],
    'Kosovo': ['Montenegro', 'North Macedonia', 'Albania', 'Serbia'],
    'Sweden': ['Norway', 'Finland'],
    'Belgium': ['France', 'Luxembourg', 'Netherlands', 'Germany'],
    'North Macedonia': ['Kosovo', 'Albania', 'Greece', 'Bulgaria', 'Serbia'],
    'Turkey': ['Greece', 'Bulgaria'],
    'Albania': ['Montenegro', 'Kosovo', 'North Macedonia', 'Greece'],
    'Bosnia & Herzegovina': ['Croatia', 'Montenegro', 'Serbia'],
    'Luxembourg': ['France', 'Belgium', 'Germany'],
    'Latvia': ['Russia', 'Lithuania', 'Belarus', 'Estonia'],
    'Norway': ['Sweden', 'Finland', 'Russia'],
    'Monaco': ['France'],
    'Finland': ['Sweden', 'Norway', 'Russia'],
    'Greece': ['North Macedonia', 'Turkey', 'Albania', 'Bulgaria'],
    'Netherlands': ['Belgium', 'Germany'],
    'Austria': ['Germany', 'Italy', 'Switzerland', 'Slovenia', 'Liechtenstein', 'Slovakia', 'Czechia', 'Hungary'],
    'Germany': ['Belgium', 'Luxembourg', 'Netherlands', 'Austria', 'Switzerland', 'Denmark', 'Poland', 'Czechia',
                'France'],
    'Italy': ['France', 'Austria', 'Switzerland', 'Slovenia', 'San Marino', 'Vatican'],
    'Switzerland': ['France', 'Austria', 'Germany', 'Italy', 'Liechtenstein'],
    'Slovenia': ['Croatia', 'Austria', 'Italy', 'Hungary'],
    'Denmark': ['Germany'],
    'Russia': ['Latvia', 'Norway', 'Finland', 'Lithuania', 'Ukraine', 'Belarus', 'Estonia', 'Poland'],
    'Bulgaria': ['North Macedonia', 'Turkey', 'Greece', 'Romania', 'Serbia'],
    'Lithuania': ['Latvia', 'Russia', 'Belarus', 'Poland'],
    'Moldova': ['Romania', 'Ukraine'],
    'Romania': ['Bulgaria', 'Moldova', 'Ukraine', 'Serbia', 'Hungary'],
    'Ukraine': ['Russia', 'Moldova', 'Romania', 'Belarus', 'Slovakia', 'Poland', 'Hungary'],
    'Spain': ['Portugal', 'France', 'Andorra'],
    'Liechtenstein': ['Austria', 'Switzerland'],
    'San Marino': ['Italy'],
    'Belarus': ['Latvia', 'Russia', 'Lithuania', 'Ukraine', 'Poland'],
    'Estonia': ['Latvia', 'Russia'],
    'Slovakia': ['Austria', 'Ukraine', 'Poland', 'Czechia', 'Hungary'],
    'Poland': ['Germany', 'Russia', 'Lithuania', 'Ukraine', 'Belarus', 'Slovakia', 'Czechia'],
    'Serbia': ['Croatia', 'Montenegro', 'Kosovo', 'North Macedonia', 'Bosnia & Herzegovina', 'Bulgaria', 'Romania',
               'Hungary'],
    'Vatican': ['Italy'],
    'Czechia': ['Austria', 'Germany', 'Slovakia', 'Poland'],
    'Hungary': ['Croatia', 'Austria', 'Slovenia', 'Romania', 'Ukraine', 'Slovakia', 'Serbia']
}

edges = {('Croatia', 'Hungary', 3.3051007420492446), ('Russia', 'Belarus', 8003.294635006126),
         ('France', 'Italy', 0.9098622654282037), ('Ukraine', 'Moldova', 16710.89569503047),
         ('Serbia', 'North Macedonia', 1633.0182764078031), ('Poland', 'Lithuania', 2.04965706252894),
         ('North Macedonia', 'Greece', 316.99580580512884), ('France', 'Monaco', 11747.885781754878),
         ('Romania', 'Bulgaria', 11749.414120973124), ('Hungary', 'Ukraine', 6.356537190758375),
         ('Austria', 'Slovakia', 9494.081810092814), ('Spain', 'Andorra', 11292.669783632895),
         ('Serbia', 'Bosnia & Herzegovina', 1309.509174381811), ('Ukraine', 'Poland', 3.8676371999696775),
         ('Lithuania', 'Russia', 12736.57725061559), ('Belarus', 'Latvia', 4646.02424502333),
         ('Slovakia', 'Czechia', 1236.0810954304782), ('Austria', 'Liechtenstein', 10515.60798067042),
         ('Italy', 'Vatican', 6283.447635310426), ('Serbia', 'Montenegro', 11284.773599640701),
         ('Russia', 'Finland', 12737.91247160966), ('Germany', 'Luxembourg', 0.6249119295302877),
         ('Germany', 'Denmark', 766.5678184704418), ('Albania', 'Kosovo', 11746.632632319031),
         ('Germany', 'Netherlands', 92.76403243396197), ('Czechia', 'Germany', 0.6809662209098974),
         ('Italy', 'San Marino', 16028.670534741219), ('Portugal', 'Spain', 11283.867620615409),
         ('Italy', 'Switzerland', 0.7202134957408972), ('Slovenia', 'Croatia', 1.1823790289776588),
         ('Russia', 'Norway', 8003.274273474284), ('Bulgaria', 'Serbia', 11286.767554625838),
         ('Italy', 'Austria', 6276.303138576045), ('Andorra', 'France', 11746.619252229571),
         ('Luxembourg', 'Belgium', 0.6217578175229205), ('Hungary', 'Romania', 3.2651102258425175),
         ('Bulgaria', 'Turkey', 5.023041448959018), ('Russia', 'Estonia', 1044.3638608474785),
         ('Italy', 'Slovenia', 4.482199838769644), ('Montenegro', 'Albania', 4.893109472578156),
         ('Finland', 'Sweden', 0.642838941200756)}


def get_stable_set(g, ind):
    max_stable_set = {ind}
    for node in g.nodes():
        if node not in max_stable_set and all(neigh not in max_stable_set for neigh in g.neighbors(node)):
            max_stable_set.add(node)
    return max_stable_set


def m_stable_set(g, st):
    G = nx.Graph(g)
    best = get_stable_set(G, st)
    g.pop(st)
    for i in g:
        cur = get_stable_set(G, i)
        if len(cur) > len(best):
            best = cur
    return best


def stable_set(g):
    m = ''
    for i in g:
        res = m_stable_set(copy.deepcopy(g), i)
        if len(m) < len(res):
            m = res
    return m


def find_max_matching(g):
    G = nx.Graph(g)
    matching = nx.algorithms.matching.max_weight_matching(G)
    return matching


if __name__ == '__main__':
    a = 0


    # min_d, max_d = min_max_degree(graph)
    # print(min_d, max_d)

    # rad, dim = get_radius_diameter(graph)

    # min_edge_color = get_minimum_edge_coloring(graph)

    # clique = bron_kerbosch(graph)

    # max_st_set = stable_set(graph)

    # max_mat = find_max_matching(graph)

    # mst = prim(graph, capital_distance)
    # print(mst)

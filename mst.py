import json
# import geopy.distance
from geopy import distance
from countryinfo import CountryInfo


graph = []


def get_dist(x, y):
    return distance.geodesic(x, y).km


with open("graph.json", 'r') as f:
    graph_dict = json.load(f)
    for i in graph_dict:
        for j in graph_dict[i]:
            country1 = CountryInfo(i)
            country2 = CountryInfo(j)
            graph.append([i, j, get_dist(country1.capital_latlng(), country2.capital_latlng())])


def find(parent, ind):
    if parent[ind] == ind:
        return ind
    return find(parent, parent[ind])


def union(parent, rank, x, y):
    xroot = find(parent, x)
    yroot = find(parent, y)

    if rank[xroot] < rank[yroot]:
        parent[xroot] = yroot
    elif rank[xroot] > rank[yroot]:
        parent[yroot] = xroot
    else:
        parent[yroot] = xroot
        rank[xroot] += 1


def kruskal_mst(g):
    result = []
    t = 0
    e = 0
    g = sorted(g, key=lambda item: item[2])

    parent = []
    rank = []

    for node in range(len(g)):
        parent.append(node)
        rank.append(0)

    while t < len(g) - 1 and e < len(g) - 1:
        u, v, w = g[t]
        t = t + 1
        x = find(parent, u)
        y = find(parent, v)

        if x != y:
            e = e + 1
            result.append([u, v, w])
            union(parent, rank, x, y)

    return result


if __name__ == '__main__':
    mst = kruskal_mst(graph)
    for i in mst:
        a, b, dist = i
        print(a, b)

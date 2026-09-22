"""Run beside weighted_graph_mechanisms.py; pip install networkx==3.6.1."""
from math import inf
from graphlib import TopologicalSorter, CycleError
import networkx as nx
from weighted_graph_mechanisms import dijkstra, bellman_ford, kruskal, topological_order


def weighted_graph(n, edges, directed=True):
    graph = nx.MultiDiGraph() if directed else nx.MultiGraph()
    graph.add_nodes_from(range(n))
    for edge_id, (u, v, weight) in enumerate(edges):
        graph.add_edge(u, v, key=edge_id, weight=weight)
    return graph


def valid_order(order, n, edges):
    positions = {vertex: index for index, vertex in enumerate(order)}
    return (set(positions) == set(range(n)) and len(order) == n
            and all(positions[u] < positions[v] for u, v, _ in edges))


def main():
    edges = [(0, 1, 10), (0, 1, 7), (0, 2, 1), (2, 1, 1),
             (1, 3, 2), (2, 3, 8), (3, 4, 3), (2, 4, 20)]
    graph = weighted_graph(6, edges)
    scratch, _, _ = dijkstra(6, edges, 0)
    lengths, paths = nx.single_source_dijkstra(graph, 0, weight='weight')
    assert scratch == [lengths.get(v, inf) for v in graph]
    for v, route in paths.items():
        assert sum(min(edge['weight'] for edge in graph[u][w].values())
                   for u, w in zip(route, route[1:])) == scratch[v]
    print('Dijkstra:', scratch)

    signed = [(0, 1, 2), (0, 2, 5), (2, 1, -4), (1, 3, 2)]
    signed_graph = weighted_graph(5, signed)
    expected, _ = bellman_ford(5, signed, 0)
    lengths = nx.single_source_bellman_ford_path_length(signed_graph, 0, weight='weight')
    assert expected == [lengths.get(v, inf) for v in signed_graph]
    print('Bellman-Ford:', expected)
    # The local solver classifies affected destinations; this API raises instead.
    bad = weighted_graph(4, [(0, 1, 1), (1, 2, -2), (2, 1, 1)])
    try:
        nx.single_source_bellman_ford_path_length(bad, 0, weight='weight')
    except nx.NetworkXUnbounded:
        print('reachable negative cycle:', 'NetworkXUnbounded')

    forest_edges = [(0, 1, 4), (0, 2, 2), (1, 2, 1), (1, 3, 5),
                    (2, 3, 5), (4, 5, -2), (0, 1, 9)]
    forest_graph = weighted_graph(7, forest_edges, directed=False)
    forest = nx.minimum_spanning_tree(forest_graph, weight='weight', algorithm='kruskal')
    total, selected, _ = kruskal(7, forest_edges)
    assert total == forest.size(weight='weight')
    assert nx.is_forest(forest) and set(forest) == set(range(7))
    assert nx.number_connected_components(forest) == nx.number_connected_components(forest_graph)
    print('minimum forest total / edges:', total, len(selected))

    dependencies = [(0, 2, 1), (1, 2, 1), (1, 3, 1), (2, 4, 1), (3, 4, 1)]
    predecessors = {v: set() for v in range(6)}
    for before, after, _ in dependencies:
        predecessors[after].add(before)
    orders = [topological_order(6, dependencies),
              list(TopologicalSorter(predecessors).static_order())]
    assert all(valid_order(order, 6, dependencies) for order in orders)
    print('topological orders valid:', True)
    predecessors[0].add(4)
    try:
        tuple(TopologicalSorter(predecessors).static_order())
    except CycleError:
        print('changed dependency:', 'CycleError')


if __name__ == '__main__':
    main()

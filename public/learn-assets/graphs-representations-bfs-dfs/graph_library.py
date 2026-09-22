"""Run beside graph_traversal_mechanisms.py; pip install networkx==3.6.1."""
import networkx as nx
from graph_traversal_mechanisms import build_graph, bfs, dfs_frames, connected_components


def make_graph(vertices, edges, directed=False):
    """Simple graph: declared unique vertices; repeated edges collapse."""
    adjacency = build_graph(vertices, edges, directed)
    graph = nx.DiGraph() if directed else nx.Graph()
    graph.add_nodes_from(adjacency)  # Includes vertices with no incident edge.
    graph.add_edges_from((u, v) for u, neighbors in adjacency.items() for v in neighbors)
    return adjacency, graph


def compare(vertices, edges, source, directed=False):
    adjacency, graph = make_graph(vertices, edges, directed)
    _, distances, parents = bfs(adjacency, source)
    library_distances = dict(nx.single_source_shortest_path_length(graph, source))
    assert distances == library_distances
    paths = nx.single_source_shortest_path(graph, source)
    for target, path in paths.items():
        assert path[0] == source and path[-1] == target
        assert len(path) - 1 == distances[target]
        assert all(graph.has_edge(u, v) for u, v in zip(path, path[1:]))
    entered, _, _ = dfs_frames(adjacency, source)
    assert set(entered) == set(nx.dfs_preorder_nodes(graph, source))
    if not directed:
        groups, _ = connected_components(adjacency)
        assert {frozenset(group) for group in groups} == {
            frozenset(group) for group in nx.connected_components(graph)}
    return [(vertex, distances.get(vertex)) for vertex in vertices]


def main():
    vertices = list('ABCDEFGH')
    edges = [('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D'),
             ('D', 'E'), ('F', 'G'), ('A', 'B'), ('E', 'E')]
    print('undirected from E:', compare(vertices, edges, 'E'))
    print('directed from E:', compare(vertices, edges, 'E', True))
    print('isolated from H:', compare(vertices, edges, 'H', True))


if __name__ == '__main__':
    main()

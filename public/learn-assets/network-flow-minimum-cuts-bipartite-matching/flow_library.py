"""Run beside network_flow_mechanisms.py; pip install networkx==3.6.1."""
import networkx as nx
from network_flow_mechanisms import edmonds_karp, bipartite_matching


def capacity_graph(n, edges):
    """Aggregate parallel capacities; ignore loops, which carry no useful s-t flow."""
    graph = nx.DiGraph()
    graph.add_nodes_from(range(n))
    for u, v, capacity in edges:
        if u != v:
            old = graph.get_edge_data(u, v, {}).get('capacity', 0)
            graph.add_edge(u, v, capacity=old + capacity)
    return graph


def check_certificate(graph, source, sink, value, flows, source_side):
    assert source in source_side and sink not in source_side
    balance = dict.fromkeys(graph, 0)
    for u, v, data in graph.edges(data=True):
        amount = flows[u][v]
        assert 0 <= amount <= data['capacity']
        balance[u] -= amount
        balance[v] += amount
    assert all(net == (-value if vertex == source else value if vertex == sink else 0)
               for vertex, net in balance.items())
    cut = sum(data['capacity'] for u, v, data in graph.edges(data=True)
              if u in source_side and v not in source_side)
    assert cut == value


def compare(n, edges, source, sink):
    scratch = edmonds_karp(n, edges, source, sink)
    graph = capacity_graph(n, edges)
    value, flows = nx.maximum_flow(graph, source, sink, capacity='capacity',
                                   flow_func=nx.algorithms.flow.edmonds_karp)
    cut, (source_side, _) = nx.minimum_cut(graph, source, sink, capacity='capacity',
                                          flow_func=nx.algorithms.flow.edmonds_karp)
    check_certificate(graph, source, sink, value, flows, source_side)
    assert value == cut == scratch['value'] == scratch['cut_capacity']
    return value


def main():
    edges = [(0, 1, 3), (0, 1, 2), (0, 2, 2), (1, 2, 1),
             (2, 1, 1), (1, 3, 3), (2, 3, 4), (1, 1, 7)]
    print('parallel / antiparallel value:', compare(5, edges, 0, 3))
    changed = [(u, v, 1 if (u, v) == (2, 3) else c) for u, v, c in edges]
    print('changed bottleneck value:', compare(5, changed, 0, 3))
    print('unreachable sink:', compare(5, edges, 0, 4))
    pairs = [(0, 0), (0, 1), (1, 0), (2, 1)]
    scratch = bipartite_matching(4, 3, pairs)
    left = {('L', i) for i in range(4)}
    right = {('R', i) for i in range(3)}
    graph = nx.Graph()
    graph.add_nodes_from(left | right)
    graph.add_edges_from((('L', u), ('R', v)) for u, v in pairs)
    matching = nx.bipartite.maximum_matching(graph, top_nodes=left)
    cover = nx.bipartite.to_vertex_cover(graph, matching, top_nodes=left)
    size = len(matching) // 2  # This dictionary stores both directions.
    assert size == len(scratch['matching']) == len(cover)
    assert all(u in cover or v in cover for u, v in graph.edges)
    print('matching / cover size:', size, len(cover))


if __name__ == '__main__':
    main()

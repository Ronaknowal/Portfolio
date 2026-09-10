"""Independent graph references: all-pairs distances, exhaustive color/order
assignments, recursive DFS and identity-preserving graph isomorphism checks.
The exact displayed programs are supplied by the JavaScript wrapper.
"""
import contextlib
import io
import itertools
import json
import random
import sys
from collections import Counter
from pathlib import Path

examples = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
namespaces = {}
for name, example in examples.items():
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(example["code"], f"displayed-{name}.py", "exec"), namespace)
    namespaces[name] = namespace
counts = Counter()
INF = float("inf")


def raises(exception_type, function, *args):
    try:
        function(*args)
    except exception_type:
        return
    raise AssertionError(f"expected {exception_type.__name__}")


def all_pairs(vertices, edges):
    distance = {(a, b): (0 if a == b else INF) for a in vertices for b in vertices}
    for first, second in edges:
        distance[first, second] = min(distance[first, second], 1)
    for middle in vertices:
        for first in vertices:
            for second in vertices:
                distance[first, second] = min(distance[first, second], distance[first, middle] + distance[middle, second])
    return distance


def recursive_dfs(graph, source):
    entered, finished, parents = [], [], {source: None}
    def visit(vertex):
        entered.append(vertex)
        for neighbor in graph[vertex]:
            if neighbor not in parents:
                parents[neighbor] = vertex
                visit(neighbor)
        finished.append(vertex)
    visit(source)
    return entered, finished, parents


build_graph = namespaces["representations"]["build_graph"]
bfs = namespaces["breadthFirst"]["bfs"]
path_to = namespaces["breadthFirst"]["path_to"]
dfs_frames = namespaces["depthFirst"]["dfs_frames"]
components = namespaces["components"]["connected_components"]
two_color = namespaces["bipartition"]["two_color"]
cycle = namespaces["directedCycle"]["has_directed_cycle"]

for directed, maximum_size in [(False, 4), (True, 3)]:
    for size in range(maximum_size + 1):
        vertices = list(range(size))
        potential = list(itertools.product(vertices, repeat=2) if directed else itertools.combinations_with_replacement(vertices, 2))
        for mask in range(1 << len(potential)):
            edges = [edge for index, edge in enumerate(potential) if mask & (1 << index)]
            # Duplicate edge entries must not multiply neighbor entries; isolates survive.
            graph = build_graph(iter(vertices), itertools.chain(edges, edges), directed)
            expected_arcs = set(edges) | (set() if directed else {(b, a) for a, b in edges})
            assert set(graph) == set(vertices)
            assert {(a, b) for a, neighbors in graph.items() for b in neighbors} == expected_arcs
            assert all(len(neighbors) == len(set(neighbors)) for neighbors in graph.values())
            distance = all_pairs(vertices, expected_arcs)
            for source in vertices:
                order, actual_distances, parents = bfs(graph, source)
                expected_distances = {target: distance[source, target] for target in vertices if distance[source, target] < INF}
                assert actual_distances == expected_distances
                assert set(order) == set(expected_distances) and len(order) == len(set(order))
                assert [actual_distances[vertex] for vertex in order] == sorted(actual_distances.values())
                assert parents[source] is None
                for target in vertices:
                    route = path_to(parents, target)
                    if target not in expected_distances:
                        assert route is None
                    else:
                        assert route[0] == source and route[-1] == target
                        assert len(route) - 1 == expected_distances[target]
                        assert all((a, b) in expected_arcs for a, b in zip(route, route[1:]))
                actual_dfs = dfs_frames(graph, source)
                assert actual_dfs == recursive_dfs(graph, source)
                assert set(actual_dfs[0]) == set(expected_distances)
                counts["bfs_distance_path_and_dfs_source_cases"] += 1
            raises(KeyError, bfs, graph, size)
            raises(KeyError, dfs_frames, graph, size)
            if directed:
                # A graph is acyclic exactly when some vertex ordering orients every edge forward.
                topological_order_exists = any(all(order.index(a) < order.index(b) for a, b in edges)
                                               for order in itertools.permutations(vertices))
                assert cycle(graph) == (not topological_order_exists)
                counts["directed_cycle_exhaustive_order_oracles"] += 1
            else:
                groups, labels = components(graph)
                assert len(labels) == size and set(labels) == set(vertices)
                assert sum(map(len, groups)) == size
                assert {vertex for group in groups for vertex in group} == set(vertices)
                for first, second in itertools.product(vertices, repeat=2):
                    assert (labels[first] == labels[second]) == (distance[first, second] < INF)
                valid_coloring_exists = any(all(colors[a] != colors[b] for a, b in edges)
                                            for colors in itertools.product((0, 1), repeat=size))
                colors = two_color(graph)
                assert (colors is not None) == valid_coloring_exists
                if colors is not None:
                    assert set(colors) == set(vertices)
                    assert all(colors[a] != colors[b] for a, b in expected_arcs)
                counts["undirected_component_and_bruteforce_coloring_graphs"] += 1
            counts["representation_graphs_with_duplicate_edges"] += 1

for vertices, edges in [([None], []), (["A", "A"], []), (["A"], [("A", "B")])]:
    raises(ValueError, build_graph, vertices, edges)
assert build_graph(["Z", "A", "Q"], [("Z", "Q"), ("Z", "A"), ("Z", "Q")], True)["Z"] == ["Q", "A"]
counts["builder_rejection_and_neighbor_order_cases"] = 4

grid_distances = namespaces["gridWavefront"]["grid_distances"]
shortest_cells = namespaces["diagonalPractice"]["shortest_cell_path"]


def grid_oracle(grid, diagonal=False):
    vertices = [(row, column) for row in range(len(grid)) for column in range(len(grid[0])) if grid[row][column] == "."]
    edges = [(a, b) for a in vertices for b in vertices if a != b and
             (max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1 if diagonal else abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1)]
    return vertices, all_pairs(vertices, edges)


for rows, columns in [(1, 1), (2, 3), (3, 3)]:
    for mask in range(1 << (rows * columns)):
        grid = ["".join("#" if mask & (1 << (row * columns + column)) else "." for column in range(columns)) for row in range(rows)]
        original = list(grid)
        vertices, reference = grid_oracle(grid)
        source_sets = [[]] + [[vertex] for vertex in vertices]
        if vertices:
            source_sets.extend([[vertices[0], vertices[-1]], [vertices[0], vertices[0]]])
        for sources in source_sets:
            actual, parents = grid_distances(grid, iter(sources))
            for row, column in itertools.product(range(rows), range(columns)):
                position = (row, column)
                expected = min((reference[source, position] for source in sources), default=INF) if position in vertices else INF
                assert actual[row][column] == (-1 if expected == INF else expected)
                if expected != INF:
                    if expected == 0:
                        assert position in sources and parents[position] is None
                    else:
                        parent = parents[position]
                        assert abs(parent[0] - row) + abs(parent[1] - column) == 1
                        assert actual[parent[0]][parent[1]] == expected - 1
                else:
                    assert position not in parents
            assert grid == original
            counts["four_neighbor_source_sets"] += 1
        diagonal_vertices, diagonal_reference = grid_oracle(grid, diagonal=True)
        start, target = (0, 0), (rows - 1, columns - 1)
        expected = diagonal_reference[start, target] + 1 if start in diagonal_vertices and target in diagonal_vertices else INF
        assert shortest_cells([[int(cell == "#") for cell in row] for row in grid]) == (-1 if expected == INF else expected)
        counts["eight_neighbor_cell_count_grids"] += 1

for grid, sources in [([], []), ([""], []), (["..", "."], []), (["x"], []), ([[""]], []), ([[".#"]], []), ([[1]], []), (["#"], [(0, 0)]), (["."], [(0, 1)]), (["."], [(-1, 0)])]:
    raises(ValueError, grid_distances, grid, sources)
    counts["invalid_grid_or_source_cases"] += 1
for grid in [[[0], [0, 0]], [[0, 2]]]:
    raises(ValueError, shortest_cells, grid)
    counts["invalid_grid_or_source_cases"] += 1
assert shortest_cells([]) == -1 and shortest_cells([[]]) == -1
assert shortest_cells([[0, 1], [1, 0]]) == 2
assert grid_distances([".#", "#."], [(0, 0)])[0][1][1] == -1
counts["diagonal_vs_orthogonal_contracts"] = 4

flood_fill = namespaces["floodFill"]["flood_fill"]
for values in itertools.product((0, 1), repeat=6):
    image = [list(values[:3]), list(values[3:])]
    for start in itertools.product(range(2), range(3)):
        source_color = image[start[0]][start[1]]
        same_color_grid = ["".join("." if color == source_color else "#" for color in row) for row in image]
        vertices, reference = grid_oracle(same_color_grid)
        for new_color in (0, 1, 7):
            original = [row.copy() for row in image]
            result = flood_fill(image, start, new_color)
            assert image == original and result is not image
            assert all(row is not image[index] for index, row in enumerate(result))
            expected = [[new_color if (row, column) in vertices and reference[start, (row, column)] < INF else image[row][column]
                         for column in range(3)] for row in range(2)]
            assert result == expected
            counts["flood_fill_regions_and_copy_contracts"] += 1
for image, start in [([], (0, 0)), ([[]], (0, 0)), ([[0], [0, 0]], (0, 0)), ([[0]], (1, 0))]:
    raises(ValueError, flood_fill, image, start, 1)
    counts["invalid_flood_fill_inputs"] += 1

Node = namespaces["cloneGraph"]["Node"]
clone_graph = namespaces["cloneGraph"]["clone_graph"]
randomizer = random.Random(91423)
for trial in range(300):
    originals = [Node("same value") for _ in range(randomizer.randrange(1, 9))]
    for node in originals:
        node.neighbors = [randomizer.choice(originals) for _ in range(randomizer.randrange(6))]
    copied = clone_graph(originals[0])
    mapping, reverse, pending = {}, {}, [(originals[0], copied)]
    while pending:
        original, replica = pending.pop()
        if original in mapping:
            assert mapping[original] is replica
            continue
        assert replica not in originals and replica not in reverse
        mapping[original], reverse[replica] = replica, original
        assert replica.value == original.value
        assert replica.neighbors is not original.neighbors
        assert len(replica.neighbors) == len(original.neighbors), "parallel entries and self-links survive"
        pending.extend(zip(original.neighbors, replica.neighbors))
    copied.neighbors.append(copied)
    assert len(originals[0].neighbors) + 1 == len(copied.neighbors)
    counts["clone_identity_bijections_with_equal_values"] += 1
assert clone_graph(None) is None

# Deep cases exercise iterative methods only, separately from small recursive oracles.
size = 2000
chain = build_graph(range(size), [(index, index + 1) for index in range(size - 1)], True)
order, distances, parents = bfs(chain, 0)
assert distances[size - 1] == size - 1 and path_to(parents, size - 1) == list(range(size))
entered, finished, _ = dfs_frames(chain, 0)
assert entered == list(range(size)) and finished == list(reversed(range(size)))
assert cycle(chain) is False
chain[size - 1].append(0)
assert cycle(chain) is True
assert grid_distances(["." * size], [(0, 0)])[0][0][-1] == size - 1
assert shortest_cells([[0] * size]) == size
counts["deep_iterative_chain_vertices"] = size
print(json.dumps({"independent_native_cases": dict(counts)}, sort_keys=True))

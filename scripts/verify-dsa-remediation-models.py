"""Author oracles for the ten DSA scratch/library extensions; no browser claim."""
import hashlib
import importlib.util
import itertools
import json
import math
import random
import sqlite3
import sys
import tempfile
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'public/learn-assets'
EVIDENCE = ROOT / 'docs/teaching/implementation-depth/dsa-remediation-models.json'
checks = []
hashes = {}


def load(topic, filename):
    path = ASSETS / topic / filename
    name = path.stem
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    hashes[str(path.relative_to(ROOT)).replace('\\', '/')] = hashlib.sha256(path.read_bytes()).hexdigest()
    return module


def check(name, condition, cases=None):
    if not condition:
        raise AssertionError(name)
    checks.append({'name': name, 'passed': True, **({'cases': cases} if cases else {})})


def trees():
    topic = 'trees-binary-search-trees'
    m = load(topic, 'tree_mechanisms.py')
    api = load(topic, 'ordered_set_library.py')
    rng = random.Random(9022)
    root, ordered, oracle = None, [], set()
    for _ in range(250):
        value = rng.randrange(-20, 21)
        if rng.randrange(2):
            root = m.insert(root, value)
            api.add_unique(ordered, value)
            oracle.add(value)
        else:
            root = m.delete(root, value)
            api.discard(ordered, value)
            oracle.discard(value)
        assert m.inorder(root) == ordered == sorted(oracle)
        for target in [-30, -5, 0, 7, 30]:
            expected = (max((x for x in oracle if x <= target), default=None),
                        min((x for x in oracle if x >= target), default=None))
            assert api.neighbors(ordered, target) == expected
            assert (m.floor_key(root, target), m.ceiling_key(root, target)) == expected
    check('Tree/bisect mixed mutations match independent set + scanning neighbor oracle', True, 250)
    slots = [2, 5, 8, 11]
    reserved = []
    for request in [4, 5, 12, 1]:
        chosen = api.neighbors(slots, request)[1]
        reserved.append(chosen)
        if chosen is not None:
            api.discard(slots, chosen)
    api.add_unique(slots, 5)
    api.add_unique(slots, 5)
    check('Tree changed reservation exercise and duplicate releases', reserved == [5, 8, None, 2] and slots == [5, 11])


def graph_traversal():
    topic = 'graphs-representations-bfs-dfs'
    m = load(topic, 'graph_traversal_mechanisms.py')
    api = load(topic, 'graph_library.py')
    arcs = list(itertools.product(range(3), repeat=2))
    for mask in range(1 << len(arcs)):
        edges = [edge for i, edge in enumerate(arcs) if mask >> i & 1]
        distances = [[0 if a == b else math.inf for b in range(4)] for a in range(4)]
        for a, b in edges:
            distances[a][b] = min(distances[a][b], 1)
        for via in range(4):
            for a in range(4):
                for b in range(4):
                    distances[a][b] = min(distances[a][b], distances[a][via] + distances[via][b])
        for source in range(4):
            actual = dict(api.compare(list(range(4)), edges + edges[:1], source, True))
            assert actual == {v: (None if math.isinf(distances[source][v]) else distances[source][v]) for v in range(4)}
    check('Traversal includes loops, duplicate edges and isolated vertices against distance closure', True, 2048)
    vertices = list('ABCDEFGHI')
    edges = [('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D'), ('D', 'E'), ('F', 'G')]
    reversed_graph = m.build_graph(vertices, [(b, a) for a, b in edges], True)
    _, actual, _ = m.bfs(reversed_graph, 'E')
    check('Incoming reachability practice', actual == {'E': 0, 'D': 1, 'B': 2, 'C': 2, 'A': 3})


def ranges():
    m = load('segment-trees-fenwick-trees-range-queries', 'range_query_mechanisms.py')
    rng = random.Random(201)
    for size in [0, 1, 3, 5, 13]:
        values = [rng.randrange(-8, 9) for _ in range(size)]
        lazy = m.LazySum(values)
        point = m.SegmentTree(values, lambda a, b: a + b, 0)
        fenwick = m.Fenwick(values)
        for step in range(80):
            low, high = sorted([rng.randrange(size + 1), rng.randrange(size + 1)])
            value = rng.randrange(-9, 10)
            if step % 2:
                lazy.set(low, high, value)
                values[low:high] = [value] * (high - low)
            else:
                lazy.add(low, high, value)
                for i in range(low, high):
                    values[i] += value
            for i, actual in enumerate(values):
                point.set(i, actual)
                fenwick.set(i, actual)
            for left in range(size + 1):
                for right in range(left, size + 1):
                    expected = sum(values[left:right])
                    assert lazy.query(left, right) == point.query(left, right) == fenwick.range_sum(left, right) == expected
    check('Local range structures every interval after mixed signed set/add updates', True, 400)
    for reverse, expected in [(False, [3, 6, 4, 4, 0]), (True, [3, 6, 6, 6, 0])]:
        tree = m.LazySum([3, -1, 4, 0, 2])
        actions = [(tree.set, 1, 4, 6), (tree.add, 2, 5, -2)]
        for action, low, high, value in (reversed(actions) if reverse else actions):
            action(low, high, value)
        assert [tree.query(i, i + 1) for i in range(5)] == expected
    check('Changed lazy update order gives totals 17 and 21', True)


def path_oracle(n, edges, source):
    best = [math.inf] * n
    def visit(vertex, seen, cost):
        best[vertex] = min(best[vertex], cost)
        for u, v, weight in edges:
            if u == vertex and v not in seen:
                visit(v, seen | {v}, cost + weight)
    visit(source, {source}, 0)
    return best


def components(n, edges):
    groups = [{v} for v in range(n)]
    for u, v, _ in edges:
        first = next(group for group in groups if u in group)
        second = next(group for group in groups if v in group)
        if first is not second:
            first.update(second)
            groups.remove(second)
    return {frozenset(group) for group in groups}


def weighted_graphs():
    topic = 'shortest-paths-spanning-trees-topological-ordering'
    m = load(topic, 'weighted_graph_mechanisms.py')
    api = load(topic, 'weighted_graph_library.py')
    import networkx as nx
    rng = random.Random(103)
    for _ in range(50):
        edges = [(rng.randrange(4), rng.randrange(4), rng.randrange(6)) for _ in range(8)]
        expected = path_oracle(5, edges, 0)
        graph = api.weighted_graph(5, edges)
        library = nx.single_source_dijkstra_path_length(graph, 0, weight='weight')
        assert m.dijkstra(5, edges, 0)[0] == expected == [library.get(v, math.inf) for v in range(5)]
        signed = [(u, v, rng.randrange(-5, 6)) for u in range(4) for v in range(u + 1, 4) if rng.randrange(2)]
        assert m.bellman_ford(5, signed, 0)[0] == path_oracle(5, signed, 0)
        forest_edges = edges[:7]
        target = components(5, forest_edges)
        optimum = math.inf
        for selected in itertools.combinations(forest_edges, 5 - len(target)):
            if components(5, selected) == target:
                optimum = min(optimum, sum(weight for _, _, weight in selected))
        assert m.kruskal(5, forest_edges)[0] == optimum
        forest = nx.minimum_spanning_tree(api.weighted_graph(5, forest_edges, False), weight='weight')
        assert forest.size(weight='weight') == optimum
    check('Weighted routes enumerate simple paths; forests enumerate edge subsets', True, 50)
    base = [(0,1,10),(0,1,7),(0,2,1),(2,1,1),(1,3,2),(2,3,8),(3,4,3),(2,4,20)]
    check('Parallel-route changed practice, zero and signed edges',
          m.dijkstra(6, base + [(0,1,0)], 0)[0] == [0,0,1,2,5,math.inf]
          and m.bellman_ford(6, base + [(0,1,-1)], 0)[0] == [0,-1,1,1,4,math.inf])


def strings():
    topic = 'string-matching-prefix-functions-rolling-hashes'
    m = load(topic, 'string_matching_mechanisms.py')
    api = load(topic, 'string_search_library.py')
    texts = [''.join(chars) for size in range(6) for chars in itertools.product('ab', repeat=size)]
    patterns = [''.join(chars) for size in range(4) for chars in itertools.product('ab', repeat=size)]
    count = 0
    for text in texts:
        for pattern in patterns:
            expected = [i for i in range(len(text) - len(pattern) + 1) if text[i:i + len(pattern)] == pattern]
            assert api.builtin_occurrences(text, pattern) == m.find_all(text, pattern) == expected
            for seam in range(len(text) + 1):
                matcher = m.StreamMatcher(pattern)
                assert matcher.feed(text[:seam]) + matcher.feed('') + matcher.feed(text[seam:]) == expected
            count += 1
    check('All-overlap built-in/KMP/stream matching against slicing at every seam', True, count)
    for pattern, expected in [('aaa', [0,1,2,3]), ('', list(range(7)))]:
        matcher = m.StreamMatcher(pattern)
        actual = [offset for chunk in ['aa', '', 'aaa', 'a'] for offset in matcher.feed(chunk)]
        assert actual == expected
    check('Changed chunk exercise includes empty pattern and empty chunk', True)


def randomized():
    m = load('randomized-algorithms-sampling-error-guarantees', 'randomized_algorithm_mechanisms.py')
    class Tape:
        def __init__(self, values): self.values = iter(values)
        def randrange(self, stop):
            value = next(self.values)
            assert 0 <= value < stop
            return value
    counts = Counter()
    for tape in itertools.product(range(3), range(4), range(5)):
        counts[tuple(sorted(m.reservoir(iter(range(5)), 2, Tape(tape))))] += 1
    check('Reservoir exact choice-tree oracle, no frequency estimate', len(counts) == 10 and set(counts.values()) == {6}, 60)
    reduced = Counter(pair for group in itertools.combinations(range(6), 4) for pair in itertools.combinations(group, 2))
    check('Changed downsampling practice gives uniform pair counts', len(reduced) == 15 and set(reduced.values()) == {6})
    picker = m.WeightedPicker([0, 2**60, 0, 1])
    check('Integer ticket route retains enormous-weight last occurrence',
          picker.pick(Tape([0])) == picker.pick(Tape([2**60 - 1])) == 1 and picker.pick(Tape([2**60])) == 3)
    seen = []
    def stream():
        for i in range(4):
            seen.append(i)
            yield i
    check('Zero reservoir capacity does not consume stream', m.reservoir(stream(), 0, random.Random(2)) == [] and seen == [])


def flow():
    topic = 'network-flow-minimum-cuts-bipartite-matching'
    m = load(topic, 'network_flow_mechanisms.py')
    api = load(topic, 'flow_library.py')
    rng = random.Random(221)
    for _ in range(100):
        edges = [(rng.randrange(4), rng.randrange(4), rng.randrange(5)) for _ in range(9)]
        cut = min(sum(c for u,v,c in edges if u in side and v not in side)
                  for mask in range(4) for side in [{0} | {v for v in [1,2] if mask >> (v-1) & 1}])
        result = m.edmonds_karp(5, edges, 0, 3)
        assert api.compare(5, edges, 0, 3) == result['value'] == cut
        balance = [0] * 5
        for (u,v,capacity), amount in zip(edges, result['flows']):
            assert 0 <= amount <= capacity
            balance[u] -= amount
            balance[v] += amount
        assert balance == [-cut,0,0,cut,0]
    check('Parallel/antiparallel flow certificates against every source-sink cut', True, 100)
    changed = [(0,1,3),(0,1,2),(0,2,2),(1,2,1),(2,1,1),(1,3,2),(2,3,2),(1,1,7)]
    check('Changed two-exit bottleneck exercise', api.compare(5, changed, 0, 3) == 4)


def geometry():
    m = load('computational-geometry-robust-predicates-convex-hulls', 'computational_geometry_mechanisms.py')
    import numpy as np
    from scipy.spatial import ConvexHull, QhullError
    rng = random.Random(97)
    compared = 0
    for _ in range(100):
        points = list({(rng.randrange(-9,10), rng.randrange(-9,10)) for _ in range(12)})
        exact = m.convex_hull(points)
        if len(exact) < 3:
            continue
        floating = np.asarray(points, dtype=np.float64)
        library = ConvexHull(floating)
        assert {tuple(floating[i]) for i in library.vertices} == set(exact)
        assert math.isclose(library.volume, abs(m.signed_area_twice(exact))/2, abs_tol=1e-10, rel_tol=1e-12)
        for direction in [(1,0),(0,1),(3,-2),(-4,-7)]:
            score = lambda point: sum(a*b for a,b in zip(point,direction))
            assert max(map(score, points)) == max(map(score, exact))
        compared += 1
    check('Nondegenerate integer hulls: library corners, exact area and support witnesses', True, compared)
    points = [(1,1),(7,1),(7,6),(1,6)]
    translated = [(x+10**6,y-10**6) for x,y in points]
    assert abs(m.signed_area_twice(translated)) == 60
    large = 2**53
    global_points = [(large+x,large+y) for x,y in [(0,0),(1,0),(1,1),(0,1)]]
    origin = global_points[0]
    shifted_first = np.array([(x-origin[0],y-origin[1]) for x,y in global_points], dtype=float)
    rounded_first = np.array(global_points, dtype=float) - np.array(origin, dtype=float)
    check('Translation practice exposes conversion-order loss', ConvexHull(shifted_first).volume == 1 and np.unique(rounded_first, axis=0).shape[0] == 1)


def persistent():
    m = load('persistent-data-structures-structural-sharing-versioned-queries', 'persistent_structures_mechanisms.py')
    from immutables import Map
    rng = random.Random(55)
    originals = [2,1,4,3,5]
    roots, maps, arrays = [m.build(originals)], [Map(enumerate(originals))], [originals]
    for _ in range(80):
        parent = rng.randrange(len(roots))
        index, value = rng.randrange(5), rng.randrange(-9,10)
        roots.append(m.assign(roots[parent], index, value))
        maps.append(maps[parent].set(index, value))
        changed = arrays[parent].copy()
        changed[index] = value
        arrays.append(changed)
        for root, mapping, expected in zip(roots, maps, arrays):
            assert [mapping[i] for i in range(5)] == expected
            for left in range(6):
                for right in range(left,6):
                    assert m.range_sum(root,left,right) == sum(expected[left:right])
    check('Branches from any old version preserve every old interval', True, 80)
    root = m.assign(m.assign(roots[0],1,-3),4,12)
    check('Historical two-position correction exercise', [m.range_sum(root,i,i+1) for i in range(5)] == [2,-3,4,3,12])


def external():
    topic = 'external-memory-algorithms-b-trees-i-o-complexity'
    m = load(topic, 'external_memory_mechanisms.py')
    api = load(topic, 'external_sort_stream.py')
    rng = random.Random(33)
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        for size in [0,1,2,7,13,50,127]:
            values = [rng.randrange(-12,13) for _ in range(size)]
            source, target = directory/'input.bin', directory/f'output-{size}.bin'
            with source.open('wb') as stream:
                api.write_records(stream,values,3)
            before = source.read_bytes()
            result = api.sort_integer_file(source,target,chunk_records=3,fan_in=2,buffer_records=2)
            with target.open('rb') as stream:
                actual = list(api.read_records(stream,4))
            assert actual == sorted(values) and Counter(actual) == Counter(values)
            assert result['records'] == size and source.read_bytes() == before
            assert not list(directory.glob('sorted-runs-*'))
            try:
                api.sort_integer_file(source,target,3,2,2)
            except ValueError:
                assert target.read_bytes() == b''.join(__import__('struct').pack('<q',x) for x in actual)
            else:
                raise AssertionError('existing destination was accepted')
        source = directory/'truncated.bin'
        source.write_bytes(b'123456789')
        target = directory/'must-not-exist.bin'
        try:
            api.sort_integer_file(source,target,3,2,2)
        except ValueError:
            assert not target.exists() and not list(directory.glob('sorted-runs-*'))
        else:
            raise AssertionError('truncated record accepted')
    check('File sorter: empty/duplicates/partial/multilevel, preserved input and cleaned workspace', True, 7)
    check('File sorter rejects truncated input and existing destinations without overwriting', True)
    connection = sqlite3.connect(':memory:')
    try:
        connection.execute('CREATE TABLE values_with_ids (id INTEGER PRIMARY KEY, reading INTEGER)')
        values = [10,20,5,12,30,7,17,10,10]
        connection.executemany('INSERT INTO values_with_ids(reading) VALUES (?)',[(v,) for v in values])
        connection.execute('CREATE INDEX value_order ON values_with_ids(reading,id)')
        rows = list(connection.execute('SELECT reading,id FROM values_with_ids ORDER BY reading,id'))
        assert rows == sorted((value,index+1) for index,value in enumerate(values))
        assert sum(value == 10 for value,_ in rows) == 3
    finally:
        connection.close()
    check('SQL changed duplicate/multicolumn ordering exercise', True)


def save(status, error=None):
    hashes['scripts/verify-dsa-remediation-models.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    EVIDENCE.write_text(json.dumps({'status': status, 'scope': 'author complementary contract/oracle checks',
        'checks': checks, 'sourceHashes': hashes, **({'error': error} if error else {})}, indent=2)+'\n', encoding='utf-8')


save('running')
try:
    for function in [trees,graph_traversal,ranges,weighted_graphs,strings,randomized,flow,geometry,persistent,external]:
        function()
        print(function.__name__, 'passed', flush=True)
    save('passed')
except Exception as error:
    save('failed', str(error))
    raise

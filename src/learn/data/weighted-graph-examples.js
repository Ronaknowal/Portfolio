// Each displayed program is complete; shared strings only avoid divergent copies within this topic.
const graphCode = `from math import inf

def adjacency(n, edges, directed=True):
    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    graph = [[] for _ in range(n)]
    for edge_id, (u, v, weight) in enumerate(edges):
        if not (isinstance(u, int) and isinstance(v, int) and 0 <= u < n and 0 <= v < n):
            raise ValueError("Vertex outside 0..n-1")
        if not isinstance(weight, int):
            raise ValueError("This exact model requires integer weights")
        graph[u].append((v, weight, edge_id))
        if not directed:
            graph[v].append((u, weight, edge_id))
    return graph

def check_source(n, source):
    if not isinstance(source, int) or not 0 <= source < n:
        raise ValueError("Source outside 0..n-1")

def route(parent, source, target):
    vertices, edge_ids = [target], []
    while target != source:
        if parent[target] is None:
            return None
        previous, edge_id = parent[target]
        edge_ids.append(edge_id)
        vertices.append(previous)
        target = previous
    return vertices[::-1], edge_ids[::-1]`;

const dijkstraCode = `${graphCode}
from heapq import heappop, heappush
from itertools import count

def dijkstra(n, edges, source):
    graph = adjacency(n, edges)
    check_source(n, source)
    if any(weight < 0 for _, _, weight in edges):
        raise ValueError("Dijkstra requires nonnegative weights")
    distance, parent = [inf] * n, [None] * n
    distance[source] = 0
    tickets = count()
    heap = [(0, next(tickets), source)]
    stale = []
    while heap:
        cost, _, u = heappop(heap)
        if cost != distance[u]:
            stale.append((u, cost))
            continue
        # This non-stale extraction finalizes u under nonnegative weights.
        for v, weight, edge_id in graph[u]:
            candidate = cost + weight
            if candidate < distance[v]:
                distance[v] = candidate
                parent[v] = (u, edge_id)
                heappush(heap, (candidate, next(tickets), v))
    return distance, parent, stale`;

const bellmanCode = `${graphCode}
from collections import deque

def bellman_ford(n, edges, source):
    graph = adjacency(n, edges)
    check_source(n, source)
    distance = [inf] * n
    distance[source] = 0
    for _ in range(n - 1):
        previous = distance
        distance = previous.copy()
        for u, v, weight in edges:
            if previous[u] != inf:
                distance[v] = min(distance[v], previous[u] + weight)
    affected = {v for u, v, weight in edges
                if distance[u] != inf and distance[u] + weight < distance[v]}
    queue = deque(affected)
    while queue:
        u = queue.popleft()
        for v, _, _ in graph[u]:
            if v not in affected:
                affected.add(v)
                queue.append(v)
    for v in affected:
        distance[v] = -inf
    # Recover finite witnesses through tight edges, avoiding zero-cycle parents.
    parent, seen = [None] * n, set()
    if distance[source] == 0:
        seen.add(source)
        queue.append(source)
    while queue:
        u = queue.popleft()
        for v, weight, edge_id in graph[u]:
            if v not in seen and distance[v] not in (inf, -inf) and distance[u] + weight == distance[v]:
                seen.add(v)
                parent[v] = (u, edge_id)
                queue.append(v)
    return distance, parent`;

const dsuCode = `${graphCode}

class DisjointSets:
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, vertex):
        while self.parent[vertex] != vertex:
            self.parent[vertex] = self.parent[self.parent[vertex]]
            vertex = self.parent[vertex]
        return vertex

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return False
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]
        return True`;

const topoCode = `${graphCode}
from collections import deque

def topological_order(n, edges):
    graph = adjacency(n, edges)
    indegree = [0] * n
    for _, v, _ in edges:
        indegree[v] += 1
    ready = deque(v for v in range(n) if indegree[v] == 0)
    order = []
    while ready:
        u = ready.popleft()
        order.append(u)
        for v, _, _ in graph[u]:
            indegree[v] -= 1
            if indegree[v] == 0:
                ready.append(v)
    if len(order) != n:
        raise ValueError("Directed cycle: no complete topological order")
    return order`;

const routeFixture = '[(0,1,10),(0,2,1),(2,1,1),(1,3,2),(2,3,8),(3,4,3),(2,4,20)]';
const forestFixture = '[(0,1,4),(0,2,2),(1,2,1),(1,3,5),(2,3,8),(2,4,10),(3,4,2),(3,5,6),(4,5,3)]';
const dependencyFixture = '[(0,2,1),(1,2,1),(1,3,1),(2,4,1),(3,4,1),(4,5,1)]';

export const weightedGraphExamples = {
  dijkstra: {
    title: 'Run a lazy priority frontier and recover an edge-identified route', language: 'python',
    code: `${dijkstraCode}

edges = ${routeFixture}
distance, parent, stale = dijkstra(6, edges, 0)
print(distance)
print(route(parent, 0, 4))
print(stale)
print(route(parent, 0, 5))`,
    expected: '[0, 2, 1, 4, 7, inf]\n([0, 2, 1, 3, 4], [1, 2, 3, 5])\n[(3, 9), (1, 10), (4, 21)]\nNone',
  },
  bellman: {
    title: 'Separate finite, unreachable and unbounded-below destinations', language: 'python',
    code: `${bellmanCode}

edges = [(0,1,2),(1,2,-4),(2,1,1),(2,3,2),(4,5,-3),(5,4,1)]
print(bellman_ford(6, edges, 0)[0])
print(bellman_ford(6, edges, 4)[0])
finite = [(0,1,2),(0,2,5),(2,1,-4),(1,3,2)]
distance, parent = bellman_ford(4, finite, 0)
print(distance)
print(route(parent, 0, 3))`,
    expected: '[0, -inf, -inf, -inf, inf, inf]\n[inf, inf, inf, inf, -inf, -inf]\n[0, 1, 5, 3]\n([0, 2, 1, 3], [1, 2, 3])',
  },
  budget: {
    title: 'Add an edge budget to the state instead of losing route feasibility', language: 'python',
    code: `${graphCode}

def edge_budget(n, edges, source, limit):
    adjacency(n, edges)
    check_source(n, source)
    if not isinstance(limit, int) or limit < 0:
        raise ValueError("Use a nonnegative edge limit")
    distance = [inf] * n
    distance[source] = 0
    for _ in range(limit):
        previous = distance
        distance = previous.copy()
        for u, v, weight in edges:
            if previous[u] != inf:
                distance[v] = min(distance[v], previous[u] + weight)
    return distance

edges = [(0,1,4),(0,2,1),(2,1,1),(1,3,1)]
print(edge_budget(4, edges, 0, 2))
print(edge_budget(4, edges, 0, 3))`,
    expected: '[0, 2, 1, 5]\n[0, 2, 1, 3]',
  },
  zeroOne: {
    title: 'Use two priority levels when every edge costs zero or one', language: 'python',
    code: `${graphCode}
from collections import deque

def zero_one_bfs(n, edges, source):
    graph = adjacency(n, edges)
    check_source(n, source)
    if any(weight not in (0, 1) for _, _, weight in edges):
        raise ValueError("Only 0/1 weights are supported")
    distance, parent = [inf] * n, [None] * n
    distance[source] = 0
    queue = deque([(0, source)])
    while queue:
        cost, u = queue.popleft()
        if cost != distance[u]:
            continue
        for v, weight, edge_id in graph[u]:
            candidate = cost + weight
            if candidate < distance[v]:
                distance[v], parent[v] = candidate, (u, edge_id)
                entry = (candidate, v)
                if weight == 0:
                    queue.appendleft(entry)
                else:
                    queue.append(entry)
    return distance, parent

edges = [(0,1,1),(0,2,0),(2,1,0),(1,3,1),(2,3,1)]
distance, parent = zero_one_bfs(5, edges, 0)
print(distance)
print(route(parent, 0, 1))`,
    expected: '[0, 0, 0, 1, inf]\n([0, 2, 1], [1, 2])',
  },
  floyd: {
    title: 'Allow one more intermediate vertex for every source-target pair', language: 'python',
    code: `${graphCode}

def floyd_warshall(n, edges):
    adjacency(n, edges)
    distance = [[inf] * n for _ in range(n)]
    for u in range(n):
        distance[u][u] = 0
    for u, v, weight in edges:
        distance[u][v] = min(distance[u][v], weight)
    for k in range(n):
        for u in range(n):
            if distance[u][k] == inf:
                continue
            for v in range(n):
                if distance[k][v] != inf:
                    distance[u][v] = min(distance[u][v], distance[u][k] + distance[k][v])
    # Read the completed finite matrix while writing a separate result.
    result = [row.copy() for row in distance]
    for k in range(n):
        if distance[k][k] < 0:
            for u in range(n):
                for v in range(n):
                    if distance[u][k] != inf and distance[k][v] != inf:
                        result[u][v] = -inf
    return result

edges = [(0,1,3),(1,2,-2),(0,2,8),(2,3,4)]
for row in floyd_warshall(4, edges):
    print(row)
print(floyd_warshall(3, [(0,1,1),(1,1,-1),(1,2,1)])[0])`,
    expected: '[0, 3, 1, 5]\n[inf, 0, -2, 2]\n[inf, inf, 0, 4]\n[inf, inf, inf, 0]\n[0, -inf, -inf]',
  },
  johnson: {
    title: 'Reweight signed edges without changing which route is best', language: 'python',
    code: `${dijkstraCode}
${bellmanCode.slice(graphCode.length)}

def johnson(n, edges):
    adjacency(n, edges)
    # A fresh source reaches every original vertex by a zero-weight edge.
    extended = list(edges) + [(n, v, 0) for v in range(n)]
    potential, _ = bellman_ford(n + 1, extended, n)
    if any(value == -inf for value in potential):
        raise ValueError("Johnson requires no negative cycle anywhere")
    adjusted = [(u, v, w + potential[u] - potential[v]) for u, v, w in edges]
    result = []
    for source in range(n):
        distance, _, _ = dijkstra(n, adjusted, source)
        result.append([value - potential[source] + potential[v]
                       if value != inf else inf for v, value in enumerate(distance)])
    return result

edges = [(0,1,2),(0,2,5),(2,1,-4),(1,3,2)]
for row in johnson(4, edges):
    print(row)`,
    expected: '[0, 1, 5, 3]\n[inf, 0, inf, 2]\n[inf, -4, 0, -2]\n[inf, inf, inf, 0]',
  },
  kruskal: {
    title: 'Accept globally light edges only when they join different components', language: 'python',
    code: `${dsuCode}

def kruskal(n, edges):
    adjacency(n, edges, directed=False)
    sets = DisjointSets(n)
    accepted, total = [], 0
    for edge_id in sorted(range(len(edges)), key=lambda i: (edges[i][2], i)):
        u, v, weight = edges[edge_id]
        if sets.union(u, v):
            accepted.append(edge_id)
            total += weight
    return total, accepted, n - len(accepted)

print(kruskal(6, ${forestFixture}))
print(kruskal(4, [(0,1,-3),(0,1,2),(2,2,-9)]))`,
    expected: '(13, [2, 1, 6, 8, 3], 1)\n(-3, [0], 3)',
  },
  prim: {
    title: 'Grow one component across its cheapest available boundary', language: 'python',
    code: `${graphCode}
from heapq import heappop, heappush

def prim_forest(n, edges):
    graph = adjacency(n, edges, directed=False)
    visited = [False] * n
    accepted, total, components = [], 0, 0
    heap = []
    def enter(u):
        visited[u] = True
        for v, weight, edge_id in graph[u]:
            if not visited[v]:
                heappush(heap, (weight, edge_id, v))
    for start in range(n):
        if visited[start]:
            continue
        components += 1
        enter(start)
        while heap:
            weight, edge_id, v = heappop(heap)
            if visited[v]:
                continue
            accepted.append(edge_id)
            total += weight
            enter(v)
    return total, accepted, components

print(prim_forest(6, ${forestFixture}))
print(prim_forest(4, [(0,1,-3),(0,1,2),(2,2,-9)]))`,
    expected: '(13, [1, 2, 3, 6, 8], 1)\n(-3, [0], 3)',
  },
  points: {
    title: 'Compute dense geometric edge costs on demand instead of storing every pair', language: 'python',
    code: `def connect_points(points):
    n = len(points)
    if n == 0:
        return 0, []
    best, parent = [float("inf")] * n, [None] * n
    used, selected = [False] * n, []
    best[0], total = 0, 0
    for _ in range(n):
        u = min((v for v in range(n) if not used[v]), key=lambda v: (best[v], v))
        used[u] = True
        total += best[u]
        if parent[u] is not None:
            selected.append((parent[u], u))
        x, y = points[u]
        for v, (a, b) in enumerate(points):
            weight = abs(x - a) + abs(y - b)
            if not used[v] and weight < best[v]:
                best[v], parent[v] = weight, u
    return total, selected

print(connect_points([(0,0),(2,0),(2,3),(5,3)]))
print(connect_points([]))`,
    expected: '(8, [(0, 1), (1, 2), (2, 3)])\n(0, [])',
  },
  topological: {
    title: 'Emit ready jobs and obtain a concrete cycle when ordering fails', language: 'python',
    code: `${topoCode}

def dfs_order_or_cycle(n, edges):
    graph = adjacency(n, edges)
    color, finished = [0] * n, []
    for start in range(n):
        if color[start]:
            continue
        color[start] = 1
        frames = [(start, 0)]
        active = [start]
        positions = {start: 0}
        while frames:
            u, index = frames[-1]
            if index == len(graph[u]):
                frames.pop()
                active.pop()
                del positions[u]
                color[u] = 2
                finished.append(u)
                continue
            frames[-1] = (u, index + 1)
            v = graph[u][index][0]
            if color[v] == 1:
                return None, active[positions[v]:] + [v]
            if color[v] == 0:
                color[v] = 1
                positions[v] = len(active)
                active.append(v)
                frames.append((v, 0))
    return finished[::-1], None

edges = ${dependencyFixture}
print(topological_order(6, edges))
print(dfs_order_or_cycle(6, edges))
print(dfs_order_or_cycle(6, edges + [(4,2,1)]))`,
    expected: '[0, 1, 2, 3, 4, 5]\n([1, 3, 0, 2, 4, 5], None)\n(None, [2, 4, 2])',
  },
  dag: {
    title: 'Process every signed-weight predecessor before its destination', language: 'python',
    code: `${topoCode}

def dag_shortest(n, edges, source):
    graph = adjacency(n, edges)
    check_source(n, source)
    order = topological_order(n, edges)
    distance, parent = [inf] * n, [None] * n
    distance[source] = 0
    for u in order:
        if distance[u] == inf:
            continue
        for v, weight, edge_id in graph[u]:
            candidate = distance[u] + weight
            if candidate < distance[v]:
                distance[v], parent[v] = candidate, (u, edge_id)
    return distance, parent

edges = [(0,1,2),(0,2,5),(2,1,-4),(1,3,2)]
distance, parent = dag_shortest(5, edges, 0)
print(distance)
print(route(parent, 0, 3))`,
    expected: '[0, 1, 5, 3, inf]\n([0, 2, 1, 3], [1, 2, 3])',
  },
  schedule: {
    title: 'Derive a feasible earliest schedule and its critical-chain certificate', language: 'python',
    code: `${topoCode}

def earliest_schedule(n, edges, duration):
    graph = adjacency(n, edges)
    if len(duration) != n or any(not isinstance(t, int) or t < 0 for t in duration):
        raise ValueError("One nonnegative integer duration per job")
    start, finish, previous = [0] * n, [0] * n, [None] * n
    for u in topological_order(n, edges):
        finish[u] = start[u] + duration[u]
        for v, _, _ in graph[u]:
            if finish[u] > start[v]:
                start[v], previous[v] = finish[u], u
    makespan = max(finish, default=0)
    chain = []
    u = finish.index(makespan) if n else None
    while u is not None:
        chain.append(u)
        u = previous[u]
    return start, finish, makespan, chain[::-1]

result = earliest_schedule(6, ${dependencyFixture}, [3,2,4,6,2,1])
for item in result:
    print(item)`,
    expected: '[0, 0, 3, 2, 8, 10]\n[3, 2, 7, 8, 10, 11]\n11\n[1, 3, 4, 5]',
  },
};

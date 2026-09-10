const flowCore = `from collections import deque

def validate_edges(n, edges):
    if type(n) is not int or n < 2:
        raise ValueError("At least two vertices are required")
    for edge in edges:
        if len(edge) != 3:
            raise ValueError("An edge is (from, to, capacity)")
        u, v, capacity = edge
        if any(type(x) is not int for x in edge):
            raise ValueError("Use integer vertices and capacities")
        if not (0 <= u < n and 0 <= v < n and capacity >= 0):
            raise ValueError("Invalid vertex or negative capacity")

class ResidualNetwork:
    def __init__(self, n, edges):
        self.edges = list(edges)
        validate_edges(n, self.edges)
        self.n = n
        self.adj = [[] for _ in range(n)]
        self.tail, self.to, self.remaining = [], [], []
        for u, v, capacity in self.edges:
            forward = len(self.to)
            self.tail.extend((u, v))
            self.to.extend((v, u))
            self.remaining.extend((capacity, 0))
            self.adj[u].append(forward)
            self.adj[v].append(forward + 1)

    def check_terminals(self, source, sink):
        if (type(source) is not int or type(sink) is not int
                or not 0 <= source < self.n or not 0 <= sink < self.n
                or source == sink):
            raise ValueError("Choose distinct valid source and sink")

    def search(self, source):
        # A dictionary avoids clearing all isolated vertices at every search.
        parent = {source: None}
        distance = {source: 0}
        queue = deque([source])
        while queue:
            u = queue.popleft()
            for arc in self.adj[u]:
                v = self.to[arc]
                if self.remaining[arc] > 0 and v not in parent:
                    parent[v] = arc
                    distance[v] = distance[u] + 1
                    queue.append(v)
        return parent, distance

    def send(self, arc, amount):
        self.remaining[arc] -= amount
        self.remaining[arc ^ 1] += amount

    def result(self, source, value, steps):
        parent, _ = self.search(source)
        side = set(parent)
        flows = [capacity - self.remaining[2 * i]
                 for i, (_, _, capacity) in enumerate(self.edges)]
        cut_edges = [i for i, (u, v, _) in enumerate(self.edges)
                     if u in side and v not in side]
        return {"value": value, "flows": flows, "source_side": sorted(side),
                "cut_edges": cut_edges,
                "cut_capacity": sum(self.edges[i][2] for i in cut_edges),
                "steps": steps}

def edmonds_karp(n, edges, source, sink, trace=False):
    network = ResidualNetwork(n, edges)
    network.check_terminals(source, sink)
    value, steps = 0, []
    while True:
        parent, _ = network.search(source)
        if sink not in parent:
            return network.result(source, value, steps)
        path = []
        vertex = sink
        while vertex != source:
            arc = parent[vertex]
            path.append(arc)
            vertex = network.tail[arc]
        path.reverse()
        amount = min(network.remaining[arc] for arc in path)
        for arc in path:
            network.send(arc, amount)
        value += amount
        if trace:
            steps.append({"delta": amount,
                          "arcs": [(arc // 2, 1 if arc % 2 == 0 else -1)
                                   for arc in path]})
`;

const defaultEdges = `edges = [(0, 1, 1), (0, 2, 1), (1, 3, 1), (1, 4, 1),
         (2, 3, 1), (3, 5, 1), (4, 5, 1)]
`;

const matchingCore = `
def bipartite_matching(left_count, right_count, pairs):
    if (type(left_count) is not int or type(right_count) is not int
            or left_count < 0 or right_count < 0):
        raise ValueError("Partition sizes must be nonnegative integers")
    pairs = list(pairs)
    for left, right in pairs:
        if (type(left) is not int or type(right) is not int
                or not 0 <= left < left_count or not 0 <= right < right_count):
            raise ValueError("Pair outside its partition")
    pairs = list(dict.fromkeys(pairs))
    source, sink = left_count + right_count, left_count + right_count + 1
    edges = [(source, left, 1) for left in range(left_count)]
    pair_offset = len(edges)
    edges.extend((left, left_count + right, 1) for left, right in pairs)
    edges.extend((left_count + right, sink, 1) for right in range(right_count))
    flow = edmonds_karp(sink + 1, edges, source, sink)
    matching = [pair for i, pair in enumerate(pairs)
                if flow["flows"][pair_offset + i] == 1]
    left_mate = dict(matching)
    right_mate = {right: left for left, right in matching}
    neighbors = [[] for _ in range(left_count)]
    for left, right in pairs:
        neighbors[left].append(right)
    seen_left = set(range(left_count)) - left_mate.keys()
    seen_right = set()
    queue = deque(sorted(seen_left))
    while queue:
        left = queue.popleft()
        for right in neighbors[left]:
            if left_mate.get(left) == right or right in seen_right:
                continue
            seen_right.add(right)
            # A reached unmatched right would give an augmenting path.
            assert right in right_mate
            mate = right_mate[right]
            if mate not in seen_left:
                seen_left.add(mate)
                queue.append(mate)
    return {"matching": matching,
            "cover_left": sorted(set(range(left_count)) - seen_left),
            "cover_right": sorted(seen_right),
            "shortage_left": sorted(seen_left),
            "shortage_neighbors": sorted(seen_right)}
`;

export const networkFlowExamples = {
  contraction: {
    title: 'Optional: sample an undirected global cut, preserving edge multiplicity',
    language: 'python',
    code: `import random

def karger_trial(n, edges, rng):
    if type(n) is not int or n < 2:
        raise ValueError("A nontrivial global cut needs at least two vertices")
    edges = list(edges)
    adjacency = [[] for _ in range(n)]
    for u, v in edges:
        if (type(u) is not int or type(v) is not int
                or not 0 <= u < n or not 0 <= v < n):
            raise ValueError("Valid original vertex IDs required")
        adjacency[u].append(v)
        adjacency[v].append(u)
    connected = {0}
    stack = [0]
    while stack:
        for v in adjacency[stack.pop()]:
            if v not in connected:
                connected.add(v)
                stack.append(v)
    if len(connected) < n:
        return 0, sorted(connected)
    owner = list(range(n))
    groups = n
    while groups > 2:
        # Original edge occurrences remain separate lottery tickets.
        active = [(u, v) for u, v in edges if owner[u] != owner[v]]
        u, v = rng.choice(active)
        keep, remove = owner[u], owner[v]
        owner = [keep if group == remove else group for group in owner]
        groups -= 1
    side = [v for v in range(n) if owner[v] == owner[0]]
    side_set = set(side)
    value = sum((u in side_set) != (v in side_set) for u, v in edges)
    return value, side

def repeat_karger(n, edges, trials, seed):
    if type(trials) is not int or trials < 1:
        raise ValueError("At least one trial")
    edges = list(edges)
    rng = random.Random(seed)
    return min((karger_trial(n, edges, rng) for _ in range(trials)),
               key=lambda result: result[0])

complete_four = [(u, v) for u in range(4) for v in range(u + 1, 4)]
print("one sampled cut", repeat_karger(4, complete_four, 1, 12))
print("best of 20", repeat_karger(4, complete_four, 20, 12))
print("disconnected", repeat_karger(4, [(0, 1), (2, 3)], 1, 7))
print("parallel edges", repeat_karger(2, [(0, 1), (0, 1), (0, 0)], 1, 7))`,
    expected: 'one sampled cut (4, [0, 3])\nbest of 20 (3, [0])\ndisconnected (0, [0, 1])\nparallel edges (2, [0])',
  },
  feasibility: {
    title: 'Audit a proposed allocation before optimizing it',
    language: 'python',
    code: `def audit_flow(n, edges, flows, source, sink):
    if (type(n) is not int or n < 2 or type(source) is not int
            or type(sink) is not int or not 0 <= source < n
            or not 0 <= sink < n or source == sink):
        raise ValueError("Distinct terminals in a network of at least two vertices")
    if len(edges) != len(flows):
        raise ValueError("One proposed flow per original edge")
    incoming, outgoing = [0] * n, [0] * n
    bad_edges = []
    for i, ((u, v, capacity), amount) in enumerate(zip(edges, flows)):
        if (any(type(x) is not int for x in (u, v, capacity, amount))
                or not 0 <= u < n or not 0 <= v < n or capacity < 0):
            raise ValueError("Integer vertices, flows and nonnegative capacities")
        if not 0 <= amount <= capacity:
            bad_edges.append(i)
        outgoing[u] += amount
        incoming[v] += amount
    bad_vertices = [v for v in range(n) if v not in (source, sink)
                    and incoming[v] != outgoing[v]]
    source_value = outgoing[source] - incoming[source]
    sink_value = incoming[sink] - outgoing[sink]
    feasible = not bad_edges and not bad_vertices and source_value == sink_value
    return feasible, bad_edges, bad_vertices, source_value, sink_value

${defaultEdges}
for proposed in ([1, 0, 1, 0, 0, 1, 0],
                 [1, 0, 0, 0, 0, 0, 0],
                 [2, 0, 2, 0, 0, 2, 0]):
    print(audit_flow(6, edges, proposed, 0, 5))`,
    expected: '(True, [], [], 1, 1)\n(False, [], [1], 1, 0)\n(False, [0, 2, 5], [], 2, 2)',
  },
  augment: {
    title: 'Run Edmonds–Karp and return a flow plus a cut',
    language: 'python',
    code: `${flowCore}
${defaultEdges}
answer = edmonds_karp(6, edges, 0, 5, trace=True)
for step in answer["steps"]:
    print("send", step["delta"], "using", step["arcs"])
print("flows", answer["flows"])
print("value", answer["value"])
print("source side", answer["source_side"])
print("cut edge IDs", answer["cut_edges"], "capacity", answer["cut_capacity"])`,
    expected: 'send 1 using [(0, 1), (2, 1), (5, 1)]\nsend 1 using [(1, 1), (4, 1), (2, -1), (3, 1), (6, 1)]\nflows [1, 1, 0, 1, 1, 1, 1]\nvalue 2\nsource side [0]\ncut edge IDs [0, 1] capacity 2',
  },
  dinic: {
    title: 'Optional: block a whole set of shortest residual paths',
    language: 'python',
    code: `${flowCore}
def dinic(n, edges, source, sink):
    network = ResidualNetwork(n, edges)
    network.check_terminals(source, sink)
    value = 0
    upper = sum(capacity for _, _, capacity in network.edges)
    while True:
        _, level = network.search(source)
        if sink not in level:
            return network.result(source, value, [])
        cursor = [0] * n

        def push(u, limit):
            if u == sink:
                return limit
            while cursor[u] < len(network.adj[u]):
                arc = network.adj[u][cursor[u]]
                v = network.to[arc]
                if (network.remaining[arc] > 0
                        and level.get(v) == level[u] + 1):
                    sent = push(v, min(limit, network.remaining[arc]))
                    if sent:
                        network.send(arc, sent)
                        return sent
                cursor[u] += 1
            return 0

        while True:
            sent = push(source, upper)
            if not sent:
                break
            value += sent

${defaultEdges}
result = dinic(6, edges, 0, 5)
print("value", result["value"])
print("flows", result["flows"])
print("cut capacity", result["cut_capacity"])`,
    expected: 'value 2\nflows [1, 1, 0, 1, 1, 1, 1]\ncut capacity 2',
  },
  matching: {
    title: 'Return a matching, minimum cover and shortage witness',
    language: 'python',
    code: `${flowCore}${matchingCore}
for pairs in ([(0, 0), (0, 1), (1, 0), (2, 2)],
              [(0, 0), (0, 1), (1, 0), (2, 1)]):
    answer = bipartite_matching(3, 3, pairs)
    print("matching", answer["matching"])
    print("cover L/R", answer["cover_left"], answer["cover_right"])
    print("shortage L/N(L)", answer["shortage_left"],
          answer["shortage_neighbors"])
print("empty", bipartite_matching(0, 0, []))`,
    expected: "matching [(0, 1), (1, 0), (2, 2)]\ncover L/R [0, 1, 2] []\nshortage L/N(L) [] []\nmatching [(0, 0), (2, 1)]\ncover L/R [] [0, 1]\nshortage L/N(L) [0, 1, 2] [0, 1]\nempty {'matching': [], 'cover_left': [], 'cover_right': [], 'shortage_left': [], 'shortage_neighbors': []}",
  },
  vertex: {
    title: 'Put a capacity on a vertex by splitting it',
    language: 'python',
    code: `${flowCore}
def vertex_capacity_flow(n, edges, vertex_limits, source, sink):
    edges = list(edges)
    validate_edges(n, edges)
    if (type(source) is not int or type(sink) is not int
            or not 0 <= source < n or not 0 <= sink < n or source == sink):
        raise ValueError("Distinct valid source and sink")
    if set(vertex_limits) != set(range(n)) - {source, sink}:
        raise ValueError("Supply one limit for each internal vertex")
    if any(type(limit) is not int or limit < 0 for limit in vertex_limits.values()):
        raise ValueError("Nonnegative integer vertex limits")
    # in(v)=2*v, out(v)=2*v+1. Start at source-out; end at sink-in.
    transformed = [(2*u + 1, 2*v, cap) for u, v, cap in edges]
    transformed.extend((2*v, 2*v + 1, limit)
                       for v, limit in vertex_limits.items())
    answer = edmonds_karp(2*n, transformed, 2*source + 1, 2*sink)
    return answer["value"], answer["flows"][:len(edges)]

edges = [(0, 1, 5), (0, 2, 5), (1, 2, 5), (1, 3, 5), (2, 3, 5)]
value, original_flows = vertex_capacity_flow(4, edges, {1: 2, 2: 3}, 0, 3)
print("value", value)
print("original edge flows", original_flows)`,
    expected: 'value 5\noriginal edge flows [2, 3, 0, 2, 3]',
  },
  supply: {
    title: 'Distinguish supply ceilings from required demand',
    language: 'python',
    code: `${flowCore}
def allocate_supplies(n, edges, supply, demand):
    edges = list(edges)
    validate_edges(n, edges)
    if set(supply) & set(demand):
        raise ValueError("Use disjoint supply and demand vertices here")
    for limits in (supply, demand):
        for v, limit in limits.items():
            if (type(v) is not int or not 0 <= v < n
                    or type(limit) is not int or limit < 0):
                raise ValueError("Valid vertices and nonnegative integer limits")
    transformed = list(edges)
    supply_ids, demand_ids = {}, {}
    for v, limit in supply.items():
        supply_ids[v] = len(transformed)
        transformed.append((n, v, limit))
    for v, limit in demand.items():
        demand_ids[v] = len(transformed)
        transformed.append((v, n + 1, limit))
    answer = edmonds_karp(n + 2, transformed, n, n + 1)
    unmet = {v: demand[v] - answer["flows"][i] for v, i in demand_ids.items()}
    unused = {v: supply[v] - answer["flows"][i] for v, i in supply_ids.items()}
    return {"delivered": answer["value"], "all_demands_met": not any(unmet.values()),
            "unmet": unmet, "unused": unused,
            "flows": answer["flows"][:len(edges)]}

edges = [(0, 2, 2), (0, 3, 1), (1, 2, 1), (1, 3, 2)]
print(allocate_supplies(4, edges, {0: 3, 1: 2}, {2: 2, 3: 2}))
print(allocate_supplies(4, edges, {0: 3, 1: 2}, {2: 4, 3: 2}))`,
    expected: "{'delivered': 4, 'all_demands_met': True, 'unmet': {2: 0, 3: 0}, 'unused': {0: 0, 1: 1}, 'flows': [2, 1, 0, 1]}\n{'delivered': 5, 'all_demands_met': False, 'unmet': {2: 1, 3: 0}, 'unused': {0: 0, 1: 0}, 'flows': [2, 1, 1, 1]}",
  },
  circulation: {
    title: 'Meet mandatory lower bounds without inventing flow',
    language: 'python',
    code: `${flowCore}
def lower_bound_circulation(n, edges):
    if type(n) is not int or n < 1:
        raise ValueError("At least one original vertex")
    edges = list(edges)
    balance = [0] * n
    residual_edges = []
    for u, v, lower, upper in edges:
        if (any(type(x) is not int for x in (u, v, lower, upper))
                or not 0 <= u < n or not 0 <= v < n or not 0 <= lower <= upper):
            raise ValueError("Valid vertices and 0 <= integer lower <= upper")
        residual_edges.append((u, v, upper - lower))
        balance[u] -= lower
        balance[v] += lower
    for v, net_lower_inflow in enumerate(balance):
        if net_lower_inflow > 0:
            residual_edges.append((n, v, net_lower_inflow))
        elif net_lower_inflow < 0:
            residual_edges.append((v, n + 1, -net_lower_inflow))
    required = sum(max(0, amount) for amount in balance)
    answer = edmonds_karp(n + 2, residual_edges, n, n + 1)
    if answer["value"] != required:
        return None
    return [lower + answer["flows"][i]
            for i, (_, _, lower, _) in enumerate(edges)]

print("cycle", lower_bound_circulation(3, [(0, 1, 2, 4), (1, 2, 1, 3), (2, 0, 0, 3)]))
print("no return", lower_bound_circulation(2, [(0, 1, 1, 3)]))
print("no mandatory flow", lower_bound_circulation(2, [(0, 1, 0, 3)]))`,
    expected: 'cycle [2, 2, 2]\nno return None\nno mandatory flow [0]',
  },
  pixels: {
    title: 'Minimize a binary labeling energy through a cut',
    language: 'python',
    code: `${flowCore}
def binary_label_cut(background, foreground, neighbors, penalty):
    neighbors = list(neighbors)
    n = len(background)
    if len(foreground) != n:
        raise ValueError("Two costs per vertex")
    if any(type(cost) is not int or cost < 0
           for cost in [*background, *foreground, penalty]):
        raise ValueError("Nonnegative integer costs and penalty")
    source, sink = n, n + 1
    edges = []
    for v in range(n):
        # A source-side vertex is foreground.
        edges.extend(((source, v, background[v]), (v, sink, foreground[v])))
    for u, v in neighbors:
        if (type(u) is not int or type(v) is not int
                or not 0 <= u < n or not 0 <= v < n or u == v):
            raise ValueError("Distinct valid neighboring vertices")
        # Each listed pair is one penalty term; list it once.
        edges.extend(((u, v, penalty), (v, u, penalty)))
    answer = edmonds_karp(n + 2, edges, source, sink)
    side = set(answer["source_side"])
    labels = [v in side for v in range(n)]
    unary = sum(foreground[v] if labels[v] else background[v] for v in range(n))
    boundary_count = sum(labels[u] != labels[v] for u, v in neighbors)
    return labels, unary, boundary_count * penalty, answer["value"]

background = [0, 1, 6, 0, 5, 6]
foreground = [6, 4, 0, 5, 1, 0]
neighbors = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]
for penalty in (0, 2, 5):
    labels, unary, pairwise, minimum = binary_label_cut(background, foreground, neighbors, penalty)
    print("penalty", penalty, "labels", ["F" if value else "B" for value in labels])
    print("unary", unary, "boundary", pairwise, "minimum", minimum)`,
    expected: "penalty 0 labels ['B', 'B', 'F', 'B', 'F', 'F']\nunary 2 boundary 0 minimum 2\npenalty 2 labels ['B', 'B', 'F', 'B', 'F', 'F']\nunary 2 boundary 6 minimum 8\npenalty 5 labels ['B', 'F', 'F', 'B', 'F', 'F']\nunary 5 boundary 10 minimum 15",
  },
};

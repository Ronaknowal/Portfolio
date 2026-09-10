const graphSetup = `from collections import deque

def build_graph(vertices, edges, directed=False):
    vertices = list(vertices)
    if any(vertex is None for vertex in vertices):
        raise ValueError("None is reserved for a missing parent")
    if len(set(vertices)) != len(vertices):
        raise ValueError("vertex labels must be unique")
    graph = {vertex: [] for vertex in vertices}
    neighbors_seen = {vertex: set() for vertex in vertices}
    def connect(first, second):
        if second not in neighbors_seen[first]:
            graph[first].append(second)
            neighbors_seen[first].add(second)
    for first, second in edges:
        if first not in graph or second not in graph:
            raise ValueError("every endpoint must be a declared vertex")
        connect(first, second)
        if not directed:
            connect(second, first)
    return graph

VERTICES = list("ABCDEFGH")
EDGES = [("A", "B"), ("A", "C"), ("B", "D"),
         ("C", "D"), ("D", "E"), ("F", "G")]
`;

const bfsImplementation = `def bfs(graph, source):
    if source not in graph:
        raise KeyError("unknown source")
    distance = {source: 0}
    parent = {source: None}
    queue = deque([source])
    order = []
    while queue:
        vertex = queue.popleft()
        order.append(vertex)
        for neighbor in graph[vertex]:
            if neighbor not in distance:
                distance[neighbor] = distance[vertex] + 1
                parent[neighbor] = vertex
                queue.append(neighbor)
    return order, distance, parent

def path_to(parent, target):
    if target not in parent:
        return None
    path = []
    while target is not None:
        path.append(target)
        target = parent[target]
    path.reverse()
    return path
`;

const gridImplementation = `from collections import deque

def grid_distances(grid, sources):
    if not grid or not grid[0]:
        raise ValueError("grid must be nonempty")
    rows, columns = len(grid), len(grid[0])
    if any(len(row) != columns for row in grid):
        raise ValueError("grid must be rectangular")
    if any(cell not in (".", "#") for row in grid for cell in row):
        raise ValueError("use . for open cells and # for walls")
    distance = [[-1] * columns for _ in range(rows)]
    parent = {}
    queue = deque()
    for row, column in sources:
        if not (0 <= row < rows and 0 <= column < columns):
            raise ValueError("source is outside the grid")
        if grid[row][column] == "#":
            raise ValueError("source is blocked")
        if (row, column) not in parent:
            distance[row][column] = 0
            parent[row, column] = None
            queue.append((row, column))
    # Increasing (row, column) neighbor order: up, left, right, down.
    directions = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    while queue:
        row, column = queue.popleft()
        for dr, dc in directions:
            following_row, following_column = row + dr, column + dc
            if not (0 <= following_row < rows and
                    0 <= following_column < columns):
                continue
            if grid[following_row][following_column] == "#":
                continue
            if distance[following_row][following_column] != -1:
                continue
            distance[following_row][following_column] = distance[row][column] + 1
            parent[following_row, following_column] = (row, column)
            queue.append((following_row, following_column))
    return distance, parent
`;

export const graphTraversalExamples = {
  representations: {
    title: 'Build one graph without losing isolated vertices',
    code: `${graphSetup}
graph = build_graph(VERTICES, EDGES)
for vertex in VERTICES:
    print(vertex, graph[vertex])
index = {vertex: position for position, vertex in enumerate(VERTICES)}
matrix = [[0] * len(VERTICES) for _ in VERTICES]
for vertex, neighbors in graph.items():
    for neighbor in neighbors:
        matrix[index[vertex]][index[neighbor]] = 1
print("matrix row A:", matrix[index["A"]])
print("matrix row H:", matrix[index["H"]])
directed = build_graph(["A", "B"], [("A", "B"), ("A", "B")], True)
print("directed:", directed)
`,
    expected: "A ['B', 'C']\nB ['A', 'D']\nC ['A', 'D']\nD ['B', 'C', 'E']\nE ['D']\nF ['G']\nG ['F']\nH []\nmatrix row A: [0, 1, 1, 0, 0, 0, 0, 0]\nmatrix row H: [0, 0, 0, 0, 0, 0, 0, 0]\ndirected: {'A': ['B'], 'B': []}",
  },
  breadthFirst: {
    title: 'Find every reachable distance and recover one route',
    code: `${graphSetup}
${bfsImplementation}
graph = build_graph(VERTICES, EDGES)
order, distance, parent = bfs(graph, "A")
print("processed:", order)
print("distances:", [(vertex, distance.get(vertex)) for vertex in VERTICES])
print("route to E:", path_to(parent, "E"))
print("route to A:", path_to(parent, "A"))
print("route to H:", path_to(parent, "H"))
`,
    expected: "processed: ['A', 'B', 'C', 'D', 'E']\ndistances: [('A', 0), ('B', 1), ('C', 1), ('D', 2), ('E', 3), ('F', None), ('G', None), ('H', None)]\nroute to E: ['A', 'B', 'D', 'E']\nroute to A: ['A']\nroute to H: None",
  },
  depthFirst: {
    title: 'Keep an explicit frame for unfinished neighbor iteration',
    code: `${graphSetup}
def dfs_frames(graph, source):
    if source not in graph:
        raise KeyError("unknown source")
    parent = {source: None}
    entered = [source]
    finished = []
    stack = [(source, iter(graph[source]))]
    while stack:
        vertex, neighbors = stack[-1]
        try:
            neighbor = next(neighbors)
        except StopIteration:
            finished.append(vertex)
            stack.pop()
            continue
        if neighbor not in parent:
            parent[neighbor] = vertex
            entered.append(neighbor)
            stack.append((neighbor, iter(graph[neighbor])))
    return entered, finished, parent

graph = build_graph(VERTICES, EDGES)
entered, finished, parent = dfs_frames(graph, "A")
print("entered:", entered)
print("finished:", finished)
print("parent of C:", parent["C"])
print("unreached:", [vertex for vertex in VERTICES if vertex not in parent])
`,
    expected: "entered: ['A', 'B', 'D', 'C', 'E']\nfinished: ['C', 'E', 'D', 'B', 'A']\nparent of C: D\nunreached: ['F', 'G', 'H']",
  },
  components: {
    title: 'Restart at each undiscovered vertex to label every component',
    code: `${graphSetup}
def connected_components(graph):
    # Requires an undirected graph with every vertex declared.
    label = {}
    groups = []
    for start in graph:
        if start in label:
            continue
        component_id = len(groups)
        group = []
        queue = deque([start])
        label[start] = component_id
        while queue:
            vertex = queue.popleft()
            group.append(vertex)
            for neighbor in graph[vertex]:
                if neighbor not in label:
                    label[neighbor] = component_id
                    queue.append(neighbor)
        groups.append(group)
    return groups, label

groups, label = connected_components(build_graph(VERTICES, EDGES))
print(groups)
print("A and E:", label["A"] == label["E"])
print("A and F:", label["A"] == label["F"])
print("empty:", connected_components({})[0])
`,
    expected: "[['A', 'B', 'C', 'D', 'E'], ['F', 'G'], ['H']]\nA and E: True\nA and F: False\nempty: []",
  },
  gridWavefront: {
    title: 'Generate grid neighbors instead of storing every edge',
    code: `${gridImplementation}
grid = ["...#.", ".#.#.", ".#...", "...#.", "....."]
distance, parent = grid_distances(grid, [(0, 0)])
print("one source:")
for row in distance:
    print(row)
multiple, _ = grid_distances(grid, [(0, 0), (4, 4)])
print("two sources:")
for row in multiple:
    print(row)
print("walls preserved:", grid[0] == "...#.")
`,
    expected: 'one source:\n[0, 1, 2, -1, 8]\n[1, -1, 3, -1, 7]\n[2, -1, 4, 5, 6]\n[3, 4, 5, -1, 7]\n[4, 5, 6, 7, 8]\ntwo sources:\n[0, 1, 2, -1, 4]\n[1, -1, 3, -1, 3]\n[2, -1, 4, 3, 2]\n[3, 4, 3, -1, 1]\n[4, 3, 2, 1, 0]\nwalls preserved: True',
  },
  floodFill: {
    title: 'Recolor one connected region, preserving the original image',
    code: `from collections import deque

def flood_fill(image, start, new_color):
    if not image or not image[0]:
        raise ValueError("image must be nonempty")
    rows, columns = len(image), len(image[0])
    if any(len(row) != columns for row in image):
        raise ValueError("image must be rectangular")
    row, column = start
    if not (0 <= row < rows and 0 <= column < columns):
        raise ValueError("start is outside image")
    result = [line.copy() for line in image]
    old_color = result[row][column]
    if old_color == new_color:
        return result
    result[row][column] = new_color
    queue = deque([(row, column)])
    while queue:
        row, column = queue.popleft()
        for dr, dc in [(-1, 0), (0, -1), (0, 1), (1, 0)]:
            following_row, following_column = row + dr, column + dc
            if (0 <= following_row < rows and 0 <= following_column < columns
                    and result[following_row][following_column] == old_color):
                result[following_row][following_column] = new_color
                queue.append((following_row, following_column))
    return result

image = [[1, 1, 0], [1, 0, 0], [0, 0, 1]]
changed = flood_fill(image, (0, 0), 7)
print(changed)
print("original:", image)
print("same color:", flood_fill(image, (0, 0), 1) == image)
`,
    expected: '[[7, 7, 0], [7, 0, 0], [0, 0, 1]]\noriginal: [[1, 1, 0], [1, 0, 0], [0, 0, 1]]\nsame color: True',
  },
  cloneGraph: {
    title: 'Copy a cyclic graph without duplicating shared nodes',
    code: `from collections import deque

class Node:
    def __init__(self, value):
        self.value = value
        self.neighbors = []

def clone_graph(start):
    if start is None:
        return None
    copies = {start: Node(start.value)}
    queue = deque([start])
    while queue:
        original = queue.popleft()
        for neighbor in original.neighbors:
            if neighbor not in copies:
                copies[neighbor] = Node(neighbor.value)
                queue.append(neighbor)
            copies[original].neighbors.append(copies[neighbor])
    return copies[start]

first, second, shared = Node("A"), Node("B"), Node("C")
first.neighbors = [second, shared]
second.neighbors = [shared]
shared.neighbors = [first]
copy = clone_graph(first)
print("new root:", copy is not first)
print("shared C:", copy.neighbors[0].neighbors[0] is copy.neighbors[1])
print("cycle retained:", copy.neighbors[1].neighbors[0] is copy)
print("empty:", clone_graph(None))
`,
    expected: 'new root: True\nshared C: True\ncycle retained: True\nempty: None',
  },
  directedCycle: {
    title: 'Recognize a back edge to active work, not just any revisit',
    code: `${graphSetup}
def has_directed_cycle(graph):
    # 0 = unseen, 1 = active frame, 2 = finished
    state = {vertex: 0 for vertex in graph}
    for start in graph:
        if state[start] != 0:
            continue
        state[start] = 1
        stack = [(start, iter(graph[start]))]
        while stack:
            vertex, neighbors = stack[-1]
            try:
                neighbor = next(neighbors)
            except StopIteration:
                state[vertex] = 2
                stack.pop()
                continue
            if state[neighbor] == 1:
                return True
            if state[neighbor] == 0:
                state[neighbor] = 1
                stack.append((neighbor, iter(graph[neighbor])))
    return False

diamond = [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")]
print("shared descendant:", has_directed_cycle(build_graph("ABCD", diamond, True)))
print("back edge:", has_directed_cycle(build_graph("ABCD", diamond + [("D", "A")], True)))
print("self loop:", has_directed_cycle(build_graph("A", [("A", "A")], True)))
print("empty:", has_directed_cycle({}))
`,
    expected: 'shared descendant: False\nback edge: True\nself loop: True\nempty: False',
  },
  bipartition: {
    title: 'Carry a two-group constraint through every component',
    code: `${graphSetup}
def two_color(graph):
    # Undirected graph; every edge must join different groups.
    color = {}
    for start in graph:
        if start in color:
            continue
        color[start] = 0
        queue = deque([start])
        while queue:
            vertex = queue.popleft()
            for neighbor in graph[vertex]:
                if neighbor not in color:
                    color[neighbor] = 1 - color[vertex]
                    queue.append(neighbor)
                elif color[neighbor] == color[vertex]:
                    return None
    return color

square = build_graph("ABCD", [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")])
triangle = build_graph("ABC", [("A", "B"), ("B", "C"), ("C", "A")])
print("square:", two_color(square))
print("triangle:", two_color(triangle))
print("empty:", two_color({}))
`,
    expected: "square: {'A': 0, 'B': 1, 'D': 1, 'C': 0}\ntriangle: None\nempty: {}",
  },
  diagonalPractice: {
    title: 'Solution: allow diagonal moves and count visited cells',
    code: `from collections import deque

def shortest_cell_path(grid):
    # Rectangular 0/1 grid: 0 open, 1 blocked. Diagonal contact is allowed.
    if not grid or not grid[0]:
        return -1
    rows, columns = len(grid), len(grid[0])
    if any(len(row) != columns for row in grid):
        raise ValueError("grid must be rectangular")
    if any(cell not in (0, 1) for row in grid for cell in row):
        raise ValueError("cells must be 0 or 1")
    if grid[0][0] or grid[-1][-1]:
        return -1
    queue = deque([(0, 0)])
    distance = {(0, 0): 1}  # count the starting cell, not zero edges
    directions = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                  if (dr, dc) != (0, 0)]
    while queue:
        row, column = queue.popleft()
        if (row, column) == (rows - 1, columns - 1):
            return distance[row, column]
        for dr, dc in directions:
            following = (row + dr, column + dc)
            r, c = following
            if (0 <= r < rows and 0 <= c < columns and grid[r][c] == 0
                    and following not in distance):
                distance[following] = distance[row, column] + 1
                queue.append(following)
    return -1

print(shortest_cell_path([[0, 1], [1, 0]]))
print(shortest_cell_path([[0]]))
print(shortest_cell_path([[1]]))
print(shortest_cell_path([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
`,
    expected: '2\n1\n-1\n-1',
  },
};

def independent_tree(weights, edges, root=0, parent_selected=False):
    if any(type(weight) is not int for weight in weights):
        raise ValueError("Use integer weights; negative weights are permitted")
    size = len(weights)
    if type(root) is not int or not 0 <= root < max(1, size) or type(parent_selected) is not bool:
        raise ValueError("Use a valid root (0 for empty) and Boolean parent condition")
    if len(edges) != max(0, size - 1):
        raise ValueError("Use n-1 edges, or zero for an empty tree")
    adjacency = [[] for _ in weights]
    seen_edges = set()
    for edge in edges:
        if len(edge) != 2 or any(type(node) is not int or not 0 <= node < size for node in edge):
            raise ValueError("Edges must name two existing node IDs")
        first, second = edge
        key = (min(first, second), max(first, second))
        if first == second or key in seen_edges:
            raise ValueError("No self-loops or duplicate edges")
        seen_edges.add(key)
        adjacency[first].append(second)
        adjacency[second].append(first)
    children = [[] for _ in weights]
    parents, visited, order = [None] * size, set(), []
    stack = [root] if size else []
    while stack:
        node = stack.pop()
        if node in visited:
            raise ValueError("A tree cannot contain a cycle")
        visited.add(node)
        order.append(node)
        for child in adjacency[node]:
            if child == parents[node]:
                continue
            if child in visited:
                raise ValueError("A tree cannot contain a cycle")
            parents[child] = node
            children[node].append(child)
            stack.append(child)
    if len(visited) != size:
        raise ValueError("A tree must be connected")
    free, blocked, take = [0] * size, [0] * size, [0] * size
    for node in reversed(order):
        blocked[node] = sum(free[child] for child in children[node])
        take[node] = weights[node] + sum(blocked[child] for child in children[node])
        free[node] = max(blocked[node], take[node])
    selected = []
    stack = [(root, parent_selected)] if size else []
    while stack:
        node, forbidden = stack.pop()
        choose = not forbidden and take[node] > blocked[node]  # Skip ties.
        if choose:
            selected.append(node)
        stack.extend((child, choose) for child in reversed(children[node]))
    value = (blocked[root] if parent_selected else free[root]) if size else 0
    return value, selected


weights = [5, 9, 2, 4, 1, 6, 3]
edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
print("root free:", independent_tree(weights, edges))
print("external parent selected:", independent_tree(weights, edges, parent_selected=True))
print("negative chain:", independent_tree([-2, -5, -1], [(0, 1), (1, 2)]))
print("empty:", independent_tree([], []))

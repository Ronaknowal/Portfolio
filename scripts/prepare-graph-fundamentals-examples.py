"""Execute complete learner programs and save their literal stdout fixtures."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = json.loads((ROOT / 'docs/teaching/evidence/graph-fundamentals-original-content.json').read_text(encoding='utf-8'))
PROGRAMS = {}


def add(key, title, question, code, note):
    PROGRAMS[key] = {'title': title, 'question': question, 'language': 'python', 'code': textwrap.dedent(code).strip(), 'note': note}


add('original', 'Find the components and build their Laplacian',
    'Which vertices can communicate, and how does each disconnected region appear in the matrix?',
    ARCHIVE['program'],
    'The complete five-vertex example uses only the standard library. The two components will reappear as two independent zero-energy signals.')

add('representations', 'Declare the vertices before reading the edges',
    'What happens to a repeated edge, a reversed arrow and a vertex absent from every edge?', r'''
def adjacency_matrix(labels, edges, directed=False):
    if len(set(labels)) != len(labels):
        raise ValueError("Vertex labels must be unique")
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for source, target, weight in edges:
        if source not in index or target not in index:
            raise ValueError("Unknown endpoint")
        if weight < 0:
            raise ValueError("This example uses nonnegative coupling weights")
        i, j = index[source], index[target]
        matrix[i][j] += weight
        if not directed and i != j:
            matrix[j][i] += weight
    return matrix

labels = ["A", "B", "C", "D"]
edges = [("A", "B", 2), ("A", "B", 1), ("B", "C", 4)]
directed = adjacency_matrix(labels, edges, directed=True)
undirected = adjacency_matrix(labels, edges)
print("directed:", directed)
print("undirected:", undirected)
print("out-strength:", [sum(row) for row in directed])
print("in-strength:", [sum(row[j] for row in directed) for j in range(4)])
print("new label order:", adjacency_matrix(["C", "A", "D", "B"], edges))
''', 'Parallel records add their coupling weights. That is an explicit modeling choice; duplicate measurements might instead require deduplication. D remains represented as an isolated vertex.')

add('walks', 'Read powers as sums over walks',
    'For A—B—C with weights 2 and 1, compare a two-step return, a two-step A-to-C walk and a three-step A-to-C walk.', r'''
def multiply(left, right):
    return [[sum(left[i][k] * right[k][j] for k in range(len(right)))
             for j in range(len(right[0]))] for i in range(len(left))]

def matrix_power(matrix, exponent):
    if exponent < 0:
        raise ValueError("A walk length is nonnegative")
    result = [[int(i == j) for j in range(len(matrix))] for i in range(len(matrix))]
    for _ in range(exponent):
        result = multiply(result, matrix)
    return result

adjacency = [[0, 2, 0], [2, 0, 1], [0, 1, 0]]
print("A^0:", matrix_power(adjacency, 0))
print("A^2:", matrix_power(adjacency, 2))
print("A-to-A, 2 steps:", matrix_power(adjacency, 2)[0][0])
print("A-to-C, 2 steps:", matrix_power(adjacency, 2)[0][2])
print("A-to-C, 3 steps:", matrix_power(adjacency, 3)[0][2])
support = [[int(value > 0) for value in row] for row in adjacency]
print("unweighted 2-step walk count:", matrix_power(support, 2)[0][2])
''', 'The return A→B→A repeats a vertex and is a walk, not a simple path. A weighted entry sums products of edge weights; it is neither a shortest distance nor a probability.')

add('incidence', 'Compute drops, currents and net outflow',
    'Can reversing an arbitrary edge orientation change the Laplacian or its energy?', r'''
def edge_calculation(n, edges, signal):
    incidence = [[int(i == u) - int(i == v) for i in range(n)]
                 for u, v, weight in edges]
    drops = [sum(row[i] * signal[i] for i in range(n)) for row in incidence]
    flows = [edge[2] * drop for edge, drop in zip(edges, drops)]
    action = [sum(incidence[e][i] * flows[e] for e in range(len(edges))) for i in range(n)]
    laplacian = [[sum(edge[2] * incidence[e][i] * incidence[e][j]
                      for e, edge in enumerate(edges)) for j in range(n)] for i in range(n)]
    energy = sum(drop * flow for drop, flow in zip(drops, flows))
    return incidence, drops, flows, action, laplacian, energy

edges = [(0, 1, 2), (1, 2, 1), (3, 4, 2)]
signal = [3, 1, 0, 2, 2]
incidence, drops, flows, action, laplacian, energy = edge_calculation(5, edges, signal)
print("C:", incidence)
print("drops:", drops)
print("flows:", flows)
print("Lx:", action)
print("energy:", energy)
reversed_edges = [(v, u, weight) for u, v, weight in edges]
reversed_result = edge_calculation(5, reversed_edges, signal)
print("reversed drops:", reversed_result[1])
print("same Laplacian:", reversed_result[4] == laplacian)
print("same energy:", reversed_result[5] == energy)
''', 'Positive flow follows the declared tail-to-head orientation. A negative value means physical flow would go the other way. The arbitrary arrows organize signs; the underlying coupling is undirected.')

add('kernel', 'Check component indicators with exact arithmetic',
    'How do two nontrivial components and an isolate create three independent null directions?', r'''
from fractions import Fraction

def exact_rank(matrix):
    rows = [[Fraction(value) for value in row] for row in matrix]
    pivot_row = 0
    for column in range(len(rows[0]) if rows else 0):
        pivot = next((i for i in range(pivot_row, len(rows)) if rows[i][column]), None)
        if pivot is None:
            continue
        rows[pivot_row], rows[pivot] = rows[pivot], rows[pivot_row]
        scale = rows[pivot_row][column]
        rows[pivot_row] = [value / scale for value in rows[pivot_row]]
        for i in range(len(rows)):
            if i != pivot_row:
                scale = rows[i][column]
                rows[i] = [a - scale * b for a, b in zip(rows[i], rows[pivot_row])]
        pivot_row += 1
        if pivot_row == len(rows):
            break
    return pivot_row

n = 6
edges = [(0, 1, 2), (1, 2, 1), (3, 4, 2)]
laplacian = [[0] * n for _ in range(n)]
for u, v, weight in edges:
    laplacian[u][u] += weight
    laplacian[v][v] += weight
    laplacian[u][v] -= weight
    laplacian[v][u] -= weight
components = [[0, 1, 2], [3, 4], [5]]
for component in components:
    indicator = [int(i in component) for i in range(n)]
    print(component, [sum(row[i] * indicator[i] for i in range(n)) for row in laplacian])
print("rank:", exact_rank(laplacian))
print("nullity:", n - exact_rank(laplacian))
# A directed arrow 0 -> 1 has a very different quadratic form.
directed_laplacian = [[1, -1], [0, 0]]
x = [1, 2]
print("directed quadratic value:", sum(x[i] * sum(directed_laplacian[i][j] * x[j] for j in range(2)) for i in range(2)))
''', 'Exact elimination confirms this finite example; the energy proof establishes the general undirected result. The directed value is negative, so the symmetric positive-semidefinite theorem cannot be imported unchanged.')

add('normalization', 'Keep the isolated row explicit',
    'For a weighted path plus an isolate, compare H L H with blindly writing I − H A H, then add a self-loop.', r'''
from math import isfinite, sqrt

def normalized_operators(adjacency):
    n = len(adjacency)
    degrees = [sum(row) for row in adjacency]
    inverse = [1 / degree if degree else 0 for degree in degrees]
    if not all(isfinite(value) for value in inverse):
        raise ValueError("Positive degrees need finite reciprocals; rescale the weights")
    inverse_root = [1 / sqrt(degree) if degree else 0 for degree in degrees]
    laplacian = [[int(i == j) * degrees[i] - adjacency[i][j] for j in range(n)] for i in range(n)]
    symmetric = [[inverse_root[i] * laplacian[i][j] * inverse_root[j] for j in range(n)] for i in range(n)]
    naive = [[int(i == j) - inverse_root[i] * adjacency[i][j] * inverse_root[j] for j in range(n)] for i in range(n)]
    transition = [[inverse[i] * adjacency[i][j] + int(i == j and degrees[i] == 0) for j in range(n)] for i in range(n)]
    return degrees, laplacian, symmetric, naive, transition

adjacency = [[0, 2, 0, 0], [2, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]]
degrees, laplacian, symmetric, naive, transition = normalized_operators(adjacency)
print("degrees:", degrees)
print("normalized isolated row:", symmetric[3])
print("naive isolated row:", naive[3])
print("stochastic isolated row:", transition[3])
ones = [sum(row) for row in symmetric]
root_degree = [sum(row[j] * sqrt(degrees[j]) for j in range(4)) for row in symmetric]
print("S times ones:", [round(value, 6) for value in ones])
print("S times sqrt(degree):", [round(value, 6) for value in root_degree])
adjacency[1][1] = 1
with_loop = normalized_operators(adjacency)
print("loop leaves L unchanged:", with_loop[1] == laplacian)
print("B row of P with loop:", with_loop[4][1])
''', 'Roundoff in floating-point square roots is displayed to six decimals. The self-loop contributes once to row-sum degree. Its contribution cancels in L but changes normalization and the walk.')

add('averaging', 'Compare two different meanings of averaging',
    'Starting from [6,0,0] on a three-vertex path, why does conservative exchange approach 2 but lazy neighbor averaging approach 1.5?', r'''
def multiply_vector(matrix, values):
    return [sum(a * b for a, b in zip(row, values)) for row in matrix]

def iterate(matrix, values, steps):
    for _ in range(steps):
        values = multiply_vector(matrix, values)
    return values

exchange = [[0.75, 0.25, 0], [0.25, 0.5, 0.25], [0, 0.25, 0.75]]
neighbor = [[0, 1, 0], [0.5, 0, 0.5], [0, 1, 0]]
lazy = [[(int(i == j) + neighbor[i][j]) / 2 for j in range(3)] for i in range(3)]
initial = [6, 0, 0]
for name, operator in [("exchange", exchange), ("neighbor", neighbor), ("lazy", lazy)]:
    print(name, "step 1:", iterate(operator, initial, 1))
    print(name, "step 2:", iterate(operator, initial, 2))
    print(name, "step 80:", [round(value, 6) for value in iterate(operator, initial, 80)])
print("initial ordinary mean:", sum(initial) / 3)
print("initial degree-weighted mean:", sum(degree * value for degree, value in zip([1, 2, 1], initial)) / 4)
''', 'These are computed linear iterations, not timed physical experiments. The neighbor rule oscillates on this bipartite path. Keeping half of the previous value removes that oscillation; it does not change which weighted mean is preserved.')

add('interpolation', 'Solve the unknown values and reject missing anchors',
    'With A fixed at 6 and C at 0, find B on the weighted path. What makes D—E undetermined, and what repairs it?', r'''
from fractions import Fraction

def harmonic_values(n, edges, anchors):
    neighbors = [[] for _ in range(n)]
    laplacian = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    for u, v, weight in edges:
        if weight < 0:
            raise ValueError("Weights must be nonnegative")
        if weight == 0 or u == v:
            continue
        neighbors[u].append(v)
        neighbors[v].append(u)
        laplacian[u][u] += weight
        laplacian[v][v] += weight
        laplacian[u][v] -= weight
        laplacian[v][u] -= weight
    seen = set()
    for start in range(n):
        if start in seen:
            continue
        component = [start]
        seen.add(start)
        for node in component:
            for neighbor in neighbors[node]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    component.append(neighbor)
        if not any(node in anchors for node in component):
            raise ValueError("Each component needs an anchor")
    unknown = [i for i in range(n) if i not in anchors]
    rows = [[laplacian[i][j] for j in unknown] +
            [-sum(laplacian[i][j] * Fraction(value) for j, value in anchors.items())]
            for i in unknown]
    for column in range(len(unknown)):
        pivot = next(i for i in range(column, len(rows)) if rows[i][column])
        rows[column], rows[pivot] = rows[pivot], rows[column]
        divisor = rows[column][column]
        rows[column] = [value / divisor for value in rows[column]]
        for i in range(len(rows)):
            if i != column:
                divisor = rows[i][column]
                rows[i] = [a - divisor * b for a, b in zip(rows[i], rows[column])]
    values = {i: Fraction(value) for i, value in anchors.items()}
    values.update({i: rows[k][-1] for k, i in enumerate(unknown)})
    return [values[i] for i in range(n)]

edges = [(0, 1, 2), (1, 2, 1), (3, 4, 2)]
try:
    harmonic_values(5, edges, {0: 6, 2: 0})
except ValueError as error:
    print(error)
print("add D anchor:", [str(x) for x in harmonic_values(5, edges, {0: 6, 2: 0, 3: 2})])
print("add C-D bridge:", [str(x) for x in harmonic_values(5, edges + [(2, 3, 1)], {0: 6, 2: 0})])
print("changed conductances:", [str(x) for x in harmonic_values(3, [(0, 1, 1), (1, 2, 3)], {0: 8, 2: 0})])
''', 'The solution minimizes a chosen smoothness energy subject to anchors. The arithmetic is exact for these rational weights. It is a model of the missing values, not an empirical guarantee that unseen measurements follow it.')

add('features', 'Count triangles without mistaking degree for clustering',
    'A triangle has one extra leaf attached to its first vertex, plus an isolate. Which vertices have locally connected neighbors?', r'''
def triangle_features(n, edges):
    adjacency = [[0] * n for _ in range(n)]
    for u, v in edges:
        if u == v:
            raise ValueError("This formula requires a simple loopless graph")
        adjacency[u][v] = adjacency[v][u] = 1
    squared = [[sum(adjacency[i][k] * adjacency[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
    cube_diagonal = [sum(squared[i][k] * adjacency[k][i] for k in range(n)) for i in range(n)]
    degree = [sum(row) for row in adjacency]
    local_triangles = [value // 2 for value in cube_diagonal]
    clustering = [2 * count / (d * (d - 1)) if d >= 2 else 0.0 for count, d in zip(local_triangles, degree)]
    return degree, local_triangles, clustering, sum(cube_diagonal) // 6

degree, local, clustering, total = triangle_features(5, [(0, 1), (1, 2), (2, 0), (0, 3)])
print("degree:", degree)
print("triangles incident to vertex:", local)
print("local clustering:", [round(value, 6) for value in clustering])
print("total triangles:", total)
''', 'Every triangle appears as six closed length-three walks: three starting vertices and two directions. Low-degree local clustering is defined as zero here; some summaries exclude those vertices, so state the convention.')

add('aggregation', 'Run one shaped graph-convolution calculation',
    'How does adding self-loops and recomputing degree change one three-node, two-feature aggregation?', r'''
from math import sqrt

def multiply(left, right):
    return [[sum(left[i][k] * right[k][j] for k in range(len(right)))
             for j in range(len(right[0]))] for i in range(len(left))]

adjacency = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
with_loops = [[adjacency[i][j] + int(i == j) for j in range(3)] for i in range(3)]
degrees = [sum(row) for row in with_loops]
normalized = [[with_loops[i][j] / sqrt(degrees[i] * degrees[j]) for j in range(3)] for i in range(3)]
features = [[1, 0], [0, 2], [1, 1]]  # 3 nodes by 2 input features
weights = [[1, -1], [0.5, 1]]         # 2 input features by 2 output features
aggregated = multiply(normalized, features)
linear = multiply(aggregated, weights)
output = [[max(0, value) for value in row] for row in linear]
print("degrees with loops:", degrees)
print("aggregated features:", [[round(x, 6) for x in row] for row in aggregated])
print("output after ReLU:", [[round(x, 6) for x in row] for row in output])
print("normalized row sums:", [round(sum(row), 6) for row in normalized])
''', 'Weights are fixed invented coefficients, not learned parameters or a performance claim. Symmetric normalization is not generally a row-stochastic average. A full trained GCN adds an objective, data splits and parameter updates.')

add('availability', 'Construct only the relationships available for a prediction',
    'At time 5, which recorded links can support a prediction? If a future A—C link is the target, can either stored direction be retained?', r'''
def visible_links(records, cutoff, held_out_pair=None):
    target = frozenset(held_out_pair) if held_out_pair is not None else None
    return [(source, destination) for time, source, destination in records
            if time <= cutoff and (target is None or frozenset((source, destination)) != target)]

records = [(2, "A", "B"), (3, "B", "C"), (6, "A", "C"), (6, "C", "A"), (4, "D", "E")]
print("available at time 5:", visible_links(records, 5))
print("held-out A-C, both stored directions removed:", visible_links(records, 10, ("A", "C")))
print("all later records:", visible_links(records, 10))
''', 'The unordered held-out pair assumes an undirected prediction target. Directed edge prediction needs an explicitly directed target policy. Merely including a test vertex is not necessarily leakage in a declared transductive task; using unavailable future evidence is.')


def main():
    output_dir = ROOT / 'scratch/graph-fundamentals-native/programs'
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, example in PROGRAMS.items():
        script = output_dir / f'{key}.py'
        script.write_text(example['code'] + '\n', encoding='utf-8')
        run = subprocess.run([sys.executable, '-X', 'utf8', '-I', str(script)], capture_output=True, text=True, encoding='utf-8', check=True)
        example['expected'] = run.stdout.rstrip()
        if run.stderr:
            raise RuntimeError(run.stderr)
    assert PROGRAMS['original']['code'] == ARCHIVE['program'].strip()
    assert PROGRAMS['original']['expected'] == ARCHIVE['output'].strip()
    destination = ROOT / 'src/learn/data/graph-fundamentals-examples.js'
    destination.write_text('// Complete Python programs, executed by prepare-graph-fundamentals-examples.py.\nexport const graphFundamentalsExamples = ' + json.dumps(PROGRAMS, ensure_ascii=False, indent=2) + ';\n', encoding='utf-8')
    record = {'programCount': len(PROGRAMS), 'originalCodeAndOutputPreserved': True, 'sourceSha256': hashlib.sha256(destination.read_bytes()).hexdigest(), 'outputs': {key: example['expected'] for key, example in PROGRAMS.items()}}
    (output_dir.parent / 'generated-results.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()

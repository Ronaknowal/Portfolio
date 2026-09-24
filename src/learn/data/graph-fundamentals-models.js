/** Bounded, deterministic teaching models. Weights are coupling strengths. */
export const graphFoundationLabels = ['A', 'B', 'C', 'D', 'E'];
export function foundationEdges(bridge = 0, extraEdge = false) {
  return [[0, 1, 2], [1, 2, 1], [2, 3, bridge], [3, 4, 2], ...(extraEdge ? [[0, 2, 1]] : [])];
}
function finiteNumber(value, name, limit = 1e6) {
  if (typeof value !== 'number' || !Number.isFinite(value) || Math.abs(value) > limit) {
    throw new RangeError(`${name} must be a finite number with magnitude at most ${limit}.`);
  }
  return value;
}
function validateVertexCount(vertexCount) {
  if (!Number.isInteger(vertexCount) || vertexCount < 0 || vertexCount > 8) {
    throw new RangeError('The teaching graph supports zero to eight declared vertices.');
  }
}
function validateVertex(vertex, vertexCount) {
  if (!Number.isInteger(vertex) || vertex < 0 || vertex >= vertexCount) {
    throw new RangeError('Every edge endpoint must be a declared vertex.');
  }
}
function squareZeros(size) {
  return Array.from({
    length: size
  }, () => Array(size).fill(0));
}
export function graphMatrixProduct(left, right) {
  if (!left.length) return [];
  const inner = left[0].length;
  if (right.length !== inner) throw new RangeError('Matrix inner dimensions must agree.');
  const columns = right[0]?.length ?? 0;
  if (left.some(row => row.length !== inner) || right.some(row => row.length !== columns)) {
    throw new RangeError('Matrices must have rectangular rows.');
  }
  return left.map(row => Array.from({
    length: columns
  }, (_, column) => row.reduce((sum, entry, index) => sum + entry * right[index][column], 0)));
}
export function graphMatrixVector(matrix, values) {
  if (matrix.some(row => row.length !== values.length)) {
    throw new RangeError('A matrix row must match the signal length.');
  }
  return matrix.map(row => row.reduce((sum, value, index) => sum + value * values[index], 0));
}
export function graphMatrices(vertexCount, edges, directed = false) {
  validateVertexCount(vertexCount);
  if (!Array.isArray(edges) || edges.length > 64 || typeof directed !== 'boolean') {
    throw new RangeError('Supply at most 64 edges and an explicit Boolean direction flag.');
  }
  const adjacency = squareZeros(vertexCount);
  const activeEdges = [];
  for (const edge of edges) {
    if (!Array.isArray(edge) || edge.length !== 3) throw new RangeError('An edge is [source, target, weight].');
    const [source, target, weight] = edge;
    validateVertex(source, vertexCount);
    validateVertex(target, vertexCount);
    finiteNumber(weight, 'Edge weight', 20);
    if (weight < 0) throw new RangeError('These coupling models require nonnegative edge weights.');
    if (weight === 0) continue;
    adjacency[source][target] += weight;
    if (!directed && source !== target) adjacency[target][source] += weight;
    activeEdges.push([...edge]);
  }
  const degrees = adjacency.map(row => row.reduce((sum, value) => sum + value, 0));
  const incoming = adjacency.map((_, column) => adjacency.reduce((sum, row) => sum + row[column], 0));
  const laplacian = adjacency.map((row, i) => row.map((value, j) => (i === j ? degrees[i] : 0) - value));
  return {
    adjacency,
    degrees,
    incoming,
    laplacian,
    activeEdges,
    directed
  };
}
export function graphConnectivity(adjacency) {
  const size = adjacency.length;
  if (adjacency.some(row => row.length !== size || row.some(value => !Number.isFinite(value) || value < 0))) {
    throw new RangeError('Connectivity needs a square nonnegative adjacency matrix.');
  }
  const reachable = adjacency.map((row, i) => row.map((value, j) => i === j || value > 0));
  for (let via = 0; via < size; via += 1) {
    for (let source = 0; source < size; source += 1) {
      for (let target = 0; target < size; target += 1) {
        reachable[source][target] ||= reachable[source][via] && reachable[via][target];
      }
    }
  }
  const grouped = related => {
    const seen = new Set();
    const components = [];
    for (let start = 0; start < size; start += 1) {
      if (seen.has(start)) continue;
      const queue = [start];
      seen.add(start);
      for (let cursor = 0; cursor < queue.length; cursor += 1) {
        for (let neighbor = 0; neighbor < size; neighbor += 1) {
          if (!seen.has(neighbor) && related(queue[cursor], neighbor)) {
            seen.add(neighbor);
            queue.push(neighbor);
          }
        }
      }
      components.push(queue.sort((a, b) => a - b));
    }
    return components;
  };
  const weakComponents = grouped((u, v) => adjacency[u][v] > 0 || adjacency[v][u] > 0);
  const strongComponents = grouped((u, v) => reachable[u][v] && reachable[v][u]);
  return {
    reachable,
    weakComponents,
    strongComponents
  };
}
export function graphWalks(adjacency, start, target, steps) {
  const size = adjacency.length;
  validateVertex(start, size);
  validateVertex(target, size);
  if (!Number.isInteger(steps) || steps < 0 || steps > 4) throw new RangeError('Use zero to four walk steps.');
  if (adjacency.some(row => row.length !== size || row.some(value => !Number.isFinite(value) || value < 0))) {
    throw new RangeError('Walks need a square nonnegative adjacency matrix.');
  }
  let power = adjacency.map((row, i) => row.map((_, j) => Number(i === j)));
  for (let index = 0; index < steps; index += 1) power = graphMatrixProduct(power, adjacency);
  const walks = [];
  function extend(path, weight) {
    if (path.length === steps + 1) {
      if (path.at(-1) === target) walks.push({
        vertices: path,
        weight
      });
      return;
    }
    const source = path.at(-1);
    for (let next = 0; next < size; next += 1) {
      if (adjacency[source][next] > 0) extend([...path, next], weight * adjacency[source][next]);
    }
  }
  extend([start], 1);
  return {
    power,
    walks,
    weightSum: walks.reduce((sum, walk) => sum + walk.weight, 0)
  };
}
export function graphEnergy(vertexCount, edges, values) {
  const graph = graphMatrices(vertexCount, edges);
  if (values.length !== vertexCount) throw new RangeError('Supply one signal value per vertex.');
  values.forEach(value => finiteNumber(value, 'Signal value'));
  const edgeTerms = graph.activeEdges.map(([source, target, weight]) => ({
    source,
    target,
    weight,
    drop: values[source] - values[target],
    flow: weight * (values[source] - values[target]),
    energy: weight * (values[source] - values[target]) ** 2
  }));
  const incidence = graph.activeEdges.map(([source, target]) => Array.from({
    length: vertexCount
  }, (_, vertex) => Number(vertex === source) - Number(vertex === target)));
  const action = graphMatrixVector(graph.laplacian, values);
  const edgeEnergy = edgeTerms.reduce((sum, edge) => sum + edge.energy, 0);
  const quadraticEnergy = values.reduce((sum, value, index) => sum + value * action[index], 0);
  const components = graphConnectivity(graph.adjacency).weakComponents;
  return {
    ...graph,
    edgeTerms,
    incidence,
    action,
    edgeEnergy,
    quadraticEnergy,
    components
  };
}
export function graphNormalizations(vertexCount, edges) {
  const graph = graphMatrices(vertexCount, edges);
  const inverse = graph.degrees.map(degree => degree === 0 ? 0 : 1 / degree);
  if (inverse.some(value => !Number.isFinite(value))) {
    throw new RangeError('Positive degrees must have finite reciprocals in the normalization model. Rescale the edge weights.');
  }
  const inverseRoot = graph.degrees.map(degree => degree === 0 ? 0 : 1 / Math.sqrt(degree));
  const positiveDegree = graph.degrees.map(degree => Number(degree > 0));
  const symmetric = graph.laplacian.map((row, i) => row.map((value, j) => inverseRoot[i] * value * inverseRoot[j]));
  const randomWalk = graph.laplacian.map((row, i) => row.map(value => inverse[i] * value));
  const transition = graph.adjacency.map((row, i) => row.map((value, j) => inverse[i] * value + Number(i === j) * (1 - positiveDegree[i])));
  const naiveIdentity = graph.adjacency.map((row, i) => row.map((value, j) => Number(i === j) - inverseRoot[i] * value * inverseRoot[j]));
  return {
    ...graph,
    inverse,
    inverseRoot,
    positiveDegree,
    symmetric,
    randomWalk,
    transition,
    naiveIdentity
  };
}
export function graphAveragingTrace(method = 'exchange', steps = 0, stepSize = 0.25) {
  if (!['exchange', 'neighbor', 'lazy'].includes(method)) throw new RangeError('Select exchange, neighbor or lazy averaging.');
  if (!Number.isInteger(steps) || steps < 0 || steps > 24) throw new RangeError('Use zero to 24 steps.');
  finiteNumber(stepSize, 'Step size', 1);
  if (stepSize < 0) throw new RangeError('The step size must be nonnegative.');
  const graph = graphNormalizations(4, [[0, 1, 1], [1, 2, 1]]);
  const update = graph.laplacian.map((row, i) => row.map((value, j) => {
    if (method === 'exchange') return Number(i === j) - stepSize * value;
    if (method === 'neighbor') return graph.transition[i][j];
    return (Number(i === j) + graph.transition[i][j]) / 2;
  }));
  const states = [[6, 0, 0, 4]];
  for (let step = 0; step < steps; step += 1) states.push(graphMatrixVector(update, states.at(-1)));
  const summaries = states.map(values => ({
    ordinaryMean: (values[0] + values[1] + values[2]) / 3,
    weightedMean: (values[0] + 2 * values[1] + values[2]) / 4,
    energy: (values[0] - values[1]) ** 2 + (values[1] - values[2]) ** 2
  }));
  return {
    ...graph,
    method,
    stepSize,
    update,
    states,
    summaries,
    convexCombination: update.every(row => row.every(entry => entry >= 0))
  };
}
function solveDense(system, rightSide) {
  const size = system.length;
  const augmented = system.map((row, i) => [...row, rightSide[i]]);
  for (let column = 0; column < size; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < size; row += 1) {
      if (Math.abs(augmented[row][column]) > Math.abs(augmented[pivot][column])) pivot = row;
    }
    if (augmented[pivot][column] === 0) throw new RangeError('The reduced system is singular.');
    [augmented[column], augmented[pivot]] = [augmented[pivot], augmented[column]];
    const divisor = augmented[column][column];
    for (let entry = column; entry <= size; entry += 1) augmented[column][entry] /= divisor;
    for (let row = 0; row < size; row += 1) {
      if (row === column) continue;
      const multiplier = augmented[row][column];
      for (let entry = column; entry <= size; entry += 1) augmented[row][entry] -= multiplier * augmented[column][entry];
    }
  }
  return augmented.map(row => row[size]);
}
export function graphHarmonicInterpolation(vertexCount, edges, anchors) {
  const graph = graphMatrices(vertexCount, edges);
  if (!anchors || typeof anchors !== 'object' || Array.isArray(anchors)) throw new RangeError('Anchors map vertex indices to fixed values.');
  const fixed = new Map();
  for (const [key, value] of Object.entries(anchors)) {
    const vertex = Number(key);
    validateVertex(vertex, vertexCount);
    if (String(vertex) !== key) throw new RangeError('Use canonical integer vertex keys.');
    fixed.set(vertex, finiteNumber(value, 'Anchor value'));
  }
  const components = graphConnectivity(graph.adjacency).weakComponents;
  const unanchored = components.filter(component => !component.some(vertex => fixed.has(vertex)));
  const unsupported = new Set(unanchored.flat());
  const unknown = Array.from({
    length: vertexCount
  }, (_, i) => i).filter(vertex => !fixed.has(vertex) && !unsupported.has(vertex));
  const reduced = unknown.map(row => unknown.map(column => graph.laplacian[row][column]));
  const rightSide = unknown.map(row => -[...fixed].reduce((sum, [column, value]) => sum + graph.laplacian[row][column] * value, 0));
  const solved = solveDense(reduced, rightSide);
  const values = Array.from({
    length: vertexCount
  }, (_, vertex) => fixed.get(vertex) ?? null);
  unknown.forEach((vertex, index) => {
    values[vertex] = solved[index];
  });
  const residuals = values.map((value, row) => value === null ? null : graph.laplacian[row].reduce((sum, coefficient, column) => sum + coefficient * (values[column] ?? 0), 0));
  return {
    ...graph,
    components,
    unanchored,
    unknown,
    reduced,
    rightSide,
    values,
    residuals,
    unique: unanchored.length === 0
  };
}
export function formatGraphNumber(value, digits = 4) {
  if (value === null) return 'undetermined';
  if (value === 0 || Object.is(value, -0)) return '0';
  if (!Number.isFinite(value)) return String(value);
  if (Math.abs(value) < 10 ** -digits || Math.abs(value) >= 1e6) return value.toExponential(2);
  return Number(value.toFixed(digits)).toString();
}

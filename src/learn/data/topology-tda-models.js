// Finite F2 teaching models. The caller chooses the represented space;
// none of these computations certifies the topology of an unknown population.
function finiteNumber(value, minimum, maximum, name) {
  if (typeof value !== "number" || !Number.isFinite(value) || value < minimum || value > maximum) {
    throw new RangeError(name + " is outside this investigation's supported range.");
  }
  return value;
}
function boundedInteger(value, minimum, maximum, name) {
  finiteNumber(value, minimum, maximum, name);
  if (!Number.isInteger(value)) throw new RangeError(name + " must be an integer.");
  return value;
}
function toggleMembers(target, source) {
  for (const value of source) {
    if (target.has(value)) target.delete(value);else target.add(value);
  }
}
function greatestMember(values) {
  let greatest = -1;
  for (const value of values) greatest = Math.max(greatest, value);
  return greatest;
}
function sortedMembers(values) {
  return [...values].sort((left, right) => left - right);
}
export function simplexKey(vertices) {
  return vertices.join("-");
}
function vertexOrder(left, right) {
  for (let index = 0; index < Math.min(left.length, right.length); index += 1) {
    if (left[index] !== right[index]) return left[index] - right[index];
  }
  return left.length - right.length;
}
export function simplexFaces(vertices) {
  if (vertices.length <= 1) return [];
  return vertices.map((_, excluded) => vertices.filter((__, index) => index !== excluded));
}
function combinations(values, size) {
  const result = [];
  function visit(start, selected) {
    if (selected.length === size) {
      result.push([...selected]);
      return;
    }
    for (let index = start; index <= values.length - (size - selected.length); index += 1) {
      selected.push(values[index]);
      visit(index + 1, selected);
      selected.pop();
    }
  }
  visit(0, []);
  return result;
}
export function normalizeFiltration(simplices) {
  if (!Array.isArray(simplices) || simplices.length > 1600) {
    throw new RangeError("Use at most 1,600 simplices in this finite model.");
  }
  const seen = new Map();
  for (const simplex of simplices) {
    if (!Array.isArray(simplex.vertices) || simplex.vertices.length < 1 || simplex.vertices.length > 4) {
      throw new RangeError("Each simplex needs one to four vertices.");
    }
    const vertices = simplex.vertices.map(vertex => boundedInteger(vertex, 0, 63, "Vertex ID")).sort((left, right) => left - right);
    if (new Set(vertices).size !== vertices.length) throw new RangeError("A simplex cannot repeat a vertex.");
    const key = simplexKey(vertices);
    if (seen.has(key)) throw new RangeError("Duplicate simplex: " + key);
    const birth = finiteNumber(simplex.birth, 0, 1000, "Filtration value");
    seen.set(key, {
      key,
      vertices,
      dimension: vertices.length - 1,
      birth
    });
  }
  for (const simplex of seen.values()) {
    for (const face of simplexFaces(simplex.vertices)) {
      const parent = seen.get(simplexKey(face));
      if (!parent || parent.birth > simplex.birth) {
        throw new RangeError("Every face must enter no later than its coface: " + simplex.key);
      }
    }
  }
  return [...seen.values()].sort((left, right) => left.birth - right.birth || left.dimension - right.dimension || vertexOrder(left.vertices, right.vertices));
}
export function complexFromFacets(facets) {
  if (!Array.isArray(facets) || facets.length > 100) throw new RangeError("Use at most 100 facets.");
  const simplices = new Map();
  for (const facet of facets) {
    if (!Array.isArray(facet) || facet.length < 1 || facet.length > 4) {
      throw new RangeError("Facet dimension must be between zero and three.");
    }
    const vertices = [...facet].sort((left, right) => left - right);
    if (new Set(vertices).size !== vertices.length) throw new RangeError("Repeated facet vertex.");
    for (let size = 1; size <= vertices.length; size += 1) {
      for (const subset of combinations(vertices, size)) {
        simplices.set(simplexKey(subset), {
          vertices: subset,
          birth: 0
        });
      }
    }
  }
  return normalizeFiltration([...simplices.values()]);
}
export function rankOverF2(columns) {
  const pivotColumns = new Map();
  for (const original of columns) {
    const column = new Set(original);
    while (column.size) {
      const pivot = greatestMember(column);
      if (!pivotColumns.has(pivot)) {
        pivotColumns.set(pivot, column);
        break;
      }
      toggleMembers(column, pivotColumns.get(pivot));
    }
  }
  return pivotColumns.size;
}
function boundaryColumns(ordered) {
  const indexByKey = new Map(ordered.map((simplex, index) => [simplex.key, index]));
  return ordered.map(simplex => simplexFaces(simplex.vertices).map(face => indexByKey.get(simplexKey(face))));
}
export function bettiAt(simplices, threshold = 1000) {
  finiteNumber(threshold, 0, 1000, "Threshold");
  const ordered = normalizeFiltration(simplices).filter(simplex => simplex.birth <= threshold);
  const columns = boundaryColumns(ordered);
  const counts = [0, 0, 0, 0];
  const ranks = [0, 0, 0, 0, 0];
  for (let dimension = 0; dimension <= 3; dimension += 1) {
    const indices = ordered.flatMap((simplex, index) => simplex.dimension === dimension ? [index] : []);
    counts[dimension] = indices.length;
    ranks[dimension] = rankOverF2(indices.map(index => columns[index]));
  }
  return {
    counts,
    ranks,
    betti: counts.map((count, dimension) => count - ranks[dimension] - ranks[dimension + 1]),
    euler: counts.reduce((sum, count, dimension) => sum + (-1) ** dimension * count, 0)
  };
}
export function chainBoundary(simplices, selectedKeys) {
  const ordered = normalizeFiltration(simplices);
  const byKey = new Map(ordered.map(simplex => [simplex.key, simplex]));
  if (!Array.isArray(selectedKeys) || new Set(selectedKeys).size !== selectedKeys.length) {
    throw new RangeError("Select each chain simplex at most once.");
  }
  const boundary = new Set();
  let dimension = null;
  for (const key of selectedKeys) {
    const simplex = byKey.get(key);
    if (!simplex) throw new RangeError("A selected simplex is absent from the complex.");
    if (dimension !== null && dimension !== simplex.dimension) throw new RangeError("A chain has one dimension.");
    dimension = simplex.dimension;
    toggleMembers(boundary, simplexFaces(simplex.vertices).map(simplexKey));
  }
  const indexByKey = new Map(ordered.map((simplex, index) => [simplex.key, index]));
  const higherColumns = ordered.filter(simplex => simplex.dimension === dimension + 1).map(simplex => simplexFaces(simplex.vertices).map(face => indexByKey.get(simplexKey(face))));
  const selectedColumn = selectedKeys.map(key => indexByKey.get(key));
  const isCycle = boundary.size === 0;
  const isBoundary = !selectedKeys.length || isCycle && rankOverF2(higherColumns) === rankOverF2([...higherColumns, selectedColumn]);
  return {
    dimension,
    boundary: [...boundary].sort(),
    isCycle,
    isBoundary
  };
}
export function persistentHomology(simplices, {
  captureTrace = false
} = {}) {
  const ordered = normalizeFiltration(simplices);
  if (captureTrace && ordered.length > 64) throw new RangeError("A detailed trace supports at most 64 simplices.");
  const boundaries = boundaryColumns(ordered);
  const reduced = [];
  const combinationsByColumn = [];
  const columnForPivot = new Map();
  const creators = new Set();
  const pairedCreators = new Set();
  const intervals = [];
  const trace = [];
  for (let columnIndex = 0; columnIndex < ordered.length; columnIndex += 1) {
    const column = new Set(boundaries[columnIndex]);
    const combination = new Set([columnIndex]);
    const stages = [];
    if (captureTrace) stages.push({
      boundary: sortedMembers(column),
      combination: sortedMembers(combination),
      addedColumn: null
    });
    while (column.size && columnForPivot.has(greatestMember(column))) {
      const earlier = columnForPivot.get(greatestMember(column));
      toggleMembers(column, reduced[earlier]);
      toggleMembers(combination, combinationsByColumn[earlier]);
      if (captureTrace) stages.push({
        boundary: sortedMembers(column),
        combination: sortedMembers(combination),
        addedColumn: earlier
      });
    }
    reduced.push(column);
    combinationsByColumn.push(combination);
    const pivot = greatestMember(column);
    if (pivot === -1) creators.add(columnIndex);else {
      columnForPivot.set(pivot, columnIndex);
      pairedCreators.add(pivot);
      intervals.push({
        dimension: ordered[pivot].dimension,
        birth: ordered[pivot].birth,
        death: ordered[columnIndex].birth,
        creator: pivot,
        destroyer: columnIndex,
        representative: sortedMembers(column)
      });
    }
    if (captureTrace) trace.push({
      column: columnIndex,
      stages,
      pivot,
      event: pivot === -1 ? "birth" : "pair"
    });
  }
  for (const creator of creators) {
    if (!pairedCreators.has(creator)) {
      intervals.push({
        dimension: ordered[creator].dimension,
        birth: ordered[creator].birth,
        death: Infinity,
        creator,
        destroyer: null,
        representative: sortedMembers(combinationsByColumn[creator])
      });
    }
  }
  intervals.sort((left, right) => left.dimension - right.dimension || left.birth - right.birth || left.death - right.death || left.creator - right.creator);
  return {
    ordered,
    boundaries,
    intervals,
    trace
  };
}
export function ripsFiltration(points, {
  maximumDimension = 2
} = {}) {
  boundedInteger(maximumDimension, 0, 3, "Maximum simplex dimension");
  if (!Array.isArray(points) || points.length > 14) throw new RangeError("Use at most 14 points.");
  for (const point of points) {
    if (!Array.isArray(point) || point.length !== 2) throw new RangeError("Each point needs two coordinates.");
    point.forEach(value => finiteNumber(value, -100, 100, "Coordinate"));
  }
  const simplices = [];
  const vertices = points.map((_, index) => index);
  for (let size = 1; size <= maximumDimension + 1; size += 1) {
    for (const subset of combinations(vertices, size)) {
      let birth = 0;
      for (let first = 0; first < subset.length; first += 1) {
        for (let second = first + 1; second < subset.length; second += 1) {
          const left = points[subset[first]];
          const right = points[subset[second]];
          birth = Math.max(birth, Math.hypot(left[0] - right[0], left[1] - right[1]));
        }
      }
      simplices.push({
        vertices: subset,
        birth
      });
    }
  }
  return normalizeFiltration(simplices);
}
export const TOPOLOGY_POINT_FIXTURES = {
  square: {
    label: "Four square corners",
    points: [[-1, -1], [1, -1], [1, 1], [-1, 1]]
  },
  rectangle: {
    label: "Rectangle: width 3, height 1",
    points: [[-1.5, -0.5], [1.5, -0.5], [1.5, 0.5], [-1.5, 0.5]]
  },
  ring: {
    label: "Eight samples on a circle",
    points: [[1, 0], [Math.SQRT1_2, Math.SQRT1_2], [0, 1], [-Math.SQRT1_2, Math.SQRT1_2], [-1, 0], [-Math.SQRT1_2, -Math.SQRT1_2], [0, -1], [Math.SQRT1_2, -Math.SQRT1_2]]
  },
  grid: {
    label: "Nine samples across a square",
    points: [-1, 0, 1].flatMap(x => [-1, 0, 1].map(y => [x, y]))
  }
};
export const TOPOLOGY_COMPLEX_FIXTURES = {
  triangle: {
    label: "Triangle edges",
    points: [[-1, -0.7], [1, -0.7], [0, 1]],
    facets: [[0, 1], [1, 2], [0, 2]],
    chainDimension: 1
  },
  filledTriangle: {
    label: "Filled triangle",
    points: [[-1, -0.7], [1, -0.7], [0, 1]],
    facets: [[0, 1, 2]],
    chainDimension: 1
  },
  square: {
    label: "Square with a diagonal",
    points: [[-1, -1], [1, -1], [1, 1], [-1, 1]],
    facets: [[0, 1], [1, 2], [2, 3], [0, 3], [0, 2]],
    chainDimension: 1
  },
  shell: {
    label: "Tetrahedron shell",
    points: [[-1, -0.8], [1, -0.8], [0, 1], [0, -0.15]],
    facets: [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]],
    chainDimension: 2
  },
  solid: {
    label: "Filled tetrahedron",
    points: [[-1, -0.8], [1, -0.8], [0, 1], [0, -0.15]],
    facets: [[0, 1, 2, 3]],
    chainDimension: 2
  }
};
function finiteDiagram(diagram) {
  if (!Array.isArray(diagram) || diagram.length > 8) throw new RangeError("Use at most eight finite diagram points.");
  return diagram.map(point => {
    if (!Array.isArray(point) || point.length !== 2) throw new RangeError("A diagram point needs birth and death.");
    const birth = finiteNumber(point[0], -20, 20, "Birth");
    const death = finiteNumber(point[1], -20, 20, "Death");
    if (death <= birth) throw new RangeError("Finite diagram points must have death greater than birth.");
    return [birth, death];
  });
}
export function evaluateDiagramMatching(firstDiagram, secondDiagram, assignments, power = Infinity) {
  const first = finiteDiagram(firstDiagram);
  const second = finiteDiagram(secondDiagram);
  if (power !== Infinity) finiteNumber(power, 1, 8, "Wasserstein power");
  if (!Array.isArray(assignments) || assignments.length !== first.length) throw new RangeError("Assign every first-diagram point.");
  const used = new Set();
  const costs = [];
  assignments.forEach((target, index) => {
    if (target === null) costs.push((first[index][1] - first[index][0]) / 2);else {
      boundedInteger(target, 0, second.length - 1, "Match index");
      if (used.has(target)) throw new RangeError("A diagram point cannot be matched twice.");
      used.add(target);
      costs.push(Math.max(Math.abs(first[index][0] - second[target][0]), Math.abs(first[index][1] - second[target][1])));
    }
  });
  second.forEach((point, index) => {
    if (!used.has(index)) costs.push((point[1] - point[0]) / 2);
  });
  return {
    costs,
    value: power === Infinity ? Math.max(0, ...costs) : costs.reduce((sum, cost) => sum + cost ** power, 0) ** (1 / power),
    unmatchedSecond: second.flatMap((_, index) => used.has(index) ? [] : [index])
  };
}
export function optimalDiagramMatching(firstDiagram, secondDiagram, power = Infinity) {
  const first = finiteDiagram(firstDiagram);
  const second = finiteDiagram(secondDiagram);
  if (first.length > 5 || second.length > 5) throw new RangeError("Exhaustive matching supports at most five points per diagram.");
  let best = null;
  let evaluated = 0;
  function visit(assignments, used) {
    if (assignments.length === first.length) {
      const result = evaluateDiagramMatching(first, second, assignments, power);
      evaluated += 1;
      if (!best || result.value < best.value) best = {
        ...result,
        assignments: [...assignments]
      };
      return;
    }
    visit([...assignments, null], used);
    for (let target = 0; target < second.length; target += 1) {
      if (!used.has(target)) visit([...assignments, target], new Set([...used, target]));
    }
  }
  visit([], new Set());
  return {
    ...best,
    evaluated
  };
}
export function pixelComplex(values, threshold) {
  if (!Array.isArray(values) || values.length !== 9) throw new RangeError("Provide a 3 by 3 image.");
  values.forEach(value => finiteNumber(value, 0, 9, "Pixel value"));
  finiteNumber(threshold, 0, 9, "Pixel threshold");
  const vertices = new Set();
  const edges = new Map();
  const activePixels = [];
  const vertexKey = (x, y) => x + "," + y;
  values.forEach((value, pixel) => {
    if (value > threshold) return;
    activePixels.push(pixel);
    const x = pixel % 3;
    const y = Math.floor(pixel / 3);
    const corners = [[x, y], [x + 1, y], [x + 1, y + 1], [x, y + 1]].map(([first, second]) => vertexKey(first, second));
    corners.forEach(corner => vertices.add(corner));
    for (let side = 0; side < 4; side += 1) {
      const endpoints = [corners[side], corners[(side + 1) % 4]].sort();
      edges.set(endpoints.join("|"), endpoints);
    }
  });
  const adjacency = new Map([...vertices].map(vertex => [vertex, []]));
  for (const [left, right] of edges.values()) {
    adjacency.get(left).push(right);
    adjacency.get(right).push(left);
  }
  const unseen = new Set(vertices);
  let components = 0;
  while (unseen.size) {
    components += 1;
    const stack = [unseen.values().next().value];
    unseen.delete(stack[0]);
    while (stack.length) {
      for (const neighbor of adjacency.get(stack.pop())) {
        if (unseen.delete(neighbor)) stack.push(neighbor);
      }
    }
  }
  const counts = [vertices.size, edges.size, activePixels.length];
  const euler = counts[0] - counts[1] + counts[2];
  return {
    vertices: [...vertices],
    edges: [...edges.values()],
    activePixels,
    counts,
    euler,
    betti: [components, components - euler]
  };
}
export function landscapeAt(diagram, time, level = 1) {
  const points = finiteDiagram(diagram);
  finiteNumber(time, -30, 30, "Landscape time");
  boundedInteger(level, 1, 9, "Landscape level");
  const tents = points.map(([birth, death]) => Math.max(0, Math.min(time - birth, death - time)));
  return {
    tents,
    value: [...tents].sort((left, right) => right - left)[level - 1] ?? 0
  };
}
function positiveNormalTail(value) {
  if (value === 0) return 0.5;
  if (value > 39) return 0;
  // Q(z) = 1/2 Q_gamma(1/2, z^2/2). Series near zero;
  // continued fraction in the tail avoids subtracting two values near one.
  const argument = value * value / 2;
  const logFactor = -argument + 0.5 * Math.log(argument) - Math.log(Math.PI) / 2;
  if (argument < 1.5) {
    let term = 2;
    let sum = term;
    for (let index = 1; index < 200; index += 1) {
      term *= argument / (index + 0.5);
      sum += term;
      if (Math.abs(term) < Math.abs(sum) * 2e-16) break;
    }
    return 0.5 * (1 - Math.exp(logFactor) * sum);
  }
  let denominator = argument + 0.5;
  let numeratorState = 1e300;
  let inverseState = 1 / denominator;
  let fraction = inverseState;
  for (let index = 1; index < 400; index += 1) {
    const numerator = -index * (index - 0.5);
    denominator += 2;
    inverseState = numerator * inverseState + denominator;
    if (Math.abs(inverseState) < 1e-300) inverseState = 1e-300;
    numeratorState = denominator + numerator / numeratorState;
    if (Math.abs(numeratorState) < 1e-300) numeratorState = 1e-300;
    inverseState = 1 / inverseState;
    const change = inverseState * numeratorState;
    fraction *= change;
    if (Math.abs(change - 1) < 4e-16) break;
  }
  return 0.5 * Math.exp(logFactor) * fraction;
}
export function normalInterval(lower, upper) {
  finiteNumber(lower, -1000, 1000, "Lower standardized endpoint");
  finiteNumber(upper, -1000, 1000, "Upper standardized endpoint");
  if (upper < lower) throw new RangeError("Interval endpoints are reversed.");
  if (upper === lower) return 0;
  const width = upper - lower;
  const midpoint = lower + width / 2;
  if (width * (1 + Math.abs(midpoint)) <= 0.01) {
    // Integrate the local Gaussian expansion: subtracting almost equal CDFs
    // would lose narrow, representable masses. The next even term is O(w^6).
    const midpointSquared = midpoint * midpoint;
    const widthSquared = width * width;
    const correction = 1 + (midpointSquared - 1) * widthSquared / 24 + (midpointSquared * midpointSquared - 6 * midpointSquared + 3) * widthSquared * widthSquared / 1920;
    return Math.exp(Math.log(width) - midpointSquared / 2 - Math.log(2 * Math.PI) / 2) * correction;
  }
  if (lower >= 0) return positiveNormalTail(lower) - positiveNormalTail(upper);
  if (upper <= 0) return positiveNormalTail(-upper) - positiveNormalTail(-lower);
  return 1 - positiveNormalTail(-lower) - positiveNormalTail(upper);
}
export function persistenceImage(diagram, {
  bandwidth = 0.5,
  xEdges = [0, 1, 2, 3, 4],
  yEdges = [0, 1, 2, 3, 4]
} = {}) {
  const points = finiteDiagram(diagram);
  finiteNumber(bandwidth, 0.1, 2, "Gaussian bandwidth");
  for (const edges of [xEdges, yEdges]) {
    if (!Array.isArray(edges) || edges.length < 2 || edges.length > 13) throw new RangeError("Use between one and twelve pixels per axis.");
    edges.forEach((edge, index) => {
      finiteNumber(edge, -20, 20, "Pixel edge");
      if (index && edge <= edges[index - 1]) throw new RangeError("Pixel edges must increase strictly.");
    });
  }
  const pixels = [];
  for (let row = 0; row < yEdges.length - 1; row += 1) {
    for (let column = 0; column < xEdges.length - 1; column += 1) {
      const contributions = points.map(([birth, death]) => {
        const persistence = death - birth;
        const weight = Math.min(persistence, 1);
        return weight * normalInterval((xEdges[column] - birth) / bandwidth, (xEdges[column + 1] - birth) / bandwidth) * normalInterval((yEdges[row] - persistence) / bandwidth, (yEdges[row + 1] - persistence) / bandwidth);
      });
      pixels.push({
        row,
        column,
        contributions,
        value: contributions.reduce((sum, value) => sum + value, 0)
      });
    }
  }
  const totalWeight = points.reduce((sum, [birth, death]) => sum + Math.min(death - birth, 1), 0);
  const capturedWeight = pixels.reduce((sum, pixel) => sum + pixel.value, 0);
  return {
    pixels,
    totalWeight,
    capturedWeight,
    omittedWeight: totalWeight - capturedWeight
  };
}
const circleCoordinate = Math.sqrt(3) / 2;
export const MAPPER_CIRCLE_POINTS = [[1, 0], [circleCoordinate, 0.5], [0.5, circleCoordinate], [0, 1], [-0.5, circleCoordinate], [-circleCoordinate, 0.5], [-1, 0], [-circleCoordinate, -0.5], [-0.5, -circleCoordinate], [0, -1], [0.5, -circleCoordinate], [circleCoordinate, -0.5]];
export function mapperGraph({
  intervalCount = 4,
  overlap = 0.4,
  clusterDistance = 0.6
} = {}) {
  boundedInteger(intervalCount, 2, 5, "Cover interval count");
  finiteNumber(overlap, 0.1, 0.65, "Cover overlap fraction");
  finiteNumber(clusterDistance, 0.1, 2.1, "Cluster edge threshold");
  const points = MAPPER_CIRCLE_POINTS;
  const width = 2 / (intervalCount - (intervalCount - 1) * overlap);
  const cover = Array.from({
    length: intervalCount
  }, (_, index) => {
    const lower = -1 + index * width * (1 - overlap);
    return {
      lower,
      upper: index === intervalCount - 1 ? 1 : lower + width
    };
  });
  const nodes = [];
  cover.forEach((interval, intervalIndex) => {
    const members = points.flatMap((point, index) => point[0] >= interval.lower && point[0] <= interval.upper ? [index] : []);
    const unseen = new Set(members);
    while (unseen.size) {
      const first = unseen.values().next().value;
      const component = [first];
      unseen.delete(first);
      for (let visited = 0; visited < component.length; visited += 1) {
        const point = points[component[visited]];
        for (const candidate of [...unseen]) {
          if (Math.hypot(point[0] - points[candidate][0], point[1] - points[candidate][1]) <= clusterDistance) {
            unseen.delete(candidate);
            component.push(candidate);
          }
        }
      }
      nodes.push({
        id: nodes.length,
        interval: intervalIndex,
        members: component.sort((left, right) => left - right)
      });
    }
  });
  const edges = [];
  for (let first = 0; first < nodes.length; first += 1) {
    for (let second = first + 1; second < nodes.length; second += 1) {
      const members = nodes[first].members.filter(member => nodes[second].members.includes(member));
      if (members.length) edges.push({
        first,
        second,
        members
      });
    }
  }
  const membership = points.map((_, point) => nodes.filter(node => node.members.includes(point)).map(node => node.id));
  return {
    points,
    cover,
    nodes,
    edges,
    membership,
    maximumMembership: Math.max(...membership.map(nodesForPoint => nodesForPoint.length))
  };
}

// Bounded, deterministic finite-graph teaching models. Eigenvectors are stored by mode.
export const spectralNames = 'ABCDEFGHIJKL';
function numberIn(value, low, high, name) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < low || value > high) {
    throw new RangeError(`${name} must be a finite number from ${low} to ${high}.`);
  }
}
export function dot(a, b) {
  return a.reduce((sum, value, i) => sum + value * b[i], 0);
}
export function multiply(matrix, vector) {
  return matrix.map(row => dot(row, vector));
}
export function formatSpectral(value, digits = 4) {
  if (value === 0) return '0';
  if (Math.abs(value) < 10 ** -digits) return value.toExponential(2);
  return Number(value.toFixed(digits)).toString();
}
export function graphFromEdges(size, edges) {
  if (!Number.isInteger(size) || size < 2 || size > 12) throw new RangeError('Use 2–12 nodes.');
  const adjacency = Array.from({
    length: size
  }, () => Array(size).fill(0));
  for (const edge of edges) {
    if (!Array.isArray(edge) || edge.length !== 3) throw new RangeError('Each edge needs two nodes and a weight.');
    const [i, j, weight] = edge;
    if (![i, j].every(v => Number.isInteger(v) && v >= 0 && v < size) || i === j) {
      throw new RangeError('Use distinct existing endpoints.');
    }
    numberIn(weight, 0, 100, 'Weight');
    if (weight > 0 && weight < 1e-6) throw new RangeError('Positive model weights must be at least 0.000001.');
    if (adjacency[i][j] !== 0) throw new RangeError('Specify each positive undirected edge once.');
    adjacency[i][j] = adjacency[j][i] = weight;
  }
  const degree = adjacency.map(row => row.reduce((sum, value) => sum + value, 0));
  const laplacian = adjacency.map((row, i) => row.map((value, j) => i === j ? degree[i] : -value));
  const normalized = laplacian.map((row, i) => row.map((value, j) => degree[i] && degree[j] ? value / Math.sqrt(degree[i] * degree[j]) : 0));
  const components = [];
  const seen = new Set();
  for (let start = 0; start < size; start++) {
    if (seen.has(start)) continue;
    const group = [start];
    seen.add(start);
    for (let cursor = 0; cursor < group.length; cursor++) {
      for (let j = 0; j < size; j++) {
        if (adjacency[group[cursor]][j] > 0 && !seen.has(j)) {
          seen.add(j);
          group.push(j);
        }
      }
    }
    components.push(group);
  }
  return {
    adjacency,
    degree,
    laplacian,
    normalized,
    components
  };
}
export function symmetricSpectrum(matrix) {
  const n = matrix.length;
  if (n < 2 || n > 12 || matrix.some(row => row.length !== n)) throw new RangeError('Use a square matrix of size 2–12.');
  if (matrix.some((row, i) => row.some((x, j) => !Number.isFinite(x) || Math.abs(x) > 1200 || Math.abs(x - matrix[j][i]) > 1e-12))) {
    throw new RangeError('The matrix must be finite, bounded and symmetric.');
  }
  const a = matrix.map(row => [...row]);
  const vectors = Array.from({
    length: n
  }, (_, i) => Array.from({
    length: n
  }, (_, j) => +(i === j)));
  const tolerance = 2e-14 * Math.max(1, ...matrix.flat().map(Math.abs));
  let converged = false;
  for (let iteration = 0; iteration < 6000; iteration++) {
    let p = 0;
    let q = 1;
    let largest = 0;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (Math.abs(a[i][j]) > largest) {
          p = i;
          q = j;
          largest = Math.abs(a[i][j]);
        }
      }
    }
    if (largest <= tolerance) {
      converged = true;
      break;
    }
    const angle = 0.5 * Math.atan2(2 * a[p][q], a[q][q] - a[p][p]);
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    const pp = a[p][p];
    const qq = a[q][q];
    const pq = a[p][q];
    for (let k = 0; k < n; k++) {
      if (k !== p && k !== q) {
        const kp = a[k][p];
        const kq = a[k][q];
        a[k][p] = a[p][k] = c * kp - s * kq;
        a[k][q] = a[q][k] = s * kp + c * kq;
      }
      const vp = vectors[k][p];
      const vq = vectors[k][q];
      vectors[k][p] = c * vp - s * vq;
      vectors[k][q] = s * vp + c * vq;
    }
    a[p][p] = c * c * pp - 2 * s * c * pq + s * s * qq;
    a[q][q] = s * s * pp + 2 * s * c * pq + c * c * qq;
    a[p][q] = a[q][p] = 0;
  }
  if (!converged) throw new Error('The bounded eigensolver did not converge.');
  const order = Array.from({
    length: n
  }, (_, i) => i).sort((i, j) => a[i][i] - a[j][j]);
  return {
    values: order.map(i => a[i][i]),
    vectors: order.map(i => {
      const column = vectors.map(row => row[i]);
      const sign = (column.find(value => Math.abs(value) > 1e-9) ?? 1) < 0 ? -1 : 1;
      return column.map(value => sign * value);
    })
  };
}
export function bridgeSpectrum(weight = 0.2, normalized = false) {
  numberIn(weight, 0, 2, 'Bridge weight');
  const edges = [[0, 1, 1], [0, 2, 1], [1, 2, 1], [3, 4, 1], [3, 5, 1], [4, 5, 1], [2, 3, weight]];
  const graph = graphFromEdges(6, edges);
  const matrix = normalized ? graph.normalized : graph.laplacian;
  return {
    ...graph,
    ...symmetricSpectrum(matrix),
    matrix
  };
}
export function graphEnergy(graph, signal) {
  if (signal.length !== graph.degree.length || signal.some(x => !Number.isFinite(x) || Math.abs(x) > 1000)) throw new RangeError('Supply one finite bounded value per node.');
  const terms = [];
  graph.adjacency.forEach((row, i) => row.forEach((weight, j) => {
    if (i < j && weight > 0) terms.push({
      i,
      j,
      weight,
      difference: signal[i] - signal[j],
      energy: weight * (signal[i] - signal[j]) ** 2
    });
  }));
  return {
    terms,
    total: terms.reduce((sum, term) => sum + term.energy, 0)
  };
}
export function cutMetrics(graph, nodes) {
  const size = graph.degree.length;
  const set = new Set(nodes);
  if (set.size !== nodes.length || !set.size || set.size === size || nodes.some(i => !Number.isInteger(i) || i < 0 || i >= size)) throw new RangeError('Choose a nonempty proper set of distinct nodes.');
  const complement = Array.from({
    length: size
  }, (_, i) => i).filter(i => !set.has(i));
  const cut = nodes.reduce((sum, i) => sum + complement.reduce((s, j) => s + graph.adjacency[i][j], 0), 0);
  const volume = nodes.reduce((sum, i) => sum + graph.degree[i], 0);
  const otherVolume = complement.reduce((sum, i) => sum + graph.degree[i], 0);
  return {
    nodes: [...nodes],
    complement,
    cut,
    volume,
    otherVolume,
    ratioCut: cut * (1 / set.size + 1 / complement.length),
    normalizedCut: volume && otherVolume ? cut * (1 / volume + 1 / otherVolume) : null,
    conductance: volume && otherVolume ? cut / Math.min(volume, otherVolume) : null
  };
}
export function cutSweep(weight = 0.2, kind = 'bridge') {
  if (!['bridge', 'unequal'].includes(kind)) throw new RangeError('Unknown graph preset.');
  numberIn(weight, 0, 2, 'Bridge weight');
  const graph = kind === 'bridge' ? bridgeSpectrum(weight) : graphFromEdges(6, [[0, 1, 3], [0, 2, 1], [1, 2, 2], [2, 3, weight], [3, 4, 1], [4, 5, 0.5]]);
  const spectrum = symmetricSpectrum(graph.normalized);
  const coordinates = spectrum.vectors[1].map((value, i) => value / Math.sqrt(graph.degree[i]));
  const order = coordinates.map((_, i) => i).sort((i, j) => coordinates[i] - coordinates[j] || i - j);
  const candidates = [];
  for (let count = 1; count < order.length; count++) {
    // Near-ties at this numerical precision remain together; the table discloses this.
    if (Math.abs(coordinates[order[count]] - coordinates[order[count - 1]]) < 1e-9) continue;
    candidates.push({
      ...cutMetrics(graph, order.slice(0, count)),
      threshold: (coordinates[order[count]] + coordinates[order[count - 1]]) / 2
    });
  }
  const best = candidates.reduce((result, candidate) => !result || candidate.conductance < result.conductance ? candidate : result, null);
  return {
    ...graph,
    spectrum,
    coordinates,
    order,
    candidates,
    best
  };
}
export function clusteringState(weight = 0.08, seedKind = 'spread') {
  numberIn(weight, 0, 1.5, 'Intergroup weight');
  if (!['spread', 'nearby'].includes(seedKind)) throw new RangeError('Unknown centroid preset.');
  const edges = [];
  for (const start of [0, 3, 6]) edges.push([start, start + 1, 1], [start, start + 2, 1], [start + 1, start + 2, 1]);
  edges.push([2, 3, weight], [5, 6, weight]);
  const graph = graphFromEdges(9, edges);
  const spectrum = symmetricSpectrum(graph.normalized);
  const raw = graph.degree.map((_, i) => spectrum.vectors.slice(0, 3).map(vector => vector[i]));
  const rowNorms = raw.map(row => Math.sqrt(dot(row, row)));
  if (rowNorms.some(norm => norm < 1e-12)) throw new Error('A selected spectral row has zero numerical norm.');
  const points = raw.map((row, i) => row.map(value => value / rowNorms[i]));
  const seeds = seedKind === 'spread' ? [0, 3, 6] : [0, 1, 3];
  let centroids = seeds.map(i => [...points[i]]);
  let previous = null;
  const frames = [{
    phase: 'Initial centroids',
    centroids,
    labels: null,
    loss: null,
    empty: [],
    note: `Start at nodes ${seeds.map(i => spectralNames[i]).join(', ')}. Distances use all three coordinates.`
  }];
  for (let iteration = 0; iteration < 12; iteration++) {
    const labels = points.map(point => {
      const distances = centroids.map(center => point.reduce((sum, x, axis) => sum + (x - center[axis]) ** 2, 0));
      return distances.indexOf(Math.min(...distances));
    });
    const loss = points.reduce((sum, point, i) => sum + point.reduce((s, x, axis) => s + (x - centroids[labels[i]][axis]) ** 2, 0), 0);
    const empty = [0, 1, 2].filter(group => !labels.includes(group));
    const stable = previous && labels.every((label, i) => label === previous[i]);
    frames.push({
      phase: stable ? 'Assignments stable' : 'Assign nearest center',
      centroids,
      labels,
      loss,
      empty,
      note: stable ? 'No assignment changed. This is a local Lloyd fixed point, not a global certificate.' : 'Hold centers fixed; choose the smallest squared distance. Exact ties choose the lower center index.'
    });
    if (stable) break;
    centroids = centroids.map((old, group) => {
      const members = points.filter((_, i) => labels[i] === group);
      return members.length ? old.map((_, axis) => members.reduce((sum, point) => sum + point[axis], 0) / members.length) : old;
    });
    const updatedLoss = points.reduce((sum, point, i) => sum + point.reduce((s, x, axis) => s + (x - centroids[labels[i]][axis]) ** 2, 0), 0);
    frames.push({
      phase: 'Move to means',
      centroids,
      labels,
      loss: updatedLoss,
      empty,
      note: 'Hold assignments fixed; replace each occupied center by its member mean. An empty center stays in place and is flagged.'
    });
    previous = labels;
  }
  return {
    ...graph,
    spectrum,
    raw,
    rowNorms,
    points,
    seeds,
    frames
  };
}
export const spectralSignals = {
  groups: [1, 1, 1, -1, -1, -1],
  noisy: [1.6, 0.4, 1, -1.6, -0.4, -1],
  spike: [1, 0, 0, 0, 0, 0],
  constant: [2, 2, 2, 2, 2, 2]
};
export function filterState(weight = 0.2, signalName = 'noisy', filter = 'heat', amount = 1) {
  if (!Object.hasOwn(spectralSignals, signalName) || !['heat', 'ridge', 'cutoff'].includes(filter)) throw new RangeError('Unknown signal or filter.');
  if (filter === 'cutoff') {
    if (!Number.isInteger(amount) || amount < 1 || amount > 6) throw new RangeError('Keep 1–6 modes.');
  } else numberIn(amount, 0, 10, 'Filter amount');
  const graph = bridgeSpectrum(weight);
  const signal = spectralSignals[signalName];
  const coefficients = graph.vectors.map(vector => dot(vector, signal));
  // The graph-theoretic zero count is exact; its computed eigenvalues may have roundoff.
  const values = graph.values.map((value, i) => i < graph.components.length ? 0 : value);
  const gains = values.map((value, i) => filter === 'heat' ? Math.exp(-amount * value) : filter === 'ridge' ? 1 / (1 + amount * value) : +(i < amount));
  const output = signal.map((_, node) => graph.vectors.reduce((sum, vector, mode) => sum + vector[node] * coefficients[mode] * gains[mode], 0));
  const discarded = coefficients.reduce((sum, value, i) => sum + (value * (1 - gains[i])) ** 2, 0);
  const repeatedBoundary = filter === 'cutoff' && amount < 6 && Math.abs(values[amount] - values[amount - 1]) < 1e-8;
  return {
    ...graph,
    signal,
    coefficients,
    gains,
    output,
    discarded,
    repeatedBoundary,
    energyBefore: graphEnergy(graph, signal).total,
    energyAfter: graphEnergy(graph, output).total
  };
}

/** Small, deterministic teaching models. Distances are ordinary Euclidean;
 * objectives use squared distance. These bounded fixtures are not a library API.
 */
export const clusteringPoints = Object.freeze([[1, 0], [1.5, 0.5], [3, 2], [3.5, 2], [7, 5], [7.5, 5.5]].map(point => Object.freeze(point)));
export const rectanglePoints = Object.freeze([[0, 0], [0, 1], [3, 0], [3, 1]].map(point => Object.freeze(point)));
function dense(values, maximum, name) {
  if (!Array.isArray(values) || !values.length || values.length > maximum || Array.from({
    length: values.length
  }, (_, index) => !Object.hasOwn(values, index)).some(Boolean)) {
    throw new RangeError(`${name} must be a nonempty dense array of at most ${maximum} entries.`);
  }
}
function checkPoints(points, maximum = 128) {
  dense(points, maximum, 'Points');
  const dimension = points[0]?.length;
  if (!Number.isInteger(dimension) || dimension < 1 || dimension > 3) throw new RangeError('Use one to three coordinates.');
  for (const point of points) {
    dense(point, 3, 'Coordinates');
    if (point.length !== dimension || point.some(value => !Number.isFinite(value) || Math.abs(value) > 10000)) {
      throw new RangeError('Use matching dimensions and finite coordinates with magnitude at most 10000.');
    }
  }
  return dimension;
}
function checkWeights(points, weights) {
  const result = weights ?? points.map(() => 1);
  dense(result, 128, 'Weights');
  if (result.length !== points.length || result.some(value => !Number.isFinite(value) || value <= 0 || value > 1000000)) {
    throw new RangeError('Provide one positive finite weight per point (at most one million).');
  }
  return result;
}
function squaredDistance(left, right) {
  return left.reduce((sum, value, coordinate) => sum + (value - right[coordinate]) ** 2, 0);
}
function validateLabels(points, labels, clusterCount) {
  dense(labels, 128, 'Labels');
  if (labels.length !== points.length || labels.some(label => !Number.isInteger(label) || label < 0 || label >= clusterCount)) {
    throw new RangeError('Each label must index a supplied center.');
  }
}
export function assignClusters(points, centers, weights) {
  const dimension = checkPoints(points);
  if (checkPoints(centers, 8) !== dimension) throw new RangeError('Center dimensions must match points.');
  const masses = checkWeights(points, weights);
  const distances = points.map(point => centers.map(center => squaredDistance(point, center)));
  // Strict comparison keeps the lowest center index on an exact distance tie.
  const labels = distances.map(row => row.reduce((best, distance, index) => distance < row[best] ? index : best, 0));
  const contributions = distances.map((row, index) => masses[index] * row[labels[index]]);
  return {
    labels,
    distances,
    contributions,
    sse: contributions.reduce((sum, value) => sum + value, 0)
  };
}
export function moveClusterMeans(points, labels, previousCenters, weights) {
  const dimension = checkPoints(points);
  if (checkPoints(previousCenters, 8) !== dimension) throw new RangeError('Center dimensions must match points.');
  const masses = checkWeights(points, weights);
  validateLabels(points, labels, previousCenters.length);
  const totals = previousCenters.map(() => 0);
  const sums = previousCenters.map(() => Array(dimension).fill(0));
  points.forEach((point, index) => {
    totals[labels[index]] += masses[index];
    point.forEach((value, coordinate) => {
      sums[labels[index]][coordinate] += masses[index] * value;
    });
  });
  const centers = sums.map((sum, cluster) => totals[cluster] > 0 ? sum.map(value => value / totals[cluster]) : [...previousCenters[cluster]]);
  const contributions = points.map((point, index) => masses[index] * squaredDistance(point, centers[labels[index]]));
  return {
    centers,
    totals,
    empty: totals.flatMap((total, index) => total === 0 ? [index] : []),
    contributions,
    sse: contributions.reduce((sum, value) => sum + value, 0)
  };
}
export function lloydTrace(points, initialCenters, weights, maxSweeps = 30) {
  if (!Number.isInteger(maxSweeps) || maxSweeps < 1 || maxSweeps > 60) throw new RangeError('Use one to sixty sweeps.');
  assignClusters(points, initialCenters, weights);
  let centers = initialCenters.map(center => [...center]);
  let previousLabels = null;
  const trace = [{
    phase: 'initial',
    centers,
    labels: null,
    sse: null,
    sweep: 0,
    empty: [],
    status: 'ready'
  }];
  for (let sweep = 1; sweep <= maxSweeps; sweep += 1) {
    const assigned = assignClusters(points, centers, weights);
    const stable = previousLabels !== null && assigned.labels.every((label, index) => label === previousLabels[index]);
    trace.push({
      phase: 'assign',
      centers,
      ...assigned,
      sweep,
      empty: centers.flatMap((_, cluster) => assigned.labels.includes(cluster) ? [] : [cluster]),
      status: stable ? 'fixed assignment' : 'continue'
    });
    if (stable) return trace;
    const moved = moveClusterMeans(points, assigned.labels, centers, weights);
    centers = moved.centers;
    trace.push({
      phase: 'move',
      labels: assigned.labels,
      sweep,
      ...moved,
      status: sweep === maxSweeps ? 'sweep budget reached' : 'continue'
    });
    previousLabels = assigned.labels;
  }
  return trace;
}
export function seedingDistribution(points, selectedIndices, quantile = 0.5) {
  checkPoints(points);
  dense(selectedIndices, 8, 'Selected rows');
  if (selectedIndices.some(index => !Number.isInteger(index) || index < 0 || index >= points.length) || new Set(selectedIndices).size !== selectedIndices.length) throw new RangeError('Use distinct valid row indices.');
  if (!Number.isFinite(quantile) || quantile < 0 || quantile >= 1) throw new RangeError('The draw position is in [0,1).');
  const distances = points.map(point => Math.min(...selectedIndices.map(index => squaredDistance(point, points[index]))));
  const total = distances.reduce((sum, value) => sum + value, 0);
  let cumulative = 0;
  const rows = distances.map((distance, index) => {
    const probability = total > 0 ? distance / total : 0;
    const start = cumulative;
    cumulative += probability;
    return {
      index,
      distance,
      probability,
      start,
      end: cumulative
    };
  });
  const selected = total === 0 ? null : (rows.find(row => row.probability > 0 && quantile < row.end) ?? rows.findLast(row => row.probability > 0)).index;
  return {
    rows,
    total,
    selected,
    quantile,
    stopped: total === 0
  };
}
function groupMean(points, members) {
  return points[0].map((_, coordinate) => members.reduce((sum, index) => sum + points[index][coordinate], 0) / members.length);
}
export function hierarchyTrace(points, linkage = 'ward') {
  checkPoints(points, 16);
  if (!['single', 'complete', 'average', 'ward'].includes(linkage)) throw new RangeError('Unknown linkage.');
  const nodes = points.map((point, index) => ({
    id: index,
    members: [index],
    mean: [...point],
    height: 0,
    sse: 0
  }));
  let active = nodes.map(node => node.id);
  const merges = [];
  while (active.length > 1) {
    let best = null;
    for (let left = 0; left < active.length; left += 1) {
      for (let right = left + 1; right < active.length; right += 1) {
        const first = nodes[active[left]],
          second = nodes[active[right]];
        const delta = first.members.length * second.members.length / (first.members.length + second.members.length) * squaredDistance(first.mean, second.mean);
        const pairDistances = first.members.flatMap(a => second.members.map(b => Math.sqrt(squaredDistance(points[a], points[b]))));
        const height = linkage === 'ward' ? Math.sqrt(2 * delta) : linkage === 'single' ? Math.min(...pairDistances) : linkage === 'complete' ? Math.max(...pairDistances) : pairDistances.reduce((sum, value) => sum + value, 0) / pairDistances.length;
        // Active IDs are sorted; the first exact tie is the lexicographically first pair.
        if (best === null || height < best.height) best = {
          left: first.id,
          right: second.id,
          height,
          delta
        };
      }
    }
    const members = [...nodes[best.left].members, ...nodes[best.right].members];
    const mean = groupMean(points, members);
    const sse = members.reduce((sum, index) => sum + squaredDistance(points[index], mean), 0);
    const merged = {
      ...best,
      id: nodes.length,
      members,
      mean,
      sse,
      size: members.length
    };
    nodes.push(merged);
    merges.push(merged);
    active = [...active.filter(id => id !== best.left && id !== best.right), merged.id].sort((a, b) => a - b);
  }
  const root = nodes.at(-1);
  function leafOrder(id) {
    const node = nodes[id];
    return node.left === undefined ? [id] : [...leafOrder(node.left), ...leafOrder(node.right)];
  }
  return {
    points: points.map(point => [...point]),
    linkage,
    nodes,
    merges,
    leafOrder: leafOrder(root.id),
    root
  };
}
export function cutHierarchy(tree, mode, value) {
  if (!tree?.points || !tree?.merges) throw new RangeError('Provide a hierarchy trace.');
  const count = tree.points.length;
  if (mode === 'count') {
    if (!Number.isInteger(value) || value < 1 || value > count) throw new RangeError('The requested count must be between one and the number of points.');
  } else if (mode === 'height') {
    if (!Number.isFinite(value) || value < 0) throw new RangeError('Cut height must be nonnegative and finite.');
  } else throw new RangeError('Choose a count or a height cut.');
  const selected = mode === 'count' ? tree.merges.slice(0, count - value) : tree.merges.filter(merge => merge.height <= value);
  const active = new Set(tree.points.map((_, index) => index));
  selected.forEach(merge => {
    active.delete(merge.left);
    active.delete(merge.right);
    active.add(merge.id);
  });
  const groups = [...active].map(id => tree.nodes[id].members).sort((left, right) => Math.min(...left) - Math.min(...right));
  const labels = tree.points.map((_, index) => groups.findIndex(group => group.includes(index)));
  return {
    groups,
    labels,
    count: groups.length,
    mergesUsed: selected.length,
    sse: [...active].reduce((sum, id) => sum + tree.nodes[id].sse, 0)
  };
}
export function featureGeometry(unitFactor = 1, verticalWeight = 1) {
  if (![1, 10].includes(unitFactor) || ![0.01, 0.25, 1, 4].includes(verticalWeight)) throw new RangeError('Use a displayed unit and weight choice.');
  const effective = unitFactor * Math.sqrt(verticalWeight);
  const points = rectanglePoints.map(([x, y]) => [x, y * effective]);
  const trials = [[0, 2], [0, 1]].map(indices => lloydTrace(points, indices.map(index => points[index])).at(-1));
  const best = trials[0].sse <= trials[1].sse ? trials[0] : trials[1];
  return {
    points,
    effective,
    trials,
    ...best
  };
}
const paletteColors = [[30, 48, 65], [45, 68, 80], [196, 128, 47], [228, 172, 64], [84, 119, 97], [113, 151, 122]];
export const palettePixels = Object.freeze(Array.from({
  length: 96
}, (_, index) => {
  const row = Math.floor(index / 12),
    column = index % 12;
  const color = row < 3 ? column < 7 ? 0 : 1 : column < 4 ? row % 2 === 0 ? 2 : 3 : (column + row) % 3 === 0 ? 4 : 5;
  return Object.freeze([...paletteColors[color]]);
}));
export function quantizePalette(clusterCount = 3) {
  if (!Number.isInteger(clusterCount) || clusterCount < 1 || clusterCount > paletteColors.length) throw new RangeError('Use one to six palette entries.');
  const counts = paletteColors.map(color => palettePixels.filter(pixel => pixel.every((value, coordinate) => value === color[coordinate])).length);
  const chosen = [counts.indexOf(Math.max(...counts))];
  while (chosen.length < clusterCount) {
    const distances = paletteColors.map(color => Math.min(...chosen.map(index => squaredDistance(color, paletteColors[index]))));
    chosen.push(distances.indexOf(Math.max(...distances)));
  }
  const trace = lloydTrace(paletteColors, chosen.map(index => paletteColors[index]), counts);
  const final = trace.at(-1);
  const roundedCenters = final.centers.map(center => center.map(Math.round));
  const reconstructed = palettePixels.map(pixel => {
    const index = paletteColors.findIndex(color => color.every((value, coordinate) => value === pixel[coordinate]));
    return roundedCenters[final.labels[index]];
  });
  const displayedSse = palettePixels.reduce((sum, pixel, index) => sum + squaredDistance(pixel, reconstructed[index]), 0);
  return {
    colors: paletteColors.map(color => [...color]),
    counts,
    initialIndices: chosen,
    trace,
    ...final,
    roundedCenters,
    reconstructed,
    displayedSse,
    meanSquaredChannelError: displayedSse / (3 * palettePixels.length)
  };
}

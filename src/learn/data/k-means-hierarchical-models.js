/** Small, deterministic teaching models. Distances are ordinary Euclidean;
 * objectives use squared distance. These bounded fixtures are not a library API.
 * Reject arithmetic that turns a nonzero squared separation or weighted term
 * into zero; finite input coordinates alone do not ensure representable costs.
 */
export const clusteringPoints = Object.freeze([[1, 0], [1.5, 0.5], [3, 2], [3.5, 2], [7, 5], [7.5, 5.5]].map(point => Object.freeze(point)));
export const rectanglePoints = Object.freeze([[0, 0], [0, 1], [3, 0], [3, 1]].map(point => Object.freeze(point)));
/** Six points exactly one unit apart along a line, plus a compact triple above
 * the middle of the chain. Unit spacing keeps the five chain distances exactly
 * equal in floating point, so the single-linkage tie at height 1 is genuine.
 * Single linkage follows the chain; complete linkage breaks it. */
export const chainPoints = Object.freeze([[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [2.5, 3.3], [3.1, 3.3], [2.8, 3.8]].map(point => Object.freeze(point)));
export const hierarchyFixtures = Object.freeze({ six: clusteringPoints, chain: chainPoints });
function dense(values, maximum, name) {
  if (!Array.isArray(values) || !values.length || values.length > maximum || Array.from({
    length: values.length
  }, (_, index) => !Object.hasOwn(values, index)).some(Boolean)) {
    throw new RangeError(`${name} must be a nonempty dense array of at most ${maximum} entries.`);
  }
}
function checkPoints(points, maximum = 256) {
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
  dense(result, 256, 'Weights');
  if (result.length !== points.length || result.some(value => !Number.isFinite(value) || value <= 0 || value > 1000000)) {
    throw new RangeError('Provide one positive finite weight per point (at most one million).');
  }
  return result;
}
function squaredDistance(left, right) {
  return left.reduce((sum, value, coordinate) => {
    const difference = value - right[coordinate];
    const square = difference ** 2;
    if (difference !== 0 && square === 0) {
      throw new RangeError('Squared separation underflows; rescale the coordinates before computing distances.');
    }
    return sum + square;
  }, 0);
}
function checkedProduct(left, right, label) {
  const result = left * right;
  if (!Number.isFinite(result) || left !== 0 && right !== 0 && result === 0) {
    throw new RangeError(`${label} is outside the representable arithmetic range; rescale the inputs.`);
  }
  return result;
}
function checkedQuotient(numerator, denominator, label) {
  const result = numerator / denominator;
  if (!Number.isFinite(result) || numerator !== 0 && result === 0) {
    throw new RangeError(`${label} is outside the representable arithmetic range; rescale the inputs.`);
  }
  return result;
}
function validateLabels(points, labels, clusterCount) {
  dense(labels, 256, 'Labels');
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
  const contributions = distances.map((row, index) => checkedProduct(masses[index], row[labels[index]], 'Weighted squared error'));
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
      sums[labels[index]][coordinate] += checkedProduct(masses[index], value, 'Weighted coordinate');
    });
  });
  const centers = sums.map((sum, cluster) => totals[cluster] > 0 ? sum.map(value => checkedQuotient(value, totals[cluster], 'Weighted mean')) : [...previousCenters[cluster]]);
  const contributions = points.map((point, index) => checkedProduct(masses[index], squaredDistance(point, centers[labels[index]]), 'Weighted squared error'));
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
/** Every way to split at most eight points into two nonempty groups, scored by
 * the squared error of the two group means. P0 is pinned to group 0 so each
 * partition appears once. Because the list is exhaustive, its first entry is
 * the exact global optimum for k = 2 under the supplied geometry. */
export function enumerateTwoGroupPartitions(points) {
  const dimension = checkPoints(points, 8);
  if (points.length < 2) throw new RangeError('Enumerate at least two points.');
  const partitions = [];
  for (let mask = 1; mask < 2 ** (points.length - 1); mask += 1) {
    const labels = points.map((_, index) => index === 0 ? 0 : mask >> index - 1 & 1);
    const groups = [0, 1].map(label => labels.flatMap((value, index) => value === label ? [index] : []));
    const centers = groups.map(members => Array.from({ length: dimension }, (_, coordinate) => checkedQuotient(members.reduce((sum, index) => sum + points[index][coordinate], 0), members.length, 'Group mean')));
    const sse = points.reduce((sum, point, index) => sum + squaredDistance(point, centers[labels[index]]), 0);
    partitions.push({ labels, groups, centers, sse });
  }
  partitions.sort((left, right) => left.sse - right.sse || left.labels.join('').localeCompare(right.labels.join('')));
  return partitions;
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
    const probability = total > 0 ? checkedQuotient(distance, total, 'D squared probability') : 0;
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
/** Deterministic 32-bit generator (mulberry32) so repeated draws are reproducible. */
function seededUniform(seed) {
  let state = seed >>> 0;
  return () => {
    state = state + 0x6D2B79F5 >>> 0;
    let value = state;
    value = Math.imul(value ^ value >>> 15, value | 1);
    value ^= value + Math.imul(value ^ value >>> 7, value | 61);
    return ((value ^ value >>> 14) >>> 0) / 4294967296;
  };
}
/** Repeat the conditional D² draw many times with a fixed seed and count how
 * often each row is selected; also report the uniform-seeding probability for
 * comparison. Frequencies estimate the probabilities in the strip; they are not
 * a second algorithm. */
export function seedingFrequencies(points, selectedIndices, draws = 200, seed = 1) {
  if (!Number.isInteger(draws) || draws < 1 || draws > 2000) throw new RangeError('Use one to two thousand draws.');
  if (!Number.isInteger(seed) || seed < 0) throw new RangeError('Use a nonnegative integer seed.');
  const distribution = seedingDistribution(points, selectedIndices, 0);
  const counts = points.map(() => 0);
  if (!distribution.stopped) {
    const next = seededUniform(seed);
    for (let draw = 0; draw < draws; draw += 1) {
      const u = next();
      const row = distribution.rows.find(entry => entry.probability > 0 && u < entry.end) ?? distribution.rows.findLast(entry => entry.probability > 0);
      counts[row.index] += 1;
    }
  }
  const farthest = distribution.stopped ? null : distribution.rows.reduce((best, row) => row.distance > best.distance ? row : best).index;
  return {
    draws,
    seed,
    counts,
    frequencies: counts.map(count => count / draws),
    uniformProbability: selectedIndices.length < points.length ? 1 / (points.length - selectedIndices.length) : 0,
    farthest,
    farthestProbability: farthest === null ? 0 : distribution.rows[farthest].probability,
    stopped: distribution.stopped
  };
}
function groupMean(points, members) {
  return points[0].map((_, coordinate) => checkedQuotient(members.reduce((sum, index) => sum + points[index][coordinate], 0), members.length, 'Cluster mean'));
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
        const sizeFactor = first.members.length * second.members.length / (first.members.length + second.members.length);
        const delta = checkedProduct(sizeFactor, squaredDistance(first.mean, second.mean), 'Ward SSE increase');
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
/** The rectangle under a changed vertical unit and squared-distance weight. The
 * optimum is found by exhaustive enumeration of all seven two-group partitions,
 * so the reported winner is exact for this geometry rather than the result of a
 * particular initialization. */
export function featureGeometry(unitFactor = 1, verticalWeight = 1) {
  if (![1, 10].includes(unitFactor) || !Number.isFinite(verticalWeight) || verticalWeight < 0.01 || verticalWeight > 4) {
    throw new RangeError('Use a displayed unit choice and a finite vertical weight from 0.01 to 4.');
  }
  const effective = unitFactor * Math.sqrt(verticalWeight);
  const points = rectanglePoints.map(([x, y]) => [x, y * effective]);
  const partitions = enumerateTwoGroupPartitions(points);
  const best = partitions[0];
  return {
    points,
    effective,
    partitions,
    labels: best.labels,
    centers: best.centers,
    sse: best.sse
  };
}
const mosaicColors = [[30, 48, 65], [45, 68, 80], [196, 128, 47], [228, 172, 64], [84, 119, 97], [113, 151, 122]];
function buildImage(columns, rows, colorAt) {
  const pixels = [];
  for (let row = 0; row < rows; row += 1) for (let column = 0; column < columns; column += 1) pixels.push(Object.freeze(colorAt(row, column).map(Math.round)));
  return Object.freeze({ columns, rows, pixels: Object.freeze(pixels) });
}
/** Three constructed rasters. The mosaic has six exact colors; the sky has
 * gradient bands, a sun and clouds with a few dozen unique colors and very
 * unequal counts; the gradient makes every pixel a unique color, so counts
 * cannot help there. None is a photograph. */
export const paletteImages = Object.freeze({
  mosaic: buildImage(12, 8, (row, column) => mosaicColors[row < 3 ? column < 7 ? 0 : 1 : column < 4 ? row % 2 === 0 ? 2 : 3 : (column + row) % 3 === 0 ? 4 : 5]),
  sky: buildImage(16, 10, (row, column) => {
    if (row >= 8) return row === 8 ? [58, 104, 66] : [46, 88, 56];
    if (Math.hypot(column - 12, row - 2) < 1.7) return Math.hypot(column - 12, row - 2) < 0.8 ? [250, 214, 96] : [236, 184, 72];
    if (row >= 4 && row <= 5 && column >= 2 && column <= 7) return column % 3 === 0 ? [228, 232, 238] : column % 3 === 1 ? [212, 218, 228] : [196, 204, 218];
    const t = row / 7;
    const tint = Math.floor(column / 4) * 6;
    return [38 + 110 * t + tint, 70 + 118 * t + tint / 2, 140 + 90 * t];
  }),
  gradient: buildImage(16, 10, (row, column) => [30 + 13 * column, 40 + 20 * row, 200 - 8 * column])
});
export const palettePixels = paletteImages.mosaic.pixels;
function uniqueColors(pixels) {
  const seen = new Map();
  pixels.forEach(pixel => {
    const key = pixel.join(',');
    if (!seen.has(key)) seen.set(key, { color: [...pixel], count: 0 });
    seen.get(key).count += 1;
  });
  const entries = [...seen.values()].sort((left, right) => right.count - left.count || left.color.join(',').localeCompare(right.color.join(',')));
  return { colors: entries.map(entry => entry.color), counts: entries.map(entry => entry.count) };
}
export function paletteLimit(imageId = 'mosaic') {
  if (!Object.hasOwn(paletteImages, imageId)) throw new RangeError('Unknown palette image.');
  return Math.min(8, uniqueColors(paletteImages[imageId].pixels).colors.length);
}
export function quantizePalette(clusterCount = 3, imageId = 'mosaic') {
  if (!Object.hasOwn(paletteImages, imageId)) throw new RangeError('Unknown palette image.');
  const image = paletteImages[imageId];
  const { colors, counts } = uniqueColors(image.pixels);
  if (!Number.isInteger(clusterCount) || clusterCount < 1 || clusterCount > Math.min(8, colors.length)) throw new RangeError(`Use one to ${Math.min(8, colors.length)} palette entries for this image.`);
  // Most frequent color first, then deterministic farthest-first: a declared start, not a D² sample.
  const chosen = [0];
  while (chosen.length < clusterCount) {
    const distances = colors.map(color => Math.min(...chosen.map(index => squaredDistance(color, colors[index]))));
    chosen.push(distances.indexOf(Math.max(...distances)));
  }
  const trace = lloydTrace(colors, chosen.map(index => colors[index]), counts, 60);
  const final = trace.at(-1);
  const roundedCenters = final.centers.map(center => center.map(Math.round));
  const colorIndex = new Map(colors.map((color, index) => [color.join(','), index]));
  const reconstructed = image.pixels.map(pixel => roundedCenters[final.labels[colorIndex.get(pixel.join(','))]]);
  const displayedSse = image.pixels.reduce((sum, pixel, index) => sum + squaredDistance(pixel, reconstructed[index]), 0);
  return {
    imageId,
    image,
    colors,
    counts,
    uniqueCount: colors.length,
    initialIndices: chosen,
    trace,
    ...final,
    roundedCenters,
    reconstructed,
    displayedSse,
    meanSquaredChannelError: displayedSse / (3 * image.pixels.length)
  };
}

// Exact on these bounded integer coordinates: every determinant is below 2^53.
function point(value) {
  if (!Array.isArray(value) || value.length !== 2 || value.some(coordinate => !Number.isInteger(coordinate) || Math.abs(coordinate) > 1_000_000)) {
    throw new RangeError('Use integer coordinate pairs between −1000000 and 1000000.');
  }
  return value;
}
export function orientation(a, b, c) {
  [a, b, c].forEach(point);
  return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
}
export function onSegment(a, b, query) {
  return orientation(a, b, query) === 0 && Math.min(a[0], b[0]) <= query[0] && query[0] <= Math.max(a[0], b[0]) && Math.min(a[1], b[1]) <= query[1] && query[1] <= Math.max(a[1], b[1]);
}
const comparePoints = (left, right) => left[0] - right[0] || left[1] - right[1];
const equalPoints = (left, right) => left[0] === right[0] && left[1] === right[1];
function uniquePoints(points) {
  points.forEach(point);
  return [...new Map(points.map(value => [value.join(','), [...value]])).values()].sort(comparePoints);
}
export function segmentState(a, b, c, d) {
  const determinants = [orientation(a, b, c), orientation(a, b, d), orientation(c, d, a), orientation(c, d, b)];
  const boxesOverlap = [0, 1].every(axis => Math.max(Math.min(a[axis], b[axis]), Math.min(c[axis], d[axis])) <= Math.min(Math.max(a[axis], b[axis]), Math.max(c[axis], d[axis])));
  const sharedEndpoints = uniquePoints([a, b, c, d].filter(query => onSegment(a, b, query) && onSegment(c, d, query)));
  let kind = 'disjoint';
  if (sharedEndpoints.length >= 2) kind = 'overlap';else if (sharedEndpoints.length === 1) kind = 'touch';else if (Math.sign(determinants[0]) * Math.sign(determinants[1]) < 0 && Math.sign(determinants[2]) * Math.sign(determinants[3]) < 0) kind = 'proper crossing';
  return {
    a,
    b,
    c,
    d,
    determinants,
    boxesOverlap,
    kind,
    sharedEndpoints
  };
}
export const segmentPresets = {
  crossing: [[1, 1], [7, 5], [1, 5], [7, 1]],
  endpoint: [[1, 1], [4, 3], [4, 3], [7, 1]],
  overlap: [[1, 3], [6, 3], [3, 3], [7, 3]],
  parallel: [[1, 1], [6, 5], [1, 2], [5, 6]],
  point: [[4, 3], [4, 3], [1, 3], [7, 3]]
};
export const hullPresets = {
  fence: [[1, 1], [3, 1], [7, 1], [7, 6], [4, 6], [1, 6], [3, 3], [5, 4], [1, 1]],
  turns: [[1, 1], [2, 4], [3, 2], [4, 5], [5, 1], [7, 4], [7, 6]],
  line: [[1, 2], [3, 3], [5, 4], [7, 5]],
  singleton: [[4, 3], [4, 3]],
  empty: []
};
export function hullTrace(input, includeBoundary = false) {
  if (!Array.isArray(input) || input.length > 50) throw new RangeError('Use at most 50 point records in the trace.');
  const sorted = uniquePoints(input);
  const frames = [];
  const collinear = sorted.length < 3 || sorted.every(value => orientation(sorted[0], sorted.at(-1), value) === 0);
  if (collinear) {
    const hull = includeBoundary || sorted.length < 2 ? sorted : [sorted[0], sorted.at(-1)];
    frames.push({
      phase: 'degenerate',
      action: 'Return the explicit empty, point or collinear contract',
      stack: hull,
      candidate: null,
      removed: null,
      determinant: null,
      lower: []
    });
    return {
      sorted,
      frames,
      hull,
      collinear,
      includeBoundary
    };
  }
  let lower = [];
  const chains = [];
  for (const [phase, sequence] of [['lower', sorted], ['upper', [...sorted].reverse()]]) {
    const stack = [];
    for (const candidate of sequence) {
      frames.push({
        phase,
        action: 'Inspect the next sorted point',
        candidate,
        stack: [...stack],
        removed: null,
        determinant: null,
        lower
      });
      while (stack.length >= 2) {
        const determinant = orientation(stack.at(-2), stack.at(-1), candidate);
        if (includeBoundary ? determinant >= 0 : determinant > 0) break;
        const removed = stack.pop();
        frames.push({
          phase,
          action: 'Remove the last point: its turn violates this chain policy',
          candidate,
          stack: [...stack],
          removed,
          determinant,
          lower
        });
      }
      stack.push(candidate);
      frames.push({
        phase,
        action: 'Append the candidate; the chain invariant is restored',
        candidate,
        stack: [...stack],
        removed: null,
        determinant: null,
        lower
      });
    }
    chains.push(stack);
    if (phase === 'lower') lower = [...stack];
  }
  const hull = [...chains[0].slice(0, -1), ...chains[1].slice(0, -1)];
  frames.push({
    phase: 'complete',
    action: 'Join chains, omitting each repeated end point',
    stack: hull,
    candidate: null,
    removed: null,
    determinant: null,
    lower
  });
  return {
    sorted,
    frames,
    hull,
    collinear,
    includeBoundary
  };
}
export function parseGridPoints(text) {
  if (typeof text !== 'string' || text.length > 800) throw new RangeError('Use at most 20 lines of x,y grid coordinates.');
  const lines = text.trim() ? text.trim().split(/\r?\n/) : [];
  if (lines.length > 20) throw new RangeError('Use at most 20 point records.');
  return lines.map((line, index) => {
    if (!/^\s*[0-8]\s*,\s*[0-8]\s*$/.test(line)) throw new RangeError(`Line ${index + 1}: use x,y with integers 0 through 8.`);
    return line.split(',').map(Number);
  });
}
export const polygonPresets = {
  courtyard: [[1, 1], [7, 1], [7, 6], [5, 6], [5, 3], [3, 3], [3, 6], [1, 6]],
  rectangle: [[1, 1], [7, 1], [7, 6], [1, 6]],
  triangle: [[1, 1], [7, 1], [4, 6]]
};
export function polygonState(vertices, query) {
  if (vertices.length < 3 || vertices.length > 50) throw new RangeError('Use a simple polygon with 3 through 50 vertices.');
  vertices.forEach(point);
  point(query);
  let doubledArea = 0;
  let crossings = 0;
  let boundary = false;
  const edges = vertices.map((a, index) => {
    const b = vertices[(index + 1) % vertices.length];
    const determinant = orientation(a, b, query);
    const onBoundary = onSegment(a, b, query);
    boundary ||= onBoundary;
    const straddles = a[1] > query[1] !== b[1] > query[1];
    const rightCrossing = straddles && (b[1] > a[1] ? determinant > 0 : determinant < 0);
    if (rightCrossing) crossings += 1;
    doubledArea += a[0] * b[1] - a[1] * b[0];
    return {
      a,
      b,
      determinant,
      onBoundary,
      straddles,
      rightCrossing,
      crossings
    };
  });
  return {
    vertices,
    query,
    edges,
    doubledArea,
    area: Math.abs(doubledArea) / 2,
    crossings,
    classification: boundary ? 'boundary' : crossings % 2 ? 'inside' : 'outside'
  };
}
export function precisionState(exponent = 27, scenario = 'products') {
  if (!Number.isInteger(exponent) || exponent < 20 || exponent > 30) throw new RangeError('Choose an exponent from 20 through 30.');
  if (!['products', 'input'].includes(scenario)) throw new RangeError('Unknown precision scenario.');
  const n = 2n ** BigInt(scenario === 'input' ? 53 : exponent);
  const points = scenario === 'input' ? [[n, 0n], [n + 1n, 0n], [n, 1n]] : [[0n, 0n], [n + 1n, n], [n, n - 1n]];
  const [a, b, c] = points;
  const exactProducts = [(b[0] - a[0]) * (c[1] - a[1]), (b[1] - a[1]) * (c[0] - a[0])];
  const [af, bf, cf] = points.map(value => value.map(Number));
  const floatProducts = [(bf[0] - af[0]) * (cf[1] - af[1]), (bf[1] - af[1]) * (cf[0] - af[0])];
  return {
    exponent,
    scenario,
    points: points.map(value => value.map(String)),
    represented: [af, bf, cf],
    exactProducts: exactProducts.map(String),
    floatProducts,
    exactDeterminant: String(exactProducts[0] - exactProducts[1]),
    floatDeterminant: floatProducts[0] - floatProducts[1],
    allInputsPreserved: points.every((value, index) => value.every((coordinate, axis) => BigInt([af, bf, cf][index][axis]) === coordinate))
  };
}

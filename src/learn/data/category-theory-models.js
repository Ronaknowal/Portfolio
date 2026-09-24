// Exact finite sets use integer indices; numeric probability/derivative views
// are bounded educational calculations, never an arbitrary-code interpreter.
function integer(value, minimum, maximum, name) {
  if (!Number.isInteger(value) || value < minimum || value > maximum) {
    throw new RangeError(name + ' is outside the stated finite model.');
  }
}
function finiteNumber(value, minimum, maximum, name) {
  if (!Number.isFinite(value) || value < minimum || value > maximum) {
    throw new RangeError(name + ' must be finite and within the displayed range.');
  }
}
export function validateFiniteMap(mapping, sourceSize, targetSize) {
  integer(sourceSize, 0, 16, 'Source size');
  integer(targetSize, 0, 16, 'Target size');
  if (!Array.isArray(mapping) || mapping.length !== sourceSize) {
    throw new TypeError('Every source element needs exactly one image.');
  }
  for (const value of mapping) integer(value, 0, targetSize - 1, 'Image');
  return mapping.slice();
}
export function composeFiniteMaps(after, before, targetSize) {
  validateFiniteMap(after, after.length, targetSize);
  validateFiniteMap(before, before.length, after.length);
  return before.map(value => after[value]);
}
export const finiteFunctionChoices = Object.freeze({
  cycle: Object.freeze([1, 2, 0]),
  reverse: Object.freeze([2, 1, 0]),
  collapse: Object.freeze([0, 0, 2]),
  identity: Object.freeze([0, 1, 2])
});
function selectedFunction(name) {
  if (!Object.hasOwn(finiteFunctionChoices, name)) throw new RangeError('Unknown finite function.');
  return finiteFunctionChoices[name];
}
export function compositionSnapshot(input = 0, first = 'cycle', second = 'collapse', third = 'reverse') {
  integer(input, 0, 2, 'Input');
  const f = selectedFunction(first);
  const g = selectedFunction(second);
  const h = selectedFunction(third);
  const gf = composeFiniteMaps(g, f, 3);
  const hg = composeFiniteMaps(h, g, 3);
  const left = composeFiniteMaps(h, gf, 3);
  const right = composeFiniteMaps(hg, f, 3);
  return {
    maps: [f.slice(), g.slice(), h.slice()],
    path: [input, f[input], g[f[input]], h[g[f[input]]]],
    left,
    right,
    associative: left.every((value, index) => value === right[index]),
    swapped: composeFiniteMaps(f, g, 3),
    intermediate: gf
  };
}
export function schemaSnapshot(sensor = 0, directSite = [0, 0, 1]) {
  integer(sensor, 0, 2, 'Sensor');
  validateFiniteMap(directSite, 3, 2);
  const deviceOf = [0, 0, 1];
  const siteOf = [0, 1];
  const composed = composeFiniteMaps(siteOf, deviceOf, 2);
  return {
    sensors: ['s0', 's1', 's2'],
    devices: ['d0', 'd1'],
    sites: ['North', 'South'],
    deviceOf,
    siteOf,
    directSite: directSite.slice(),
    composed,
    sensor,
    path: [sensor, deviceOf[sensor], composed[sensor]],
    violations: composed.flatMap((site, index) => site === directSite[index] ? [] : [index])
  };
}
export function naturalitySnapshot(values = [2, 0, 1], operation = 'reverse', mapping = 'reverse') {
  if (!Array.isArray(values) || values.length > 8) throw new RangeError('Use at most eight list entries.');
  validateFiniteMap(values, values.length, 3);
  if (!['reverse', 'sort'].includes(operation)) throw new RangeError('Unknown list operation.');
  const f = selectedFunction(mapping);
  const transform = list => operation === 'reverse' ? list.slice().reverse() : list.slice().sort((a, b) => a - b);
  const transformed = transform(values);
  const mapped = values.map(value => f[value]);
  const topThenRight = transformed.map(value => f[value]);
  const leftThenBottom = transform(mapped);
  return {
    original: values.slice(),
    transformed,
    mapped,
    topThenRight,
    leftThenBottom,
    mapping: f.slice(),
    commutesHere: topThenRight.every((value, index) => value === leftThenBottom[index])
  };
}
export function productSnapshot(mode = 'complete', row = 0, column = 0) {
  if (!['complete', 'missing', 'duplicate'].includes(mode)) throw new RangeError('Unknown candidate product.');
  integer(row, 0, 1, 'Row');
  integer(column, 0, 1, 'Column');
  const points = [{
    id: 'p00',
    row: 0,
    column: 0
  }, {
    id: 'p01',
    row: 0,
    column: 1
  }, {
    id: 'p10',
    row: 1,
    column: 0
  }, {
    id: 'p11',
    row: 1,
    column: 1
  }].filter(point => mode !== 'missing' || point.id !== 'p00');
  if (mode === 'duplicate') points.push({
    id: 'q00',
    row: 0,
    column: 0
  });
  const mediators = points.filter(point => point.row === row && point.column === column);
  const counts = [0, 1].map(r => [0, 1].map(c => points.filter(point => point.row === r && point.column === c).length));
  return {
    points,
    row,
    column,
    mediators,
    counts,
    universal: counts.flat().every(count => count === 1)
  };
}
export function pullbackPairs(leftMap, rightMap, targetSize) {
  validateFiniteMap(leftMap, leftMap.length, targetSize);
  validateFiniteMap(rightMap, rightMap.length, targetSize);
  return leftMap.flatMap((image, left) => rightMap.flatMap((other, right) => image === other ? [[left, right]] : []));
}
export function multiplyStochastic(first, second) {
  const validate = matrix => {
    if (!Array.isArray(matrix) || matrix.length < 1 || matrix.length > 8 || !Array.isArray(matrix[0])) {
      throw new RangeError('Use nonempty finite matrices with at most eight rows and columns.');
    }
    const width = matrix[0].length;
    integer(width, 1, 8, 'Matrix width');
    for (const row of matrix) {
      if (!Array.isArray(row) || row.length !== width) throw new TypeError('Matrix rows must have the same length.');
      for (const value of row) finiteNumber(value, 0, 1, 'Probability');
      if (Math.abs(row.reduce((sum, value) => sum + value, 0) - 1) > 1e-12) {
        throw new RangeError('Every stochastic row must sum to one.');
      }
    }
    return width;
  };
  const middle = validate(first);
  const width = validate(second);
  if (middle !== second.length) throw new RangeError('The intermediate state spaces do not match.');
  return first.map(row => Array.from({
    length: width
  }, (_, column) => row.reduce((sum, probability, index) => sum + probability * second[index][column], 0)));
}
export function stochasticCopySnapshot(percent = 50) {
  integer(percent, 0, 100, 'Probability percentage');
  const p = percent / 100;
  const marginal = [1 - p, p];
  const copied = [[1 - p, 0], [0, p]];
  const independent = marginal.map(first => marginal.map(second => first * second));
  return {
    p,
    marginal,
    copied,
    independent,
    copiedDisagreement: 0,
    independentDisagreement: 2 * p * (1 - p),
    copyCommutes: percent === 0 || percent === 100
  };
}
export const tangentFunctionChoices = Object.freeze({
  shiftedSquare: Object.freeze({
    first: 'f(x) = x + 1',
    second: 'g(y) = y²'
  }),
  cubicAffine: Object.freeze({
    first: 'f(x) = x³',
    second: 'g(y) = 2y − 1'
  }),
  squareSquare: Object.freeze({
    first: 'f(x) = x²',
    second: 'g(y) = y²'
  })
});
export function tangentSnapshot(input = 0, tangent = 1, choice = 'shiftedSquare', cotangent = 1) {
  finiteNumber(input, -3, 3, 'Input');
  finiteNumber(tangent, -2, 2, 'Tangent');
  finiteNumber(cotangent, -2, 2, 'Cotangent');
  if (!Object.hasOwn(tangentFunctionChoices, choice)) throw new RangeError('Unknown polynomial composition.');
  const firstValue = choice === 'shiftedSquare' ? input + 1 : choice === 'cubicAffine' ? input ** 3 : input ** 2;
  const firstDerivative = choice === 'shiftedSquare' ? 1 : choice === 'cubicAffine' ? 3 * input ** 2 : 2 * input;
  const secondValue = choice === 'cubicAffine' ? 2 * firstValue - 1 : firstValue ** 2;
  const secondDerivative = choice === 'cubicAffine' ? 2 : 2 * firstValue;
  const derivativeAtWrongPoint = choice === 'cubicAffine' ? 2 : 2 * input;
  const intermediateTangent = firstDerivative * tangent;
  const outputTangent = secondDerivative * intermediateTangent;
  const intermediateCotangent = secondDerivative * cotangent;
  const inputCotangent = firstDerivative * intermediateCotangent;
  const directDerivative = choice === 'shiftedSquare' ? 2 * (input + 1) : choice === 'cubicAffine' ? 6 * input ** 2 : 4 * input ** 3;
  return {
    input,
    tangent,
    cotangent,
    firstValue,
    secondValue,
    firstDerivative,
    secondDerivative,
    intermediateTangent,
    outputTangent,
    intermediateCotangent,
    inputCotangent,
    directDerivative,
    composedDerivative: firstDerivative * secondDerivative,
    wrongDerivative: firstDerivative * derivativeAtWrongPoint,
    outputPairing: cotangent * outputTangent,
    inputPairing: inputCotangent * tangent
  };
}
function indices(mask, size) {
  return Array.from({
    length: size
  }, (_, index) => index).filter(index => (mask & 1 << index) !== 0);
}
export function adjunctionSnapshot(sourceMask = 3, targetMask = 1, mapping = [0, 0, 1, 2]) {
  integer(sourceMask, 0, 15, 'Source selection');
  integer(targetMask, 0, 7, 'Target selection');
  validateFiniteMap(mapping, 4, 3);
  const source = indices(sourceMask, 4);
  const target = indices(targetMask, 3);
  const image = [...new Set(source.map(point => mapping[point]))].sort((a, b) => a - b);
  const preimage = mapping.flatMap((imagePoint, point) => target.includes(imagePoint) ? [point] : []);
  const saturation = mapping.flatMap((imagePoint, point) => image.includes(imagePoint) ? [point] : []);
  const imageFailures = image.filter(point => !target.includes(point));
  const sourceFailures = source.filter(point => !preimage.includes(point));
  return {
    mapping: mapping.slice(),
    source,
    target,
    image,
    preimage,
    saturation,
    imageFailures,
    sourceFailures,
    imageContained: imageFailures.length === 0,
    sourceContained: sourceFailures.length === 0
  };
}

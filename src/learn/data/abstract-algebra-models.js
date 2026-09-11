// D4 has eight elements. gh applies h first, then g; every action is on the left.
export const squareElementNames = Object.freeze(['e', 'r', 'r²', 'r³', 's', 'rs', 'r²s', 'r³s']);
export const squareVertices = Object.freeze([[1, 0], [0, 1], [-1, 0], [0, -1]].map(Object.freeze));
export const squareGroup = Object.freeze(Array.from({
  length: 8
}, (_, index) => index));
export const rotationGroup = Object.freeze([0, 1, 2, 3]);
function integer(value, minimum, maximum, name) {
  if (!Number.isInteger(value) || value < minimum || value > maximum) {
    throw new RangeError(`${name} must be an integer from ${minimum} to ${maximum}.`);
  }
  return value;
}
function denseArray(values, length, name) {
  if (!Array.isArray(values) || values.length !== length || Array.from({
    length
  }, (_, index) => index).some(index => !Object.hasOwn(values, index))) {
    throw new TypeError(`${name} must contain exactly ${length} explicit entries.`);
  }
  return values;
}
const mod4 = value => (value % 4 + 4) % 4;
const element = value => integer(value, 0, 7, 'Square element');
export function composeSquare(g, h) {
  element(g);
  element(h);
  const k = g % 4;
  const b = Math.floor(g / 4);
  const l = h % 4;
  const c = Math.floor(h / 4);
  return mod4(k + (b ? -l : l)) + 4 * (b ^ c);
}
export function inverseSquare(g) {
  element(g);
  const b = Math.floor(g / 4);
  return mod4((b ? 1 : -1) * (g % 4)) + 4 * b;
}
export function vertexImage(g, vertex) {
  element(g);
  integer(vertex, 0, 3, 'Vertex');
  return mod4(g % 4 + (g < 4 ? vertex : -vertex));
}
export function squarePermutation(g) {
  return Array.from({
    length: 4
  }, (_, vertex) => vertexImage(g, vertex));
}
export function squareMatrix(g) {
  const imageX = squareVertices[vertexImage(g, 0)];
  const imageY = squareVertices[vertexImage(g, 1)];
  return [[imageX[0], imageY[0]], [imageX[1], imageY[1]]];
}
export function moveSquareData(g, values) {
  element(g);
  denseArray(values, 4, 'Square data');
  const result = Array(4);
  values.forEach((value, source) => {
    result[vertexImage(g, source)] = value;
  });
  return result;
}
export function compositionState(g = 1, h = 4, vertex = 1) {
  element(g);
  element(h);
  integer(vertex, 0, 3, 'Tracked vertex');
  const labels = ['A', 'B', 'C', 'D'];
  const product = composeSquare(g, h);
  const reversed = composeSquare(h, g);
  return {
    g,
    h,
    vertex,
    product,
    reversed,
    inverse: inverseSquare(product),
    firstRoute: [vertex, vertexImage(h, vertex), vertexImage(product, vertex)],
    secondRoute: [vertex, vertexImage(g, vertex), vertexImage(reversed, vertex)],
    firstStates: [labels, moveSquareData(h, labels), moveSquareData(product, labels)],
    secondStates: [labels, moveSquareData(g, labels), moveSquareData(reversed, labels)],
    table: squareGroup.map(row => squareGroup.map(column => composeSquare(row, column)))
  };
}
export function generatedSubgroup(generators) {
  if (!Array.isArray(generators) || generators.length > 8) throw new TypeError('Use at most eight generators.');
  denseArray(generators, generators.length, 'Generators').forEach(element);
  const found = new Set([0]);
  const frontier = [0];
  while (frontier.length) {
    const current = frontier.shift();
    for (const generator of generators) {
      const next = composeSquare(generator, current);
      if (!found.has(next)) {
        found.add(next);
        frontier.push(next);
      }
    }
  }
  return [...found].sort((a, b) => a - b);
}
function checkedSubgroup(subgroup) {
  if (!Array.isArray(subgroup) || subgroup.length < 1 || subgroup.length > 8) throw new TypeError('Use a nonempty square subgroup.');
  denseArray(subgroup, subgroup.length, 'Subgroup').forEach(element);
  const values = [...new Set(subgroup)].sort((a, b) => a - b);
  if (!values.includes(0) || values.some(g => values.some(h => !values.includes(composeSquare(g, inverseSquare(h)))))) {
    throw new RangeError('The selected set is not a subgroup.');
  }
  return values;
}
export function leftCosets(subgroup) {
  const values = checkedSubgroup(subgroup);
  const cosets = [];
  for (const g of squareGroup) {
    const members = values.map(h => composeSquare(g, h)).sort((a, b) => a - b);
    if (!cosets.some(coset => coset.members.join(',') === members.join(','))) cosets.push({
      representative: g,
      members
    });
  }
  return cosets;
}
export function permutationCycles(g) {
  element(g);
  const visited = new Set();
  const cycles = [];
  for (let start = 0; start < 4; start += 1) {
    if (visited.has(start)) continue;
    const cycle = [];
    let next = start;
    do {
      cycle.push(next);
      visited.add(next);
      next = vertexImage(g, next);
    } while (next !== start);
    cycles.push(cycle);
  }
  return cycles;
}
function checkedColors(colors, colorCount = 3) {
  denseArray(colors, 4, 'Vertex colors').forEach(value => integer(value, 0, colorCount - 1, 'Color'));
  return colors;
}
export function orbitState(colors = [1, 0, 1, 0], rotationsOnly = false) {
  checkedColors(colors);
  if (typeof rotationsOnly !== 'boolean') throw new TypeError('Choose whether only rotations are allowed.');
  const group = rotationsOnly ? rotationGroup : squareGroup;
  const states = [];
  for (const g of group) {
    const transformed = moveSquareData(g, colors);
    let entry = states.find(row => row.colors.every((value, index) => value === transformed[index]));
    if (!entry) {
      entry = {
        colors: transformed,
        transformations: []
      };
      states.push(entry);
    }
    entry.transformations.push(g);
  }
  const stabilizer = group.filter(g => moveSquareData(g, colors).every((value, index) => value === colors[index]));
  return {
    colors: [...colors],
    group: [...group],
    states,
    stabilizer,
    orbitSize: states.length
  };
}
export function fixedColoringCount(colorCount = 2, rotationsOnly = false) {
  integer(colorCount, 2, 4, 'Number of colors');
  if (typeof rotationsOnly !== 'boolean') throw new TypeError('Choose whether only rotations are allowed.');
  const group = rotationsOnly ? rotationGroup : squareGroup;
  const rows = group.map(g => ({
    g,
    cycles: permutationCycles(g),
    fixed: colorCount ** permutationCycles(g).length
  }));
  const numerator = rows.reduce((sum, row) => sum + row.fixed, 0);
  return {
    colorCount,
    group: [...group],
    rows,
    numerator,
    orbitCount: numerator / group.length,
    totalColorings: colorCount ** 4
  };
}
export function cosetProductState(normal = true, representativeIndex = 0) {
  if (typeof normal !== 'boolean') throw new TypeError('Choose a normal or nonnormal subgroup.');
  const subgroup = normal ? [...rotationGroup] : [0, 4];
  integer(representativeIndex, 0, subgroup.length - 1, 'Representative index');
  const representative = subgroup[representativeIndex];
  const product = composeSquare(representative, 1);
  const cosets = leftCosets(subgroup);
  const output = cosets.find(coset => coset.members.includes(product));
  const expected = cosets.find(coset => coset.members.includes(1));
  return {
    normal,
    subgroup,
    representative,
    product,
    cosets,
    output: output.members,
    expected: expected.members,
    sameOutput: output === expected
  };
}
export function permutationMatrix(g) {
  const permutation = squarePermutation(g);
  return Array.from({
    length: 4
  }, (_, destination) => permutation.map(image => Number(image === destination)));
}
function quarterNumber(value, bound, name) {
  if (!Number.isFinite(value) || Math.abs(value) > bound || !Number.isInteger(value * 4)) {
    throw new RangeError(`${name} must be a multiple of 0.25 between ${-bound} and ${bound}.`);
  }
  return value;
}
function checkedMatrix(matrix) {
  denseArray(matrix, 4, 'Matrix').forEach(row => denseArray(row, 4, 'Matrix row').forEach(value => {
    if (!Number.isFinite(value) || Math.abs(value) > 32) throw new RangeError('Weights must be finite and between −32 and 32.');
  }));
  return matrix;
}
export function matrixProduct(a, b) {
  return a.map(row => b[0].map((_, column) => row.reduce((sum, value, k) => sum + value * b[k][column], 0)));
}
export function applyMatrix(matrix, values) {
  return matrix.map(row => row.reduce((sum, value, index) => sum + value * values[index], 0));
}
export function averageSquareMap(matrix, rotationsOnly = false) {
  checkedMatrix(matrix);
  if (typeof rotationsOnly !== 'boolean') throw new TypeError('Choose whether only rotations are allowed.');
  const group = rotationsOnly ? rotationGroup : squareGroup;
  const result = Array.from({
    length: 4
  }, () => Array(4).fill(0));
  for (const g of group) {
    const term = matrixProduct(matrixProduct(permutationMatrix(inverseSquare(g)), matrix), permutationMatrix(g));
    term.forEach((row, i) => row.forEach((value, j) => {
      result[i][j] += value / group.length;
    }));
  }
  return result;
}
export const rawSquareWeights = Object.freeze([[2, 2, 4, 8], [3, 5, 7, 9], [6, 10, 11, 12], [13, 14, 15, 16]].map(Object.freeze));
export function equivarianceState(values = [1, 2, 4, 8], mode = 'tied', g = 4, weights = [2, 0.5, 0]) {
  denseArray(values, 4, 'Sensor readings').forEach(value => quarterNumber(value, 20, 'Reading'));
  denseArray(weights, 3, 'Tied weights').forEach(value => quarterNumber(value, 4, 'Tied weight'));
  element(g);
  if (!['raw', 'rotations', 'averaged', 'tied', 'shift'].includes(mode)) throw new RangeError('Select a listed map.');
  const [self, neighbor, opposite] = weights;
  const tiedRow = [self, neighbor, opposite, neighbor];
  const matrix = mode === 'raw' ? rawSquareWeights.map(row => [...row]) : mode === 'rotations' ? averageSquareMap(rawSquareWeights, true) : mode === 'averaged' ? averageSquareMap(rawSquareWeights) : mode === 'shift' ? permutationMatrix(1) : Array.from({
    length: 4
  }, (_, row) => Array.from({
    length: 4
  }, (_, column) => tiedRow[mod4(column - row)]));
  const transformedInput = moveSquareData(g, values);
  const mapped = applyMatrix(matrix, values);
  const mapThenTransform = moveSquareData(g, mapped);
  const transformThenMap = applyMatrix(matrix, transformedInput);
  const difference = transformThenMap.map((value, index) => value - mapThenTransform[index]);
  const commutators = squareGroup.map(transform => {
    const p = permutationMatrix(transform);
    const left = matrixProduct(matrix, p);
    const right = matrixProduct(p, matrix);
    const residual = left.map((row, i) => row.map((value, j) => value - right[i][j]));
    return {
      g: transform,
      residual,
      maximum: Math.max(...residual.flat().map(Math.abs))
    };
  });
  return {
    values: [...values],
    g,
    mode,
    matrix,
    mapped,
    transformedInput,
    mapThenTransform,
    transformThenMap,
    difference,
    commutators,
    poolBefore: mapped.reduce((sum, value) => sum + value, 0) / 4,
    poolAfter: transformThenMap.reduce((sum, value) => sum + value, 0) / 4,
    inputSpecificDefect: Math.max(...difference.map(Math.abs)),
    allInputDefect: Math.max(...commutators.map(row => row.maximum))
  };
}
export function parseSensorReadings(text) {
  if (typeof text !== 'string') throw new TypeError('Enter four comma-separated readings.');
  const parts = text.split(',').map(value => value.trim());
  if (parts.length !== 4 || parts.some(value => !/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$/.test(value))) throw new RangeError('Enter four decimal readings separated by commas.');
  const values = parts.map(Number);
  values.forEach(value => quarterNumber(value, 20, 'Reading'));
  return values;
}
export function modularState(modulus = 6, multiplier = 2, target = 4) {
  integer(modulus, 2, 12, 'Modulus');
  integer(multiplier, 0, modulus - 1, 'Multiplier');
  integer(target, 0, modulus - 1, 'Target');
  const outputs = Array.from({
    length: modulus
  }, (_, value) => multiplier * value % modulus);
  const preimages = Array.from({
    length: modulus
  }, (_, value) => outputs.flatMap((output, index) => output === value ? [index] : []));
  return {
    modulus,
    multiplier,
    target,
    outputs,
    preimages,
    solutions: preimages[target],
    inverse: outputs.indexOf(1),
    injective: new Set(outputs).size === modulus
  };
}
export function formatAlgebraNumber(value) {
  if (!Number.isFinite(value)) throw new RangeError('Expected a finite display value.');
  return String(Object.is(value, -0) ? 0 : value);
}

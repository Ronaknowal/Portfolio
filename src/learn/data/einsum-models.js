export const contractionPresets = {
  product: {
    name: 'Matrix product',
    expression: 'ik,kj->ij',
    operands: [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
  },
  elementwise: {
    name: 'Keep both shared labels',
    expression: 'ij,ij->ij',
    operands: [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
  },
  diagonal: {
    name: 'Keep the diagonal',
    expression: 'ii->i',
    operands: [[[1, 2], [3, 4]]]
  },
  trace: {
    name: 'Sum the diagonal',
    expression: 'ii->',
    operands: [[[1, 2], [3, 4]]]
  },
  columns: {
    name: 'Sum over rows',
    expression: 'ij->j',
    operands: [[[1, 2], [3, 4]]]
  },
  transpose: {
    name: 'Reorder axes',
    expression: 'ij->ji',
    operands: [[[1, 2], [3, 4]]]
  },
  outer: {
    name: 'Every pair of vector entries',
    expression: 'i,j->ij',
    operands: [[1, 2], [3, 4]]
  },
  dot: {
    name: 'One shared vector index',
    expression: 'i,i->',
    operands: [[1, 2], [3, 4]]
  }
};
function arrayShape(value) {
  if (!Array.isArray(value)) {
    if (!Number.isFinite(value)) throw new Error('Use finite numeric entries.');
    return [];
  }
  if (!value.length || value.length > 4) throw new Error('This inspector uses axis sizes from 1 to 4.');
  const child = arrayShape(value[0]);
  for (const item of value) {
    if (JSON.stringify(arrayShape(item)) !== JSON.stringify(child)) throw new Error('Use rectangular arrays.');
  }
  return [value.length, ...child];
}
function assignments(labels, sizes) {
  let rows = [{}];
  for (const label of labels) {
    rows = rows.flatMap(row => Array.from({
      length: sizes[label]
    }, (_, index) => ({
      ...row,
      [label]: index
    })));
  }
  return rows;
}

// A bounded explicit interpreter for the inspector, not a replacement for NumPy.
export function inspectContraction(expression, operands) {
  if (typeof expression !== 'string' || !Array.isArray(operands) || operands.length < 1 || operands.length > 2) {
    throw new Error('Use one or two operands and an explicit expression.');
  }
  const compact = expression.replace(/\s/g, '');
  if (!/^[a-z]+(?:,[a-z]+)?->[a-z]*$/.test(compact)) {
    throw new Error('Use lower-case labels and an explicit -> output; this inspector does not parse ellipses.');
  }
  const [input, output] = compact.split('->');
  const inputs = input.split(',');
  if (inputs.length !== operands.length) throw new Error('The comma-separated terms must match the number of displayed operands.');
  if (new Set(output).size !== output.length) throw new Error('Each output label must occur only once.');
  const shapes = operands.map(arrayShape);
  const sizes = {};
  inputs.forEach((labels, operandIndex) => {
    if (labels.length !== shapes[operandIndex].length) throw new Error('Give one label to each displayed operand axis.');
    const localSizes = {};
    [...labels].forEach((label, axis) => {
      const size = shapes[operandIndex][axis];
      if (localSizes[label] !== undefined && localSizes[label] !== size) throw new Error('Repeated axes within one operand must have equal lengths.');
      localSizes[label] = size;
      if (sizes[label] !== undefined && sizes[label] !== size && sizes[label] !== 1 && size !== 1) {
        throw new Error('Shared labels across operands need matching lengths or a singleton length.');
      }
      sizes[label] = Math.max(sizes[label] ?? 1, size);
    });
  });
  if ([...output].some(label => sizes[label] === undefined)) throw new Error('Every output label must occur in an input.');
  const labels = Object.keys(sizes);
  if (labels.length > 4) throw new Error('This inspector is limited to four distinct labels.');
  const reduced = labels.filter(label => !output.includes(label));
  const cells = assignments([...output], sizes).map(free => {
    const terms = assignments(reduced, sizes).map(dummy => {
      const index = {
        ...free,
        ...dummy
      };
      const sources = inputs.map((operandLabels, operandIndex) => {
        const coordinates = [...operandLabels].map((label, axis) => shapes[operandIndex][axis] === 1 ? 0 : index[label]);
        const value = coordinates.reduce((array, coordinate) => array[coordinate], operands[operandIndex]);
        return {
          operand: operandIndex,
          coordinates,
          value
        };
      });
      return {
        indices: index,
        sources,
        product: sources.reduce((product, source) => product * source.value, 1)
      };
    });
    return {
      coordinates: [...output].map(label => free[label]),
      terms,
      value: terms.reduce((total, term) => total + term.product, 0)
    };
  });
  return {
    expression: compact,
    inputs,
    output,
    shapes,
    sizes,
    reduced,
    outputShape: [...output].map(label => sizes[label]),
    cells
  };
}
export const attentionFixtures = [{
  queries: [[1, 0], [0, 1]],
  keys: [[1, 0], [0, 1], [1, 1]],
  values: [[2, 0], [0, 4], [2, 2]]
}, {
  queries: [[1, 1], [-1, 0]],
  keys: [[2, 0], [0, 1], [-1, 1]],
  values: [[10, 0], [0, 6], [4, 2]]
}];
export function attentionRow(batch, query, allowed, scaled = true) {
  if (!Number.isInteger(batch) || batch < 0 || batch >= attentionFixtures.length || !Number.isInteger(query) || query < 0 || query > 1) {
    throw new Error('Select an available batch and query.');
  }
  if (!Array.isArray(allowed) || allowed.length !== 3 || allowed.some(value => typeof value !== 'boolean') || typeof scaled !== 'boolean') {
    throw new Error('Supply one boolean mask value per key and a boolean scaling choice.');
  }
  if (!allowed.some(Boolean)) throw new Error('No allowed key: a normalized attention row is undefined. Allow a key to continue.');
  const fixture = attentionFixtures[batch];
  const q = fixture.queries[query];
  const scores = fixture.keys.map(key => key.reduce((sum, value, d) => sum + q[d] * value, 0) / (scaled ? Math.sqrt(q.length) : 1));
  const maximum = Math.max(...scores.filter((_, key) => allowed[key]));
  const exponentials = scores.map((score, key) => allowed[key] ? Math.exp(score - maximum) : 0);
  const denominator = exponentials.reduce((sum, value) => sum + value, 0);
  const weights = exponentials.map(value => value / denominator);
  const rows = scores.map((score, key) => ({
    key,
    products: fixture.keys[key].map((value, d) => value * q[d]),
    score,
    weight: weights[key],
    value: fixture.values[key],
    contribution: fixture.values[key].map(value => value * weights[key])
  }));
  const context = fixture.values[0].map((_, d) => rows.reduce((sum, row) => sum + row.contribution[d], 0));
  return {
    q,
    rows,
    context,
    denominator,
    maximum
  };
}
export function contractionOrders(dimensions) {
  if (!Array.isArray(dimensions) || dimensions.length !== 4 || dimensions.some(value => !Number.isInteger(value) || value < 1 || value > 100)) {
    throw new Error('Use four integer axis lengths between 1 and 100.');
  }
  const [a, b, c, d] = dimensions;
  return {
    left: {
      firstShape: [a, c],
      firstSize: a * c,
      firstCost: a * b * c,
      lastCost: a * c * d,
      total: a * b * c + a * c * d
    },
    right: {
      firstShape: [b, d],
      firstSize: b * d,
      firstCost: b * c * d,
      lastCost: a * b * d,
      total: b * c * d + a * b * d
    },
    outputShape: [a, d]
  };
}
const multiply = (left, right) => left.map(row => right[0].map((_, j) => row.reduce((sum, value, k) => sum + value * right[k][j], 0)));
const transpose = matrix => matrix[0].map((_, j) => matrix.map(row => row[j]));
const matvec = (matrix, vector) => matrix.map(row => row.reduce((sum, value, j) => sum + value * vector[j], 0));
const dot = (left, right) => left.reduce((sum, value, index) => sum + value * right[index], 0);
export const basisPresets = {
  identity: [[1, 0], [0, 1]],
  shear: [[1, 1], [0, 1]],
  stretch: [[2, 0], [0, 1]],
  rotation: [[0, -1], [1, 0]]
};
export const basisVectors = [[3, 2], [-1, 2], [2, -1], [0, 0]];
export function basisChange(basis, vector) {
  if (!Object.hasOwn(basisPresets, basis) || !Array.isArray(vector) || vector.length !== 2 || vector.some(value => !Number.isFinite(value))) {
    throw new Error('Use a listed basis and a finite two-component vector.');
  }
  const S = basisPresets[basis];
  const determinant = S[0][0] * S[1][1] - S[0][1] * S[1][0];
  const inverse = [[S[1][1] / determinant, -S[0][1] / determinant], [-S[1][0] / determinant, S[0][0] / determinant]];
  const coordinates = matvec(inverse, vector);
  const functional = [2, -1];
  const newFunctional = matvec(transpose(S), functional);
  const map = [[2, 1], [0, 1]];
  const newMap = multiply(multiply(inverse, map), S);
  const metric = multiply(transpose(S), S);
  return {
    S,
    coordinates,
    newFunctional,
    newMap,
    metric,
    reconstructed: matvec(S, coordinates),
    mapped: matvec(map, vector),
    mappedNew: matvec(newMap, coordinates),
    measurement: dot(functional, vector),
    newMeasurement: dot(newFunctional, coordinates),
    normSquared: dot(vector, vector),
    coordinateSquares: dot(coordinates, coordinates),
    metricNormSquared: dot(coordinates, matvec(metric, coordinates))
  };
}
export const einsumNumber = value => Math.abs(value) < 1e-12 ? '0' : Number(value.toFixed(6)).toString();

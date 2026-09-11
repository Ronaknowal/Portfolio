export const rosterNames = ['Ada', 'Bo', 'Cam', 'Dee'];
function integerInRange(value, lower, upper, name) {
  if (!Number.isInteger(value) || value < lower || value > upper) {
    throw new RangeError(`${name} must be an integer from ${lower} through ${upper}.`);
  }
}
function isBooleanSquareMatrix(matrix, size) {
  if (!Array.isArray(matrix) || matrix.length !== size) return false;
  for (let rowIndex = 0; rowIndex < size; rowIndex += 1) {
    if (!Object.hasOwn(matrix, rowIndex)) return false;
    const row = matrix[rowIndex];
    if (!Array.isArray(row) || row.length !== size) return false;
    for (let columnIndex = 0; columnIndex < size; columnIndex += 1) {
      if (!Object.hasOwn(row, columnIndex) || typeof row[columnIndex] !== 'boolean') return false;
    }
  }
  return true;
}
export function selectRoster(firstMask, secondMask, operation) {
  integerInRange(firstMask, 0, 15, 'Training membership');
  integerInRange(secondMask, 0, 15, 'Badge membership');
  const operations = {
    intersection: (first, second) => first && second,
    union: (first, second) => first || second,
    difference: (first, second) => first && !second,
    reverseDifference: (first, second) => second && !first,
    complement: first => !first,
    symmetricDifference: (first, second) => first !== second
  };
  if (!Object.hasOwn(operations, operation)) throw new RangeError('Unknown set operation.');
  const members = rosterNames.map((name, index) => {
    const first = Boolean(firstMask & 1 << index);
    const second = Boolean(secondMask & 1 << index);
    return {
      name,
      index,
      first,
      second,
      region: first ? second ? 'both' : 'first' : second ? 'second' : 'neither',
      selected: operations[operation](first, second)
    };
  });
  return {
    members,
    first: members.filter(member => member.first).map(member => member.name),
    second: members.filter(member => member.second).map(member => member.name),
    selected: members.filter(member => member.selected).map(member => member.name)
  };
}
export const propositionLabels = {
  p: 'P',
  q: 'Q',
  notP: '¬P',
  notQ: '¬Q',
  implication: 'P → Q',
  converse: 'Q → P',
  contrapositive: '¬Q → ¬P',
  equivalence: 'P ↔ Q',
  conjunction: 'P ∧ Q',
  disjunction: 'P ∨ Q',
  exclusiveOr: 'P xor Q'
};
export function evaluateProposition(name, p, q) {
  if (typeof p !== 'boolean' || typeof q !== 'boolean') throw new TypeError('Use Boolean truth values.');
  switch (name) {
    case 'p':
      return p;
    case 'q':
      return q;
    case 'notP':
      return !p;
    case 'notQ':
      return !q;
    case 'implication':
      return !p || q;
    case 'converse':
      return !q || p;
    case 'contrapositive':
      return q || !p;
    case 'equivalence':
      return p === q;
    case 'conjunction':
      return p && q;
    case 'disjunction':
      return p || q;
    case 'exclusiveOr':
      return p !== q;
    default:
      throw new RangeError('Unknown proposition.');
  }
}
export function inspectArgument(premises, conclusion) {
  if (!Array.isArray(premises) || premises.length > 6) throw new RangeError('Use at most six premises.');
  for (const name of [...premises, conclusion]) {
    if (!Object.hasOwn(propositionLabels, name)) throw new RangeError('Unknown proposition.');
  }
  const worlds = [false, true].flatMap(p => [false, true].map(q => {
    const premiseValues = premises.map(name => evaluateProposition(name, p, q));
    const admitted = premiseValues.every(Boolean);
    const conclusionValue = evaluateProposition(conclusion, p, q);
    return {
      p,
      q,
      premiseValues,
      admitted,
      conclusion: conclusionValue,
      counterexample: admitted && !conclusionValue
    };
  }));
  return {
    worlds,
    valid: worlds.every(world => !world.counterexample),
    consistent: worlds.some(world => world.admitted),
    counterexamples: worlds.filter(world => world.counterexample)
  };
}
export function inspectQuantifiers(matrix, rowCount, columnCount) {
  integerInRange(rowCount, 0, 3, 'Job count');
  integerInRange(columnCount, 0, 3, 'Reviewer count');
  if (!isBooleanSquareMatrix(matrix, 3)) {
    throw new TypeError('Supply a three-by-three Boolean board; selected domain sizes may be zero.');
  }
  const rows = Array.from({
    length: rowCount
  }, (_, index) => index);
  const columns = Array.from({
    length: columnCount
  }, (_, index) => index);
  const rowWitnesses = rows.map(row => columns.filter(column => matrix[row][column]));
  const rowFailures = rows.filter(row => rowWitnesses[row].length === 0);
  const commonWitnesses = columns.filter(column => rows.every(row => matrix[row][column]));
  const columnFailures = columns.map(column => rows.find(row => !matrix[row][column]) ?? null);
  const eachHasSomeone = rowFailures.length === 0;
  const someoneCoversAll = commonWitnesses.length > 0;
  const everyoneCoversAll = rows.every(row => columns.every(column => matrix[row][column]));
  const someoneCoversSomething = rows.some(row => columns.some(column => matrix[row][column]));
  return {
    rows,
    columns,
    rowWitnesses,
    rowFailures,
    commonWitnesses,
    columnFailures,
    eachHasSomeone,
    someoneCoversAll,
    everyoneCoversAll,
    someoneCoversSomething
  };
}
export function inspectRelation(size, pairs) {
  integerInRange(size, 0, 6, 'Relation universe size');
  if (!Array.isArray(pairs) || pairs.length > 100) throw new RangeError('Use at most 100 relation pairs.');
  const matrix = Array.from({
    length: size
  }, () => Array(size).fill(false));
  for (const pair of pairs) {
    if (!Array.isArray(pair) || pair.length !== 2 || !Object.hasOwn(pair, 0) || !Object.hasOwn(pair, 1)) {
      throw new TypeError('A relation pair has two indices.');
    }
    integerInRange(pair[0], 0, size - 1, 'Relation index');
    integerInRange(pair[1], 0, size - 1, 'Relation index');
    matrix[pair[0]][pair[1]] = true;
  }
  const indices = Array.from({
    length: size
  }, (_, index) => index);
  const failures = {
    reflexive: null,
    symmetric: null,
    transitive: null,
    antisymmetric: null
  };
  for (const first of indices) {
    if (!matrix[first][first] && failures.reflexive === null) failures.reflexive = [first];
    for (const second of indices) {
      if (matrix[first][second] && !matrix[second][first] && failures.symmetric === null) {
        failures.symmetric = [first, second];
      }
      if (first !== second && matrix[first][second] && matrix[second][first] && failures.antisymmetric === null) {
        failures.antisymmetric = [first, second];
      }
      for (const third of indices) {
        if (matrix[first][second] && matrix[second][third] && !matrix[first][third] && failures.transitive === null) {
          failures.transitive = [first, second, third];
        }
      }
    }
  }
  const properties = Object.fromEntries(Object.entries(failures).map(([name, failure]) => [name, failure === null]));
  const equivalence = properties.reflexive && properties.symmetric && properties.transitive;
  const partialOrder = properties.reflexive && properties.antisymmetric && properties.transitive;
  let classes = null;
  if (equivalence) {
    const seen = new Set();
    classes = [];
    for (const first of indices) {
      if (seen.has(first)) continue;
      const members = indices.filter(second => matrix[first][second]);
      members.forEach(member => seen.add(member));
      classes.push(members);
    }
  }
  const covers = partialOrder ? indices.flatMap(first => indices.filter(second => first !== second && matrix[first][second] && !indices.some(middle => middle !== first && middle !== second && matrix[first][middle] && matrix[middle][second])).map(second => [first, second])) : null;
  const minimal = partialOrder ? indices.filter(first => indices.every(second => first === second || !matrix[second][first])) : null;
  const maximal = partialOrder ? indices.filter(first => indices.every(second => first === second || !matrix[first][second])) : null;
  const least = partialOrder ? indices.filter(first => indices.every(second => matrix[first][second])) : null;
  const greatest = partialOrder ? indices.filter(first => indices.every(second => matrix[second][first])) : null;
  const levels = partialOrder ? Array(size).fill(0) : null;
  if (partialOrder) {
    // Every strict order chain has at most size-1 edges. Relax only cover edges.
    for (let pass = 0; pass < size; pass += 1) {
      for (const [lower, upper] of covers) levels[upper] = Math.max(levels[upper], levels[lower] + 1);
    }
  }
  return {
    matrix,
    properties,
    failures,
    equivalence,
    partialOrder,
    classes,
    covers,
    minimal,
    maximal,
    least,
    greatest,
    levels
  };
}
export function relationPreset(name) {
  let labels;
  let predicate;
  switch (name) {
    case 'moduloThree':
      labels = ['0', '1', '2', '3', '4', '5'];
      predicate = (first, second) => first % 3 === second % 3;
      break;
    case 'nearby':
      labels = ['0', '1', '2', '3'];
      predicate = (first, second) => Math.abs(first - second) <= 1;
      break;
    case 'identity':
      labels = ['0', '1', '2', '3'];
      predicate = (first, second) => first === second;
      break;
    case 'divisors':
      labels = ['1', '2', '3', '6'];
      predicate = (first, second) => Number(labels[second]) % Number(labels[first]) === 0;
      break;
    case 'noLeast':
      labels = ['2', '3', '6'];
      predicate = (first, second) => Number(labels[second]) % Number(labels[first]) === 0;
      break;
    default:
      throw new RangeError('Unknown relation preset.');
  }
  const pairs = labels.flatMap((_, first) => labels.flatMap((__, second) => predicate(first, second) ? [[first, second]] : []));
  return {
    labels,
    pairs
  };
}
export function oddSquareStep(size) {
  integerInRange(size, 0, 7, 'Square stage');
  const cells = Array.from({
    length: size + 1
  }, (_, row) => Array.from({
    length: size + 1
  }, (__, column) => ({
    row,
    column,
    added: row === size || column === size
  }))).flat();
  return {
    size,
    cells,
    before: size * size,
    added: 2 * size + 1,
    after: (size + 1) ** 2
  };
}
export function diagonalSubset(matrix) {
  if (!Array.isArray(matrix) || matrix.length > 8 || !isBooleanSquareMatrix(matrix, matrix.length)) {
    throw new TypeError('Use a square Boolean membership table of size zero through eight.');
  }
  const subset = matrix.map((row, index) => !row[index]);
  return {
    subset,
    members: subset.flatMap((included, index) => included ? [index] : []),
    differences: matrix.map((row, index) => ({
      row: index,
      member: index,
      proposed: row[index],
      constructed: subset[index],
      unequal: row[index] !== subset[index]
    }))
  };
}

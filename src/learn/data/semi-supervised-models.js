/** Small, deterministic teaching models. These fixtures are not banknote fits. */
export const graphDefault = {
  nodes: [{ name: 'A', label: 0 }, { name: 'B', label: null }, { name: 'C', label: null }, { name: 'D', label: 1 }],
  edges: [[0, 1, 1], [1, 2, 1], [2, 3, 1]],
};
export const coTrainingRows = [
  { views: ['red', 'round'], label: 0 }, { views: ['blue', 'square'], label: 1 },
  { views: ['red', 'triangle'], label: null }, { views: ['green', 'triangle'], label: null },
  { views: ['orange', 'square'], label: null }, { views: ['orange', 'hexagon'], label: null },
  { views: ['red', 'square'], label: null },
];

export function solveLinear(matrix, rhs) {
  const rows = matrix.map((row, i) => [...row, rhs[i]]);
  for (let column = 0; column < rows.length; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < rows.length; row += 1) {
      if (Math.abs(rows[row][column]) > Math.abs(rows[pivot][column])) pivot = row;
    }
    if (Math.abs(rows[pivot][column]) < 1e-13) throw new Error('The anchored system is numerically singular.');
    [rows[column], rows[pivot]] = [rows[pivot], rows[column]];
    const diagonal = rows[column][column];
    for (let j = column; j <= rows.length; j += 1) rows[column][j] /= diagonal;
    for (let row = 0; row < rows.length; row += 1) {
      if (row === column) continue;
      const factor = rows[row][column];
      for (let j = column; j <= rows.length; j += 1) rows[row][j] -= factor * rows[column][j];
    }
  }
  return rows.map(row => row.at(-1));
}

function finiteNumber(value, minimum, maximum, description) {
  if (String(value).trim() === '') throw new Error(`Enter ${description}.`);
  const number = Number(value);
  if (!Number.isFinite(number) || number < minimum || number > maximum) {
    throw new Error(`${description} must be between ${minimum} and ${maximum}.`);
  }
  return number;
}
const parseLabel = (value) => {
  if (value === '?' || value === '') return null;
  if (value === '0' || value === '1') return Number(value);
  throw new Error('A label must be 0, 1 or ? for unknown.');
};
const lines = value => value.trim() ? value.trim().split(/\r?\n/).filter(row => row.trim()) : [];

export function parseGraph(nodeText, edgeText) {
  const nodes = lines(nodeText).map(row => {
    const parts = row.split(',').map(value => value.trim());
    if (parts.length !== 2 || !/^[A-Za-z][A-Za-z0-9]{0,7}$/.test(parts[0])) {
      throw new Error('Use one node per line: name,label. Names: 1–8 letters/digits, starting with a letter.');
    }
    return { name: parts[0], label: parseLabel(parts[1]) };
  });
  if (nodes.length < 2 || nodes.length > 10) throw new Error('Use 2–10 nodes.');
  if (new Set(nodes.map(node => node.name)).size !== nodes.length) throw new Error('Node names must be unique.');
  const pairs = new Set();
  const edges = lines(edgeText).map(row => {
    const parts = row.split(',').map(value => value.trim());
    if (parts.length !== 3) throw new Error('Use one edge per line: from,to,weight.');
    const first = nodes.findIndex(node => node.name === parts[0]);
    const second = nodes.findIndex(node => node.name === parts[1]);
    if (first < 0 || second < 0 || first === second) throw new Error('Each edge needs two different existing nodes.');
    const key = [first, second].sort((a, b) => a - b).join(':');
    if (pairs.has(key)) throw new Error('List each undirected edge only once.');
    pairs.add(key);
    return [first, second, finiteNumber(parts[2], 0, 5, 'Edge weight')];
  }).filter(edge => edge[2] > 0);
  return { nodes, edges };
}

export function propagateGraph(graph, mode = 'hard', alpha = 0.8) {
  if (!['hard', 'soft'].includes(mode)) throw new Error('Choose hard or soft propagation.');
  if (mode === 'soft') alpha = finiteNumber(alpha, 0, 0.99, 'Alpha');
  const count = graph.nodes.length;
  const weights = Array.from({ length: count }, () => Array(count).fill(0));
  graph.edges.forEach(([a, b, weight]) => { weights[a][b] = weight; weights[b][a] = weight; });
  const degree = weights.map(row => row.reduce((sum, weight) => sum + weight, 0));
  const anchored = Array(count).fill(false);
  const seen = new Set();
  for (let start = 0; start < count; start += 1) {
    if (seen.has(start)) continue;
    const component = [start];
    seen.add(start);
    for (let position = 0; position < component.length; position += 1) {
      weights[component[position]].forEach((weight, neighbor) => {
        if (weight > 0 && !seen.has(neighbor)) { seen.add(neighbor); component.push(neighbor); }
      });
    }
    if (component.some(index => graph.nodes[index].label !== null)) component.forEach(index => { anchored[index] = true; });
  }
  const symmetric = weights.map((row, i) => row.map((weight, j) => degree[i] && degree[j] ? (weight / Math.sqrt(degree[i])) / Math.sqrt(degree[j]) : 0));
  let equilibrium;
  let trace;
  let residual = 0;
  let converged = true;
  if (mode === 'hard') {
    const unknown = graph.nodes.flatMap((node, index) => node.label === null && anchored[index] ? [index] : []);
    // Row-normalized harmonic equations preserve a common rescaling of all
    // conductances. Small but well-conditioned graphs are not singular.
    const matrix = unknown.map(i => unknown.map(j => Number(i === j) - weights[i][j] / degree[i]));
    const rhs = unknown.map(i => weights[i].reduce((sum, weight, j) => sum + (weight / degree[i]) * (graph.nodes[j].label ?? 0), 0));
    const solution = solveLinear(matrix, rhs);
    equilibrium = graph.nodes.map((node, index) => !anchored[index] ? null : node.label ?? solution[unknown.indexOf(index)]);
    residual = unknown.reduce((largest, index, row) => Math.max(largest, Math.abs(degree[index] * (matrix[row].reduce((sum, entry, j) => sum + entry * solution[j], 0) - rhs[row]))), 0);
    let state = graph.nodes.map((node, index) => !anchored[index] ? null : node.label ?? 0.5);
    trace = [state];
    for (let step = 0; step < 5000; step += 1) {
      const next = graph.nodes.map((node, i) => !anchored[i] ? null : node.label ?? weights[i].reduce((sum, weight, j) => sum + (weight / degree[i]) * (state[j] ?? 0), 0));
      trace.push(next);
      const error = next.reduce((largest, value, i) => Math.max(largest, Math.abs((value ?? 0) - (equilibrium[i] ?? 0))), 0);
      state = next;
      if (error < 1e-10) break;
      if (step === 4999) converged = false;
    }
  } else {
    const evidence = graph.nodes.map(node => [Number(node.label === 0), Number(node.label === 1)]);
    const matrix = symmetric.map((row, i) => row.map((value, j) => Number(i === j) - alpha * value));
    const solved = [0, 1].map(classIndex => solveLinear(matrix, evidence.map(row => (1 - alpha) * row[classIndex])));
    equilibrium = graph.nodes.map((_, i) => [solved[0][i], solved[1][i]]);
    residual = Math.max(...matrix.map((row, i) => Math.max(...[0, 1].map(c => Math.abs(row.reduce((sum, entry, j) => sum + entry * equilibrium[j][c], 0) - (1 - alpha) * evidence[i][c])))));
    let state = evidence;
    trace = [state];
    for (let step = 0; step < 5000; step += 1) {
      const next = evidence.map((row, i) => [0, 1].map(c => alpha * symmetric[i].reduce((sum, weight, j) => sum + weight * state[j][c], 0) + (1 - alpha) * row[c]));
      trace.push(next);
      const error = Math.max(...next.flatMap((row, i) => row.map((value, c) => Math.abs(value - equilibrium[i][c]))));
      state = next;
      if (error < 1e-10) break;
      if (step === 4999) converged = false;
    }
  }
  const readout = state => mode === 'hard' ? state : state.map(row => row[0] + row[1] > 0 ? row[1] / (row[0] + row[1]) : null);
  return { weights, degree, anchored, symmetric, equilibrium, scores: readout(equilibrium), trace, readout, residual, converged, mode };
}

export function scoreSide(value) {
  if (value === null) return 'unavailable';
  if (Math.abs(value - 0.5) < 1e-9) return 'tie';
  return value < 0.5 ? 'below' : 'above';
}

export function parsePrototype(observedText, poolText, threshold, query) {
  const observed = lines(observedText).map(row => {
    const parts = row.split(',').map(value => value.trim());
    if (parts.length !== 2 || !['0', '1'].includes(parts[1])) throw new Error('Observed rows use x,class with class 0 or 1.');
    return { x: finiteNumber(parts[0], -10, 10, 'Observed coordinate'), label: Number(parts[1]) };
  });
  if (observed.length < 2 || observed.length > 10 || new Set(observed.map(point => point.label)).size !== 2) throw new Error('Use 2–10 observed points with at least one example of each class.');
  const pool = poolText.trim() ? poolText.split(/[\s,]+/).map(value => finiteNumber(value, -10, 10, 'Unlabeled coordinate')) : [];
  if (pool.length > 20) throw new Error('Use at most 20 unlabeled points.');
  return { observed, pool, threshold: finiteNumber(threshold, 0.5, 0.99, 'Threshold'), query: finiteNumber(query, -10, 10, 'Query coordinate') };
}

export function prototypeScores(x, means) {
  const logits = means.map(mean => -((x - mean) ** 2));
  const maximum = Math.max(...logits);
  const exponentials = logits.map(value => Math.exp(value - maximum));
  const total = exponentials[0] + exponentials[1];
  return exponentials.map(value => value / total);
}

export function prototypeTraining({ observed, pool, threshold, query }, maxRounds = 20) {
  const points = observed.map((point, index) => ({ ...point, origin: 'observed', id: `L${index}` }));
  let remaining = pool.map((x, index) => ({ x, id: `U${index}`, label: null, origin: 'unlabeled' }));
  const meansOf = () => [0, 1].map(label => {
    const group = points.filter(point => point.label === label);
    return group.reduce((sum, point) => sum + point.x, 0) / group.length;
  });
  const initial = meansOf();
  const history = [];
  for (let round = 1; round <= maxRounds; round += 1) {
    const before = meansOf();
    const proposals = remaining.map(point => {
      const probabilities = prototypeScores(point.x, before);
      const label = probabilities[1] > probabilities[0] ? 1 : 0;
      return { ...point, label, probabilities, accepted: probabilities[label] >= threshold, round, origin: `pseudo round ${round}` };
    });
    const accepted = proposals.filter(point => point.accepted);
    points.push(...accepted);
    const ids = new Set(accepted.map(point => point.id));
    remaining = remaining.filter(point => !ids.has(point.id));
    history.push({ round, before, proposals, accepted, after: meansOf(), points: points.map(point => ({ ...point })), remaining: [...remaining] });
    if (!accepted.length || !remaining.length) break;
  }
  const final = meansOf();
  const boundary = means => Math.abs(means[0] - means[1]) < 1e-12 ? null : (means[0] + means[1]) / 2;
  const initialBoundary = boundary(initial);
  const finalBoundary = boundary(final);
  const movement = initialBoundary === null || finalBoundary === null ? 'unavailable' : Math.abs(finalBoundary - initialBoundary) < 1e-9 ? 'unchanged' : finalBoundary > initialBoundary ? 'right' : 'left';
  const queryClass = means => {
    const probabilities = prototypeScores(query, means);
    return Math.abs(probabilities[0] - probabilities[1]) < 1e-12 ? 'tie' : probabilities[1] > probabilities[0] ? 1 : 0;
  };
  const beforeQuery = queryClass(initial);
  const afterQuery = queryClass(final);
  return { initial, final, initialBoundary, finalBoundary, movement, history, points, remaining, beforeQuery, afterQuery,
    queryChange: beforeQuery === 'tie' || afterQuery === 'tie' ? 'tie' : beforeQuery === afterQuery ? 'no' : 'yes',
    capped: history.length === maxRounds && remaining.length > 0 && history.at(-1).accepted.length > 0 };
}

export function parseViews(text) {
  const rows = lines(text).map(row => {
    const parts = row.split('|').map(value => value.trim());
    if (parts.length !== 3 || parts.slice(0, 2).some(value => value.length < 1 || value.length > 20)) throw new Error('Use view1 | view2 | label, with 1–20 characters per category and label 0, 1 or ?.');
    return { views: parts.slice(0, 2), label: parseLabel(parts[2]) };
  });
  if (rows.length < 2 || rows.length > 20) throw new Error('Use 2–20 paired rows.');
  return rows;
}

export function learnCategoryRules(rows, labels, view) {
  const received = new Map();
  rows.forEach((row, index) => {
    if (labels[index] === null) return;
    const category = row.views[view];
    if (!received.has(category)) received.set(category, new Set());
    received.get(category).add(labels[index]);
  });
  return Object.fromEntries([...received].filter(([, values]) => values.size === 1).map(([category, values]) => [category, [...values][0]]));
}

export function categoricalCoTraining(rows, maxRounds = 8) {
  const labels = [rows.map(row => row.label), rows.map(row => row.label)];
  const provenance = [0, 1].map(() => rows.map(row => row.label === null ? null : { kind: 'observed' }));
  const history = [];
  for (let round = 1; round <= maxRounds; round += 1) {
    const rulesBefore = labels.map((values, view) => learnCategoryRules(rows, values, view));
    const predictions = [0, 1].map(view => rows.map(row => Object.hasOwn(rulesBefore[view], row.views[view]) ? rulesBefore[view][row.views[view]] : null));
    const offers = [];
    const conflicts = [];
    rows.forEach((row, index) => {
      if (predictions[0][index] !== null && predictions[1][index] !== null && predictions[0][index] !== predictions[1][index]) { conflicts.push(index); return; }
      for (const donor of [0, 1]) {
        const recipient = 1 - donor;
        if (predictions[donor][index] !== null && labels[recipient][index] === null) {
          const category = row.views[donor];
          const evidenceRows = rows.flatMap((candidate, j) => candidate.views[donor] === category && labels[donor][j] !== null ? [{ row: j, provenance: provenance[donor][j] }] : []);
          offers.push({ row: index, donor, recipient, label: predictions[donor][index], donorCategory: category, recipientCategory: row.views[recipient], evidenceRows,
            newCategory: !Object.hasOwn(rulesBefore[recipient], row.views[recipient]) });
        }
      }
    });
    for (const offer of offers) {
      labels[offer.recipient][offer.row] = offer.label;
      provenance[offer.recipient][offer.row] = { kind: 'pseudo', round, donor: offer.donor, evidenceRows: offer.evidenceRows };
    }
    const rulesAfter = labels.map((values, view) => learnCategoryRules(rows, values, view));
    const learnedKeys = new Set(offers.filter(offer => offer.newCategory && Object.hasOwn(rulesAfter[offer.recipient], offer.recipientCategory)).map(offer => `${offer.recipient}:${offer.recipientCategory}`));
    history.push({ round, rulesBefore, predictions, offers, conflicts, labelsAfter: labels.map(values => [...values]), rulesAfter,
      provenance: provenance.map(values => [...values]), newRules: learnedKeys.size });
    if (!offers.length) break;
  }
  return { history, labels, provenance, rules: labels.map((values, view) => learnCategoryRules(rows, values, view)),
    capped: history.length === maxRounds && history.at(-1).offers.length > 0 };
}

export function coTrainingAnswer(result, rows, target, view) {
  const predictions = [0, 1].map(index => Object.hasOwn(result.rules[index], rows[target].views[index]) ? result.rules[index][rows[target].views[index]] : null);
  if (predictions[0] !== null && predictions[1] !== null && predictions[0] !== predictions[1]) return 'conflict';
  return predictions[view] === null ? 'unknown' : String(predictions[view]);
}

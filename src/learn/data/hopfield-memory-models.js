// Rows are memory records; vectors are column coordinates conceptually.
// Max-subtracted log-sum-exp keeps both tiny class mass and large scores meaningful.
export const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
export const norm = a => Math.sqrt(dot(a, a));
export const logSumExp = a => { const m = Math.max(...a); return m + Math.log(a.reduce((s, x) => s + Math.exp(x - m), 0)); };
export const softmax = a => { const z = logSumExp(a); return a.map(x => Math.exp(x - z)); };
export const weightedRows = (rows, weights) => rows[0].map((_, j) => rows.reduce((s, row, i) => s + weights[i] * row[j], 0));
export const unit = a => { const d = Math.max(norm(a), 1e-12); return a.map(x => x / d); };
export const distance = (a, b) => norm(a.map((x, i) => x - b[i]));
export function binaryStore(patterns) {
  const d = patterns[0].length;
  return Array.from({ length: d }, (_, i) => Array.from({ length: d }, (_, j) => i === j ? 0 : patterns.reduce((s, row) => s + row[i] * row[j], 0) / d));
}
export const binaryEnergy = (w, state) => -.5 * state.reduce((s, x, i) => s + x * dot(w[i], state), 0);
export function binaryRecall(patterns, cue, order = [0, 1, 2, 3], sweeps = 8) {
  const w = binaryStore(patterns), trace = [{ state: [...cue], energy: binaryEnergy(w, cue), coordinate: null, field: null, votes: [], changed: false }];
  let state = [...cue], converged = false;
  for (let sweep = 0; sweep < sweeps; sweep++) {
    let changes = 0;
    for (const coordinate of order) {
      const votes = w[coordinate].map((x, j) => x * state[j]), field = votes.reduce((a, b) => a + b, 0), before = state[coordinate];
      state = [...state]; state[coordinate] = field === 0 ? before : Math.sign(field);
      const changed = state[coordinate] !== before; changes += Number(changed);
      trace.push({ state, energy: binaryEnergy(w, state), coordinate, field, votes, changed, sweep: sweep + 1, before });
    }
    if (!changes) { converged = true; break; }
  }
  const same = (a, b) => a.every((x, i) => x === b[i]);
  const category = patterns.some(row => same(row, state)) ? 'stored pattern' : patterns.some(row => same(row.map(x => -x), state)) ? 'inverse of a stored pattern' : 'neither a stored pattern nor its inverse';
  return { w, trace, final: state, converged, category };
}
export const continuousEnergy = (memories, q, beta) => .5 * dot(q, q) - logSumExp(memories.map(x => beta * dot(x, q))) / beta;
export function continuousRead(memories, q, beta) {
  const scores = memories.map(x => dot(x, q)), logits = scores.map(x => beta * x), weights = softmax(logits), read = weightedRows(memories, weights);
  const jacobian = q.map((_, i) => q.map((__, j) => beta * (memories.reduce((s, x, k) => s + weights[k] * x[i] * x[j], 0) - read[i] * read[j])));
  const [a, b] = jacobian[0], c = jacobian[1][1], sensitivity = .5 * (a + c + Math.hypot(a - c, 2 * b));
  return { scores, logits, weights, read, energy: continuousEnergy(memories, q, beta), nextEnergy: continuousEnergy(memories, read, beta), jacobian, sensitivity, step: distance(q, read) };
}
export function continuousTrace(memories, q, beta, steps = 12) {
  const trace = [];
  for (let i = 0; i <= steps; i++) { const result = continuousRead(memories, q, beta); trace.push({ q, ...result }); q = result.read; }
  return trace;
}
export function associate(records, q, beta) {
  const scores = records.map(r => dot(r.key, q)), logits = scores.map(x => beta * x), weights = softmax(logits);
  return { scores, logits, weights, output: weightedRows(records.map(r => r.value), weights), contributions: records.map((r, i) => r.value.map(x => x * weights[i])) };
}
export function queryTraining(keys, q, target, beta, rate) {
  const weights = softmax(keys.map(k => beta * dot(k, q))), gradient = q.map((_, j) => beta * (keys.reduce((s, k, i) => s + weights[i] * k[j], 0) - keys[target][j]));
  return { weights, loss: -Math.log(weights[target]), gradient, next: q.map((x, j) => x - rate * gradient[j]) };
}
export function prepareDigitBank(bank, model) {
  const pixels = bank.memory.map(row => row.pixels.map(x => x / 16));
  const project = x => model.projection ? model.projection.map(row => dot(row, x)) : x;
  return { pixels, keys: pixels.map(x => unit(project(x))), project };
}
export function digitRead(rawPixels, bank, model, prepared = prepareDigitBank(bank, model)) {
  const query = unit(prepared.project(rawPixels.map(x => x / 16))), scores = prepared.keys.map(k => dot(k, query)), logits = scores.map(x => model.beta * x), logZ = logSumExp(logits), logWeights = logits.map(x => x - logZ), weights = logWeights.map(Math.exp);
  const logClasses = Array.from({ length: 10 }, (_, label) => logSumExp(logWeights.filter((_, i) => bank.memory[i].label === label))), classes = logClasses.map(Math.exp), maximum = Math.max(...classes), winners = classes.flatMap((x, i) => Math.abs(x - maximum) < 1e-12 ? [i] : []);
  const top = weights.map((weight, index) => ({ weight, index })).sort((a, b) => b.weight - a.weight || a.index - b.index).slice(0, 3);
  return { query, scores, logits, weights, logClasses, classes, winners, top, readPixels: weightedRows(prepared.pixels, weights) };
}
export const digitMse = (a, b) => a.reduce((s, x, i) => s + (x - b[i]) ** 2, 0) / a.length;
export const occludeDigit = pixels => pixels.map((x, i) => i % 8 === 3 || i % 8 === 4 ? 0 : x);
export function convexHull(points) {
  const rows = [...new Map(points.map(x => [x.join(','), x])).values()].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (rows.length < 3) return rows;
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]), half = list => { const h = []; for (const p of list) { while (h.length > 1 && cross(h.at(-2), h.at(-1), p) <= 0) h.pop(); h.push(p); } return h; };
  return [...half(rows).slice(0, -1), ...half([...rows].reverse()).slice(0, -1)];
}
// Split cells into triangles: each linear triangle has at most one segment per
// level, avoiding an ambiguous marching-squares saddle connectivity decision.
export function energyContours(memories, beta, extent = 2, nx = 61, ny = 41) {
  const grid = Array.from({ length: ny }, (_, j) => Array.from({ length: nx }, (_, i) => { const p = [-extent + 2 * extent * i / (nx - 1), -extent + 2 * extent * j / (ny - 1)]; return { p, e: continuousEnergy(memories, p, beta) }; }));
  const energies = grid.flat().map(x => x.e), low = Math.min(...energies), high = Math.max(...energies), levels = Array.from({ length: 8 }, (_, i) => low + (high - low) * (i + 1) / 9);
  return levels.map(level => { const segments = []; for (let j = 0; j < ny - 1; j++) for (let i = 0; i < nx - 1; i++) {
    const [a, b, c, d] = [grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]];
    for (const t of [[a, b, c], [a, c, d]]) { const hits = []; for (let k = 0; k < 3; k++) { const x = t[k], y = t[(k + 1) % 3]; if ((x.e < level) !== (y.e < level)) { const u = (level - x.e) / (y.e - x.e); hits.push(x.p.map((v, z) => v + u * (y.p[z] - v))); } } if (hits.length === 2) segments.push(hits); }
  } return { level, segments }; });
}
export function marginBound(count, beta, gap, maximumNorm) {
  const a = (count - 1) * Math.exp(-beta * gap);
  return { targetMass: 1 / (1 + a), errorBound: 2 * maximumNorm * a / (1 + a) };
}
export function storageCost(count, dimension, bytes) {
  return { bytes: BigInt(count) * BigInt(dimension) * BigInt(bytes), scores: BigInt(count), multiplications: BigInt(count) * BigInt(dimension) };
}
export const parityMemories = [[-1, -1, -1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]];
export const higherOrderEnergy = (state, power) => -parityMemories.reduce((sum, x) => sum + dot(x, state) ** power, 0);

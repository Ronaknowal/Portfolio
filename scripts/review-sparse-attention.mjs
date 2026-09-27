// Independent contract probes: new inputs and different formulations, no refit.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { sparseRead, causalEdges, graphReach, featureMemory, randomFeatureRead, projectedRead, blockOccupancy, forecastTrajectory } from '../src/learn/data/sparse-attention-models.js';

let state = 849031, comparisons = 0, maximumError = 0;
const random = () => ((state = (Math.imul(state, 1664525) + 1013904223) >>> 0) / 2 ** 32);
const vector = (n, positive = false) => Array.from({ length: n }, () => positive ? .1 + random() * 2 : random() * 4 - 2);
const inner = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
function close(a, b, tolerance = 2e-11) {
  if (Array.isArray(a)) { assert.equal(a.length, b.length); a.forEach((v, i) => close(v, b[i], tolerance)); return; }
  const error = Math.abs(a - b); maximumError = Math.max(maximumError, error); comparisons++;
  assert.ok(error <= tolerance, `${a} != ${b}; error ${error}`);
}
function weighted(weights, values) {
  const total = weights.reduce((a, b) => a + b, 0);
  return values[0].map((_, k) => values.reduce((s, v, i) => s + v[k] * weights[i] / total, 0));
}
for (let trial = 0; trial < 100; trial++) {
  const n = 2 + trial % 11, logits = vector(n), values = Array.from({ length: n }, () => vector(3));
  const legal = Array.from({ length: n }, (_, i) => i === 0 || random() > .5);
  const result = sparseRead(logits, values, legal);
  const weights = logits.map((x, i) => legal[i] ? Math.exp(x) : 0);
  close(result.output, weighted(weights, values));
  close(result.output, sparseRead(logits.map(x => x + 1000), values, legal).output);
  close(sparseRead(logits, values.map(() => [-3, 2, .5]), legal).output, [-3, 2, .5]);
  const error = Math.hypot(...result.output.map((x, i) => x - result.dense[i]));
  const bound = 2 * Math.max(...values.map(v => Math.hypot(...v))) * result.removedMass;
  assert.ok(error <= bound + 1e-12);
  assert.equal(sparseRead(logits, values, legal.map(() => false)).output, null);

  const query = vector(3, true), keys = Array.from({ length: n }, () => vector(3, true));
  const memory = featureMemory(query, keys, values);
  close(memory.output, weighted(keys.map(k => inner(query, k)), values));
  close(memory.output, featureMemory(query.map(x => x * 7), keys, values).output);
  close(memory.output, featureMemory(query, keys.toReversed(), values.toReversed()).output);
  assert.equal(featureMemory([0, 0, 0], keys, values).output, null);
  const evicted = featureMemory(query, keys.slice(1), values.slice(1));
  close(evicted.matrix, memory.matrix.map((row, i) => row.map((x, j) => x - keys[0][i] * values[0][j])));

  const mask = Array.from({ length: n }, (_, r) => Array.from({ length: n }, (_, c) => c <= r && random() > .55));
  const source = trial % n, depth = 1 + trial % 4;
  let reachable = new Set([source]);
  for (let d = 0; d < depth; d++) {
    const next = new Set(reachable);
    for (const donor of reachable) for (let receiver = donor; receiver < n; receiver++) if (mask[receiver][donor]) next.add(receiver);
    reachable = next;
  }
  const graph = graphReach(mask, source, depth);
  assert.deepEqual(graph.stages.at(-1), Array.from({ length: n }, (_, i) => reachable.has(i)));
  graph.paths.filter(Boolean).forEach(path => path.slice(1).forEach((r, i) => assert.ok(r === path[i] || mask[r][path[i]])));
  const w = 1 + trial % 8, edges = causalEdges(n, w), m = Math.min(n, w);
  assert.equal(edges.flat().filter(Boolean).length, m * (m + 1) / 2 + (n - m) * m);

  const coefficients = vector(n), p = trial % n, before = projectedRead(values.map(v => v[0]), coefficients, p);
  const changed = values.map((v, i) => v[0] + (i > p ? 3 : 0));
  close(projectedRead(changed, coefficients, p).prefix, before.prefix);
}
for (let trial = 0; trial < 35; trial++) {
  const q = Array.from({ length: 5 }, () => vector(2)), k = Array.from({ length: 5 }, () => vector(2));
  const v = Array.from({ length: 5 }, () => vector(3)), projection = Array.from({ length: 13 }, () => vector(2));
  const result = randomFeatureRead(q, k, v, projection);
  // Unstabilized direct exponential products: separate arithmetic trust root
  // from the runtime's logarithmic common-key/query scaling.
  q.forEach((query, receiver) => {
    const weights = k.map((key, donor) => donor > receiver ? 0 : projection.reduce((s, w) => s + Math.exp(inner(w, query) - inner(query, query) / 2) * Math.exp(inner(w, key) - inner(key, key) / 2), 0));
    close(result.output[receiver], weighted(weights, v));
  });
  const changed = structuredClone(k); changed[4][0] += .5;
  close(randomFeatureRead(q, changed, v, projection).output.slice(0, 4), result.output.slice(0, 4));
  const mask = Array.from({ length: 8 }, () => Array.from({ length: 8 }, () => random() > .8));
  for (const size of [1, 2, 4, 8]) {
    const occupied = new Set();
    mask.forEach((row, r) => row.forEach((on, c) => { if (on) occupied.add(`${Math.floor(r / size)},${Math.floor(c / size)}`); }));
    assert.equal(blockOccupancy(mask, size).candidates, occupied.size * size ** 2);
  }
}
let networkCases = 0;
for (const mode of ['dense', 'window', 'kernel']) {
  const { weights } = JSON.parse(fs.readFileSync(`public/learn-assets/sparse-linear-attention-variants/${mode}-forecast.json`, 'utf8'));
  for (const n of [8, 17, 33, 44]) {
    const points = Array.from({ length: n }, () => [random(), random()]);
    const full = forecastTrajectory(weights, points, mode);
    let cache = null, chunked = [];
    for (let start = 0; start < n; start += 3) {
      const part = forecastTrajectory(weights, points.slice(start, start + 3), mode, { cache, start });
      cache = part.cache; chunked.push(...part.predictions);
    }
    close(full.predictions, chunked);
    const changed = structuredClone(points); changed[n - 3][0] = 1 - changed[n - 3][0];
    close(full.predictions.slice(0, n - 3), forecastTrajectory(weights, changed, mode).predictions.slice(0, n - 3));
    if (mode === 'window') {
      const old = structuredClone(points); old[0][1] = 1 - old[0][1];
      close(full.forecast, forecastTrajectory(weights, old, mode).forecast);
      assert.equal(cache[0].keys.length, 5);
    }
    networkCases++;
  }
}
const report = { passed: true, comparisons, maximumError, networkCases, groups: [
  '100 independent sparse renormalization/oracle, shift, constant, bound and empty-set cases',
  '100 direct feature pair/state, rescaling, permutation and evicted outer-product cases',
  '100 set-propagation causal graph/path and exact window-count cases',
  '35 fixed-projection direct exponential product oracles plus future-key invariance',
  '35 masks at four block sizes against occupied-coordinate set counts',
  '12 new bounded network inputs: three-token chunk equality, future intervention and window remote null',
], limitations: 'Complementary arithmetic/structural probes; author native PyTorch parity remains the separate network oracle. No fresh fits or GPU benchmark.' };
fs.writeFileSync('docs/teaching/deep-learning-completion/sparse-linear-attention-variants/independent-model-checks.json', JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report));

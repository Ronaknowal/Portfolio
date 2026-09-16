import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import { U_POINTS, radiusGraph, gaussianRow, directedMembership, fuzzyUnion, umapConnection, idealPair, tsneObjective, neighborOrder, neighborAudit, queryNeighbors, ripsComplex } from '../src/learn/data/manifold-models.js';
import { MANIFOLD_DIGITS, MANIFOLD_FIXTURES } from '../src/learn/data/manifold-data.js';

let assertions = 0;
const close = (actual, expected, tolerance = 1e-10) => { assertions += 1; assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} differs from ${expected} by ${Math.abs(actual - expected)}`); };
const equal = (actual, expected) => { assertions += 1; assert.deepEqual(actual, expected); };
const rejects = fn => { assertions += 1; assert.throws(fn); };
const groups = [];
function checked(name, fn) { fn(); groups.push(name); }

checked('Radius graph: route, shortcuts, disconnected, edits, ties, independent shortest-path oracle', () => {
  equal(radiusGraph(U_POINTS, .75).distance, null);
  equal(radiusGraph(U_POINTS, 1).path, ['A', 'B', 'C', 'D', 'E', 'F', 'G']);
  close(radiusGraph(U_POINTS, 1).distance, 6);
  close(radiusGraph(U_POINTS, 1.5).distance, 2 + 2 * Math.SQRT2);
  close(radiusGraph(U_POINTS, 2).distance, 2);
  close(radiusGraph(U_POINTS, 3).distance, 2);
  equal(radiusGraph(U_POINTS.map(p => p.id === 'D' ? { ...p, y: 3 } : p), 1).distance, null);
  close(radiusGraph(U_POINTS.map(p => p.id === 'D' ? { ...p, y: 3 } : p), 1, 'A', 'B').distance, 1);
  rejects(() => radiusGraph(U_POINTS.map(p => p.id === 'B' ? { ...p, y: 0 } : p), 1));
  rejects(() => radiusGraph(U_POINTS, NaN));
  rejects(() => radiusGraph(U_POINTS, 1, 'A', 'A'));
  // Floyd–Warshall is independent of production's simple-path enumeration.
  for (const radius of [.5, 1, 1.25, 1.5, 2, 2.75]) {
    const points = U_POINTS.map((p, i) => ({ ...p, x: p.x + (i === 3 ? .25 : 0) }));
    const distances = points.map((p, i) => points.map((q, j) => i === j ? 0 : Math.hypot(p.x - q.x, p.y - q.y) <= radius ? Math.hypot(p.x - q.x, p.y - q.y) : Infinity));
    for (let k = 0; k < 7; k += 1) for (let i = 0; i < 7; i += 1) for (let j = 0; j < 7; j += 1) distances[i][j] = Math.min(distances[i][j], distances[i][k] + distances[k][j]);
    for (let i = 0; i < 7; i += 1) for (let j = i + 1; j < 7; j += 1) {
      const graph = radiusGraph(points, radius, points[i].id, points[j].id);
      if (Number.isFinite(distances[i][j])) close(graph.distance, distances[i][j]); else equal(graph.distance, null);
    }
  }
  const diamond = [{ id: 'A', x: 0, y: 0 }, { id: 'B', x: 1, y: 1 }, { id: 'C', x: 1, y: -1 }, { id: 'D', x: 2, y: 0 }];
  equal(radiusGraph(diamond, 1.5, 'A', 'D').path, ['A', 'B', 'D']);
  equal(radiusGraph([...diamond].reverse(), 1.5, 'A', 'D').path, ['A', 'B', 'D']);
});

checked('Gaussian normalization, entropy, tiny weights, equal-distance and common-scale invariance', () => {
  for (const sigma of [.25, .5, 1, 2, 3]) {
    const row = gaussianRow([1, 2, 3], sigma);
    const independent = [1, 2, 3].map(d => Math.exp(-d * d / (2 * sigma * sigma)));
    const total = independent.reduce((a, b) => a + b, 0);
    row.probabilities.forEach((p, i) => close(p, independent[i] / total));
    close(row.probabilities.reduce((a, b) => a + b), 1);
    close(gaussianRow([2, 2, 2], sigma).perplexity, 3);
    gaussianRow([2, 4, 6], 2 * sigma).probabilities.forEach((p, i) => close(p, row.probabilities[i]));
  }
  close(gaussianRow([1, 2, 3], 1).probabilities[0], .8055124120405226);
  close(gaussianRow([1, 2, 3], 1).perplexity, MANIFOLD_FIXTURES.probability_rows.unequal['1'].perplexity);
  close(gaussianRow([6, 6, 6], .25).perplexity, 3);
  let previous = 0;
  for (let sigma = .25; sigma <= 3; sigma += .25) { const value = gaussianRow([.75, 2.75, 5.5], sigma).perplexity; assert.ok(value >= previous); previous = value; assertions += 1; }
  rejects(() => gaussianRow([1, 2, 3], 0));
  rejects(() => gaussianRow([-1, 2, 3], 1));
});

checked('UMAP directed scales, absent support, fuzzy union and ideal pair boundary cases', () => {
  const connection = umapConnection({ distance: 2, rhoI: 1, rhoJ: 1, sigmaI: 1 / Math.log(2), sigmaJ: 1 / Math.log(4) });
  close(connection.forward, .5); close(connection.reverse, .25); close(connection.weight, .625);
  close(directedMembership(.5, 1, 1), 1);
  close(directedMembership(2, 1, 1, false), 0);
  close(fuzzyUnion(.5, 0), .5); close(fuzzyUnion(0, 0), 0); close(fuzzyUnion(1, 1), 1);
  close(fuzzyUnion(.2, .9), .92);
  const calibration = [1, 1 + Math.log(2), 1 + Math.log(2)].map(d => directedMembership(d, 1, 1));
  close(calibration.reduce((a, b) => a + b), Math.log2(4));
  for (const separation of [.5, 1, 2]) close(idealPair(.625, separation).cost, MANIFOLD_FIXTURES.pair_objective_w_0_625?.[separation] ?? MANIFOLD_FIXTURES['pair_objective_w_0.625'][separation]);
  const optimum = idealPair(.625, 1).optimumSeparation;
  close(optimum, Math.sqrt(.6));
  assert.ok(idealPair(.625, optimum).cost < idealPair(.625, optimum - .1).cost); assertions += 1;
  assert.ok(idealPair(.625, optimum).cost < idealPair(.625, optimum + .1).cost); assertions += 1;
  close(idealPair(1, 0).cost, 0);
  equal(idealPair(0, 0).cost, Infinity);
  equal(idealPair(0, 1).optimumSeparation, Infinity);
  close(idealPair(1, 1).optimumSeparation, 0);
  rejects(() => idealPair(.5, -1)); rejects(() => directedMembership(1, 1, 0));
});

let gradientMaxError = 0;
checked('Exact t-SNE: normalized P/Q, independent finite differences, total force, rigid invariance and offline trace', () => {
  const fixture = MANIFOLD_FIXTURES.t_sne_tiny, P = fixture.P;
  const Y = fixture.initial_Y;
  close(P.flat().reduce((a, b) => a + b), 1);
  P.forEach((row, i) => row.forEach((value, j) => close(value, P[j][i])));
  const initial = tsneObjective(P, Y);
  close(initial.normalization, 4);
  close(initial.Q[0][1], .125);
  close(initial.cost, fixture.initial_cost);
  close(initial.pairContributions[0][1][0], 4 * (P[0][1] - .125) * .5);
  initial.gradient.forEach((row, i) => close(row[0], fixture.initial_gradient[i][0]));
  close(initial.gradient.flat().reduce((a, b) => a + b), 0);
  const independentCost = points => {
    const weights = points.map((p, i) => points.map((q, j) => i === j ? 0 : 1 / (1 + p.reduce((s, value, axis) => s + (value - q[axis]) ** 2, 0))));
    const total = weights.flat().reduce((a, b) => a + b);
    return P.reduce((sum, row, i) => sum + row.reduce((s, p, j) => s + (p ? p * (Math.log(p) - Math.log(weights[i][j]) + Math.log(total)) : 0), 0), 0);
  };
  for (const map of [Y, [[-.7, .3], [.2, -.4], [.8, .5], [1.5, -1.1]]]) {
    const { gradient, cost } = tsneObjective(P, map);
    close(cost, independentCost(map));
    map.forEach((row, i) => row.forEach((_, axis) => {
      const plus = map.map(p => [...p]), minus = map.map(p => [...p]); plus[i][axis] += 1e-6; minus[i][axis] -= 1e-6;
      const estimate = (independentCost(plus) - independentCost(minus)) / 2e-6;
      gradientMaxError = Math.max(gradientMaxError, Math.abs(estimate - gradient[i][axis]));
      close(estimate, gradient[i][axis], 1e-9);
    }));
    close(tsneObjective(P, map.map(row => row.map(x => 3 - x))).cost, cost);
  }
  for (const item of fixture.trace) close(tsneObjective(P, item.Y).cost, item.cost);
  equal(fixture.trace.map(item => item.step), [0, 1, 10, 50, 100, 200]);
  close(fixture.trace.at(-1).cost, fixture.final_cost);
  rejects(() => tsneObjective(P.map(row => row.map(p => p * 12)), Y));
});

checked('Stable source-ID neighbor audit: false ranks, nulls, query30 reversal and every native R metric', () => {
  const X = [[0], [1], [3], [7]], Y = [[0], [5], [1], [11]];
  const audit = neighborAudit(X, Y, 1, ['A', 'B', 'C', 'D']);
  close(audit.retention, 0); close(audit.trustworthiness, .5); equal(audit.falsePenalty, 4);
  close(neighborAudit(X, X, 1).retention, 1); close(neighborAudit(X, X, 1).trustworthiness, 1);
  close(neighborAudit(X, X.map(([x]) => [3 - x]), 1).retention, 1);
  equal(neighborOrder([[0], [1], [-1]], [10, 30, 20])[0].map(item => item.sourceRow), [20, 30]);
  const rows = MANIFOLD_DIGITS.rows, ids = rows.map(row => row.sourceRow), input = rows.map(row => row.pixels.map(p => p / 16));
  for (const layout of MANIFOLD_DIGITS.layouts) for (const k of [5, 10, 20]) close(neighborAudit(input, layout.coordinates, k, ids).retention, layout.metrics[k].retention, 1e-12);
  const query = key => queryNeighbors(rows, MANIFOLD_DIGITS.layouts.find(layout => layout.key === key).coordinates, 30, 10);
  equal(query('pca').count, 6); equal(query('tsne-p30-s7').count, 5);
  equal(query('pca').input.map(item => item.sourceRow), [0, 166, 229, 160, 36, 276, 140, 266, 178, 79]);
  equal(query('pca').map.map(item => item.sourceRow), [78, 229, 79, 306, 276, 0, 130, 266, 132, 166]);
  equal(query('tsne-p30-s7').map.map(item => item.sourceRow), [0, 276, 266, 286, 256, 130, 48, 166, 229, 10]);
  rejects(() => queryNeighbors(rows, [], 30, 10));
  rejects(() => neighborAudit(X, Y, 2));
});

checked('MDS Gram/distance/eigenvalue recovery, LLE recipe and filled-versus-unfilled Rips topology', () => {
  const { B, distances, coordinates } = MANIFOLD_FIXTURES.classical_mds;
  coordinates.forEach((x, i) => coordinates.forEach((y, j) => { close(Math.abs(x - y), distances[i][j]); close(x * y, B[i][j]); }));
  close(coordinates.reduce((sum, x) => sum + x * x, 0), 38 / 3);
  close(coordinates.reduce((a, b) => a + b), 0);
  const { weights, input, output } = MANIFOLD_FIXTURES.lle;
  close(weights[0] + weights[1], 1);
  close(weights[0] * input[0] + weights[1] * input[2], input[1]);
  close(weights[0] * output[0] + weights[1] * output[2], output[1]);
  close((4 / 7) * -2 + (3 / 7) * 12, 4);
  const square = MANIFOLD_FIXTURES.square_rips;
  equal(ripsComplex(square.points, .99).beta0, 4);
  const loop = ripsComplex(square.points, 1); equal(loop.edges.length, 4); equal(loop.triangles.length, 0); equal(loop.beta1, 1);
  const filled = ripsComplex(square.points, Math.SQRT2); equal(filled.edges.length, 6); equal(filled.triangles.length, 4); equal(filled.tetrahedra.length, 1); equal(filled.beta1, 0);
  const duplicate = ripsComplex(square.projected, 0); equal(duplicate.edges, [[0, 3], [1, 2]]); equal(duplicate.beta0, 2); equal(duplicate.beta1, 0);
  equal(ripsComplex(square.projected, 1).beta1, 0);
});

const files = ['src/learn/data/manifold-models.js', 'src/learn/data/manifold-data.js', 'scripts/verify-manifold-models.mjs'];
const evidence = { status: 'passed', generatedAt: new Date().toISOString(), command: 'node scripts/verify-manifold-models.mjs', groups, assertions, gradientMaxError, sourceHashes: Object.fromEntries(files.map(file => [file, crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex')])) };
fs.writeFileSync('docs/teaching/evidence/manifold-models.json', `${JSON.stringify(evidence, null, 2)}\n`);
console.log(`PASS: ${groups.length} model groups, ${assertions} assertions; gradient max error ${gradientMaxError}.`);

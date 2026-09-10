import fs from 'node:fs';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import * as model from '../src/learn/data/spectral-graph-models.js';
import { spectralGraphExamples } from '../src/learn/data/spectral-graph-examples.js';

const directory = 'scratch/spectral-verification';
fs.mkdirSync(directory, { recursive: true });
const cases = { spectra: [], cuts: [], clusters: [], filters: [], arbitrary: [] };
for (let i = 0; i <= 40; i++) {
  const weight = i / 20;
  cases.spectra.push({ weight, state: model.bridgeSpectrum(weight) });
  for (const kind of ['bridge', 'unequal']) cases.cuts.push({ weight, kind, state: model.cutSweep(weight, kind) });
  for (const signal of Object.keys(model.spectralSignals)) {
    for (const filter of ['heat', 'ridge']) {
      for (let a = 0; a <= 40; a++) {
        const amount = a / 4;
        const state = model.filterState(weight, signal, filter, amount);
        cases.filters.push({ weight, signal, filter, amount, output: state.output, gains: state.gains, discarded: state.discarded, energyBefore: state.energyBefore, energyAfter: state.energyAfter });
      }
    }
    for (let amount = 1; amount <= 6; amount++) {
      const state = model.filterState(weight, signal, 'cutoff', amount);
      cases.filters.push({ weight, signal, filter: 'cutoff', amount, state });
    }
  }
}
for (let i = 0; i <= 75; i++) for (const seed of ['spread', 'nearby']) cases.clusters.push({ weight: i / 50, seed, state: model.clusteringState(i / 50, seed) });
let rng = 42;
function random() {
  rng = (Math.imul(1664525, rng) + 1013904223) >>> 0;
  return rng / 2 ** 32;
}
for (let trial = 0; trial < 240; trial++) {
  const n = 2 + trial % 11;
  const edges = [];
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (random() < 0.25) edges.push([i, j, [1e-6, 0.1, 1, 3, 100][Math.floor(random() * 5)]]);
  const graph = model.graphFromEdges(n, edges);
  cases.arbitrary.push({ graph, spectrum: model.symmetricSpectrum(graph.laplacian), normalized: model.symmetricSpectrum(graph.normalized) });
}
const invalid = [
  () => model.graphFromEdges(1, []), () => model.graphFromEdges(13, []), () => model.graphFromEdges(3, [[0, 0, 1]]),
  () => model.graphFromEdges(3, [[0, 3, 1]]), () => model.graphFromEdges(3, [[0, 1, -1]]), () => model.graphFromEdges(3, [[0, 1, 1e-20]]),
  () => model.graphFromEdges(3, [[0, 1, 1], [1, 0, 1]]), () => model.graphFromEdges(3, [[0, 1, Infinity]]),
  () => model.symmetricSpectrum([[1, 2], [0, 1]]), () => model.symmetricSpectrum([[1, NaN], [NaN, 1]]),
  () => model.bridgeSpectrum(-1), () => model.bridgeSpectrum(3), () => model.cutSweep(0.2, 'missing'),
  () => model.cutMetrics(model.bridgeSpectrum(), []), () => model.cutMetrics(model.bridgeSpectrum(), [0, 0]),
  () => model.clusteringState(2), () => model.clusteringState(1, 'missing'),
  () => model.filterState(1, 'toString'), () => model.filterState(1, 'spike', 'missing'),
  () => model.filterState(1, 'spike', 'cutoff', 2.5), () => model.filterState(1, 'spike', 'heat', -1),
];
invalid.forEach(fn => assert.throws(fn));
const source = fs.readFileSync('scratch/spectral-authoring/original-lesson.jsx', 'utf8');
const original = source.match(/<CodeBlock language="python">\{`([\s\S]*?)`\}<\/CodeBlock>/)[1];
assert.equal(spectralGraphExamples[0].code, original);
fs.writeFileSync(`${directory}/cases.json`, JSON.stringify({ ...cases, examples: spectralGraphExamples, invalid: invalid.length }));
const python = process.env.LESSON_PYTHON || 'scratch/lesson-tools/Scripts/python.exe';
const result = spawnSync(python, ['scripts/verify-spectral-graph-native.py'], { encoding: 'utf8', maxBuffer: 4_000_000 });
process.stdout.write(result.stdout || '');
process.stderr.write(result.stderr || '');
if (result.status !== 0) process.exit(result.status || 1);

import assert from 'node:assert/strict';
import { mkdir, writeFile } from 'node:fs/promises';
import { weightedProbeState, spectralSketchState, sketchSpectra, rowSketchState, traceProbeState, traceMatrices } from '../src/learn/data/randomized-linear-algebra-models.js';

const probes = [];
for (let first = -2; first <= 2; first += 1) {
  for (let second = -2; second <= 2; second += 1) {
    for (let third = -2; third <= 2; third += 1) probes.push(weightedProbeState([first, second, third]));
  }
}
const spectra = [];
for (const preset of Object.keys(sketchSpectra)) {
  for (let rank = 1; rank <= 4; rank += 1) {
    for (let oversampling = 0; oversampling <= 6 - rank; oversampling += 1) {
      for (let iterations = 0; iterations <= 3; iterations += 1) {
        for (const seed of [1, 7, 42, 99]) spectra.push(spectralSketchState(preset, rank, oversampling, iterations, seed));
      }
    }
  }
}
const rows = Array.from({ length: 256 }, (_, mask) => rowSketchState(
  Array.from({ length: 8 }, (_, index) => index).filter(index => mask & (1 << index))));
const traces = [];
for (const preset of Object.keys(traceMatrices)) {
  for (const seed of [1, 7, 42, 99, 999]) {
    for (let count = 0; count <= 32; count += 1) traces.push(traceProbeState(preset, seed, count));
  }
}
for (const invalid of [() => weightedProbeState([1, 2]), () => weightedProbeState([1, 2, NaN]),
  () => weightedProbeState([1, 2, 3]), () => spectralSketchState('missing', 2, 2, 1, 7),
  () => spectralSketchState('fast', 0, 2, 1, 7), () => spectralSketchState('fast', 2, 5, 1, 7),
  () => spectralSketchState('fast', 2, 2, 4, 7), () => spectralSketchState('fast', 2, 2, 1, 0),
  () => rowSketchState([0, 0]), () => rowSketchState([8]), () => traceProbeState('missing', 7, 2),
  () => traceProbeState('coupled', 1000, 2), () => traceProbeState('coupled', 7, 33)]) assert.throws(invalid, RangeError);
assert.equal(weightedProbeState([1, 1, -1]).direction, null);
assert.deepEqual(spectralSketchState('slow', 2, 1, 1, 7), spectralSketchState('slow', 2, 1, 1, 7));
for (let width = 1; width < 6; width += 1) {
  const current = spectralSketchState('slow', 1, width - 1, 0, 7);
  const extended = spectralSketchState('slow', 1, width, 0, 7);
  assert.deepEqual(current.omega, extended.omega.map(row => row.slice(0, width)));
}
await mkdir('scratch/randomized-linear-algebra-review', { recursive: true });
await writeFile('scratch/randomized-linear-algebra-review/model-fixtures.json', JSON.stringify({ probes, spectra, rows, traces }));
console.log(JSON.stringify({ probes: probes.length, spectra: spectra.length, rowSubsets: rows.length, traces: traces.length, invalidGroups: 13 }));

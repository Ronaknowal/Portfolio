import fs from 'node:fs';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import * as m from '../src/learn/data/complex-transforms-models.js';
import { complexTransformExamples } from '../src/learn/data/complex-transforms-examples.js';

const directory = 'scratch/complex-transforms-verification';
fs.mkdirSync(directory, { recursive: true });
const data = { examples: complexTransformExamples, arithmetic: [], roots: [], projections: [], square: [], fourier: [], aliases: [], windows: [], convolution: [], filters: [], integrals: [], regions: [] };
data.synthesis = [];
for (const phase of [-Math.PI, -.7, 0, Math.PI / 2]) for (const time of [0, .03125, .25, .61, 1]) data.synthesis.push(m.phasorSynthesis(time, phase));
for (const z of [[0, 0], [1, 2], [-3, .25], [1e-100, -1e-100]]) {
  for (const w of [[0, 0], [2, -1], [-.1, 3], [1e-100, 0]]) {
    for (const op of ['add', 'multiply', 'divide']) {
      if (op === 'divide' && w.every(v => v === 0)) continue;
      data.arithmetic.push({ z, w, op, result: m.complexOperation(z, w, op).result });
    }
  }
  for (const count of [1, 2, 3, 4, 7, 12]) data.roots.push({ z, count, result: m.rootsOfComplex(z, count) });
}
for (const phase of [-Math.PI, -.4, 0, Math.PI / 2]) for (const k of [-3, -1, 0, 1, 2, 3]) {
  const result = m.harmonicProjection(1.3, .7, phase, -.2, k);
  data.projections.push({ phase, k, coefficient: result.coefficient, points: [0, 1, 17, 95, 192, 256].map(i => result.points[i]) });
}
for (const terms of [1, 2, 4, 16, 64]) data.square.push(m.squareConvergence(terms));
for (const count of [1, 2, 3, 4, 5, 7, 8, 16, 31]) {
  for (let seed = 0; seed < 4; seed += 1) {
    const values = Array.from({ length: count }, (_, n) => [((n * 7 + seed * 3) % 13 - 6) / 3, seed % 2 ? Math.sin(n + .2) : 0]);
    for (const removePair of [false, true]) {
      const bin = count > 1 ? 1 : 0;
      data.fourier.push({ values, bin, removePair, result: m.finiteFourier(values, bin, removePair) });
    }
  }
}
for (const frequency of [0, 3, 8, 13, 16, 24]) for (const sampleRate of [4, 16, 31.5, 32]) for (const phase of [-1.1, 0, Math.PI / 3]) {
  const state = m.aliasState(frequency, sampleRate, phase);
  delete state.curves;
  data.aliases.push(state);
}
for (const count of [32, 64, 128]) for (const padded of [count, 256, 512]) for (const window of ['rectangular', 'hann']) for (const tone of [3, 5.5, 8.125]) {
  const state = m.windowSpectrum(count, padded, window, tone);
  state.curve = state.curve.filter((_, i) => i % 96 === 0);
  data.windows.push(state);
}
for (const values of [[1, 2, 0, -1], [2, -3, .5], [-1, 0, 2, 3, 4]]) for (const kernel of [[1, 1], [.25, -.5], [2]]) for (const circular of [false, true]) {
  for (let index = 0; index < (circular ? values.length : values.length + kernel.length - 1); index += 1) data.convolution.push(m.convolutionState(values, kernel, circular, index));
}
for (const rate of [.5, 2 * Math.PI, 20]) for (const phase of [-Math.PI, .3, Math.PI / 2]) for (const initial of [-3, 0, 2]) {
  const state = m.filterResponse(rate, phase, initial);
  state.points = state.points.filter((_, i) => i % 64 === 0);
  data.filters.push(state);
}
for (const real of [-8, -.5, -1e-12, 0, 1e-12, .5, 8]) for (const imaginary of [-8, -1e-12, 0, .3, 8]) for (const horizon of [0, 1e-12, .1, 1, 8]) data.integrals.push({ real, imaginary, horizon, value: m.finiteExponentialIntegral(real, imaginary, horizon) });
for (const side of ['left', 'right']) for (const sigma of [-2, -1, 0]) for (const omega of [0, .7]) data.regions.push(m.laplaceRegion(1, sigma, omega, 4, side));
const rejects = [
  () => m.complexOperation([1, 0], [0, 0], 'divide'),
  () => m.complexOperation([1, 0], [Number.MIN_VALUE, 0], 'divide'),
  () => m.complexOperation([1, NaN], [0, 0]),
  () => m.complexOperation(Array(2), [1, 0]),
  () => m.rootsOfComplex([1, 0], 0),
  () => m.harmonicProjection(1, 1, 0, 0, .5),
  () => m.squareConvergence(65), () => m.dft([]), () => m.dft(Array(3)),
  () => m.dft([1, Infinity]), () => m.dft([1, 2], 'yes'),
  () => m.parseRealSamples('1,,2,3'), () => m.parseRealSamples('1,NaN,2,3'),
  () => m.parseRealSamples('1,1e-999,2,3'), () => m.parseRealSamples('1,5,2,3'),
  () => m.windowSpectrum(64, 32), () => m.windowSpectrum(64, 64, 'unknown'),
  () => m.convolutionState([1], [1, 2]), () => m.laplaceRegion(1, 0, 0, Infinity),
];
rejects.forEach(check => assert.throws(check, RangeError));
data.rejected = rejects.length;
fs.writeFileSync(`${directory}/fixtures.json`, JSON.stringify(data));
const result = spawnSync('scratch/lesson-tools/Scripts/python.exe', ['scripts/verify-complex-transforms.py'], { stdio: 'inherit' });
process.exitCode = result.status ?? 1;

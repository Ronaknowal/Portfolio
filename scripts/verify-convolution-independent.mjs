// Complementary reviewer route: forward Boolean dependency propagation, exact
// polynomial path counts, and arbitrary transpose dot-product identities.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import { observedAncestors, receptiveTrace, cnnAxisLayers, averagingProfile, transpose1d } from '../src/learn/data/convolution-models.js';
const evidence = 'docs/teaching/evidence/convolution-independent-model.json';
fs.writeFileSync(evidence, JSON.stringify({ passed: false, status: 'running' }));
let ancestryCases = 0, polynomialCases = 0, adjointCases = 0;
function near(a, b, tolerance = 1e-10) { assert.ok(Math.abs(a - b) <= tolerance * Math.max(1, Math.abs(a), Math.abs(b)), `${a} != ${b}`); }
// Forward push each source identity through positive all-ones convolutions.
const families = [cnnAxisLayers, [{ k: 3, s: 2, left: 1, right: 1 }, { k: 3, d: 2, s: 1, left: 2, right: 2 }], [{ k: 4, left: 1, right: 2 }, { k: 2, s: 2 }], [{ k: 3, d: 2, left: 2, right: 2 }, { k: 3, d: 2, left: 2, right: 2 }]];
for (const layers of families) {
  let dependencies = Array.from({ length: 32 }, (_, i) => new Set([i]));
  for (let level = 0; level < layers.length; level++) {
    const { k, d = 1, s = 1, left = 0, right = 0 } = layers[level];
    const padded = [...Array.from({ length: left }, () => new Set()), ...dependencies, ...Array.from({ length: right }, () => new Set())];
    const next = [];
    for (let start = 0; start + d * (k - 1) < padded.length; start += s) {
      const sources = new Set();
      for (let tap = 0; tap < k; tap++) for (const id of padded[start + tap * d]) sources.add(id);
      next.push(sources);
    }
    dependencies = next;
    const trace = receptiveTrace(layers.slice(0, level + 1));
    assert.equal(dependencies.length, trace.at(-1).n);
    for (let i = 0; i < dependencies.length; i++) {
      assert.deepEqual(observedAncestors(layers.slice(0, level + 1), i), [...dependencies[i]].sort((a, b) => a - b));
      ancestryCases++;
    }
  }
}
// Integer coefficients of (1+z+z²)^L via multinomial counts, not recursive convolution.
const factorial = n => { let value = 1n; for (let i = 2n; i <= BigInt(n); i++) value *= i; return value; };
for (const depth of [1, 2, 3, 5, 10, 20]) {
  const counts = Array(2 * depth + 1).fill(0n);
  for (let twos = 0; twos <= depth; twos++) for (let ones = 0; ones + twos <= depth; ones++) {
    const zeros = depth - twos - ones;
    counts[2 * twos + ones] += factorial(depth) / (factorial(twos) * factorial(ones) * factorial(zeros));
  }
  const probabilities = counts.map(value => Number(value) / 3 ** depth);
  for (const threshold of [0.001, 0.013, 0.2]) {
    const result = averagingProfile(depth, threshold), peak = Math.max(...probabilities);
    result.coefficients.forEach((value, i) => near(value, probabilities[i], 1e-12));
    const accepted = probabilities.flatMap((value, i) => value >= threshold * peak ? [i - depth] : []);
    assert.deepEqual(result.retained, accepted); polynomialCases++;
  }
}
for (const inputLength of [1, 2, 5]) for (const kernel of [[1, -1], [0.25, 1, -0.7], [0, 0, 0]]) for (const stride of [1, 2, 3]) {
  const dual = Array.from({ length: inputLength }, (_, i) => Math.cos(i * 1.7));
  const input = Array.from({ length: (inputLength - 1) * stride + kernel.length }, (_, i) => Math.sin(i * 0.83));
  const output = dual.map((_, i) => kernel.reduce((sum, weight, tap) => sum + weight * input[i * stride + tap], 0));
  const reverse = transpose1d(dual, kernel, stride).output;
  near(output.reduce((sum, value, i) => sum + value * dual[i], 0), input.reduce((sum, value, i) => sum + value * reverse[i], 0)); adjointCases++;
}
const source = 'src/learn/data/convolution-models.js';
fs.writeFileSync(evidence, JSON.stringify({ passed: true, ancestryCases, polynomialCases, adjointCases, method: 'Forward source-set propagation vs backward tracing; exact multinomial path counts vs recursive averaging; arbitrary strided dot-product identities.', sourceHash: crypto.createHash('sha256').update(fs.readFileSync(source)).digest('hex') }, null, 2));
console.log(`Independent convolution JS review: ${ancestryCases} ancestry, ${polynomialCases} polynomial, ${adjointCases} adjoint cases passed.`);

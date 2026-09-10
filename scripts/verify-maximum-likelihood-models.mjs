import assert from 'node:assert/strict';
import { bernoulli, betaDensity, betaPosterior, locationFit, normalPosterior, parseBinary, parseMeasurements, samplingMass, transformedBetaDensity, logit } from '../src/learn/data/maximum-likelihood-models.js';
const close = (a, b, tolerance = 2e-10) => assert(Math.abs(a - b) <= tolerance * Math.max(1, Math.abs(a), Math.abs(b)), `${a} != ${b}`);
let eventChecks = 0;
for (let n = 0; n <= 9; n += 1) {
  for (let numerator = 0; numerator <= 10; numerator += 1) {
    const p = numerator / 10;
    const byCount = Array(n + 1).fill(0);
    for (let bits = 0; bits < 2 ** n; bits += 1) {
      let mass = 1;
      let successes = 0;
      for (let i = 0; i < n; i += 1) {
        const success = (bits >> i) & 1;
        successes += success;
        mass *= success ? p : 1 - p;
      }
      close(bernoulli(successes, n - successes, p).value, mass);
      byCount[successes] += mass;
      eventChecks += 1;
    }
    if (n) {
      const rows = samplingMass(n, p);
      rows.forEach(row => close(row.mass, byCount[row.k]));
      close(rows.reduce((sum, row) => sum + row.mass, 0), 1);
      close(rows.reduce((sum, row) => sum + row.mass * row.estimate, 0), p);
      close(rows.reduce((sum, row) => sum + row.mass * (row.estimate - p) ** 2, 0), p * (1 - p) / n);
      assert(Object.isFrozen(rows) && Object.isFrozen(rows[0]));
    }
  }
}
function simpson(fn, low = 0, high = 1, panels = 2000) {
  const step = (high - low) / panels;
  let sum = fn(low) + fn(high);
  for (let i = 1; i < panels; i += 1) sum += (i % 2 ? 4 : 2) * fn(low + i * step);
  return sum * step / 3;
}
let posteriorChecks = 0;
for (const a of [1, 2, 5, 12]) for (const b of [1, 2, 5, 12]) for (const s of [0, 1, 3, 20]) for (const f of [0, 1, 4, 20]) {
  const post = betaPosterior(s, f, a, b);
  const density = p => betaDensity(p, post.a, post.b);
  close(simpson(density), 1, 1e-8);
  close(simpson(p => p * density(p)), post.mean, 1e-8);
  close(simpson(p => (p - post.mean) ** 2 * density(p)), post.variance, 1e-8);
  if (post.mode !== null) for (let i = 0; i <= 100; i += 1) assert(density(i / 100) <= density(post.mode) + 1e-10);
  else assert.equal(post.a + post.b, 2);
  posteriorChecks += 1;
}
close(simpson(p => betaDensity(p, 5, 3), .5, .75), 8681 / 16384);
close(simpson(eta => transformedBetaDensity(eta, 5, 3), 0, Math.log(3)), 8681 / 16384);
for (let i = -400; i <= 400; i += 1) assert(transformedBetaDensity(i / 100, 5, 3) <= transformedBetaDensity(logit(5 / 8), 5, 3) + 1e-12);
const data = Object.freeze([2, 3, 4, 7]);
for (let center = -10; center <= 35; center += .25) {
  const fit = locationFit(data, center);
  close(fit.squared, 14 + 4 * (center - 4) ** 2);
  assert(fit.absolute >= 6);
  assert(Object.isFrozen(fit) && Object.isFrozen(fit.residuals));
}
assert.deepEqual(normalPosterior(data, 4, 0, 1), { mean: 2, variance: .5 });
for (const variance of [.5, 4, 9]) for (const priorMean of [-4, 0, 2, 10]) for (const priorVariance of [.25, 1, 8]) {
  const post = normalPosterior(data, variance, priorMean, priorVariance);
  const objective = center => data.reduce((sum, value) => sum + (value - center) ** 2 / (2 * variance), 0) + (center - priorMean) ** 2 / (2 * priorVariance);
  for (let center = -10; center <= 15; center += .25) {
    close(objective(center) - objective(post.mean), (center - post.mean) ** 2 / (2 * post.variance));
  }
}
// Repeated two-outcome convolution checks all browser-supported sample sizes.
for (const p of [0, .01, .2, .5, .97, 1]) {
  let masses = [1];
  for (let n = 1; n <= 30; n += 1) {
    const next = Array(n + 1).fill(0);
    masses.forEach((mass, k) => { next[k] += mass * (1 - p); next[k + 1] += mass * p; });
    masses = next;
    samplingMass(n, p).forEach(row => close(row.mass, masses[row.k]));
  }
}
assert.equal(locationFit([2, 3, 4, 7, 27], 0).mean, 8.6);
assert.equal(bernoulli(0, 0, 0).mle, null);
assert.equal(bernoulli(4, 0, 1).log, 0);
assert.equal(bernoulli(3, 1, 1).log, -Infinity);
assert.deepEqual(parseBinary(''), []);
assert.deepEqual(parseBinary('1, 0 1'), [1, 0, 1]);
assert.deepEqual(parseMeasurements('2 -3 .4'), [2, -3, .4]);
for (const bad of ['1 two', '2', Array(41).fill('1').join(' ')]) assert.throws(() => parseBinary(bad));
for (const bad of ['', 'Infinity', '51', 'NaN']) assert.throws(() => parseMeasurements(bad));
assert.throws(() => bernoulli(30, 30, .5));
assert.throws(() => samplingMass(0, .5));
assert.throws(() => betaPosterior(1, 1, .5, 2));
console.log(`Models passed: ${eventChecks} independent sequence enumerations; ${posteriorChecks} posterior integration/mode cases; normalized sampling masses, coordinate interval conservation, residual/normal identities and input boundaries.`);

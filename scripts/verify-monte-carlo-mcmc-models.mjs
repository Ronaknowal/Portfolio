import assert from 'node:assert/strict';
import { finiteMetropolis, independentEstimate, metropolisTrace, leapfrog, exactHarmonic, harmonicEnergy, hamiltonianTrajectory, nutsExpansion, stationaryMeanVariance, twoStateTrace, seededRandom } from '../src/learn/data/monte-carlo-mcmc-models.js';

const close = (actual, expected, tolerance = 1e-10) => assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`);
const average = values => values.reduce((a, b) => a + b, 0) / values.length;
let cases = 0;
// Independent accepted-flow identity: min(pi_i*q_ij, pi_j*q_ji).
for (let a = 1; a <= 9; a += 1) for (let b = 1; b <= 9; b += 1) {
  const weights = [a, b, 3];
  const q = [[.1, .8, .1], [.2, .1, .7], [.6, .3, .1]];
  const result = finiteMetropolis(weights, q);
  for (let i = 0; i < 3; i += 1) {
    close(result.transition[i].reduce((x, y) => x + y), 1);
    close(result.afterOne[i], result.target[i]);
    for (let j = 0; j < 3; j += 1) if (i !== j) close(result.flow[i][j], Math.min(result.target[i] * q[i][j], result.target[j] * q[j][i]));
  }
  cases += 1;
}
close(finiteMetropolis().acceptance[0][1], .625);
assert.ok(Math.abs(finiteMetropolis(undefined, undefined, false).afterOne[0] - .2) > .01);
for (const evaluations of [2, 10, 100, 2000]) for (const antithetic of [false, true]) {
  const result = independentEstimate({ evaluations, antithetic });
  close(result.estimate, average(result.rows.flatMap(row => row.contributions)));
  close(result.exactVariance, antithetic ? 1 / (90 * evaluations) : 4 / (45 * evaluations));
  for (const row of result.rows) if (antithetic) close(row.inputs[0] + row.inputs[1], 1);
}
assert.equal(independentEstimate({ evaluations: 2, antithetic: true }).mcse, null);
for (const scale of [.01, .12, .5, 1]) {
  const result = metropolisTrace({ iterations: 500, scale });
  let previous = .5;
  for (const row of result.rows) {
    close(row.before, previous);
    // Compare direct density ratio (rather than the implementation's logs).
    const f = x => x > 0 && x < 1 ? x ** 9 * (1 - x) ** 3 : 0;
    assert.equal(row.accept, row.uniform < Math.min(1, f(row.proposal) / f(previous)));
    close(row.state, row.accept ? row.proposal : previous);
    previous = row.state;
  }
}
// Independent matrix representation of kick/drift/kick and analytic orbit.
for (const sigma of [.25, .5, 1, 2]) for (const epsilon of [.01, .1, .5, 1]) {
  const a = 1 - epsilon ** 2 / (2 * sigma ** 2);
  const b = epsilon;
  const c = -epsilon / sigma ** 2 * (1 - epsilon ** 2 / (4 * sigma ** 2));
  close(a * a - b * c, 1);
  const actual = leapfrog(.8, -.3, epsilon, sigma);
  close(actual.position, a * .8 + b * -.3);
  close(actual.momentum, c * .8 + a * -.3);
  const reversed = leapfrog(actual.position, actual.momentum, -epsilon, sigma);
  close(reversed.position, .8); close(reversed.momentum, -.3);
  const orbit = exactHarmonic(.8, -.3, 1.3, sigma);
  close(harmonicEnergy(orbit.position, orbit.momentum, sigma), harmonicEnergy(.8, -.3, sigma));
}
const first = hamiltonianTrajectory({ steps: 1 });
close(first.rows[1].halfMomentum, .6); close(first.rows[1].position, 1.12); close(first.rows[1].momentum, .488); close(first.rows[1].energyError, .001272);
assert.ok(hamiltonianTrajectory({ sigma: .25, step: 1, steps: 12 }).divergent);
const coarse = hamiltonianTrajectory({ step: .2, steps: 5 }).rows.at(-1);
const fine = hamiltonianTrajectory({ step: .1, steps: 10 }).rows.at(-1);
const exact = exactHarmonic(1, .7, 1);
assert.ok(Math.hypot(fine.position - exact.position, fine.momentum - exact.momentum) < Math.hypot(coarse.position - exact.position, coarse.momentum - exact.momentum) / 3);

// Enumerate the probability of every 0/1 path; independent finite variance oracle.
for (let n = 2; n <= 10; n += 1) for (const rho of [-.8, 0, .8]) for (const thinning of [1, 2]) {
  const m = Math.floor(n / thinning);
  let variance = 0;
  for (let bits = 0; bits < 2 ** n; bits += 1) {
    const states = Array.from({ length: n }, (_, i) => (bits >> i) & 1);
    let probability = .5;
    for (let i = 1; i < n; i += 1) probability *= states[i] === states[i - 1] ? (1 + rho) / 2 : (1 - rho) / 2;
    const retained = states.filter((_, i) => (i + 1) % thinning === 0);
    variance += probability * (average(retained) - .5) ** 2;
  }
  close(stationaryMeanVariance(n, rho, thinning).variance, variance);
  assert.equal(stationaryMeanVariance(n, rho, thinning).retained, m);
  cases += 1;
}
assert.ok(stationaryMeanVariance(100, -.8).equivalentIID > 100);
assert.ok(stationaryMeanVariance(100, -.8, 2).variance > stationaryMeanVariance(100, -.8).variance);
assert.equal(twoStateTrace({ draws: 20 }).length, 20);

// Independent reconstruction of each NUTS leaf from a matrix power at its time.
// Also check slice eligibility and the distinction between rejected subtrees and
// whole-tree termination. This verifies the construction, not global convergence.
const reasons = new Set();
for (let seed = 1; seed <= 200; seed += 1) for (const step of [.1, .4, 1.5]) {
  const result = nutsExpansion({ seed, step });
  let pool = [result.initial];
  for (const snapshot of result.snapshots) {
    reasons.add(snapshot.stopReason);
    for (const state of snapshot.explored) {
      let q = result.initial.position, r = result.initial.momentum;
      const epsilon = Math.sign(state.time) * step;
      const a = 1 - epsilon ** 2 / 2;
      const c = -epsilon * (1 - epsilon ** 2 / 4);
      for (let i = 0; i < Math.abs(state.time); i += 1) [q, r] = [a * q + epsilon * r, c * q + a * r];
      close(state.position, q, 1e-8); close(state.momentum, r, 1e-8);
      assert.equal(state.onSlice, state.validEnergy && result.logSlice <= -(q * q + r * r) / 2);
    }
    if (!snapshot.subtreeAccepted) assert.deepEqual(snapshot.candidates, pool);
    snapshot.candidates.forEach(state => assert.ok(result.logSlice <= -harmonicEnergy(state.position, state.momentum) + 1e-12));
    assert.ok(snapshot.explored.length <= 2 ** snapshot.depth - 1);
    pool = snapshot.candidates;
  }
  assert.ok(pool.includes(result.selected));
  assert.ok(Object.isFrozen(result) && Object.isFrozen(result.snapshots) && Object.isFrozen(pool));
  cases += 1;
}
for (const reason of ['Depth cap', 'Internal subtree turn', 'Whole-tree turn']) assert.ok(reasons.has(reason));
assert.throws(() => independentEstimate({ evaluations: 3, antithetic: true }));
assert.throws(() => nutsExpansion({ maxDepth: 8 }));
assert.throws(() => metropolisTrace({ scale: NaN }));
assert.throws(() => stationaryMeanVariance(1, 1));
assert.throws(() => seededRandom(0));
assert.throws(() => finiteMetropolis([1e308, 1e308, 1e308]));
console.log(`Monte Carlo/MCMC model checks passed: ${cases} compound cases, exact path enumeration, flow identities, independent linear dynamics and NUTS leaf/candidate contracts.`);

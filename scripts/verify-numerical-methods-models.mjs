import assert from 'node:assert/strict';
import fs from 'node:fs';
import { adaptiveSimpson, bracketTrace, compositeQuadrature, derivativeEstimate, formatNumerical, integralProblems, newtonTrace, pumpCalibration, rootProblems, sampledTrapezoid } from '../src/learn/data/numerical-methods-models.js';

const output = { brackets: [], newton: [], quadrature: [], adaptive: [], derivatives: [], samples: [], calibration: [] };
for (const target of [.2, .7, 2, 5, 17, 81]) {
  for (const tolerance of [1e-3, 1e-8, 1e-12]) {
    const state = bracketTrace(x => x * x - target, 0, 10, { tolerance, maximumSteps: 100 });
    for (const row of state.steps) {
      assert(row.lower <= Math.sqrt(target) && row.upper >= Math.sqrt(target));
      assert.equal(row.width, 10 / 2 ** row.iteration);
    }
    output.brackets.push({ target, tolerance, state });
  }
}
assert.equal(bracketTrace(x => x, 0, 1).status, 'endpoint root');
assert.equal(bracketTrace(x => x * x + 1, -1, 1).status, 'no opposite endpoint signs');
assert.equal(bracketTrace(x => 1 / x, -1, 1).status, 'nonfinite evaluation');
assert.equal(bracketTrace(x => x <= 1 ? -1 : 1, 1, 1 + Number.EPSILON, { tolerance: Number.MIN_VALUE }).status, 'arithmetic stagnation');
assert.equal(bracketTrace(x => x < .3 ? -Number.MAX_VALUE : Number.MAX_VALUE, 0, 1).status, 'bracket tolerance reached');
assert.throws(() => bracketTrace(x => x, -Number.MAX_VALUE, Number.MAX_VALUE));
assert.throws(() => bracketTrace(x => x, 0, 1, { tolerance: 0 }));
for (const [key, problem] of Object.entries(rootProblems)) {
  for (const start of [-1, 0, .5, 1.5, 2]) {
    for (const safeguarded of [false, true]) {
      const state = newtonTrace(problem, start, { safeguarded, maximumSteps: 20 });
      for (const row of state.steps) {
        assert.equal(row.nextValue, problem.f(row.next));
        if (safeguarded) assert(row.next >= row.lower + .1 * (row.upper - row.lower) && row.next <= row.upper - .1 * (row.upper - row.lower));
      }
      output.newton.push({ key, start, safeguarded, state });
    }
  }
}
assert.deepEqual(newtonTrace(rootProblems.cycle, 0, { maximumSteps: 4 }).steps.map(row => row.next), [1, 0, 1, 0]);
for (const power of [0, 1, 2, 3, 4, 7]) {
  for (const lower of [-2, 0, 1]) {
    for (const panels of [2, 4, 8, 16, 64]) {
      for (const method of ['trapezoid', 'simpson']) {
        output.quadrature.push({ power, lower, upper: lower + 1.5, panels, method, state: compositeQuadrature(x => x ** power, lower, lower + 1.5, panels, method) });
      }
    }
  }
}
assert.throws(() => compositeQuadrature(Math.sin, 0, 1, 3, 'simpson'));
assert.throws(() => compositeQuadrature(() => Infinity, 0, 1));
assert.throws(() => compositeQuadrature(x => x, 1, 1 + Number.EPSILON, 4));
for (const key of Object.keys(integralProblems)) {
  for (const tolerance of [1e-3, 1e-6, 1e-9]) {
    for (const maximumDepth of [1, 3, 8, 12]) {
      const state = adaptiveSimpson(integralProblems[key].f, 0, 1, { tolerance, maximumDepth });
      if (state.estimate !== null) {
        assert(Math.abs(state.leaves.reduce((sum, leaf) => sum + leaf.budget, 0) - tolerance) < 1e-15);
        assert.equal(state.leaves[0].lower, 0);
        assert.equal(state.leaves.at(-1).upper, 1);
        state.leaves.slice(1).forEach((leaf, index) => assert.equal(leaf.lower, state.leaves[index].upper));
      }
      output.adaptive.push({ key, tolerance, maximumDepth, state });
    }
  }
}
const blind = adaptiveSimpson(integralProblems.blind.f, 0, 1);
assert.equal(blind.estimate, 0);
assert.equal(blind.estimatedError, 0);
assert.equal(blind.samples.length, 5);
assert.equal(adaptiveSimpson(integralProblems.peak.f, 0, 1, { maximumEvaluations: 5 }).estimate, null);
assert.equal(adaptiveSimpson(() => Infinity, 0, 1).status, 'nonfinite evaluation');
assert.equal(adaptiveSimpson(x => x, 1, 1 + Number.EPSILON).status, 'arithmetic stagnation');
for (const x of [-2, 0, 1, 3]) for (const h of [.1, .01, .001, 1e-6]) for (const stencil of ['forward', 'central', 'second', 'boundary']) {
  output.derivatives.push({ x, h, stencil, state: derivativeEstimate(x, h, stencil) });
}
assert.equal(derivativeEstimate(1, 1e-20).status, 'sample coordinates coincide');
assert.throws(() => derivativeEstimate(1, 0));
assert.equal(formatNumerical(1e-20), '1.000e-20');
for (const shift of [-4, 0, 3]) for (const scale of [.5, 2, 7]) {
  const xs = [0, .25, 1, 3].map(x => shift + scale * x);
  const ys = xs.map(x => 2 * x + 3);
  output.samples.push({ xs, ys, state: sampledTrapezoid(xs, ys) });
}
assert.throws(() => sampledTrapezoid([0, 0], [1, 2]));
for (const target of [.61, .7, .8, .89]) for (const candidate of [2, 2.3, 2.44, 3, 3.6, 4]) for (const panels of [2, 8, 32, 128]) {
  output.calibration.push(pumpCalibration(target, panels, candidate));
}
assert.throws(() => pumpCalibration(.5));
fs.mkdirSync('scratch/numerical-methods-verification', { recursive: true });
fs.writeFileSync('scratch/numerical-methods-verification/model-fixtures.json', JSON.stringify(output));
console.log(JSON.stringify(Object.fromEntries(Object.entries(output).map(([key, values]) => [key, values.length]))));

// Reviewer-owned changed fixtures; production models are imported unchanged.
import fs from 'node:fs';
import * as model from '../src/learn/data/numerical-methods-models.js';
import { numericalMethodsExamples as examples } from '../src/learn/data/numerical-methods-examples.js';

const cases = { examples, adaptive: [], brackets: [], newton: [], derivatives: [], calibration: [] };
for (const degree of [0, 1, 2, 3, 4, 5]) {
  for (const [a, b] of [[-1, 2], [0.25, 1.75], [-2, -0.5]]) {
    const f = x => (x - 0.125) ** degree;
    cases.adaptive.push({ degree, a, b, shift: 0.125,
      result: model.adaptiveSimpson(f, a, b, { tolerance: 1e-7, maximumDepth: 12 }) });
  }
}
for (const scale of [1, 1000, -4]) {
  cases.adaptive.push({ blind: true, scale, a: 0, b: 1,
    result: model.adaptiveSimpson(x => scale * model.integralProblems.blind.f(x), 0, 1,
      { tolerance: 1e-12, maximumDepth: 20 }) });
}
cases.workFailures = [1, 3, 5, 9].map(maximumEvaluations => ({ maximumEvaluations,
  result: model.adaptiveSimpson(model.integralProblems.peak.f, 0, 1,
    { tolerance: 1e-12, maximumEvaluations }) }));
for (const root of [-1.75, -0.3, 0.125, 1.35]) {
  for (const scale of [-1e150, 1e-150, 1, 1e150]) {
    const f = x => scale * Math.expm1(x - root);
    const derivative = x => scale * Math.exp(x - root);
    const lower = root - 0.75, upper = root + 1.25;
    cases.brackets.push({ root, scale, result: model.bracketTrace(f, lower, upper,
      { tolerance: 1e-8, maximumSteps: 40 }) });
    cases.newton.push({ root, scale, result: model.newtonTrace({ f, derivative, lower, upper },
      upper, { safeguarded: true, maximumSteps: 40, tolerance: Math.abs(scale) * 1e-12 }) });
  }
}
for (const x of [-1.75, -0.125, 0, 1.25]) {
  for (const h of [0.25, 0.03125, 1 / 1024]) {
    for (const stencil of ['forward', 'central', 'second', 'boundary']) {
      cases.derivatives.push({ x, h, stencil, result: model.derivativeEstimate(x, h, stencil) });
    }
  }
}
for (const target of [0.6, 0.675, 0.79, 0.88, 0.9]) {
  for (const panels of [3, 9, 17, 65]) {
    for (const candidate of [2, 2.375, 3.25, 4]) {
      cases.calibration.push({ target, panels, candidate,
        result: model.pumpCalibration(target, panels, candidate) });
    }
  }
}
fs.mkdirSync('scratch/numerical-methods-independent-review', { recursive: true });
fs.writeFileSync('scratch/numerical-methods-independent-review/fixtures.json', JSON.stringify(cases));
console.log('Reviewer fixtures exported from unchanged numerical sources.');

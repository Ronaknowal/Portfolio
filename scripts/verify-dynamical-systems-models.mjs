import fs from "node:fs";
import { createHash } from "node:crypto";
import assert from "node:assert/strict";
import * as models from "../src/learn/data/dynamical-systems-models.js";

const root = "scratch/dynamical-systems-review";
fs.mkdirSync(root, { recursive: true });
const fixtures = { sourceSha256: createHash("sha256").update(fs.readFileSync("src/learn/data/dynamical-systems-models.js")).digest("hex") };
fixtures.cooling = [];
for (const decay of [0, 0.1, 0.5, 1, 2]) {
  for (const step of [0.05, 0.5, 1, 2]) {
    const input = { decay, step, steps: 12, initial: -3 };
    fixtures.cooling.push({ input, output: models.coolingTrace(input) });
  }
}
fixtures.scalar = [];
for (const kind of ["pitchfork", "tilted"]) {
  const parameters = kind === "pitchfork" ? [-1, -0.1, 0, 0.1, 1] :
    [-0.6, -models.CUBIC_FOLD, -models.CUBIC_FOLD + 1e-8, 0, models.CUBIC_FOLD - 1e-8, models.CUBIC_FOLD, 0.6];
  for (const parameter of parameters) for (const initial of [-1.7, -0.05, 0, 0.2, 1.6]) {
    const input = { kind, parameter, initial, duration: 4, steps: 400 };
    fixtures.scalar.push({ input, output: models.scalarFlowState(input) });
  }
}
fixtures.planar = [];
for (const mode of ["center", "spiral", "saddle", "transient", "hopf"]) {
  const parameters = mode === "hopf" ? [-1, -0.01, -1e-12, -Number.MIN_VALUE, 0, Number.MIN_VALUE, 1e-12, 0.01, 0.4, 1] : [0.4];
  for (const parameter of parameters) for (const initial of [[0, 0], [0, 1], [0.2, -0.3], [1.5, 0.4]]) {
    const input = { mode, parameter, initial, duration: 6, steps: 120 };
    fixtures.planar.push({ input, output: models.planarTrace(input) });
  }
}
fixtures.logistic = [];
for (const growth of [0, 0.5, 1, 2.5, 3, 3.2, 1 + Math.sqrt(6), 3.5, 3.83, 3.9, 4]) {
  for (const initial of [0, 0.125, 0.2, 0.5, 0.75, 1]) {
    const input = { growth, initial, steps: 10 };
    fixtures.logistic.push({ input, output: models.logisticTrace(input) });
  }
}
fixtures.sensitivity = [];
for (const growth of [0, 2.5, 3.2, 3.9, 4]) {
  for (const initial of [0, 0.2, 0.5, 0.75]) {
    for (const difference of [0, 1e-6]) {
      const input = { growth, initial, difference, steps: 8 };
      fixtures.sensitivity.push({ input, output: models.logisticSensitivity(input) });
    }
  }
}
fixtures.estimates = [0, 2.5, 3.2, 3.5, 3.83, 3.9, 4].map(growth => {
  const input = { growth, initial: 0.217, burn: 1000, samples: 2000 };
  return { input, output: models.logisticLyapunovEstimate(input) };
});
fixtures.criticalEstimate = models.logisticLyapunovEstimate({ growth: 4, initial: 0.5, burn: 0, samples: 3 });
fixtures.fixedEstimate = models.logisticLyapunovEstimate({ growth: 4, initial: 0.75, burn: 0, samples: 20 });
fixtures.atlas = models.bifurcationAtlas({ minimum: 3, maximum: 4, columns: 31, burn: 40, retained: 16 });
fixtures.cylinders = [];
for (let depth = 1; depth <= 9; depth += 1) {
  for (let value = 0; value < 2 ** depth; value += 1) {
    const word = value.toString(2).padStart(depth, "0").replaceAll("0", "L").replaceAll("1", "R");
    fixtures.cylinders.push(models.tentCylinder(word));
  }
}
fixtures.conjugacy = Array.from({ length: 301 }, (_, index) => {
  const value = index / 300;
  return { value, tent: models.tentStep(value), logistic: models.logisticCoordinate(value),
    cdf: models.logisticInvariantCdf(value) };
});
fixtures.quantized = [];
for (const bits of [1, 3, 8, 12]) {
  for (const numerator of [0, 1, Math.floor(2 ** bits / 3), 2 ** bits]) {
    const input = { bits, numerator, steps: 20 };
    fixtures.quantized.push({ input, output: models.quantizedTentTrace(input) });
  }
}
fixtures.oscillator = [];
for (const step of [0.05, 0.1, 0.2, 0.5, 1, 2, 2.5]) {
  for (const initial of [[1, 0], [0, 1], [0.2, -0.3], [0, 0]]) {
    const input = { step, initial, steps: 24 };
    fixtures.oscillator.push({ input, output: models.oscillatorTrace(input) });
  }
}
fixtures.lorenz = [];
for (const rho of [0.5, 1, 5, 10, 28]) {
  for (const initial of [[0, 0, 0], [1, 1, 1], [-3, 4, 5], [10, -2, 15]]) {
    const input = { rho, initial, duration: 1, steps: 400 };
    fixtures.lorenz.push({ input, output: models.lorenzTrace(input) });
  }
}
const invalid = [
  () => models.coolingTrace({ decay: NaN }),
  () => models.coolingTrace({ initial: Number.MIN_VALUE }),
  () => models.scalarFlowState({ kind: "missing" }),
  () => models.scalarFlowState({ duration: 12, steps: 10 }),
  () => models.planarTrace({ mode: "missing" }),
  () => models.planarTrace({ initial: [1] }),
  () => models.planarTrace({ parameter: Infinity }),
  () => models.logisticTrace({ growth: 4.1 }),
  () => models.logisticTrace({ initial: -0.1 }),
  () => models.logisticSensitivity({ difference: 1e-20 }),
  () => models.logisticSensitivity({ initial: 1, difference: 0.01 }),
  () => models.logisticLyapunovEstimate({ samples: 0 }),
  () => models.bifurcationAtlas({ minimum: 3, maximum: 3 }),
  () => models.bifurcationAtlas({ columns: 100000 }),
  () => models.tentCylinder("AB"),
  () => models.tentStep(Infinity),
  () => models.logisticInvariantCdf(1.1),
  () => models.quantizedTentTrace({ bits: 0 }),
  () => models.quantizedTentTrace({ numerator: -1 }),
  () => models.oscillatorTrace({ step: 0 }),
  () => models.oscillatorTrace({ step: 2.5, steps: 1000 }),
  () => models.lorenzTrace({ duration: 10, steps: 3 }),
  () => models.lorenzTrace({ initial: [1, NaN, 2] }),
];
for (const fail of invalid) assert.throws(fail, RangeError);
fixtures.invalidCases = invalid.length;
fs.writeFileSync(root + "/model-fixtures.json", JSON.stringify(fixtures) + "\n");
console.log("Exported exact production-model fixtures; " + invalid.length + " unsupported input cases rejected.");

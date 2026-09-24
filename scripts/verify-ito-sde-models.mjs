import fs from 'node:fs';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import * as m from '../src/learn/data/ito-sde-models.js';

const directory = 'scratch/ito-sde-verification';
fs.mkdirSync(directory, { recursive: true });
const fixtures = { errors: [], ou: [], growth: [], integrals: [], grouped: [], paths: [] };
for (const mu of [-0.5, -0.13, 0, 1e-8, 0.4, 1]) {
  for (const sigma of [0, 0.05, 0.2, 0.6, 1.2]) {
    for (const horizon of [0.25, 1, 2]) for (const steps of [1, 2, 8, 64, 512]) {
      for (const method of ['euler', 'milstein']) {
        const input = { mu, sigma, horizon, steps, method, initial: 1.3 };
        fixtures.errors.push({ input, output: m.gbmErrorMoments(input) });
      }
    }
  }
}
for (const theta of [0, 0.05, 0.3, 1, 3]) for (const step of [1/8192, 1/512, 0.25, 1, 4]) {
  fixtures.ou.push({ theta, step, output: m.ouNoiseMoments(theta, step) });
}
for (const mu of [-0.5, 0, 0.4, 1]) for (const sigma of [0, 0.05, 0.3, 1.2]) {
  for (const horizon of [0, 0.25, 1, 4]) {
    const input = { mu, sigma, horizon, initial: 0.7 };
    fixtures.growth.push({ input, output: m.growthLaw(input) });
  }
}
const random = m.normalSource(21);
for (let sample = 0; sample < 60; sample += 1) {
  const increments = Array.from({ length: sample % 8 + 1 }, () => Math.round(random() * 8) / 16);
  fixtures.integrals.push({ increments, output: m.integralTrace(increments, 1.5) });
}
const increments = m.brownianIncrements({ seed: 19, steps: 256, horizon: 1 });
for (const group of [1, 2, 4, 16, 64, 256]) {
  const active = m.groupIncrements(increments, group);
  fixtures.grouped.push({ increments, group, active });
  for (const sigma of [0, 0.3, 1.2]) {
    const input = { mu: -0.13, sigma, initial: 1.3, horizon: 1, increments: active };
    fixtures.paths.push({ input, output: m.growthPath(input) });
  }
}
fixtures.ouPaths = [0, 0.05, 1, 3].map(theta => ({ theta,
  output: m.ouPath({ theta, initial: -0.7, target: 0.3, eta: 0.4, steps: 32, seed: 17 }) }));
fixtures.ouLaws = [];
for (const theta of [0, 0.05, 0.3, 1, 3]) for (const eta of [0, 0.4, 1.2]) {
  for (const initialVariance of [0, 0.6, 4]) for (const time of [0, 1/8192, 0.125, 1, 4]) {
    const input = { theta, eta, initialVariance, time, initial: -0.7, target: 0.3 };
    fixtures.ouLaws.push({ input, output: m.ouLaw(input) });
  }
}
fixtures.sampled = ['euler', 'milstein'].map(method => ({ method,
  output: m.sampledErrors({ method, samples: 4096, steps: 64, seed: 29 }) }));
const rejected = [
  () => m.normalSource(0), () => m.normalSource(1.5),
  () => m.brownianIncrements({ steps: 513 }),
  () => m.groupIncrements([], 1), () => m.groupIncrements([1, 2, 3], 2),
  () => m.groupIncrements([NaN], 1), () => m.integralTrace([Infinity]),
  () => m.growthLaw({ sigma: 1e-300 }), () => m.growthLaw({ initial: 0 }),
  () => m.growthLaw({ horizon: 1e-300 }), () => m.growthPath({ horizon: 0 }),
  () => m.growthPath({ increments: Array(512).fill(-20), sigma: 1.2 }),
  () => m.ouNoiseMoments(1e-300, 1), () => m.ouNoiseMoments(1, 1e-300),
  () => m.ouLaw({ initialVariance: -1 }), () => m.ouLaw({ eta: -1 }),
  () => m.ouPath({ steps: 1e6 }), () => m.gbmErrorMoments({ method: 'unknown' }),
  () => m.gbmErrorMoments({ horizon: 4 }), () => m.gbmErrorMoments({ steps: 0 }),
  () => m.sampledErrors({ samples: 4 }), () => m.sampledErrors({ samples: 4096, steps: 512 }),
];
rejected.forEach(check => assert.throws(check, RangeError));
fixtures.rejected = rejected.length;
assert(Object.isFrozen(fixtures.paths[0].output.rows[0]));
const first = m.brownianIncrements({ seed: 2 });
assert.deepEqual(first, m.brownianIncrements({ seed: 2 }));
assert.notDeepEqual(first, m.brownianIncrements({ seed: 3 }));
fs.writeFileSync(`${directory}/model-fixtures.json`, JSON.stringify(fixtures));
const result = spawnSync('scratch/lesson-tools/Scripts/python.exe',
  ['scripts/verify-ito-sde-models.py'], { encoding: 'utf8', stdio: 'inherit' });
if (result.status !== 0) process.exit(result.status ?? 1);

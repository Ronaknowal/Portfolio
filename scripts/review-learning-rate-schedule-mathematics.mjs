import assert from 'node:assert/strict';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { resolve } from 'node:path';
import { buildPlateauTrace, buildRateSchedule, scheduleNoiseMoments } from '../src/learn/data/learning-rate-schedule-models.js';

const directory = resolve('scratch/learning-rate-schedule-independent');
mkdirSync(directory, { recursive: true });
const cycles = [];
for (const total of [3, 7, 12, 22]) {
  for (const rise of [...new Set([2, Math.floor(total / 2), total - 1])].filter(value => value >= 2)) {
    cycles.push({ total, rise, states: buildRateSchedule({ kind: 'one-cycle', total, rise }) });
  }
}
const noiseCases = [];
for (const rates of [[.1, .3], [.25, 0, .1], [0, 0], [.01, .7, .2, .1]]) {
  for (const curvature of [.5, 4, 8]) {
    for (const noise of [0, .5, 2]) {
      noiseCases.push({ rates, curvature, noise, states: scheduleNoiseMoments(rates.map(rate => ({ rate })), { curvature, noise, initialError: 2 }) });
    }
  }
}
const plateaus = [];
for (const threshold of [0, .125]) {
  for (const patience of [0, 2]) {
    for (const cooldown of [0, 2]) {
      for (const minimumRate of [0, .025]) {
        const metrics = [1, .875, .875, .75, .75, .75, .75, .5, .5, .5, .5, .5];
        const options = { threshold, patience, cooldown, minimumRate, initialRate: .2 };
        plateaus.push({ metrics, options, states: buildPlateauTrace(metrics, options) });
      }
    }
  }
}
writeFileSync(resolve(directory, 'fixtures.json'), JSON.stringify({ cycles, noiseCases, plateaus }));
const check = spawnSync(resolve('scratch/lesson-tools/Scripts/python.exe'), ['-X', 'utf8', '-I', 'scripts/review-learning-rate-schedule-mathematics.py'], { encoding: 'utf8', timeout: 120000 });
assert.equal(check.status, 0, check.stderr || check.stdout);
const files = ['src/learn/data/topics/learning-rate-schedules-cosine-warmup-onecyclelr.jsx', 'src/learn/data/learning-rate-schedule-models.js', 'src/learn/data/learning-rate-schedule-examples.js', 'src/learn/components/lesson-labs/LearningRateScheduleLabs.jsx'];
const result = { checkedAt: new Date().toISOString(), sourceHashes: Object.fromEntries(files.map(file => [file, createHash('sha256').update(readFileSync(file)).digest('hex')])), ...JSON.parse(check.stdout), findings: [] };
writeFileSync(resolve(directory, 'results.json'), JSON.stringify(result, null, 2));
console.log(JSON.stringify(result, null, 2));

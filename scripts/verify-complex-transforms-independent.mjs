import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { complexTransformExamples } from '../src/learn/data/complex-transforms-examples.js';
import * as models from '../src/learn/data/complex-transforms-models.js';

const directory = path.resolve('scratch/complex-transforms-independent-review');
fs.mkdirSync(directory, { recursive: true });
const author = JSON.parse(fs.readFileSync('docs/teaching/evidence/complex-transforms-author-review.json', 'utf8'));
const payload = {
  checkedAt: new Date().toISOString(),
  productionSources: author.productionSources.map(({ path: source }) => ({ path: source, sha256: crypto.createHash('sha256').update(fs.readFileSync(source)).digest('hex') })),
  examples: complexTransformExamples,
  integrals: [],
  filters: [],
  windows: [],
  square: [1, 7, 31, 64].map(terms => models.squareConvergence(terms)),
  filteredDft: models.finiteFourier([4, -4, 4, 4, 4, 4, 4, -4], 1, true),
};
for (const q of [[1e-18, 0], [0, 1e-18], [1e-15, -1e-16], [.125, .75], [-.125, 1.5], [0, 0]]) {
  for (const time of [.25, 1, 8]) payload.integrals.push({ q, time, result: models.finiteExponentialIntegral(...q, time) });
}
for (const rate of [.75, 3.5, 12, 20]) {
  for (const phase of [-2.1, .4, Math.PI / 4]) {
    for (const initial of [-3, 1.25]) {
      const state = models.filterResponse(rate, phase, initial);
      payload.filters.push({ rate, phase, initial, points: [0, 19, 83, 256, 512].map(i => state.points[i]) });
    }
  }
}
for (const count of [32, 64, 128]) {
  for (const window of ['rectangular', 'hann']) {
    for (const tone of [3.125, 6.375, 8.875]) {
      const state = models.windowSpectrum(count, 512, window, tone);
      payload.windows.push({ count, window, tone, bins: state.bins, densityIntegral: state.densityIntegral, weightedMeanSquare: state.weightedMeanSquare });
    }
  }
}
fs.writeFileSync(path.join(directory, 'payload.json'), `${JSON.stringify(payload, null, 2)}\n`);
const run = spawnSync(path.resolve('scratch/lesson-tools/Scripts/python.exe'), ['scripts/verify-complex-transforms-independent.py'], { encoding: 'utf8', env: { ...process.env, PYTHONIOENCODING: 'utf-8' } });
process.stdout.write(run.stdout);
process.stderr.write(run.stderr);
if (run.status !== 0) process.exit(run.status ?? 1);

import fs from 'node:fs';
import crypto from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { conditioningStabilityExamples } from '../src/learn/data/conditioning-stability-examples.js';
import * as models from '../src/learn/data/conditioning-stability-models.js';

const folder = 'scratch/conditioning-independent-review';
fs.mkdirSync(folder, { recursive: true });
const author = JSON.parse(fs.readFileSync('docs/teaching/evidence/conditioning-stability-author-review.json', 'utf8'));
for (const source of author.productionSources) {
  const actual = crypto.createHash('sha256').update(fs.readFileSync(source.path)).digest('hex');
  if (actual !== source.sha256) throw new Error(`Author freeze mismatch: ${source.path}`);
}
const fixtures = {
  productionSources: author.productionSources,
  examples: conditioningStabilityExamples,
  cancellation: [-1, 0, 1].flatMap(sign => [0, 1, 17, 40, 52, 53, 54, 60].map(k => ({ sign, k, state: models.cancellationState(k, sign) }))),
  propagation: [-0.5, 0.5, 0.9, 1, 1.1].flatMap(q => ['constant', 'alternating', 'pulse'].map(mode => ({ q, mode, state: models.propagationState(q, mode, 19) }))),
  stored: [Number.MIN_VALUE, -Number.MIN_VALUE, 2 ** -1022, Number.MAX_VALUE, -0, 0.1, 1 + 2 ** -52].map(value => ({value, fraction: models.storedNumberFraction(value).map(String)})),
};
fs.writeFileSync(`${folder}/fixtures.json`, JSON.stringify(fixtures, null, 2));
const result = spawnSync('scratch/lesson-tools/Scripts/python.exe', ['scripts/verify-conditioning-independent.py'], { encoding: 'utf8' });
process.stdout.write(result.stdout || '');
process.stderr.write(result.stderr || '');
if (result.status !== 0) process.exit(result.status || 1);

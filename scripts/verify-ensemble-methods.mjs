import fs from 'node:fs';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import * as m from '../src/learn/data/ensemble-methods-models.js';
import { ensembleExamples } from '../src/learn/data/ensemble-methods-examples.js';

const directory = 'scratch/ensemble-methods';
fs.mkdirSync(directory, { recursive: true });
const data = { examples: ensembleExamples, votes: [], blends: [], bags: [], boosting: [], oof: [], invalid: 0 };
for (const a of [0, .25, .5, .75, 1]) for (const b of [0, .25, .5, .75, 1]) for (const c of [0, .25, .5, .75, 1]) {
  for (const weight of [0, .5, 1, 2, 4]) data.votes.push({ p: [a, b, c], weights: [1, 1, weight], result: m.votingState([a, b, c], [1, 1, weight]) });
}
for (const preset of Object.keys(m.errorBlendPresets)) for (let index = 0; index <= 100; index += 1) data.blends.push({ preset, weight: index / 100, result: m.errorBlendState(preset, index / 100) });
function bags(prefix, size, minimum = 0) {
  if (prefix.length === size) { data.bags.push({ draws: prefix, result: m.fitBootstrapStump(prefix) }); return; }
  for (let value = minimum; value < 6; value += 1) bags([...prefix, value], size, value);
}
for (let size = 1; size <= 5; size += 1) bags([], size);
for (const preset of Object.keys(m.boostingPresets)) data.boosting.push({ preset, result: m.signedBoostingTrace(preset, 12) });
for (const mode of ['honest', 'leaky']) for (let count = 0; count <= 3; count += 1) for (const query of [0, .5, 1.25, 2.5, 4.5, 5]) data.oof.push(m.oofOwnershipState(mode, count, query));
for (const call of [
  () => m.votingState([, .5, .5]), () => m.votingState([.5, .5, .5], [0, 0, 0]), () => m.votingState([.5, NaN, 1]), () => m.votingState([.5, .5, .5], [-1, 1, 1]),
  () => m.fitBootstrapStump([]), () => m.fitBootstrapStump([, 1]), () => m.fitBootstrapStump([.5]), () => m.fitBootstrapStump([6]),
  () => m.errorBlendState('toString'), () => m.errorBlendState('copies', Infinity), () => m.signedBoostingTrace('mixed', 1.5),
  () => m.bootstrapState('constructor'), () => m.bootstrapState('repeated', 3), () => m.oofOwnershipState('other'), () => m.oofOwnershipState('honest', 2, NaN),
]) { assert.throws(call); data.invalid += 1; }
if (process.argv.includes('--models-only')) {
  assert.deepEqual(data, JSON.parse(fs.readFileSync(directory + '/verification-input.json')));
  const result = { checkedAt: new Date().toISOString(), status: 'passed', scope: 'Current model states and example records exactly equal the independently checked native input; source formatting preserved normalized AST.', modelSha256: createHash('sha256').update(fs.readFileSync('src/learn/data/ensemble-methods-models.js')).digest('hex') };
  fs.writeFileSync(directory + '/model-format-recheck.json', JSON.stringify(result, null, 2) + '\n');
  console.log(JSON.stringify(result, null, 2));
  process.exit(0);
}
fs.writeFileSync(directory + '/verification-input.json', JSON.stringify(data));
const execution = spawnSync(process.env.LESSON_PYTHON || 'scratch/lesson-tools/Scripts/python.exe', ['scripts/verify-ensemble-methods.py'], { encoding: 'utf8', timeout: 240000, env: { ...process.env, OMP_NUM_THREADS: '1', LOKY_MAX_CPU_COUNT: '2' } });
process.stdout.write(execution.stdout || '');
process.stderr.write(execution.stderr || '');
assert.equal(execution.status, 0, 'Independent native/oracle check failed');

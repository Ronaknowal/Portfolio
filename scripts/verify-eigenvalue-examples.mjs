import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { spawnSync } from 'node:child_process';
import { eigenvalueExamples } from '../src/learn/data/eigenvalue-examples.js';
import { eigenDirectionPresets, eigenDirectionState, repeatedMapPresets, repeatedMapStarts, repeatedMapTrace, pcaDirectionDatasets, pcaDirectionState } from '../src/learn/data/eigenvalue-models.js';

const directory = path.resolve('scratch/eigenvalue-verification');
fs.mkdirSync(directory, { recursive: true });
const fixtures = { directions: [], repetitions: [], pca: [] };
for (const preset of Object.keys(eigenDirectionPresets)) {
  for (let angle = 0; angle <= 360; angle += 5) fixtures.directions.push({ preset, angle, state: eigenDirectionState(preset, angle) });
}
for (const preset of Object.keys(repeatedMapPresets)) {
  for (const start of Object.keys(repeatedMapStarts)) fixtures.repetitions.push({ preset, start, trace: repeatedMapTrace(preset, start, 30) });
}
for (const dataset of Object.keys(pcaDirectionDatasets)) {
  for (let angle = 0; angle <= 360; angle += 5) fixtures.pca.push({ dataset, angle, state: pcaDirectionState(dataset, angle) });
}
for (const invalid of [-1, 361, NaN, Infinity]) {
  assert.throws(() => eigenDirectionState('scalar', invalid));
  assert.throws(() => pcaDirectionState('line', invalid));
}
assert.throws(() => eigenDirectionState('missing'));
assert.throws(() => pcaDirectionState('missing'));
assert.throws(() => repeatedMapTrace('missing'));
assert.throws(() => repeatedMapTrace('decay', 'missing'));
for (const steps of [-1, 31, 0.5, NaN]) assert.throws(() => repeatedMapTrace('decay', 'mixed', steps));
const before = eigenDirectionState();
before.matrix[0][0] = 999;
assert.equal(eigenDirectionState().matrix[0][0], 2, 'A caller mutated a preset through returned state');

fs.writeFileSync(path.join(directory, 'models.json'), JSON.stringify(fixtures));
fs.writeFileSync(path.join(directory, 'programs.json'), JSON.stringify(eigenvalueExamples));
const python = process.env.LESSON_PYTHON || path.resolve('scratch/lesson-tools/Scripts/python.exe');
const result = spawnSync(python, ['-I', path.resolve('scripts/verify-eigenvalue-native.py'), directory], {
  encoding: 'utf8', timeout: 60000, env: { ...process.env, PYTHONIOENCODING: 'utf-8' },
});
process.stdout.write(result.stdout || '');
process.stderr.write(result.stderr || '');
assert.equal(result.status, 0, result.error?.message || 'Independent eigenvalue verification failed');

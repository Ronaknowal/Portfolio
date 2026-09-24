import fs from 'node:fs';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import assert from 'node:assert/strict';
import { numpyFoundationsExamples } from '../src/learn/data/numpy-foundations-examples.js';
import { scalingExamples } from '../src/learn/data/scaling-examples.js';

const program = numpyFoundationsExamples.loopReference.code + `
from math import fsum
for rows, columns in [(1, 1), (2, 5), (7, 3)]:
    raw = np.arange(rows * columns, dtype=float).reshape(rows, columns) / 4
    offsets = np.arange(columns, dtype=float) / 2
    saved = raw.copy()
    actual, means = calibrate_sensor_means(raw, offsets)
    expected = [[float(raw[i,j])-float(offsets[j]) for j in range(columns)] for i in range(rows)]
    expected_means = [fsum(row[j] for row in expected)/rows for j in range(columns)]
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_allclose(means, expected_means, atol=1e-12)
    np.testing.assert_array_equal(raw, saved)
    assert not np.shares_memory(raw, actual)
    shifted, shifted_means = calibrate_sensor_means(raw+32, offsets+32)
    np.testing.assert_array_equal(shifted, actual)
    np.testing.assert_array_equal(shifted_means, means)
print('Independent coordinate sums, non-square shapes, alias protection and translation invariance passed')
`;
const run = spawnSync('scratch/lesson-tools/Scripts/python.exe', ['-'], { input: program, encoding: 'utf8', timeout: 60000 });
assert.equal(run.status, 0, run.stderr);
const nativeLines = run.stdout.trim().split(/\r?\n/);
assert.equal(nativeLines.slice(0, 4).join('\n'), numpyFoundationsExamples.loopReference.output);
const files = ['src/learn/data/numpy-foundations-examples.js', 'src/learn/data/topics/numpy-arrays-broadcasting-vectorization.jsx'];
const report = { status: 'passed', checkedAt: new Date().toISOString(), reviewer: 'integration owner; not the NumPy author', scope: 'New loop-to-library addition only; no new full-module correctness claim', checks: ['Displayed output matches native execution', 'Independent fsum and scalar-coordinate oracle on three shapes', 'Input remains unchanged and output does not alias it', 'Joint input/offset translation leaves corrected values unchanged'], sourceHashes: Object.fromEntries(files.map(file => [file, createHash('sha256').update(fs.readFileSync(file)).digest('hex')])) };
fs.writeFileSync('docs/teaching/evidence/implementation-depth-numpy-independent.json', JSON.stringify(report, null, 2) + '\n');
const scalingCode = scalingExamples.fittedStateParity.code + `
permutation = [2, 0, 3, 1]
reordered_state = fit_preparation(train[permutation], [train_categories[i] for i in permutation])
for key in state:
    np.testing.assert_array_equal(reordered_state[key], state[key])
np.testing.assert_allclose(transform_preparation(state, query[::-1], query_categories[::-1]), manual[::-1], atol=1e-12)
shift = np.array([10., 20., 0.])
translated_state = fit_preparation(train + shift, train_categories)
np.testing.assert_allclose(transform_preparation(translated_state, query + shift, query_categories), manual, atol=1e-12)
query_missing = np.array([[15., 500., np.nan]])
mask = np.isnan(query_missing)
extension = np.column_stack([transform_preparation(state, query_missing, ['b']), mask.astype(float)])
np.testing.assert_allclose(extension[0], [-1/np.sqrt(2),np.sqrt(2),0,0,1,0,0,0,1], atol=1e-12)
assert all(np.array_equal(state[key], before) for key, before in snapshot.items())
print('Permutation, translation, missingness extension and state ownership passed')
`;
const scalingRun = spawnSync('scratch/lesson-tools/Scripts/python.exe', ['-'], { input: scalingCode, encoding: 'utf8', timeout: 60000 });
assert.equal(scalingRun.status, 0, scalingRun.stderr);
assert.equal(scalingRun.stdout.trim().split(/\r?\n/).slice(0, 5).join('\n'), scalingExamples.fittedStateParity.expected);
const scalingFiles = ['src/learn/data/scaling-examples.js', 'src/learn/data/topics/feature-scaling-encoding-imputation.jsx'];
const scalingReport = { status: 'passed', checkedAt: new Date().toISOString(), reviewer: 'integration owner; not the Feature Scaling author', scope: 'New fitted-state implementation and practice only', checks: ['Exact displayed output replay', 'Training and query permutation invariance', 'Joint train/query translation leaves standardized output unchanged', 'Missingness extension has independently calculated coordinates and indicators', 'No transform mutates fitted state'], sourceHashes: Object.fromEntries(scalingFiles.map(file => [file, createHash('sha256').update(fs.readFileSync(file)).digest('hex')])) };
fs.writeFileSync('docs/teaching/evidence/implementation-depth-scaling-independent.json', JSON.stringify(scalingReport, null, 2) + '\n');
console.log(JSON.stringify({ numpy: report.status, scaling: scalingReport.status }));

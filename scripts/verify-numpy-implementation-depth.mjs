import fs from 'node:fs';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { parse } from '@babel/parser';
import { numpyFoundationsExamples } from '../src/learn/data/numpy-foundations-examples.js';

const example = numpyFoundationsExamples.loopReference;
const program = `import contextlib, io, json, sys
import numpy as np
packet = json.load(sys.stdin)
namespace = {}
output = io.StringIO()
with contextlib.redirect_stdout(output):
    exec(packet['code'], namespace)
assert output.getvalue().strip() == packet['output'].strip()
f = namespace['calibrate_sensor_means']
raw = np.array([[2., -4., 0.], [6., 8., 10.]])
original = raw.copy()
corrected, means = f(raw, [1., -2., .5])
np.testing.assert_allclose(corrected, [[1., -2., -.5], [5., 10., 9.5]])
np.testing.assert_allclose(means, [3., 4., 4.5])
assert np.array_equal(raw, original)
assert not np.shares_memory(corrected, raw)
single, single_mean = f([[0.]], [0.])
assert single.shape == (1, 1) and single_mean.tolist() == [0.]
for bad_raw, bad_offsets in [([], []), ([[]], []), ([1, 2], [0]), ([[1, 2]], [0]), ([[1]], [[0]]), ([[float('nan')]], [0]), ([[1]], [float('inf')])]:
    try:
        f(bad_raw, bad_offsets)
    except ValueError:
        pass
    else:
        raise AssertionError('invalid contract accepted')
print(json.dumps({'status': 'passed', 'checks': ['complete displayed program matches output', 'changed non-square table and signed offsets', 'unchanged input and independent output storage', 'one-cell zero case', 'seven invalid contract cases'], 'python': sys.version.split()[0], 'numpy': np.__version__, 'stdout': output.getvalue()}))
`;
const result = spawnSync('scratch/lesson-tools/Scripts/python.exe', ['-c', program], {input:JSON.stringify(example), encoding:'utf8'});
if (result.status !== 0) throw new Error(result.stderr || result.stdout);
const files = ['src/learn/data/numpy-foundations-examples.js', 'src/learn/data/topics/numpy-arrays-broadcasting-vectorization.jsx', 'scripts/verify-numpy-implementation-depth.mjs'];
for (const file of files.filter(file => !file.startsWith('scripts/'))) parse(fs.readFileSync(file, 'utf8'), {sourceType:'module', plugins:['jsx']});
const receipt = {...JSON.parse(result.stdout), timestamp:new Date().toISOString(), scope:'bounded implementation-depth repair; browser/build integration is separate', sourceHashes:Object.fromEntries(files.map(file => [file, createHash('sha256').update(fs.readFileSync(file)).digest('hex')]))};
fs.writeFileSync('docs/teaching/implementation-depth/numpy-repair.json', JSON.stringify(receipt,null,2) + '\n');
console.log('NumPy implementation-depth example: complete native replay, changed shape, ownership and invalid-contract checks passed; body/examples parse.');

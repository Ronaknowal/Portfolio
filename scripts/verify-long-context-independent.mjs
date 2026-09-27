// Complementary reviewer check: fresh altered-input native inference and baseline
// export parity. This does not replay training or replace browser inspection.
import fs from 'node:fs';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { trajectoryForward, meanTrajectory } from '../src/learn/data/long-context-models.js';

const base = 'public/learn-code/long-context-sequence-models-transformer-xl-griffin-perceiver';
const output = 'docs/teaching/evidence/long-context/independent-checks.json';
fs.writeFileSync(output, JSON.stringify({ status: 'incomplete', reason: 'Review running' }) + '\n');
const source = `${base}/trajectory-models.json`;
const data = JSON.parse(fs.readFileSync(source, 'utf8'));
const cases = [];
for (const specimen of [data.specimens[0], data.specimens[17], data.specimens.at(-1)]) {
  const records = specimen.records.map((r, i) => ({ ...r, valid: i % 3 === 0,
    x: i === 21 ? .9 : r.x, y: i === 21 ? .05 : r.y }));
  cases.push(records, records.map((r, i) => ({ ...r, position: records[records.length - 1 - i].position })));
}
const python = String.raw`
import sys,json,runpy
from pathlib import Path
import numpy as np
import torch
from scipy.special import softmax
torch.set_num_threads(2)
p=Path('public/learn-code/long-context-sequence-models-transformer-xl-griffin-perceiver')
n=runpy.run_path(str(p/'latent_trajectory_classifier.py'))
saved=np.load(p/'small-fits.npz',allow_pickle=False)
cases=json.load(sys.stdin)
results=[]
for rows in cases:
    frames=torch.tensor([[[2*r['x']-1,2*r['y']-1,r['position']] for r in rows]],dtype=torch.float32)
    mask=torch.tensor([[r['valid'] for r in rows]],dtype=torch.bool)
    out={}
    for count in [1,4]:
        for seed in [11,29]:
            key=f'latents{count}_seed{seed}'
            model=n['LatentClassifier'](count)
            model.load_state_dict({k.split('::')[1]:torch.from_numpy(saved[k]) for k in saved.files if k.startswith(key+'::') and k.split('::')[1] in model.state_dict()})
            with torch.no_grad():
                logits,_=model.eval()(frames,mask)
            out[key]=logits[0].tolist()
    xy=np.array([[2*r['x']-1,2*r['y']-1] for r in rows if r['valid']])
    out['mean']=softmax(saved['mean_coef']@xy.mean(0)+saved['mean_intercept']).tolist()
    results.append(out)
print(json.dumps(results))
`;
const result = spawnSync('scratch/lesson-tools/Scripts/python.exe', ['-c', python],
  { input: JSON.stringify(cases), encoding: 'utf8', maxBuffer: 2 ** 20 });
assert.equal(result.status, 0, result.stderr);
const native = JSON.parse(result.stdout);
let maxLogitError = 0, maxBaselineError = 0;
for (let i = 0; i < cases.length; i++) {
  for (const [name, parameters] of Object.entries(data.models)) {
    const actual = trajectoryForward(parameters, cases[i]).logits;
    actual.forEach((value, j) => { maxLogitError = Math.max(maxLogitError, Math.abs(value - native[i][name][j])); });
  }
  meanTrajectory(data.mean, cases[i]).probabilities.forEach((value, j) => {
    maxBaselineError = Math.max(maxBaselineError, Math.abs(value - native[i].mean[j]));
  });
}
assert.ok(maxLogitError < 5e-5, `Altered masked/tag inputs: ${maxLogitError}`);
assert.ok(maxBaselineError < 1e-12, `Mean-coordinate baseline: ${maxBaselineError}`);
const files = [source, `${base}/small-fits.npz`, `${base}/latent_trajectory_classifier.py`,
  'src/learn/data/long-context-models.js', 'scripts/verify-long-context-independent.mjs'];
const report = { status: 'passed', reviewer: 'state_space_implementation',
  checks: ['24 fresh PyTorch masked/coordinate-edited/position-tag reassigned model cases',
    'Six NumPy/SciPy baseline results from NPZ, compared against JSON-export JS baseline'],
  maxLogitError, maxBaselineError,
  files: Object.fromEntries(files.map(path => [path, crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex')])) };
fs.writeFileSync(output, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report));

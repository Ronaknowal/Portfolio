import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import * as models from '../src/learn/data/algebra-functions-models.js';
import { algebraFunctionsExamples } from '../src/learn/data/algebra-functions-examples.js';

const directory = path.resolve('scratch/algebra-functions-independent');
fs.mkdirSync(directory, { recursive: true });
const author = JSON.parse(fs.readFileSync('docs/teaching/evidence/algebra-functions-author-review.json', 'utf8'));
for (const source of author.production) {
  assert.equal(crypto.createHash('sha256').update(fs.readFileSync(source.path)).digest('hex'), source.sha256);
}
const growth = [];
for (const rate of [-0.5, -0.2, 0, 0.1, 0.2, 0.5, 1]) {
  for (const factor of [0.13, 0.7, 1, 1.7, 3.4, 7.9]) {
    growth.push(models.growthState(rate, 4, factor));
  }
}
const compositions = Array.from({ length: 33 }, (_, i) => models.compositionState((i - 16) / 4));
const probes = [];
for (const value of [-3.75, -2.25, -0.25, 0, 0.25, 2.25, 3.75]) {
  for (const kind of ['affine', 'square', 'reciprocal']) {
    for (const restricted of [false, true]) probes.push(models.functionProbeState(kind, value, restricted));
  }
}
const logStates = [0.5, 2, 10].flatMap(base => [-2.75, -1.25, -0.25, 0.75, 1.75, 2.75].map(exponent => models.logarithmState(base, exponent)));
fs.writeFileSync(path.join(directory, 'source-and-models.json'), JSON.stringify({
  at: new Date().toISOString(), authorFrozenAt: author.frozenAt, production: author.production,
  growth, compositions, probes, logStates, examples: algebraFunctionsExamples,
}, null, 2));
console.log(JSON.stringify({ sourceHashesMatch: true, growth: growth.length, compositions: compositions.length, probes: probes.length, logStates: logStates.length, programs: algebraFunctionsExamples.length }));

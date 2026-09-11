// Independent final-source review; preserve the author's first freeze before any amendment.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import * as models from '../src/learn/data/real-analysis-models.js';
import { realAnalysisExamples } from '../src/learn/data/real-analysis-examples.js';

const directory = 'scratch/real-analysis-independent';
fs.mkdirSync(directory, { recursive: true });
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const packetPath = 'docs/teaching/evidence/real-analysis-author-review.json';
const packet = JSON.parse(fs.readFileSync(packetPath, 'utf8'));
const archive = `${directory}/original-author-freeze`;
if (!fs.existsSync(archive)) {
  fs.mkdirSync(archive, { recursive: true });
  fs.copyFileSync(packetPath, `${archive}/author-review.json`);
  for (const source of packet.sources) {
    assert.equal(hash(source.path), source.sha256);
    const destination = `${archive}/${source.path}`;
    fs.mkdirSync(path.dirname(destination), { recursive: true });
    fs.copyFileSync(source.path, destination);
  }
}
const payload = {
  checkedAt: new Date().toISOString(),
  sources: packet.sources.map(source => ({ path: source.path, sha256: hash(source.path) })),
  examples: realAnalysisExamples,
  tails: [], brackets: [], blocks: [], triangles: [], series: [], bernstein: [], writers: [],
};
for (const [p, q] of [[3, 29], [7, 57], [9, 27], [2, 997], [1, 999], [19, 4]]) {
  for (const index of [1, Math.max(1, Math.floor(q / p) - 1), Math.max(1, Math.floor(q / p)), Math.max(1, Math.floor(q / p) + 1)]) {
    payload.tails.push({ p, q, index, ...models.sequenceTail(p, q, index) });
  }
}
for (const target of [2, 3, 5]) for (const steps of [1, 7, 13, 16, 23, 24]) {
  payload.brackets.push(models.dyadicBracket(target, steps));
}
for (const n of [3, 7, 31, 127, 509]) for (const family of ['harmonic', 'telescoping']) {
  payload.blocks.push(models.cauchyBlock(n, family));
}
for (const n of [3, 7, 19, 97]) for (const scaling of ['unit-height', 'unit-area', 'shrinking-height']) {
  for (const grid of [7, 23]) payload.triangles.push(models.triangleFamily(n, scaling, grid, 1 / 7));
}
for (const n of [3, 17, 63, 257, 1023]) for (const x of [-1, -0.875, -0.125, 0, 0.375, 0.875, 1]) {
  payload.series.push(models.inverseSquareSeries(n, x));
}
for (const n of [3, 11, 37, 127, 251]) for (const x of [0, 0.0625, 0.3125, 0.5, 0.9375, 1]) {
  for (const corner of [0.1875, 0.6875]) {
    payload.bernstein.push(models.bernsteinApproximation(n, x, corner, 2.5));
  }
}
for (const block of [0, 3, 7, 12]) for (const observer of [0, 1 / 16, 7 / 16, 15 / 16]) {
  const j = Math.floor(observer * 2 ** block);
  for (const position of [...new Set([0, j, Math.max(0, j - 1), 2 ** block - 1])]) {
    payload.writers.push(models.typewriterInterval(block, position, observer));
  }
}
fs.writeFileSync(`${directory}/payload.json`, JSON.stringify(payload, null, 2));
console.log('Archived author freeze and exported independent changed inputs.');

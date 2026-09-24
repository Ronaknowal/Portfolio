import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import * as models from '../src/learn/data/abstract-algebra-models.js';
import { abstractAlgebraExamples } from '../src/learn/data/abstract-algebra-examples.js';

const directory = 'scratch/abstract-algebra-independent';
const authorPath = 'docs/teaching/evidence/abstract-algebra-author-review.json';
const hash = file => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const author = JSON.parse(fs.readFileSync(authorPath, 'utf8'));
const originalSources = author.sources || author.productionSources;
const arrowAmendment = { path: 'src/learn/components/lesson-labs/AbstractAlgebraLabs.jsx', sha256: '9ea9a96115ee8d32e424eeb687764c5853dc02945cbb019a726b9429f42cf0a0' };
const sources = originalSources.map(source => source.path === arrowAmendment.path ? { ...source, authorSha256: source.sha256, sha256: arrowAmendment.sha256, amended: true } : source);
assert.equal(sources.length, 6);
fs.mkdirSync(directory, { recursive: true });
const baselinePath = `${directory}/author-baseline.json`;
if (!fs.existsSync(baselinePath)) {
  fs.copyFileSync(authorPath, baselinePath);
  for (const source of originalSources) {
    assert.equal(hash(source.path), source.sha256, 'Original author bytes must be archived before any amendment.');
    const archivedPath = path.join(directory, 'author-sources', source.path);
    fs.mkdirSync(path.dirname(archivedPath), { recursive: true });
    fs.copyFileSync(source.path, archivedPath);
  }
}
assert.equal(hash(baselinePath), hash(authorPath));
for (const source of sources) {
  assert.equal(hash(source.path), source.sha256, source.path);
  assert.equal(hash(path.join(directory, 'author-sources', source.path)), source.authorSha256 || source.sha256);
}
const basis = Array.from({ length: 16 }, (_, index) => Array.from({ length: 4 }, (_, i) => Array.from({ length: 4 }, (_, j) => Number(i * 4 + j === index))));
const changedMatrices = Array.from({ length: 20 }, (_, sample) => Array.from({ length: 4 }, (_, i) => Array.from({ length: 4 }, (_, j) => (((sample + 3) * (i + 7) * (j + 2)) % 129 - 64) / 4)));
const projections = [false, true].map(rotationsOnly => ({
  rotationsOnly,
  columns: basis.map(matrix => models.averageSquareMap(matrix, rotationsOnly).flat()),
  changed: changedMatrices.map(matrix => ({ matrix, averaged: models.averageSquareMap(matrix, rotationsOnly) })),
}));
const states = [];
for (const values of [[-20, 0.25, 19.75, -7.5], [0, 0, 0, 0], [20, -20, 20, -20]]) {
  for (const mode of ['raw', 'rotations', 'averaged', 'tied', 'shift']) {
    for (const g of [1, 3, 5, 7]) states.push(models.equivarianceState(values, mode, g, [-3.75, 2.25, -0.5]));
  }
}
const subgroups = Array.from({ length: 256 }, (_, mask) => Array.from({ length: 8 }, (_, g) => g).filter(g => mask & (1 << g))).map(generators => ({ generators, generated: models.generatedSubgroup(generators) }));
const uniqueSubgroups = [...new Map(subgroups.map(row => [row.generated.join(','), row.generated])).values()];
const payload = {
  preparedAt: new Date().toISOString(), sources, authorPacket: { path: authorPath, sha256: hash(authorPath) },
  reviewAmendment: 'Move three reverse-cycle arrow endpoints outside destination circles; models, examples and body unchanged.',
  examples: abstractAlgebraExamples, projections, states,
  subgroups: uniqueSubgroups.map(subgroup => ({ subgroup, cosets: models.leftCosets(subgroup) })),
  fourColorCounts: [false, true].map(rotationsOnly => models.fixedColoringCount(4, rotationsOnly)),
};
fs.writeFileSync(`${directory}/payload.json`, JSON.stringify(payload, null, 2));
console.log(`Prepared ${projections.length} complete projection operators, ${changedMatrices.length * 2} changed matrices and ${states.length} changed sensor states from final production exports.`);

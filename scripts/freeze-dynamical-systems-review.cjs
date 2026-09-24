const fs = require('node:fs');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const hash = file => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const read = file => JSON.parse(fs.readFileSync(file, 'utf8'));
const browser = read('scratch/dynamical-systems-browser/results.json');
const models = read('scratch/dynamical-systems-review/model-results.json');
const native = read('scratch/dynamical-systems-review/native-independent-results.json');
assert.equal(browser.passed, true);
for (const source of browser.sources) assert.equal(hash(source.file), source.sha256, source.file);
assert.equal(models.sourceSha256, hash('src/learn/data/dynamical-systems-models.js'));
assert.equal(native.examplesSha256, hash('src/learn/data/dynamical-systems-examples.js'));
const files = [...browser.sources.map(source => source.file), 'src/learn/data/curriculum/blueprints/dynamical-systems-theory-chaos.js'];
const screenshots = ['state-figure-320.png', 'equilibrium-branches-390.png', 'bifurcation-1440.png', 'reading-8-320.png', 'phase-line-1440.png', 'hopf-1440.png', 'folding-320.png', 'lorenz-1440.png', 'numerical-energy-1440.png', 'critical-sensitivity-320.png', 'reading-1-390.png'];
const result = {
  topicId: 'dynamical-systems-theory-chaos',
  authorFrozenAt: new Date().toISOString(),
  status: 'author-verified; independent review and production integration pending',
  production: files.map(file => ({ path: file, sha256: hash(file), bytes: fs.statSync(file).size })),
  preservation: read('docs/teaching/evidence/dynamical-systems-original-content.json'),
  modelEvidence: models, nativeEvidence: native, browserEvidence: browser,
  screenshotsActuallyOpened: screenshots.map(name => {
    const file = 'scratch/dynamical-systems-browser/' + name;
    return { file, sha256: hash(file), scope: 'Actual image opened by root; finite chosen state, not a user learning study' };
  }),
  limitations: ['Bounded authored model and native inputs', 'Short-time ODE comparison does not certify long-time chaotic trajectories', 'Large labelled plots scroll horizontally on mobile', 'No full-video viewing, deployment or user-acceptance claim'],
};
fs.writeFileSync('docs/teaching/evidence/dynamical-systems-author-review.json', JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify({ topicId: result.topicId, authorFrozenAt: result.authorFrozenAt, production: result.production }, null, 2));

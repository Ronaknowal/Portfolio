const fs = require('node:fs');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const destination = 'docs/teaching/evidence/decision-theory-independent-review.json';
assert(!fs.existsSync(destination), 'Independent closure is immutable; use an amendment record.');
const digest = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const entry = file => ({ path: file, sha256: digest(file), bytes: fs.statSync(file).size });
const directory = 'scratch/decision-theory-independent';
const native = JSON.parse(fs.readFileSync(`${directory}/native-results.json`, 'utf8'));
const browser = JSON.parse(fs.readFileSync(`${directory}/browser-results.json`, 'utf8'));
assert(native.passed && browser.passed);
assert.deepEqual(native.sources, browser.sources);
for (const source of native.sources) {
  assert.equal(digest(source.path), source.sha256, source.path);
  assert.equal(digest(`${directory}/author-sources/${source.path}`), source.sha256);
}
const images = ['changed-loss-envelope', 'changed-state-weighting', 'changed-quantity-criterion', 'fallback-sampling-contract', 'three-slot-allocation', 'changed-information-timing', 'off-optimum-mixture', 'changed-partial-atom', 'changed-capacity-three-branches', 'inline-0', 'inline-1', 'changed-practice', 'actual-output', 'alternate-resources'].map(name => `${name}-320.png`);
images.push('changed-capacity-three-branches-1440.png');
const result = {
  reviewedAt: new Date().toISOString(), topicId: 'decision-theory-risk-cost-sensitive-decisions',
  status: 'independent review closed; no unresolved material findings; production integration and user acceptance separate',
  sources: native.sources,
  authorPacket: entry('docs/teaching/evidence/decision-theory-author-review.json'),
  preservedAuthorPacket: entry(`${directory}/author-baseline.json`),
  productionAmendmentsAfterAuthorFreeze: [],
  native: { file: entry(`${directory}/native-results.json`), result: native },
  browser: { file: entry(`${directory}/browser-results.json`), result: browser },
  finalImagesActuallyOpened: images.map(name => entry(`${directory}/${name}`)),
  report: entry('docs/teaching/DECISION-THEORY-INDEPENDENT-REVIEW.md'),
  scripts: ['scripts/verify-decision-theory-independent.mjs', 'scripts/verify-decision-theory-independent.py', 'scripts/review-decision-theory-independent.cjs'].map(entry),
  limitations: ['Finite independent review; no learner study, universal arithmetic guarantee, user acceptance, production build or deployment.']
};
fs.writeFileSync(destination, JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify({ destination, reviewedAt: result.reviewedAt, sources: result.sources.length, imagesOpened: images.length }));

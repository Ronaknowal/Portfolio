const fs = require('node:fs');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const read = file => JSON.parse(fs.readFileSync(file));
const entry = path => ({ path, sha256: createHash('sha256').update(fs.readFileSync(path)).digest('hex'), bytes: fs.statSync(path).size });
const author = read('docs/teaching/evidence/sets-logic-author-review.json');
const native = read('scratch/sets-logic-independent/native-results.json');
const browser = read('scratch/sets-logic-independent/browser-results.json');
assert(native.passed && browser.passed);
for (const source of author.sources) assert.equal(entry(source.path).sha256, source.sha256);
assert.equal(browser.modelSha256, entry('src/learn/data/sets-logic-models.js').sha256);
const viewed = [
  'all-four-in-overlap-320', 'separate-witnesses-no-common-390', 'changed-transitivity-obligation-320',
  'missing-loop-390', 'incomparable-minimal-elements-320', 'changed-conjunction-argument-1440',
  'changed-square-border-320', 'changed-diagonal-escape-390', 'inline-0-320', 'inline-1-390',
  'inline-2-390', 'inline-3-320', 'inline-4-1440', 'representative-independence-practice-390',
];
const record = {
  topicId: 'sets-logic-relations-proof-techniques', reviewedAt: new Date().toISOString(),
  reviewer: 'root; independent of the topic author', status: 'closed; no remaining material finding within review scope',
  authorFrozenAt: author.authorFrozenAt, production: author.sources,
  sourceRead: 'Complete body/design/brief/models/eleven actual code strings/labs/CSS/eleven practice groups; final changed validator also inspected.',
  finding: { type: 'malformed sparse-array validation', resolved: true, priorModelSha256: 'd20028bb2576028eb9fbfd1d9df57611ae7cf7daab9112169a7358db9013de32', finalModelSha256: browser.modelSha256, originalEvidence: 'docs/teaching/evidence/sets-logic-author-review-before-sparse-validation.json' },
  native, browser,
  actualImagesOpened: viewed.map(name => entry('scratch/sets-logic-independent/' + name + '.png')),
  support: ['docs/teaching/SETS-LOGIC-INDEPENDENT-REVIEW.md', 'scripts/verify-sets-logic-independent.mjs', 'scripts/verify-sets-logic-independent.py', 'scripts/review-sets-logic-independent.cjs'].map(entry),
  limits: ['Selected primary-source passages only; no full book/video viewing.', 'Finite complementary cases do not replace the mathematical proof review.', 'Author full browser checks are attributed, not claimed as independently repeated.', 'Production integration and user acceptance are separate.'],
};
fs.writeFileSync('docs/teaching/evidence/sets-logic-independent-review.json', JSON.stringify(record, null, 2) + '\n');
console.log(JSON.stringify({ reviewedAt: record.reviewedAt, sources: record.production.length, imagesOpened: viewed.length }));

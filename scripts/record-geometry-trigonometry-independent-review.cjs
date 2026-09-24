const fs = require('node:fs');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const read = path => JSON.parse(fs.readFileSync(path));
const entry = path => ({ path, sha256: createHash('sha256').update(fs.readFileSync(path)).digest('hex'), bytes: fs.statSync(path).size });
const author = read('docs/teaching/evidence/geometry-trigonometry-author-review.json');
const native = read('scratch/geometry-trigonometry-independent/native-results.json');
const browser = read('scratch/geometry-trigonometry-independent/browser-results.json');
assert(native.passed && browser.passed);
for (const source of [...author.production, ...browser.sources]) assert.equal(entry(source.path).sha256, source.sha256);
const viewed = [
  'reflex-sector-320', 'smaller-similar-triangle-390', 'third-quadrant-components-320',
  'actual-ssa-circle-390', 'passive-quarter-turn-320', 'active-quarter-turn-1440',
  'inline-0-390', 'inline-1-320', 'inline-3-320', 'inline-4-390',
  'invalid-input-retains-bearing-320', 'changed-ssa-practice-390',
];
const record = {
  topicId: author.topicId, reviewedAt: new Date().toISOString(),
  reviewer: 'root; independent of topic author', status: 'closed; no remaining material finding within review scope',
  authorFrozenAt: author.frozenAt, production: author.production,
  sourceRead: 'Complete ten-section body/design/brief/models/nine actual programs and outputs/labs/CSS/two checkpoints/twelve practice groups.',
  preFreezeCopyCorrections: ['Vectors is not called later in the module.', 'The arc uses fixed drawing scale, not a responsive CSS-pixel guarantee.'],
  native, browser,
  actualImagesOpened: viewed.map(name => entry('scratch/geometry-trigonometry-independent/' + name + '.png')),
  support: ['docs/teaching/GEOMETRY-TRIGONOMETRY-INDEPENDENT-REVIEW.md', 'scripts/verify-geometry-trigonometry-independent.mjs', 'scripts/verify-geometry-trigonometry-independent.py', 'scripts/review-geometry-trigonometry-independent.cjs'].map(entry),
  limits: ['Finite high-precision cases do not establish an unrestricted floating-point bound.', 'SVG arc sampling is checked with explicit drawing tolerances.', 'Author full browser checks remain attributed.', 'Production integration and user acceptance are separate.'],
};
fs.writeFileSync('docs/teaching/evidence/geometry-trigonometry-independent-review.json', JSON.stringify(record, null, 2) + '\n');
console.log(JSON.stringify({ reviewedAt: record.reviewedAt, sources: record.production.length, imagesOpened: viewed.length }));

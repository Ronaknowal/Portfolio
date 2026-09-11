const fs = require('node:fs');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const read = file => JSON.parse(fs.readFileSync(file, 'utf8'));
const hash = file => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const bind = path => ({ path, sha256: hash(path) });
const output = 'docs/teaching/evidence/linked-traversal-extension-independent-review.json';
assert(!fs.existsSync(output), 'Preserve the immutable independent review.');
const authorPath = 'docs/teaching/evidence/linked-traversal-extension-author-review.json';
const author = read(authorPath);
const nativePath = 'scratch/linked-extension-independent/native-results.json';
const browserPath = 'scratch/linked-extension-independent/browser/results.json';
const native = read(nativePath), browser = read(browserPath);
assert(native.passed && browser.passed);
const sources = author.productionSources;
assert.equal(sources.length, 11);
for (const source of sources) {
  assert.equal(hash(source.path), source.sha256, source.path);
  assert.equal(native.sources.find(row => row.path === source.path).sha256, source.sha256);
  assert.equal(browser.sources.find(row => row.path === source.path).sha256, source.sha256);
}
assert.equal(native.authorPacketSha256, hash(authorPath));
const opened = ['changed-cycle-meeting-1440', 'changed-cycle-entry-320', 'changed-even-cut-320', 'signed-resolution-320', 'changed-histogram-geometry-1440', 'changed-histogram-geometry-320', 'changed-inclusive-result-320', 'alias-practice-320', 'tied-rectangle-practice-320', 'reading-9-1440', 'reading-11-320', 'reading-12-320'];
const packet = {
  reviewedAt: new Date().toISOString(), status: 'independent review closed; production integration pending',
  topicId: 'linked-lists-stacks-queues', reviewer: '/root', authorPacket: bind(authorPath), sources,
  productionAmendments: [], narrative: bind('docs/teaching/LINKED-TRAVERSAL-MONOTONIC-INDEPENDENT-REVIEW.md'),
  native: { record: bind(nativePath), checkedAt: native.checkedAt, checks: native.checks },
  browser: { record: bind(browserPath), checkedAt: browser.checkedAt, results: browser.results },
  scripts: ['scripts/verify-linked-extension-independent.mjs', 'scripts/verify-linked-extension-independent.py', 'scripts/review-linked-extension-independent.cjs'].map(bind),
  openedImages: opened.map(name => bind(`scratch/linked-extension-independent/browser/${name}.png`)),
  conservation: 'Original author archive and conservation attribution retained; original ten practice placements include the documented intentional LC141 transfer-note update.',
  limits: ['Scoped independent read and complementary verification, not a repeated full audit of every earlier implementation.', 'No judge submissions, observed learner study, physical-device or full screen-reader study.', 'Production integration and user acceptance remain separate.'],
};
fs.writeFileSync(output, JSON.stringify(packet, null, 2) + '\n');
console.log(JSON.stringify({ reviewedAt: packet.reviewedAt, sources: sources.length, openedImages: opened.length }));

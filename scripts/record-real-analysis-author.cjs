const fs = require('node:fs');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');

const destination = 'docs/teaching/evidence/real-analysis-author-review.json';
assert(!fs.existsSync(destination), 'Preserve the frozen author packet; amendments need a separate record.');
const digest = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const entry = file => ({ path: file, sha256: digest(file), bytes: fs.statSync(file).size });
const nativePath = 'scratch/real-analysis-verification/results.json';
const browserPath = 'scratch/real-analysis-browser/results.json';
const native = JSON.parse(fs.readFileSync(nativePath, 'utf8'));
const browser = JSON.parse(fs.readFileSync(browserPath, 'utf8'));
assert(browser.passed);
assert.deepEqual(native.sourceHashes, browser.sourceHashes);
for (const [file, expected] of Object.entries(native.sourceHashes)) assert.equal(digest(file), expected, file);
const names = ['investigation-0', 'investigation-1', 'investigation-2', 'investigation-3', 'missed-unit-area-peak', 'both-derivative-bounds', 'changed-series-endpoint', 'changed-corner-weights', 'half-open-boundary-hit', 'reading-figure-2', 'reading-figure-3', 'equation-13', 'changed-practice-13', 'changed-output', 'learning-resources'];
const packet = {
  authorFrozenAt: new Date().toISOString(),
  topicId: 'real-analysis-sequences-modes-of-convergence',
  status: 'author-verified; independent review, production integration and user acceptance separate',
  sources: Object.keys(native.sourceHashes).map(entry),
  records: { native: { file: entry(nativePath), result: native }, browser: { file: entry(browserPath), result: browser } },
  finalImagesActuallyOpened: names.map(name => entry(`scratch/real-analysis-browser/${name}-320.png`)),
  design: entry('docs/teaching/REAL-ANALYSIS-LESSON-DESIGN.md'),
  report: entry('docs/teaching/REAL-ANALYSIS-VERIFICATION.md'),
  reviewAttention: ['Recheck the painted minus sign in the changed native series select; selected DOM value and calculated result agree.'],
  limitations: ['Author checks extend preserved independent design fixtures; whole-lesson independent review is still pending.', 'No complete network-warning audit or observed learner study is claimed.', 'No production integration or user acceptance is inferred from publication.']
};
fs.writeFileSync(destination, JSON.stringify(packet, null, 2) + '\n');
console.log(JSON.stringify({ destination, authorFrozenAt: packet.authorFrozenAt, sources: packet.sources.length, imagesOpened: packet.finalImagesActuallyOpened.length }));

import fs from 'node:fs';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import { pdeExamples } from '../src/learn/data/pde-examples.js';
const hash = path => crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex');
const previousPath = 'docs/teaching/evidence/pde-author-review-before-independent-amendment.json';
const packet = JSON.parse(fs.readFileSync(previousPath));
const original = fs.readFileSync('docs/teaching/archive/pde-before-independent-amendment/src/learn/data/pde-examples.js.txt', 'utf8');
const oldExamples = JSON.parse(original.slice(original.indexOf('{'), original.lastIndexOf('}') + 1));
for (const [name, example] of Object.entries(pdeExamples)) {
  assert.equal(example.expected, oldExamples[name].expected, `Original stdout: ${name}`);
  if (name !== 'wave') assert.deepEqual(example, oldExamples[name], `Unchanged original program: ${name}`);
}
const browserPath = 'docs/teaching/evidence/pde-independent-amendment-browser.json';
const browser = JSON.parse(fs.readFileSync(browserPath));
const native = JSON.parse(fs.readFileSync('scratch/pde-verification/results.json'));
const boundaryPath = 'docs/teaching/evidence/pde-wave-boundary-verification.json';
const boundary = JSON.parse(fs.readFileSync(boundaryPath));
assert.equal(browser.passed, true);
for (const source of browser.sourceHashes) assert.equal(hash(source.path), source.sha256);
for (const [source, expected] of Object.entries(native.numericSourceHashes)) assert.equal(hash(source), expected);
for (const [source, expected] of Object.entries(boundary.sourceHashes)) assert.equal(hash(source), expected);
const openedImages = ['arithmetic-final-320', 'maximum-final-320', 'units-final-320', 'wave-final-390', 'wave-final-1440', 'arithmetic-final-1440', 'units-final-390'].map(name => {
  const path = `scratch/pde-wave-boundary/${name}.png`;
  return { path, sha256: hash(path), reviewed: 'Actually opened by the author after final amendment; no claim to have opened all captured images.' };
});
packet.previousAuthorFreeze = { path: previousPath, sha256: hash(previousPath), frozenAt: packet.frozenAt };
packet.originalFullBrowser = packet.browser;
packet.originalOpenedImages = packet.openedImages;
packet.sourceHashes = browser.sourceHashes;
packet.frozenAt = new Date().toISOString();
packet.openedImages = openedImages;
fs.copyFileSync('scratch/pde-verification/results.json', 'docs/teaching/evidence/pde-native-verification.json');
packet.native = { path: 'docs/teaching/evidence/pde-native-verification.json', sha256: hash('docs/teaching/evidence/pde-native-verification.json'), checkedAt: native.checkedAt, checks: native.checks, maxScaledError: native.maxScaledError };
packet.browser = { path: browserPath, sha256: hash(browserPath), checkedAt: browser.checkedAt, scope: 'Focused final-source amendment review:24 wave states,15 current program/code/output checks, exact loaded-module boundary, keyboard reset, three repaired/new paragraphs, all22 existing formula widths and document/error checks per1440/390/320. The earlier complete176-state review is retained separately, not rerun or relabeled as final-source.', summaries: browser.results };
packet.independentReviewAmendment = {
  originals: 'docs/teaching/evidence/pde-independent-amendment-originals.json',
  findings: [
    'Primitive subtraction could make a nonnegative wave velocity integral negative at a tiny support-edge time. Replaced with16 positive polynomial terms on a clipped offset interval; shifted bump factors preserve small endpoint distances in JS and displayed Python.',
    'Maximum-principle derivatives were only assumed in the interior. The proof now works on every T′<T and extends to T by continuity.',
    'Restored the physical coefficient s/(2k)=1 K/m² and declared SI coordinate values in the simplified rod formulas; decay-rate units are explicit.',
  ],
  conservation: 'Fourteen complete program objects and all15 expected stdout strings remain exact. Lab/CSS/brief hashes are unchanged. Body changes:two qualifications and one arithmetic explanation, plus LF line-ending normalization.',
  boundary: { path: boundaryPath, sha256: hash(boundaryPath), ...boundary },
};
fs.writeFileSync('docs/teaching/evidence/pde-author-review.json', JSON.stringify(packet, null, 2) + '\n');
console.log(JSON.stringify({ frozenAt: packet.frozenAt, sourceHashes: packet.sourceHashes, openedImages: openedImages.length }, null, 2));

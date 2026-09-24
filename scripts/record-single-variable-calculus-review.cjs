const fs = require('node:fs');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');

function fingerprint(path) {
  const bytes = fs.readFileSync(path);
  return { path, sha256: crypto.createHash('sha256').update(bytes).digest('hex'), bytes: bytes.length };
}
function record(path) {
  return { file: fingerprint(path), result: JSON.parse(fs.readFileSync(path, 'utf8')) };
}
const native = record('scratch/single-variable-calculus-verification/results.json');
const browser = record('scratch/single-variable-calculus-browser/results.json');
const reading = record('scratch/single-variable-calculus-browser/final-reading-results.json');
assert(native.result.passed && browser.result.passed && reading.result.passed);
const sources = native.result.sources.map(source => {
  const actual = fingerprint(source.path);
  assert.equal(actual.sha256, source.sha256, source.path);
  return actual;
});
assert.equal(sources.length, 6);
const destination = 'docs/teaching/evidence/single-variable-calculus-author-review.json';
assert(!fs.existsSync(destination), 'Do not overwrite a frozen author packet.');
const reviewedImageNames = [
  'cusp-tied-endpoints-390', 'log-outside-series-radius-390',
  'invalid-interval-retains-state-320',
];
const packet = {
  authorFrozenAt: new Date().toISOString(),
  topicId: 'single-variable-calculus-limits-derivatives-integrals',
  status: 'author-verified; independent closure, production integration and user acceptance separate',
  sources,
  records: { native, browser, finalReading: reading,
    formatConservation: record('scratch/single-variable-calculus-review/format-conservation.json') },
  openedFinalAmendmentImages: reviewedImageNames.map(name => fingerprint(`scratch/single-variable-calculus-browser/${name}.png`)),
  earlierVisualReview: 'The author opened all 24 narrow-screen equation images, the four inline figures, changed program/practice and selected lab images during authoring. The final reading run repeats geometry and exact displayed-program checks on the final sources; only the three explicitly listed final amendment images are claimed here as reopened after that rerun. The verification Markdown distinguishes unchanged regions from amended ones.',
  baseline: fingerprint('scratch/single-variable-calculus-original-plan.json'),
  teachingInventory: { sections: 14, investigations: 8, inlineFigures: 4, completePrograms: 15, changedPractice: 13, displayedEquations: 24 },
  limitations: [
    'Finite exact and high-precision cases support the implementation; the general mathematical proofs were read separately.',
    'Plots are finite sampled drawings. Limit guarantees and extrema classifications use their stated analytic or exact decimal rules, not pixel proximity.',
    'Taylor truncation bounds are evaluated approximately and do not bound floating-point roundoff or constitute outward-rounded certified intervals.',
    'Browser code examples are displayed Python programs; the page does not run arbitrary Python.',
    'Production route/loading integration, user acceptance and observed learning outcomes remain separate.'
  ]
};
fs.writeFileSync(destination, JSON.stringify(packet, null, 2) + '\n');
console.log(JSON.stringify({ destination, frozenAt: packet.authorFrozenAt, sources: sources.length,
  browserPassed: browser.result.passed, readingPassed: reading.result.passed }));

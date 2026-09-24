import fs from 'node:fs';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';

const hash = path => crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex');
const native = JSON.parse(fs.readFileSync('scratch/pde-verification/results.json', 'utf8'));
const browser = JSON.parse(fs.readFileSync('scratch/pde-browser/results.json', 'utf8'));
assert.equal(browser.passed, true);
assert.deepEqual(browser.results.map(row => row.width), [1440, 390, 320]);
for (const [path, expected] of Object.entries(native.numericSourceHashes)) assert.equal(hash(path), expected);
for (const row of browser.sourceHashes) assert.equal(hash(row.path), row.sha256);
const openedNames = [
  'inline-1-1440.png', 'inline-2-1440.png', 'reading-5-1440.png', 'default-5-1440.png',
  'reading-5-320.png', 'inline-2-320.png', 'default-3-390.png', 'inline-7-320.png',
  'burgers-fan-320.png', 'poisson-incompatible-390.png', 'heat-exact-initial-320.png', 'default-8-390.png',
  'reading-11-320.png', 'reading-9-390.png', 'inline-8-390.png', 'practice-open-320.png'
];
const openedImages = openedNames.map(name => {
  const path = `scratch/pde-browser/${name}`;
  return { path, sha256: hash(path), reviewed: 'Actually opened with view_image by the author after the final source changes; capture is not being counted as visual inspection by itself.' };
});
const destination = 'docs/teaching/evidence';
fs.mkdirSync(destination, { recursive: true });
fs.copyFileSync('scratch/pde-verification/results.json', `${destination}/pde-native-verification.json`);
fs.copyFileSync('scratch/pde-browser/results.json', `${destination}/pde-browser-review.json`);
const packet = {
  topicId: 'partial-differential-equations-conservation-boundary-conditions',
  title: 'Partial Differential Equations, Conservation & Boundary Conditions',
  modulePosition: 56,
  frozenAt: new Date().toISOString(),
  status: 'author-verified; independent and production integration reviews remain separate',
  sourceHashes: browser.sourceHashes,
  baseline: { path: 'docs/teaching/evidence/pde-original-plan.json', sha256: hash('docs/teaching/evidence/pde-original-plan.json'), disposition: 'Previously planned; no old authored body or program existed for this stable identity. All starting commitments, identity and module position retained.' },
  native: { path: `${destination}/pde-native-verification.json`, sha256: hash(`${destination}/pde-native-verification.json`), checkedAt: native.checkedAt, checks: native.checks, maxScaledError: native.maxScaledError },
  browser: { path: `${destination}/pde-browser-review.json`, sha256: hash(`${destination}/pde-browser-review.json`), checkedAt: browser.checkedAt, summaries: browser.results.map(({ width, states, disclosures, controls, programs, equations, documentOverflow, errors, warnings, failedRequests }) => ({ width, states, disclosures, controls, programs, equations, documentOverflow, errors, warnings, failedRequests })) },
  openedImages,
  sourceRead: ['Complete body and every displayed derivation', 'All 15 complete programs, questions, captured stdout and interpretations', 'Pure models and all visual/lab/CSS source', '14 changed exercises with hints and explained solutions', 'Approved design, individual brief and incoming/destination notes'],
  resolvedAuthorFindings: [
    'Kept the exact initial heat state separate and hid the positive-time control while initial data are active.',
    'Reversed Burgers data now draw diverging characteristics for an expansion jump; they do not reuse the compressive shock geometry.',
    'Factored thin-slice integral/flux differences before evaluation; neighboring floating-point endpoint regressions pass against high-precision identities.',
    'Actual screenshots exposed missing arrowheads on compound paths; each directed flux now has its own path and marker.',
    'Reflowed 15 formula blocks, including a second narrow pass for the Burgers specialization, without shrinking type. All 22 final blocks fit 320px.'
  ],
  checksDoNotEstablish: ['General PDE existence or convergence theorems beyond the locally stated proofs and hypotheses', 'Arbitrary floating-point accuracy outside the documented bounded model contracts', 'Numerical PDE discretization quality, solver convergence or production integration', 'A beginner user study, complete video viewing or user acceptance'],
  design: 'docs/teaching/PARTIAL-DIFFERENTIAL-EQUATIONS-LESSON-DESIGN.md',
  verification: 'docs/teaching/PARTIAL-DIFFERENTIAL-EQUATIONS-VERIFICATION.md'
};
fs.writeFileSync(`${destination}/pde-author-review.json`, JSON.stringify(packet, null, 2) + '\n');
console.log(JSON.stringify({ frozenAt: packet.frozenAt, sourceHashes: packet.sourceHashes, openedImages: openedImages.length }, null, 2));

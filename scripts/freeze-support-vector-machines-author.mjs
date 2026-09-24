import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
const sourcePaths = [
  'src/learn/data/topics/support-vector-machines-svm.jsx',
  'src/learn/data/support-vector-machines-models.js',
  'src/learn/data/support-vector-machines-examples.js',
  'src/learn/data/svm-validation-fixtures.js',
  'src/learn/components/lesson-labs/SupportVectorMachineLabs.jsx',
  'src/learn/components/lesson-labs/support-vector-machines-labs.css',
  'src/learn/data/curriculum/blueprints/support-vector-machines-svm.js',
];
const scratch = 'scratch/support-vector-machines-verification';
const destination = 'docs/teaching/evidence/support-vector-machines';
const packetPath = 'docs/teaching/evidence/support-vector-machines-author-review.json';
assert(!fs.existsSync(packetPath), 'An author packet already exists; preserve it before an amendment.');
fs.mkdirSync(destination, { recursive: true });
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const read = file => JSON.parse(fs.readFileSync(file));
const sourceHashes = sourcePaths.map(path => ({ path, sha256: hash(path) }));
const browser = read(`${scratch}/browser/results.json`);
assert.deepEqual(browser.sourceHashes, sourceHashes);
assert.equal(browser.records.length, 3);
assert(browser.records.every(record => !record.failure && !record.errors.length && !record.failedRequests.length && record.programs === 14));
const images = [
  ['margin-geometry-1440.png', 'Equal-scale normal, projection and score corridor at desktop size'],
  ['margin-geometry-320.png', 'Same mechanism and labels readable at 320 pixels'],
  ['support-motion-390.png', 'Moved point, old certificate failure and refitted support identities'],
  ['conflicting-bias-320.png', 'Hinge states and explicit undefined distance when w=0'],
  ['kernel-lift-1440.png', 'Input score field versus actual polynomial feature view'],
  ['kernel-contributions-390.png', 'Unclipped signed contribution scale and program transition'],
  ['zero-curvature-pair-320.png', 'Feasible pair geometry and linear objective gain'],
  ['measured-selection-1440.png', 'Actual selected model, measured grid and data-role control'],
  ['target-units-tube-390.png', 'Vertical target tolerance, dynamic plot and costs'],
  ['dual-reading-320.png', 'Final narrow dual derivation and proof prose'],
  ['regression-reading-390.png', 'Unit conversion explanation and visible runnable question'],
].map(([name, purpose]) => {
  const original = `${scratch}/browser/${name}`, stored = `${destination}/${name}`;
  fs.copyFileSync(original, stored);
  assert.equal(hash(original), hash(stored));
  return { path: stored, originalCapturePath: original, sha256: hash(stored), purpose, actuallyOpened: true };
});
const evidence = [
  ['design-fixtures.json', 'design-fixtures.json'],
  ['program-results.json', 'program-results.json'],
  ['model-results.json', 'model-results.json'],
  ['pair-update-results.json', 'pair-update-results.json'],
  ['browser/behavior-before-equation-fit.json', 'behavior-before-equation-fit.json'],
  ['browser/behavior-320-before-equation-fit.json', 'behavior-320-before-equation-fit.json'],
  ['browser/results.json', 'final-reading-results.json'],
].map(([original, filename]) => {
  const originalPath = `${scratch}/${original}`, target = `${destination}/${filename}`;
  fs.copyFileSync(originalPath, target);
  return { path: target, originalWorkingPath: originalPath, sha256: hash(target) };
});
const packet = {
  topicId: 'support-vector-machines-svm', title: 'Support Vector Machines (SVM)', moduleId: 'classical-ml', modulePosition: 5,
  status: 'author-reviewed; independent review, production integration and learner acceptance remain separate',
  frozenAt: new Date().toISOString(), sourceHashes,
  design: 'docs/teaching/SUPPORT-VECTOR-MACHINES-LESSON-DESIGN.md',
  verification: 'docs/teaching/SUPPORT-VECTOR-MACHINES-VERIFICATION.md',
  originalArchive: 'docs/teaching/archive/support-vector-machines-before-rewrite/manifest.json',
  evidence, openedImages: images,
  numerical: read(`${scratch}/model-results.json`),
  pairAmendment: { checkedAt: read(`${scratch}/pair-update-results.json`).executedAt, cases: 7, actualStdoutUnchanged: true, unaffectedProgramsRetained: 13 },
  browser: {
    behavioral: ['behavior-before-equation-fit.json', 'behavior-320-before-equation-fit.json'],
    coverage: '50 states, 35 focusable controls, 33 keyboard disclosures and 14 actual programs per width; 390/320 runs then exposed formula overflow, retained honestly in their records.',
    final: `${destination}/final-reading-results.json`,
    correctionApplicability: 'Final equations and all rendered program text checked at every width. Flat same-label tie corrected in JS and standalone Python; focused loaded-module and visible pair checks passed. Other lab logic and native program bytes unchanged from the cited passing behavior/numerical evidence.',
  },
  commands: ['scratch/lesson-tools/Scripts/python.exe scripts/build-support-vector-machines-examples.py', 'node scripts/verify-support-vector-machines-models.mjs', 'node scripts/verify-support-vector-machines-pair-update.mjs', 'node scripts/review-support-vector-machines-lesson.cjs', 'node scripts/review-support-vector-machines-lesson.cjs all reading'],
  limits: ['Finite numeric domains and CPU demonstration data, not production optimizer or universal generalization guarantees.', 'The MIT official transcript segment was read; no complete video viewing claim.', 'Final source review uses bounded relevant correction checks, not repeated unrelated historical suites.'],
  retirement: { exactDisposablePaths: [...['model-cases.json', 'browser/failure-1440.png', 'browser/failure-390.png', 'browser/failure-320.png'].map(file => `${scratch}/${file}`), ...images.map(image => image.originalCapturePath)], reason: 'Regenerable working cases and superseded failed screenshots retired; final meaningful captures relocated byte-for-byte to the recorded durable paths. Failure/repair JSON history is retained.' },
};
fs.writeFileSync(packetPath, JSON.stringify(packet, null, 2)+'\n');
console.log(JSON.stringify({ packet: packetPath, frozenAt: packet.frozenAt, sources: sourceHashes.length, openedImages: images.length }));

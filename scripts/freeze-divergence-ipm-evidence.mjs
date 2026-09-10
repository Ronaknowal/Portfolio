import fs from 'node:fs';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';

const read = path => JSON.parse(fs.readFileSync(path, 'utf8'));
const finalReading = read('scratch/divergence-ipm-browser/final-reading-results.json');
const sourceHashes = finalReading.hashes.map(item => {
  const actual = crypto.createHash('sha256').update(fs.readFileSync(item.file)).digest('hex');
  assert.equal(actual, item.sha256, item.file);
  return { ...item };
});
const finalEvidence = {
  topicId: 'f-divergences-integral-probability-metrics',
  title: 'f-Divergences & Integral Probability Metrics',
  modulePosition: 29,
  frozenAt: new Date().toISOString(),
  status: {
    authorImplementation: 'complete',
    authorNativeModel: 'passed',
    authorActualBrowser: 'passed',
    independentReview: 'parent owns; reported channel-underflow defect repaired and regression checked',
    integratedProduction: 'parent owns subsequent snapshot',
    userAcceptance: 'not claimed',
  },
  sourceHashes,
  originalBody: {
    path: 'scratch/divergence-ipm-native-verification/original-lesson.jsx',
    sha256: crypto.createHash('sha256').update(fs.readFileSync('scratch/divergence-ipm-native-verification/original-lesson.jsx')).digest('hex'),
    originalOutputPreserved: '0.208 0.054 0.3',
  },
  native: read('scratch/divergence-ipm-native-verification/results.json'),
  fullBrowser: read('scratch/divergence-ipm-browser/results.json'),
  finalReadingAndChannelRepair: finalReading,
  formatting: read('scratch/divergence-ipm-native-verification/formatting-results.json'),
  channelRepair: {
    reportedBy: 'parent independent reviewer',
    previousModelSha256: '7117c02922fc53dad85760877e1840d395f37aaf8811377b03c2b396ebfc933f',
    counterexample: 'processDivergence([100,1e-8],[1e-8,100], [[Number.MIN_VALUE,1],[0,1]])',
    defect: 'positive product rounded to zero, creating false support loss and infinite processed KL/JS',
    repair: 'reject positive channel entries below 1e-8; true zero permitted; safe positive processed mass lower bound 1.25e-19',
    modelRegressions: 'exact counterexample and just-below-range reject; five safe boundary/identity/constant maps independently checked',
    browser: 'actual loaded export verified at 1440,390,320',
  },
  screenshotsActuallyOpened: [
    'observer-lipschitz-390.png', 'kernel-two-features-390.png', 'ratio-support-390.png',
    'section-6-1440.png', 'moving-small-disjoint-320.png', 'critic-negative-bound-320.png',
    'permutation-small-bandwidth-390.png', 'ratio-support-320.png',
    'inline-figure-1-320.png', 'inline-figure-2-320.png', 'inline-figure-3-320.png',
    'section-2-320.png', 'section-7-390.png', 'section-8-320.png',
    'final-mean-existence-320.png', 'final-bias-moment-contract-320.png', 'final-bias-moment-contract-1440.png',
  ].map(name => 'scratch/divergence-ipm-browser/' + name),
  records: {
    design: 'docs/teaching/F-DIVERGENCES-IPMS-LESSON-DESIGN.md',
    verification: 'docs/teaching/F-DIVERGENCES-IPMS-VERIFICATION.md',
    destinationNotes: [
      'docs/teaching/topic-notes/functional-analysis-rkhs.md',
      'docs/teaching/topic-notes/generative-adversarial-networks-gan-fundamentals.md',
    ],
  },
  limits: [
    'Finite educational model arithmetic, not arbitrary-range probability software.',
    'No measured deployment benchmarks, beginner study, screen-reader listening session or user approval claimed.',
    'Video target and creator description verified; recording not watched in full and large slides not fully reviewed.',
    'Independent integration and full-goal status remain separate from author freeze.',
  ],
};
fs.writeFileSync('scratch/divergence-ipm-native-verification/final-source-hashes.json', JSON.stringify({ frozenAt: finalEvidence.frozenAt, sourceHashes }, null, 2) + '\n');
fs.writeFileSync('docs/teaching/evidence/f-divergences-ipms-verification.json', JSON.stringify(finalEvidence, null, 2) + '\n');
console.log(JSON.stringify({ frozenAt: finalEvidence.frozenAt, sourceHashes }, null, 2));

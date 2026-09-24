// Freeze the bounded complementary review only against the author's final sources.
import fs from 'node:fs';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';

const hash = path => crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex');
const numericalPath = 'docs/teaching/evidence/classical-tree-knn-independent-numerical.json';
const numerical = JSON.parse(fs.readFileSync('scratch/tree-knn-independent/results.json'));
const definitions = [
  {
    stem: 'decision-tree',
    topicId: 'decision-trees-random-forests',
    initialFreeze: '2026-09-11T13:17:59.284Z',
    initialModelHash: 'c518194566df93953774c47fa76e373d078b280d73310d2f6e9cf3b8b25bea1c',
    reviewedScope: 'Complete fourteen-section body, all thirteen changed practice tasks, full pure model and labs/CSS, individual brief/design, and exact displayed BinaryTree/BinaryForest definitions. Reused the author\'s ten complete program executions and three-width behavioral evidence.',
    sourceFindings: [
      {
        issue: 'Sparse arrays were skipped by Array.some validation.',
        originalReproductions: ['impurity(Array(2)) returned 0', 'treePrediction(growTree(), Array(2)) followed a plausible leaf using undefined coordinates'],
        repair: 'Author replaced hole-skipping checks with explicit own-index validation. Complementary dense/sparse and changed tree checks close this finding.',
      },
    ],
    images: [
      'scratch/decision-tree-browser/tree-partitions-final-figure-1440.png',
      'scratch/decision-tree-browser/tree-pruning-final-figure-320.png',
      'scratch/decision-tree-browser/tree-xor-final-figure-390.png',
      'scratch/decision-tree-browser/tree-bootstrap-changed-320.png',
      'scratch/decision-tree-browser/tree-permutation-changed-320.png',
    ],
    assessment: 'Actual leaf partitions and equality paths, subtree costs, bootstrap multiplicities/OOB exclusion and relationship-breaking permutations make different mechanisms visible. Narrow plots deliberately scroll with visible instructions and numerical/text equivalents; the opened bootstrap and permutation captures show readable controls and state. No additional teaching or mathematical defect found within this bounded review.',
    primarySourcesRead: ['https://scikit-learn.org/stable/modules/ensemble.html#forest'],
  },
  {
    stem: 'knn',
    topicId: 'k-nearest-neighbors-knn',
    initialFreeze: '2026-09-11T13:59:23.262Z',
    initialModelHash: 'b9eb84f09a36ed8cdad9b392de8c1d4e074fc33353e065f646e188a7ef9bb48b',
    initialExamplesHash: 'b07eacd797f1e489ee16b9a27d2a74e16a782e08620c101c51246335aa3d0264',
    reviewedScope: 'Complete thirteen-section body, all twelve changed practice tasks, full pure model and labs/CSS, individual brief/design/verification, actual validation fixture, and exact displayed LocalNeighbors and KD build/search definitions. Reused the author\'s eleven complete program executions and three-width behavioral evidence except its focused amendments.',
    sourceFindings: [
      {
        issue: 'Sparse query/candidate arrays bypassed Array.some validation.',
        originalReproductions: ['neighborReport({query:Array(2)}) produced NaN distances and a plausible label', 'neighborReport({candidates:Array(2),k:1}) failed later on undefined.distance'],
        repair: 'Author validates explicit own indices and distinct declared candidate identities before arithmetic.',
      },
      {
        issue: 'The displayed Python Euclidean norm squared tiny differences to zero.',
        originalReproductions: ["LocalNeighbors(k=1).fit([[0.],[1e-200]],['A','B']).predict([[1e-200]]) returned A instead of B", 'The displayed KD helper chose index 0 instead of the exact-match index 1 on the same coordinates'],
        repair: 'Author uses stable hypot reduction in both actual displayed helpers; independently changed exact-match queries at ordinary, tiny and large scales preserve the correct neighbor identity.',
      },
      {
        issue: 'The browser helper multiplied two tiny nonzero norms before cosine division.',
        originalReproductions: ["metricDistance([1e-200,0],[1e-200,0],'cosine') rejected two nonzero vectors"],
        repair: 'Author normalizes each vector by its separate nonzero hypot before forming the dot product; six complementary tiny-scale angle cases pass.',
      },
    ],
    images: [
      'scratch/knn-browser/learning-state-1440-0.png',
      'scratch/knn-browser/learning-state-1440-1.png',
      'scratch/knn-browser/learning-state-390-1.png',
      'scratch/knn-browser/learning-state-390-2.png',
      'scratch/knn-browser/learning-state-320-1.png',
      'scratch/knn-browser/learning-state-320-2.png',
    ],
    assessment: 'The equal-scale neighbor contour, within-row unit contributions, visited-node/split-plane trace, candidate-ID comparison and explicit volume model serve distinct learning jobs. The validation figure uses the actual recorded losses and does not invent a U-shape. Some author element screenshots contain the fixed page header across the capture; that is retained as an evidence limitation rather than claimed unobscured reading. Narrow spatial plots explicitly scroll. No additional mathematical or teaching defect found within this bounded review.',
    primarySourcesRead: ['https://scikit-learn.org/stable/modules/neighbors.html', 'https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote02_kNN.html'],
  },
];

for (const definition of definitions) {
  const amendmentPath = `docs/teaching/evidence/${definition.stem}-input-amendment.json`;
  const amendment = JSON.parse(fs.readFileSync(amendmentPath));
  assert.equal(amendment.topicId, definition.topicId);
  for (const source of amendment.sourceHashes) assert.equal(hash(source.path), source.sha256, source.path);
  assert.deepEqual(numerical.sourceHashes[definition.stem], amendment.sourceHashes);
  const destination = `docs/teaching/evidence/${definition.stem}-independent-review.json`;
  assert(!fs.existsSync(destination), `${destination} already exists; preserve prior closure.`);
}
fs.copyFileSync('scratch/tree-knn-independent/results.json', numericalPath);
for (const definition of definitions) {
  const { images, ...review } = definition;
  const authorPath = `docs/teaching/evidence/${definition.stem}-author-review.json`;
  const author = JSON.parse(fs.readFileSync(authorPath));
  const amendmentPath = `docs/teaching/evidence/${definition.stem}-input-amendment.json`;
  const amendment = JSON.parse(fs.readFileSync(amendmentPath));
  const packet = {
    reviewedAt: new Date().toISOString(),
    reviewer: '/root/scientific_visual_improvements',
    status: 'independent review closed; no unresolved material finding in the stated scope',
    ...review,
    authorPacket: { path: authorPath, sha256: hash(authorPath), frozenAt: author.frozenAt },
    amendmentPacket: { path: amendmentPath, sha256: hash(amendmentPath), amendedAt: amendment.amendedAt },
    originalSourceArchive: amendment.originalSourceArchive,
    sourceHashes: amendment.sourceHashes,
    complementaryEvidence: { path: numericalPath, sha256: hash(numericalPath), checkedAt: numerical.checkedAt },
    scripts: ['scripts/verify-classical-tree-knn-independent.mjs', 'scripts/verify-classical-tree-knn-independent.py'].map(path => ({ path, sha256: hash(path) })),
    openedAuthorImages: images.map(path => ({ path, sha256: hash(path) })),
    limits: 'A complementary review, not a fresh full browser/author-suite run, arbitrary-float proof, learner study or user acceptance. Author-operated controls, complete outputs and baseline conservation are separately attributed to the exact author packet. The listed images were actually opened by this reviewer; no other image inspection is claimed.',
  };
  fs.writeFileSync(`docs/teaching/evidence/${definition.stem}-independent-review.json`, JSON.stringify(packet, null, 2) + '\n');
  console.log(`${definition.stem}: ${packet.reviewedAt}, ${packet.sourceHashes.length} final sources`);
}

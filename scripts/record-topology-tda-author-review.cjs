const fs = require('node:fs');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const read = file => JSON.parse(fs.readFileSync(file));
const hash = file => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const entry = path => ({ path, sha256: hash(path), bytes: fs.statSync(path).size });
const productionPaths = [
  'src/learn/data/topics/topology-topological-data-analysis-tda.jsx',
  'src/learn/data/topology-tda-models.js',
  'src/learn/data/topology-tda-examples.js',
  'src/learn/components/lesson-labs/TopologyTdaLabs.jsx',
  'src/learn/components/lesson-labs/topology-tda-labs.css',
  'src/learn/data/curriculum/blueprints/topology-topological-data-analysis-tda.js',
];
const browser = read('scratch/topology-tda-browser/results.json');
const models = read('scratch/topology-tda-review/model-oracle-results.json');
const native = read('docs/teaching/evidence/topology-tda-native-verification.json');
assert(browser.passed && models.passed && native.passed && native.originalPreservedExactly);
for (const row of browser.sources) assert.equal(hash(row.file), row.sha256);
assert.equal(browser.records.length, 3);
assert(browser.records.every(row => row.states === 458));
const viewed = [
  'chain-boundary-1440', 'square-persistence-1440', 'boundary-xor-1440', 'matching-optimum-1440',
  'pixel-ring-1440', 'landscape-image-1440', 'mapper-loop-1440',
  'inline-0-1440', 'inline-1-1440', 'inline-2-1440',
  'landscape-image-320', 'reduction-matrix-320', 'mapper-higher-nerve-390',
  'reading-3-320', 'reading-9-390', 'reading-13-390',
];
const record = {
  topicId: 'topology-topological-data-analysis-tda', modulePosition: 40,
  authorFreezeAt: new Date().toISOString(),
  status: 'author-frozen; independent review, production integration and user acceptance are separate',
  production: productionPaths.map(entry),
  support: [
    'docs/teaching/TOPOLOGY-TDA-LESSON-DESIGN.md',
    'docs/teaching/TOPOLOGY-TDA-VERIFICATION.md',
    'docs/teaching/evidence/topology-tda-original-content.json',
    'scripts/export-topology-tda-model-fixtures.mjs',
    'scripts/verify-topology-tda-models.py',
    'scripts/verify-topology-tda-native.py',
    'scripts/review-topology-tda-lesson.cjs',
    'scripts/format-topology-tda-source.cjs',
    'docs/teaching/topic-notes/category-theory-emerging-use-in-ml.md',
    'docs/teaching/topic-notes/t-sne-umap-manifold-learning.md',
    'docs/teaching/topic-notes/geometric-topological-deep-learning.md',
  ].map(entry),
  original: { sourceSha256: '0e944c5c549399e52954b5a490530c2ee3a068b0342f87095ae785a004292d71', codeAndOutputPreservedExactly: true },
  modelVerification: models, nativeVerification: native, browserVerification: browser,
  visualReview: { openedAt: new Date().toISOString(), scope: 'Author actually opened these sixteen final images through view_image; other captured images are not counted as opened.', images: viewed.map(name => entry('scratch/topology-tda-browser/' + name + '.png')) },
  formatting: read('scratch/topology-tda-review/format-conservation.json'),
  limits: ['Finite bounded mathematical and numerical checks are not general topology-inference proofs.', 'Subnormal binary64 probabilities have separately reported rounding/underflow, without a relative-accuracy guarantee.', 'Official video identities and links were checked; full playback was not reviewed.', 'GUDHI3.13.0 is a documentation version; all displayed programs use the actual Python standard library.', 'Root production integration and user acceptance are separate.'],
};
fs.writeFileSync('docs/teaching/evidence/topology-tda-author-review.json', JSON.stringify(record, null, 2) + '\n');
console.log(JSON.stringify({ authorFreezeAt: record.authorFreezeAt, sources: record.production, openedImages: viewed.length }, null, 2));

const fs = require('node:fs');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const read = file => JSON.parse(fs.readFileSync(file));
const hash = file => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const entry = path => ({ path, sha256: hash(path), bytes: fs.statSync(path).size });
const production = [
  'src/learn/data/topics/category-theory-emerging-use-in-ml.jsx',
  'src/learn/data/category-theory-models.js',
  'src/learn/data/category-theory-examples.js',
  'src/learn/components/lesson-labs/CategoryTheoryLabs.jsx',
  'src/learn/components/lesson-labs/category-theory-labs.css',
  'src/learn/data/curriculum/blueprints/category-theory-emerging-use-in-ml.js',
];
const browser = read('scratch/category-theory-browser/results.json');
const keyboard = read('scratch/category-theory-browser/keyboard-results.json');
const native = read('scratch/category-theory-verification/native-results.json');
assert(browser.passed && keyboard.passed && native.passed);
for (const source of browser.sources) assert.equal(hash(source.file), source.sha256);
assert(browser.records.length === 3 && browser.records.every(row => row.states === 272));
const viewed = [
  'default-lab-0-320', 'default-lab-1-390', 'default-lab-2-390', 'default-lab-3-320',
  'default-lab-4-320', 'default-lab-5-320', 'default-lab-6-390', 'inline-0-320',
  'inline-1-390', 'inline-2-320', 'default-lab-0-1440', 'naturality-counterexample-1440',
  'reading-1-390', 'reading-7-320', 'reading-10-320', 'reading-12-390', 'keyboard-program-320',
];
const record = {
  topicId: 'category-theory-emerging-use-in-ml', modulePosition: 41,
  authorFreezeAt: new Date().toISOString(),
  status: 'author-frozen; independent review, production integration and user acceptance are separate',
  production: production.map(entry),
  support: [
    'docs/teaching/CATEGORY-THEORY-LESSON-DESIGN.md', 'docs/teaching/CATEGORY-THEORY-VERIFICATION.md',
    'docs/teaching/evidence/category-theory-original-content.json',
    'docs/teaching/topic-notes/category-theory-emerging-use-in-ml.md',
    'scripts/build-category-theory-examples.py', 'scripts/verify-category-theory-models.mjs',
    'scripts/verify-category-theory-native.py', 'scripts/review-category-theory-lesson.cjs',
    'scripts/review-category-theory-keyboard.cjs', 'scripts/format-category-theory-source.cjs',
  ].map(entry),
  nativeVerification: native, browserVerification: browser, keyboardVerification: keyboard,
  formatting: read('scratch/category-theory-review/format-conservation.json'),
  researchFiles: read('scratch/category-theory-research/fetch-results.json'),
  visualReview: { openedAt: new Date().toISOString(), scope: 'Author actually opened these seventeen final screenshots using view_image; other captures are not counted as viewed.', images: viewed.map(name => entry('scratch/category-theory-browser/' + name + '.png')) },
  limits: ['Finite implementation checks do not establish general category-theoretic laws; the lesson supplies separate proofs.', 'Video metadata and official companion material were reviewed, not complete playback.', 'Author heuristic review does not imply a beginner study or user acceptance.', 'Independent review and integrated production loading remain separate.'],
};
fs.writeFileSync('docs/teaching/evidence/category-theory-author-review.json', JSON.stringify(record, null, 2) + '\n');
console.log(JSON.stringify({ authorFreezeAt: record.authorFreezeAt, production: record.production }, null, 2));

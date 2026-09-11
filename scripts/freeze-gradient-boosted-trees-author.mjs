import fs from 'node:fs';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';

const directory = 'scratch/gradient-boosted-trees-verification';
const destination = 'docs/teaching/evidence/gradient-boosted-trees-author-review.json';
assert(!fs.existsSync(destination), 'Preserve the existing author freeze before an amendment.');
const hash = path => crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex');
const read = path => JSON.parse(fs.readFileSync(path, 'utf8'));
const browser = read(`${directory}/browser/results.json`);
const native = read(`${directory}/native-results.json`);
const programs = read(`${directory}/program-results.json`);
const sourceHashes = browser.sourceHashes;
for (const source of sourceHashes) assert.equal(hash(source.path), source.sha256, source.path);
assert.equal(sourceHashes.length, 6);
assert.equal(native.modelSha256, sourceHashes.find(source => source.path.endsWith('-models.js')).sha256);
assert.equal(native.examplesSha256, sourceHashes.find(source => source.path.endsWith('-examples.js')).sha256);
assert.deepEqual(browser.records.map(record => record.width), [1440, 390, 320]);
for (const record of browser.records) {
  assert.equal(record.passed, true);
  assert.equal(record.states.length, 82);
  assert.equal(record.programs.length, 15);
  assert.equal(record.equations.length, 14);
  assert.equal(record.keyboard.length, 33);
  assert.deepEqual(record.errors, []);
  assert.deepEqual(record.failedRequests, []);
}
const opened = read(`${directory}/actually-opened-final-images.json`);
const captured = browser.records.flatMap(record => record.screenshots);
for (const image of opened.images) {
  assert.equal(hash(image.path), image.sha256, image.path);
  assert(captured.some(record => record.path === image.path && record.sha256 === image.sha256), image.path);
}
assert(opened.images.length >= 16);
const archivePath = 'docs/teaching/archive/gradient-boosted-trees-before-rewrite/manifest.json';
const result = {
  topicId: 'gradient-boosted-trees-xgboost-lightgbm-catboost',
  title: 'Gradient Boosted Trees (XGBoost, LightGBM, CatBoost)',
  moduleId: 'classical-ml',
  modulePosition: 4,
  status: 'author-reviewed; independent review and production integration remain separate',
  frozenAt: new Date().toISOString(),
  sourceHashes,
  design: 'docs/teaching/GRADIENT-BOOSTED-TREES-LESSON-DESIGN.md',
  verification: 'docs/teaching/GRADIENT-BOOSTED-TREES-VERIFICATION.md',
  originalArchive: { path: archivePath, sha256: hash(archivePath), manifest: read(archivePath) },
  authorEvidence: {
    native: { path: `${directory}/native-results.json`, sha256: hash(`${directory}/native-results.json`), result: native },
    programs: { path: `${directory}/program-results.json`, sha256: hash(`${directory}/program-results.json`), result: programs },
    browser: { path: `${directory}/browser/results.json`, sha256: hash(`${directory}/browser/results.json`), result: browser },
    actuallyOpenedFinalImages: opened,
    sourceFormatting: read(`${directory}/formatting.json`),
  },
  commands: [
    'scratch/lesson-tools/Scripts/python.exe scripts/build-gradient-boosted-trees-examples.py',
    'node scripts/verify-gradient-boosted-trees-models.mjs',
    'node scripts/review-gradient-boosted-trees-lesson.cjs',
  ],
  limitations: [
    'CPU demonstration fixtures, not controlled library speed/quality rankings, real-world deployment evidence or GPU/distributed execution.',
    'Finite JS teaching domains and explicit curvature arithmetic guards; not a production optimizer for arbitrary numeric input.',
    'StatQuest video identity, creator description/prerequisites and chapter list verified; transcript remained empty and selected playback attempts failed. No substantive video watch is claimed. CatBoost official video index/identity verified, corresponding paper read.',
    'Exact stdout is from the pinned workspace versions and one CPU thread; other versions/platforms can differ.',
    'Independent mathematical review, integrated production checks and user approval are not inferred from this author freeze.',
  ],
};
fs.writeFileSync(destination, JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify({ path: destination, frozenAt: result.frozenAt, sourceHashes, openedImages: opened.images.length }, null, 2));

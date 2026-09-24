import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const read = file => JSON.parse(fs.readFileSync(file, 'utf8'));
const target = 'docs/teaching/evidence/bitwise-foundations-author-review.json';
assert(!fs.existsSync(target), 'Do not overwrite the frozen author packet.');
const native = read('scratch/bitwise-author/native-results.json');
const browser = read('scratch/bitwise-author/browser/results.json');
assert(native.passed && browser.passed);
assert.deepEqual(native.sources.map(({path,sha256})=>({path,sha256})), browser.sources);
const sources = native.sources.map(source => {
  const bytes = fs.readFileSync(source.path);
  assert.equal(hash(bytes), source.sha256);
  const archive = path.join('scratch/bitwise-author-freeze', source.path).replaceAll('\\','/');
  fs.mkdirSync(path.dirname(archive), { recursive: true }); fs.writeFileSync(archive, bytes);
  return { ...source, archive };
});
const original = read('docs/teaching/evidence/bitwise-foundations-original.json');
const originals = original.sources.map(source => {
  const bytes = fs.readFileSync(source.archive); assert.equal(hash(bytes), source.sha256);
  return { path: source.path, sha256: source.sha256, bytes: bytes.length, base64: bytes.toString('base64') };
});
fs.writeFileSync('docs/teaching/evidence/bitwise-foundations-original-sources.json', JSON.stringify({ capturedAt: original.capturedAt, sourceManifest: 'docs/teaching/evidence/bitwise-foundations-original.json', sources: originals }, null, 2) + '\n');
const openedNames = ['changed-sets-320','changed-word-320','shift-origins-320','sparse-borrow-320',
  'changed-parity-320','invalid-parity-promise-320','two-partitions-320','changed-bitmap-answer-320',
  'word-default-1440','reading-9-1440','reading-10-320','bitwise-practice-320','references-320','invalid-events-320','zero-count-320'];
const imagesOpened = openedNames.map(name => {
  const item = browser.images.find(image => image.path.endsWith(`/${name}.png`)); assert(item);
  assert.equal(hash(fs.readFileSync(item.path)), item.sha256);
  return { ...item, opened: true, inspectedBy: 'root author, actual view_image calls after final browser capture' };
});
const record = { frozenAt: new Date().toISOString(), topicId: 'arrays-strings-hash-maps',
  status: 'author-verified extension; independent review and production integration pending',
  designRecord: 'docs/teaching/BITWISE-FOUNDATIONS-EXTENSION-DESIGN.md',
  verificationRecord: 'docs/teaching/BITWISE-FOUNDATIONS-EXTENSION-VERIFICATION.md',
  originalManifest: 'docs/teaching/evidence/bitwise-foundations-original.json',
  originalSourcesArchive: 'docs/teaching/evidence/bitwise-foundations-original-sources.json',
  sources, native, browser: { ...browser, images: undefined }, imagesOpened,
  preserved: { originalTeachingSubtrees: 62, originalPrograms: 6, originalProblems: 10, originalSupportFilesByteIdentical: 4 },
  added: { fullPrograms: 5, investigations: 4, computedPartitionFigure: 1, visibleChangedTasks: 4, officialProblems: 5 },
  resolvedBeforeFreeze: ['Explicit accessible names for seven controls; select labels shortened to retain full meaning at narrow widths.', 'Screenshot harness accepts the natural bottom-of-page clamp; a complete final browser pass followed.'],
  limits: 'No judge submissions, newly watched video, benchmark, screen-reader session or user study. Source inspection and actual checks are stated in the verification record. General correctness is argued in the authored proofs.' };
fs.writeFileSync(target, JSON.stringify(record, null, 2) + '\n');
console.log(`Frozen ${sources.length} source hashes and ${imagesOpened.length} actually opened screenshots at ${record.frozenAt}.`);

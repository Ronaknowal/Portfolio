import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { bridges } from './build-dsa-library-bridges.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const prefix = 'docs/teaching/implementation-depth/';
const native = JSON.parse(fs.readFileSync(path.join(root, prefix, 'dsa-remediation-native.json'), 'utf8'));
const models = JSON.parse(fs.readFileSync(path.join(root, prefix, 'dsa-remediation-models.json'), 'utf8'));
if (native.status !== 'passed' || models.status !== 'passed') throw new Error('Author checks must pass first');
const sha = file => crypto.createHash('sha256').update(fs.readFileSync(path.join(root, file))).digest('hex');
for (const record of [native, models]) for (const [file, expected] of Object.entries(record.sourceHashes)) {
  if (sha(file) !== expected) throw new Error(`Evidence no longer matches ${file}`);
}
const components = ['TreeLibraryBridge','GraphTraversalLibraryBridge','RangeQueryLibraryBridge',
  'WeightedGraphLibraryBridge','StringSearchLibraryBridge','SamplingLibraryBridge','FlowLibraryBridge',
  'GeometryLibraryBridge','PersistentMapLibraryBridge','ExternalMemoryLibraryBridge'];
const evidence = [prefix + 'dsa-remediation-native.json', prefix + 'dsa-remediation-models.json'];
function descendants(directory) {
  return fs.readdirSync(path.join(root, directory), { withFileTypes: true }).flatMap(entry =>
    entry.isDirectory() ? descendants(`${directory}/${entry.name}`) : [`${directory}/${entry.name}`]);
}
const noteIds = new Set([bridges[1][0], bridges[3][0], bridges[6][0], bridges[7][0]]);
const topics = bridges.map(([id, stem, keys], index) => {
  const body = `src/learn/data/topics/${id}.jsx`;
  const section = `src/learn/components/lesson-labs/${components[index]}.jsx`;
  const unchangedOwner = `src/learn/data/${stem}-examples.js`;
  const changedFiles = [body, section, `src/learn/data/${stem}-library-output.json`,
    ...descendants(`public/learn-assets/${id}`),
    ...(noteIds.has(id) ? [`docs/teaching/topic-notes/${id}.md`] : [])];
  return { id, body, section, status: 'author-complete-awaiting-independent-and-integration',
    unchangedMechanismOwners: [{ path: unchangedOwner, exampleKeys: keys }],
    changedFiles, evidence, sourceHashes: Object.fromEntries([...changedFiles, unchangedOwner].map(file => [file, sha(file)])) };
});
const sharedAuthorFiles = ['scripts/build-dsa-library-bridges.mjs', 'scripts/fetch-range-library.py',
  'scripts/verify-dsa-library-bridges.mjs', 'scripts/verify-dsa-remediation-models.py',
  'scripts/fixtures/range-library-oracle.cpp', 'scripts/write-dsa-remediation-manifest.mjs',
  prefix + 'DSA-REMEDIATION.md', ...evidence];
const sharedDependencies = ['src/learn/components/lesson-labs/MechanismProgram.jsx', 'src/learn/components/lesson-labs/mechanism-program.css'];
fs.writeFileSync(path.join(root, prefix, 'dsa-remediation.json'), JSON.stringify({
  date: '2026-09-22', status: 'author-complete-awaiting-independent-and-integration',
  report: prefix + 'DSA-REMEDIATION.md', runtime: native.runtime,
  topics, sharedAuthorFiles, sharedAuthorHashes: Object.fromEntries(sharedAuthorFiles.map(file => [file, sha(file)])),
  sharedDependencies: Object.fromEntries(sharedDependencies.map(file => [file, sha(file)])),
  integration: { build: 'not run; coordinating task owns', browser: 'not run; coordinating task owns',
    independentReview: 'pending; author evidence is not independent review', ledger: 'unchanged by author' },
}, null, 2) + '\n');
console.log(`Bound ${topics.length} topics and ${topics.reduce((sum, topic) => sum + topic.changedFiles.length, 0)} scoped changed/new files`);

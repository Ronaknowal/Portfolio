// Immutable build evidence for explicitly selected author-ready lessons.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';

const directory = 'docs/teaching/deep-learning-completion';
const hash = file => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const read = file => JSON.parse(fs.readFileSync(file, 'utf8'));
const baseline = read(`${directory}/baseline.json`);
const topicIds = process.argv.slice(2);
assert.ok(topicIds.length, 'Pass explicit author-ready topic IDs.');
assert.equal(new Set(topicIds).size, topicIds.length);
const topicSources = {}, sourceHashes = {};
for (const id of topicIds) {
  assert.ok(baseline.topicIds.includes(id), `Outside scoped increment: ${id}`);
  const author = read(`${directory}/${id}/implementation-checks.json`);
  assert.equal(author.passed, true);
  topicSources[id] = author.sourceFiles.filter(file => file.startsWith('src/') || file.startsWith('public/'));
  assert.ok(topicSources[id].length);
  for (const file of topicSources[id]) {
    const actual = hash(file);
    assert.equal(actual, author.sourceHashes[file], `Stale author-ready source: ${file}`);
    sourceHashes[file] = actual;
  }
}
const startedAt = new Date().toISOString();
fs.mkdirSync(`${directory}/builds`, { recursive: true });
const logPath = 'scratch/deep-learning-review-build.log';
const log = fs.openSync(logPath, 'w');
const result = spawnSync(process.execPath, ['node_modules/vite/bin/vite.js', 'build'], { stdio: ['ignore', log, log] });
fs.closeSync(log);
const changedDuringBuild = Object.keys(sourceHashes).filter(file => hash(file) !== sourceHashes[file]);
const coveredTopicIds = topicIds.filter(id => topicSources[id].every(file => !changedDuringBuild.includes(file)));
const receipt = {
  startedAt, completedAt: new Date().toISOString(), command: 'node node_modules/vite/bin/vite.js build',
  passed: result.status === 0, topicIds, coveredTopicIds, topicSources, sourceHashes, changedDuringBuild,
  buildManifestHash: fs.existsSync('dist/.vite/manifest.json') ? hash('dist/.vite/manifest.json') : null,
  logPath, limitations: 'A build establishes bundling and source identity only. It does not certify numerical, independent or rendered learning quality. Existing shared-navigation chunk warning is not hidden.',
};
const receiptPath = `${directory}/builds/${startedAt.replace(/[:.]/g, '-')}.json`;
assert.ok(!fs.existsSync(receiptPath));
fs.writeFileSync(receiptPath, JSON.stringify(receipt, null, 2) + '\n');
fs.writeFileSync(`${directory}/current-build.json`, JSON.stringify({ receiptPath }, null, 2) + '\n');
console.log(JSON.stringify({ receiptPath, passed: receipt.passed, coveredTopicIds, changedDuringBuild }));
if (result.status !== 0) console.log(fs.readFileSync(logPath, 'utf8').slice(-6000));
process.exitCode = result.status ?? 1;

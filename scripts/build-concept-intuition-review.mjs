// Build a review snapshot and bind it to the actual author-ready runtime sources.
import fs from 'node:fs';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';

const directory = 'docs/teaching/concept-intuition';
const baseline = JSON.parse(fs.readFileSync('docs/teaching/evidence/concept-intuition-baseline.json'));
const hash = file => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const sourceFiles = new Set();
for (const id of baseline.topicIds) {
  const receipt = `${directory}/${id}/author-checks.json`;
  if (!fs.existsSync(receipt)) continue;
  const author = JSON.parse(fs.readFileSync(receipt));
  for (const file of author.sourceFiles) if (file.startsWith('src/')) sourceFiles.add(file);
}
const sourceHashes = Object.fromEntries([...sourceFiles].map(file => [file, hash(file)]));
const startedAt = new Date().toISOString();
const log = fs.openSync('scratch/concept-intuition-build.log', 'w');
const build = spawnSync(process.execPath, ['node_modules/vite/bin/vite.js', 'build'], { stdio: ['ignore', log, log] });
fs.closeSync(log);
const changedDuringBuild = Object.keys(sourceHashes).filter(file => hash(file) !== sourceHashes[file]);
const receipt = {
  startedAt, completedAt: new Date().toISOString(), command: 'node node_modules/vite/bin/vite.js build',
  passed: build.status === 0, sourceHashes, changedDuringBuild,
  unchangedThroughBuild: Object.fromEntries(Object.entries(sourceHashes).filter(([file]) => !changedDuringBuild.includes(file))),
  limitations: 'Existing large-chunk warning is retained in the build log; this is build evidence, not teaching or browser review.',
};
const destination = `${directory}/build-source-checkpoint.json`;
if (fs.existsSync(destination)) {
  const previous = JSON.parse(fs.readFileSync(destination));
  const archive = `${directory}/build-${previous.startedAt.replace(/[:.]/g, '-')}.json`;
  if (!fs.existsSync(archive)) fs.copyFileSync(destination, archive);
}
fs.writeFileSync(destination, JSON.stringify(receipt, null, 2) + '\n');
console.log(JSON.stringify({ passed: receipt.passed, files: sourceFiles.size, changedDuringBuild, completedAt: receipt.completedAt }));
process.exitCode = build.status ?? 1;

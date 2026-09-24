const fs = require('node:fs');
const { spawnSync } = require('node:child_process');

const directory = 'scratch/foundations-completion-integration';
fs.mkdirSync(directory, { recursive: true });
for (const [name, script, ...args] of [
  ['curriculum', 'scripts/verify-curriculum.mjs'],
  ['artifacts', 'scripts/verify-learning-artifacts.mjs'],
  ['source-boundary', 'scripts/verify-content-import-boundary.mjs'],
  ['runtime-organization', 'scripts/verify-runtime-source-organization.mjs'],
  ['progress-before', 'scripts/verify-dsa-math-foundations-progress.mjs'],
  ['inventory-before', 'scripts/build-curriculum-inventory.mjs'],
  ['build', 'node_modules/vite/bin/vite.js', 'build'],
]) {
  const result = spawnSync(process.execPath, [script, ...args], {
    encoding: 'utf8', maxBuffer: 24 * 1024 * 1024,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  fs.writeFileSync(`${directory}/${name}.log`, result.stdout + result.stderr);
  if (result.status !== 0) throw new Error(`${name} failed; inspect its saved log.`);
  console.log(`${name}: passed`);
}

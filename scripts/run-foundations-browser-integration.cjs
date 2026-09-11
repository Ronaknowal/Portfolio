const fs = require('node:fs');
const { spawnSync } = require('node:child_process');

const directory = 'scratch/foundations-completion-integration';
fs.mkdirSync(directory, { recursive: true });
const env = {
  ...process.env,
  PLAYWRIGHT_PACKAGE: process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright',
  PERFORMANCE_BASE_URL: 'http://127.0.0.1:4173',
  LEARNING_BASE_URL: 'http://127.0.0.1:4173',
};
for (const [name, script] of [
  ['loading', 'scripts/verify-learning-load-boundaries.cjs'],
  ['routes', 'scripts/review-module-order.cjs'],
]) {
  const result = spawnSync(process.execPath, [script], {
    env, encoding: 'utf8', maxBuffer: 24 * 1024 * 1024,
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  fs.writeFileSync(`${directory}/${name}.log`, result.stdout + result.stderr);
  if (result.status !== 0) throw new Error(`${name} failed; inspect its saved log.`);
  console.log(`${name}: passed`);
}

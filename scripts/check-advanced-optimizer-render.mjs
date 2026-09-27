import fs from 'node:fs';
import assert from 'node:assert/strict';
import { build } from 'esbuild';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
const id = 'advanced-optimizers-lion-sophia-prodigy-schedule-free';
const file = 'scratch/optimizer-render-check.mjs';
const errors = [], original = console.error;
try {
  await build({ entryPoints: [`src/learn/data/topics/${id}.jsx`], bundle: true, platform: 'node', format: 'esm', jsx: 'automatic', external: ['react', 'react-dom', 'react/jsx-runtime'], loader: { '.css': 'empty' }, outfile: file });
  console.error = (...args) => errors.push(args.map(String).join(' '));
  const module = await import(`../${file}`);
  const html = renderToStaticMarkup(createElement(module.default.content));
  assert.deepEqual(errors, []);
  assert.ok(!html.includes('NaN') && !html.includes('Infinity'));
  for (const section of ['optimizer-lion', 'optimizer-sophia', 'optimizer-prodigy', 'optimizer-schedule-free', 'optimizer-memory', 'optimizer-digit']) assert.ok(html.includes(`data-lab="${section}"`), section);
  const result = { passed: true, htmlCharacters: html.length, checks: ['Complete initial lesson renders without exceptions or React warnings', 'All six semantic investigations present with immediate arithmetic or honest asset loading state', 'Finite initial numeric output'], limitations: ['Server render checks do not establish painted geometry, browser interaction or network recovery. Those remain the integration owner’s checks.'] };
  fs.writeFileSync(`docs/teaching/deep-learning-completion/${id}/render-checks.json`, JSON.stringify(result, null, 2) + '\n');
  original(JSON.stringify(result));
} finally {
  console.error = original;
  fs.rmSync(file, { force: true });
}

import fs from 'node:fs';
import assert from 'node:assert/strict';
import { createServer } from 'vite';
import react from '@vitejs/plugin-react';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { rotaryConventionTrace, xposPairTrace } from '../src/learn/data/position-convention-models.js';
import { alibiCompetition, cachePositionRead, positionCacheDefault } from '../src/learn/data/positional-encoding-models.js';

const checks = [];
function close(name, actual, expected, tolerance = 1e-10) {
  const a = [actual].flat(Infinity), b = [expected].flat(Infinity);
  assert.equal(a.length, b.length);
  const error = Math.max(...a.map((v, i) => Math.abs(v - b[i])));
  assert.ok(Number.isFinite(error) && error < tolerance, name + ': ' + error);
  checks.push({ name, passed: true, maxAbsoluteError: error, tolerance });
}
// Explicit four-by-four matrices are a separate reference for the paired and
// half-split routes. The partial head sets nonrotary block angles to zero.
function matrixRotation(v, id, base, pairs) {
  const a = pairs > 0 ? id : 0, b = pairs > 1 ? id / Math.sqrt(base) : 0;
  const R = [[Math.cos(a), -Math.sin(a), 0, 0], [Math.sin(a), Math.cos(a), 0, 0],
    [0, 0, Math.cos(b), -Math.sin(b)], [0, 0, Math.sin(b), Math.cos(b)]];
  return R.map(row => row.reduce((s, value, i) => s + value * v[i], 0));
}
for (const base of [10, 10000, 1e6]) for (const [m, n] of [[0, 0], [3, 7], [-128, 256], [512, 129]]) for (const pairs of [0, 1, 2]) {
  const q = [.8, -.5, .3, 1.2], k = [1, .25, -.5, .75];
  const result = rotaryConventionTrace(q, k, m, n, base, pairs);
  const rq = matrixRotation(q, m, base, pairs), rk = matrixRotation(k, n, base, pairs);
  close('Half-split restored ' + [base, m, n, pairs], result.restored, matrixRotation(q, m, base, 2));
  close('Partial contribution ' + [base, m, n, pairs], result.contributions, [rq[0] * rk[0] + rq[1] * rk[1], rq[2] * rk[2] + rq[3] * rk[3]]);
}
for (const m of [0, 1, 128, 512, 1024]) for (const n of [0, 128, 512, 1024]) {
  const t = xposPairTrace(m, n);
  close('XPos relative product ' + [m, n], t.product, Math.exp((m - n) / 512 * Math.log(2 / 7)));
  close('XPos dot relative angle ' + [m, n], t.scaledQuery[0] * t.scaledKey[0] + t.scaledQuery[1] * t.scaledKey[1], t.relativeAmplitude * Math.cos(m - n));
  close('XPos changed norms ' + [m, n], [Math.hypot(...t.scaledQuery), Math.hypot(...t.scaledKey)], [t.queryAmplitude, t.keyAmplitude]);
}
for (const length of [3, 4, 8]) for (const mode of ['rope', 'alibi']) {
  const s = { ...positionCacheDefault(), mode };
  while (s.ids.length < length) { s.ids.push(s.queryId); s.keys.push([1, 0, 0, 1]); s.values.push([0, 1]); }
  const result = cachePositionRead(s);
  close('Expanded cache normalized ' + [mode, length], result.weights.reduce((a, b) => a + b), 1);
  close('Expanded cache record reversal ' + [mode, length], result.output, cachePositionRead({ ...s, ids: [...s.ids].reverse(), keys: [...s.keys].reverse(), values: [...s.values].reverse() }).output);
  if (mode === 'alibi') close('ALiBi final components ' + length, result.scores, result.content.map((v, i) => v - s.slope * (s.queryId - s.ids[i])));
}
const competition = alibiCompetition([1, 1, -.5], [1, 4, 3], 4, .5);
close('Selected equal-content pair odds', competition.weights[1] / competition.weights[0], Math.exp(1.5));
const server = await createServer({ configFile: false, plugins: [react()], server: { middlewareMode: true, watch: null }, optimizeDeps: { noDiscovery: true }, appType: 'custom', logLevel: 'error' });
try {
  const topic = await server.ssrLoadModule('/src/learn/data/topics/positional-encodings-sinusoidal-learned-rope-alibi.jsx');
  const html = renderToStaticMarkup(React.createElement(topic.default.content));
  for (const phrase of ['Pin current clock as reference', 'Half-split output', 'Equalize selected', 'Add cache record', 'Distance bias', 'XPos query ID', 'cached positions and key coordinates']) {
    assert.ok(html.includes(phrase), 'SSR missing ' + phrase);
    checks.push({ name: 'Full lesson SSR: ' + phrase, passed: true });
  }
  assert.ok(!html.includes('NaN'));
} finally { await server.close(); }
fs.writeFileSync('docs/teaching/deep-learning-completion/positional-encodings-sinusoidal-learned-rope-alibi/convention-checks.json', JSON.stringify({ passed: true, checks, limits: ['SSR verifies render and default calculations; painted geometry and control interactions remain the integration browser check.'] }, null, 2) + '\n');
console.log(checks.length + ' convention, amplitude, variable-cache and full-render checks passed.');

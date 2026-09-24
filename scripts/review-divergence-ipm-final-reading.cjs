const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const files = [
  'src/learn/data/topics/f-divergences-integral-probability-metrics.jsx',
  'src/learn/data/divergence-ipm-models.js',
  'src/learn/data/divergence-ipm-examples.js',
  'src/learn/components/lesson-labs/DivergenceIpmLabs.jsx',
  'src/learn/components/lesson-labs/divergence-ipm-labs.css',
  'src/learn/data/curriculum/blueprints/f-divergences-integral-probability-metrics.js',
];
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 } });
      await page.routeWebSocket('**', socket => socket.close());
      const errors = [], failedRequests = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('requestfailed', request => failedRequests.push(request.url()));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/f-divergences-integral-probability-metrics?module=math-foundations', { waitUntil: 'networkidle' });
      await page.locator('.divergence-ipm-lesson').waitFor();
      await page.evaluate(() => document.fonts.ready);
      const mean = page.locator('.divergence-ipm-lesson p').filter({ hasText: 'For the measurable finite-feature and Gaussian kernels used here' });
      const bias = page.locator('.divergence-ipm-lesson p').filter({ hasText: 'the bias is the sum of feature variances' });
      for (const [name, target] of [['mean-existence', mean], ['bias-moment-contract', bias]]) {
        assert.equal(await target.count(), 1);
        await target.evaluate(element => window.scrollTo(0, window.scrollY + element.getBoundingClientRect().top - 175));
        await page.screenshot({ path: 'scratch/divergence-ipm-browser/final-' + name + '-' + width + '.png' });
        assert.ok(await target.evaluate(element => element.scrollWidth <= element.clientWidth + 1));
      }
      const geometry = await page.evaluate(() => ({
        width: document.documentElement.scrollWidth,
        fonts: [...document.fonts].some(font => font.family === 'Space Grotesk' && font.status === 'loaded'),
      }));
      assert.equal(geometry.width, width);
      assert.ok(geometry.fonts);
      assert.deepEqual(errors, []); assert.deepEqual(failedRequests, []);
      const channelBoundary = await page.evaluate(async () => {
        const { processDivergence } = await import('/src/learn/data/divergence-ipm-models.js');
        let rejected = false;
        try {
          processDivergence([100, 1e-8], [1e-8, 100], [[Number.MIN_VALUE, 1], [0, 1]]);
        } catch (error) { rejected = error instanceof RangeError; }
        const minimum = processDivergence([100, 1e-8], [1e-8, 100], [[1e-8, 1 - 1e-8], [0, 1]]);
        const zeros = processDivergence([100, 1e-8], [1e-8, 100], [[0, 1], [0, 1]]);
        return { rejected, finiteMinimum: Object.values(minimum.after).every(Number.isFinite), js: minimum.after.js, kl: minimum.after.kl, beforeKl: minimum.before.kl, exactZeroOutput: zeros.outputP[0] === 0 && zeros.outputQ[0] === 0, mergedKl: zeros.after.kl };
      });
      assert.ok(channelBoundary.rejected && channelBoundary.finiteMinimum && channelBoundary.exactZeroOutput);
      assert.ok(channelBoundary.js <= Math.LN2 && channelBoundary.kl <= channelBoundary.beforeKl);
      assert.equal(channelBoundary.mergedKl, 0);
      results.push({ width, geometry, errors, failedRequests, meanExistenceAndBiasContractRead: true, channelBoundary });
      await page.close();
    }
  } finally { await browser.close(); }
  const hashes = files.map(file => ({ file, sha256: crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex') }));
  const result = { at: new Date().toISOString(), results, hashes };
  fs.writeFileSync('scratch/divergence-ipm-browser/final-reading-results.json', JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result, null, 2));
})().catch(error => { console.error(error); process.exitCode = 1; });

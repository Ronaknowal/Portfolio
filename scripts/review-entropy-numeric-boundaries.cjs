const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('fs');
const path = require('path');
const assert = require('assert/strict');

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1050 } });
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => {
        if (['error', 'warning'].includes(message.type())) errors.push(message.text());
      });
      page.on('requestfailed', request => errors.push(request.url() + ': ' + request.failure().errorText));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/entropy-cross-entropy-kl-divergence');
      const mismatch = page.locator('[data-entropy-lab="mismatch"]');
      await mismatch.waitFor();
      await page.evaluate(() => document.fonts.ready);
      const source = mismatch.getByRole('textbox', { name: 'Source weights P', exact: true });
      const prediction = mismatch.getByRole('textbox', { name: 'Model weights Q', exact: true });
      const apply = mismatch.getByRole('button', { name: 'Apply distributions', exact: true });
      await source.fill('1,1,0,0');
      await prediction.fill('100,1e-310,0,0');
      await apply.click();
      assert.equal(await mismatch.getByRole('alert').count(), 0);
      const previous = await mismatch.locator('.entropy-metrics').innerText();
      assert(!previous.includes('∞'));
      for (const draft of ['100,1e-322,0,0', '100,1e-9999,0,0']) {
        await prediction.fill(draft);
        await apply.click();
        assert((await mismatch.getByRole('alert').innerText()).includes('too small'));
        assert.equal(await mismatch.locator('.entropy-metrics').innerText(), previous);
      }
      await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = 'hidden'));
      await mismatch.screenshot({ path: path.resolve('scratch/entropy-browser/numeric-error-' + width + '.png') });
      await prediction.fill('100,0,0,0');
      await apply.click();
      assert.equal(await mismatch.getByRole('alert').count(), 0);
      assert((await mismatch.locator('.entropy-metrics').innerText()).includes('∞'));
      await mismatch.getByRole('button', { name: 'Reset mismatch', exact: true }).click();
      const maximum = page.locator('[data-entropy-lab="maximum"]');
      assert((await maximum.innerText()).includes('0.000001 through 1.999999'));
      for (const mean of ['0', '0.5', '1', String(10 / 7), '1.8', '2']) {
        await maximum.getByRole('combobox', { name: 'Required mean m', exact: true }).selectOption(mean);
        for (const fraction of ['0', '0.5', '1']) {
          await maximum.getByRole('slider', { name: 'Position along feasible slice', exact: true }).fill(fraction);
          const gap = maximum.locator('.entropy-metrics>div').filter({ has: page.locator('dt', { hasText: 'KL(q ∥ p*), bits' }) }).locator('dd');
          assert(Number.isFinite(Number(await gap.innerText())));
        }
      }
      await maximum.getByRole('button', { name: 'Reset mean constraint', exact: true }).click();
      await maximum.screenshot({ path: path.resolve('scratch/entropy-browser/maximum-range-final-' + width + '.png') });
      await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = ''));
      const moduleChecks = await page.evaluate(async () => {
        const model = await import('/src/learn/data/entropy-information-models.js');
        const rejected = [1e-200, 1e-100, 1e-12, 2 - 1e-12].map(mean => {
          try { model.maxEntropyState(mean, 1); return false; } catch (error) { return error instanceof RangeError; }
        });
        const accepted = [0, 1e-6, 2 - 1e-6, 2].map(mean => Number.isFinite(model.maxEntropyState(mean, 1).gap));
        return { rejected, accepted };
      });
      assert(moduleChecks.rejected.every(Boolean));
      assert(moduleChecks.accepted.every(Boolean));
      const overflow = await page.evaluate(() => ({ viewport: document.documentElement.clientWidth, width: document.documentElement.scrollWidth }));
      assert(overflow.width <= overflow.viewport + 1);
      assert.deepEqual(await maximum.locator('p').evaluateAll(nodes => nodes.filter(node => node.scrollWidth > node.clientWidth + 1).map(node => node.textContent)), []);
      assert.deepEqual(errors, []);
      results.push({ width, rejectedDrafts: 2, tinyFiniteDraft: true, intentionalZeroInfinity: true, maximumControlStates: 18, moduleChecks, overflow, errors });
      await page.close();
    }
  } finally {
    await browser.close();
  }
  fs.writeFileSync('scratch/entropy-browser/numeric-boundary-results.json', JSON.stringify({ passed: true, timestamp: new Date().toISOString(), results }, null, 2));
  console.log(results);
})().catch(error => { console.error(error); process.exit(1); });

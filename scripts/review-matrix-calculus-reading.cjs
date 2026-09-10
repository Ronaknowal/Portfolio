const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const directory = path.resolve(__dirname, '../scratch/matrix-calculus-browser');
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  for (const width of [1440, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/matrix-calculus-jacobians');
    await page.locator('[data-lab="local-jacobian"]').waitFor();
    assert.equal(await page.getByText(/Consider the two-input function/).count(), 1);
    const chain = page.locator('[data-lab="chain-rule"]');
    await chain.locator('select').first().selectOption('reverse');
    assert.equal(await chain.locator('select').first().locator('option:checked').innerText(), 'Reverse loss gradient');
    await chain.getByRole('button', { name: 'Show complete trace', exact: true }).click();
    await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = 'hidden'));
    await chain.screenshot({ path: path.join(directory, 'reverse-final-' + width + '.png') });
    await page.locator('.lesson-sources').screenshot({ path: path.join(directory, 'sources-' + width + '.png') });
    await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = ''));
    const difference = page.locator('[data-lab="finite-difference"]');
    const steps = [1, 0.1, 0.01, 0.0001, 0.000001, 0.00000001, 1e-10, 1e-12, 1e-14, 1e-16];
    for (let index = 0; index < steps.length; index++) {
      await difference.locator('select').nth(2).selectOption({ value: String(index) });
      assert.equal(await difference.locator('select').nth(2).inputValue(), String(index));
      const expected = 52 + (Math.log10(steps[index]) + 16) / 16 * 262;
      await page.waitForFunction(value => Math.abs(Number(document.querySelector('[data-lab="finite-difference"] .calculus-marker').getAttribute('x1')) - value) < 1e-9, expected);
      const observed = Number(await difference.locator('.calculus-marker').getAttribute('x1'));
      assert.ok(Math.abs(observed - expected) < 1e-5, JSON.stringify({ index, observed, expected }));
    }
    await page.locator('.matrix-calculus-lesson').evaluate(node => node.querySelectorAll('details').forEach(details => details.open = true));
    assert.equal(await page.locator('.katex-error').count(), 0);
    assert.equal(await page.locator('.katex-display').evaluateAll(nodes => nodes.some(node => node.scrollWidth > node.clientWidth + 2)), false);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 2), false);
    assert.equal(await page.locator('.lesson-sources a').count(), 9);
    assert.deepEqual(errors, []);
    results.push({ width, markerSteps: 10, sources: 9, mathOverflow: false, pageOverflow: false, errors });
    await page.close();
  }
  fs.writeFileSync(path.join(directory, 'reading-results.json'), JSON.stringify({ date: new Date().toISOString(), results }, null, 2));
  await browser.close();
})().catch(error => { console.error(error); process.exit(1); });

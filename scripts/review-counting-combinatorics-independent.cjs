const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const directory = path.resolve('scratch/counting-combinatorics-independent-review');

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      const errors = [], failedRequests = [];
      await page.routeWebSocket('**', socket => socket.close());
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => {
        if (message.type() === 'error' && !message.text().includes('[vite]')) errors.push(message.text());
      });
      page.on('requestfailed', request => failedRequests.push(request.url()));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/counting-combinatorics-mathematical-induction?module=math-foundations', { waitUntil: 'domcontentloaded' });
      const lesson = page.locator('.counting-lesson');
      await lesson.waitFor();
      await page.evaluate(() => document.fonts.ready);
      const fonts = await page.evaluate(() => [...document.fonts].filter(font => font.status === 'loaded').map(font => font.family));
      assert(fonts.length > 0);
      const fibers = lesson.getByRole('region', { name: 'Outcome identity and fibers investigation', exact: true });
      await fibers.getByLabel('Available labels', { exact: true }).selectOption('4');
      await fibers.getByLabel('Positions', { exact: true }).selectOption('3');
      await fibers.getByLabel('Repeated labels', { exact: true }).selectOption('true');
      assert.match(await fibers.getByRole('status').first().innerText(), /64 ordered descriptions\s+20 unordered outcomes/);
      assert.match(await fibers.getByRole('status').last().innerText(), /1, 3, 6/);
      const coefficients = lesson.getByRole('region', { name: 'Generating coefficient construction investigation', exact: true });
      for (const [station, capacity] of [['A', 4], ['B', 0], ['C', 4]]) {
        await coefficients.getByLabel(`Station ${station} capacity`, { exact: true }).selectOption(String(capacity));
      }
      await coefficients.getByLabel('Target degree', { exact: true }).selectOption('4');
      assert.match(await coefficients.getByRole('status').innerText(), /Coefficient of x\^4: 5\./);
      const remove = coefficients.getByRole('button', { name: 'Remove last factor', exact: true });
      await remove.focus();
      assert(await remove.evaluate(node => document.activeElement === node && getComputedStyle(node).outlineStyle !== 'none'));
      await page.keyboard.press('Enter');
      assert.match(await coefficients.getByRole('status').innerText(), /Coefficient of x\^4: 1\./);
      const restore = coefficients.getByRole('button', { name: 'Include next factor', exact: true });
      await restore.focus();
      await page.keyboard.press('Enter');
      assert.match(await coefficients.getByRole('status').innerText(), /Coefficient of x\^4: 5\./);
      const region = coefficients.locator('.counting-coefficients');
      await region.evaluate(node => window.scrollTo({ top: scrollY + node.getBoundingClientRect().top - 85, behavior: 'instant' }));
      await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
      const imagePath = `${directory}/changed-coefficient-independent-${width}.png`;
      await page.screenshot({ path: imagePath });
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      assert.deepEqual(errors, []);
      assert.deepEqual(failedRequests, []);
      results.push({ width, fonts, changedFiberCounts: [64, 20, [1, 3, 6]], changedCoefficient: [4, 0, 4], target: 4, countsBeforeRemoveRestored: [5, 1, 5], keyboardActions: 2, errors, failedRequests, imagePath });
      await page.close();
    }
  } finally {
    await browser.close();
  }
  const record = { checkedAt: new Date().toISOString(), status: 'passed', scope: 'Reviewer changed states and real keyboard subset; does not replace or relabel author comprehensive browser evidence.', results };
  fs.writeFileSync(`${directory}/browser-results.json`, JSON.stringify(record, null, 2) + '\n');
  console.log(JSON.stringify(record, null, 2));
})().catch(error => { console.error(error); process.exitCode = 1; });

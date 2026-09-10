const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const directory = 'scratch/bayesian-inference-review/reading';
fs.mkdirSync(directory, { recursive: true });
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/bayesian-inference-conjugate-priors?module=math-foundations', { waitUntil: 'networkidle' });
      const lesson = page.locator('.bayesian-inference-lesson');
      await lesson.waitFor();
      await page.evaluate(() => document.fonts.ready);
      let captures = 0;
      const capture = async (locator, name) => {
        await locator.evaluate(node => node.scrollIntoView({ block: 'center', behavior: 'instant' }));
        await page.screenshot({ path: `${directory}/${name}-${width}.png` });
        captures += 1;
      };
      const checkpoints = lesson.locator('.lesson-check').filter({ hasNot: page.locator('h3') });
      assert.equal(await checkpoints.count(), 2);
      for (const checkpoint of await checkpoints.all()) {
        assert.ok((await checkpoint.locator(':scope > p').innerText()).length > 50);
        const summary = checkpoint.locator('summary');
        await summary.focus();
        await page.keyboard.press('Enter');
        assert.ok((await checkpoint.locator('details p').innerText()).length > 100);
      }
      for (const [index, figure] of (await lesson.locator('.bayesian-plot').all()).entries()) {
        await capture(figure, `plot-${index}`);
      }
      for (const [index, figure] of (await lesson.locator('.bayesian-inline').all()).entries()) {
        await capture(figure, `inline-${index}`);
      }
      await lesson.getByText('Deeper: when both mean and noise variance are unknown', { exact: true }).click();
      for (const index of [4, 5, 8, 9, 13, 14, 15]) {
        await capture(lesson.locator('.katex-display').nth(index), `equation-${index}`);
      }
      await capture(lesson.locator('.python-example').last().locator('h3'), 'native-code');
      await capture(lesson.locator('.python-example').last().locator(':scope > div').last(), 'native-output');
      await capture(lesson.locator('.lesson-sources'), 'sources');
      await lesson.getByRole('slider', { name: 'Successes', exact: true }).focus();
      const keyboard = [];
      for (let index = 0; index < 20; index += 1) {
        await page.keyboard.press('Tab');
        keyboard.push(await page.evaluate(() => ({ tag: document.activeElement.tagName, outline: getComputedStyle(document.activeElement).outlineStyle, name: document.activeElement.getAttribute('aria-label') || document.activeElement.textContent.slice(0, 80) })));
      }
      const mathOverflow = await lesson.locator('.katex-display').evaluateAll(nodes => nodes.filter(node => node.scrollWidth > node.clientWidth + 1).map(node => node.textContent.slice(0, 100)));
      const dimensions = await page.evaluate(() => ({ viewport: innerWidth, page: document.documentElement.scrollWidth }));
      assert.deepEqual(errors, []);
      assert.deepEqual(mathOverflow, []);
      assert.ok(dimensions.page <= dimensions.viewport + 1);
      results.push({ width, at: new Date().toISOString(), visibleAnsweredCheckpoints: 2, captures, keyboard, mathOverflow, dimensions, errors });
      await page.close();
    }
    fs.writeFileSync(`${directory}/results.json`, JSON.stringify({ at: new Date().toISOString(), results }, null, 2));
    console.log(JSON.stringify(results.map(result => ({ width: result.width, at: result.at, checkpoints: result.visibleAnsweredCheckpoints, mathOverflow: result.mathOverflow }))));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

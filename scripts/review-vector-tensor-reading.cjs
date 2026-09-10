const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  const directory = path.resolve(__dirname, '../scratch/vector-tensor-browser');
  for (const width of [1440, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/vectors-matrices-tensor-operations');
    await page.locator('[data-lab="vector-projection"]').waitFor();
    assert.equal(await page.locator('.katex-error').count(), 0);
    assert.equal(await page.locator('.lesson-sources a').count(), 14);
    assert.equal(await page.locator('a[href="https://www.youtube.com/watch?v=kYB8IZa5AuE"]').count(), 1);
    assert.equal(await page.locator('.vectors-lesson').getByText('orthonormal', { exact: true }).count(), 1);
    for (const [name, heading] of [['dot-reading', /^3\./], ['map-reading', /^4\./], ['practice-reading', /^10\./]]) {
      const target = page.getByRole('heading', { name: heading });
      await target.evaluate(node => window.scrollTo(0, window.scrollY + node.getBoundingClientRect().top - 100));
      await page.screenshot({ path: path.join(directory, name + '-' + width + '.png') });
    }
    const sources = page.locator('.lesson-sources');
    await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = 'hidden'));
    await sources.screenshot({ path: path.join(directory, 'sources-' + width + '.png') });
    await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = ''));
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 2), false);
    assert.deepEqual(errors, []);
    results.push({ width, sourceLinks: 14, mathErrors: 0, pageErrors: 0, horizontalPageOverflow: false });
    await page.close();
  }
  fs.writeFileSync(path.join(directory, 'reading-results.json'), JSON.stringify({ date: new Date().toISOString(), results }, null, 2));
  await browser.close();
})().catch(error => { console.error(error); process.exit(1); });

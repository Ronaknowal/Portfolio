const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const directory = path.resolve(__dirname, '../scratch/eigenvalue-lesson-review');
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  for (const width of [1440, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.routeWebSocket('**', socket => socket.close());
    await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/eigenvalues-eigenvectors?module=math-foundations');
    const lesson = page.locator('.eigenvalue-lesson');
    await lesson.waitFor();
    assert.equal(await lesson.locator('.lesson-sources a').count(), 10);
    assert.equal(await lesson.locator('.lesson-sources > ul > li').count(), 6);
    assert.ok((await lesson.textContent()).includes('[1,1]P=[1,1]'));
    assert.ok((await lesson.textContent()).includes('Its real counterpart may need 2×2 blocks'));
    const equations = await lesson.locator('.katex-display').evaluateAll(nodes => nodes.filter(node => node.getBoundingClientRect().height > 0 && node.scrollWidth > node.clientWidth + 2).map(node => node.textContent));
    assert.deepEqual(equations, [], 'Wide displayed equations');
    for (const section of [2, 3, 4, 5, 7, 8]) {
      const heading = lesson.getByRole('heading', { name: new RegExp(`^${section}\\.`) });
      await heading.evaluate(node => window.scrollTo(0, window.scrollY + node.getBoundingClientRect().top - 100));
      await page.screenshot({ path: path.join(directory, `reading-${section}-${width}.png`) });
    }
    const recurrence = lesson.getByRole('region', { name: 'Repeated matrix update investigation' });
    await recurrence.getByLabel('Update rule').selectOption('shear');
    await recurrence.getByLabel('Starting vector').selectOption('mixed');
    const labels = await recurrence.locator('.eigen-history text').evaluateAll(nodes => nodes.map(node => ({ text: node.textContent, x: node.getBBox().x, width: node.getBBox().width })));
    assert.ok(labels.every(label => label.x >= -0.5 && label.x + label.width <= 370.5), 'Norm axis label leaves SVG canvas');
    assert.equal(await page.locator('.katex-error').count(), 0);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
    assert.deepEqual(errors, []);
    results.push({ width, referenceLinks: 10, readingSections: 6, formulaOverflow: equations, longestNormLabels: labels, errors });
    await page.close();
  }
  await browser.close();
  fs.writeFileSync(path.join(directory, 'reading-results.json'), JSON.stringify({ reviewedAt: new Date().toISOString(), results }, null, 2) + '\n');
  console.log(JSON.stringify({ results }, null, 2));
})().catch(error => { console.error(error); process.exit(1); });

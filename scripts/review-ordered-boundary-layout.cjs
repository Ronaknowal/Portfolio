const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const directory = path.resolve(__dirname, '../scratch/ordered-pattern-lesson-review');

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  try {
    for (const width of [1440, 390]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      await page.goto(`${process.env.LEARNING_BASE_URL || 'http://127.0.0.1:5173'}/learn/path/full-curriculum/binary-search-sorting-two-pointer-patterns?module=data-structures-algorithms`);
      const boundary = page.getByRole('region', { name: 'Boundary search investigation', exact: true });
      await boundary.waitFor();
      const scroll = boundary.locator('.ordered-scroll');
      assert.ok(await scroll.evaluate(node => node.scrollWidth <= node.clientWidth + 1), 'six-element default fits');
      await boundary.getByRole('button', { name: 'Next comparison' }).click();
      await boundary.scrollIntoViewIfNeeded();
      await page.addStyleTag({ content: '.learn-nav { visibility: hidden !important; }' });
      await boundary.screenshot({ path: path.join(directory, `boundary-regions-${width}.png`) });
      const largeValues = Array.from({ length: 16 }, (_, index) => Math.round(-1000 + index * 2000 / 15));
      await boundary.getByLabel('Boundary sorted values').fill(largeValues.join(','));
      await boundary.getByRole('button', { name: 'Apply array' }).click();
      assert.equal(await boundary.getByRole('alert').count(), 0);
      assert.equal(await boundary.locator('.ordered-value').count(), 16);
      const textFits = await boundary.locator('.ordered-value > strong').evaluateAll(nodes => nodes.every(node => {
        const range = document.createRange();
        range.selectNodeContents(node);
        return range.getBoundingClientRect().width <= node.getBoundingClientRect().width + 1;
      }));
      assert.ok(textFits, 'four-digit signed values fit their cells');
      await scroll.focus();
      await page.keyboard.press('ArrowRight');
      await page.waitForFunction(() => document.querySelector('.ordered-scroll').scrollLeft > 0);
      const numericCells = await boundary.locator('.ordered-value > strong').allTextContents();
      assert.deepEqual(numericCells.map(Number), largeValues);
      assert.ok(await boundary.evaluate(node => node.scrollWidth <= node.clientWidth + 1));
      await boundary.screenshot({ path: path.join(directory, `boundary-long-${width}.png`) });
      await boundary.getByLabel('Boundary sorted values').fill('');
      await boundary.getByRole('button', { name: 'Apply array' }).click();
      assert.equal(await boundary.locator('[data-boundary]').getAttribute('data-boundary'), '0');
      assert.equal(await boundary.locator('.ordered-gap').count(), 1);
      const rates = page.getByRole('region', { name: 'Feasible rate investigation', exact: true });
      await rates.scrollIntoViewIfNeeded();
      await rates.screenshot({ path: path.join(directory, `rates-budget-${width}.png`) });
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
      results.push({ width, defaultFits: true, largeSignedValuesFit: true, longStripKeyboardScroll: true, emptyBoundary: 0, pageOverflow: false });
      await page.close();
    }
    fs.writeFileSync(path.join(directory, 'boundary-layout-results.json'), JSON.stringify({ checkedOn: new Date().toISOString(), results }, null, 2) + '\n');
    console.log(JSON.stringify(results));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });

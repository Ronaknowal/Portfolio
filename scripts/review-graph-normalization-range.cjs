const assert = require('node:assert/strict');
const fs = require('node:fs');
const {chromium} = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

(async () => {
  const browser = await chromium.launch({channel: 'msedge', headless: true});
  const records = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({viewport: {width, height: 1000}, reducedMotion: 'reduce'});
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/graph-fundamentals-adjacency-laplacian-connectivity?module=math-foundations');
      const program = page.locator('.python-example').filter({has: page.getByRole('heading', {name: 'Keep the isolated row explicit', exact: true})});
      await program.waitFor();
      await page.evaluate(() => document.fonts.ready);
      const text = await program.innerText();
      assert(text.includes('from math import isfinite, sqrt'));
      assert(text.includes('if not all(isfinite(value) for value in inverse):'));
      assert(text.includes('stochastic isolated row: [0, 0, 0, 1]'));
      const geometry = await program.evaluate(node => {
        const question = node.previousElementSibling;
        scrollTo({top: scrollY + question.getBoundingClientRect().top - 80, behavior: 'instant'});
        return {question: question.textContent, pageOverflow: document.documentElement.scrollWidth > innerWidth + 1};
      });
      assert(geometry.question.startsWith('Before running:'));
      assert(!geometry.pageOverflow);
      assert(await page.evaluate(() => document.fonts.check('16px "Space Grotesk"')));
      await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
      await page.waitForTimeout(180);
      const capture = `normalization-range-amendment-${width}.png`;
      await page.screenshot({path: 'scratch/graph-fundamentals-browser/' + capture});
      assert.deepEqual(errors, []);
      records.push({width, ...geometry, capture, errors, guardAndOutputVisible: true});
      await page.close();
    }
    fs.writeFileSync('scratch/graph-fundamentals-browser/range-amendment-results.json', JSON.stringify({checkedAt: new Date().toISOString(), allPassed: true, records}, null, 2));
    console.log(JSON.stringify(records, null, 2));
  } finally {
    await browser.close();
  }
})().catch(error => {console.error(error); process.exitCode = 1;});

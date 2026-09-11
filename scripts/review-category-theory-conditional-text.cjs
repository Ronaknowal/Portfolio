const assert = require('node:assert/strict');
const fs = require('node:fs');
const { createHash } = require('node:crypto');
const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1050 }, reducedMotion: 'reduce' });
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/category-theory-emerging-use-in-ml?module=math-foundations', { waitUntil: 'domcontentloaded' });
      const paragraph = page.locator('.category-theory-lesson p').filter({ hasText: 'With prior p(x) and channel' });
      await paragraph.waitFor(); await page.evaluate(() => document.fonts.ready);
      const text = await paragraph.innerText();
      assert(text.includes('For q(y)> 0, the reverse conditional'));
      assert(!text.includes('&gt'));
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
      await paragraph.screenshot({ path: `scratch/category-theory-browser/conditional-text-${width}.png` });
      assert.deepEqual(errors, []);
      records.push({ width, text, errors });
      await page.close();
    }
    const file = 'src/learn/data/topics/category-theory-emerging-use-in-ml.jsx';
    fs.writeFileSync('scratch/category-theory-browser/conditional-text-results.json', JSON.stringify({ verifiedAt: new Date().toISOString(), passed: true, bodySha256: createHash('sha256').update(fs.readFileSync(file)).digest('hex'), records }, null, 2) + '\n');
    console.log('Category positive-evidence condition renders correctly at all three widths.');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

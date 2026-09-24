const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const directory = 'scratch/lesson-scroll-preference-review';
const base = process.env.REVIEW_BASE_URL || 'http://127.0.0.1:4173';
fs.mkdirSync(directory, { recursive: true });

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  try {
    for (const width of [1440, 390]) {
      for (const preference of ['reduce', 'no-preference']) {
        for (const id of ['linux-basics-filesystems-processes', 'convex-optimization']) {
          const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: preference });
          const errors = [];
          page.on('pageerror', error => errors.push(error.message));
          await page.goto(`${base}/learn/path/full-curriculum/${id}`, { waitUntil: 'networkidle' });
          const links = page.locator('.lesson-intro nav a');
          await links.first().waitFor();
          assert.equal(await page.evaluate(() => getComputedStyle(document.documentElement).scrollBehavior), preference === 'reduce' ? 'auto' : 'smooth');
          const arrivals = [];
          for (const index of [0, (await links.count()) - 1]) {
            const link = links.nth(index);
            const href = await link.getAttribute('href');
            await link.scrollIntoViewIfNeeded();
            await link.focus();
            await page.keyboard.press('Enter');
            await page.waitForFunction(hash => {
              const target = document.getElementById(decodeURIComponent(hash.slice(1)));
              if (!target || location.hash !== hash) return false;
              const box = target.getBoundingClientRect();
              return box.top >= 60 && box.bottom < innerHeight;
            }, href);
            if (preference === 'no-preference') await page.waitForTimeout(750);
            const box = await page.evaluate(hash => {
              const target = document.getElementById(decodeURIComponent(hash.slice(1)));
              const bounds = target.getBoundingClientRect();
              return { top: bounds.top, bottom: bounds.bottom, text: target.textContent };
            }, href);
            assert.ok(box.top >= 60 && box.bottom < 1000);
            arrivals.push({ href, ...box });
          }
          assert.deepEqual(errors, []);
          results.push({ id, width, preference, arrivals, errors });
          await page.close();
        }
      }
    }
    fs.writeFileSync(`${directory}/results.json`, JSON.stringify({ checkedAt: new Date().toISOString(), results }, null, 2));
    console.log(JSON.stringify(results));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

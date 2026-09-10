const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const directory = 'scratch/lesson-intro-navigation-review';
fs.mkdirSync(directory, { recursive: true });
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  try {
    for (const width of [1440, 390, 320]) {
      for (const id of ['greedy-algorithms-exchange-arguments', 'dynamic-programming-states-transitions-optimization', 'linux-basics-filesystems-processes']) {
        const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
        await page.routeWebSocket('**', socket => socket.close());
        const errors = [];
        page.on('pageerror', error => errors.push(error.message));
        await page.goto(`http://127.0.0.1:5173/learn/path/full-curriculum/${id}`, { waitUntil: 'networkidle' });
        const intro = page.locator('.lesson-intro');
        await intro.waitFor();
        const items = await intro.locator('nav li').evaluateAll(nodes => nodes.map(node => {
          const rect = node.getBoundingClientRect();
          return { left: rect.left, right: rect.right, top: rect.top, bottom: rect.bottom, text: node.textContent };
        }));
        for (let index = 1; index < items.length; index += 1) {
          const previous = items[index - 1];
          const current = items[index];
          if (Math.abs(previous.top - current.top) < 1) assert.ok(current.left - previous.right >= 17, 'Adjacent navigation items overlap or lack spacing');
        }
        assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1));
        const links = intro.locator('nav a');
        for (const index of [0, (await links.count()) - 1]) {
          const link = links.nth(index);
          const href = await link.getAttribute('href');
          await link.focus();
          await page.keyboard.press('Enter');
          assert.equal(new URL(page.url()).hash, href);
          assert.ok(await page.evaluate(anchor => Boolean(document.getElementById(decodeURIComponent(anchor.slice(1)))), href));
        }
        await intro.scrollIntoViewIfNeeded();
        await intro.screenshot({ path: `${directory}/${id}-${width}.png` });
        assert.deepEqual(errors, []);
        results.push({ id, width, links: items.length });
        await page.close();
      }
    }
    fs.writeFileSync(`${directory}/results.json`, JSON.stringify({ at: new Date().toISOString(), results }, null, 2));
    console.log(JSON.stringify(results));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

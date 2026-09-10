const assert = require('node:assert/strict');
const fs = require('node:fs');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const base = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:5173';
const route = '/learn/path/full-curriculum/hashing-collision-resolution-amortized-analysis?module=data-structures-algorithms';
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  try {
    for (const width of [1440, 390]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      await page.goto(base + route);
      const navigation = page.getByRole('navigation', { name: 'In this lesson', exact: true });
      await navigation.waitFor();
      const destinations = await navigation.locator('a').evaluateAll(links => links.map(link => link.getAttribute('href')));
      const arrivals = [];
      for (const destination of destinations) {
        await navigation.locator(`a[href="${destination}"]`).focus();
        await page.keyboard.press('Enter');
        await page.waitForFunction(hash => window.location.hash === hash, destination);
        const heading = page.locator(`[id="${destination.slice(1)}"]`);
        await page.waitForFunction(id => {
          const bounds = document.getElementById(id).getBoundingClientRect();
          return bounds.top >= 50 && bounds.top < 180;
        }, destination.slice(1));
        const box = await heading.boundingBox();
        assert(box.y + box.height <= 1000, destination);
        arrivals.push({ destination, top: box.y, text: await heading.innerText() });
      }
      results.push({ width, keyboardArrivals: arrivals });
      await page.close();
    }
    const evidence = { checkedAt: new Date().toISOString(), base, results };
    fs.writeFileSync('scratch/hashing-amortized-browser/anchor-arrivals.json', JSON.stringify(evidence, null, 2));
    console.log(JSON.stringify(evidence, null, 2));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const directory = 'scratch/convex-optimization-review/reading';
fs.mkdirSync(directory, { recursive: true });
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1100 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/convex-optimization?module=math-foundations');
      const lesson = page.locator('.convex-optimization-lesson');
      await lesson.waitFor();
      await page.evaluate(() => document.fonts.ready);
      assert.ok((await lesson.innerText()).includes('.venv\\Scripts\\Activate.ps1'), 'PowerShell activation command changed');
      for (const index of [0, 7]) {
        const link = lesson.locator('.lesson-intro a').nth(index);
        const hash = await link.getAttribute('href');
        await link.scrollIntoViewIfNeeded();
        await link.focus(); await page.keyboard.press('Enter');
        assert.equal(new URL(page.url()).hash, hash);
        await page.waitForFunction(id => {
          const bounds = document.getElementById(id)?.getBoundingClientRect();
          return bounds && bounds.top >= 0 && bounds.top < innerHeight * 0.45;
        }, hash.slice(1));
      }
      await lesson.locator('details').evaluateAll(nodes => nodes.forEach(node => node.open = true));
      for (const index of [0, 2, 3, 4, 5, 6, 7]) {
        const heading = lesson.locator('h2').nth(index);
        await heading.evaluate(node => window.scrollTo({ top: window.scrollY + node.getBoundingClientRect().top - 90, behavior: 'instant' }));
        const position = await heading.boundingBox();
        assert.ok(position.y >= 70 && position.y <= 115, `Section ${index + 1} did not arrive in the reading viewport: ${position.y}`);
        await page.screenshot({ path: `${directory}/section-${index + 1}-${width}.png` });
      }
      await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = 'hidden'));
      const math = lesson.locator('.katex-display');
      for (let index = 0; index < await math.count(); index += 1) {
        const equation = math.nth(index);
        const dimensions = await equation.evaluate(node => ({ width: node.clientWidth, scroll: node.scrollWidth }));
        assert.ok(dimensions.scroll <= dimensions.width + 2);
        if (width === 320) await equation.screenshot({ path: `${directory}/equation-${index}-${width}.png` });
      }
      await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = ''));
      const summaries = lesson.locator('summary');
      for (const index of [0, (await summaries.count()) - 1]) {
        const summary = summaries.nth(index);
        await summary.evaluate(node => node.parentElement.open = false);
        await summary.focus(); await page.keyboard.press('Enter');
        assert.equal(await summary.evaluate(node => node.parentElement.open), true);
      }
      const links = await lesson.locator('.lesson-sources a').evaluateAll(nodes => nodes.map(node => ({ href: node.href, title: node.textContent, target: node.target, rel: node.rel })));
      assert.ok(links.length >= 10);
      assert.ok(links.every(link => link.href.startsWith('https://') && link.title && link.target === '_blank' && link.rel.includes('noreferrer')));
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 2), false);
      assert.deepEqual(errors, []);
      results.push({ width, displayedEquations: await math.count(), sectionsCaptured: 7, resourceLinks: links.length, keyboardDisclosures: 2, keyboardAnchorArrivals: 2, errors });
      await page.close();
    }
    fs.writeFileSync(`${directory}/results.json`, JSON.stringify({ checkedAt: new Date().toISOString(), results }, null, 2));
    console.log(JSON.stringify(results));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

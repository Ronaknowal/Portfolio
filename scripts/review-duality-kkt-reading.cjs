const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const directory = path.resolve(__dirname, '../scratch/duality-kkt-browser');
fs.mkdirSync(directory, { recursive: true });

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  const errors = [];
  for (const width of [1440, 390, 320]) {
    const page = await browser.newPage({ viewport: { width, height: 1050 }, reducedMotion: 'reduce' });
    page.on('pageerror', error => errors.push(error.message));
    await page.routeWebSocket('**', socket => socket.close());
    await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/convex-duality-lagrangian-methods-kkt-conditions');
    await page.locator('.duality-lesson').waitFor();
    const lesson = page.locator('.duality-lesson');
    await lesson.locator('.duality-lab').first().waitFor();
    await lesson.locator('.katex-display').first().waitFor();
    for (let index = 0; index < 9; index++) {
      const heading = lesson.locator('h2').nth(index);
      await heading.evaluate(node => window.scrollTo(0, node.getBoundingClientRect().top + scrollY - 100));
      await page.screenshot({ path: path.join(directory, `reading-${index + 1}-${width}.png`) });
    }
    // Show the static geometry fully, independently of a fixed navigation overlay.
    await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = 'hidden'));
    for (let index = 0; index < 2; index++) {
      await lesson.locator('.duality-inline').nth(index).screenshot({ path: path.join(directory, `inline-${index + 1}-${width}.png`) });
    }
    await lesson.locator('.lesson-sources').screenshot({ path: path.join(directory, `sources-${width}.png`) });
    await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = ''));
    await lesson.locator('details').evaluateAll(nodes => nodes.forEach(node => node.open = true));
    const scrollableTables = lesson.locator('.lesson-table-wrap');
    let keyboardScrolledTables = 0;
    for (let index = 0; index < await scrollableTables.count(); index++) {
      const region = scrollableTables.nth(index);
      if (await region.evaluate(node => node.scrollWidth > node.clientWidth + 1)) {
        await region.focus();
        for (let press = 0; press < 6; press++) await page.keyboard.press('ArrowRight');
        await page.waitForFunction(node => node.scrollLeft > 0, await region.elementHandle());
        keyboardScrolledTables++;
      }
    }
    const math = await lesson.locator('.katex-display').evaluateAll(nodes => nodes.map(node => ({
      text: node.textContent.slice(0, 100), width: node.clientWidth, scroll: node.scrollWidth,
    })));
    const overflowingMath = math.filter(row => row.scroll > row.width + 2);
    assert.deepEqual(overflowingMath, [], JSON.stringify(overflowingMath));
    const plotTextOutside = await lesson.locator('svg.duality-plot').evaluateAll(nodes => nodes.flatMap((svg, index) => {
      const boundary = svg.getBoundingClientRect();
      return [...svg.querySelectorAll('text')].filter(node => {
        const rect = node.getBoundingClientRect();
        return rect.left < boundary.left - 1 || rect.right > boundary.right + 1 || rect.top < boundary.top - 1 || rect.bottom > boundary.bottom + 1;
      }).map(node => ({ plot: index, text: node.textContent }));
    }));
    const pageOverflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1);
    assert.deepEqual(plotTextOutside, []);
    assert.equal(pageOverflow, false);
    assert.equal(await lesson.locator('.katex-error').count(), 0);
    results.push({ width, ordinaryReadingSections: 9, inlineFigures: 2, keyboardScrolledTables, formulaCount: math.length, overflowingMath, plotTextOutside, pageOverflow });
    await page.close();
  }
  await browser.close();
  assert.deepEqual(errors, []);
  fs.writeFileSync(path.join(directory, 'reading-results.json'), JSON.stringify({ checkedAt: new Date().toISOString(), results, errors }, null, 2));
  console.log(JSON.stringify(results, null, 2));
})().catch(error => { console.error(error); process.exit(1); });

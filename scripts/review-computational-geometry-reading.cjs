const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const directory = path.resolve('scratch/computational-geometry-browser');

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/computational-geometry-robust-predicates-convex-hulls?module=data-structures-algorithms');
      const lesson = page.locator('.computational-geometry-lesson');
      await lesson.waitFor();
      const record = { width, captures: [] };
      const reading = async (locator, name) => {
        await locator.evaluate(element => window.scrollTo(0, element.getBoundingClientRect().top + window.scrollY - 100));
        await page.screenshot({ path: path.join(directory, `detail-${name}-${width}.png`) });
        record.captures.push(`detail-${name}-${width}.png`);
      };
      await reading(lesson.locator('.geometry-calculation'), 'calculation');
      await reading(lesson.getByRole('heading', { name: 'Why the scan works', exact: true }), 'hull-invariant');
      await reading(lesson.getByRole('heading', { name: 'Locate a query with a boundary-first ray', exact: true }), 'ray-rule');
      await reading(lesson.locator('.python-example').nth(0), 'python-program');
      const example = lesson.locator('.python-example').nth(0);
      const code = example.locator(':scope > div').nth(0);
      const codeOverflow = await code.evaluate(element => ({ client: element.clientWidth, scroll: element.scrollWidth, mode: getComputedStyle(element).overflowX }));
      if (codeOverflow.scroll > codeOverflow.client + 1) assert(['auto', 'scroll'].includes(codeOverflow.mode));
      await reading(example.locator('.lesson-note'), 'python-output');
      await reading(lesson.locator('.lesson-sources'), 'sources');
      if (width === 320) {
        const hull = lesson.getByRole('region', { name: 'Monotone convex hull construction' });
        await hull.getByLabel('Point records, one x,y per line').fill('0,0\n8,0\n8,8\n0,8\n4,4');
        await hull.getByRole('button', { name: 'Apply point records', exact: true }).click();
        await hull.getByRole('button', { name: 'Show completed hull', exact: true }).click();
        await reading(hull.locator('figure'), 'edge-point-labels');
        const precision = lesson.getByRole('region', { name: 'Exact and floating predicate comparison' });
        await precision.getByLabel('Failure stage').selectOption('input');
        await reading(precision.locator('.geometry-arithmetic'), 'exact-input');
      }
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
      assert.deepEqual(await lesson.locator('.geometry-calculation, .geometry-lab, .geometry-inline').evaluateAll(elements => elements.filter(element => element.scrollWidth > element.clientWidth + 1).map(element => element.className)), []);
      record.nativeCodeOverflow = codeOverflow;
      results.push(record);
      await page.close();
    }
    const result = { checkedAt: new Date().toISOString(), results };
    fs.writeFileSync(path.join(directory, 'reading-results.json'), JSON.stringify(result, null, 2));
    console.log(JSON.stringify(result, null, 2));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

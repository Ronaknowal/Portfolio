const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1080 } });
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/algebra-functions-exponentials-logarithms', { waitUntil: 'domcontentloaded', timeout: 60000 });
      const lab = page.getByRole('region', { name: 'Growth and scale investigation' });
      await lab.waitFor(); await page.evaluate(() => document.fonts.ready);
      await lab.getByRole('checkbox').check();
      const shape = await lab.locator('svg').evaluate(svg => ({ lines: [...svg.querySelectorAll('polyline')].map(line => line.getAttribute('points').split(' ').map(pair => pair.split(',').map(Number))), text: svg.textContent }));
      assert.equal(shape.lines[1].length, 81);
      const points = shape.lines[1];
      const first = points[0], middle = points[40], last = points[80];
      const fraction = (middle[1] - first[1]) / (last[1] - first[1]);
      const expected = Math.log10(180 / 100) / Math.log10(260 / 100);
      assert(Math.abs(fraction - expected) < 1e-12);
      assert(Math.abs(fraction - .5) > .1);
      assert((await lab.innerText()).includes('180'));
      await lab.evaluate(node => scrollTo({ top: node.getBoundingClientRect().top + scrollY - 84, behavior: 'instant' }));
      await page.screenshot({ path: `scratch/algebra-functions-browser/additive-log-repair-${width}.png` });
      await lab.getByRole('checkbox').uncheck();
      const linear = await lab.locator('polyline').nth(1).getAttribute('points');
      const rows = linear.split(' ').map(pair => pair.split(',').map(Number));
      assert(Math.abs((rows[40][1] - rows[0][1]) / (rows[80][1] - rows[0][1]) - .5) < 1e-12);
      assert(await page.evaluate(() => document.fonts.check('16px "Space Grotesk"')));
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      assert.deepEqual(errors, []);
      records.push({ width, additiveSamples: points.length, logMidpointFraction: fraction, independentExpectedFraction: expected, actualQuantity: 180, linearMidpointCorrect: true, errors });
      await page.close();
    }
  } finally { await browser.close(); }
  const result = { at: new Date().toISOString(), passed: true, records };
  fs.writeFileSync('scratch/algebra-functions-browser/additive-log-repair-results.json', JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result));
})().catch(error => { console.error(error); process.exit(1); });

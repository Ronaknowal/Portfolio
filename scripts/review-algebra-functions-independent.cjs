const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const path = require('node:path');

const directory = path.resolve('scratch/algebra-functions-independent');
fs.mkdirSync(directory, { recursive: true });

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [];
  try {
    for (const width of [390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1080 }, reducedMotion: 'reduce' });
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/algebra-functions-exponentials-logarithms', { waitUntil: 'domcontentloaded', timeout: 60000 });
      const growth = page.getByRole('region', { name: 'Growth and scale investigation' });
      await growth.waitFor();
      await page.evaluate(() => document.fonts.ready);
      assert(await page.evaluate(() => document.fonts.check('16px "Space Grotesk"')));
      const cases = [];
      for (const rate of [0.2, -0.5, 1]) {
        await growth.getByRole('combobox', { name: 'Growth rate', exact: true }).selectOption(String(rate));
        await growth.getByRole('checkbox').check();
        const lines = await growth.locator('polyline').evaluateAll(nodes => nodes.map(node => node.getAttribute('points').trim().split(/\s+/).map(pair => pair.split(',').map(Number))));
        assert.equal(lines.length, 2);
        assert(lines.every(line => line.length === 81));
        const additive = lines[1];
        const fractions = additive.map((point, i) => {
          const actual = (point[1] - additive[0][1]) / (additive[80][1] - additive[0][1]);
          const expected = Math.log((100 + 2 * i) / 100) / Math.log(260 / 100);
          assert(Math.abs(actual - expected) < 2e-14);
          return actual;
        });
        assert(Math.abs(fractions[40] - 0.5) > 0.1);
        const exponential = lines[0];
        for (let index = 0; index <= 80; index++) {
          const actual = (exponential[index][1] - exponential[0][1]) / (exponential[80][1] - exponential[0][1]);
          assert(Math.abs(actual - index / 80) < 2e-14);
        }
        assert((await growth.locator('.algebra-result').innerText()).includes('fixed addition gives 180 units'));
        if (rate === 0.2) {
          for (let attempt = 0; attempt < 3; attempt++) {
            await growth.locator('svg').evaluate(node => window.scrollTo({ top: scrollY + node.getBoundingClientRect().top - 110, behavior: 'instant' }));
            await page.waitForTimeout(250);
            const rectangle = await growth.locator('svg').boundingBox();
            if (rectangle.y >= 80 && rectangle.y < 200) break;
          }
          const captureRectangle = await growth.locator('svg').boundingBox();
          assert(captureRectangle.y >= 80 && captureRectangle.y < 200);
          await page.screenshot({ path: path.join(directory, `independent-log-geometry-${width}.png`) });
        }
        await growth.getByRole('checkbox').uncheck();
        const linear = await growth.locator('polyline').nth(1).evaluate(node => node.getAttribute('points').trim().split(/\s+/).map(pair => pair.split(',').map(Number)));
        for (let index = 0; index <= 80; index++) {
          const actual = (linear[index][1] - linear[0][1]) / (linear[80][1] - linear[0][1]);
          assert(Math.abs(actual - index / 80) < 2e-12, `Linear pixel-ratio roundoff: ${actual - index / 80}`);
        }
        cases.push({ rate, actualLogMidpointFraction: fractions[40], additiveSamplesChecked: 81, exponentialSamplesChecked: 81, linearSamplesChecked: 81 });
      }
      assert.deepEqual(errors, []);
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      records.push({ width, originalFontLoaded: true, cases, errors });
      await page.close();
    }
  } finally { await browser.close(); }
  const result = { at: new Date().toISOString(), passed: true, records };
  fs.writeFileSync(path.join(directory, 'browser-results.json'), JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result));
})().catch(error => { console.error(error); process.exit(1); });

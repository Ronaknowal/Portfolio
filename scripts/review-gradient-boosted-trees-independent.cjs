const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const directory = 'scratch/gradient-boosted-trees-independent';

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1050 }, reducedMotion: 'reduce' });
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => { if (message.type() === 'error' && !message.text().includes('WebSocket')) errors.push(message.text()); });
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/gradient-boosted-trees-xgboost-lightgbm-catboost', { waitUntil: 'domcontentloaded', timeout: 60000 });
      const lesson = page.locator('.gradient-boosted-trees-lesson');
      await lesson.waitFor({ timeout: 60000 });
      await page.evaluate(() => document.fonts.ready);
      assert(await page.evaluate(() => document.fonts.check('16px "Space Grotesk"')));
      const correction = page.locator('[data-gbt-lab="correction"]');
      await correction.getByRole('textbox').fill('1, 1, 1, 1, 1, 1');
      await correction.getByRole('button', { name: 'Apply targets and restart' }).click();
      assert((await correction.innerText()).includes('0 → 0'));
      await correction.getByRole('textbox').fill('1, 2');
      await correction.getByRole('button', { name: 'Apply targets and restart' }).click();
      assert((await correction.getByRole('alert').innerText()).includes('previous valid data'));
      assert((await correction.innerText()).includes('0 → 0'));
      const ordered = page.locator('[data-gbt-lab="ordered"]');
      await ordered.getByRole('checkbox', { name: 'Reverse the permutation' }).check();
      await ordered.getByRole('slider', { name: 'Prior strength', exact: true }).fill('2');
      const prefix = ordered.locator('dl div').first();
      const before = await prefix.innerText();
      const checkbox = ordered.getByRole('checkbox', { name: "Flip the selected row's target" });
      await checkbox.focus();
      await page.keyboard.press('Space');
      assert(await checkbox.isChecked());
      assert.equal(await prefix.innerText(), before);
      assert((await prefix.innerText()).includes('0.66667'));
      const validation = page.locator('[data-gbt-lab="validation"]');
      await validation.getByRole('slider', { name: 'Correction tree depth', exact: true }).fill('3');
      await validation.getByRole('button', { name: 'Show validation choice' }).click();
      const geometry = await validation.locator('polyline').evaluateAll(lines => lines.map(line => {
        const points = [...line.points].map(p => [p.x, p.y]);
        return { within: points.every(p => p[0] >= 43.9 && p[1] >= 19.9 && p[1] <= 192.1), axisAligned: points.every((p, i) => !i || Math.abs(p[0] - points[i - 1][0]) < 1e-8 || Math.abs(p[1] - points[i - 1][1]) < 1e-8) };
      }));
      assert(geometry.every(item => item.within));
      assert(geometry.at(-1).axisAligned);
      assert.equal(await lesson.locator('.python-example').count(), 15);
      assert.deepEqual(await lesson.locator('nav a').evaluateAll(nodes => nodes.map(n => n.hash).filter(h => !document.getElementById(h.slice(1)))), []);
      const fit = await page.evaluate(() => ({ width: innerWidth, scroll: document.documentElement.scrollWidth, equations: [...document.querySelectorAll('.katex-display')].map(node => ({ client: node.clientWidth, scroll: node.scrollWidth })) }));
      assert(fit.scroll <= width + 1);
      assert(fit.equations.every(item => item.scroll <= item.client + 1));
      const target = width === 1440 ? validation : ordered;
      await target.evaluate(node => scrollTo({ top: scrollY + node.getBoundingClientRect().top - 85, behavior: 'instant' }));
      await page.screenshot({ path: `${directory}/changed-${width}.png` });
      assert.deepEqual(errors, []);
      records.push({ width, states: 7, font: 'Space Grotesk loaded', errors, fit, geometry, image: `${directory}/changed-${width}.png` });
      await page.close();
    }
  } finally { await browser.close(); }
  fs.writeFileSync(`${directory}/browser-results.json`, JSON.stringify({ checkedAt: new Date().toISOString(), scope: 'Seven complementary operated states per width; source reading and original author full-suite evidence remain distinct.', records }, null, 2) + '\n');
  console.log('Passed complementary GBDT interactions at 1440, 390, 320.');
})().catch(error => { console.error(error); process.exitCode = 1; });

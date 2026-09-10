const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const folder = path.resolve('scratch/constrained-multiobjective-browser');
fs.mkdirSync(folder, { recursive: true });

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  const errors = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1050 }, reducedMotion: 'reduce' });
      page.on('pageerror', error => errors.push(error.message));
      await page.routeWebSocket('**', socket => socket.close());
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/constrained-multi-objective-optimization');
      const lesson = page.locator('.constrained-lesson');
      await lesson.locator('.constrained-lab').first().waitFor();
      for (let index = 0; index < 9; index++) {
        await lesson.locator('h2').nth(index).evaluate(node => window.scrollTo(0, node.getBoundingClientRect().top + scrollY - 100));
        await page.screenshot({ path: path.join(folder, `reading-${index + 1}-${width}.png`) });
      }
      await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = 'hidden'));
      for (let index = 0; index < 2; index++) {
        await lesson.locator('.constrained-inline').nth(index).screenshot({ path: path.join(folder, `inline-${index + 1}-${width}.png`) });
      }
      await lesson.locator('.lesson-sources').screenshot({ path: path.join(folder, `sources-${width}.png`) });
      await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = ''));

      const visits = [];
      for (let labIndex = 0; labIndex < 5; labIndex++) {
        const controls = lesson.locator('.constrained-lab').nth(labIndex).locator('button:enabled,input:enabled,select:enabled');
        await controls.first().focus();
        for (let index = 0; index < await controls.count(); index++) {
          assert(await controls.nth(index).evaluate(node => node === document.activeElement), `lab ${labIndex} tab ${index}`);
          const style = await controls.nth(index).evaluate(node => ({ name: node.getAttribute('aria-label') || node.textContent.trim(), outline: getComputedStyle(node).outlineStyle, width: getComputedStyle(node).outlineWidth }));
          assert.notEqual(style.outline, 'none');
          assert(parseFloat(style.width) > 0);
          visits.push(style);
          await page.keyboard.press('Tab');
        }
      }
      const slider = lesson.getByRole('slider', { name: 'Target coordinate 1', exact: true });
      await slider.focus();
      await page.keyboard.press('Home');
      await page.keyboard.press('ArrowRight');
      assert.equal(await slider.inputValue(), '-1.9');
      const select = lesson.locator('[data-lab="constraint-penalty"] select');
      await select.focus();
      await page.keyboard.press('End');
      await page.keyboard.press('Enter');
      assert.equal(await select.inputValue(), 'barrier');
      await lesson.locator('.constrained-practice summary').last().focus();
      await page.keyboard.press('Space');
      assert.equal(await lesson.locator('.constrained-practice details[open]').count(), 1);
      await lesson.locator('details').evaluateAll(nodes => nodes.forEach(node => node.open = true));
      let keyboardScrolledTables = 0;
      const tables = lesson.locator('.lesson-table-wrap,.constrained-table');
      for (let index = 0; index < await tables.count(); index++) {
        const table = tables.nth(index);
        if (await table.evaluate(node => node.scrollWidth > node.clientWidth + 1)) {
          await table.focus();
          for (let press = 0; press < 6; press++) await page.keyboard.press('ArrowRight');
          await page.waitForFunction(node => node.scrollLeft > 0, await table.elementHandle());
          keyboardScrolledTables++;
        }
      }
      const math = await lesson.locator('.katex-display').evaluateAll(nodes => nodes.map(node => ({ text: node.textContent.slice(0, 100), width: node.clientWidth, scroll: node.scrollWidth })));
      const overflowingMath = math.filter(row => row.scroll > row.width + 2);
      const clippedText = await lesson.locator('svg.constrained-plot').evaluateAll(nodes => nodes.flatMap((svg, index) => {
        const boundary = svg.getBoundingClientRect();
        return [...svg.querySelectorAll('text')].filter(node => {
          const box = node.getBoundingClientRect();
          return box.left < boundary.left - 1 || box.right > boundary.right + 1 || box.top < boundary.top - 1 || box.bottom > boundary.bottom + 1;
        }).map(node => ({ plot: index, text: node.textContent }));
      }));
      const pageOverflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1);
      results.push({ width, ordinarySections: 9, inlineFigures: 2, keyboardControls: visits.length, visits, keyboardScrolledTables, mathCount: math.length, overflowingMath, clippedText, pageOverflow });
      fs.writeFileSync(path.join(folder, 'reading-in-progress.json'), JSON.stringify(results, null, 2));
      assert.deepEqual(overflowingMath, []);
      assert.deepEqual(clippedText, []);
      assert.equal(pageOverflow, false);
      assert.equal(await lesson.locator('.katex-error').count(), 0);
      await page.close();
    }
  } finally { await browser.close(); }
  assert.deepEqual(errors, []);
  fs.writeFileSync(path.join(folder, 'reading-results.json'), JSON.stringify({ at: new Date().toISOString(), status: 'passed', results, errors }, null, 2));
  console.log(results.map(({ width, ordinarySections, inlineFigures, keyboardControls, mathCount, keyboardScrolledTables }) => ({ width, ordinarySections, inlineFigures, keyboardControls, mathCount, keyboardScrolledTables })));
})().catch(error => { console.error(error); process.exitCode = 1; });

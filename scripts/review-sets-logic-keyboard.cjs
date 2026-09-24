const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const directory = path.resolve('scratch/sets-logic-browser');
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [], errors = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      page.on('pageerror', error => errors.push(error.message));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/sets-logic-relations-proof-techniques?module=math-foundations');
      const lesson = page.locator('.sets-logic-lesson');
      await lesson.waitFor(); await page.evaluate(() => document.fonts.ready);
      const diagonal = lesson.getByRole('region', { name: 'Diagonal missing subset investigation', exact: true });
      await diagonal.getByRole('button', { name: 'Reveal the diagonal subset', exact: true }).focus();
      await page.keyboard.press('Enter');
      const controls = [];
      for (const select of await lesson.getByRole('combobox').all()) {
        const options = await select.locator('option').evaluateAll(nodes => nodes.map(node => node.value));
        await select.focus();
        assert(await select.evaluate(node => document.activeElement === node && getComputedStyle(node).outlineStyle !== 'none'));
        await page.keyboard.press('Home'); await page.keyboard.press('ArrowDown'); await page.keyboard.press('Enter');
        assert.equal(await select.inputValue(), options[1]);
        controls.push(await select.getAttribute('aria-label'));
      }
      const relation = lesson.getByRole('region', { name: 'Relation properties and grouping investigation', exact: true });
      await relation.getByLabel('Relation preset').selectOption('moduloThree');
      const matrix = relation.getByRole('region', { name: 'Editable relation matrix; scroll horizontally if necessary', exact: true });
      await matrix.focus();
      const extent = await matrix.evaluate(node => ({ visible: node.clientWidth, full: node.scrollWidth }));
      if (extent.full > extent.visible + 1) {
        for (let index = 0; index < 15; index++) await page.keyboard.press('ArrowRight');
        await page.waitForTimeout(200);
        assert(await matrix.evaluate(node => node.scrollLeft > 0));
      }
      for (const lab of await lesson.locator('.sets-logic-lab').all()) {
        await lab.getByRole('button', { name: 'Reset', exact: true }).focus(); await page.keyboard.press('Enter');
      }
      await lesson.locator('.katex-display').nth(1).evaluate(node => window.scrollTo({ top: node.getBoundingClientRect().top + scrollY - 85, behavior: 'instant' }));
      await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
      const capture = `keyboard-final-negation-${width}.png`;
      await page.screenshot({ path: path.join(directory, capture) });
      records.push({ width, controls, matrix: extent, capture, fonts: await page.evaluate(() => [...document.fonts].filter(font => font.status === 'loaded').map(font => font.family)) });
      await page.close();
    }
    assert.deepEqual(errors, []);
    fs.writeFileSync(path.join(directory, 'keyboard-results.json'), JSON.stringify({ checkedAt: new Date().toISOString(), passed: true, records, errors }, null, 2));
    console.log('Native select Home/ArrowDown/Enter, horizontal matrix panning and six keyboard resets passed at all widths.');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

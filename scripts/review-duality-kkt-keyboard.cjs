const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const folder = path.resolve('scratch/duality-kkt-browser');

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1050 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/convex-duality-lagrangian-methods-kkt-conditions');
      await page.locator('.duality-lab').first().waitFor();
      const visits = [];
      for (let labIndex = 0; labIndex < 4; labIndex++) {
        const controls = page.locator('.duality-lab').nth(labIndex).locator('button:enabled,input,select');
        const total = await controls.count();
        await controls.first().focus();
        for (let index = 0; index < total; index++) {
          assert(await controls.nth(index).evaluate(node => node === document.activeElement), `lab ${labIndex} tab ${index}`);
          const state = await controls.nth(index).evaluate(node => ({
            name: node.getAttribute('aria-label') || node.textContent.trim(),
            outline: getComputedStyle(node).outlineStyle,
            outlineWidth: getComputedStyle(node).outlineWidth,
          }));
          assert.notEqual(state.outline, 'none');
          assert(parseFloat(state.outlineWidth) > 0);
          visits.push(state);
          await page.keyboard.press('Tab');
        }
      }
      const select = page.getByLabel('Value function', { exact: true });
      await select.focus();
      await page.keyboard.press('End');
      await page.keyboard.press('Enter');
      assert.equal(await select.inputValue(), 'kink');
      const price = page.getByRole('slider', { name: 'Optimal kink multiplier', exact: true });
      await price.focus();
      await page.keyboard.press('Home');
      await page.keyboard.press('ArrowRight');
      assert.equal(await price.inputValue(), '0.25');
      const practice = page.locator('.duality-practice').first();
      await practice.locator('summary').last().focus();
      await page.keyboard.press('Space');
      assert.equal(await practice.locator('details[open]').count(), 1);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
      results.push({ width, orderedEnabledControls: visits.length, visits, conditionalKinkKeyboard: true, practiceKeyboard: true });
      await page.close();
    }
  } finally { await browser.close(); }
  fs.writeFileSync(path.join(folder, 'keyboard-results.json'), JSON.stringify({ at: new Date().toISOString(), results }, null, 2));
  console.log(results.map(row => ({ width: row.width, orderedEnabledControls: row.orderedEnabledControls, conditionalKinkKeyboard: true })));
})().catch(error => { console.error(error); process.exitCode = 1; });

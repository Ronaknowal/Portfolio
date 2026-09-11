const assert = require('node:assert/strict');
const fs = require('node:fs');
const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1050 }, reducedMotion: 'reduce' });
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/category-theory-emerging-use-in-ml?module=math-foundations', { waitUntil: 'domcontentloaded' });
      const lesson = page.locator('.category-theory-lesson');
      await lesson.locator('h2').last().waitFor();
      await page.evaluate(() => document.fonts.ready);
      const composition = lesson.getByRole('region', { name: 'Typed composition investigation', exact: true });
      const input = composition.getByRole('combobox', { name: 'Starting element in A', exact: true });
      await input.focus();
      await page.keyboard.press('Home');
      await page.keyboard.press('ArrowDown');
      await page.keyboard.press('Enter');
      assert.equal(await input.inputValue(), '1');
      assert((await composition.locator('.category-result').innerText()).includes('sends 1 to'));
      const reset = composition.getByRole('button', { name: 'Reset investigation' });
      await reset.focus();
      assert(await reset.evaluate(node => getComputedStyle(node).outlineStyle !== 'none'));
      await page.keyboard.press('Enter');
      assert.equal(await input.inputValue(), '0');
      const example = lesson.locator('.python-example').first();
      await example.evaluate(node => window.scrollTo({ top: node.getBoundingClientRect().top + scrollY - 85, behavior: 'instant' }));
      await page.screenshot({ path: `scratch/category-theory-browser/keyboard-program-${width}.png` });
      records.push({ width, nativeSelectKeys: true, resetEnter: true, focusVisible: true });
      await page.close();
    }
    fs.writeFileSync('scratch/category-theory-browser/keyboard-results.json', JSON.stringify({ verifiedAt: new Date().toISOString(), passed: true, records }, null, 2) + '\n');
    console.log('Category native select and reset keyboard checks passed at all three widths.');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

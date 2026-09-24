const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const path = require('node:path');
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
      const lesson = page.locator('.algebra-lesson');
      await lesson.waitFor(); await page.evaluate(() => document.fonts.ready);
      const capture = async (locator, name) => {
        await locator.first().evaluate(node => window.scrollTo({ top: node.getBoundingClientRect().top + scrollY - 84, behavior: 'instant' }));
        await page.screenshot({ path: path.resolve(`scratch/algebra-functions-browser/${name}-${width}.png`) });
      };
      const boundary = await page.evaluate(async () => {
        const m = await import('/src/learn/data/algebra-functions-models.js');
        const calls = [() => m.functionProbeState('square', 1e-200), () => m.compositionState(1e-200), () => m.quadraticState(1, -1e-300), () => m.growthState(1e-320, 2), () => m.logarithmState(2, 1e-300)];
        const rejected = calls.map(call => { try { call(); return false; } catch (error) { return error instanceof RangeError; } });
        return { rejected, solution: m.equationState(3, 6, 21).solution, square: m.functionProbeState('square', 2).y, roots: m.quadraticState(3, -4).roots, growth: m.growthState(.2, 4).active.growth, log: m.logarithmState(2, 1.5).value };
      });
      assert.deepEqual(boundary.rejected, [true, true, true, true, true]);
      assert.equal(boundary.solution, 5); assert.equal(boundary.square, 4); assert.deepEqual(boundary.roots, [1, 5]);
      assert((await lesson.locator('[aria-label="Growth and scale investigation"]').innerText()).includes('207.36'));
      const functionLab = lesson.locator('[aria-label="Function and inverse investigation"]');
      await functionLab.getByRole('slider').focus(); await page.keyboard.press('ArrowRight');
      assert((await functionLab.innerText()).includes('f(2.25) = 5.0625'));
      await functionLab.evaluate(node => window.scrollTo({ top: node.getBoundingClientRect().top + scrollY - 84, behavior: 'instant' }));
      await page.screenshot({ path: path.resolve(`scratch/algebra-functions-browser/final-grid-${width}.png`) });
      assert(await page.evaluate(() => document.fonts.check('16px "Space Grotesk"')));
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      assert.equal(await lesson.locator('.katex-error').count(), 0); assert.deepEqual(errors, []);
      const mathWidths = await lesson.locator('.katex-display').evaluateAll(nodes => nodes.map((node,index) => ({index,width:node.clientWidth,content:node.scrollWidth})));
      assert.equal(mathWidths.length, 7);
      assert.deepEqual(mathWidths.filter(row => row.content > row.width + 2), []);
      for (let i=0; i<mathWidths.length; i++) await capture(lesson.locator('.katex-display').nth(i), `final-equation-${i}`);
      await capture(lesson.locator('h2').nth(6), 'final-reading-7');
      await capture(lesson.locator('p').filter({ hasText: 'For b>0 and a positive integer n' }), 'final-fractional-powers');
      const code = lesson.locator('.python-example').last().locator('div[style*="white-space: pre"]').first();
      await code.focus();
      await page.keyboard.press('ArrowRight');
      await page.waitForTimeout(150);
      const codeGeometry = await code.evaluate(node => ({ width: node.clientWidth, content: node.scrollWidth, left: node.scrollLeft, focused: node === document.activeElement }));
      if (codeGeometry.content > codeGeometry.width) assert(codeGeometry.focused && codeGeometry.left > 0, 'Keyboard access to locally scrolled code');
      await capture(code, 'final-code-keyboard');
      records.push({ width, boundary, actualKeyboardGrid: true, codeGeometry, fonts: true, mathWidths, errors });
      await page.close();
    }
  } finally { await browser.close(); }
  const result = { at: new Date().toISOString(), passed: true, records };
  fs.writeFileSync(path.resolve('scratch/algebra-functions-browser/boundary-results.json'), JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result));
})().catch(error => { console.error(error); process.exit(1); });

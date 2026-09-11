const assert = require('node:assert/strict');
const fs = require('node:fs');
const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const directory = 'scratch/single-variable-calculus-independent-review';

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1050 }, reducedMotion: 'reduce' });
      const errors = [], failedRequests = [], captures = [];
      await page.routeWebSocket('**', socket => socket.close());
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => {
        if (message.type() === 'error' && !message.text().includes('[vite]')) errors.push(message.text());
      });
      page.on('requestfailed', request => failedRequests.push(request.url()));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/single-variable-calculus-limits-derivatives-integrals?module=math-foundations', { waitUntil: 'domcontentloaded' });
      const lesson = page.locator('.single-calculus-lesson');
      await lesson.waitFor();
      await page.evaluate(() => document.fonts.ready);
      const fonts = await page.evaluate(() => [...document.fonts].filter(font => font.status === 'loaded').map(font => font.family));
      assert(fonts.includes('Space Grotesk'));
      const lab = index => lesson.locator('.calculus-lab').nth(index);
      const keyboard = async (target, name) => {
        const button = target.getByRole('button', { name, exact: true });
        await button.focus();
        assert(await button.evaluate(node => document.activeElement === node && getComputedStyle(node).outlineStyle !== 'none'));
        await page.keyboard.press('Enter');
      };
      const capture = async (target, name) => {
        await target.evaluate(node => window.scrollTo({ top: node.getBoundingClientRect().top + scrollY - 85, behavior: 'instant' }));
        await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
        const file = `${directory}/${name}-${width}.png`;
        await page.screenshot({ path: file });
        captures.push(file);
      };

      await lab(1).getByRole('slider', { name: 'Output tolerance epsilon' }).fill('0.84');
      await lab(1).getByRole('slider', { name: 'Input radius delta' }).fill('0.2');
      assert.match(await lab(1).locator('.calculus-readout').innerText(), /guarantees the requested accuracy/);
      await lab(1).getByRole('combobox').selectOption('hole');
      assert.match(await lab(1).locator('.calculus-readout').innerText(), /f\(2\)=6/);
      assert.match(await lab(1).locator('.calculus-readout').innerText(), /guarantees/);
      await lab(1).getByRole('combobox').selectOption('jump');
      assert.match(await lab(1).locator('.calculus-readout').innerText(), /fails.*Interior witness/s);

      await lab(3).getByRole('combobox').selectOption('cusp');
      await lab(3).getByRole('textbox', { name: 'Left endpoint' }).fill('2');
      await lab(3).getByRole('textbox', { name: 'Right endpoint' }).fill('2.0000000000000004');
      await keyboard(lab(3), 'Apply interval');
      assert.match(await lab(3).locator('.calculus-signs').innerText(), /increasing/);
      assert.doesNotMatch(await lab(3).locator('.calculus-candidates > div').first().innerText(), /maximum/);
      assert.match(await lab(3).locator('.calculus-candidates > div').last().innerText(), /maximum/);
      await capture(lab(3).locator('.calculus-candidates'), 'amended-cusp-adjacent');
      await lab(3).getByRole('combobox').selectOption('motion');
      await lab(3).getByRole('textbox', { name: 'Left endpoint' }).fill('1');
      await lab(3).getByRole('textbox', { name: 'Right endpoint' }).fill('1.0000000000000002');
      await keyboard(lab(3), 'Apply interval');
      assert.match(await lab(3).locator('.calculus-signs').innerText(), /decreasing/);
      assert.match(await lab(3).locator('.calculus-candidates > div').first().innerText(), /maximum/);
      assert.doesNotMatch(await lab(3).locator('.calculus-candidates > div').first().innerText(), /minimum/);
      const beforeError = await lab(3).locator('.calculus-candidates').innerText();
      await lab(3).getByRole('textbox', { name: 'Left endpoint' }).fill('');
      await keyboard(lab(3), 'Apply interval');
      assert.equal(await lab(3).locator('.calculus-candidates').innerText(), beforeError);
      assert.match(await lab(3).getByRole('alert').innerText(), /last valid interval is retained/);
      await keyboard(lab(3), 'Reset extrema');

      await lab(4).getByRole('slider', { name: 'Upper time T' }).fill('2.25');
      await lab(4).getByRole('combobox', { name: 'Rectangle count' }).selectOption('3');
      await lab(4).getByRole('combobox', { name: 'Sample in each strip' }).selectOption('left');
      const geometry = await lab(4).locator('svg').first().evaluate(svg => {
        const clip = svg.querySelector('clipPath rect');
        return { plotWidth: +clip.getAttribute('width'), plotHeight: +clip.getAttribute('height'), rectangles: [...svg.querySelectorAll('rect.positive-panel,rect.negative-panel')].map(rect => ({ width: +rect.getAttribute('width'), height: +rect.getAttribute('height'), className: rect.getAttribute('class') })) };
      });
      assert.equal(geometry.rectangles.length, 3);
      for (const [index, velocity] of [9, 1.6875, -2.25].entries()) {
        const rect = geometry.rectangles[index];
        assert(Math.abs(rect.width - .75*geometry.plotWidth/4) < 1e-9);
        assert(Math.abs(rect.height - Math.abs(velocity)*geometry.plotHeight/14) < 1e-9);
        assert.equal(rect.className, velocity > 0 ? 'positive-panel' : 'negative-panel');
      }
      if (width === 320) await capture(lab(4).locator('.calculus-two-plots'), 'changed-signed-geometry');

      await lab(6).getByRole('slider', { name: 'Polynomial degree' }).fill('12');
      await lab(6).getByRole('slider', { name: 'Evaluation input' }).fill('0.05');
      const taylor = await lab(6).innerText();
      assert.match(taylor, /floating-point roundoff/);
      assert.match(taylor, /can exceed a tiny analytic bound/);
      assert.match(taylor, /2\.061e-27/);
      await capture(lab(6).locator('.calculus-readout'), 'amended-taylor-roundoff');
      const checkpoint = lesson.locator('.lesson-check').filter({ hasText: 'If a continuous velocity is negative' });
      assert.equal(await checkpoint.count(), 1);
      const question = checkpoint.locator('p').first();
      await checkpoint.locator('summary').focus();
      await page.keyboard.press('Enter');
      assert(await lesson.getByText('Wherever velocity is differentiable, acceleration is nonnegative;', { exact: false }).isVisible());
      if (width === 320) await capture(question, 'amended-acceleration-checkpoint');
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      assert.deepEqual(errors, []);
      assert.deepEqual(failedRequests, []);
      records.push({ width, fonts, captures, signedRectangleGeometry: geometry, errors, failedRequests, keyboardActions: 5 });
      await page.close();
    }
  } finally {
    await browser.close();
  }
  const result = { checkedAt: new Date().toISOString(), status: 'passed', records, scope: 'Reviewer amendment closures and independent actual signed-rectangle geometry, with keyboard and actual fonts; author comprehensive browser evidence is separate.' };
  fs.writeFileSync(`${directory}/browser-results.json`, JSON.stringify(result, null, 2) + '\n');
  console.log(JSON.stringify({ checkedAt: result.checkedAt, widths: records.map(r => r.width), captures: records.reduce((n,r) => n+r.captures.length,0) }));
})().catch(error => { console.error(error); process.exitCode = 1; });

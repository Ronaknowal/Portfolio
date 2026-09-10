const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const directory = path.resolve(__dirname, '../scratch/complexity-recursion-lesson-review');
fs.mkdirSync(directory, { recursive: true });

(async () => {
  const { sumCallTrace } = await import('../src/learn/data/complexity-recursion-models.js');
  const { complexityRecursionExamples } = await import('../src/learn/data/complexity-recursion-examples.js');
  const base = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:5173';
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  const errors = [];
  try {
    for (const width of [1440, 390]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      // Other lesson authors may edit generated navigation while this stateful
      // check runs. Hold this page's loaded source stable; production has no HMR.
      await page.routeWebSocket('**', socket => socket.close());
      page.on('pageerror', error => errors.push(error.message));
      await page.goto(`${base}/learn/path/full-curriculum/complexity-analysis-recursion?module=data-structures-algorithms`);
      await page.locator('.complexity-recursion-lesson').waitFor();
      const lesson = page.locator('.complexity-recursion-lesson');
      const lattice = page.getByRole('region', { name: 'Iteration lattice investigation', exact: true });
      const frames = page.getByRole('region', { name: 'Recursive frames investigation', exact: true });
      const recurrence = page.getByRole('region', { name: 'Recurrence level investigation', exact: true });
      async function screenshot(locator, name) {
        await locator.scrollIntoViewIfNeeded();
        const captureStyle = await page.addStyleTag({ content: '.learn-nav { visibility: hidden !important; }' });
        await locator.screenshot({ path: path.join(directory, `${name}-${width}.png`) });
        await captureStyle.evaluate(node => node.remove());
      }
      let loopCases = 0;
      for (const pattern of ['triangle', 'square', 'doubling']) {
        await lattice.getByLabel('Loop pattern').selectOption(pattern);
        for (const size of [0, 1, 2, 6, 16]) {
          const slider = lattice.getByRole('slider');
          await slider.focus();
          await page.keyboard.press('Home');
          for (let index = 0; index < size; index += 1) await page.keyboard.press('ArrowRight');
          assert.equal(await slider.inputValue(), String(size));
          const expected = pattern === 'square' ? size ** 2 : pattern === 'triangle' ? Math.abs(size * (size - 1) / 2) : size * (size > 1 ? Math.ceil(Math.log2(size)) : 0);
          assert.equal(Number(await lattice.locator('[data-lattice-total]').getAttribute('data-lattice-total')), expected);
          assert.equal(await lattice.locator('.is-work').count(), expected);
          if (size) {
            await lattice.getByRole('button', { name: `Inspect outer index ${size - 1}`, exact: true }).click();
            assert.match(await lattice.locator('.complexity-readout').innerText(), new RegExp(`Row ${size - 1}:`));
          }
          loopCases += 1;
        }
      }
      await lattice.getByRole('button', { name: 'Reset lattice' }).click();
      assert.equal(await lattice.getByRole('slider').inputValue(), '6');
      assert.equal(await lattice.getByLabel('Loop pattern').inputValue(), 'triangle');
      await lattice.getByRole('button', { name: 'Inspect outer index 4', exact: true }).click();
      await screenshot(lattice, 'lattice');

      let frameStates = 0;
      for (const values of [[], [3, 1, 4], [0], [-2, 5, 0, -9], [1, 2, 3, 4, 5, 6, 7, 8]]) {
        await frames.getByLabel('Suffix sum values').fill(values.join(', '));
        await frames.getByRole('button', { name: 'Apply values' }).click();
        const trace = sumCallTrace(values);
        for (let step = 0; step < trace.length; step += 1) {
          const state = trace[step];
          assert.equal(Number(await frames.locator('[data-frame-step]').getAttribute('data-frame-step')), step);
          assert.equal(Number(await frames.locator('[data-frame-count]').getAttribute('data-frame-count')), state.frames.length);
          const rendered = await frames.locator('.complexity-frame').allTextContents();
          const expectedFrames = [...state.frames].reverse();
          expectedFrames.forEach((frame, index) => {
            assert.ok(rendered[index].includes(`suffix(${frame.index})`));
            if (frame.phase === 'return') assert.ok(rendered[index].includes(`return ${frame.value}`));
            if (frame.phase === 'waiting') assert.ok(rendered[index].includes(`${values[frame.index]} + ?`));
          });
          if (values.length === 3 && step === 4) await screenshot(frames, 'pending-frames');
          frameStates += 1;
          if (step + 1 < trace.length) await frames.getByRole('button', { name: 'Advance one event' }).click();
        }
        assert.ok(await frames.getByRole('button', { name: 'Advance one event' }).isDisabled());
        assert.match(await frames.locator('.complexity-frames').innerText(), new RegExp(`Caller receives ${values.reduce((sum, value) => sum + value, 0)}`));
        await frames.getByRole('button', { name: 'Back one event' }).click();
        assert.equal(Number(await frames.locator('[data-frame-step]').getAttribute('data-frame-step')), trace.length - 2);
        await frames.getByRole('button', { name: 'Restart trace' }).click();
        assert.equal(await frames.locator('[data-frame-step]').getAttribute('data-frame-step'), '0');
      }
      for (const invalid of ['1,', '100', 'a', '1,2,3,4,5,6,7,8,9']) {
        const before = await frames.locator('.complexity-array').innerText();
        await frames.getByLabel('Suffix sum values').fill(invalid);
        await frames.getByRole('button', { name: 'Apply values' }).click();
        assert.ok(await frames.getByRole('alert').isVisible());
        assert.equal(await frames.locator('.complexity-array').innerText(), before);
      }
      await frames.getByRole('button', { name: 'Reset frames' }).click();
      assert.equal(await frames.getByLabel('Suffix sum values').inputValue(), '3, 1, 4');
      assert.equal(await frames.getByRole('alert').count(), 0);
      await frames.getByRole('button', { name: 'Advance one event' }).focus();
      await page.keyboard.press('Enter');
      assert.equal(await frames.locator('[data-frame-step]').getAttribute('data-frame-step'), '1');

      let recurrenceCases = 0;
      for (const pattern of ['chainConstant', 'chainLinear', 'halfConstant', 'halfLinear', 'twoHalfLinear']) {
        await recurrence.locator('select').first().selectOption(pattern);
        for (const size of [1, 8, 32]) {
          await recurrence.getByLabel('Recurrence input size').selectOption(String(size));
          const expected = { chainConstant: size, chainLinear: size * (size + 1) / 2, halfConstant: Math.log2(size) + 1, halfLinear: 2 * size - 1, twoHalfLinear: size * (Math.log2(size) + 1) }[pattern];
          assert.equal(Number(await recurrence.locator('[data-recurrence-total]').getAttribute('data-recurrence-total')), expected);
          const rows = await recurrence.locator('tbody tr').evaluateAll(nodes => nodes.map(node => [...node.children].map(cell => cell.textContent)));
          assert.equal(rows.reduce((sum, row) => sum + parseInt(row[4], 10), 0), expected);
          for (let depth = 0; depth < rows.length; depth += 1) {
            await recurrence.getByRole('button', { name: `Inspect recurrence depth ${depth}`, exact: true }).click();
            assert.equal(await recurrence.locator('.complexity-node-strip > span').count(), Number(rows[depth][1]));
          }
          recurrenceCases += 1;
        }
      }
      await recurrence.getByRole('button', { name: 'Reset recurrence' }).click();
      await screenshot(recurrence, 'recurrence-levels');
      for (const [index, name] of ['return-ladder', 'slice-storage'].entries()) {
        await screenshot(lesson.locator('.complexity-inline').nth(index), name);
      }
      const examples = lesson.locator('.python-example');
      assert.equal(await examples.count(), 10);
      for (const example of Object.values(complexityRecursionExamples)) {
        const program = examples.filter({ has: page.getByRole('heading', { name: example.title, exact: true, includeHidden: true }) });
        const content = await program.textContent();
        assert.ok(content.includes(example.code), `${example.title}: complete code`);
        assert.ok(content.includes(example.expected), `${example.title}: expected output`);
      }
      const anchors = await lesson.locator('nav a').evaluateAll(nodes => nodes.map(node => node.getAttribute('href')));
      assert.equal(new Set(anchors).size, 9);
      for (const anchor of anchors) assert.equal(await page.locator(`[id="${anchor.slice(1)}"]`).count(), 1, anchor);
      const practice = lesson.locator('.dsa-practice');
      assert.equal(await practice.locator('a[href*="leetcode.com/problems/"]').count(), 7);
      assert.equal(await practice.locator('details[open]').count(), 0);
      const hint = practice.locator('summary').first();
      await hint.focus();
      await page.keyboard.press('Enter');
      assert.equal(await practice.locator('details[open]').count(), 1);
      await page.keyboard.press('Space');
      assert.equal(await practice.locator('details[open]').count(), 0);
      await screenshot(practice, 'practice');
      await lesson.locator('a[href="#guided-dsa-practice"]').click();
      await page.waitForFunction(() => Math.abs(document.querySelector('#guided-dsa-practice').getBoundingClientRect().top) < 160);
      for (const lab of [lattice, frames, recurrence]) {
        const heights = await lab.locator('button,input,select').evaluateAll(nodes => nodes.map(node => node.getBoundingClientRect().height));
        assert.ok(heights.every(height => height >= 43), heights.join(','));
        assert.ok(await lab.evaluate(node => node.scrollWidth <= node.clientWidth + 2));
      }
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 2));
      assert.equal(await lesson.locator('a[href*="binary-search-sorting-two-pointer-patterns"]').count(), 1);
      results.push({ width, loopCases, frameStates, recurrenceCases, programs: 10, practiceLinks: 7, anchors: 9, keyboardResetInvalid: 'passed' });
      console.log(`Complexity/recursion ${width}px passed.`);
      await page.close();
    }
    assert.deepEqual(errors, []);
    fs.writeFileSync(path.join(directory, 'results.json'), JSON.stringify({ date: new Date().toISOString(), results, errors }, null, 2));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });

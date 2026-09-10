const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const directory = path.resolve(__dirname, '../scratch/ordered-pattern-lesson-review');
fs.mkdirSync(directory, { recursive: true });

(async () => {
  const { boundaryTrace, pairTrace, windowTrace, rateState } = await import('../src/learn/data/ordered-pattern-models.js');
  const { orderedPatternExamples } = await import('../src/learn/data/ordered-pattern-examples.js');
  const base = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:5173';
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  const errors = [];
  try {
    for (const width of [1440, 390]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      // Keep the initial source stable during concurrent authoring's Vite HMR.
      await page.routeWebSocket('**', socket => socket.close());
      page.on('pageerror', error => errors.push(error.message));
      await page.goto(`${base}/learn/path/full-curriculum/binary-search-sorting-two-pointer-patterns?module=data-structures-algorithms`);
      const lesson = page.locator('.ordered-pattern-lesson');
      await lesson.waitFor();
      const boundary = page.getByRole('region', { name: 'Boundary search investigation', exact: true });
      const merge = page.getByRole('region', { name: 'Stable merge investigation', exact: true });
      const pair = page.getByRole('region', { name: 'Pair elimination investigation', exact: true });
      const window = page.getByRole('region', { name: 'Moving window investigation', exact: true });
      const rates = page.getByRole('region', { name: 'Feasible rate investigation', exact: true });
      async function slider(locator, value, minimum) {
        await locator.focus();
        await page.keyboard.press('Home');
        for (let offset = minimum; offset < value; offset += 1) await page.keyboard.press('ArrowRight');
        assert.equal(await locator.inputValue(), String(value));
      }
      async function screenshot(locator, name) {
        await locator.scrollIntoViewIfNeeded();
        const captureStyle = await page.addStyleTag({ content: '.learn-nav { visibility: hidden !important; }' });
        await locator.screenshot({ path: path.join(directory, `${name}-${width}.png`) });
        await captureStyle.evaluate(node => node.remove());
      }
      await screenshot(lesson.locator('.lesson-intro'), 'route');
      let boundaryStates = 0;
      for (const values of [[], [4], [2, 4, 4, 4, 7, 9], [4, 4, 4], [-10, -5, 0, 10, 20]]) {
        await boundary.getByLabel('Boundary sorted values').fill(values.join(','));
        await boundary.getByRole('button', { name: 'Apply array' }).click();
        for (const side of ['left', 'right']) {
          await boundary.getByLabel('Boundary rule').selectOption(side);
          for (const target of [-10, 4, 20]) {
            await slider(boundary.getByRole('slider'), target, -10);
            // Home can leave the same value unchanged; explicitly restart.
            await boundary.getByRole('button', { name: 'Restart trace' }).click();
            const model = boundaryTrace(values, target, side);
            for (let step = 0; step < model.steps.length; step += 1) {
              const state = model.steps[step];
              assert.equal(await boundary.locator('.ordered-value').count(), values.length);
              assert.equal(await boundary.locator('.possible-gap').count(), state.high - state.low + 1);
              assert.equal(await boundary.locator('.proved-before').count(), state.low);
              assert.equal(await boundary.locator('.proved-after').count(), values.length - state.high);
              if (state.done) assert.equal(await boundary.locator('[data-boundary]').getAttribute('data-boundary'), String(model.boundary));
              else assert.match(await boundary.locator('.is-middle').innerText(), new RegExp(String(values[state.middle])));
              boundaryStates += 1;
              if (!state.done) await boundary.getByRole('button', { name: 'Next comparison' }).click();
            }
            assert.ok(await boundary.getByRole('button', { name: 'Next comparison' }).isDisabled());
          }
        }
      }
      for (const invalid of ['2,1', '1,', 'NaN', Array(17).fill('1').join(','), '1001']) {
        const old = await boundary.locator('.ordered-boundary-strip').innerText();
        await boundary.getByLabel('Boundary sorted values').fill(invalid);
        await boundary.getByRole('button', { name: 'Apply array' }).click();
        assert.ok(await boundary.getByRole('alert').isVisible());
        assert.equal(await boundary.locator('.ordered-boundary-strip').innerText(), old);
      }
      await boundary.getByRole('button', { name: 'Reset boundaries' }).click();
      assert.equal(await boundary.getByRole('alert').count(), 0);
      await boundary.getByRole('button', { name: 'Next comparison' }).focus();
      await page.keyboard.press('Enter');
      assert.equal(await boundary.locator('.proved-after').count(), 3);
      await screenshot(boundary, 'boundary-regions');
      await boundary.getByRole('button', { name: 'Previous comparison' }).click();
      assert.equal(await boundary.locator('.proved-after').count(), 0);

      await merge.getByRole('button', { name: 'Take left head' }).click();
      assert.match(await merge.locator('.ordered-readout').innerText(), /smaller/);
      assert.equal(await merge.locator('.ordered-output small').count(), 0);
      for (const lane of ['right', 'left', 'right', 'left', 'left', 'right']) await merge.getByRole('button', { name: `Take ${lane} head` }).click();
      assert.match(await merge.locator('.ordered-readout').innerText(), /original tie order changed/);
      assert.ok(await merge.getByRole('button', { name: 'Take left head' }).isDisabled());
      await screenshot(merge, 'merge-unstable');
      await merge.getByRole('button', { name: 'Reset merge' }).click();
      for (const lane of ['right', 'left', 'left', 'left', 'right', 'right']) await merge.getByRole('button', { name: `Take ${lane} head` }).click();
      assert.match(await merge.locator('.ordered-readout').innerText(), /original tie order preserved/);
      assert.deepEqual(await merge.locator('.ordered-output small').allTextContents(), ['ID D', 'ID A', 'ID B', 'ID C', 'ID E', 'ID F']);
      await screenshot(merge, 'merge-stable');

      let pairStates = 0;
      for (const target of [0, 3, 6, 11, 20]) {
        await slider(pair.getByRole('slider'), target, 0);
        await pair.getByRole('button', { name: 'Restart trace' }).click();
        const model = pairTrace([1, 2, 4, 5, 7, 9], target);
        for (let step = 0; step < model.steps.length; step += 1) {
          const state = model.steps[step];
          assert.equal(await pair.locator('[data-pair-action]').getAttribute('data-pair-action'), state.action);
          const count = Math.max(0, state.right - state.left + 1);
          assert.equal(await pair.locator('.is-candidate').count(), count * (count - 1) / 2);
          if (target === 6 && step === 2) await screenshot(pair, 'pair-elimination');
          pairStates += 1;
          if (step + 1 < model.steps.length) await pair.getByRole('button', { name: 'Next pair' }).click();
        }
      }
      await pair.getByRole('button', { name: 'Reset pairs' }).click();
      assert.equal(await pair.getByRole('slider').inputValue(), '11');

      let windowStates = 0;
      for (const [preset, values] of Object.entries({ positive: [2, 1, 3, 2, 4], zeros: [0, 0, 5], empty: [] })) {
        await window.getByLabel('Window input').selectOption(preset);
        for (const target of [1, 6, 15]) {
          await slider(window.getByRole('slider'), target, 1);
          await window.getByRole('button', { name: 'Restart trace' }).click();
          const model = windowTrace(values, target);
          for (let step = 0; step < model.steps.length; step += 1) {
            const state = model.steps[step];
            assert.equal(await window.locator('[data-window-step]').getAttribute('data-window-step'), String(step));
            assert.equal(await window.locator('.in-window').count(), state.right - state.left);
            assert.match(await window.locator('.ordered-readout').innerText(), new RegExp(`Current sum: ${state.total}`));
            if (preset === 'positive' && target === 6 && state.action === 'qualifies' && state.left === 3) await screenshot(window, 'window-best');
            windowStates += 1;
            if (step + 1 < model.steps.length) await window.getByRole('button', { name: 'Next event' }).click();
          }
        }
      }
      await window.getByRole('button', { name: 'Reset window' }).click();
      assert.equal(await window.getByLabel('Window input').inputValue(), 'positive');

      let rateCases = 0;
      for (const budget of [2, 3, 6, 10]) {
        await slider(rates.getByRole('slider'), budget, 2);
        for (const speed of [1, 3, 7]) {
          await rates.getByRole('button', { name: new RegExp(`^${speed} / slot`) }).click();
          const model = rateState(speed, budget);
          assert.equal(await rates.locator('[data-rate-total]').getAttribute('data-rate-total'), String(model.total));
          assert.equal(await rates.locator('.ordered-rate-jobs > div > div > span').count(), model.total);
          rateCases += 1;
        }
        if (budget === 2) await screenshot(rates, 'rates-impossible');
      }
      await rates.getByRole('button', { name: 'Reset rates' }).click();
      await screenshot(rates, 'rates-budget');

      for (const [index, name] of ['partition-regions', 'compaction', 'interval-union', 'prefix-cancellation'].entries()) {
        await screenshot(lesson.locator('.ordered-inline').nth(index), name);
      }
      const examples = lesson.locator('.python-example');
      assert.equal(await examples.count(), 17);
      for (const example of Object.values(orderedPatternExamples)) {
        const program = examples.filter({ has: page.getByRole('heading', { name: example.title, exact: true, includeHidden: true }) });
        const content = await program.textContent();
        assert.ok(content.includes(example.code), `${example.title}: complete code`);
        assert.ok(content.includes(example.expected), `${example.title}: expected output`);
      }
      const anchors = await lesson.locator('nav a').evaluateAll(nodes => nodes.map(node => node.getAttribute('href')));
      assert.equal(new Set(anchors).size, 10);
      for (const anchor of anchors) assert.equal(await page.locator(`[id="${anchor.slice(1)}"]`).count(), 1, anchor);
      const practice = lesson.locator('.dsa-practice');
      assert.equal(await practice.locator('a[href*="leetcode.com/problems/"]').count(), 14);
      assert.equal(await practice.locator('details[open]').count(), 0);
      const hint = practice.locator('summary').first();
      await hint.focus();
      await page.keyboard.press('Enter');
      assert.equal(await practice.locator('details[open]').count(), 1);
      await page.keyboard.press('Enter');
      assert.equal(await practice.locator('details[open]').count(), 0);
      await screenshot(practice.locator('.dsa-practice__stage').first(), 'practice');
      await lesson.locator('a[href="#guided-dsa-practice"]').click();
      await page.waitForFunction(() => Math.abs(document.querySelector('#guided-dsa-practice').getBoundingClientRect().top) < 160);
      for (const region of await lesson.locator('.ordered-lab, .ordered-inline').all()) {
        const overflow = await region.evaluate(node => node.scrollWidth > node.clientWidth + 1);
        assert.equal(overflow, false, await region.getAttribute('aria-label'));
      }
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false, 'page overflow');
      results.push({ width, boundaryStates, pairStates, windowStates, rateCases, mergePaths: 2, examples: 17, practiceLinks: 14, anchors: anchors.length, overflow: false });
      await page.close();
    }
    assert.deepEqual(errors, []);
    const record = { checkedOn: new Date().toISOString(), base, results, errors };
    fs.writeFileSync(path.join(directory, 'results.json'), JSON.stringify(record, null, 2) + '\n');
    console.log(JSON.stringify(record, null, 2));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });

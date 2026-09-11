const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const directory = path.resolve('scratch/conditioning-stability-browser');
fs.mkdirSync(directory, { recursive: true });

(async () => {
  const models = await import('../src/learn/data/conditioning-stability-models.js');
  const { conditioningStabilityExamples: examples } = await import('../src/learn/data/conditioning-stability-examples.js');
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1080 }, reducedMotion: 'reduce' });
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/conditioning-stability-numerical-analysis', { waitUntil: 'domcontentloaded', timeout: 60000 });
      const lesson = page.locator('.conditioning-lesson');
      await lesson.waitFor({ timeout: 60000 });
      await page.evaluate(() => document.fonts.ready);
      assert(await page.evaluate(() => document.fonts.check('16px "Space Grotesk"')));
      const capture = async (locator, name) => {
        await locator.first().evaluate(node => scrollTo({ top: node.getBoundingClientRect().top + scrollY - 84, behavior: 'instant' }));
        await page.waitForTimeout(120);
        await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
        await page.screenshot({ path: path.join(directory, `${name}-${width}.png`) });
      };
      const lab = name => lesson.locator(`.conditioning-lab[aria-label="${name}"]`);
      const slider = async (area, name, value) => { await area.getByRole('slider', { name, exact: true }).fill(String(value)); };
      let states = 0;
      assert.equal(await lesson.locator('h2').count(), 10);
      assert.equal(await lesson.locator('.conditioning-lab').count(), 6);
      assert.equal(await lesson.locator('.conditioning-figure').count(), 2);
      assert.deepEqual(await lesson.locator('nav a').evaluateAll(nodes => nodes.map(node => node.hash).filter(hash => !document.getElementById(hash.slice(1)))), []);
      for (let i = 0; i < 10; i += 1) await capture(lesson.locator('h2').nth(i), `reading-${i + 1}`);
      for (let i = 0; i < 2; i += 1) await capture(lesson.locator('.conditioning-figure').nth(i), `inline-${i + 1}`);

      const rounding = lab('Rounding cells investigation');
      for (const bin of [0, 1]) {
        await rounding.getByRole('combobox', { name: 'Exponent bin' }).selectOption(String(bin));
        for (const half of [0, 1, 3, 8, 15, 16]) {
          await slider(rounding, 'Half-step position', half);
          const expected = models.roundingCellState(bin, half);
          assert((await rounding.locator('.conditioning-readout').innerText()).includes(`stored = ${expected.rounded}`));
          states += 1;
        }
      }
      await slider(rounding, 'Half-step position', 3); await capture(rounding, 'rounding-tie');
      await rounding.getByRole('button').focus(); await page.keyboard.press('Enter');
      await rounding.getByRole('slider').focus(); await page.keyboard.press('ArrowRight');
      assert.equal(await rounding.getByRole('slider').inputValue(), '2');
      const wideRounding = rounding.locator('.conditioning-drawing');
      await wideRounding.focus(); await page.keyboard.press('End');

      const cancellation = lab('Cancellation investigation');
      for (const sign of [1, -1, 0]) {
        await cancellation.getByRole('combobox').selectOption(String(sign));
        for (const exponent of [0, 1, 26, 54, 60]) {
          await slider(cancellation, 'Dyadic exponent k', exponent);
          const expected = models.cancellationState(exponent, sign);
          assert((await cancellation.locator('.conditioning-readout').innerText()).includes(`Reference ≈ ${models.formatConditioning(expected.reference, 12)}`));
          if (sign !== 0) {
            const dots = await cancellation.locator('circle[data-exponent]').evaluateAll(nodes => nodes.map(node => ({ x: Number(node.getAttribute('cx')), y: Number(node.getAttribute('cy')), error: Number(node.dataset.error) })));
            assert(dots.length > 60 && dots.every(dot => Number.isFinite(dot.x) && Number.isFinite(dot.y) && dot.error > 0));
          }
          states += 1;
        }
      }
      await capture(cancellation, 'zero-reference');
      await cancellation.getByRole('combobox').selectOption('-1'); await slider(cancellation, 'Dyadic exponent k', 0);
      assert((await cancellation.getByRole('status').innerText()).includes('derivative is unbounded'));
      await capture(cancellation, 'domain-endpoint');
      await cancellation.getByRole('button').click(); await capture(cancellation, 'cancellation-path');
      await capture(cancellation.locator('svg'), 'cancellation-chart');
      const chartLabels = await cancellation.locator('svg text').evaluateAll(nodes => nodes.map(node => ({ text: node.textContent, left: node.getBBox().x, right: node.getBBox().x + node.getBBox().width })));
      assert(chartLabels.every(label => label.left >= 0 && label.right <= 440), 'Clipped cancellation label');
      await cancellation.getByText('Inspect the exact reference interval', { exact: true }).click();
      assert.equal(await cancellation.locator('.conditioning-rational').first().innerText(), models.cancellationState().referenceLower);

      const measurement = lab('Measurement sensitivity investigation');
      for (const exponent of [1, 4, 7]) for (const change of [-2, -1, 0, 1, 2]) {
        await slider(measurement, 'Separation exponent', exponent);
        await slider(measurement, 'Second reading change in units of 1/256', change);
        const expected = models.measurementSensitivityState(exponent, change);
        assert.equal(await measurement.locator('circle[data-solution]').getAttribute('data-solution'), expected.solution.join(','));
        states += 1;
      }
      await capture(measurement, 'measurement-sensitive');
      await measurement.getByRole('checkbox').check();
      assert((await measurement.locator('.conditioning-readout').innerText()).includes('no solution'));
      await slider(measurement, 'Second reading change in units of 1/256', 0);
      assert((await measurement.locator('.conditioning-readout').innerText()).includes('infinitely many'));
      await capture(measurement, 'coincident-lines');
      await measurement.getByRole('button').click();

      const backward = lab('Backward error investigation');
      for (const power of [1, 3, 6, 9]) for (const scaled of [false, true]) {
        await slider(backward, 'Small row exponent', power);
        await backward.getByRole('checkbox').setChecked(scaled);
        const expected = models.backwardWitnessState(power, scaled);
        assert((await backward.locator('.conditioning-readout').innerText()).includes(`Normwise η = ${models.formatConditioning(expected.normwise)}`));
        states += 1;
      }
      await capture(backward, 'backward-scaled'); await backward.getByRole('button').click(); await capture(backward, 'backward-zero-entry');

      const sum = lab('Summation investigation');
      for (const preset of Object.keys(models.summationPresets)) {
        await sum.getByRole('combobox').selectOption(preset);
        const expected = models.summationState(preset);
        for (let step = 1; step <= expected.steps.length; step += 1) {
          await slider(sum, 'Inspect addition', step);
          assert((await sum.locator('.conditioning-addition').innerText()).includes(`Exact local sum minus stored subtotal: ${expected.steps[step - 1].lost}`));
          states += 1;
        }
        assert((await sum.locator('.conditioning-readout').innerText()).includes(`Neumaier = ${models.formatConditioning(expected.neumaier, 15)}`));
      }
      await sum.getByRole('combobox').selectOption('reordered'); await capture(sum, 'sum-reordered');
      await sum.getByRole('combobox').selectOption('positive'); await slider(sum, 'Inspect addition', 3); await capture(sum, 'sum-positive');
      assert.equal(await sum.locator('.conditioning-tree > li > span').innerText(), String(models.summationState('positive').balanced.value));
      const tree = sum.locator('.conditioning-tree-scroll'); await tree.focus();
      await page.keyboard.press('ArrowRight');
      await page.waitForTimeout(160);
      const treeGeometry = await tree.evaluate(node => ({ offset: node.scrollLeft, width: node.clientWidth, content: node.scrollWidth }));
      if (treeGeometry.content > treeGeometry.width) assert(treeGeometry.offset > 0, 'Keyboard must scroll the wide tree');
      await tree.evaluate(node => { node.scrollLeft = node.scrollWidth; });
      await capture(tree, 'sum-tree-scrolled');

      const propagation = lab('Error propagation investigation');
      for (const q of [-0.5, 0.5, 1, 1.1]) for (const mode of ['constant', 'alternating', 'pulse']) {
        await propagation.getByRole('combobox', { name: 'Multiplier q' }).selectOption(String(q));
        await propagation.getByRole('combobox', { name: 'Disturbance pattern' }).selectOption(mode);
        await slider(propagation, 'Propagation steps', 24);
        const expected = models.propagationState(q, mode, 24).frames.at(-1);
        assert((await propagation.locator('.conditioning-readout').innerText()).includes(`Final signed error = ${models.formatConditioning(expected.error)}`));
        const dot = propagation.locator('circle[data-step="24"]');
        assert.equal(Number(await dot.getAttribute('data-error')), expected.error);
        const labels = await propagation.locator('svg text').evaluateAll(nodes => nodes.map(node => ({ left: node.getBBox().x, right: node.getBBox().x + node.getBBox().width })));
        assert(labels.every(label => label.left >= 0 && label.right <= 440), 'Clipped propagation label');
        states += 1;
      }
      await capture(propagation, 'propagation-pulse');
      for (const steps of [4, 8, 16, 32]) {
        await propagation.getByRole('combobox', { name: 'Fixed-time refinement' }).selectOption(String(steps));
        assert.equal(Number(await propagation.locator('[data-refinement-error]').getAttribute('data-refinement-error')), models.unstableRefinementState(steps).finalError);
        states += 1;
      }
      await capture(propagation.locator('.conditioning-refinement'), 'unstable-refinement');
      await propagation.getByRole('button').click();

      // Open all teaching depth and answers, then inspect actual questions/code/output.
      await lesson.locator('details').evaluateAll(nodes => nodes.forEach(node => { node.open = true; }));
      assert.equal(await lesson.locator('.python-example').count(), examples.length);
      for (let index = 0; index < examples.length; index += 1) {
        const block = lesson.locator('.python-example').nth(index);
        assert((await block.innerText()).includes(examples[index].title));
        const codes = [await block.innerText()];
        assert(codes.some(code => code.includes(examples[index].code.trim())), 'Actual code mismatch '+examples[index].id);
        assert(codes.some(code => code.includes(examples[index].expected.trim())), 'Actual output mismatch '+examples[index].id);
        assert(await lesson.getByText(examples[index].question, { exact: false }).count());
      }
      assert.equal(await lesson.locator('.lesson-check h3').count(), 10);
      for (const item of await lesson.locator('.lesson-check').all()) assert((await item.innerText()).length > 200);
      assert.equal(await lesson.locator('.katex-error').count(), 0);
      const equations = await lesson.locator('.katex-display').evaluateAll(nodes => nodes.map((node, index) => ({ index, width: node.getBoundingClientRect().width, contentWidth: node.scrollWidth, text: node.textContent })));
      assert(equations.every(item => item.contentWidth <= item.width + 1), 'Equation overflow: ' + JSON.stringify(equations.filter(item => item.contentWidth > item.width + 1)));
      const overflow = await page.evaluate(() => ({ page: document.documentElement.scrollWidth, viewport: innerWidth }));
      assert(overflow.page <= width + 1, `Page overflow at ${width}: ${overflow.page}`);
      await capture(lesson.locator('.python-example').last(), 'complete-report');
      await capture(lesson.locator('.lesson-check h3').last(), 'practice-report');
      await capture(lesson.locator('.lesson-sources'), 'sources');
      const fonts = await page.evaluate(() => ({ sans: document.fonts.check('16px "Space Grotesk"'), mono: document.fonts.check('16px "JetBrains Mono"') }));
      assert.deepEqual(errors, []);
      records.push({ width, states, fonts, errors, overflow, equations, chartLabels, treeGeometry, programs: examples.length, practices: 10 });
      console.log(`Completed ${width}: ${states} operated states`);
      await page.close();
    }
    fs.writeFileSync(path.join(directory, 'results.json'), JSON.stringify({ completedAt: new Date().toISOString(), records }, null, 2));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

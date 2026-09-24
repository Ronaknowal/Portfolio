const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const directory = path.resolve('scratch/sampling-measurement-browser');
fs.mkdirSync(directory, { recursive: true });

(async () => {
  const models = await import('../src/learn/data/sampling-measurement-models.js');
  const { samplingMeasurementExamples: examples } = await import('../src/learn/data/sampling-measurement-examples.js');
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1080 }, reducedMotion: 'reduce' });
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/sampling-measurement-experimental-design', { waitUntil: 'domcontentloaded', timeout: 60000 });
      const lesson = page.locator('.sampling-lesson');
      await lesson.waitFor({ timeout: 60000 });
      await page.evaluate(() => document.fonts.ready);
      assert(await page.evaluate(() => document.fonts.check('16px "Space Grotesk"')));
      const capture = async (locator, name) => {
        await locator.first().evaluate(node => scrollTo({ top: node.getBoundingClientRect().top + scrollY - 84, behavior: 'instant' }));
        await page.screenshot({ path: path.join(directory, `${name}-${width}.png`) });
      };
      const slider = async (label, value) => {
        const control = lesson.getByRole('slider', { name: label, exact: true });
        await control.fill(String(value));
        await control.dispatchEvent('input');
      };
      const lab = name => lesson.locator(`.sampling-lab[aria-label="${name}"]`);
      assert.equal(await lesson.locator('h2').count(), 10);
      assert.equal(await lesson.locator('.sampling-lab').count(), 5);
      assert.equal(await lesson.locator('.sampling-figure').count(), 5);
      assert.deepEqual(await lesson.locator('nav a').evaluateAll(nodes => nodes.map(node => node.hash).filter(hash => !document.getElementById(hash.slice(1)))), []);
      for (let i = 0; i < 10; i += 1) await capture(lesson.locator('h2').nth(i), `reading-${i + 1}`);
      for (let i = 0; i < 5; i += 1) await capture(lesson.locator('.sampling-figure').nth(i), `inline-${i + 1}`);
      let states = 0;
      const finite = lab('Finite sampling investigation');
      for (const frame of ['complete', 'partial']) {
        await finite.getByRole('combobox').selectOption(frame);
        for (let n = 1; n <= (frame === 'complete' ? 8 : 4); n += 1) {
          await slider('Sample size', n);
          const result = models.finiteSampleState(undefined, frame === 'complete' ? null : [0, 1, 2, 3], n);
          for (const index of [0, result.samples.length - 1]) {
            await slider('Selected subset', index + 1);
            const text = await finite.locator('.sampling-readout').innerText();
            assert(text.includes(`Selected mean = ${models.formatSampling(result.samples[index].mean)}`));
            assert(text.includes(`Variance = ${models.formatSampling(result.formulaVariance)}`));
            const probabilities = await finite.locator('rect[data-probability]').evaluateAll(nodes => nodes.map(node => Number(node.dataset.probability)));
            assert(Math.abs(probabilities.reduce((a, b) => a + b, 0) - 1) < 1e-10);
            states += 1;
          }
        }
      }
      assert((await finite.locator('.sampling-readout').innerText()).includes('Variance = 0; bias = -4; MSE = 16'));
      await capture(finite, 'incomplete-census');
      await capture(finite.locator('svg'), 'incomplete-distribution');
      await finite.getByRole('button', { name: 'Reset sampling' }).focus(); await page.keyboard.press('Enter');
      await finite.getByRole('slider', { name: 'Sample size', exact: true }).focus(); await page.keyboard.press('ArrowRight');
      assert.equal(await finite.getByRole('slider', { name: 'Sample size', exact: true }).inputValue(), '3');

      const inclusion = lab('Inclusion weighting investigation');
      for (const mode of ['unequal', 'equal', 'uncovered']) {
        await inclusion.getByRole('combobox', { name: 'Selection design' }).selectOption(mode);
        const result = models.inclusionDesignState(mode);
        for (let index = 0; index < result.samples.length; index += 1) {
          if (result.samples[index].probability === 0) continue;
          await inclusion.getByRole('combobox', { name: 'Inspect subset' }).selectOption(String(index));
          if (result.covered) assert((await inclusion.locator('.sampling-readout').innerText()).includes(`sum / fixed N = ${models.formatSampling(result.samples[index].ht)}`));
          else assert((await inclusion.getByRole('status').innerText()).includes('zero inclusion probability'));
          states += 1;
        }
      }
      await capture(inclusion, 'uncovered-weights');
      await inclusion.getByRole('button', { name: 'Reset weights' }).click();
      await inclusion.getByRole('combobox', { name: 'Inspect subset' }).selectOption('5');
      await capture(inclusion, 'weighted-contributions');
      await inclusion.getByText('Inspect all six sample probabilities', { exact: true }).click();
      assert.equal(await inclusion.locator('details table tbody tr').count(), 6);

      const reading = lab('Independent units and repeated readings investigation');
      for (const [G, m] of [[1, 16], [4, 4], [16, 1], [1, 1], [16, 16]]) {
        await slider('Independent units G', G); await slider('Readings per unit m', m);
        for (const [u, e, b] of [[4, 1, 0], [0, 1, 2], [9, 0, -3], [0, 0, 0]]) {
          await slider('Unit variance', u); await slider('Reading variance', e); await slider('Fixed offset b', b);
          const result = models.groupedMeasurementState(G, m, u, e, b);
          const text = await reading.locator('.sampling-readout').innerText();
          assert(text.includes(`Variance = ${models.formatSampling(result.variance)}`));
          assert(text.includes(`MSE = variance + b² = ${models.formatSampling(result.mse)}`));
          assert.equal(await reading.locator('.reading-groups i').count(), G * m);
          const widths = await reading.locator('svg rect').evaluateAll(nodes => nodes.map(node => Number(node.getAttribute('width'))));
          assert(Math.abs(widths[0] - 200 * result.unitComponent / 9) < 1e-10);
          assert(Math.abs(widths[1] - 200 * result.readingComponent / 9) < 1e-10);
          states += 1;
        }
      }
      await capture(reading.locator('.sampling-readout'), 'zero-variance');
      await reading.getByRole('button', { name: 'Reset readings' }).click();
      await capture(reading, 'grouped-readings'); await capture(reading.locator('svg'), 'variance-decomposition');

      const assignment = lab('Random assignment investigation');
      for (const design of ['complete', 'prognostic', 'mixed']) for (const effect of ['constant', 'heterogeneous']) {
        await assignment.getByRole('combobox', { name: 'Assignment rule' }).selectOption(design);
        await assignment.getByRole('combobox', { name: 'Individual effects' }).selectOption(effect);
        const result = models.assignmentState(design, effect);
        for (let index = 0; index < result.states.length; index += 1) {
          await slider('Allocation number', index + 1);
          const text = await assignment.locator('.sampling-readout').innerText();
          assert(text.includes(`Realized contrast = ${models.formatSampling(result.states[index].difference)}`));
          assert(text.includes(`Exact assignment variance = ${models.formatSampling(result.variance)}`));
          assert.equal(await assignment.getByRole('cell', { name: 'unobserved', exact: true }).count(), 6);
          states += 1;
        }
      }
      await capture(assignment, 'unobserved-outcomes');
      await assignment.getByRole('checkbox').focus(); await page.keyboard.press('Space');
      assert.equal(await assignment.getByRole('cell', { name: 'unobserved', exact: true }).count(), 0);
      await capture(assignment.locator('svg'), 'mixed-pair-distribution');
      await assignment.getByRole('combobox', { name: 'Assignment rule' }).selectOption('prognostic');
      await assignment.getByRole('combobox', { name: 'Individual effects' }).selectOption('constant');
      await capture(assignment, 'blocked-table'); await capture(assignment.locator('svg'), 'blocked-distribution');
      await assignment.getByRole('button', { name: 'Reset assignment' }).click();
      assert.equal(await assignment.getByRole('checkbox').isChecked(), false);

      const factorial = lab('Factorial interaction investigation');
      assert.equal(await factorial.locator('.fourth-revealed').count(), 0);
      await factorial.getByRole('button', { name: 'Reveal fourth cell', exact: true }).focus(); await page.keyboard.press('Enter');
      for (const interaction of [-6, -2, 0, 4, 6]) for (const share of [0, .25, .5, 1]) {
        await slider('Interaction contrast', interaction); await slider('Share with B at one', share);
        assert((await factorial.locator('.sampling-readout').innerText()).includes(`declared B mixture: ${models.formatSampling(models.factorialState(interaction, share).averageAEffect)}`));
        states += 1;
      }
      await slider('Interaction contrast', -2); await slider('Share with B at one', .75);
      await capture(factorial, 'factorial-changed');
      await factorial.getByRole('button', { name: 'Reset factorial' }).click();
      assert.equal(await factorial.locator('.fourth-revealed').count(), 0);

      const checkpoints = lesson.locator('div.lesson-check');
      assert.equal(await checkpoints.count(), 2);
      for (let i = 0; i < 2; i += 1) {
        assert((await checkpoints.nth(i).locator('p').first().innerText()).length > 80);
        await checkpoints.nth(i).getByText('Show explanation', { exact: true }).click();
        assert((await checkpoints.nth(i).locator('details div').innerText()).length > 140);
      }
      const practices = lesson.locator('section.lesson-check');
      assert.equal(await practices.count(), 11);
      for (let i = 0; i < 11; i += 1) {
        await practices.nth(i).getByText('Hint', { exact: true }).click();
        await practices.nth(i).getByText('Explained solution', { exact: true }).click();
        assert((await practices.nth(i).locator('details').last().innerText()).length > 180);
      }
      await capture(practices.nth(2), 'practice-weighting');
      await capture(practices.nth(9), 'practice-attrition');
      await capture(practices.nth(10), 'practice-protocol');
      await lesson.getByText('Deeper: the finite randomization variance and what remains unobserved', { exact: true }).click();
      for (const example of examples) {
        const container = lesson.locator('.python-example').filter({ has: page.getByRole('heading', { name: example.title, exact: true }) });
        const text = (await container.innerText()).replace(/\r\n/g, '\n');
        assert(text.includes(example.code.trim()), 'Actual code ' + example.id);
        assert(text.includes(example.expected), 'Actual output ' + example.id);
        assert((await lesson.innerText()).includes(example.question), 'Visible question ' + example.id);
      }
      assert.equal(await lesson.locator('.python-example').count(), 12);
      await capture(lesson.locator('.python-example').nth(2), 'native-inclusion');
      await capture(lesson.locator('.python-example').last(), 'native-protocol');
      await capture(lesson.locator('.lesson-sources'), 'sources');
      const scrollCode = lesson.locator('.python-example').nth(2).locator('div[style*="white-space: pre"]').first();
      await scrollCode.focus(); await page.keyboard.press('ArrowRight'); await page.waitForTimeout(150);
      const codeScroll = await scrollCode.evaluate(node => ({ width: node.clientWidth, content: node.scrollWidth, left: node.scrollLeft, focused: node === document.activeElement }));
      if (codeScroll.content > codeScroll.width) assert(codeScroll.focused && codeScroll.left > 0);
      const math = await lesson.locator('.katex-display').evaluateAll(nodes => nodes.map((node, index) => ({ index, width: node.clientWidth, content: node.scrollWidth })));
      const svgText = await lesson.locator('svg text').evaluateAll(nodes => nodes.map(node => { const box = node.getBBox(); const view = node.ownerSVGElement.viewBox.baseVal; return { text: node.textContent, x: box.x, right: box.x + box.width, bottom: box.y + box.height, width: view.width, height: view.height }; }));
      const overflow = await page.evaluate(() => ({ document: document.documentElement.scrollWidth, viewport: innerWidth }));
      const record = { width, fonts: true, states, sections: 10, labs: 5, figures: 5, programs: 12, checkpoints: 2, practices: 11, keyboard: true, math, svgText, codeScroll, overflow, errors };
      records.push(record);
      fs.writeFileSync(path.join(directory, 'progress.json'), JSON.stringify({ at: new Date().toISOString(), records }, null, 2));
      assert.deepEqual(math.filter(item => item.content > item.width + 2), [], 'Displayed equations fit');
      assert.deepEqual(svgText.filter(item => item.x < -1 || item.right > item.width + 1 || item.bottom > item.height + 1), [], 'SVG labels fit');
      assert.equal(await lesson.locator('.katex-error').count(), 0);
      assert(overflow.document <= width + 1);
      assert.deepEqual(errors, []);
      await page.close();
    }
  } finally { await browser.close(); }
  const result = { at: new Date().toISOString(), passed: true, records };
  fs.writeFileSync(path.join(directory, 'results.json'), JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result));
})().catch(error => { console.error(error); process.exit(1); });

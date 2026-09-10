const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const directory = path.resolve('scratch/randomized-algorithms-browser');
fs.mkdirSync(directory, { recursive: true });
const normalize = value => value.replace(/\s+/g, ' ').trim();

(async () => {
  const models = await import(pathToFileURL(path.resolve('src/learn/data/randomized-algorithm-models.js')));
  const { randomizedAlgorithmExamples: examples } = await import(pathToFileURL(path.resolve('src/learn/data/randomized-algorithm-examples.js')));
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [], errors = [];
  try {
    for (const width of [1440, 390]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      page.on('pageerror', error => errors.push(`${width}: ${error.message}`));
      await page.goto((process.env.LEARNING_BASE_URL || 'http://127.0.0.1:5173') + '/learn/path/full-curriculum/randomized-algorithms-sampling-error-guarantees?module=data-structures-algorithms');
      const lesson = page.locator('.randomized-algorithms-lesson');
      await lesson.waitFor();
      const record = { width, anchors: [], mappingCases: 0, shuffleSteps: 0, reservoirSteps: 0, selectionSteps: 0, probeCases: 0, budgetCases: 0, figures: [], examples: [] };
      const keyButton = async (lab, name, key = 'Enter') => { await lab.getByRole('button', { name, exact: true }).focus(); await page.keyboard.press(key); };
      const capture = async (locator, name) => {
        const style = await page.addStyleTag({ content: '.learn-nav { visibility: hidden !important; }' });
        await locator.screenshot({ path: path.join(directory, `${name}-${width}.png`) });
        await style.evaluate(element => element.remove());
      };
      const anchors = lesson.locator('nav[aria-label="In this lesson"] a');
      for (let index = 0; index < await anchors.count(); index++) {
        const href = await anchors.nth(index).getAttribute('href');
        await anchors.nth(index).focus();
        await page.keyboard.press('Enter');
        await page.waitForFunction(hash => location.hash === hash, href);
        await page.waitForTimeout(100);
        const box = await page.locator(`[id="${href.slice(1)}"]`).boundingBox();
        assert(box.y >= 50 && box.y < 200, `anchor ${href} arrived at ${box.y}`);
        record.anchors.push({ href, top: box.y });
        await page.screenshot({ path: path.join(directory, `reading-${index + 1}-${width}.png`) });
      }
      const mapping = lesson.locator('[data-lab="uniform-choice"]');
      for (const target of [2, 3, 4, 5, 6, 7]) for (const mode of ['modulo', 'reject']) {
        await mapping.getByLabel('Number of outputs').selectOption(String(target));
        await mapping.getByLabel('Mapping', { exact: true }).selectOption(mode);
        const model = models.rejectionMap(8, target, mode === 'reject');
        const counts = await mapping.locator('.random-bars strong').allTextContents();
        assert.deepEqual(counts, model.counts.map(count => `${count}/${model.accepted}`));
        record.mappingCases++;
      }
      await mapping.getByLabel('Number of outputs').selectOption('3');
      await mapping.getByLabel('Mapping', { exact: true }).selectOption('modulo');
      await capture(mapping, 'modulo-bias');
      await keyButton(mapping, 'Reset', 'Space');
      assert.equal(await mapping.getByLabel('Mapping', { exact: true }).inputValue(), 'reject');
      await capture(mapping, 'unbiased-rejection');

      const shuffle = lesson.locator('[data-lab="shuffle"]');
      for (const choices of [[0, 0, 0], [3, 2, 1], [1, 1, 0]]) {
        await keyButton(shuffle, 'Reset');
        for (let step = 0; step < choices.length; step++) {
          await shuffle.getByLabel('Chosen active index').selectOption(String(choices[step]));
          await keyButton(shuffle, 'Swap and fix');
          const state = models.shuffleTrace(choices.slice(0, step + 1)).at(-1);
          assert.deepEqual(await shuffle.locator('.random-cells strong').allTextContents(), state.values);
          record.shuffleSteps++;
        }
        assert(await shuffle.getByRole('button', { name: 'Swap and fix', exact: true }).isDisabled());
      }
      await keyButton(shuffle, 'Previous');
      assert(!(await shuffle.getByRole('button', { name: 'Swap and fix', exact: true }).isDisabled()));
      await capture(shuffle, 'shuffle-fixed-suffix');
      await keyButton(shuffle, 'Reset');

      const reservoir = lesson.locator('[data-lab="reservoir"]');
      for (const capacity of [1, 2, 3]) for (const policy of ['replace', 'discard']) {
        await reservoir.getByLabel('Reservoir size k').selectOption(String(capacity));
        await keyButton(reservoir, 'Reset');
        const choices = [];
        for (let seen = capacity; seen < 6; seen++) {
          choices.push(policy === 'replace' ? 0 : seen);
          await reservoir.getByLabel('Next draw j').selectOption(String(choices.at(-1)));
          await keyButton(reservoir, 'Process next record');
          const state = models.reservoirTrace(capacity, choices).at(-1);
          assert.deepEqual(await reservoir.locator('.random-cells strong').allTextContents(), state.sample.map(index => 'ABCDEFGH'[index]));
          record.reservoirSteps++;
        }
        assert(await reservoir.getByRole('button', { name: 'Process next record', exact: true }).isDisabled());
        await reservoir.locator('summary').click();
        assert.equal(await reservoir.locator('.random-subsets > span').count(), models.reservoirDistribution(6, capacity).length);
        await reservoir.locator('summary').click();
      }
      await reservoir.getByLabel('Reservoir size k').selectOption('2');
      await reservoir.getByLabel('Next draw j').selectOption('1');
      await keyButton(reservoir, 'Process next record');
      await reservoir.locator('summary').click();
      await capture(reservoir, 'reservoir-joint-distribution');
      await keyButton(reservoir, 'Previous');
      assert((await reservoir.locator('[data-result="reservoir"]').innerText()).includes('Initial fill'));

      const selection = lesson.locator('[data-lab="random-selection"]');
      const values = [8, 1, 6, 3, 9, 2, 7, 4, 5];
      for (const rank of [0, 4, 8]) for (const policy of ['minimum', 'maximum']) {
        await selection.getByLabel('Wanted sorted rank (zero-based)').selectOption(String(rank));
        await keyButton(selection, 'Reset');
        const choices = [];
        let state = models.quickselectTrace(values, rank, choices).at(-1);
        while (state.result === null) {
          const pivot = policy === 'minimum' ? Math.min(...state.active) : Math.max(...state.active);
          choices.push(state.active.indexOf(pivot));
          await selection.getByLabel('Pivot occurrence').selectOption(String(choices.at(-1)));
          await keyButton(selection, 'Partition');
          state = models.quickselectTrace(values, rank, choices).at(-1);
          const result = await selection.locator('[data-result="selection"]').innerText();
          assert(result.includes(`Scanned elements: ${state.work}.`), result);
          if (state.result !== null) assert(result.includes(`Exact result ${state.result}.`), result);
          record.selectionSteps++;
        }
        assert(await selection.getByRole('button', { name: 'Partition', exact: true }).isDisabled());
      }
      await keyButton(selection, 'Previous');
      await selection.getByLabel('Wanted sorted rank (zero-based)').selectOption('4');
      await keyButton(selection, 'Reset');
      await selection.getByLabel('Pivot occurrence').selectOption('2');
      await keyButton(selection, 'Partition');
      assert((await selection.locator('[data-result="selection"]').innerText()).includes('Retained 5 values; local rank 4.'));
      await capture(selection, 'selection-surviving-partition');
      await keyButton(selection, 'Reset');

      const verification = lesson.locator('[data-lab="random-verification"]');
      for (const fixture of ['correct', 'cancellation', 'even']) {
        await verification.getByLabel('Claimed product').selectOption(fixture);
        for (const bits of [[0, 0], [0, 1], [1, 0], [1, 1]]) {
          await verification.getByLabel('Probe coordinate r0').selectOption(String(bits[0]));
          await verification.getByLabel('Probe coordinate r1').selectOption(String(bits[1]));
          const state = models.productProbe(fixture, bits);
          const result = await verification.locator('[data-result="verification"]').innerText();
          assert(result.startsWith(state.passes ? 'Probe passes' : 'Mismatch found'));
          const pathText = await verification.locator('.random-probe-path').innerText();
          assert(pathText.includes(`residual = [${state.residual.join(', ')}]`));
          record.probeCases++;
        }
      }
      await keyButton(verification, 'Reset');
      assert((await verification.locator('[data-result="amplification"]').innerText()).includes('12.50%'));
      await capture(verification, 'verification-cancellation');
      await verification.getByLabel('Randomness across tests').selectOption('same');
      assert((await verification.locator('[data-result="amplification"]').innerText()).includes('3 tests false acceptance 50.00%'));
      await verification.getByRole('button', { name: '[1,0] detects error', exact: true }).focus();
      await page.keyboard.press('Enter');
      assert((await verification.locator('[data-result="verification"]').innerText()).startsWith('Mismatch found'));
      await capture(verification, 'verification-independent-versus-reused');
      await keyButton(verification, 'Reset');

      const budget = lesson.locator('[data-lab="sample-budget"]');
      for (const epsilon of [0.2, 0.1, 0.05, 0.02]) for (const delta of [0.1, 0.05, 0.01, 0.001]) {
        await budget.getByLabel('Absolute tolerance ε').selectOption(String(epsilon));
        await budget.getByLabel('Failure budget δ').selectOption(String(delta));
        assert((await budget.locator('[data-result="budget"]').innerText()).includes(`Sufficient budget: ${models.sampleBudget(epsilon, delta)}.`));
        record.budgetCases++;
      }
      await keyButton(budget, 'Reset', 'Space');
      await capture(budget, 'sample-budget');

      const figures = lesson.locator('[data-figure]');
      for (let index = 0; index < await figures.count(); index++) {
        const figure = figures.nth(index);
        const name = await figure.getAttribute('data-figure');
        assert.equal(await figure.evaluate(element => element.scrollWidth > element.clientWidth + 1), false, `${name} overflow`);
        await capture(figure, name);
        record.figures.push(name);
      }
      const codeBlocks = lesson.locator('.python-example');
      const entries = Object.entries(examples);
      assert.equal(await codeBlocks.count(), entries.length);
      for (let index = 0; index < entries.length; index++) {
        const title = await codeBlocks.nth(index).getByRole('heading').innerText();
        const [name, example] = entries.find(([, fixture]) => fixture.title === title);
        const displayed = await codeBlocks.nth(index).locator(':scope > div').evaluateAll(elements => elements.map(element => [...element.childNodes].filter(child => child.nodeType === Node.TEXT_NODE).map(child => child.textContent).join('')));
        assert.deepEqual(displayed.map(normalize), [normalize(example.code), normalize(example.expected)], name);
        record.examples.push(name);
      }
      const practice = lesson.locator('.dsa-practice');
      assert.equal(await practice.locator('a[href^="https://leetcode.com/problems/"]').count(), 6);
      assert.equal(await practice.locator('details[open]').count(), 0);
      assert.equal(await lesson.locator('.random-exercise').count(), 7);
      assert.equal(await lesson.locator('.random-exercise details[open]').count(), 0);
      const overflow = await lesson.locator('.random-lab').evaluateAll(elements => elements.filter(element => element.scrollWidth > element.clientWidth + 1).map(element => element.dataset.lab));
      assert.deepEqual(overflow, []);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
      await capture(lesson.locator('.lesson-intro'), 'intro');
      await capture(lesson.locator('.lesson-sources'), 'sources');
      results.push(record);
      await page.close();
    }
    assert.deepEqual(errors, []);
    const result = { checkedAt: new Date().toISOString(), results, errors };
    fs.writeFileSync(path.join(directory, 'results.json'), JSON.stringify(result, null, 2));
    console.log(JSON.stringify(result, null, 2));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

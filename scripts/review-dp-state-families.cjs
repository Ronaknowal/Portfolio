const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const directory = path.resolve(__dirname, '../scratch/dp-state-families-browser');
fs.mkdirSync(directory, { recursive: true });

(async () => {
  const { rewardTrace, gridPlan, defaultGrid, lcsPlan, capacityTrace, subsetRoutePlan } = await import('../src/learn/data/dynamic-programming-models.js');
  const { dynamicProgrammingExamples } = await import('../src/learn/data/dynamic-programming-examples.js');
  const base = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:5173';
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  const errors = [];
  const failedRequests = [];
  const hashes = require('./dp-state-families-source-hashes.cjs');
  const sourceHashes = hashes();
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      // Freeze this test page against unrelated authors' Vite HMR state resets.
      await page.routeWebSocket('**', socket => socket.close());
      page.on('pageerror', error => errors.push(error.message));
      page.on('requestfailed', request => failedRequests.push(request.url()));
      page.on('console', message => { if (message.type() === 'error' && !message.text().startsWith('[vite]')) errors.push(message.text()); });
      await page.goto(`${base}/learn/path/full-curriculum/dynamic-programming-states-transitions-optimization?module=data-structures-algorithms`);
      const lesson = page.locator('.dynamic-programming-lesson');
      await lesson.waitFor();
      await page.evaluate(() => document.fonts.ready);
      assert(await page.evaluate(() => document.fonts.check('16px "Space Grotesk"')));
      assert.equal(await page.locator('vite-error-overlay').count(), 0);
      const rewards = page.getByRole('region', { name: 'Reward dependency investigation', exact: true });
      const grid = page.getByRole('region', { name: 'Grid dependency investigation', exact: true });
      const sequences = page.getByRole('region', { name: 'Sequence alignment investigation', exact: true });
      const capacity = page.getByRole('region', { name: 'Capacity generation investigation', exact: true });
      const subset = page.getByRole('region', { name: 'Subset endpoint investigation', exact: true });
      async function capture(locator, name) {
        await locator.scrollIntoViewIfNeeded();
        const style = await page.addStyleTag({ content: '.learn-nav { visibility: hidden !important; }' });
        await locator.screenshot({ path: path.join(directory, `${name}-${width}.png`) });
        await style.evaluate(node => node.remove());
      }
      await capture(lesson.locator('.lesson-intro'), 'route');
      let rewardStates = 0;
      for (const values of [[4, 7, 2, 9], [], [-4, -2], [1, 1, 1, 1, 1, 1, 1]]) {
        await rewards.getByLabel('Session rewards', { exact: true }).fill(values.join(','));
        await rewards.getByRole('button', { name: 'Apply rewards' }).click();
        for (const method of ['memo', 'table']) {
          await rewards.getByLabel('Evaluation order').selectOption(method);
          // Changing to the same option leaves the current step; reapply explicitly.
          await rewards.getByRole('button', { name: 'Apply rewards' }).click();
          const model = rewardTrace(values, method);
          for (let step = 0; step < model.frames.length; step += 1) {
            assert.equal(await rewards.locator('[data-dp-status]').innerText(), model.frames[step].message);
            if (step + 1 < model.frames.length) await rewards.getByRole('button', { name: 'Next state', exact: true }).click();
            rewardStates += 1;
          }
          assert(await rewards.getByRole('button', { name: 'Next state', exact: true }).isDisabled());
        }
      }
      await rewards.getByLabel('Session rewards', { exact: true }).fill('3,NaN');
      await rewards.getByRole('button', { name: 'Apply rewards' }).click();
      assert(await rewards.getByRole('alert').isVisible());
      await rewards.getByRole('button', { name: 'Reset reward lab' }).click();
      await rewards.getByRole('button', { name: 'Next state', exact: true }).focus();
      await page.keyboard.press('Enter');
      assert.match(await rewards.locator('[data-dp-status]').innerText(), /index|beginning/);
      await capture(rewards, 'reward-requests');
      await rewards.getByRole('button', { name: 'Finish states' }).click();
      await capture(rewards, 'reward-complete');
      let gridStates = 0;
      for (const blocked of [[], ['0,1'], ['0,1', '1,0'], ['0,0']]) {
        await grid.getByRole('button', { name: 'Reset grid lab' }).click();
        for (const cell of blocked) await grid.getByRole('button', { name: new RegExp(`^Cell ${cell},`) }).click();
        const model = gridPlan(defaultGrid, blocked);
        for (let step = 0; step < model.steps.length; step += 1) {
          const state = model.steps[step];
          const current = grid.locator('.dp-grid .dp-selected');
          assert.match(await current.getAttribute('aria-label'), new RegExp(`Cell ${state.row},${state.column},`));
          assert.equal(await current.locator('strong').innerText(), String(state.costs[state.row][state.column] ?? '∞'));
          if (step + 1 < model.steps.length) await grid.getByRole('button', { name: 'Next cell', exact: true }).click();
          gridStates += 1;
        }
        const traceButton = grid.getByRole('button', { name: 'Trace cheapest witness' });
        assert.equal(await traceButton.isDisabled(), model.result === null);
        if (model.result !== null) {
          await traceButton.click();
          assert.equal(await grid.locator('.dp-grid .dp-chosen').count(), model.path.length);
          if (blocked.length === 0) await capture(grid, 'grid-witness');
        }
      }
      await capture(grid, 'grid-unreachable');
      await grid.getByRole('button', { name: 'Reset grid lab' }).click();
      await grid.getByRole('button', { name: /^Cell 0,1,/ }).focus();
      await page.keyboard.press('Space');
      assert.equal(await grid.getByRole('button', { name: /^Cell 0,1,/ }).getAttribute('aria-pressed'), 'true');
      let sequenceStates = 0;
      for (const [first, second] of [['CABAC', 'ABC'], ['AB', 'BA'], ['', ''], ['AAAAAA', 'AAA'], ['ABCDEF', 'FEDCBA']]) {
        await sequences.getByLabel('First sequence', { exact: true }).fill(first);
        await sequences.getByLabel('Second sequence', { exact: true }).fill(second);
        await sequences.getByRole('button', { name: 'Apply sequences' }).click();
        const model = lcsPlan(first, second);
        for (let row = 0; row <= first.length; row += 1) {
          for (let column = 0; column <= second.length; column += 1) {
            await sequences.getByRole('button', { name: `LCS prefix ${row},${column} length ${model.lengths[row][column]}`, exact: true }).click();
            assert.match(await sequences.locator('[data-dp-status]').innerText(), new RegExp(`L\\(${row},${column}\\)`));
            sequenceStates += 1;
          }
        }
        await sequences.getByRole('button', { name: 'Start witness trace' }).click();
        for (let step = 0; step < model.trace.length; step += 1) {
          assert.equal(await sequences.locator('.dp-alignment .dp-chosen').count(), model.trace[step].pairs.length * 2);
          if (step + 1 < model.trace.length) await sequences.getByRole('button', { name: 'Next choice', exact: true }).click();
        }
        assert.match(await sequences.innerText(), new RegExp(`length ${model.result}`));
        if (first === 'CABAC') await capture(sequences, 'alignment-witness');
        if (first === 'ABCDEF') await capture(sequences, 'alignment-wide');
      }
      await sequences.getByLabel('First sequence', { exact: true }).fill('A😀');
      await sequences.getByRole('button', { name: 'Apply sequences' }).click();
      assert(await sequences.getByRole('alert').isVisible());
      await sequences.getByRole('button', { name: 'Reset sequence lab' }).click();
      await capture(sequences, 'alignment-dependencies');
      let capacityStates = 0;
      for (const budget of [0, 2, 4, 6, 8]) {
        for (const direction of ['descending', 'ascending']) {
          await capacity.getByRole('button', { name: 'Reset capacity lab' }).click();
          await capacity.getByLabel('Capacity', { exact: true }).selectOption(String(budget));
          await capacity.getByLabel('Capacity order', { exact: true }).selectOption(direction);
          const model = capacityTrace(undefined, budget, direction);
          for (let step = 0; step < model.frames.length; step += 1) {
            assert.equal(await capacity.locator('[data-dp-status]').innerText(), model.frames[step].message);
            assert.deepEqual(await capacity.locator('.dp-capacities strong').allTextContents(), model.frames[step].best.map(String));
            if (step + 1 < model.frames.length) await capacity.getByRole('button', { name: 'Next update', exact: true }).click();
            capacityStates += 1;
          }
          if (budget === 6) await capture(capacity, `capacity-${direction}`);
        }
      }
      let subsetStates = 0;
      const routeModel = subsetRoutePlan();
      for (const mask of [7, 15, 0, 1, 5, 14]) {
        await subset.getByRole('button', { name: 'Reset subset lab' }).click();
        for (let bit = 0; bit < 4; bit += 1) if ((mask & (1 << bit)) !== (7 & (1 << bit))) await subset.getByRole('button', { name: new RegExp(`membership bit ${bit}$`) }).click();
        for (let endpoint = 0; endpoint < 4; endpoint += 1) {
          await subset.getByLabel('Endpoint', { exact: true }).selectOption(String(endpoint));
          const status = await subset.locator('[data-dp-status]').innerText();
          const cost = routeModel.best[mask][endpoint];
          assert(cost === null ? status.includes('unreachable') : status.includes(`= ${cost}.`));
          subsetStates += 1;
        }
      }
      await subset.getByRole('button', { name: 'Reset subset lab' }).click();
      await capture(subset, 'subset-endpoint');
      await subset.getByRole('button', { name: 'Toggle B membership bit 1' }).focus();
      await page.keyboard.press('Space');
      assert.equal(await subset.getByRole('button', { name: 'Toggle B membership bit 1' }).getAttribute('aria-pressed'), 'false');
      for (const [index, figure] of (await lesson.locator('.dp-inline').all()).entries()) await capture(figure, `inline-${index}`);

      const families = await require('./review-dp-state-families-controls.cjs')(page, lesson, width, directory);
      // The independent solution is hidden until explicitly opened.
      const commandAnswer = lesson.locator('.lesson-check').filter({ hasText: 'Explain a recurrence and one full implementation' }).locator('summary');
      await commandAnswer.click();
      const examples = lesson.locator('.python-example');
      const { dpStateFamiliesExamples } = await import('../src/learn/data/dp-state-families-examples.js');
      const allExamples = {...dynamicProgrammingExamples,...Object.fromEntries(Object.entries(dpStateFamiliesExamples).map(([key,value])=>['family-'+key,value]))};
      assert.equal(await examples.count(), Object.keys(allExamples).length);
      for (const example of Object.values(allExamples)) {
        const block = examples.filter({ has: page.getByRole('heading', { name: example.title, exact: true }) });
        assert.equal(await block.count(), 1);
        const text = (await block.innerText()).replace(/\r\n/g, '\n');
        assert(text.includes(example.expected), example.title);
        const blocks = await block.evaluate(node=>[...node.children].filter(child=>getComputedStyle(child).whiteSpace==='pre').map(child=>[...child.childNodes].filter(node=>node.nodeType===Node.TEXT_NODE).map(node=>node.textContent).join('')));
        assert.equal(blocks[0].trim(),example.code.trim());
        assert.equal(blocks[1].trim(),example.expected.trim());
      }
      const practice = lesson.locator('.dsa-practice');
      assert.equal(await practice.locator('a[href^="https://leetcode.com/problems/"]').count(), 15);
      assert.equal(await practice.locator('details[open]').count(), 0);
      await practice.locator('summary').first().focus();
      await page.keyboard.press('Enter');
      assert.equal(await practice.locator('details[open]').count(), 1);
      await capture(practice.locator('.dsa-practice__stage').first(), 'practice');
      const anchors = await lesson.locator('nav[aria-label="In this lesson"] a').evaluateAll(nodes => nodes.map(node => node.getAttribute('href')));
      for (const anchor of anchors) assert.equal(await lesson.locator(`[id="${anchor.slice(1)}"]`).count(), 1, `missing ${anchor}`);
      for (const lab of [rewards, grid, sequences, capacity, subset]) {
        assert.equal(await lab.evaluate(node => node.scrollWidth > node.clientWidth + 1), false, 'lab overflow');
      }
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false, 'page overflow');
      results.push({ width, rewardStates, gridStates, sequenceStates, capacityStates, subsetStates, families, examples: Object.keys(allExamples).length, practice: 15, anchors: anchors.length, overflow: false });
      await page.close();
    }
    assert.deepEqual(errors, []);
    assert.deepEqual(failedRequests, []);
    assert.deepEqual(hashes(),sourceHashes);
    const result = { checkedAt: new Date().toISOString(), base, sourceHashes, results, errors, failedRequests };
    fs.writeFileSync(path.join(directory, 'results.json'), JSON.stringify(result, null, 2));
    fs.writeFileSync('docs/teaching/evidence/dp-state-families-browser.json', JSON.stringify(result, null, 2));
    console.log(JSON.stringify(result, null, 2));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

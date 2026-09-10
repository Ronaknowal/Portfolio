const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

const directory = path.resolve('scratch/algorithm-correctness-browser');
fs.mkdirSync(directory, { recursive: true });
const base = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:5173';
const route = '/learn/path/full-curriculum/algorithm-correctness-loop-invariants-termination?module=data-structures-algorithms';
const normalize = text => text.replace(/\s+/g, ' ').trim();

(async () => {
  const models = await import(pathToFileURL(path.resolve('src/learn/data/algorithm-correctness-models.js')));
  const { algorithmCorrectnessExamples: examples } = await import(pathToFileURL(path.resolve('src/learn/data/algorithm-correctness-examples.js')));
  const { default: practice } = await import(pathToFileURL(path.resolve('src/learn/data/practice/algorithm-correctness-loop-invariants-termination.js')));
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  const errors = [];
  try {
    for (const width of [1440, 390]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      page.on('pageerror', error => errors.push(`${width}: ${error.message}`));
      await page.routeWebSocket('**', socket => socket.close());
      await page.goto(base + route);
      await page.locator('[data-lab="search-invariant"]').waitFor();
      const record = { width, searchStates: 0, compactionStates: 0, partitionStates: 0, euclidStates: 0 };
      async function capture(locator, name) {
        const style = await page.addStyleTag({ content: '.learn-nav { visibility: hidden !important; }' });
        await locator.screenshot({ path: path.join(directory, `${name}-${width}.png`) });
        await style.evaluate(element => element.remove());
      }
      async function advance(lab) {
        await lab.getByRole('button', { name: 'Next state', exact: true }).focus();
        await page.keyboard.press('Enter');
      }
      async function reset(lab) {
        await lab.getByRole('button', { name: 'Reset investigation' }).focus();
        await page.keyboard.press('Space');
      }
      const search = page.locator('[data-lab="search-invariant"]');
      async function searchCase(values, target, skip, screenshot) {
        await search.getByLabel('Search values', { exact: true }).fill(values.join(','));
        await search.getByLabel('Target', { exact: true }).fill(String(target));
        await search.getByRole('button', { name: 'Apply search input' }).click();
        await search.getByRole('checkbox').setChecked(skip);
        const trace = models.searchProofTrace(values, target, skip);
        for (let index = 0; index < trace.length; index++) {
          const state = trace[index];
          assert.equal(await search.locator('[data-result="search-action"]').innerText(), state.action);
          assert.match(await search.locator('.proof-state-label').innerText(), new RegExp(`i = ${state.index} · n − i = ${state.variant}`));
          if (state.finished) assert.match(await search.locator('[data-result="search-result"]').innerText(), new RegExp(`Returned ${state.result}`));
          if (index + 1 < trace.length) await advance(search);
          record.searchStates++;
        }
        if (screenshot) await capture(search, screenshot);
      }
      await searchCase([4, 9, 2, 9], 9, false, 'search-first-match');
      await search.getByLabel('Candidate invariant', { exact: true }).selectOption('bounds');
      assert.match(await search.locator('dl').innerText(), /position 1 matches/);
      await search.getByLabel('Candidate invariant', { exact: true }).selectOption('whole');
      assert.match(await search.locator('dl').innerText(), /initialized state does not satisfy/);
      await search.getByLabel('Candidate invariant', { exact: true }).selectOption('prefix');
      await searchCase([4, 9, 2, 9], 9, true, 'search-skipped-match');
      assert.match(await search.locator('dl').innerText(), /i=0 advances to i=2/);
      await searchCase([4, 9, 2, 9], 7, false);
      await searchCase([], 9, false, 'search-empty');
      await searchCase([9], 9, false);
      await search.getByLabel('Search values', { exact: true }).fill('1,,2');
      await search.getByRole('button', { name: 'Apply search input' }).click();
      assert.match(await search.getByRole('alert').innerText(), /comma-separated/);
      assert.match(await search.locator('[data-result="search-result"]').innerText(), /Returned 0/);
      await reset(search);
      assert.equal(await search.getByLabel('Search values', { exact: true }).inputValue(), '4, 9, 2, 9');

      const compaction = page.locator('[data-lab="compaction-invariant"]');
      async function compactionCase(values, removed, screenshot) {
        await compaction.getByLabel('Compaction values', { exact: true }).fill(values.join(','));
        await compaction.getByLabel('Value to remove', { exact: true }).fill(String(removed));
        await compaction.getByRole('button', { name: 'Apply compaction input' }).click();
        const trace = models.compactionTrace(values, removed);
        for (let index = 0; index < trace.states.length; index++) {
          const state = trace.states[index];
          assert.equal(await compaction.locator('[data-result="compaction-action"]').innerText(), state.action);
          const displayed = await compaction.getByRole('region', { name: 'Actual array · kept prefix / reusable gap / unread suffix', exact: true }).locator('.proof-cell strong').allTextContents();
          assert.deepEqual(displayed.map(Number), state.working.map(item => item.value));
          assert.equal(await compaction.locator('.proof-check.is-false').count(), 0);
          if (screenshot && index === Math.min(4, trace.states.length - 1)) await capture(compaction, screenshot);
          if (index + 1 < trace.states.length) await advance(compaction);
          record.compactionStates++;
        }
      }
      await compactionCase([5, 0, 5, 2, 0, 7], 0, 'compaction-occurrences');
      await capture(compaction, 'compaction-finished');
      await compactionCase([0, 0, 0], 0);
      await compactionCase([3, -1, 3], 0);
      await compactionCase([], 0, 'compaction-empty');
      await compactionCase([0, 1, 0, 2, 0, 3, 0, 4], 0, 'compaction-long');
      await compaction.getByLabel('Value to remove', { exact: true }).fill('1.5');
      await compaction.getByRole('button', { name: 'Apply compaction input' }).click();
      assert.match(await compaction.getByRole('alert').innerText(), /integer/);
      await reset(compaction);
      assert.match(await compaction.locator('.proof-state-label').innerText(), /read = 0 · write = 0/);

      const partition = page.locator('[data-lab="partition-invariant"]');
      async function partitionCase(values, faulty, screenshot) {
        await partition.getByLabel('Partition categories', { exact: true }).fill(values.join(','));
        await partition.getByRole('button', { name: 'Apply partition input' }).click();
        await partition.getByRole('checkbox').setChecked(faulty);
        const trace = models.partitionTrace(values, faulty);
        for (let index = 0; index < trace.length; index++) {
          const state = trace[index];
          assert.equal(await partition.locator('[data-result="partition-action"]').innerText(), state.action);
          assert.deepEqual((await partition.locator('.proof-cell strong').allTextContents()).map(Number), state.working.map(item => item.value));
          assert.equal(await partition.locator('.proof-check.is-false').count(), Object.values(state.checks).filter(value => !value).length);
          if (index + 1 < trace.length) await advance(partition);
          record.partitionStates++;
        }
        if (screenshot) await capture(partition, screenshot);
      }
      await partitionCase([1, 2, 0], false, 'partition-correct');
      await partitionCase([1, 2, 0], true, 'partition-skipped');
      await partitionCase([2], true);
      await partitionCase([2, 0, 1, 2, 0, 1], false, 'partition-regions');
      await partitionCase([1, 1, 1], false);
      await partitionCase([], false, 'partition-empty');
      await partition.getByLabel('Partition categories', { exact: true }).fill('1,3,0');
      await partition.getByRole('button', { name: 'Apply partition input' }).click();
      assert.match(await partition.getByRole('alert').innerText(), /only the categories/);
      await reset(partition);
      assert.equal(await partition.getByRole('checkbox').isChecked(), false);

      const euclid = page.locator('[data-lab="euclid-termination"]');
      async function euclidCase(first, second, screenshot) {
        await euclid.getByLabel('First integer', { exact: true }).fill(String(first));
        await euclid.getByLabel('Second integer', { exact: true }).fill(String(second));
        await euclid.getByRole('button', { name: 'Apply Euclid input' }).click();
        const trace = models.euclidTrace(first, second);
        for (let index = 0; index < trace.length; index++) {
          const state = trace[index];
          const expected = state.terminal ? `b = 0: return a = ${state.a}` : `${state.a} = ${state.quotient} × ${state.b} + ${state.remainder}`;
          assert.equal(await euclid.locator('[data-result="euclid-equation"]').innerText(), expected);
          assert.deepEqual((await euclid.locator('[data-result="euclid-divisors"] span').allTextContents()).map(Number), state.commonDivisors);
          if (screenshot && index === 0) await capture(euclid, screenshot);
          if (index + 1 < trace.length) await advance(euclid);
          record.euclidStates++;
        }
      }
      await euclidCase(84, 30, 'euclid-remainder');
      await capture(euclid, 'euclid-finished');
      await euclidCase(30, 84);
      await euclidCase(0, 12, 'euclid-zero-first');
      await euclidCase(12, 0);
      await euclidCase(96, 95);
      await euclid.getByLabel('First integer', { exact: true }).fill('0');
      await euclid.getByLabel('Second integer', { exact: true }).fill('0');
      await euclid.getByRole('button', { name: 'Apply Euclid input' }).click();
      assert.match(await euclid.getByRole('alert').innerText(), /at least one positive/);
      await reset(euclid);
      assert.match(await euclid.locator('[data-result="euclid-equation"]').innerText(), /84 = 2 × 30 \+ 24/);

      for (const lab of [search, compaction, partition, euclid]) {
        await lab.getByRole('button', { name: 'Finish trace', exact: true }).focus();
        await page.keyboard.press('Enter');
        const lastAction = await lab.locator('[data-result]').first().innerText();
        await lab.getByRole('button', { name: 'Previous state', exact: true }).focus();
        await page.keyboard.press('Space');
        assert.equal(await lab.getByRole('button', { name: 'Next state', exact: true }).isEnabled(), true);
        await lab.getByRole('button', { name: 'Finish trace', exact: true }).click();
        assert.equal(await lab.locator('[data-result]').first().innerText(), lastAction);
        await reset(lab);
      }
      record.previousFinishResetKeyboard = 4;

      const figures = page.locator('.proof-inline');
      assert.equal(await figures.count(), 4);
      for (let index = 0; index < await figures.count(); index++) await capture(figures.nth(index), `inline-${index}`);
      const headingLinks = page.locator('nav[aria-label="In this lesson"] a');
      record.anchors = await headingLinks.count();
      for (const href of await headingLinks.evaluateAll(links => links.map(link => link.getAttribute('href')))) {
        assert.equal(await page.locator(`[id="${href.slice(1)}"]`).count(), 1, href);
      }
      const practiceSection = page.locator('#guided-dsa-practice').locator('..');
      record.practice = practice.groups.flatMap(group => group.problems).length;
      const problemLinks = await page.locator('a[href^="https://leetcode.com/problems/"]').evaluateAll(links => links.map(link => ({ href: link.href, target: link.target, rel: link.rel })));
      assert.equal(problemLinks.length, record.practice);
      for (const link of problemLinks) {
        assert.equal(link.target, '_blank');
        assert.match(link.rel, /noopener/);
      }
      const hint = practiceSection.locator('summary').filter({ hasText: /^Optional hint/ }).first();
      assert.equal(await hint.locator('..').getAttribute('open'), null);
      await hint.focus();
      await page.keyboard.press('Enter');
      assert.notEqual(await hint.locator('..').getAttribute('open'), null);
      await capture(practiceSection, 'practice-hint');
      // Native example fixtures are compared with the actual rendered code and stdout.
      for (const details of await page.locator('.correctness-lesson details').all()) {
        await details.evaluate(element => { element.open = true; });
      }
      const blocks = await page.locator('.correctness-lesson .python-example > div').evaluateAll(elements => elements.map(element => Array.from(element.childNodes).filter(node => node.nodeType === Node.TEXT_NODE).map(node => node.textContent).join('')));
      for (const [key, example] of Object.entries(examples)) {
        assert(blocks.some(text => normalize(text) === normalize(example.code)), `${width}: code ${key}`);
        assert(blocks.some(text => normalize(text) === normalize(example.expected)), `${width}: stdout ${key}`);
      }
      record.examples = Object.keys(examples).length;
      for (const details of await page.locator('.correctness-lesson details').all()) {
        await details.evaluate(element => { element.open = false; });
      }
      // Read in the normal closed-disclosure state, preserving site navigation.
      const headings = page.locator('.correctness-lesson h2');
      record.readingSections = await headings.count();
      for (let index = 0; index < record.readingSections; index++) {
        await headings.nth(index).evaluate(element => window.scrollTo(0, element.getBoundingClientRect().top + window.scrollY - 82));
        await page.screenshot({ path: path.join(directory, `reading-${index}-${width}.png`) });
      }
      await page.evaluate(() => window.scrollTo(0, 0));
      await page.screenshot({ path: path.join(directory, `reading-intro-${width}.png`) });
      record.pageOverflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1);
      assert.equal(record.pageOverflow, false);
      record.labOverflow = await page.locator('.proof-lab, .proof-inline').evaluateAll(elements => elements.filter(element => element.scrollWidth > element.clientWidth + 1).map(element => element.getAttribute('aria-label')));
      assert.deepEqual(record.labOverflow, []);
      if (width === 390) {
        await compactionCase([0, 1, 0, 2, 0, 3, 0, 4], 0);
        const strip = compaction.getByRole('region', { name: 'Actual array · kept prefix / reusable gap / unread suffix', exact: true });
        await strip.focus();
        await page.keyboard.press('ArrowRight');
        await page.keyboard.press('ArrowRight');
        await page.waitForTimeout(250);
        assert(await strip.evaluate(element => element.scrollLeft > 0));
        record.keyboardLocalScroll = true;
      }
      results.push(record);
      await page.close();
    }
    assert.deepEqual(errors, []);
    const evidence = { checkedAt: new Date().toISOString(), base, results, errors };
    fs.writeFileSync(path.join(directory, 'results.json'), JSON.stringify(evidence, null, 2));
    console.log(JSON.stringify(evidence, null, 2));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const directory = path.resolve('scratch/hashing-amortized-browser');
fs.mkdirSync(directory, { recursive: true });
const base = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:5173';
const route = '/learn/path/full-curriculum/hashing-collision-resolution-amortized-analysis?module=data-structures-algorithms';
const normalize = text => text.replace(/\s+/g, ' ').trim();

(async () => {
  const models = await import(pathToFileURL(path.resolve('src/learn/data/hashing-amortized-models.js')));
  const { hashingAmortizedExamples: examples } = await import(pathToFileURL(path.resolve('src/learn/data/hashing-amortized-examples.js')));
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [], errors = [];
  try {
    for (const width of [1440, 390]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      page.on('pageerror', error => errors.push(`${width}: ${error.message}`));
      await page.routeWebSocket('**', socket => socket.close());
      await page.goto(base + route);
      await page.locator('[data-lab="probe-chain"]').waitFor();
      const record = { width, probeStates: 0, familyCases: 0, resizeStates: 0, denseStates: 0 };
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
        await lab.getByRole('button', { name: 'Reset', exact: true }).focus();
        await page.keyboard.press('Space');
      }
      const probe = page.locator('[data-lab="probe-chain"]');
      async function probeCase(program, capacity, faulty, name) {
        await probe.getByLabel('Operation script', { exact: true }).fill(program);
        await probe.getByLabel('Initial capacity', { exact: true }).selectOption(String(capacity));
        await probe.getByRole('button', { name: 'Apply script', exact: true }).click();
        await probe.getByRole('checkbox').setChecked(faulty);
        const states = models.probeProgramTrace(program, capacity, faulty).states;
        for (let index = 0; index < states.length; index++) {
          const state = states[index];
          assert.equal(await probe.locator('[data-result="probe-action"]').innerText(), state.action);
          assert.equal(await probe.locator('[data-result="probe-load"]').innerText(), `${state.live} / ${state.deleted} / ${state.capacity}`);
          assert.equal(await probe.locator('.hash-slot').count(), state.capacity);
          assert.deepEqual((await probe.locator('.hash-slot strong').allTextContents()).map(Number), state.slots.filter(slot => slot && !slot.deleted).map(slot => slot.key));
          assert.equal(await probe.locator('[data-result="probe-invariant"] li').count(), state.issues.length);
          if (name && ((faulty && state.phase === 'commit' && !state.agrees) || (!faulty && state.phase === 'commit' && state.deleted))) {
            await capture(probe, name);
            name = undefined;
          }
          if (index + 1 < states.length) await advance(probe);
          record.probeStates++;
        }
        if (name) await capture(probe, name);
      }
      await probeCase(models.defaultProbeProgram, 8, false, 'probe-tombstone');
      await probeCase(models.defaultProbeProgram, 8, true, 'probe-failure');
      await probeCase('put 3 1\nput 7 2\nput 11 3\nput 15 4\nget -1\nput 19 5\ndel 7\nput 19 5\nget 19\nrebuild 8', 4, false, 'probe-full');
      await probeCase('put -1 0\nput 15 1\nget 15', 16, false, 'probe-wrap');
      if (width === 390) {
        const strip = probe.locator('.hash-strip');
        const active = probe.locator('.hash-active');
        const box = await strip.boundingBox(), cell = await active.boundingBox();
        assert(cell.x >= box.x - 1 && cell.x + cell.width <= box.x + box.width + 1);
        record.activeProbeVisible = true;
      }
      await probeCase('', 4, false, 'probe-empty');
      await probe.getByLabel('Operation script', { exact: true }).fill('put 1 nope');
      await probe.getByRole('button', { name: 'Apply script', exact: true }).click();
      assert.match(await probe.getByRole('alert').innerText(), /Line 1/);
      assert.match(await probe.locator('[data-result="probe-action"]').innerText(), /Every slot starts EMPTY/);
      await reset(probe);

      const family = page.locator('[data-lab="hash-family"]');
      async function familyCase(keys, query, a, b, name) {
        await family.getByLabel('Fixed keys', { exact: true }).fill(keys);
        await family.getByRole('button', { name: 'Apply keys', exact: true }).click();
        await family.getByLabel('Query key', { exact: true }).selectOption(String(query));
        await family.getByLabel('Multiplier a', { exact: true }).selectOption(String(a));
        await family.getByLabel('Offset b', { exact: true }).selectOption(String(b));
        const state = models.hashFamilyState(keys, a, b, query);
        assert.equal(await family.locator('[data-result="family-selected"]').innerText(), String(state.selectedLength));
        assert.equal(await family.locator('[data-result="family-mean"]').innerText(), `${state.totalLength} / 272 ≈ ${(state.totalLength / 272).toFixed(6)}`);
        assert.equal(await family.locator('svg rect').count(), 272);
        assert.equal(await family.locator('svg rect[stroke="#fff"]').count(), 1);
        assert.deepEqual(await family.locator('.hash-frequency strong').allTextContents(), state.distribution.map(row => `${row.count} / 272`));
        if (name) await capture(family, name);
        record.familyCases++;
      }
      await familyCase('1,5,9,13', 9, 1, 0, 'family-exact');
      await familyCase('0,2,4,6,8,10,12,14', 14, 16, 16, 'family-eight');
      await familyCase('16', 16, 1, 0, 'family-single');
      await family.getByLabel('Fixed keys', { exact: true }).fill('1,1');
      await family.getByRole('button', { name: 'Apply keys', exact: true }).click();
      assert.match(await family.getByRole('alert').innerText(), /distinct/);
      assert.equal(await family.locator('[data-result="family-selected"]').innerText(), '1');
      await reset(family);
      assert.equal(await family.getByLabel('Query key', { exact: true }).inputValue(), '9');

      const resize = page.locator('[data-lab="resize-accounting"]');
      async function resizeCase(mode, scenario, name) {
        await resize.getByLabel('Comparison', { exact: true }).selectOption(mode);
        if (mode === 'shrink') await resize.getByLabel('Operation sequence', { exact: true }).selectOption(scenario);
        const operations = mode === 'growth' ? models.resizeScenarios.append : models.resizeScenarios[scenario];
        const left = models.resizeTrace(operations, mode === 'growth' ? 'never' : 'half');
        const right = models.resizeTrace(operations, mode === 'growth' ? 'never' : 'quarter', mode === 'growth' ? 'one' : 'double');
        for (let index = 0; index < left.length; index++) {
          for (const [side, trace] of [left, right].entries()) {
            assert.equal(await resize.locator(`[data-result="resize-total-${side}"]`).innerText(), String(trace[index].total));
          }
          if (index + 1 < left.length) await advance(resize);
          record.resizeStates++;
        }
        await capture(resize, name);
      }
      await resizeCase('growth', 'boundary', 'resize-geometric');
      await resizeCase('shrink', 'boundary', 'resize-thrashing');
      await resizeCase('shrink', 'drain', 'resize-empty');
      await reset(resize);

      const dense = page.locator('[data-lab="dense-set"]');
      for (const removed of [30, 40, 99]) {
        await dense.getByLabel('Value to remove', { exact: true }).selectOption(String(removed));
        const states = models.denseSetTrace([10, 30, 20, 40], removed);
        for (let index = 0; index < states.length; index++) {
          assert.equal(await dense.locator('[data-result="dense-action"]').innerText(), states[index].action);
          assert.deepEqual((await dense.locator('.hash-dense-array strong').allTextContents()).map(Number), states[index].items);
          if (removed === 30 && index === 1) await capture(dense, 'dense-during-move');
          if (index + 1 < states.length) await advance(dense);
          record.denseStates++;
        }
        await capture(dense, `dense-after-${removed}`);
      }
      await reset(dense);
      for (const lab of [probe, resize, dense]) {
        await lab.getByRole('button', { name: 'Finish trace', exact: true }).focus();
        await page.keyboard.press('Enter');
        assert(await lab.getByRole('button', { name: 'Next state', exact: true }).isDisabled());
        await lab.getByRole('button', { name: 'Previous state', exact: true }).focus();
        await page.keyboard.press('Space');
        assert(await lab.getByRole('button', { name: 'Next state', exact: true }).isEnabled());
        await reset(lab);
      }
      record.keyboardPreviousFinishReset = 3;
      const figures = page.locator('.hash-inline');
      assert.equal(await figures.count(), 3);
      for (let index = 0; index < 3; index++) await capture(figures.nth(index), `inline-${index}`);
      const links = page.locator('nav[aria-label="In this lesson"] a');
      record.anchors = await links.count();
      for (const href of await links.evaluateAll(elements => elements.map(element => element.getAttribute('href')))) {
        assert.equal(await page.locator(`[id="${href.slice(1)}"]`).count(), 1, href);
      }
      const practice = page.locator('#guided-dsa-practice').locator('..');
      const hint = practice.locator('summary').filter({ hasText: /^Optional hint/ }).first();
      assert.equal(await hint.locator('..').getAttribute('open'), null);
      await hint.focus();
      await page.keyboard.press('Enter');
      assert.notEqual(await hint.locator('..').getAttribute('open'), null);
      record.practiceLinks = await practice.locator('a[href^="https://leetcode.com/problems/"][target="_blank"]').count();
      assert.equal(record.practiceLinks, 5);
      for (const details of await page.locator('.hashing-lesson details').all()) await details.evaluate(element => { element.open = true; });
      const blocks = await page.locator('.hashing-lesson .python-example > div').evaluateAll(elements => elements.map(element => Array.from(element.childNodes).filter(node => node.nodeType === Node.TEXT_NODE).map(node => node.textContent).join('')));
      for (const [key, example] of Object.entries(examples)) {
        assert(blocks.some(text => normalize(text) === normalize(example.code)), `${width}: code ${key}`);
        assert(blocks.some(text => normalize(text) === normalize(example.expected)), `${width}: output ${key}`);
      }
      record.completeExamples = Object.keys(examples).length;
      const potential = page.locator('.hashing-lesson details').filter({ has: page.locator('summary', { hasText: 'Deeper: a potential for both append and pop' }) });
      await capture(potential, 'potential-mixed-proof');
      for (const details of await page.locator('.hashing-lesson details').all()) await details.evaluate(element => { element.open = false; });
      const headings = page.locator('.hashing-lesson h2');
      record.readingSections = await headings.count();
      for (let index = 0; index < record.readingSections; index++) {
        await headings.nth(index).evaluate(element => window.scrollTo(0, element.getBoundingClientRect().top + window.scrollY - 82));
        await page.screenshot({ path: path.join(directory, `reading-${index}-${width}.png`) });
      }
      await page.evaluate(() => window.scrollTo(0, 0));
      await page.screenshot({ path: path.join(directory, `reading-intro-${width}.png`) });
      record.pageOverflow = await page.evaluate(() => document.documentElement.scrollWidth > window.innerWidth + 1);
      assert.equal(record.pageOverflow, false);
      record.figureOverflow = await page.locator('.hash-lab, .hash-inline').evaluateAll(elements => elements.filter(element => element.scrollWidth > element.clientWidth + 1).map(element => element.getAttribute('aria-label')));
      assert.deepEqual(record.figureOverflow, []);
      if (width === 390) {
        const strip = probe.locator('.hash-strip');
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
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

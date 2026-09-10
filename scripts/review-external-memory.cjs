const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const directory = path.resolve('scratch/external-memory-browser');
fs.mkdirSync(directory, { recursive: true });
const normalize = value => value.replace(/\s+/g, ' ').trim();

(async () => {
  const model = await import(pathToFileURL(path.resolve('src/learn/data/external-memory-models.js')));
  const { externalMemoryExamples: examples } = await import(pathToFileURL(path.resolve('src/learn/data/external-memory-examples.js')));
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [], errors = [];
  try {
    for (const width of [1440, 390]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      page.on('pageerror', error => errors.push(`${width}: ${error.message}`));
      await page.goto((process.env.LEARNING_BASE_URL || 'http://127.0.0.1:5173') + '/learn/path/full-curriculum/external-memory-algorithms-b-trees-i-o-complexity?module=data-structures-algorithms');
      const lesson = page.locator('.external-memory-lesson');
      await lesson.waitFor();
      const record = { width, anchors: [], bufferStates: 0, treeStates: 0, rangeCases: 0, mergeCases: 0, crashStates: 0, examples: [], screenshots: [] };
      const keyButton = async (region, name, key = 'Enter') => {
        await region.getByRole('button', { name, exact: true }).focus();
        await page.keyboard.press(key);
      };
      const setRange = async (region, name, value) => {
        const input = region.getByRole('slider', { name, exact: true });
        const minimum = Number(await input.getAttribute('min'));
        await input.focus();
        await page.keyboard.press('Home');
        for (let step = minimum; step < value; step++) await page.keyboard.press('ArrowRight');
        assert.equal(await input.inputValue(), String(value));
      };
      const fact = (region, name) => region.locator('.external-facts > div').filter({ has: page.locator('dt', { hasText: name }) }).locator('dd');
      const shot = async (locator, name) => {
        await locator.scrollIntoViewIfNeeded();
        await locator.evaluate(element => window.scrollTo(0, element.getBoundingClientRect().top + window.scrollY - 75));
        await page.screenshot({ path: path.join(directory, `${name}-${width}.png`) });
        record.screenshots.push(`${name}-${width}.png`);
      };
      const anchors = lesson.locator('nav[aria-label="In this lesson"] a');
      assert.equal(await anchors.count(), 9);
      for (let index = 0; index < await anchors.count(); index++) {
        const href = await anchors.nth(index).getAttribute('href');
        await anchors.nth(index).focus();
        await page.keyboard.press('Enter');
        await page.waitForFunction(hash => location.hash === hash, href);
        await page.waitForTimeout(100);
        const box = await page.locator(`[id="${href.slice(1)}"]`).boundingBox();
        assert(box && box.y >= 45 && box.y < 200, `anchor ${href}: ${box?.y}`);
        record.anchors.push({ href, top: box.y });
        await page.screenshot({ path: path.join(directory, `reading-${index + 1}-${width}.png`) });
      }
      const buffer = lesson.getByRole('region', { name: 'Pages and buffer residency', exact: true });
      await shot(buffer, 'buffer-initial');
      for (const pattern of ['sequential', 'strided', 'writes', 'reuse']) {
        await buffer.getByLabel('Request pattern').selectOption(pattern);
        const state = model.bufferTrace(4, 2, pattern);
        for (let index = 0; index < state.frames.length; index++) {
          const frame = state.frames[index];
          assert((await buffer.locator('.external-current').innerText()).includes(frame.action));
          assert.equal(await fact(buffer, 'Page reads').innerText(), String(frame.reads));
          assert.equal(await fact(buffer, 'Page writes').innerText(), String(frame.writes));
          assert.equal(await fact(buffer, 'Resident hits').innerText(), String(frame.hits));
          assert.equal(await buffer.locator('.external-resident .is-dirty').count(), frame.resident.filter(page => page.dirty).length);
          record.bufferStates++;
          if (index + 1 < state.frames.length) await keyButton(buffer, 'Next request');
        }
      }
      await buffer.getByLabel('Request pattern').selectOption('strided');
      await buffer.getByLabel('Buffer frames').selectOption('4');
      await keyButton(buffer, 'Finish request', 'Space');
      assert.equal(await fact(buffer, 'Page reads').innerText(), '4');
      await buffer.getByLabel('Records per page').selectOption('1');
      await keyButton(buffer, 'Finish request');
      assert.equal(await fact(buffer, 'Page reads').innerText(), '16');
      await keyButton(buffer, 'Reset buffer experiment');
      assert.equal(await buffer.getByLabel('Records per page').inputValue(), '4');
      assert.equal(await buffer.getByLabel('Buffer frames').inputValue(), '2');
      assert.equal(await fact(buffer, 'Page reads').innerText(), '0');

      const tree = lesson.getByRole('region', { name: 'B-tree page repairs', exact: true });
      for (const preset of ['insert', 'delete', 'duplicates']) {
        await tree.getByLabel('Tree scenario').selectOption(preset);
        const state = model.btreeTrace(2, preset);
        for (let index = 0; index < state.frames.length; index++) {
          const frame = state.frames[index];
          assert((await tree.locator('.external-current').innerText()).includes(frame.action));
          const expected = [];
          const walk = root => { expected.push(...root.keys); root.children.forEach(walk); };
          walk(frame.root);
          const shown = await tree.locator('figure .external-keys > span').allTextContents();
          assert.deepEqual(shown.map(Number), expected);
          record.treeStates++;
          if (index + 1 < state.frames.length) await keyButton(tree, 'Next tree event');
        }
        await shot(tree.locator('figure'), `tree-final-${preset}`);
      }
      await tree.getByLabel('Minimum degree t').selectOption('3');
      await tree.getByLabel('Tree scenario').selectOption('insert');
      await keyButton(tree, 'Finish tree event');
      await setRange(tree, 'Key to try', 17);
      await keyButton(tree, 'Search final tree');
      assert((await tree.locator('p[role="status"]').innerText()).includes('17 is present'));
      assert.equal(await tree.locator('figure .is-active').count(), model.btreeTrace(3, 'insert').search(17).visited.length);
      await shot(tree.locator('figure'), 'tree-search');
      await setRange(tree, 'Key to try', 18);
      assert.equal(await tree.locator('p[role="status"]').count(), 0);
      await keyButton(tree, 'Apply to final tree');
      await keyButton(tree, 'Search final tree');
      assert((await tree.locator('p[role="status"]').innerText()).includes('18 is present'));
      await keyButton(tree, 'Previous tree event');
      assert.equal(await tree.locator('p[role="status"]').count(), 0);
      await keyButton(tree, 'Reset tree experiment');
      assert.equal(await tree.getByLabel('Minimum degree t').inputValue(), '2');
      assert.equal(await tree.getByRole('slider').inputValue(), '8');
      await shot(tree, 'tree-reset');

      const range = lesson.getByRole('region', { name: 'B-plus linked leaf range', exact: true });
      await shot(lesson.locator('.external-contrast'), 'bplus-contrast');
      for (const [low, high, capacity, fanout] of [[10,27,3,3],[20,20,3,3],[36,40,3,3],[9,3,3,3],[0,40,2,2],[5,29,4,4]]) {
        await setRange(range, 'Start of inclusive range', low);
        await setRange(range, 'End of inclusive range', high);
        await range.getByLabel('Records per leaf').selectOption(String(capacity));
        await range.getByLabel('Internal fanout').selectOption(String(fanout));
        await keyButton(range, 'Finish range page');
        const state = model.bplusRange(low, high, capacity, fanout);
        assert.equal(await fact(range, 'Records returned so far').innerText(), state.result.join(', ') || 'none');
        assert.equal(await fact(range, 'Cold query page reads so far').innerText(), String(state.visited.length));
        record.rangeCases++;
      }
      await keyButton(range, 'Reset range experiment');
      await shot(range, 'bplus-initial');
      await shot(range.locator('figure'), 'bplus-pages');

      const merge = lesson.getByRole('region', { name: 'External merge runs and transfers', exact: true });
      await shot(merge, 'merge-initial');
      for (const [n,b,p] of [[32,4,3],[7,2,3],[48,4,4],[0,4,3],[12,8,8],[20,2,4]]) {
        await merge.getByLabel('Record count').selectOption(String(n));
        await merge.getByLabel('Records per transfer page').selectOption(String(b));
        await merge.getByLabel('Available record-buffer pages').selectOption(String(p));
        const state = model.mergePlan(n,b,p);
        if (!n) {
          assert((await merge.getByRole('status').innerText()).includes('zero transfers'));
          assert.equal(await merge.getByRole('button', { name: 'Next merge pass', exact: true }).count(), 0);
        } else {
          for (let index = 0; index < state.stages.length; index++) {
            const stage = state.stages[index];
            assert.equal(await fact(merge, 'This pass: page reads').innerText(), String(stage.reads));
            assert.equal(await fact(merge, 'This pass: page writes').innerText(), String(stage.writes));
            assert.equal(await merge.locator('.external-run').count(), stage.runs.length);
            for (let run = 0; run < stage.runs.length; run++) {
              assert.deepEqual((await merge.locator('.external-run').nth(run).locator('.external-keys > span').allTextContents()).map(Number), stage.runs[run]);
            }
            if (index + 1 < state.stages.length) await keyButton(merge, 'Next merge pass');
          }
        }
        record.mergeCases++;
      }
      await keyButton(merge, 'Reset merge experiment');
      await shot(merge.locator('.external-runs'), 'merge-runs');

      const crash = lesson.getByRole('region', { name: 'Shadow page crash recovery', exact: true });
      for (const early of [false, true]) {
        await crash.getByLabel('Publication order').selectOption(early ? 'early' : 'safe');
        const state = model.shadowCommitTrace(early);
        for (let index = 0; index < state.frames.length; index++) {
          const frame = state.frames[index];
          const expected = frame.missing.length ? `invalid reachable root; missing ${frame.missing.join(', ')}` : `recover value ${frame.recoveredValue}`;
          assert((await crash.getByRole('status').innerText()).includes(expected));
          assert((await crash.locator('.external-current').innerText()).includes(frame.action));
          assert.equal(await crash.locator('.is-pending').count(), frame.pending.length);
          record.crashStates++;
          if (early && index === 2) await shot(crash.locator('figure'), 'crash-invalid-pages');
          if (index + 1 < state.frames.length) await keyButton(crash, 'Next commit stage');
        }
      }
      await keyButton(crash, 'Reset commit experiment');
      await shot(crash, 'crash-reset');

      const examplesOnPage = lesson.locator('.python-example');
      assert.equal(await examplesOnPage.count(), Object.keys(examples).length);
      for (let index = 0; index < await examplesOnPage.count(); index++) {
        const block = examplesOnPage.nth(index);
        const title = await block.getByRole('heading').innerText();
        const [name, example] = Object.entries(examples).find(([,fixture]) => fixture.title === title);
        const displayed = await block.locator(':scope > div').evaluateAll(elements => elements.map(element => [...element.childNodes].filter(child => child.nodeType === Node.TEXT_NODE).map(child => child.textContent).join('')));
        assert.deepEqual(displayed.map(normalize), [normalize(example.code), normalize(example.expected)], name);
        record.examples.push(name);
      }
      const exercises = lesson.locator('.external-exercise');
      assert.equal(await exercises.count(), 7);
      assert.equal(await exercises.locator('details[open]').count(), 0);
      for (let index = 0; index < 7; index++) {
        await exercises.nth(index).locator('summary').focus();
        await page.keyboard.press('Enter');
        assert.equal(await exercises.nth(index).locator('details[open]').count(), 1);
        await page.keyboard.press('Enter');
      }
      const practice = lesson.locator('.dsa-practice');
      assert.equal(await practice.locator('details[open]').count(), 0);
      const links = practice.locator('a[href^="https://leetcode.com/problems/"]');
      assert.equal(await links.count(), 4);
      for (const link of await links.all()) {
        assert.equal(await link.getAttribute('target'), '_blank');
        assert((await link.getAttribute('rel')).includes('noreferrer'));
      }
      await practice.locator('.dsa-practice__extension > summary').focus();
      await page.keyboard.press('Enter');
      assert(await practice.getByRole('link', { name: /148/ }).isVisible());
      await page.keyboard.press('Enter');
      assert.equal(await lesson.locator('.lesson-sources a').count(), 5);
      await shot(lesson.locator('.lesson-sources'), 'sources');
      await shot(lesson.locator('.lesson-intro'), 'intro');
      const overflow = await lesson.locator('.external-lab, .external-inline, .external-calculation').evaluateAll(elements => elements.filter(element => element.scrollWidth > element.clientWidth + 1).map(element => element.getAttribute('aria-label') || element.className));
      assert.deepEqual(overflow, []);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
      results.push(record);
      await page.close();
    }
    assert.deepEqual(errors, []);
    const result = { checkedAt: new Date().toISOString(), results, errors };
    fs.writeFileSync(path.join(directory, 'results.json'), JSON.stringify(result, null, 2));
    console.log(JSON.stringify(result, null, 2));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

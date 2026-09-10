const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const directory = path.resolve(__dirname, '../scratch/heap-trie-lesson-review');
fs.mkdirSync(directory, { recursive: true });
const ascending = values => [...values].sort((a, b) => a - b);
const labelWords = words => [...words].sort().map(word => word || 'ε');

async function finish(lab, inspect = async () => {}) {
  let states = 0;
  while (true) {
    await inspect(); states++;
    const next = lab.locator('.heap-trie-steps button').nth(1);
    if (await next.isDisabled()) return states;
    assert.ok(states < 180, 'bounded trace terminates');
    await next.click();
  }
}

async function capture(page, element, name) {
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = 'hidden'));
  await element.screenshot({ path: path.join(directory, name) });
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = ''));
}

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const errors = [], results = [];
  for (const width of [1440, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(process.env.LESSON_URL || 'http://127.0.0.1:5173/learn/path/full-curriculum/heaps-priority-queues-tries');
    const heap = page.locator('[data-lab="heap-operations"]'); await heap.waitFor();
    const topK = page.locator('[data-lab="top-k-stream"]');
    const trie = page.locator('[data-lab="trie-prefix"]');
    assert.equal(await page.locator('[data-lab]').count(), 3);
    assert.equal(await page.locator('.python-example').count(), 10);
    assert.equal(await page.locator('.heap-trie-inline-figure').count(), 2);
    const practice = page.locator('[data-practice-topic="heaps-priority-queues-tries"]');
    assert.equal(await practice.locator('details[open]').count(), 0);
    assert.equal(await practice.locator('a[href^="https://leetcode.com/problems/"]').count(), 10);
    assert.equal(await practice.locator('a:not([rel="noopener noreferrer"])').count(), 0);
    const optional = practice.locator('.dsa-practice__extension > summary');
    await optional.focus(); await page.keyboard.press('Enter');
    assert.equal(await optional.evaluate(element => element.parentElement.open), true);
    const hint = practice.locator('.dsa-practice__problem details > summary').first();
    await hint.focus(); await page.keyboard.press('Space');
    assert.equal(await hint.evaluate(element => element.parentElement.open), true);
    await page.keyboard.press('Space');
    assert.equal(await hint.evaluate(element => element.parentElement.open), false);
    await capture(page, practice.locator('.dsa-practice__stage').first(), `practice-foundation-${width}.png`);
    const anchors = await page.locator('.lesson-intro a[href^="#"]').evaluateAll(links => links.map(link => link.hash.slice(1)));
    assert.equal(anchors.length, 9);
    for (const anchor of anchors) assert.equal(await page.locator(`[id="${anchor}"]`).count(), 1);
    await page.locator('.lesson-intro a[href="#guided-dsa-practice"]').click();
    assert.equal(new URL(page.url()).hash, '#guided-dsa-practice');
    assert.ok(await page.locator('a[href*="/graphs-representations-bfs-dfs"]').count() >= 1);

    const heapCases = [];
    for (const fixture of [
      { values: [1, 2, 9, 7, 5], operation: 'push', key: 0, expected: [0, 2, 1, 7, 5, 9] },
      { values: [1, 2, 9, 7, 5], operation: 'pop', expected: [2, 5, 9, 7], removed: 1 },
      { values: [7, 2, 9, 1, 5], operation: 'build', expected: [1, 2, 9, 7, 5] },
      { values: [1, 4, 2, 8, 7, 3, 9], operation: 'pop', expected: [2, 4, 3, 8, 7, 9], removed: 1 },
      { values: [1, 2, 2], operation: 'pop', expected: [2, 2], removed: 1 },
      { values: [3], operation: 'pop', expected: [], removed: 3 },
      { values: [], operation: 'pop', expected: [], terminal: /No minimum to remove/ },
      { values: [], operation: 'push', key: -2, expected: [-2] },
      { values: [7, 2, 9, 1, 5], operation: 'push', key: 0, expected: [7, 2, 9, 1, 5], terminal: /Repair the starting heap first/ },
      { values: Array.from({ length: 12 }, (_, index) => index + 1), operation: 'push', key: 13, expected: Array.from({ length: 12 }, (_, index) => index + 1), terminal: /Browser size limit/ },
    ]) {
      await heap.getByLabel('Starting array · up to 12 integers', { exact: true }).fill(fixture.values.join(','));
      await heap.locator('select').selectOption(fixture.operation);
      if (fixture.operation === 'push') await heap.getByLabel('Value to push', { exact: true }).fill(String(fixture.key));
      await heap.getByRole('button', { name: 'Run heap operation', exact: true }).click();
      const states = await finish(heap, async () => {
        const array = (await heap.locator('.heap-array-strip strong').allTextContents()).map(Number);
        const drawn = (await heap.locator('.heap-node-value').allTextContents()).map(Number);
        assert.deepEqual(drawn, array, 'array and tree agree at every repair step');
      });
      assert.deepEqual((await heap.locator('.heap-array-strip strong').allTextContents()).map(Number), fixture.expected);
      assert.match(await heap.getByRole('status').innerText(), fixture.terminal || /Heap operation complete/);
      if (fixture.removed !== undefined) assert.match(await heap.locator('.heap-trie-metrics').innerText(), new RegExp(`Returned minimum: ${fixture.removed}`));
      if (fixture.values.length === 7) await capture(page, heap, `heap-right-child-${width}.png`);
      heapCases.push({ operation: fixture.operation, input: fixture.values, states });
    }
    const savedHeap = await heap.locator('svg').getAttribute('aria-label');
    for (const invalid of ['1,,2', '100', '1.5', '1,2,3,4,5,6,7,8,9,10,11,12,13']) {
      await heap.getByLabel('Starting array · up to 12 integers', { exact: true }).fill(invalid);
      await heap.getByRole('button', { name: 'Run heap operation', exact: true }).click();
      assert.equal(await heap.getByRole('alert').count(), 1);
      assert.equal(await heap.locator('svg').getAttribute('aria-label'), savedHeap);
    }
    await heap.getByRole('button', { name: 'Unordered array → build', exact: true }).click();
    assert.equal(await heap.getByRole('alert').count(), 0);
    await heap.getByRole('button', { name: 'Reset investigation', exact: true }).click();
    const index = heap.getByRole('button', { name: 'Inspect index 2, value 9', exact: true });
    await index.focus(); await page.keyboard.press('Enter');
    assert.equal(await index.getAttribute('aria-pressed'), 'true');
    assert.match(await heap.locator('.heap-index-inspection').innerText(), /Selected index 2, value 9/);
    await heap.locator('.heap-trie-steps button').nth(1).click();
    await capture(page, heap, `heap-insertion-${width}.png`);
    await heap.getByRole('button', { name: 'Back', exact: true }).click();
    assert.match(await heap.locator('.heap-trie-steps').innerText(), /Step 1/);

    const topKCases = [];
    for (const fixture of [
      { stream: [5, 1, 9, 3, 9, 2], k: 3 },
      { stream: [5, 1, 9, 3, 9, 2], k: 8 },
      { stream: [-3, -3, 2, 2, -1], k: 1 },
      { stream: [], k: 2 },
      { stream: [0, 0, 0, 0], k: 2 },
    ]) {
      await topK.getByLabel('Stream · up to 16 integers', { exact: true }).fill(fixture.stream.join(','));
      await topK.getByLabel('k · number of occurrences', { exact: true }).fill(String(fixture.k));
      await topK.getByRole('button', { name: 'Read this stream', exact: true }).click();
      const states = await finish(topK, async () => {
        const active = await topK.locator('p.heap-trie-note').first().innerText();
        const consumed = Number(active.match(/(\d+) of \d+ occurrences committed/)[1]);
        const expected = ascending(fixture.stream.slice(0, consumed)).reverse().slice(0, fixture.k);
        const retained = (await topK.locator('.heap-array-strip strong').allTextContents()).map(Number);
        assert.deepEqual(ascending(retained).reverse(), expected);
        assert.equal(await topK.locator('.top-k-stream-ribbon .is-retained').count(), retained.length);
        assert.equal(await topK.locator('.top-k-stream-ribbon .is-discarded').count(), consumed - retained.length);
        assert.equal((await topK.locator('.top-k-boundary > strong').innerText()).trim(), consumed < fixture.k ? 'not available yet' : String(expected.at(-1)));
      });
      topKCases.push({ ...fixture, states });
      if (fixture.k === 3) await capture(page, topK, `stream-duplicates-${width}.png`);
    }
    const savedStream = await topK.locator('.top-k-stream-ribbon').textContent();
    for (const invalid of ['0', '9', '1.5', 'x']) {
      await topK.getByLabel('k · number of occurrences', { exact: true }).fill(invalid);
      await topK.getByRole('button', { name: 'Read this stream', exact: true }).click();
      assert.equal(await topK.getByRole('alert').count(), 1);
      assert.equal(await topK.locator('.top-k-stream-ribbon').textContent(), savedStream);
    }
    await topK.getByRole('button', { name: 'Reset investigation', exact: true }).click();
    assert.equal(await topK.getByLabel('k · number of occurrences', { exact: true }).inputValue(), '3');

    const words = ['car', 'cart', 'cat', 'dog'];
    const trieCases = [];
    for (const fixture of [
      { words, query: 'car', operation: 'exact', outcome: /word is stored/ },
      { words, query: 'ca', operation: 'exact', outcome: /only a prefix/ },
      { words, query: 'ca', operation: 'prefix', matches: ['car', 'cart', 'cat'] },
      { words, query: 'car', operation: 'delete' },
      { words, query: 'cart', operation: 'delete' },
      { words, query: 'ca', operation: 'delete', outcome: /not a stored word/ },
      { words, query: 'cab', operation: 'delete', outcome: /absent/ },
      { words, query: 'car', operation: 'insert', outcome: /existing terminal/ },
      { words, query: 'ca', operation: 'insert' },
      { words, query: '', operation: 'insert' },
      { words: ['', 'car'], query: '', operation: 'exact', outcome: /word is stored/ },
      { words: ['', 'car'], query: '', operation: 'delete' },
      { words: [], query: '', operation: 'prefix', matches: [] },
      { words: [], query: '', operation: 'exact', outcome: /only a prefix/ },
      { words: [], query: 'dog', operation: 'insert' },
    ]) {
      await trie.getByLabel('Starting words · comma-separated', { exact: true }).fill(fixture.words.map(word => word || 'ε').join(','));
      await trie.getByLabel('Word or prefix', { exact: true }).fill(fixture.query);
      await trie.locator('select').selectOption(fixture.operation);
      await trie.getByRole('button', { name: 'Trace these characters', exact: true }).click();
      let sawClearMarker = false;
      const states = await finish(trie, async () => {
        const rows = await trie.locator('tbody tr').evaluateAll(nodes => nodes.map(node => [...node.querySelectorAll('th,td')].map(cell => cell.textContent)));
        const terminalWords = labelWords(rows.filter(row => row[1] === 'yes').map(row => row[0] === 'ε' ? '' : row[0]));
        const visibleWords = (await trie.locator('.trie-word-results > p').first().innerText()).trim();
        assert.equal(visibleWords, terminalWords.length ? terminalWords.join(' · ') : 'No stored words.');
        if (fixture.operation === 'delete' && fixture.query === 'car' && (await trie.getByRole('status').innerText()).startsWith('Clear only')) {
          sawClearMarker = true;
          assert.ok(rows.some(row => row[0] === 'car' && row[1] === 'no' && row[2] === 't → cart'));
          await capture(page, trie, `trie-clear-marker-${width}.png`);
        }
      });
      const expected = new Set(fixture.words);
      if (fixture.operation === 'insert') expected.add(fixture.query);
      if (fixture.operation === 'delete') expected.delete(fixture.query);
      const expectedText = labelWords(expected).join(' · ') || 'No stored words.';
      assert.equal((await trie.locator('.trie-word-results > p').first().innerText()).trim(), expectedText);
      if (fixture.outcome) assert.match(await trie.getByRole('status').innerText(), fixture.outcome);
      if (fixture.matches) assert.equal((await trie.locator('.trie-word-results > p').nth(1).innerText()).trim(), fixture.matches.length ? labelWords(fixture.matches).join(' · ') : 'No stored words match.');
      if (fixture.operation === 'delete' && fixture.query === 'car') assert.equal(sawClearMarker, true);
      if (fixture.operation === 'delete' && fixture.query === 'cart') await capture(page, trie, `trie-pruning-${width}.png`);
      trieCases.push({ operation: fixture.operation, query: fixture.query, states });
    }
    const savedTrie = await trie.locator('svg').getAttribute('aria-label');
    for (const invalid of ['Car', 'abcdefg', 'car,,dog', 'abcdef,ghijkl,mnopqr,stuvwx,yzzzzz,zabcde']) {
      await trie.getByLabel('Starting words · comma-separated', { exact: true }).fill(invalid);
      await trie.getByRole('button', { name: 'Trace these characters', exact: true }).click();
      assert.equal(await trie.getByRole('alert').count(), 1);
      assert.equal(await trie.locator('svg').getAttribute('aria-label'), savedTrie);
    }
    await trie.getByRole('button', { name: 'Reset investigation', exact: true }).click();
    await trie.getByLabel('Word or prefix', { exact: true }).fill('C');
    await trie.getByRole('button', { name: 'Trace these characters', exact: true }).click();
    assert.equal(await trie.getByRole('alert').count(), 1);
    await trie.getByRole('button', { name: 'Reset investigation', exact: true }).click();
    assert.equal(await trie.getByLabel('Word or prefix', { exact: true }).inputValue(), 'car');
    const exactDetails = trie.locator('.trie-picture details > summary');
    await exactDetails.focus(); await page.keyboard.press('Enter');
    assert.equal(await exactDetails.evaluate(element => element.parentElement.open), true);

    for (const lab of [heap, topK, trie]) {
      const region = lab.locator('.heap-trie-diagram-scroll');
      await region.focus(); await page.keyboard.press('ArrowRight');
      assert.equal(await region.evaluate(element => document.activeElement === element), true);
      const control = lab.locator('input,select,button').first();
      await control.focus(); await page.keyboard.press('Tab');
      assert.equal(await lab.evaluate(element => element.contains(document.activeElement)), true);
      const sizes = await lab.locator('input,select,button').evaluateAll(nodes => nodes.map(node => node.getBoundingClientRect().height));
      assert.ok(sizes.every(height => height >= 43));
      assert.equal(await lab.evaluate(element => element.scrollWidth > element.clientWidth + 2), false);
    }
    for (let index = 0; index < 2; index++) await capture(page, page.locator('.heap-trie-inline-figure').nth(index), `inline-${index + 1}-${width}.png`);
    await page.locator('.lesson-pilot').evaluate(element => element.querySelectorAll('details').forEach(details => details.open = true));
    assert.ok((await page.locator('.python-example pre').allTextContents()).every(text => text.trim()));
    assert.ok(await page.locator('.lesson-sources a').count() >= 5);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 2), false);
    results.push({ width, heapCases, topKCases, trieCases, programs: 10, practiceLinks: 10, anchors: anchors.length });
    console.log(`Heaps/Tries ${width}px integrated checks passed.`);
    await page.close();
  }
  assert.deepEqual(errors, []);
  fs.writeFileSync(path.join(directory, 'results.json'), JSON.stringify({ date: new Date().toISOString(), results, errors }, null, 2));
  await browser.close();
  console.log('Heaps/Tries: native controls, intermediate array/tree/trie agreement, top-k prefixes, contracts, reset/errors, keyboard, code, practice, anchors and overflow passed.');
})().catch(error => { console.error(error); process.exit(1); });

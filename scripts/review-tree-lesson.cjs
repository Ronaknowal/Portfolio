const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const repository = path.resolve(__dirname, '..');
const directory = path.join(repository, 'scratch/tree-lesson-review');
fs.mkdirSync(directory, { recursive: true });
const sample = [8, 3, 10, 1, 6, 14, 4, 7, 13];
const expectedOrders = {
  preorder: [8, 3, 1, 6, 4, 7, 10, 14, 13],
  inorder: [1, 3, 4, 6, 7, 8, 10, 13, 14],
  postorder: [1, 4, 7, 6, 3, 13, 14, 10, 8],
  'level-order': [8, 3, 10, 1, 6, 14, 4, 7, 13],
};

async function traceToEnd(lab, check = async () => {}) {
  let steps = 0;
  while (true) {
    await check();
    const next = lab.locator('.tree-trace-controls button').nth(1);
    if (await next.isDisabled()) break;
    assert.ok(steps++ < 150, 'bounded trace terminates');
    await next.click();
  }
  return steps + 1;
}

async function screenshot(page, locator, name) {
  // A sticky page navigation bar otherwise paints over a tall element capture.
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = 'hidden'));
  await locator.screenshot({ path: path.join(directory, name) });
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = ''));
}

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const errors = [], results = [];
  for (const width of [1440, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
    page.on('pageerror', error => errors.push(error.message));
    await page.goto((process.env.LESSON_URL || 'http://127.0.0.1:5173/learn/path/full-curriculum/trees-binary-search-trees'));
    const search = page.locator('[data-lab="tree-search"]');
    await search.waitFor();
    assert.equal(await page.locator('[data-lab]').count(), 3);
    assert.equal(await page.locator('.python-example').count(), 10);
    assert.equal(await page.locator('.tree-inline-figure').count(), 4);
    const practice = page.locator('[data-practice-topic="trees-binary-search-trees"]');
    assert.equal(await practice.count(), 1);
    assert.equal(await practice.locator('details[open]').count(), 0, 'practice help starts closed');
    assert.equal(await practice.locator('a[href^="https://leetcode.com/problems/"]').count(), 10);
    assert.equal(await practice.locator('a:not([rel="noopener noreferrer"])').count(), 0);
    const extension = practice.locator('.dsa-practice__extension > summary');
    await extension.focus();
    await page.keyboard.press('Enter');
    assert.equal(await practice.locator('.dsa-practice__extension[open]').count(), 1);
    const hint = practice.locator('.dsa-practice__problem details > summary').first();
    await hint.focus(); await page.keyboard.press('Space');
    assert.equal(await hint.evaluate(element => element.parentElement.open), true);
    await page.keyboard.press('Space');
    assert.equal(await hint.evaluate(element => element.parentElement.open), false);
    await screenshot(page, practice, `practice-${width}.png`);

    const anchors = await page.locator('.lesson-intro a[href^="#"]').evaluateAll(links => links.map(link => link.hash.slice(1)));
    assert.equal(anchors.length, 9);
    for (const anchor of anchors) assert.equal(await page.locator(`[id="${anchor}"]`).count(), 1, anchor);
    await page.locator('.lesson-intro a[href="#guided-dsa-practice"]').click();
    assert.equal(new URL(page.url()).hash, '#guided-dsa-practice');
    assert.ok(await page.locator('a[href*="/heaps-priority-queues-tries"]').count() >= 1, 'next syllabus topic stays Heaps/Tries');

    let searchStates = 0;
    searchStates += await traceToEnd(search);
    assert.match(await search.locator('.tree-path').innerText(), /8 → 3 → 6 → 7/);
    assert.match(await search.locator('.tree-readouts').innerText(), /4 node comparisons/);
    assert.match(await search.locator('[role="status"]').innerText(), /Key found/);
    await search.getByLabel('Target key', { exact: true }).fill('5');
    await search.getByRole('button', { name: 'Trace this operation', exact: true }).click();
    searchStates += await traceToEnd(search);
    assert.match(await search.locator('[role="status"]').innerText(), /empty child/);
    assert.match(await search.locator('.tree-bounds strong').innerText(), /4 < key < 6/);
    await search.locator('select').selectOption('insert');
    await search.getByRole('button', { name: 'Trace this operation', exact: true }).click();
    searchStates += await traceToEnd(search);
    assert.match(await search.locator('.tree-readouts').innerText(), /10 nodes/);
    assert.match(await search.locator('svg').getAttribute('aria-label'), /n10, key 5/);
    await search.getByLabel('Target key', { exact: true }).fill('6');
    await search.getByRole('button', { name: 'Trace this operation', exact: true }).click();
    searchStates += await traceToEnd(search);
    assert.match(await search.locator('[role="status"]').innerText(), /duplicate/);
    assert.match(await search.locator('.tree-readouts').innerText(), /9 nodes/);
    await search.locator('select').selectOption('search');
    await search.getByLabel('Target key', { exact: true }).fill('14');
    await search.getByRole('button', { name: 'Same keys, ascending order', exact: true }).click();
    searchStates += await traceToEnd(search);
    assert.match(await search.locator('.tree-readouts').innerText(), /height 8 edges/);
    assert.match(await search.locator('.tree-readouts').innerText(), /9 node comparisons/);
    await screenshot(page, search, `search-skew-${width}.png`);
    await search.getByRole('button', { name: 'Empty tree', exact: true }).click();
    searchStates += await traceToEnd(search);
    assert.match(await search.locator('.tree-readouts').innerText(), /0 nodes.*height −?[-−]?1 edges.*0 node comparisons/s);
    await search.locator('select').selectOption('insert');
    await search.getByRole('button', { name: 'Trace this operation', exact: true }).click();
    searchStates += await traceToEnd(search);
    assert.match(await search.locator('.tree-readouts').innerText(), /1 nodes/);
    assert.match(await search.locator('svg').getAttribute('aria-label'), /n1, key 14.*root/);
    await search.getByLabel('Insertion order (up to 12 integer entries)', { exact: true }).fill('1,2,3,4,5,6,7,8,9,10,11,12');
    await search.getByRole('button', { name: 'Build this tree', exact: true }).click();
    searchStates += await traceToEnd(search);
    assert.match(await search.locator('[role="status"]').innerText(), /Browser size limit/);
    const preservedTree = await search.locator('svg').getAttribute('aria-label');
    for (const invalid of ['1,,2', '1,2.5', '100', '1,2,3,4,5,6,7,8,9,10,11,12,13']) {
      await search.getByLabel('Insertion order (up to 12 integer entries)', { exact: true }).fill(invalid);
      await search.getByRole('button', { name: 'Build this tree', exact: true }).click();
      assert.equal(await search.getByRole('alert').count(), 1);
      assert.equal(await search.locator('svg').getAttribute('aria-label'), preservedTree);
    }
    await search.getByLabel('Target key', { exact: true }).fill('3.5');
    await search.getByRole('button', { name: 'Trace this operation', exact: true }).click();
    assert.equal(await search.getByRole('alert').count(), 1);
    await search.getByRole('button', { name: 'Empty tree', exact: true }).click();
    assert.equal(await search.locator('svg').getAttribute('aria-label'), preservedTree, 'invalid target keeps preset rebuild atomic');
    await search.getByRole('button', { name: 'Reset investigation', exact: true }).click();
    assert.equal(await search.getByLabel('Target key', { exact: true }).inputValue(), '7');
    assert.equal(await search.getByRole('alert').count(), 0);
    await search.getByRole('button', { name: 'Enlarge tree labels', exact: true }).focus();
    await page.keyboard.press('Enter');
    assert.equal(await search.getByRole('button', { name: 'Fit whole tree', exact: true }).getAttribute('aria-pressed'), 'true');
    const enlargedRegion = search.locator('.tree-diagram-scroll');
    await enlargedRegion.focus(); await page.keyboard.press('ArrowRight');
    assert.equal(await enlargedRegion.evaluate(element => element === document.activeElement), true);
    await screenshot(page, search, `search-enlarged-${width}.png`);
    await search.getByRole('button', { name: 'Fit whole tree', exact: true }).click();
    await search.locator('.tree-structure-details > summary').focus(); await page.keyboard.press('Enter');
    assert.equal(await search.locator('tbody tr').count(), 9);
    await search.locator('.tree-structure-details > summary').press('Enter');

    const traversal = page.locator('[data-lab="tree-traversal"]');
    const traversalStates = {};
    for (const [order, expected] of Object.entries(expectedOrders)) {
      await traversal.locator('select').selectOption(order);
      traversalStates[order] = await traceToEnd(traversal, async () => {
        const output = await traversal.locator('.tree-output strong').allTextContents();
        assert.deepEqual(output.map(Number), expected.slice(0, output.length));
        assert.equal(new Set(output).size, output.length);
      });
      assert.deepEqual((await traversal.locator('.tree-output strong').allTextContents()).map(Number), expected);
      assert.equal(await traversal.locator('.tree-frontier > li').count(), 0);
      await traversal.getByRole('button', { name: 'Back', exact: true }).click();
      assert.equal(await traversal.locator('.tree-trace-controls button').nth(1).isEnabled(), true);
    }
    await traversal.getByRole('button', { name: 'Reset investigation', exact: true }).click();
    for (let index = 0; index < 4; index++) await traversal.locator('.tree-trace-controls button').nth(1).click();
    await screenshot(page, traversal, `traversal-stack-${width}.png`);
    await traversal.locator('select').selectOption('level-order');
    for (let index = 0; index < 6; index++) await traversal.locator('.tree-trace-controls button').nth(1).click();
    await screenshot(page, traversal, `traversal-queue-${width}.png`);

    const deletion = page.locator('[data-lab="tree-deletion"]');
    const deletionCases = [];
    for (const fixture of [
      { preset: 'sample', key: 8, values: sample, removed: 'n3', root: 'n1' },
      { preset: 'sample', key: 1, values: sample, removed: 'n4', root: 'n1' },
      { preset: 'sample', key: 14, values: sample, removed: 'n6', root: 'n1' },
      { preset: 'sample', key: 3, values: sample, removed: 'n7', root: 'n1' },
      { preset: 'sample', key: 99, values: sample, removed: null, root: 'n1' },
      { preset: 'successor', key: 20, values: [20, 10, 40, 30, 50, 35], removed: 'n4', root: 'n1' },
      { preset: 'single', key: 8, values: [8], removed: 'n1', root: null },
      { preset: 'one', key: 8, values: [8, 3], removed: 'n1', root: 'n2' },
      { preset: 'empty', key: 8, values: [], removed: null, root: null },
    ]) {
      await deletion.locator('select').selectOption(fixture.preset);
      await deletion.getByLabel('Key to delete', { exact: true }).fill(String(fixture.key));
      await deletion.getByRole('button', { name: 'Trace deletion', exact: true }).click();
      let temporaryDuplicates = 0;
      const steps = await traceToEnd(deletion, async () => {
        assert.ok((await deletion.locator('.tree-node-id').allTextContents()).every(label => /^n\d+$/.test(label)), 'short identity labels avoid child-edge intersections');
        if (await deletion.locator('.tree-feedback.is-transient').count()) {
          temporaryDuplicates++;
          assert.match(await deletion.locator('.tree-transient-note').innerText(), /duplicate key/);
          if (fixture.preset === 'successor') await screenshot(page, deletion, `delete-intermediate-${width}.png`);
        }
      });
      const rows = await deletion.locator('tbody tr').evaluateAll(nodes => nodes.map(node => [...node.querySelectorAll('th,td')].map(cell => cell.textContent)));
      assert.deepEqual(rows.map(row => Number(row[1])).sort((a, b) => a - b), fixture.values.filter(key => key !== fixture.key).sort((a, b) => a - b));
      assert.match(await deletion.locator('table caption').textContent(), new RegExp('root → ' + (fixture.root || 'empty')));
      assert.equal(rows.some(row => row[0] === fixture.removed), false);
      assert.match(await deletion.locator('.tree-validity').innerText(), /ordering holds/);
      if (fixture.preset === 'successor') {
        assert.equal(rows.find(row => row[0] === 'n3')[2], 'n6', '35 reconnects through 40.left');
        assert.equal(rows.find(row => row[0] === 'n1')[1], '30', 'target object holds successor key');
        assert.equal(temporaryDuplicates, 1);
        await screenshot(page, deletion, `delete-reconnected-${width}.png`);
      }
      deletionCases.push({ preset: fixture.preset, key: fixture.key, steps, temporaryDuplicates });
    }
    await deletion.getByLabel('Key to delete', { exact: true }).fill('x');
    await deletion.getByRole('button', { name: 'Trace deletion', exact: true }).click();
    assert.equal(await deletion.getByRole('alert').count(), 1);
    await deletion.getByRole('button', { name: 'Reset investigation', exact: true }).click();
    assert.equal(await deletion.locator('select').inputValue(), 'sample');
    assert.equal(await deletion.getByRole('alert').count(), 0);

    const visualMetrics = [];
    for (let index = 0; index < await page.locator('.tree-inline-figure').count(); index++) {
      const figure = page.locator('.tree-inline-figure').nth(index);
      await screenshot(page, figure, `inline-${index + 1}-${width}.png`);
      visualMetrics.push(await figure.locator('svg text').evaluateAll(nodes => nodes.map(node => {
        const matrix = node.getScreenCTM();
        return { text: node.textContent, renderedFontSize: parseFloat(getComputedStyle(node).fontSize) * matrix.a };
      })));
    }
    // Open local worked solutions only after verifying progressive disclosure defaults.
    await page.locator('.lesson-pilot').evaluate(element => element.querySelectorAll('details').forEach(details => details.open = true));
    assert.ok((await page.locator('.python-example pre').allTextContents()).every(text => text.trim().length > 0));
    assert.ok(await page.locator('.lesson-sources a').count() >= 5);
    for (const lab of [search, traversal, deletion]) {
      const control = lab.locator('input,select,button').first();
      await control.focus(); await page.keyboard.press('Tab');
      assert.equal(await lab.evaluate(element => element.contains(document.activeElement)), true);
      const sizes = await lab.locator('input,select,button').evaluateAll(nodes => nodes.map(node => node.getBoundingClientRect().height));
      assert.ok(sizes.every(height => height >= 43), `touch targets: ${sizes}`);
      assert.equal(await lab.evaluate(element => element.scrollWidth > element.clientWidth + 2), false);
    }
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 2), false, 'page overflow');
    results.push({ width, searchStates, traversalStates, deletionCases, visualMetrics, anchorCount: anchors.length, displayedPrograms: 10, linkedProblems: 10 });
    console.log(`Trees ${width}px interactions and checks passed.`);
    await page.close();
  }
  assert.deepEqual(errors, []);
  fs.writeFileSync(path.join(directory, 'results.json'), JSON.stringify({ date: new Date().toISOString(), results, errors }, null, 2));
  await browser.close();
  console.log('Trees browser checks passed at1440px and390px: search/insert/invalid/reset/limits, all traversal steps,9 deletion cases, identity/link readouts, keyboard disclosures,10 programs,10 practice links,9 anchors and no overflow/errors.');
})().catch(error => { console.error(error); process.exit(1); });

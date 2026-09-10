const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const directory = path.resolve(__dirname, '../scratch/union-find-lesson-review');
fs.mkdirSync(directory, { recursive: true });

function components(n, edges) {
  const neighbors = Array.from({ length: n }, () => []);
  for (const [a, b] of edges) {
    neighbors[a].push(b);
    neighbors[b].push(a);
  }
  const seen = new Set();
  const groups = [];
  for (let node = 0; node < n; node++) {
    if (seen.has(node)) continue;
    seen.add(node);
    const queue = [node];
    for (let i = 0; i < queue.length; i++) {
      for (const neighbor of neighbors[queue[i]]) {
        if (!seen.has(neighbor)) {
          seen.add(neighbor);
          queue.push(neighbor);
        }
      }
    }
    groups.push(queue.sort((a, b) => a - b));
  }
  return groups;
}

async function table(lab) {
  return lab.locator('tbody tr').evaluateAll(rows => rows.map(row => [...row.cells].map(cell => cell.textContent.trim())));
}

async function checkForest(lab, expected) {
  const rows = await table(lab);
  const grouped = new Map();
  for (const row of rows) {
    const item = Number(row[0]);
    const representative = Number(row[2]);
    if (!grouped.has(representative)) grouped.set(representative, []);
    grouped.get(representative).push(item);
    let current = item;
    const seen = new Set();
    while (Number(rows[current][1]) !== current) {
      assert.ok(!seen.has(current), 'parent forest must not cycle');
      seen.add(current);
      current = Number(rows[current][1]);
    }
    assert.equal(current, representative);
  }
  const actual = [...grouped.values()].sort((a, b) => a[0] - b[0]);
  if (expected) assert.deepEqual(actual, expected);
  for (const [root, group] of grouped) assert.equal(Number(rows[root][3]), group.length);
  assert.equal(await lab.locator('svg line').count(), rows.length - grouped.size);
  return actual;
}

async function finish(lab, expectedDuring) {
  let states = 0;
  while (true) {
    await checkForest(lab, expectedDuring);
    states++;
    const button = lab.getByRole('button', { name: 'Next step', exact: true });
    if (await button.isDisabled()) return states;
    assert.ok(states < 50);
    await button.click();
  }
}

async function capture(page, target, filename) {
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = 'hidden'));
  await target.screenshot({ path: path.join(directory, filename) });
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = ''));
}

function gridCount(active) {
  const unseen = new Set(active);
  let count = 0;
  while (unseen.size) {
    count++;
    const start = unseen.values().next().value;
    unseen.delete(start);
    const queue = [start];
    for (let i = 0; i < queue.length; i++) {
      for (const other of [...unseen]) {
        const node = queue[i];
        if (Math.abs(Math.floor(other / 5) - Math.floor(node / 5)) + Math.abs(other % 5 - node % 5) === 1) {
          unseen.delete(other);
          queue.push(other);
        }
      }
    }
  }
  return count;
}

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  const errors = [];
  for (const width of [1440, 390, 320]) {
    const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(process.env.LESSON_URL || 'http://127.0.0.1:5173/learn/path/full-curriculum/disjoint-sets-union-find?module=data-structures-algorithms');
    const unions = page.locator('[data-union-find-lab="unions"]');
    await unions.waitFor({ timeout: 60000 });
    const compression = page.locator('[data-union-find-lab="compression"]');
    const grid = page.locator('[data-union-find-lab="islands"]');
    assert.equal(await page.locator('[data-union-find-lab]').count(), 3);
    assert.equal(await page.locator('.python-example').count(), 10);
    assert.equal(await page.locator('[data-practice-topic="disjoint-sets-union-find"] a[href^="https://leetcode.com/problems/"]').count(), 5);
    const anchors = await page.locator('.lesson-intro a[href^="#"]').evaluateAll(nodes => nodes.map(node => node.hash.slice(1)));
    for (const id of anchors) assert.equal(await page.locator(`[id="${id}"]`).count(), 1, id);
    await capture(page, page.locator('.uf-inline'), `equivalent-forests-${width}.png`);
    await capture(page, unions, `union-singletons-${width}.png`);

    let unionStates = 0;
    const inputEdges = [[0, 1], [2, 3], [1, 2], [0, 3], [3, 3], [6, 7], [3, 6]];
    const edges = [];
    for (const [a, b] of inputEdges) {
      await unions.getByLabel('First element', { exact: true }).fill(String(a));
      await unions.getByLabel('Second element', { exact: true }).fill(String(b));
      await unions.getByRole('button', { name: 'Apply union', exact: true }).click();
      unionStates += await finish(unions);
      edges.push([a, b]);
      await checkForest(unions, components(8, edges));
    }
    await unions.getByLabel('First element', { exact: true }).fill('1.5');
    await unions.getByRole('button', { name: 'Apply union', exact: true }).click();
    assert.match(await unions.getByRole('alert').innerText(), /whole-number/);
    await checkForest(unions, components(8, edges));
    await unions.getByLabel('First element', { exact: true }).fill('8');
    await unions.getByRole('button', { name: 'Apply union', exact: true }).click();
    assert.match(await unions.getByRole('alert').innerText(), /0 to 7/);
    await unions.getByLabel('Attachment policy', { exact: true }).selectOption('unweighted');
    await unions.getByRole('button', { name: 'Consecutive links', exact: true }).click();
    assert.match(await unions.locator('.uf-readout').innerText(), /depth: 7 edges/);
    await capture(page, unions, `chain-policy-${width}.png`);
    await unions.getByLabel('Attachment policy', { exact: true }).selectOption('size');
    await unions.getByRole('button', { name: 'Consecutive links', exact: true }).click();
    assert.match(await unions.locator('.uf-readout').innerText(), /depth: 1 edges/);
    await unions.getByRole('button', { name: 'Reset singletons', exact: true }).focus();
    await page.keyboard.press('Enter');
    await checkForest(unions, components(8, []));
    const scroll = unions.locator('.uf-forest');
    await scroll.focus();
    assert.equal(await scroll.evaluate(node => getComputedStyle(node).outlineStyle), 'solid');
    if (width < 400) {
      await page.keyboard.press('ArrowRight');
      await page.waitForTimeout(200);
      assert.ok(await scroll.evaluate(node => node.scrollLeft > 0));
    }
    let compressionStates = 0;
    for (let node = 0; node < 8; node++) {
      await compression.getByLabel('Find element', { exact: true }).selectOption(String(node));
      await compression.getByRole('button', { name: 'Trace selected find', exact: true }).click();
      compressionStates += await finish(compression, [[0, 1, 2, 3, 4, 5, 6, 7]]);
      const rows = await table(compression);
      assert.equal(rows[node][1], '0');
    }
    await capture(page, compression, `compression-result-${width}.png`);
    await compression.getByRole('button', { name: 'Find again on result', exact: true }).click();
    assert.match(await compression.locator('.uf-compression-path').innerText(), /1 upward hops/);
    await finish(compression, [[0, 1, 2, 3, 4, 5, 6, 7]]);
    await compression.getByRole('button', { name: 'Reset balanced seed', exact: true }).click();
    assert.match(await compression.locator('.uf-compression-path').innerText(), /7 → 6 → 4 → 0/);
    await capture(page, compression, `compression-before-${width}.png`);

    const active = new Set();
    for (const index of [0, 2, 1, 6, 1, 24, 23, 18, 17, 12, 7, 8]) {
      await grid.locator('.uf-cell').nth(index).click();
      active.add(index);
      assert.equal(Number(await grid.locator('.uf-island-count strong').innerText()), gridCount(active));
    }
    await grid.getByRole('button', { name: 'Load ring around center', exact: true }).click();
    assert.equal(await grid.locator('.uf-island-count strong').innerText(), '1');
    const center = grid.getByRole('button', { name: 'Cell 2, 2: closed; open cell', exact: true });
    await center.focus();
    await page.keyboard.press('Space');
    assert.equal(await grid.locator('.uf-island-count strong').innerText(), '1');
    assert.match(await grid.locator('.uf-status').innerText(), /− 1 successful joins = 1/);
    await capture(page, grid, `island-ring-${width}.png`);
    await grid.getByRole('button', { name: 'Reset grid', exact: true }).click();
    assert.equal(await grid.locator('.uf-island-count strong').innerText(), '0');
    assert.equal(await grid.locator('[aria-pressed="true"]').count(), 0);

    const practice = page.locator('[data-practice-topic="disjoint-sets-union-find"]');
    assert.equal(await practice.locator('details[open]').count(), 0);
    const hint = practice.locator('summary').first();
    await hint.focus();
    await page.keyboard.press('Enter');
    assert.equal(await hint.evaluate(node => node.parentElement.open), true);
    await page.keyboard.press('Enter');
    await capture(page, practice.locator('.dsa-practice__stage').first(), `practice-${width}.png`);
    const overflow = await page.evaluate(() => ({ width: innerWidth, scroll: document.documentElement.scrollWidth }));
    assert.ok(overflow.scroll <= overflow.width + 1, JSON.stringify(overflow));
    results.push({ width, unionStates, compressionStates, gridEvents: 14, anchors: anchors.length, overflow, keyboard: 'forms, reset, local SVG scroll, grid and hints passed' });
    await page.close();
  }
  assert.deepEqual(errors, []);
  fs.writeFileSync(path.join(directory, 'results.json'), JSON.stringify({ results, errors }, null, 2));
  await browser.close();
  console.log(JSON.stringify({ results, errors }, null, 2));
})().catch(error => { console.error(error); process.exit(1); });

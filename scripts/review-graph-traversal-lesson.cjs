const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const directory = path.resolve(__dirname, '../scratch/graph-traversal-lesson-review');
fs.mkdirSync(directory, { recursive: true });
const vertices = [...'ABCDEFGH'];
const edges = [['A', 'B'], ['A', 'C'], ['B', 'D'], ['C', 'D'], ['D', 'E'], ['F', 'G']];

function graphOracle(inputEdges, directed) {
  const matrix = vertices.map(() => vertices.map(() => 0));
  for (const [from, to] of inputEdges) {
    matrix[vertices.indexOf(from)][vertices.indexOf(to)] = 1;
    if (!directed) matrix[vertices.indexOf(to)][vertices.indexOf(from)] = 1;
  }
  const distance = matrix.map((row, index) => row.map((value, column) => index === column ? 0 : value ? 1 : Infinity));
  for (let middle = 0; middle < 8; middle++) for (let from = 0; from < 8; from++) for (let to = 0; to < 8; to++) distance[from][to] = Math.min(distance[from][to], distance[from][middle] + distance[middle][to]);
  const neighbors = Object.fromEntries(vertices.map((vertex, index) => [vertex, vertices.filter((_, column) => matrix[index][column])]));
  return { matrix, distance, neighbors };
}

function recursiveOrders(neighbors, source) {
  const entered = [], finished = [], seen = new Set();
  const visit = vertex => { seen.add(vertex); entered.push(vertex); for (const neighbor of neighbors[vertex]) if (!seen.has(neighbor)) visit(neighbor); finished.push(vertex); };
  visit(source); return { entered, finished };
}

async function finish(lab, check = async () => {}) {
  let states = 0;
  while (true) {
    await check(); states++;
    const next = lab.locator('.graph-trace-controls button').nth(1);
    if (await next.isDisabled()) return states;
    assert.ok(states < 160); await next.click();
  }
}

async function capture(page, element, name) {
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = 'hidden'));
  await element.screenshot({ path: path.join(directory, name) });
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = ''));
}

async function gridReference(lab) {
  const cells = await lab.locator('.grid-wavefront-cell').evaluateAll(nodes => nodes.map(node => ({ wall: node.classList.contains('is-wall'), source: node.classList.contains('is-source') })));
  const distance = cells.map((cell, index) => cells.map((other, next) => index === next ? 0 : !cell.wall && !other.wall && Math.abs(Math.floor(index / 5) - Math.floor(next / 5)) + Math.abs(index % 5 - next % 5) === 1 ? 1 : Infinity));
  for (let middle = 0; middle < 25; middle++) for (let from = 0; from < 25; from++) for (let to = 0; to < 25; to++) distance[from][to] = Math.min(distance[from][to], distance[from][middle] + distance[middle][to]);
  const sources = cells.flatMap((cell, index) => cell.source ? [index] : []);
  return cells.map((cell, index) => cell.wall ? '#' : Math.min(...sources.map(source => distance[source][index])));
}

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [], errors = [];
  for (const width of [1440, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(process.env.LESSON_URL || 'http://127.0.0.1:5173/learn/path/full-curriculum/graphs-representations-bfs-dfs');
    const representation = page.locator('[data-lab="graph-representation"]'); await representation.waitFor();
    const search = page.locator('[data-lab="graph-search"]'), grid = page.locator('[data-lab="grid-wavefront"]');
    assert.equal(await page.locator('[data-lab]').count(), 3);
    assert.equal(await page.locator('.python-example').count(), 10);
    const practice = page.locator('[data-practice-topic="graphs-representations-bfs-dfs"]');
    assert.equal(await practice.locator('details[open]').count(), 0);
    assert.equal(await practice.locator('a[href^="https://leetcode.com/problems/"]').count(), 10);
    assert.equal(await practice.locator('a:not([rel="noopener noreferrer"])').count(), 0);
    const extension = practice.locator('.dsa-practice__extension > summary'); await extension.focus(); await page.keyboard.press('Enter');
    assert.equal(await extension.evaluate(element => element.parentElement.open), true);
    const hint = practice.locator('.dsa-practice__problem details > summary').first(); await hint.focus(); await page.keyboard.press('Space');
    assert.equal(await hint.evaluate(element => element.parentElement.open), true); await page.keyboard.press('Space');
    assert.equal(await hint.evaluate(element => element.parentElement.open), false);
    await capture(page, practice.locator('.dsa-practice__stage').first(), `practice-${width}.png`);
    const anchors = await page.locator('.lesson-intro a[href^="#"]').evaluateAll(links => links.map(link => link.hash.slice(1)));
    assert.equal(anchors.length, 9); for (const anchor of anchors) assert.equal(await page.locator(`[id="${anchor}"]`).count(), 1);
    await page.locator('.lesson-intro a[href="#guided-dsa-practice"]').click(); assert.equal(new URL(page.url()).hash, '#guided-dsa-practice');
    assert.ok(await page.locator('a[href*="/disjoint-sets-union-find"]').count() >= 1);

    const representationCases = [];
    for (const fixture of [
      { edges, directed: false }, { edges, directed: true },
      { edges: [['A', 'B'], ['B', 'A'], ['A', 'A'], ['A', 'B']], directed: true },
      { edges: [['A', 'B'], ['B', 'A'], ['A', 'A']], directed: false },
      { edges: [], directed: false },
      { edges: [['H', 'H'], ['B', 'B'], ['B', 'C'], ['C', 'B']], directed: true },
    ]) {
      await representation.getByLabel('Edges · comma-separated, such as A-B', { exact: true }).fill(fixture.edges.map(edge => edge.join('-')).join(','));
      await representation.locator('select').selectOption(fixture.directed ? 'directed' : 'undirected');
      await representation.getByRole('button', { name: 'Apply graph', exact: true }).click();
      const reference = graphOracle(fixture.edges, fixture.directed);
      const matrix = await representation.locator('.graph-adjacency-matrix tbody tr').evaluateAll(rows => rows.map(row => [...row.querySelectorAll('td')].map(cell => Number(cell.textContent))));
      assert.deepEqual(matrix, reference.matrix);
      assert.equal(await representation.locator('.graph-vertex').count(), 8);
      for (const vertex of vertices) {
        const selected = representation.getByRole('button', { name: `Inspect outgoing neighbors of ${vertex}`, exact: true });
        await selected.click(); assert.equal(await selected.getAttribute('aria-pressed'), 'true');
        assert.equal((await representation.locator('.graph-selected-adjacency strong').nth(1).innerText()).trim(), reference.neighbors[vertex].join(', ') || 'no vertex');
      }
      const graphEdges = await representation.locator('.graph-edge').evaluateAll(nodes => nodes.map(node => ({ curve: node.getAttribute('d'), arrow: node.getAttribute('marker-end') })));
      assert.ok(graphEdges.every(edge => fixture.directed ? Boolean(edge.arrow) : !edge.arrow));
      if (fixture.edges.length === 4 && fixture.edges[0][0] === 'A') {
        assert.equal(graphEdges.filter(edge => edge.curve.includes('Q')).length, 2);
        await capture(page, representation, `direction-reciprocal-loop-${width}.png`);
      }
      representationCases.push({ directed: fixture.directed, edges: fixture.edges, drawnEdges: graphEdges.length });
    }
    const savedGraph = await representation.locator('svg').getAttribute('aria-label');
    for (const invalid of ['A-I', 'a-b', 'A-B,,C-D', Array(25).fill('A-B').join(',')]) {
      await representation.getByLabel('Edges · comma-separated, such as A-B', { exact: true }).fill(invalid);
      await representation.getByRole('button', { name: 'Apply graph', exact: true }).click();
      assert.equal(await representation.getByRole('alert').count(), 1); assert.equal(await representation.locator('svg').getAttribute('aria-label'), savedGraph);
    }
    await representation.getByRole('button', { name: 'Reset graph', exact: true }).click(); assert.equal(await representation.getByRole('alert').count(), 0);

    const searchCases = [];
    for (const fixture of [
      { method: 'bfs', source: 'A', directed: false }, { method: 'dfs', source: 'A', directed: false },
      { method: 'bfs', source: 'A', directed: true }, { method: 'dfs', source: 'A', directed: true },
      { method: 'bfs', source: 'F', directed: false }, { method: 'dfs', source: 'H', directed: false },
      { method: 'bfs', source: 'E', directed: true }, { method: 'bfs', source: 'E', directed: false },
    ]) {
      await search.locator('.graph-controls select').nth(0).selectOption(fixture.method);
      await search.locator('.graph-controls select').nth(1).selectOption(fixture.source);
      await search.locator('.graph-controls select').nth(2).selectOption(fixture.directed ? 'directed' : 'undirected');
      await search.getByRole('button', { name: 'Start selected search', exact: true }).click();
      const reference = graphOracle(edges, fixture.directed), traversal = recursiveOrders(reference.neighbors, fixture.source);
      let capturedPending = false;
      const states = await finish(search, async () => {
        const rows = await search.locator('.graph-search-table tbody tr').evaluateAll(nodes => nodes.map(node => [...node.querySelectorAll('th,td')].map(cell => cell.textContent)));
        const byVertex = Object.fromEntries(rows.map(row => [row[0], row]));
        const drawLabels = await search.locator('.graph-vertex-state').allTextContents();
        for (let index = 0; index < 8; index++) {
          const row = rows[index], vertex = row[0], reached = row.at(-1) !== 'unreached';
          if (reached) {
            assert.ok(reference.distance[vertices.indexOf(fixture.source)][index] < Infinity);
            if (vertex !== fixture.source) assert.ok(reference.neighbors[row[1]].includes(vertex));
            if (fixture.method === 'bfs') assert.equal(Number(row[2]), reference.distance[vertices.indexOf(fixture.source)][index]);
          }
          if (fixture.method === 'bfs') assert.equal(drawLabels[index], reached ? `d = ${row[2]}` : 'unreached');
        }
        const queued = await search.locator('.graph-queue strong').allTextContents();
        assert.equal(queued.length, new Set(queued).size);
        if (fixture.method === 'bfs') assert.deepEqual(queued.map(vertex => Number(byVertex[vertex][2])), queued.map(vertex => Number(byVertex[vertex][2])).sort((a, b) => a - b));
        const entryText = await search.locator('.graph-order').first().innerText();
        const entered = entryText.split(' → ');
        if (fixture.method === 'dfs') assert.deepEqual(entered, traversal.entered.slice(0, entered.length));
        if (!capturedPending && fixture.source === 'A' && !fixture.directed && (queued.length === 2 || await search.locator('.graph-frame-stack > li').count() === 3)) {
          await capture(page, search, `${fixture.method}-pending-${width}.png`); capturedPending = true;
        }
      });
      const entered = (await search.locator('.graph-order').first().innerText()).split(' → ');
      assert.deepEqual([...entered].sort(), traversal.entered.slice().sort());
      if (fixture.method === 'dfs') assert.equal(await search.locator('.graph-order').nth(1).innerText(), traversal.finished.join(' → '));
      assert.equal(await search.locator('.graph-queue > li, .graph-frame-stack > li').count(), 0);
      await search.locator('.graph-route-control select').selectOption(fixture.source);
      assert.match(await search.locator('.graph-route-result').innerText(), /0 edges/);
      if (fixture.source === 'A') {
        await search.locator('.graph-route-control select').selectOption('H'); assert.match(await search.locator('.graph-route-result').innerText(), /unreachable/);
        await search.locator('.graph-route-control select').selectOption('C');
        if (!fixture.directed) assert.match(await search.locator('.graph-route-result').innerText(), fixture.method === 'bfs' ? /A → C · 1 edge/ : /A → B → D → C · 3 edges/);
      }
      await search.getByRole('button', { name: 'Back', exact: true }).click(); assert.equal(await search.locator('.graph-trace-controls button').nth(1).isEnabled(), true);
      searchCases.push({ ...fixture, states });
    }
    await search.getByRole('button', { name: 'Reset investigation', exact: true }).click();
    assert.equal(await search.locator('.graph-controls select').nth(0).inputValue(), 'bfs');

    const gridCases = [];
    async function verifyGrid(name, targetResult) {
      const expected = await gridReference(grid);
      const states = await finish(grid, async () => {
        const values = await grid.locator('.grid-wavefront-cell strong').allTextContents();
        for (let index = 0; index < 25; index++) {
          if (expected[index] === '#') assert.equal(values[index], '#');
          else if (values[index] !== '·') assert.equal(Number(values[index]), expected[index]);
        }
      });
      const values = await grid.locator('.grid-wavefront-cell strong').allTextContents();
      assert.deepEqual(values, expected.map(value => value === '#' ? '#' : value === Infinity ? '·' : String(value)));
      assert.equal((await grid.locator('.grid-distance-result > strong').innerText()).trim(), targetResult);
      gridCases.push({ name, states, targetResult });
    }
    const cell = (row, column) => grid.getByRole('button', { name: new RegExp(`^Row ${row}, column ${column}:`) });
    await verifyGrid('default single source', '8 moves'); await capture(page, grid, `grid-single-${width}.png`);
    await grid.locator('.graph-controls select').nth(0).selectOption('multiple'); await verifyGrid('target is a source', '0 moves');
    await grid.locator('.graph-controls select').nth(1).selectOption('target'); await cell(0, 4).click();
    await verifyGrid('nearest of two sources', '4 moves'); await capture(page, grid, `grid-multiple-${width}.png`);
    await grid.locator('.graph-controls select').nth(0).selectOption('single'); await verifyGrid('same target from one source', '8 moves');
    await cell(0, 3).click(); assert.equal(await grid.getByRole('alert').count(), 1, 'blocked target rejects');
    await grid.locator('.graph-controls select').nth(1).selectOption('walls');
    await cell(0, 1).click(); await cell(1, 0).click(); await verifyGrid('isolated source', 'unreachable');
    await capture(page, grid, `grid-unreachable-${width}.png`);
    await cell(0, 0).click(); assert.equal(await grid.getByRole('alert').count(), 1, 'source must remain open');
    await cell(0, 4).click(); assert.equal(await grid.getByRole('alert').count(), 1, 'target must remain open');
    await grid.getByRole('button', { name: 'Clear all walls', exact: true }).click(); await verifyGrid('wall-free Manhattan comparison', '4 moves');
    await cell(4, 4).focus(); await page.keyboard.press('Space'); assert.equal(await cell(4, 4).getAttribute('aria-pressed'), 'true');
    await grid.locator('.graph-controls select').nth(0).selectOption('multiple'); assert.equal(await cell(4, 4).getAttribute('aria-pressed'), 'false', 'new source is opened');
    await grid.locator('.graph-controls select').nth(1).selectOption('target'); await cell(0, 0).focus(); await page.keyboard.press('Enter');
    await verifyGrid('source equals target by keyboard', '0 moves');
    await grid.getByRole('button', { name: 'Reset investigation', exact: true }).click();
    assert.equal(await grid.locator('.graph-controls select').nth(0).inputValue(), 'single');
    assert.equal(await grid.getByRole('alert').count(), 0);

    const cycleFigure = page.locator('.graph-inline-figure');
    for (const region of await cycleFigure.locator('.graph-diagram-scroll').all()) {
      const geometry = await region.evaluate(element => {
        const bounds = element.getBoundingClientRect();
        return { overflow: element.scrollWidth > element.clientWidth + 1,
          nodesFit: [...element.querySelectorAll('.graph-vertex')].every(node => {
            const box = node.getBoundingClientRect();
            return box.left >= bounds.left && box.right <= bounds.right;
          }) };
      });
      assert.equal(geometry.overflow, false, 'small cycle contrast fits without panning');
      assert.equal(geometry.nodesFit, true, 'all four cycle-contrast nodes fit together');
    }
    await capture(page, cycleFigure, `cycle-contrast-${width}.png`);
    const cloneFigure = page.locator('figure').filter({ hasText: 'Preserve sharing and cycles; replace object identity' });
    assert.equal(await cloneFigure.locator('svg').count(), 2); await capture(page, cloneFigure, `clone-identity-${width}.png`);
    for (const lab of [representation, search, grid]) {
      const control = lab.locator('input,select,button').first(); await control.focus(); await page.keyboard.press('Tab');
      assert.equal(await lab.evaluate(element => element.contains(document.activeElement)), true);
      const sizes = await lab.locator('input,select,button').evaluateAll(nodes => nodes.map(node => node.getBoundingClientRect().height));
      assert.ok(sizes.every(height => height >= 43), `native targets ${sizes}`);
      assert.equal(await lab.evaluate(element => element.scrollWidth > element.clientWidth + 2), false);
    }
    for (const lab of [representation, search]) {
      const region = lab.locator('.graph-diagram-scroll'); await region.focus(); await page.keyboard.press('ArrowRight');
      assert.equal(await region.evaluate(element => element === document.activeElement), true);
      if (width === 390) await page.waitForFunction(element => element.scrollLeft > 0, await region.elementHandle());
    }
    await page.locator('.lesson-pilot').evaluate(element => element.querySelectorAll('details').forEach(details => details.open = true));
    assert.ok((await page.locator('.python-example pre').allTextContents()).every(text => text.trim()));
    assert.ok(await page.locator('.lesson-sources a').count() >= 5);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 2), false);
    results.push({ width, representationCases, searchCases, gridCases, programs: 10, practiceLinks: 10, anchors: anchors.length });
    console.log(`Graphs ${width}px integrated checks passed.`); await page.close();
  }
  assert.deepEqual(errors, []);
  fs.writeFileSync(path.join(directory, 'results.json'), JSON.stringify({ date: new Date().toISOString(), results, errors }, null, 2));
  await browser.close(); console.log('Graphs: representation, directed edges, BFS/DFS states/routes, grids, invalid/reset, keyboard, code, practice, anchors and no overflow/errors passed.');
})().catch(error => { console.error(error); process.exit(1); });

const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const directory = path.resolve(__dirname, '../scratch/weighted-graph-lesson-review');
fs.mkdirSync(directory, { recursive: true });

async function capture(page, target, filename) {
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => { node.style.visibility = 'hidden'; }));
  await target.screenshot({ path: path.join(directory, filename) });
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => { node.style.visibility = ''; }));
}
async function button(lab, name) { await lab.getByRole('button', { name, exact: true }).click(); }
async function trace(lab, inspect = async () => {}) {
  let count = 0;
  while (true) {
    await inspect();
    count++;
    const next = lab.getByRole('button', { name: 'Next step', exact: true });
    if (await next.isDisabled()) return count;
    assert(count < 150, 'Finite bounded trace');
    await next.click();
  }
}
async function edit(lab, text) {
  const disclosure = lab.locator('.weighted-graph-editor');
  if (!(await disclosure.evaluate(node => node.open))) await disclosure.locator('summary').click();
  await lab.getByLabel('Edge draft', { exact: true }).fill(text);
  await button(lab, 'Apply graph');
}
async function reset(lab) {
  const disclosure = lab.locator('.weighted-graph-editor');
  if (!(await disclosure.evaluate(node => node.open))) await disclosure.locator('summary').click();
  await button(lab, 'Reset example');
  await disclosure.locator('summary').click();
}

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  for (const width of [1440, 390, 320]) {
    const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
    await page.routeWebSocket('**', socket => socket.close());
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(process.env.LESSON_URL || 'http://127.0.0.1:5173/learn/path/full-curriculum/shortest-paths-spanning-trees-topological-ordering?module=data-structures-algorithms');
    const dijkstra = page.locator('section[aria-label="Dijkstra priority frontier investigation"]');
    await dijkstra.waitFor({ timeout: 60000 });
    const bellman = page.locator('section[aria-label="Bellman Ford edge budget investigation"]');
    const forest = page.locator('section[aria-label="Minimum spanning forest investigation"]');
    const topo = page.locator('section[aria-label="Topological ready frontier investigation"]');
    assert.equal(await page.locator('.python-example').count(), 12);
    assert.equal(await page.locator('.weighted-graph-lab').count(), 4);
    await page.screenshot({ path: path.join(directory, `reading-entry-${width}.png`) });
    const anchors = await page.locator('.lesson-intro a[href^="#"]').evaluateAll(nodes => nodes.map(node => node.hash.slice(1)));
    for (const anchor of anchors) {
      assert.equal(await page.locator(`[id="${anchor}"]`).count(), 1);
      await page.locator(`.lesson-intro a[href="#${anchor}"]`).click();
      await page.waitForFunction(id => { const top = document.getElementById(id).getBoundingClientRect().top; return top >= -1 && top < innerHeight; }, anchor);
    }
    await capture(page, page.locator('.weighted-graph-objectives'), `objectives-${width}.png`);
    await capture(page, page.locator('.weighted-graph-figure').last(), `critical-timeline-${width}.png`);
    const counts = {};
    let sawStale = false;
    counts.dijkstra = await trace(dijkstra, async () => {
      const message = await dijkstra.getByRole('status').innerText();
      if (message.startsWith('C → B:')) {
        assert.equal(await dijkstra.locator('.weighted-graph-frontier li').filter({ hasText: 'B' }).count(), 2);
        await capture(page, dijkstra, `dijkstra-improvement-${width}.png`);
      }
      if (message.startsWith('Discard B at 10')) sawStale = true;
    });
    assert(sawStale);
    assert.match(await dijkstra.innerText(), /A → C → B → D → E; cost 7/);
    await capture(page, dijkstra, `dijkstra-final-${width}.png`);
    await dijkstra.getByLabel('Route target', { exact: true }).selectOption('5');
    assert.match(await dijkstra.innerText(), /unreachable from this source/);
    await dijkstra.getByLabel('Source', { exact: true }).selectOption('5');
    counts.dijkstra += await trace(dijkstra);
    assert.match(await dijkstra.innerText(), /Selected target F: F; cost 0/);
    await edit(dijkstra, 'A B -1');
    assert.match(await dijkstra.getByRole('alert').innerText(), /nonnegative/);
    await reset(dijkstra);
    await edit(dijkstra, 'A B 0\nB C 0\nC A 0');
    counts.dijkstra += await trace(dijkstra);
    assert.match(await dijkstra.getByRole('status').innerText(), /queue is empty/);
    await reset(dijkstra);
    await dijkstra.getByRole('button', { name: 'Next step', exact: true }).focus();
    await page.keyboard.press('Enter');
    assert.match(await dijkstra.getByRole('status').innerText(), /Finalize A/);
    await dijkstra.getByRole('button', { name: 'Previous', exact: true }).focus();
    await page.keyboard.press('Space');
    assert.match(await dijkstra.getByRole('status').innerText(), /empty route costs 0/);

    counts.bellman = await trace(bellman);
    assert.deepEqual((await bellman.innerText()).match(/Downstream affected region: ([^.]+)\./)[1].split(', ').sort(), ['B', 'C', 'D']);
    await capture(page, bellman, `negative-cycle-region-${width}.png`);
    await bellman.getByLabel('Source', { exact: true }).selectOption('4');
    counts.bellman += await trace(bellman);
    const classification = await bellman.locator('table').first().locator('tbody tr').last().innerText();
    assert.equal((classification.match(/−∞/g) || []).length, 2);
    await edit(bellman, 'A B 2\nB C -4\nC D 2\nE F -3\nF E 1');
    await bellman.getByLabel('Source', { exact: true }).selectOption('0');
    counts.bellman += await trace(bellman);
    assert.match(await bellman.innerText(), /Downstream affected region: none/);
    await reset(bellman);

    counts.forest = await trace(forest);
    assert.match(await forest.getByRole('status').innerText(), /5 accepted edges, total 13/);
    await capture(page, forest, `kruskal-final-${width}.png`);
    await forest.getByLabel('Growth rule', { exact: true }).selectOption('prim');
    counts.forest += await trace(forest);
    assert.match(await forest.getByRole('status').innerText(), /5 accepted edges, total 13/);
    await capture(page, forest, `prim-final-${width}.png`);
    await edit(forest, 'A B -2\nC D -2');
    counts.forest += await trace(forest);
    assert.match(await forest.getByRole('status').innerText(), /2 accepted edges, total -4/);
    assert.match(await forest.innerText(), /2 edges · 4 components/);
    await edit(forest, 'A B 1\nB A 2');
    assert.equal(await forest.getByRole('alert').count(), 1);
    await edit(forest, '');
    counts.forest += await trace(forest);
    assert.match(await forest.innerText(), /0 edges · 6 components/);
    await reset(forest);

    await button(topo, 'Emit B');
    assert.equal(await topo.getByRole('button', { name: 'Emit C', exact: true }).count(), 0);
    for (const name of ['A', 'C', 'D', 'E', 'F']) await button(topo, `Emit ${name}`);
    assert.match(await topo.getByRole('status').innerText(), /Complete/);
    assert.deepEqual(await topo.locator('.weighted-graph-order li').allTextContents(), ['B', 'A', 'C', 'D', 'E', 'F']);
    await button(topo, 'Undo emission');
    assert.equal(await topo.getByRole('button', { name: 'Emit F', exact: true }).count(), 1);
    await button(topo, 'Load cycle example');
    for (const name of ['A', 'B', 'D']) await button(topo, `Emit ${name}`);
    assert.match(await topo.getByRole('status').innerText(), /Cycle witness: C → E → C.*Residual vertices: C, E, F/);
    await capture(page, topo, `topological-cycle-${width}.png`);
    await reset(topo);
    await edit(topo, '');
    assert.equal(await topo.locator('.weighted-graph-ready button').count(), 6);
    await reset(topo);
    const disclosure = topo.locator('.weighted-graph-editor > summary');
    await disclosure.focus();
    await page.keyboard.press('Enter');
    assert(await disclosure.evaluate(node => node.parentElement.open));
    await page.keyboard.press('Enter');
    assert(!(await disclosure.evaluate(node => node.parentElement.open)));

    const scroll = bellman.locator('.weighted-graph-scroll').first();
    await scroll.focus();
    assert.equal(await scroll.evaluate(node => getComputedStyle(node).outlineStyle), 'solid');
    if (await scroll.evaluate(node => node.scrollWidth > node.clientWidth)) {
      await page.keyboard.press('ArrowRight');
      await page.waitForTimeout(150);
      assert(await scroll.evaluate(node => node.scrollLeft > 0));
    }
    const hint = page.locator('.dsa-practice details summary').first();
    assert(!(await hint.evaluate(node => node.parentElement.open)));
    await hint.focus();
    await page.keyboard.press('Enter');
    assert(await hint.evaluate(node => node.parentElement.open));
    const links = await page.locator('.dsa-practice a[href*="leetcode.com/problems/"]').evaluateAll(nodes => nodes.map(node => ({ href: node.href, target: node.target, rel: node.rel })));
    assert.equal(links.length, 7);
    assert(links.every(link => link.target === '_blank' && link.rel.includes('noopener')));
    const drawings = await page.locator('.weighted-graph-picture svg').evaluateAll(nodes => nodes.map(node => ({ width: node.getBoundingClientRect().width, container: node.parentElement.parentElement.clientWidth })));
    assert(drawings.every(item => item.width <= item.container + 1), JSON.stringify(drawings));
    const overflow = await page.evaluate(() => ({ viewport: innerWidth, page: document.documentElement.scrollWidth }));
    assert.equal(overflow.page, overflow.viewport);
    assert.deepEqual(errors, []);
    results.push({ width, counts, anchors: anchors.length, programs: 12, practiceLinks: links.length, drawings, overflow, errors });
    await page.close();
  }
  await browser.close();
  fs.writeFileSync(path.join(directory, 'results.json'), JSON.stringify(results, null, 2));
  console.log(JSON.stringify(results));
})().catch(error => { console.error(error); process.exit(1); });

const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const output = path.resolve(__dirname, '../scratch/network-flow-lesson-review');
fs.mkdirSync(output, { recursive: true });

async function capture(page, target, filename) {
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => { node.style.visibility = 'hidden'; }));
  await target.screenshot({ path: path.join(output, filename) });
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => { node.style.visibility = ''; }));
}
async function click(lab, label) {
  await lab.getByRole('button', { name: label, exact: true }).click();
}
async function openDetail(lab, label) {
  const summary = lab.locator('summary').filter({ hasText: label }).first();
  const details = summary.locator('..');
  if (!(await details.evaluate(element => element.open))) await summary.click();
  return details;
}

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  for (const width of [1440, 390, 320]) {
    const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
    await page.routeWebSocket('**', socket => socket.close());
    const errors = [];
    const consoleErrors = [];
    const failedRequests = [];
    page.on('requestfailed', request => failedRequests.push({ url: request.url(), failure: request.failure() }));
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', message => { if (message.type() === 'error' && !message.text().startsWith('[vite] failed to connect to websocket.')) consoleErrors.push({ text: message.text(), location: message.location() }); });
    await page.goto(process.env.LESSON_URL || 'http://127.0.0.1:5173/learn/path/full-curriculum/network-flow-minimum-cuts-bipartite-matching?module=data-structures-algorithms');
    const conservation = page.getByRole('region', { name: 'Flow conservation investigation', exact: true });
    const augment = page.getByRole('region', { name: 'Residual augmentation investigation', exact: true });
    const matching = page.getByRole('region', { name: 'Bipartite matching and cover investigation', exact: true });
    const pixels = page.getByRole('region', { name: 'Binary labeling minimum cut investigation', exact: true });
    await conservation.waitFor({ timeout: 60000 });
    assert.equal(await page.locator('.python-example').count(), 9);
    await page.screenshot({ path: path.join(output, `reading-entry-${width}.png`) });
    const anchors = await page.locator('.lesson-intro a[href^="#"]').evaluateAll(nodes => nodes.map(node => node.hash.slice(1)));
    for (const id of anchors) {
      assert.equal(await page.locator(`[id="${id}"]`).count(), 1, id);
      await page.locator(`.lesson-intro a[href="#${id}"]`).click();
      await page.waitForFunction(anchor => { const top = document.getElementById(anchor).getBoundingClientRect().top; return top >= -1 && top < innerHeight; }, id);
    }
    await capture(page, page.getByRole('figure', { name: 'Residual edge identity figure', exact: true }), `residual-pairs-${width}.png`);
    await capture(page, page.getByRole('figure', { name: 'Vertex capacity transformation', exact: true }), `vertex-split-${width}.png`);
    await click(conservation, 'Stranded at A');
    assert.match(await conservation.getByRole('status').innerText(), /Conservation violated at A/);
    await capture(page, conservation, `conservation-invalid-${width}.png`);
    await click(conservation, 'One balanced route');
    assert.match(await conservation.getByRole('status').innerText(), /Feasible flow.*1/);
    for (const direction of ['S → A', 'A → C', 'C → T']) await conservation.getByLabel(`Proposed flow ${direction}`, { exact: true }).selectOption('2');
    assert.match(await conservation.getByRole('status').innerText(), /Capacity violated on e0, e2, e5/);
    const reset = conservation.getByRole('button', { name: 'Reset flows', exact: true });
    await reset.focus();
    await page.keyboard.press('Enter');
    assert.match(await conservation.getByRole('status').innerText(), /Feasible flow.*0/);
    const balanceTable = conservation.locator('.nf-table-scroll').last();
    await balanceTable.focus();
    assert.equal(await balanceTable.evaluate(element => document.activeElement === element), true);
    assert.ok(await balanceTable.evaluate(element => element.matches(':focus-visible') && parseFloat(getComputedStyle(element).outlineWidth) >= 2 && getComputedStyle(element).outlineStyle === 'solid'));

    await click(augment, 'Preview next path');
    assert.match(await augment.getByRole('status').innerText(), /Bottleneck 1/);
    await click(augment, 'Send bottleneck');
    await click(augment, 'Preview next path');
    assert.match(await augment.getByRole('status').innerText(), /includes cancellation/);
    assert.match(await augment.locator('table caption').innerText(), /S → B → C → A → D → T/);
    assert.match(await augment.locator('table').innerText(), /Cancel e2/);
    assert.equal(await augment.locator('svg .nf-cancel').count(), 1);
    await capture(page, augment, `augmentation-reverse-${width}.png`);
    await click(augment, 'Send bottleneck');
    assert.match(await augment.getByRole('status').innerText(), /Maximum flow = 2/);
    assert.match(await augment.locator('.nf-cut-inspector').innerText(), /both are optimal/);
    await click(augment, 'A outside');
    assert.match(await augment.locator('.nf-cut-inspector').innerText(), /looser upper bound/);
    assert.equal(await augment.locator('svg .nf-source-side').count(), 2);
    assert.equal(await augment.locator('svg .nf-cut-edge').count(), 3);
    await click(augment, 'Use residual source side');
    await capture(page, augment, `flow-cut-certificate-${width}.png`);
    await openDetail(augment, 'Edit original capacities');
    await augment.getByLabel('Capacity S → A', { exact: true }).fill('1.5');
    await click(augment, 'Apply capacities');
    assert.match(await augment.getByRole('alert').innerText(), /active network is unchanged/);
    assert.match(await augment.getByRole('status').innerText(), /Maximum flow = 2/);
    for (const field of await augment.locator('input').all()) await field.fill('0');
    await click(augment, 'Apply capacities');
    assert.match(await augment.getByRole('status').innerText(), /Maximum flow = 0/);
    assert.match(await augment.locator('.nf-cut-inspector').innerText(), /both are optimal/);
    await click(augment, 'Reset network');
    await click(augment, 'Preview next path');
    await click(augment, 'Send bottleneck');
    await click(augment, 'Back');
    assert.match(await augment.getByRole('status').innerText(), /0 augmentations applied/);

    for (let index = 0; index < 3; index++) await click(matching, 'Augment matching');
    assert.match(await matching.getByRole('status').innerText(), /Maximum matching size\s*3/);
    assert.match(await matching.innerText(), /remove A–1/);
    await click(matching, 'Reveal cover and shortage');
    await capture(page, matching, `matching-perfect-${width}.png`);
    await click(matching, 'Three workers, two neighbors');
    for (let index = 0; index < 2; index++) await click(matching, 'Augment matching');
    await click(matching, 'Reveal cover and shortage');
    assert.match(await matching.locator('.nf-cover-proof').innerText(), /3 workers.*2 distinct allowed tasks/);
    assert.equal(await matching.locator('svg .nf-cover-node').count(), 2);
    assert.equal(await matching.locator('svg .nf-reached-ring').count(), 5);
    await capture(page, matching, `matching-shortage-${width}.png`);
    await click(matching, 'Clear compatibility');
    await click(matching, 'Reveal cover and shortage');
    assert.match(await matching.locator('.nf-cover-proof').innerText(), /3 workers.*0 distinct allowed tasks/);
    const pair = matching.getByRole('button', { name: 'Compatibility A to 1', exact: true });
    await pair.focus();
    await page.keyboard.press('Space');
    assert.equal(await pair.getAttribute('aria-pressed'), 'true');
    assert.equal(await matching.locator('.nf-cover-proof').count(), 0);
    await click(matching, 'Augment matching');
    assert.match(await matching.getByRole('status').innerText(), /Maximum matching size\s*1/);

    assert.match(await pixels.getByRole('status').innerText(), /=\s*8/);
    await pixels.getByLabel('Neighbor disagreement penalty', { exact: true }).selectOption('5');
    assert.match(await pixels.getByRole('status').innerText(), /=\s*17/);
    await click(pixels, 'Apply a minimum-cut labeling');
    assert.match(await pixels.getByRole('status').innerText(), /=\s*15/);
    assert.match(await pixels.innerText(), /Current excess cost\s*0/);
    const cell = pixels.getByRole('button', { name: /Cell 0, background/ });
    await cell.focus();
    await page.keyboard.press('Enter');
    assert.match(await pixels.innerText(), /Current excess cost\s*6/);
    await click(pixels, 'Apply a minimum-cut labeling');
    await openDetail(pixels, 'Inspect the cut-to-energy');
    assert.equal(await pixels.locator('tbody tr').count(), 6);
    await capture(page, pixels, `binary-minimum-${width}.png`);
    await click(pixels, 'Reset labeling');
    assert.match(await pixels.getByRole('status').innerText(), /=\s*8/);

    const hint = page.locator('summary').filter({ hasText: 'Hint: compare the internal totals first' });
    await hint.focus();
    await page.keyboard.press('Space');
    assert.equal(await hint.locator('..').evaluate(element => element.open), true);
    await openDetail(page, 'Solution and acceptance checks');
    const dinic = await openDetail(page, 'Deeper implementation: Dinic');
    assert.equal(await dinic.locator('.python-example').count(), 1);
    const contraction = await openDetail(page, "Connect to Randomized Algorithms: Karger's");
    assert.equal(await contraction.locator('.python-example').count(), 1);
    const leetcode = page.locator('a[href^="https://leetcode.com/problems/"]');
    assert.equal(await leetcode.count(), 2);
    for (const link of await leetcode.all()) {
      assert.equal(await link.getAttribute('target'), '_blank');
      assert.match(await link.getAttribute('rel'), /noopener/);
    }
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1);
    assert.equal(overflow, false, `Page overflow ${width}`);
    fs.writeFileSync(path.join(output, `network-${width}.json`), JSON.stringify({ failedRequests, consoleErrors }, null, 2));
    assert.deepEqual(errors, []);
    const externalFontErrors = consoleErrors.filter(error => error.location.url.startsWith('https://fonts.googleapis.com/') && error.text.includes('ERR_NETWORK_ACCESS_DENIED'));
    const lessonConsoleErrors = consoleErrors.filter(error => !externalFontErrors.includes(error));
    assert.deepEqual(lessonConsoleErrors, []);
    results.push({ width, anchors: anchors.length, nativePrograms: 9, officialPracticeLinks: 2, keyboard: true, invalidDraftPreserved: true, maximumFlowAndCut: 2, shortage: [3, 2], binaryEnergy: 15, overflow, errors, lessonConsoleErrors, externalFontErrors });
    await page.close();
  }
  fs.writeFileSync(path.join(output, 'results.json'), JSON.stringify(results, null, 2));
  await browser.close();
  console.log(JSON.stringify(results, null, 2));
})().catch(error => { console.error(error); process.exit(1); });

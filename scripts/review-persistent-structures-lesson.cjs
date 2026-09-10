const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const output = path.resolve(__dirname, '../scratch/persistent-structures-lesson-review');
fs.mkdirSync(output, { recursive: true });

async function capture(page, target, filename) {
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => { node.style.visibility = 'hidden'; }));
  await target.screenshot({ path: path.join(output, filename) });
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => { node.style.visibility = ''; }));
}
async function openDetail(lab, label) {
  const summary = lab.locator('summary').filter({ hasText: label }).first();
  const details = summary.locator('..');
  if (!(await details.evaluate(element => element.open))) await summary.click();
  return details;
}
async function button(lab, name) {
  await lab.getByRole('button', { name, exact: true }).click();
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
    page.on('console', message => {
      if (message.type() === 'error' && !message.text().startsWith('[vite] failed to connect to websocket.')) consoleErrors.push({ text: message.text(), location: message.location() });
    });
    await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/persistent-data-structures-structural-sharing-versioned-queries?module=data-structures-algorithms');
    const copy = page.getByRole('region', { name: 'Path copying and version ownership investigation', exact: true });
    const history = page.getByRole('region', { name: 'Per-index historical predecessor lookup', exact: true });
    const rank = page.getByRole('region', { name: 'Prefix history order statistic investigation', exact: true });
    await copy.waitFor({ timeout: 60000 });
    assert.equal(await page.locator('.python-example').count(), 7);
    const anchors = await page.locator('.lesson-intro a[href^="#"]').evaluateAll(nodes => nodes.map(node => node.hash.slice(1)));
    for (const id of anchors) {
      assert.equal(await page.locator(`[id="${id}"]`).count(), 1, id);
      await page.locator(`.lesson-intro a[href="#${id}"]`).click();
      await page.waitForFunction(anchor => { const top = document.getElementById(anchor).getBoundingClientRect().top; return top >= -1 && top < innerHeight; }, id);
    }
    for (const [name, selector] of [['stack', '.persistent-stack'], ['path-overview', '.persistent-copy-figure']]) await capture(page, page.locator(selector), `${name}-${width}.png`);
    assert.match(await copy.locator('.persistent-result').first().innerText(), /v1 \[1,4\).*13/);
    assert(await copy.locator('.persistent-dag text').filter({ hasText: /^v[01] root$/ }).evaluateAll(nodes => nodes.every(node => {
      const bounds = node.getBoundingClientRect();
      const region = node.ownerSVGElement.parentElement.getBoundingClientRect();
      return bounds.left >= region.left - 1 && bounds.right <= region.right + 1;
    })));
    await capture(page, copy, `path-default-${width}.png`);
    await copy.getByLabel('Query version', { exact: true }).selectOption('0');
    await button(copy, 'Run historical query');
    assert.match(await copy.locator('.persistent-result').first().innerText(), /v0 \[1,4\).*8/);
    await copy.getByLabel('Assignment index', { exact: true }).fill('0');
    await copy.getByLabel('New integer value', { exact: true }).fill('-2');
    await button(copy, 'Create branch');
    assert.match(await copy.getByRole('status').innerText(), /Created v2 from v0. 3 new nodes/);
    assert.match(await copy.locator('.persistent-result').first().innerText(), /v2 \[0,5\).*11/);
    await openDetail(copy, 'Inspect physical identities');
    assert.equal(await copy.locator('table tbody tr').count(), 15);
    await capture(page, copy.locator('details').first(), `branch-identities-${width}.png`);
    await copy.getByLabel('Query version', { exact: true }).selectOption('1');
    await button(copy, 'Run historical query');
    assert.match(await copy.locator('.persistent-result').first().innerText(), /v1 \[0,5\).*20/);
    const ownership = await openDetail(copy, 'Which nodes survive');
    await ownership.getByLabel('Keep v0', { exact: true }).uncheck();
    await ownership.getByLabel('Keep v2', { exact: true }).uncheck();
    assert.match(await ownership.locator('.persistent-result').innerText(), /9 reachable \/ 15 allocated/);
    await capture(page, ownership, `ownership-one-root-${width}.png`);
    await ownership.getByLabel('Keep v1', { exact: true }).focus();
    await page.keyboard.press('Space');
    assert.match(await ownership.locator('.persistent-result').innerText(), /0 reachable/);
    const beforeInvalid = await copy.locator('.persistent-result').first().innerText();
    await copy.getByLabel('Range start, inclusive', { exact: true }).fill('4');
    await copy.getByLabel('Range end, exclusive', { exact: true }).fill('2');
    await button(copy, 'Run historical query');
    assert.match(await copy.getByRole('alert').innerText(), /unchanged/);
    assert.equal(await copy.locator('.persistent-result').first().innerText(), beforeInvalid);
    await copy.getByLabel('Range start, inclusive', { exact: true }).fill('2');
    await button(copy, 'Run historical query');
    assert.match(await copy.locator('.persistent-result').first().innerText(), /v1 \[2,2\).*0.*empty/);
    const reset = copy.getByRole('button', { name: 'Reset versions', exact: true });
    await reset.focus();
    await page.keyboard.press('Enter');
    assert.match(await copy.getByRole('status').innerText(), /Fresh v0/);
    await copy.getByLabel('Assignment index', { exact: true }).fill('2');
    await copy.getByLabel('New integer value', { exact: true }).fill('4');
    await button(copy, 'Create branch');
    assert.match(await copy.getByRole('status').innerText(), /0 new nodes/);
    for (let i = 0; i < 6; i++) await button(copy, 'Create branch');
    await button(copy, 'Create branch');
    assert.match(await copy.getByRole('alert').innerText(), /Eight versions reached/);
    await button(copy, 'Reset versions');
    const initial = await openDetail(copy, 'Try a different initial array');
    await initial.getByRole('textbox').fill('100');
    await button(initial, 'Build fresh array');
    assert.match(await copy.getByRole('alert').innerText(), /−?99|99/);
    await initial.getByRole('textbox').fill('7');
    await button(initial, 'Build fresh array');
    assert.match(await copy.locator('.persistent-result').first().innerText(), /v0 \[0,1\).*7/);
    await copy.getByLabel('Assignment index', { exact: true }).fill('0');
    await copy.getByLabel('New integer value', { exact: true }).fill('-9');
    await button(copy, 'Create branch');
    assert.match(await copy.getByRole('status').innerText(), /1 new nodes/);
    const graph = copy.locator('.persistent-scroll').first();
    await graph.focus();
    await page.keyboard.press('ArrowRight');
    assert(await graph.evaluate(element => document.activeElement === element && element.matches(':focus-visible') && parseFloat(getComputedStyle(element).outlineWidth) >= 2));
    await button(copy, 'Center root handles');
    if (width === 320) assert(await graph.evaluate(element => element.scrollLeft > 0));

    await capture(page, history, `history-middle-${width}.png`);
    await history.getByLabel('Saved snapshot', { exact: true }).selectOption('4');
    assert.match(await history.locator('.persistent-result').innerText(), /9/);
    await history.getByLabel('Array index', { exact: true }).selectOption('2');
    assert.match(await history.locator('.persistent-result').innerText(), /= 0/);
    await button(history, 'Reset lookup');
    await openDetail(history, 'Inspect the binary-search');
    await capture(page, history, `history-search-${width}.png`);

    assert.match(await rank.getByRole('status').innerText(), /Left half contains 3 occurrences/);
    await button(rank, 'Next rank step');
    assert.match(await rank.getByRole('status').innerText(), /Left half contains 2 occurrences.*rank 1/);
    await capture(page, rank, `rank-decision-${width}.png`);
    await button(rank, 'Next rank step');
    assert.match(await rank.getByRole('status').innerText(), /bucket remains: 3/);
    await rank.getByLabel('Rank, starting at 1', { exact: true }).fill('9');
    await button(rank, 'Apply query');
    assert.match(await rank.getByRole('alert').innerText(), /unchanged/);
    assert.match(await rank.getByRole('status').innerText(), /bucket remains: 3/);
    await rank.getByLabel('Array values', { exact: true }).fill('4, 4, 4');
    await rank.getByLabel('Subarray start', { exact: true }).fill('0');
    await rank.getByLabel('Subarray end, exclusive', { exact: true }).fill('3');
    await rank.getByLabel('Rank, starting at 1', { exact: true }).fill('2');
    await button(rank, 'Apply query');
    assert.match(await rank.getByRole('status').innerText(), /bucket remains: 4/);
    await capture(page, rank, `rank-duplicates-${width}.png`);
    await rank.getByLabel('Array values', { exact: true }).fill('-9, 8, 3, 1, 0, 7, -3, 2');
    await rank.getByLabel('Subarray end, exclusive', { exact: true }).fill('8');
    await rank.getByLabel('Rank, starting at 1', { exact: true }).fill('8');
    await button(rank, 'Apply query');
    while (!(await rank.getByRole('button', { name: 'Next rank step', exact: true }).isDisabled())) await button(rank, 'Next rank step');
    assert.match(await rank.getByRole('status').innerText(), /bucket remains: 8/);

    const practice = page.locator('.dsa-practice');
    assert.equal(await practice.locator('a[href*="leetcode.com/problems/"]').count(), 2);
    const checkpoint = page.locator('.lesson-check').first();
    assert.equal(await checkpoint.locator('details').evaluate(element => element.open), false);
    await checkpoint.locator('summary').focus();
    await page.keyboard.press('Enter');
    assert.equal(await checkpoint.locator('details').evaluate(element => element.open), true);
    const task = page.locator('.persistent-practice-task').first();
    assert.equal(await task.locator('details').first().evaluate(element => element.open), false);
    assert.equal(await task.locator('details').last().evaluate(element => element.open), false);
    await task.locator('summary').first().focus();
    await page.keyboard.press('Enter');
    assert.equal(await task.locator('details').first().evaluate(element => element.open), true);
    assert.equal(await task.locator('details').last().evaluate(element => element.open), false);
    await capture(page, task, `practice-hint-${width}.png`);
    for (const id of ['2-share-what-did-not-change', '3-copy-the-path-to-a-changed-value', '7-preserve-lazy-updates-without-mutating-children']) {
      await page.evaluate(anchor => { const target = document.getElementById(anchor); window.scrollTo({ top: window.scrollY + target.getBoundingClientRect().top - 180, behavior: 'instant' }); }, id);
      await page.screenshot({ path: path.join(output, `ordinary-${id}-${width}.png`) });
    }
    const figureMetrics = await page.locator('.persistent-copy-figure svg').evaluate(svg => ({ width: svg.getBoundingClientRect().width, fontSize: getComputedStyle(svg.querySelector('text')).fontSize, viewWidth: svg.viewBox.baseVal.width }));
    assert(figureMetrics.width / figureMetrics.viewWidth * parseFloat(figureMetrics.fontSize) >= 13.5);
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
    assert.deepEqual(errors, []);
    const lessonConsoleErrors = consoleErrors.filter(error => !error.location.url.startsWith('https://fonts.googleapis.com/'));
    assert.deepEqual(lessonConsoleErrors, []);
    results.push({ width, anchors, figureMetrics, errors, consoleErrors, failedRequests, passed: true });
    await page.close();
  }
  fs.writeFileSync(path.join(output, 'results.json'), JSON.stringify(results, null, 2));
  await browser.close();
  console.log('PASS: desktop/390/320 branching, ownership, history, rank, invalid/no-op/bounds, keyboard, 10 anchors, ordinary reading and screenshots.');
})().catch(error => { console.error(error); process.exit(1); });

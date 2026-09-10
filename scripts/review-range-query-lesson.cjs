const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const directory = path.resolve(__dirname, '../scratch/range-query-lesson-review');
fs.mkdirSync(directory, { recursive: true });

async function capture(page, target, name) {
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = 'hidden'));
  await target.screenshot({ path: path.join(directory, name) });
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = ''));
}
async function values(lab) {
  return (await lab.locator('.range-cells').first().locator('strong').allTextContents()).map(Number);
}
async function trace(lab, inspect = () => {}) {
  let count = 0;
  while (true) {
    await inspect();
    count++;
    const next = lab.getByRole('button', { name: 'Next event', exact: true });
    if (await next.isDisabled()) break;
    assert(count < 200, 'Trace must be finite');
    await next.click();
  }
  return count;
}
async function fill(lab, name, value) { await lab.getByLabel(name, { exact: true }).fill(String(value)); }
async function button(lab, name) { await lab.getByRole('button', { name, exact: true }).click(); }

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  for (const width of [1440, 390, 320]) {
    const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
    await page.routeWebSocket('**', socket => socket.close());
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(process.env.LESSON_URL || 'http://127.0.0.1:5173/learn/path/full-curriculum/segment-trees-fenwick-trees-range-queries?module=data-structures-algorithms');
    const segment = page.locator('section[aria-label="Segment range investigation"]');
    await segment.waitFor({ timeout: 60000 });
    const fenwick = page.locator('section[aria-label="Fenwick block investigation"]');
    const lazy = page.locator('section[aria-label="Lazy range propagation investigation"]');
    const maximum = page.locator('section[aria-label="Moving maximum investigation"]');
    const shortest = page.locator('section[aria-label="Signed shortest range investigation"]');
    assert.equal(await page.locator('.python-example').count(), 13);
    assert.equal(await page.locator('.range-lab').count(), 5);
    const anchors = await page.locator('.lesson-intro a[href^="#"]').evaluateAll(nodes => nodes.map(node => node.hash.slice(1)));
    for (const anchor of anchors) assert.equal(await page.locator(`[id="${anchor}"]`).count(), 1, anchor);
    await page.screenshot({ path: path.join(directory, `reading-entry-${width}.png`) });
    for (const anchor of anchors) {
      await page.locator(`.lesson-intro a[href="#${anchor}"]`).click();
      assert.equal(new URL(page.url()).hash, `#${anchor}`);
      await page.waitForFunction(id => {
        const top = document.getElementById(id).getBoundingClientRect().top;
        return top >= -1 && top < innerHeight;
      }, anchor, { timeout: 5000 });
      const top = await page.locator(`[id="${anchor}"]`).evaluate(node => node.getBoundingClientRect().top);
      assert(top >= -1 && top < 1000, `Anchor ${anchor} arrives in viewport`);
    }
    for (let index = 0; index < 4; index++) await capture(page, page.locator('.range-figure').nth(index), `inline-${index + 1}-${width}.png`);

    const counts = {};
    counts.segment = await trace(segment);
    assert.match(await segment.getByRole('status').innerText(), /Combine left then right: 8\./);
    await capture(page, segment, `segment-query-${width}.png`);
    await button(segment, 'Assign point');
    counts.segment += await trace(segment);
    assert.deepEqual(await values(segment), [2, 1, 8, 4]);
    assert.equal(await segment.locator('.range-node-value').first().textContent(), '15');
    for (const [array, operation, expected] of [['-5,-2,-4', 'max', '-2'], ['4,3,2,1,5', 'min', '1'], ['', 'sum', '0']]) {
      await fill(segment, 'Array draft · up to 8 integers, −99…99', array);
      await segment.getByLabel('Combine draft', { exact: true }).selectOption(operation);
      await button(segment, 'Apply array and combine');
      counts.segment += await trace(segment);
      assert.match(await segment.getByRole('status').innerText(), new RegExp(`Combine left then right: ${expected}\\.`));
    }
    await button(segment, 'Reset segment lab');
    await button(segment, 'Finish operation');
    await fill(segment, 'Left boundary', '');
    await button(segment, 'Query range');
    assert.match(await segment.getByRole('alert').innerText(), /blank is not zero/);
    await fill(segment, 'Left boundary', '2');
    await fill(segment, 'Right boundary · excluded', '1');
    await button(segment, 'Query range');
    assert.equal(await segment.getByRole('alert').count(), 1);
    await button(segment, 'Reset segment lab');

    counts.fenwick = await trace(fenwick);
    assert.match(await fenwick.getByRole('status').innerText(), /prefix sum is 17/);
    assert.deepEqual(await fenwick.locator('.fenwick-lane.is-selected > span > b').allTextContents(), ['4', '6', '7']);
    await capture(page, fenwick, `fenwick-prefix-${width}.png`);
    await button(fenwick, 'Add at point');
    counts.fenwick += await trace(fenwick);
    assert.deepEqual(await fenwick.locator('.fenwick-lane.is-selected > span > b').allTextContents(), ['5', '6', '8']);
    await button(fenwick, 'Read prefix');
    counts.fenwick += await trace(fenwick);
    assert.match(await fenwick.getByRole('status').innerText(), /prefix sum is 20/);
    await fill(fenwick, 'Prefix end · excluded', '0');
    await button(fenwick, 'Read prefix');
    counts.fenwick += await trace(fenwick);
    assert.match(await fenwick.getByRole('status').innerText(), /prefix sum is 0/);
    await fill(fenwick, 'Array draft · up to 8 integers, −99…99', '1.5');
    await button(fenwick, 'Apply array');
    assert.equal(await fenwick.getByRole('alert').count(), 1);
    await button(fenwick, 'Reset Fenwick lab');

    counts.lazy = await trace(lazy);
    assert.deepEqual(await values(lazy), [5, 4, 6, 7]);
    assert.equal(await lazy.locator('.range-node-value').first().textContent(), '22');
    assert.equal(await lazy.locator('.range-node-tag').first().textContent(), 'add 3');
    await capture(page, lazy, `lazy-deferred-${width}.png`);
    await button(lazy, 'Run range operation');
    let sawPush = false;
    counts.lazy += await trace(lazy, async () => {
      if ((await lazy.getByRole('status').innerText()).startsWith('Push map')) {
        sawPush = true;
        await capture(page, lazy, `lazy-push-${width}.png`);
      }
    });
    assert(sawPush);
    assert.deepEqual(await values(lazy), [5, 5, 5, 7]);
    await lazy.getByLabel('Next operation', { exact: true }).selectOption('query');
    await button(lazy, 'Run range operation');
    counts.lazy += await trace(lazy);
    assert.match(await lazy.getByRole('status').innerText(), /sum 10/);
    for (const [kind, amount, expected] of [['set', 0, [0, 0, 0, 0]], ['add', -2, [-2, -2, -2, -2]]]) {
      await lazy.getByLabel('Next operation', { exact: true }).selectOption(kind);
      await fill(lazy, 'Left boundary', '0');
      await fill(lazy, 'Right boundary · excluded', '4');
      await fill(lazy, 'Amount · −20…20', amount);
      await button(lazy, 'Run range operation');
      counts.lazy += await trace(lazy);
      assert.deepEqual(await values(lazy), expected);
    }
    await button(lazy, 'Reset lazy lab');
    await button(lazy, 'Finish operation');
    await fill(lazy, 'Amount · −20…20', '');
    await lazy.getByLabel('Next operation', { exact: true }).selectOption('query');
    await button(lazy, 'Run range operation');
    counts.lazy += await trace(lazy);
    assert.match(await lazy.getByRole('status').innerText(), /sum 10/);
    await fill(lazy, 'Array draft · up to 8 integers, −99…99', '');
    await button(lazy, 'Apply array');
    counts.lazy += await trace(lazy);
    assert.deepEqual(await values(lazy), []);
    assert.match(await lazy.getByRole('status').innerText(), /sum 0/);
    await button(lazy, 'Reset lazy lab');

    counts.maximum = await trace(maximum);
    assert.deepEqual(await maximum.locator('tbody tr').evaluateAll(nodes => nodes.map(node => [...node.children].map(cell => cell.textContent))), [['[0,3)', '4', '0'], ['[1,4)', '5', '3'], ['[2,5)', '5', '3'], ['[3,6)', '5', '3'], ['[4,7)', '3', '5'], ['[5,8)', '3', '5']]);
    await capture(page, maximum, `moving-max-${width}.png`);
    await fill(maximum, 'Array draft · up to 8 integers, −20…20', '2,2,2');
    await fill(maximum, 'Window width · 1…8', '2');
    await button(maximum, 'Apply maximum inputs');
    counts.maximum += await trace(maximum);
    assert.deepEqual(await maximum.locator('tbody tr td:last-child').allTextContents(), ['1', '2']);
    await fill(maximum, 'Window width · 1…8', '4');
    await button(maximum, 'Apply maximum inputs');
    counts.maximum += await trace(maximum);
    assert.match(await maximum.getByRole('status').innerText(), /0 full windows/);
    await fill(maximum, 'Window width · 1…8', '0');
    await button(maximum, 'Apply maximum inputs');
    assert.equal(await maximum.getByRole('alert').count(), 1);
    await button(maximum, 'Reset maximum lab');

    counts.shortest = await trace(shortest);
    assert.match(await shortest.getByRole('status').innerText(), /\[2,3\).*length 1 and sum 5/);
    await capture(page, shortest, `signed-shortest-${width}.png`);
    for (const [array, target, expected] of [['2,-1,2', 3, /\[0,3\).*length 3/], ['-2,-1', 1, /No nonempty/], ['', 1, /No nonempty/]]) {
      await fill(shortest, 'Array draft · up to 8 integers, −20…20', array);
      await fill(shortest, 'Positive target · 1…100', target);
      await button(shortest, 'Apply shortest inputs');
      counts.shortest += await trace(shortest);
      assert.match(await shortest.getByRole('status').innerText(), expected);
    }
    await fill(shortest, 'Positive target · 1…100', '');
    await button(shortest, 'Apply shortest inputs');
    assert.match(await shortest.getByRole('alert').innerText(), /blank is not zero/);
    await button(shortest, 'Reset shortest lab');
    await shortest.getByRole('button', { name: 'Next event', exact: true }).focus();
    await page.keyboard.press('Enter');
    assert.match(await shortest.getByRole('status').innerText(), /Consider end boundary 0/);
    await shortest.getByRole('button', { name: 'Previous', exact: true }).focus();
    await page.keyboard.press('Space');
    assert.match(await shortest.getByRole('status').innerText(), /Prefix P\[0\]=0/);
    if (width <= 390) {
      const tree = segment.locator('.range-tree');
      assert(await tree.evaluate(node => node.parentElement.clientWidth >= node.getBoundingClientRect().width), 'Four-leaf tree fits narrow reading area');
    }
    const scroll = fenwick.locator('.range-scroll').first();
    await scroll.focus();
    assert.equal(await scroll.evaluate(node => getComputedStyle(node).outlineStyle), 'solid');
    if (width < 500) {
      await page.keyboard.press('ArrowRight');
      await page.waitForTimeout(150);
      assert(await scroll.evaluate(node => node.scrollLeft > 0));
    }
    const disclosure = shortest.getByText('Read the exact prefix table', { exact: true });
    await disclosure.focus();
    await page.keyboard.press('Enter');
    assert(await disclosure.evaluate(node => node.parentElement.open));
    const hints = page.locator('.dsa-practice details summary');
    if (await hints.count()) { await hints.first().focus(); await page.keyboard.press('Enter'); assert(await hints.first().evaluate(node => node.parentElement.open)); }
    const links = await page.locator('.dsa-practice a[href*="leetcode.com/problems/"]').evaluateAll(nodes => nodes.map(node => ({ href: node.href, target: node.target, rel: node.rel })));
    assert.equal(links.length, 5);
    assert(links.every(link => link.target === '_blank' && link.rel.includes('noopener')));
    const overflow = await page.evaluate(() => ({ viewport: innerWidth, page: document.documentElement.scrollWidth }));
    assert.equal(overflow.page, overflow.viewport);
    assert.deepEqual(errors, []);
    results.push({ width, counts, anchors: anchors.length, programs: 13, practiceLinks: links.length, overflow, errors });
    await page.close();
  }
  await browser.close();
  fs.writeFileSync(path.join(directory, 'results.json'), JSON.stringify(results, null, 2));
  console.log(JSON.stringify(results));
})().catch(error => { console.error(error); process.exit(1); });

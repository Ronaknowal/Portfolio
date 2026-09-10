// Targeted review of layout corrections after the full interaction matrix.
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert = require('node:assert/strict');
const path = require('node:path');
const fs = require('node:fs');
const directory = path.resolve(__dirname, '../scratch/greedy-exchange-lesson-review');
fs.mkdirSync(directory, { recursive: true });

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  const errors = [];
  for (const width of [1440, 390, 320]) {
    const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
    await page.routeWebSocket('**', socket => socket.close());
    page.on('pageerror', error => errors.push(error.message));
    await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/greedy-algorithms-exchange-arguments?module=data-structures-algorithms');
    const lab = page.locator('[data-gx-lab="deadlines"]');
    await lab.waitFor({ timeout: 60000 });
    for (const [id, processing, deadline] of [['A', 1, 30], ['B', 8, 0], ['C', 1, 30]]) {
      await lab.getByLabel(`Job ${id} processing time`, { exact: true }).fill(String(processing));
      await lab.getByLabel(`Job ${id} deadline`, { exact: true }).fill(String(deadline));
    }
    await lab.getByRole('button', { name: 'Apply jobs', exact: true }).click();
    await lab.getByRole('button', { name: 'Order all by deadline', exact: true }).click();
    const metrics = await lab.locator('.gx-job').evaluateAll(nodes => nodes.map(node => ({
      id: node.querySelector('strong').textContent,
      width: node.getBoundingClientRect().width,
      labels: [...node.querySelectorAll('small,span')].map(label => ({
        width: label.getBoundingClientRect().width,
        height: label.getBoundingClientRect().height,
        text: label.textContent,
        nowrap: getComputedStyle(label).whiteSpace,
      })),
    })));
    for (const job of metrics) {
      assert.ok(job.width >= 50, JSON.stringify(job));
      for (const label of job.labels) {
        assert.equal(label.nowrap, 'nowrap');
        assert.ok(label.width <= job.width - 1, JSON.stringify(job));
      }
    }
    await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = 'hidden'));
    await lab.screenshot({ path: path.join(directory, `deadline-readable-${width}.png`) });
    const current = lab.locator('h4', { hasText: 'Current order' }).locator('..');
    const scroller = current.locator('.gx-scroll');
    await page.keyboard.press('Tab');
    await scroller.focus();
    if (width < 400) {
      await page.keyboard.press('End');
      await page.keyboard.press('ArrowRight');
      await page.waitForTimeout(200);
      assert.ok(await scroller.evaluate(node => node.scrollLeft > 0));
      await scroller.evaluate(node => node.scrollLeft = node.scrollWidth);
      await current.screenshot({ path: path.join(directory, `deadline-readable-tail-${width}.png`) });
    }
    assert.match(await current.innerText(), /Maximum signed lateness 8; maximum tardiness 8/);
    const reach = page.locator('[data-gx-figure="reach"]');
    const stages = await reach.locator('g[data-reach]').evaluateAll(nodes => nodes.map(node => ({
      reach: Number(node.dataset.reach),
      index: Number(node.dataset.inspected),
      band: Number(node.querySelector('rect').getAttribute('width')),
      inspected: node.querySelectorAll('.gx-reach-inspected').length,
      known: node.querySelectorAll('.gx-reach-known').length,
    })));
    assert.deepEqual(stages, [
      { reach: 2, index: 0, band: 122, inspected: 1, known: 2 },
      { reach: 4, index: 1, band: 212, inspected: 1, known: 4 },
      { reach: 5, index: 4, band: 257, inspected: 1, known: 5 },
    ]);
    assert.deepEqual(await reach.locator('.gx-reach-value').allTextContents(), ['2', '3', '0', '0', '1', '0']);
    assert.equal(await reach.locator('svg').evaluate(node => node.getBoundingClientRect().width), 280);
    await reach.screenshot({ path: path.join(directory, `reach-continuous-${width}.png`) });
    const capacity = page.locator('[data-gx-lab="capacity"]');
    await capacity.getByLabel('Capacity', { exact: true }).focus();
    await page.keyboard.press('Home');
    for (let value = 0; value < 31; value++) await page.keyboard.press('ArrowRight');
    assert.match(await capacity.locator('.gx-status').innerText(), /Density-first value: 164/);
    assert.equal(await capacity.locator('.gx-capacity-track .gx-job--C strong').count(), 0);
    assert.equal(await capacity.locator('.gx-capacity-track .gx-job--C').getAttribute('title'), 'C: 1 weight');
    await capacity.screenshot({ path: path.join(directory, `capacity-thin-fractional-${width}.png`) });
    await capacity.getByRole('checkbox', { name: 'Allow fractional items', exact: true }).setChecked(false);
    assert.match(await capacity.innerText(), /Unused capacity: 1/);
    assert.equal(await capacity.locator('.gx-unused span').count(), 0);
    assert.match(await capacity.locator('.gx-status').innerText(), /Density-first value: 160/);
    await capacity.screenshot({ path: path.join(directory, `capacity-thin-unused-${width}.png`) });
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth);
    assert.equal(overflow, width);
    results.push({ width, labels: metrics, reachablePrefixStages: stages, thinCapacity: 'exact table retained; no clipped inline labels', overflow });
    await page.close();
  }
  assert.deepEqual(errors, []);
  await browser.close();
  fs.writeFileSync(path.join(directory, 'visual-layout-results.json'), JSON.stringify({ results, errors }, null, 2));
  console.log('Readable short-job labels, keyboard local scroll, continuous reachable prefixes, thin-capacity labels, numerical conservation and page overflow passed at 1440/390/320.');
})().catch(error => { console.error(error); process.exit(1); });

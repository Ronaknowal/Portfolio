const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const directory = path.resolve(__dirname, '../scratch/dynamic-programming-lesson-review');

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  const errors = [];
  try {
    for (const width of [1440, 390]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      page.on('pageerror', error => errors.push(error.message));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/dynamic-programming-states-transitions-optimization?module=data-structures-algorithms');
      const lesson = page.locator('.dynamic-programming-lesson');
      await lesson.waitFor();
      const reward = page.getByRole('region', { name: 'Reward dependency investigation', exact: true });
      const graph = reward.getByRole('region', { name: 'Suffix dependency graph', exact: true });
      assert(await graph.evaluate(node => node.scrollWidth <= node.clientWidth + 1), 'Default complete graph must fit');
      async function capture(locator, name) {
        await locator.scrollIntoViewIfNeeded();
        const style = await page.addStyleTag({ content: '.learn-nav{visibility:hidden!important}' });
        await locator.screenshot({ path: path.join(directory, `${name}-${width}.png`) });
        await style.evaluate(node => node.remove());
      }
      await reward.getByRole('button', { name: 'Finish states' }).click();
      await capture(reward, 'reward-compact-final');
      await reward.getByLabel('Session rewards').fill('-9,20,-9,20,-9,20,-9');
      await reward.getByRole('button', { name: 'Apply rewards' }).click();
      while (!(await reward.getByRole('button', { name: 'Next state', exact: true }).isDisabled())) {
        await reward.getByRole('button', { name: 'Next state', exact: true }).click();
        assert(await graph.evaluate(node => {
          const selected = node.querySelector('.dp-node-active');
          if (!selected) return true;
          const outer = node.getBoundingClientRect();
          const inner = selected.getBoundingClientRect();
          return inner.left >= outer.left - 1 && inner.right <= outer.right + 1;
        }), 'Active reward state visible');
      }
      if (width === 390) {
        await graph.focus();
        for (let press = 0; press < 10; press += 1) await page.keyboard.press('ArrowRight');
        await page.waitForTimeout(150);
        assert(await graph.evaluate(node => node.scrollLeft > 0));
      }
      await capture(reward, 'reward-long-scroll');
      const capacity = page.getByRole('region', { name: 'Capacity generation investigation', exact: true });
      const strip = capacity.getByRole('region', { name: 'Capacity array and update generations', exact: true });
      let checkedWrites = 0;
      for (const direction of ['descending', 'ascending']) {
        await capacity.getByRole('button', { name: 'Reset capacity lab' }).click();
        await capacity.getByLabel('Capacity order', { exact: true }).selectOption(direction);
        while (!(await capacity.getByRole('button', { name: 'Next update', exact: true }).isDisabled())) {
          await capacity.getByRole('button', { name: 'Next update', exact: true }).click();
          assert(await strip.evaluate(node => {
            const selected = node.querySelector('.dp-selected');
            const outer = node.getBoundingClientRect();
            const inner = selected.getBoundingClientRect();
            return inner.left >= outer.left - 1 && inner.right <= outer.right + 1;
          }), 'Written capacity remains visible');
          checkedWrites += 1;
        }
        await capture(capacity, `capacity-visible-${direction}`);
      }
      if (width === 390) {
        await strip.focus();
        for (let press = 0; press < 25; press += 1) await page.keyboard.press('ArrowLeft');
        await page.waitForTimeout(150);
        assert.equal(await strip.evaluate(node => node.scrollLeft), 0);
      }
      // Inspect ordinary reading locations without unfolding every lab/answer.
      await page.reload();
      await lesson.waitFor();
      const headings = await lesson.locator('h2').all();
      for (let index = 0; index < headings.length; index += 1) {
        await headings[index].scrollIntoViewIfNeeded();
        await page.screenshot({ path: path.join(directory, `reading-${index}-${width}.png`) });
      }
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
      results.push({ width, defaultGraphFits: true, activeLongGraphVisible: true, capacityWritesVisible: checkedWrites, keyboardLocalScrolling: width === 390, readingViews: headings.length, overflow: false });
      await page.close();
    }
    assert.deepEqual(errors, []);
    const record = { checkedAt: new Date().toISOString(), results, errors };
    fs.writeFileSync(path.join(directory, 'layout-results.json'), JSON.stringify(record, null, 2));
    console.log(JSON.stringify(record, null, 2));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

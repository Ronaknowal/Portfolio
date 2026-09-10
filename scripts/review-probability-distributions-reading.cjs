const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const directory = 'scratch/probability-distributions-review/browser';
fs.mkdirSync(directory, { recursive: true });
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      const errors = [];
      const consoleErrors = [];
      const environmentMessages = [];
      await page.routeWebSocket('**', socket => socket.close());
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => {
        if (message.type() !== 'error') return;
        if (message.text().startsWith('[vite] failed to connect to websocket.') || /Failed to load resource: net::ERR_(?:CONNECTION_CLOSED|INTERNET_DISCONNECTED|NAME_NOT_RESOLVED|BLOCKED_BY_CLIENT|NETWORK_ACCESS_DENIED)/.test(message.text())) environmentMessages.push(message.text());
        else consoleErrors.push(message.text());
      });
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/probability-distributions-bayes-theorem?module=math-foundations', { waitUntil: 'networkidle' });
      const lesson = page.locator('.probability-distributions-lesson');
      await lesson.waitFor();
      assert.equal(await lesson.locator('p p, p div, p section, p table').count(), 0);
      const targets = [
        ['bayes-area', lesson.locator('[data-investigation="probability-bayes"] .probability-mass-strip').first()],
        ['count-cdf', lesson.locator('[data-investigation="probability-counts"] .probability-paired-plots')],
        ['density-cdf', lesson.locator('[data-investigation="probability-density"] .probability-paired-plots')],
        ['arrivals-timeline', lesson.locator('.probability-timeline')],
        ['same-moments', lesson.locator('.probability-tail-figure')],
        ['normal-units', lesson.locator('.probability-normal-units')],
        ['first-native', lesson.locator('.python-example').first()],
      ];
      for (const [name, locator] of targets) {
        await locator.evaluate(node => node.scrollIntoView({ block: 'center' }));
        await page.screenshot({ path: `${directory}/ordinary-${name}-${width}.png` });
      }
      const firstButton = lesson.locator('[data-investigation="probability-events"] button').first();
      await firstButton.focus();
      const keyboard = [];
      for (let index = 0; index < 20; index += 1) {
        await page.keyboard.press('Tab');
        const focused = await page.evaluate(() => {
          const node = document.activeElement;
          const style = getComputedStyle(node);
          return { tag: node.tagName, name: node.getAttribute('aria-label') || node.textContent.trim().slice(0, 70), outlineStyle: style.outlineStyle, outlineWidth: style.outlineWidth, inside: Boolean(node.closest('.probability-distributions-lesson')) };
        });
        assert.ok(focused.inside);
        assert.notEqual(focused.outlineStyle, 'none');
        assert.notEqual(focused.outlineWidth, '0px');
        keyboard.push(focused);
      }
      await page.screenshot({ path: `${directory}/keyboard-${width}.png` });
      await lesson.locator('details').evaluateAll(nodes => nodes.forEach(node => { node.open = true; }));
      const tableScrolls = [];
      for (const table of await lesson.locator('.lesson-table-wrap').all()) {
        const dimensions = await table.evaluate(node => ({ width: node.clientWidth, scroll: node.scrollWidth }));
        if (dimensions.scroll <= dimensions.width + 2) continue;
        await table.focus();
        for (let press = 0; press < 8; press += 1) await page.keyboard.press('ArrowRight');
        await page.waitForFunction(node => node.scrollLeft > 0, await table.elementHandle());
        tableScrolls.push({ ...dimensions, left: await table.evaluate(node => node.scrollLeft) });
      }
      const dimensions = await page.evaluate(() => ({ width: innerWidth, scroll: document.documentElement.scrollWidth }));
      assert.ok(dimensions.scroll <= width + 1);
      assert.deepEqual(errors, []);
      assert.deepEqual(consoleErrors, []);
      results.push({ width, keyboard, tableScrolls, dimensions, errors, consoleErrors, environmentMessages });
      await page.close();
    }
    fs.writeFileSync(`${directory}/reading-results.json`, JSON.stringify({ checkedAt: new Date().toISOString(), results }, null, 2));
    console.log(JSON.stringify(results.map(result => ({ width: result.width, keyboardControls: result.keyboard.length, scrollableTables: result.tableScrolls.length, errors: result.errors, consoleErrors: result.consoleErrors }))));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

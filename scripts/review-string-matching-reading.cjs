const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const directory = path.resolve('scratch/string-matching-browser');

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  try {
    for (const width of [1440, 390]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      await page.goto((process.env.LEARNING_BASE_URL || 'http://127.0.0.1:5173') + '/learn/path/full-curriculum/string-matching-prefix-functions-rolling-hashes?module=data-structures-algorithms');
      await page.locator('.string-matching-lesson').waitFor();
      const intro = page.locator('.string-matching-lesson .lesson-intro');
      await intro.screenshot({ path: path.join(directory, `intro-${width}.png`) });
      const finished = [];
      for (const [labId, resultId, expected] of [['prefix-border', 'prefix-action', 'Store π[6]=1.'], ['kmp-alignment', 'kmp-action', 'Text exhausted.'], ['chunk-matcher', 'stream-action', 'After 4 chunks:'], ['rolling-fingerprint', 'rolling-action', 'Window start 4;']]) {
        const lab = page.locator(`[data-lab="${labId}"]`);
        await lab.getByRole('button', { name: 'Finish', exact: true }).focus();
        await page.keyboard.press('Enter');
        assert((await lab.locator(`[data-result="${resultId}"]`).innerText()).startsWith(expected));
        await lab.getByRole('button', { name: 'Previous', exact: true }).focus();
        await page.keyboard.press('Enter');
        assert(!(await lab.getByRole('button', { name: 'Finish', exact: true }).isDisabled()));
        await lab.getByRole('button', { name: 'Reset', exact: true }).focus();
        await page.keyboard.press('Space');
        assert(await lab.getByRole('button', { name: 'Previous', exact: true }).isDisabled());
        finished.push(labId);
      }
      const sources = page.locator('.string-matching-lesson .lesson-sources');
      assert.equal(await sources.locator('a').count(), 6);
      assert.equal(await sources.locator('a[target="_blank"]').count(), 6);
      await sources.screenshot({ path: path.join(directory, `sources-${width}.png`) });
      results.push({ width, finishPreviousResetKeyboard: finished, introLinks: await intro.locator('a').count(), annotatedSources: 6 });
      await page.close();
    }
    const result = { checkedAt: new Date().toISOString(), results };
    fs.writeFileSync(path.join(directory, 'reading-controls-results.json'), JSON.stringify(result, null, 2));
    console.log(JSON.stringify(result, null, 2));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });

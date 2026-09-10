const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const directory = 'scratch/learning-rate-schedule-browser';
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/learning-rate-schedules-cosine-warmup-onecyclelr?module=math-foundations', { waitUntil: 'networkidle' });
      const lesson = page.locator('.learning-rate-schedules-lesson');
      await lesson.waitFor();
      await lesson.locator('.lesson-intro').screenshot({ style: '.learn-nav { visibility: hidden; }', path: `${directory}/intro-${width}.png` });
      assert.ok((await lesson.innerText()).includes('.venv\\Scripts\\Activate.ps1'));
      const exercises = lesson.locator(':scope > details');
      assert.equal(await exercises.count(), 7);
      for (let index = 0; index < 7; index += 1) {
        const exercise = exercises.nth(index);
        const outer = exercise.locator(':scope > summary');
        await outer.focus(); await page.keyboard.press('Enter');
        for (const name of ['Hint', index < 5 ? 'Explained solution' : 'Acceptance and explanation']) {
          const summary = exercise.getByText(name, { exact: true });
          await summary.focus(); await page.keyboard.press('Enter');
        }
        assert.ok(await exercise.evaluate(node => node.open && [...node.querySelectorAll('details')].every(item => item.open)));
        await exercise.screenshot({ style: '.learn-nav { visibility: hidden; }', path: `${directory}/practice-${index + 1}-${width}.png` });
      }
      assert.ok((await exercises.nth(5).innerText()).includes('.066142 instead of .057201'));
      assert.ok((await exercises.nth(6).innerText()).includes('[.103916,.103974,.106589]'));
      const clock = lesson.locator('[data-lab="schedule-clock"]');
      await clock.getByRole('button', { name: 'Finish', exact: true }).click();
      const summary = clock.getByText('Inspect the actual consumed indices and rates', { exact: true });
      await summary.focus(); await page.keyboard.press('Enter');
      const table = clock.getByRole('region', { name: 'Consumed schedule indices and rates, horizontally scrollable' });
      await table.focus();
      await page.keyboard.press('ArrowRight');
      const scroll = await table.evaluate(node => ({ width: node.clientWidth, content: node.scrollWidth, focused: document.activeElement === node }));
      assert.equal(scroll.focused, true);
      await table.screenshot({ style: '.learn-nav { visibility: hidden; }', path: `${directory}/consumed-table-${width}.png` });
      const sources = lesson.locator('.lesson-sources');
      assert.equal(await sources.locator('a').count(), 14);
      await sources.screenshot({ style: '.learn-nav { visibility: hidden; }', path: `${directory}/sources-${width}.png` });
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 2), false);
      assert.deepEqual(errors, []);
      results.push({ width, exercises: 7, keyboardReveals: 21, sourceLinks: 14, table: scroll, errors });
      await page.close();
    }
    fs.writeFileSync(`${directory}/reading-results.json`, JSON.stringify({ checkedAt: new Date().toISOString(), results }, null, 2));
    console.log(JSON.stringify(results));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

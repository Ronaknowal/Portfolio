const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const normalize = value => value.replace(/\s+/g, ' ').trim();
(async () => {
  const { numericalMethodsExamples: examples } = await import(pathToFileURL(path.resolve('src/learn/data/numerical-methods-examples.js')));
  const directory = path.resolve('scratch/numerical-methods-browser');
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [], errors = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      page.on('pageerror', error => errors.push(error.message));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/numerical-methods-finite-differences-quadrature-root-finding?module=math-foundations', { waitUntil: 'domcontentloaded', timeout: 60000 });
      const lesson = page.locator('.numerical-methods-lesson');
      await lesson.waitFor(); await page.evaluate(() => document.fonts.ready);
      const answer = lesson.locator('.nm-practice').nth(8).locator(':scope > details > summary').nth(1);
      await answer.focus(); await page.keyboard.press('Enter');
      const programs = [];
      for (const [key, example] of Object.entries(examples)) {
        const block = lesson.locator('.python-example').filter({ has: page.getByRole('heading', { name: example.title, exact: true }) });
        assert.equal(await block.count(), 1);
        const text = normalize(await block.innerText());
        assert(text.includes(normalize(example.code)));
        assert(text.includes(normalize(example.expected)));
        assert(normalize(await block.evaluate(node => node.previousElementSibling.textContent)).includes(normalize(example.question)));
        programs.push(key);
      }
      assert.equal(programs.length, 13);
      const nested = lesson.locator('.python-example').filter({ has: page.getByRole('heading', { name: examples.nestedCalibration.title, exact: true }) });
      const captures = [];
      for (const [target, name] of [[nested, 'nested-program'], [nested.locator('.lesson-note'), 'nested-output'], [lesson.locator('.nm-practice').nth(8), 'formatted-practice']]) {
        await target.evaluate(node => window.scrollTo({ top: node.getBoundingClientRect().top + scrollY - 90, behavior: 'instant' }));
        await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
        await page.waitForTimeout(150);
        const filename = `final-${name}-${width}.png`;
        await page.screenshot({ path: path.join(directory, filename) }); captures.push(filename);
      }
      for (const equation of await lesson.locator('.katex-display').all()) assert(await equation.evaluate(node => node.scrollWidth <= node.clientWidth + 1));
      assert(!(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1)));
      records.push({ width, programs, captures, fonts: await page.evaluate(() => [...document.fonts].filter(font => font.status === 'loaded').map(font => font.family)) });
      await page.close();
    }
    assert.deepEqual(errors, []);
    fs.writeFileSync(path.join(directory, 'final-program-results.json'), JSON.stringify({ checkedAt: new Date().toISOString(), passed: true, records, errors }, null, 2));
    console.log('Final 13 actual programs, exact output/prompt placement and nested-search reading passed at 1440/390/320.');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

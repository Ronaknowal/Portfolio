const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const { pathToFileURL } = require('node:url');
const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const normalize = value => value.replace(/\s+/g, ' ').trim();
(async () => {
  const { recommenderExamples } = await import(pathToFileURL(path.resolve('src/learn/data/recommender-examples.js')));
  const example = recommenderExamples.neighbors;
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const rows = [], errors = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 950 } });
      await page.routeWebSocket('**', socket => socket.close());
      page.on('pageerror', error => errors.push(error.message));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/recommender-systems-collaborative-filtering-matrix-factorization?module=classical-ml-supervised');
      const program = page.locator('.python-example').filter({ has: page.getByRole('heading', { name: example.title, exact: true }) });
      await program.waitFor();
      await page.evaluate(() => document.fonts.ready);
      const fonts = await page.evaluate(() => [...document.fonts].filter(font => font.status === 'loaded').map(font => font.family));
      assert(fonts.some(font => font.includes('Space Grotesk')));
      assert(fonts.some(font => font.includes('JetBrains Mono')));
      const blocks = await program.locator(':scope > div').evaluateAll(nodes => nodes.map(node => [...node.childNodes].filter(child => child.nodeType === Node.TEXT_NODE).map(child => child.textContent).join('')));
      assert.equal(normalize(blocks[0]), normalize(example.code));
      assert.equal(normalize(blocks[1]), normalize(example.expected));
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
      if (width === 390) {
        await program.evaluate(node => window.scrollTo({ top: node.getBoundingClientRect().top + scrollY - 80, behavior: 'instant' }));
        await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
        await page.waitForTimeout(150);
        await page.screenshot({ path: 'scratch/recommender-browser/fallback-final-390.png' });
      }
      rows.push({ width, actualCodeAndOutput: true, fonts });
      await page.close();
    }
    assert.deepEqual(errors, []);
    fs.writeFileSync('docs/teaching/evidence/recommender-fallback-browser.json', JSON.stringify({ checkedAt: new Date().toISOString(), rows, errors, sourceHash: crypto.createHash('sha256').update(fs.readFileSync('src/learn/data/recommender-examples.js')).digest('hex') }, null, 2) + '\n');
    console.log(JSON.stringify(rows));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });

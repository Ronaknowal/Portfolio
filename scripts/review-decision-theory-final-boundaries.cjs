const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      const errors = [];
      page.on('pageerror', error => errors.push(String(error)));
      await page.routeWebSocket('**', socket => socket.close());
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/decision-theory-risk-cost-sensitive-decisions?module=math-foundations');
      const lesson = page.locator('.decision-lesson');
      await lesson.waitFor(); await page.evaluate(() => document.fonts.ready);
      await page.addStyleTag({ content: 'html { scroll-behavior: auto !important; }' });
      assert.equal(await lesson.locator('.decision-investigation').count(), 8);
      const scope = lesson.getByRole('region', { name: 'Hold the state fixed, then average the states', exact: true });
      const prior = scope.getByRole('slider');
      await prior.fill('0.2'); await prior.dispatchEvent('input');
      assert((await scope.locator('.decision-result').innerText()).includes('Best rules: Always low or Follow signal'));
      const capstone = lesson.getByRole('region', { name: 'Test first; reallocate after the result', exact: true });
      assert.equal(await capstone.locator('.decision-candidate-costs p').filter({ hasText: 'optimal' }).count(), 2);
      assert((await lesson.getByRole('region', { name: 'Change the consequence criterion explicitly', exact: true }).locator('.decision-result').innerText()).includes('CVaR 55;'));
      const fallback = lesson.getByRole('region', { name: 'Add a real fallback, then share limited slots', exact: true });
      assert((await fallback.innerText()).includes('This strip samples 101 probabilities'));
      await fallback.locator('.decision-region').evaluate(node => window.scrollTo({ top: node.getBoundingClientRect().top + scrollY - 90, behavior: 'instant' }));
      await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
      await page.screenshot({ path: `scratch/decision-theory-browser/final-fallback-sampling-${width}.png` });
      const summary = lesson.locator('summary').filter({ hasText: 'Derive the finite hinge formula and its ties' });
      await summary.focus(); await page.keyboard.press('Enter');
      const details = summary.locator('..');
      assert((await details.innerText()).includes('no finite threshold need attain it'));
      assert((await details.innerText()).includes('at any threshold at or below the smallest loss'));
      await details.evaluate(node => window.scrollTo({ top: node.getBoundingClientRect().top + scrollY - 90, behavior: 'instant' }));
      await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
      await page.screenshot({ path: `scratch/decision-theory-browser/final-cvar-endpoint-${width}.png` });
      const math = await details.locator('.katex-display').evaluateAll(nodes => nodes.map(node => ({ width: node.clientWidth, scroll: node.scrollWidth })));
      assert(math.every(node => node.scroll <= node.width + 1));
      assert(!(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1)));
      assert.deepEqual(errors, []);
      const fonts = await page.evaluate(() => [...document.fonts].filter(font => font.status === 'loaded').map(font => font.family));
      assert(fonts.some(font => font.includes('Space Grotesk')));
      records.push({ width, allEightCurrentLabsRendered: true, exactRuleTie: true, coOptimalTests: true, atomTail: true, endpointKeyboardAndRead: true, fonts, math, errors, capture: `final-cvar-endpoint-${width}.png` });
      await page.close();
    }
    const modelReport = JSON.parse(fs.readFileSync('scratch/decision-theory-review/model-results.json', 'utf8'));
    const sources = modelReport.sources.map(source => ({ path: source.path, sha256: crypto.createHash('sha256').update(fs.readFileSync(source.path)).digest('hex') }));
    fs.writeFileSync('scratch/decision-theory-browser/final-boundaries.json', JSON.stringify({ checkedAt: new Date().toISOString(), passed: true, records, sources, scope: 'After the final Object.hasOwn validation-only amendment: current-source rendering, valid-array interactions, exact ties, CVaR endpoint statement and narrow formula at all three widths. Full interaction/read evidence remains in results.json.' }, null, 2));
    console.log('Final current-source boundary reading and interaction checks passed.');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

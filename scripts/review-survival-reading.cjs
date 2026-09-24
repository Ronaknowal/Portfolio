const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const directory = path.resolve('scratch/survival/browser');
(async () => {
  const original = JSON.parse(fs.readFileSync(path.join(directory, 'results.json'), 'utf8'));
  const sourceHashes = Object.fromEntries(Object.keys(original.sourceHashes).map(file => [file, crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex')]));
  for (const file of Object.keys(sourceHashes)) if (!file.endsWith('/SurvivalLabs.jsx') && !file.includes('/topics/')) assert.equal(sourceHashes[file], original.sourceHashes[file]);
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [], errors = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      page.on('pageerror', error => errors.push(error.message));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/survival-analysis-cox-regression-kaplan-meier-hazard-models?module=classical-ml', { waitUntil: 'domcontentloaded' });
      const lesson = page.locator('.survival-lesson');
      await lesson.waitFor(); await page.evaluate(() => document.fonts.ready);
      const fonts = await page.evaluate(() => [...document.fonts].filter(font => font.status === 'loaded').map(font => font.family));
      assert(fonts.some(font => font.includes('Space Grotesk')) && fonts.some(font => font.includes('JetBrains Mono')));
      const captures = [];
      async function shot(target, name) {
        await target.evaluate(node => window.scrollTo({ top: node.getBoundingClientRect().top + scrollY - 90, behavior: 'instant' }));
        await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
        await page.waitForTimeout(500);
        const file = `${name}-${width}.png`;
        await page.screenshot({ path: path.join(directory, file) });
        captures.push(file);
      }
      const prereqs = await lesson.locator('.lesson-intro > p a').evaluateAll(nodes => nodes.map(node => node.getAttribute('href')));
      assert(prereqs.some(href => href.includes('/linear-logistic-regression?')) && prereqs.some(href => href.includes('/hypothesis-testing-confidence-intervals?')));
      for (const summary of await lesson.locator('details > summary').all()) { if (!(await summary.evaluate(node => node.parentElement.open))) { await summary.focus(); await page.keyboard.press('Enter'); } }
      const equations = await lesson.locator('.katex-display').evaluateAll(nodes => nodes.map((node, index) => ({ index, text: node.textContent, scroll: node.scrollWidth, client: node.clientWidth })));
      const overflow = equations.filter(row => row.scroll > row.client + 2);
      const geometry = await lesson.evaluate(node => { const bounds = node.getBoundingClientRect(); return [...node.querySelectorAll('p,h2,h3,summary,svg,figure,input,select,button')].filter(element => element.getClientRects().length && !element.closest('.lesson-table-wrap,.sv-table-scroll')).filter(element => { const box = element.getBoundingClientRect(); return box.left < bounds.left - 3 || box.right > bounds.right + 3; }).map(element => element.textContent.slice(0,120)); });
      assert.deepEqual(geometry, []);
      assert.equal(await lesson.locator('.katex-error').count(), 0);
      const cox = lesson.getByRole('region', { name: 'Let the event compete inside its risk set', exact: true });
      assert(Math.abs(Number(await cox.getByLabel('Cox coefficient beta', { exact: false }).inputValue()) - Math.log(2)) < 1e-14);
      const bars = await cox.locator('.sv-risk-weights > div').evaluateAll(nodes => nodes.map(node => { const bar = node.querySelector('.sv-weight-track span'); return { label: node.innerText, color: getComputedStyle(bar).backgroundColor, width: bar.getBoundingClientRect().width, height: bar.getBoundingClientRect().height }; }));
      assert(bars.every(bar => bar.width > 0 && bar.height > 0));
      assert.equal(bars[0].color, 'rgb(231, 185, 74)');
      if (width === 1440) { await shot(lesson.locator('h2').nth(7), 'final-cox-reading'); await shot(lesson.locator('h2').nth(12), 'final-competing-reading'); }
      if (width === 390) { await shot(cox.locator('.sv-risk-weights'), 'final-cox-weight-paint'); await shot(lesson.locator('.katex-display').nth(4), 'final-hazard-equation'); await shot(lesson.locator('.katex-display').nth(13), 'final-censor-equation'); }
      if (width === 320) { await shot(cox.locator('.sv-risk-weights'), 'final-cox-weight-paint'); await shot(lesson.locator('.katex-display').nth(6), 'final-logrank-equation'); await shot(lesson.locator('.katex-display').nth(9), 'final-score-equation'); await shot(lesson.locator('.katex-display').nth(14), 'final-discrete-equation'); }
      records.push({ width, fonts, prereqs, equations, overflow, geometry, bars, captures });
      await page.close();
    }
    assert.deepEqual(errors, []);
    fs.writeFileSync(path.join(directory,'final-reading-results.json'),JSON.stringify({checkedAt:new Date().toISOString(),sourceHashes,records,errors,passed:records.every(row=>row.overflow.length===0),scope:'Final mathematical line layout, prose, linked prerequisites and actual initial Cox bar painting. Earlier26states/69keyboard actions per width retained for unchanged behavioral mechanisms.'},null,2)+'\n');
    console.log(JSON.stringify(records.map(row=>({width:row.width,overflow:row.overflow,captures:row.captures})),null,2));
  } finally { await browser.close(); }
})().catch(error=>{console.error(error);process.exit(1);});

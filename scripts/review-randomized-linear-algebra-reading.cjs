const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const directory = 'scratch/randomized-linear-algebra-review';

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/randomized-linear-algebra?module=math-foundations');
      const lesson = page.locator('.randomized-linear-algebra-lesson');
      await lesson.waitFor();
      await page.evaluate(() => document.fonts.ready);
      const links = lesson.locator('.lesson-intro a[href^="#"]');
      for (const index of [0, 7]) {
        const link = links.nth(index);
        const hash = await link.getAttribute('href');
        await link.focus();
        await page.keyboard.press('Enter');
        assert.equal(new URL(page.url()).hash, hash);
        assert.equal(await page.evaluate(id => Boolean(document.getElementById(id)), hash.slice(1)), true);
      }
      const hints = lesson.locator('.lesson-check details summary');
      assert.ok(await hints.count());
      await hints.first().focus();
      await page.keyboard.press('Enter');
      assert.equal(await hints.first().evaluate(node => node.parentElement.open), true);
      await lesson.locator('details').evaluateAll(nodes => nodes.forEach(node => node.open = true));
      const mathOverflow = await lesson.locator('.katex-display').evaluateAll(nodes => nodes.filter(node => node.scrollWidth > node.clientWidth + 2).map(node => ({ text: node.textContent, width: node.clientWidth, scroll: node.scrollWidth })));
      assert.deepEqual(mathOverflow, [], 'Displayed equation overflow with all deeper content open');
      assert.equal(await lesson.locator('.katex-error').count(), 0);
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
      for (const section of [1, 3, 5, 6, 7, 8]) {
        const heading = lesson.getByRole('heading', { level: 2, name: new RegExp(`^${section}\\.`) });
        await heading.evaluate(node => window.scrollTo(0, window.scrollY + node.getBoundingClientRect().top - 90));
        await page.screenshot({ path: `${directory}/final-reading-${section}-${width}.png` });
      }
      await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = 'hidden'));
      for (const [name, selector] of [
        ['intro', '.lesson-intro'], ['sources', '.lesson-sources'],
        ['range-bound', '.katex-display'],
      ]) {
        const locator = name === 'range-bound' ? lesson.locator(selector).filter({ hasText: 'τ' }) : lesson.locator(selector);
        await locator.screenshot({ path: `${directory}/final-${name}-${width}.png` });
      }
      if (width === 320) {
        for (const [index, name] of ['probe', 'spectrum', 'rows', 'trace'].entries()) {
          await lesson.locator('.rla-lab').nth(index).screenshot({ path: `${directory}/final-${name}-${width}.png` });
        }
      }
      await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = ''));
      assert.equal(await lesson.locator('.lesson-sources a').count(), 6);
      assert.ok((await lesson.innerText()).includes('Multivariate Calculus & Gradients'));
      assert.deepEqual(errors, []);
      results.push({ width, openDetailsMath: 'pass', keyboardAnchors: 2, readingSections: 6, sources: 6, errors });
      await page.close();
    }
    fs.writeFileSync(path.join(directory, 'final-reading-results.json'), JSON.stringify({ at: new Date().toISOString(), results }, null, 2));
    console.log(JSON.stringify(results));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

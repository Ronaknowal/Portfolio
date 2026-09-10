const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const directory = 'scratch/mutual-information-browser-review-fonts';

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  for (const width of [1440, 390, 320]) {
    const page = await browser.newPage({ viewport: { width, height: 1000 } });
    await page.routeWebSocket('**', socket => socket.close());
    const errors = [], failedRequests = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('requestfailed', request => failedRequests.push(request.url()));
    await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/mutual-information-information-bottleneck?module=math-foundations', { waitUntil: 'networkidle' });
    await page.locator('.mutual-information-lesson').waitFor();
    await page.evaluate(() => document.fonts.ready);
    const sectionLinks = await page.locator('.lesson-intro nav a').evaluateAll(elements => elements.map(element => ({ text: element.textContent, target: element.hash, exists: !!document.getElementById(element.hash.slice(1)) })));
    assert.equal(sectionLinks.length, 10);
    assert.ok(sectionLinks.every(link => link.exists));
    const project = page.locator('.mi-practice').last();
    assert.equal(await page.locator('.mi-practice').count(), 10);
    const solution = project.getByText('Show explained solution', { exact: true });
    await solution.focus();
    await solution.press('Enter');
    assert.match(await project.innerText(), /0, −.112288 and .887712/);
    await project.evaluate(element => window.scrollTo(0, window.scrollY + element.getBoundingClientRect().top - 175));
    await page.screenshot({ path: `${directory}/final-project-reading-${width}.png` });
    for (const [label, text] of [['free-energy', 'For fixed q, a minimizing r'], ['variational-gap', 'C is the decoder’s expected cross-entropy'], ['continuous-limit', 'At σ=0, the observation']]) {
      const prose = page.locator('.mutual-information-lesson p').filter({ hasText: text }).first();
      await prose.evaluate(element => window.scrollTo(0, window.scrollY + element.getBoundingClientRect().top - 430));
      await page.screenshot({ path: `${directory}/final-${label}-reading-${width}.png` });
    }
    const samples = page.getByRole('region', { name: 'Mutual information estimation investigation', exact: true });
    await samples.getByRole('combobox', { name: 'Categories per variable', exact: true }).selectOption('8');
    const counts = samples.getByRole('region', { name: 'Observed pair counts', exact: true });
    let horizontalScroll = null;
    if (width === 320) {
      await counts.focus();
      for (let i = 0; i < 8; i += 1) await counts.press('ArrowRight');
      await page.waitForTimeout(250);
      horizontalScroll = await counts.evaluate(element => ({ offset: element.scrollLeft, active: document.activeElement === element, width: element.clientWidth, total: element.scrollWidth }));
      assert.ok(horizontalScroll.offset > 0 && horizontalScroll.active);
      await counts.evaluate(element => window.scrollTo(0, window.scrollY + element.getBoundingClientRect().top - 180));
      await page.screenshot({ path: `${directory}/final-counts-keyboard-scroll-${width}.png` });
    }
    const geometry = await page.evaluate(() => {
      const root = document.querySelector('.mutual-information-lesson');
      return {
        viewport: document.documentElement.clientWidth, page: document.documentElement.scrollWidth,
        equationOverflow: [...root.querySelectorAll('.katex-display')].filter(element => element.scrollWidth > element.parentElement.clientWidth + 1).length,
        equationErrors: root.querySelectorAll('.katex-error').length,
        loadedOriginalFont: [...document.fonts].some(font => font.family === 'Space Grotesk' && font.status === 'loaded'),
      };
    });
    assert.equal(errors.length, 0);
    assert.equal(failedRequests.length, 0);
    assert.equal(geometry.page, width);
    assert.equal(geometry.equationOverflow, 0);
    assert.equal(geometry.equationErrors, 0);
    assert.ok(geometry.loadedOriginalFont);
    results.push({ width, sectionLinks, practice: 10, horizontalScroll, geometry, errors, failedRequests });
    fs.writeFileSync(`${directory}/final-reading-results.json`, JSON.stringify(results, null, 2));
    await page.close();
  }
  await browser.close();
  console.log('Final font-enabled reading passed at 1440/390/320, including the complete changed-task practice and actual keyboard horizontal table scrolling.');
})().catch(error => { console.error(error); process.exit(1); });

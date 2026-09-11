const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 320, height: 1000 } });
    await page.routeWebSocket('**', socket => socket.close());
    await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/stochastic-processes-markov-chains-brownian-motion-poisson?module=math-foundations', { waitUntil: 'networkidle' });
    await page.locator('.stochastic-processes-lesson').waitFor();
    await page.evaluate(() => document.fonts.ready);
    const markov = page.getByRole('region', { name: 'Markov probability propagation', exact: true });
    const contributions = markov.locator('.process-flow-options button');
    for (let index = 0; index < 4; index += 1) {
      await contributions.nth(index).focus();
      await page.keyboard.press('Enter');
      assert.equal(await contributions.nth(index).getAttribute('aria-pressed'), 'true');
    }
    const reserve = page.getByRole('region', { name: 'First passage and absorption', exact: true });
    await reserve.getByLabel(/^Upper reserve boundary/).selectOption('8');
    await reserve.getByLabel(/^Starting reserve/).selectOption('8');
    const scroller = reserve.locator('.process-wide');
    await scroller.focus();
    await page.keyboard.press('ArrowRight');
    await page.waitForTimeout(200);
    const keyboardOffset = await scroller.evaluate(node => node.scrollLeft);
    assert.ok(keyboardOffset > 0);
    await scroller.evaluate(node => { node.scrollLeft = node.scrollWidth; });
    await page.addStyleTag({ content: '.learn-nav { visibility:hidden !important; }' });
    await scroller.screenshot({ path: 'scratch/stochastic-processes-browser/final-reserve-scrolled-320.png' });
    const result = { checkedAt: new Date().toISOString(), width: 320, keyboardContributions: 4,
      keyboardScrollOffset: keyboardOffset, lastStatesScreenshot: 'final-reserve-scrolled-320.png',
      documentWidth: await page.evaluate(() => document.documentElement.scrollWidth) };
    assert.equal(result.documentWidth, 320);
    fs.writeFileSync('scratch/stochastic-processes-browser/detail-navigation-results.json', JSON.stringify(result, null, 2));
    console.log(JSON.stringify(result, null, 2));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

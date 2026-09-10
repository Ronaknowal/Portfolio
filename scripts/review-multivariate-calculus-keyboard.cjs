const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const directory = path.resolve(__dirname, '../scratch/multivariate-browser');
  const results = [];
  for (const width of [1440, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 1000 } });
    await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/multivariate-calculus-gradients');
    const lesson = page.locator('.multivariate-lesson');
    await lesson.locator('[data-lab="local-gradient"]').waitFor();
    const descent = lesson.locator('[data-lab="gradient-descent"]');
    await descent.getByRole('button', { name: 'Next step', exact: true }).focus();
    await page.keyboard.press('Enter');
    assert.equal(await descent.locator('[data-result="descent-point"]').innerText(), '(2.4, -1.2)');
    await descent.getByRole('button', { name: 'Previous step', exact: true }).focus();
    await page.keyboard.press('Space');
    assert.equal(await descent.locator('[data-result="descent-point"]').innerText(), '(3, -2)');
    const select = lesson.locator('[data-lab="approach-paths"]').getByLabel(/^Approach path/);
    await select.focus(); await page.keyboard.press('Home'); await page.keyboard.press('ArrowDown');
    assert.equal(await select.inputValue(), 'line');
    const baseInput = lesson.locator('[data-lab="local-gradient"]').getByLabel('Base x', { exact: true });
    await baseInput.focus(); await page.keyboard.press('Control+A'); await page.keyboard.type('-1'); await page.keyboard.press('Enter');
    assert.ok((await lesson.locator('[data-lab="local-gradient"]').innerText()).includes('(-1, 1) / (-2, 4)'));
    const setup = lesson.getByText('For the optional programs, use Python 3 with NumPy installed.', { exact: false });
    assert.equal(await setup.count(), 1);
    await setup.evaluate(node => window.scrollTo(0, node.getBoundingClientRect().top + scrollY - 100));
    await page.screenshot({ path: path.join(directory, `program-setup-${width}.png`) });
    assert.equal(await lesson.locator('.katex-error').count(), 0);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
    results.push({ width, enterButton: true, spaceButton: true, keyboardSelect: true, keyboardForm: true, completeProgramSetup: true });
    await page.close();
  }
  await browser.close();
  fs.writeFileSync(path.join(directory, 'keyboard-results.json'), JSON.stringify({ status: 'passed', checkedAt: new Date().toISOString(), results }, null, 2));
  console.log(JSON.stringify(results));
})().catch(error => { console.error(error); process.exit(1); });

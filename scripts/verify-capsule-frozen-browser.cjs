// Focused follow-up for visible grid guidance; preserve unchanged full-suite evidence.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const crypto = require('node:crypto');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const directory = 'docs/teaching/evidence/capsule-browser';
const reportPath = `${directory}/report.json`;
const base = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4194';
const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
(async () => {
  const previous = JSON.parse(fs.readFileSync(reportPath, 'utf8'));
  assert.equal(previous.passed, true);
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const captures = [];
  try {
    const page = await browser.newPage({ viewport: { width: 390, height: 1000 }, reducedMotion: 'reduce' });
    await require('./lib/lesson-browser-fonts.cjs')(page);
    await page.goto(`${base}/learn/path/full-curriculum/capsule-networks?module=deep-learning-fundamentals`, { waitUntil: 'commit' });
    await page.locator('.capsule-frozen > summary').click();
    const lab = page.locator('[data-lab="capsule-frozen"]');
    await lab.locator('[data-result="frozen"]').waitFor();
    for (const width of [390, 320]) {
      await page.setViewportSize({ width, height: 1000 });
      await page.evaluate(() => document.fonts.ready);
      assert.match(await lab.innerText(), /All 8 columns remain available.*scroll the grid horizontally/);
      const region = lab.getByRole('region', { name: 'Editable pixel grid; scroll horizontally or select with the row and column controls below', exact: true });
      assert.ok(await region.evaluate(node => node.scrollWidth > node.clientWidth));
      await region.focus(); await region.press('ArrowRight');
      await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
      assert.ok(await region.evaluate(node => node.scrollLeft > 0), 'Focused pixel grid must scroll with the keyboard');
      await region.getByRole('button', { name: /^Pixel row 0, column 7:/ }).click();
      assert.equal(await lab.getByRole('spinbutton', { name: 'Selected pixel row', exact: true }).inputValue(), '0');
      assert.equal(await lab.getByRole('spinbutton', { name: 'Selected pixel column', exact: true }).inputValue(), '7');
      assert.ok(await region.evaluate(node => node.scrollLeft > 0));
      await lab.getByRole('spinbutton', { name: 'Selected pixel intensity', exact: true }).fill('0.3');
      assert.equal(await lab.getByRole('slider', { name: 'Selected pixel intensity slider', exact: true }).inputValue(), '0.3');
      assert.doesNotMatch(await lab.locator('[data-result="frozen"]').innerText(), /Maximum score change 0\. Exact/);
      await lab.getByRole('button', { name: 'Reset frozen investigation', exact: true }).click();
      await region.evaluate(node => { node.scrollLeft = 0; });
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
      assert.equal(await lab.locator('.capsule-pixels-editable button').evaluateAll(items => items.filter(item => item.getBoundingClientRect().width < 43.5 || item.getBoundingClientRect().height < 43.5).length), 0);
      const file = `${directory}/capsule-frozen-${width}.png`;
      await lab.screenshot({ path: file, style: '.workspace-nav, .learning-skip { visibility: hidden !important; }' });
      const bytes = fs.readFileSync(file); captures.push({ file, bytes: bytes.length, sha256: hash(bytes) });
    }
    const author = JSON.parse(fs.readFileSync('docs/teaching/evidence/capsule-author.json', 'utf8'));
    previous.sourceHashes = author.sources;
    previous.captures = previous.captures.map(capture => captures.find(newCapture => newCapture.file === capture.file) || capture);
    previous.visibleGridGuidanceRecheck = { passed: true, checkedAt: new Date().toISOString(), widths: [390, 320], checks: ['Visible local-scroll and numeric-selection guidance', 'Actual keyboard horizontal scrolling and pointer selection of last column update row/column', 'Changed real pixel inference and exact range/numeric value', 'Reset,44px cells and no document overflow'], limit: 'Only explanatory hint changed; prior source reconstructed exactly before receipt refresh. Other full-suite model/source/retry interactions are reused unchanged.' };
    fs.writeFileSync(reportPath, JSON.stringify(previous, null, 2) + '\n');
    console.log('PASS: focused frozen-grid guidance and actual last-column interaction at390/320.');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

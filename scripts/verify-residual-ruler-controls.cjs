// Regression coverage for the actual editable forecast diamonds, not click-only presets.
const fs = require('node:fs');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const base = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4192';
const topic = 'evaluation-metrics-precision-recall-f1-auc-roc-ap-r-mae';
const hash = path => createHash('sha256').update(fs.readFileSync(path)).digest('hex');
const sources = [
  'src/learn/components/lesson-labs/EvaluationMetricsFigures.jsx',
  'src/learn/components/lesson-labs/EvaluationMetricsLabs.jsx',
  'src/learn/components/lesson-labs/evaluation-metrics.css',
  'src/learn/data/evaluation-metrics-models.js',
  `src/learn/data/topics/${topic}.jsx`,
  `docs/teaching/drafts/${topic}/live-exploration.md`,
  `docs/teaching/drafts/${topic}/visual-specifications.md`,
  'scripts/verify-residual-ruler-controls.cjs',
];
const sourceHashes = Object.fromEntries(sources.map(path => [path, hash(path)]));
const fonts = JSON.parse(fs.readFileSync('scratch/kmeans-revision-review/fonts/manifest.json', 'utf8').replace(/^\uFEFF/, ''));
const results = [], screenshots = [], errors = [], targetDimensions = [];
const settle = page => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
const value = async slider => Number(await slider.getAttribute('aria-valuenow'));
async function screenPoint(slider, fraction) {
  return slider.evaluate((handle, fraction) => {
    const svg = handle.ownerSVGElement;
    const point = svg.createSVGPoint();
    point.x = 24 + fraction * 252; point.y = 26;
    const screen = point.matrixTransform(svg.getScreenCTM());
    return { x: screen.x, y: screen.y };
  }, fraction);
}
async function dragTo(page, slider, target, offset = 0) {
  await slider.scrollIntoViewIfNeeded();
  const min = Number(await slider.getAttribute('aria-valuemin'));
  const max = Number(await slider.getAttribute('aria-valuemax'));
  const start = await screenPoint(slider, (await value(slider) - min) / (max - min));
  const end = await screenPoint(slider, (target - min) / (max - min));
  await page.mouse.move(start.x + offset, start.y);
  await page.mouse.down();
  await page.mouse.move(end.x + offset, end.y, { steps: 6 });
  await settle(page);
  assert.equal(await value(slider), target, 'actual pointer movement updates forecast');
  assert.equal(Number(await slider.getAttribute('aria-valuemin')), min, 'domain stays fixed during drag');
  assert.equal(Number(await slider.getAttribute('aria-valuemax')), max, 'domain stays fixed during drag');
  await page.mouse.up();
  await settle(page);
}
async function snapshot(page, locator, name) {
  const path = `docs/teaching/evidence/screenshots/residual-ruler-${name}.png`;
  const style = await page.addStyleTag({ content: '.learn-nav{visibility:hidden!important}' });
  try { await locator.screenshot({ path }); } finally { await style.evaluate(node => node.remove()); }
  screenshots.push({ path, sha256: hash(path) });
}
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const context = await browser.newContext({ viewport: { width: 1366, height: 950 }, hasTouch: true });
  await context.route('https://fonts.googleapis.com/**', route => route.fulfill({ path: fonts.stylesheet, contentType: 'text/css', headers: { 'access-control-allow-origin': '*' } }));
  await context.route('https://fonts.gstatic.com/**', route => fonts.files[route.request().url()] ? route.fulfill({ path: fonts.files[route.request().url()], contentType: 'font/ttf', headers: { 'access-control-allow-origin': '*' } }) : route.continue());
  const page = await context.newPage();
  page.on('pageerror', error => errors.push(error.message));
  try {
    await page.goto(`${base}/learn/path/full-curriculum/${topic}?module=classical-ml`, { waitUntil: 'domcontentloaded' });
    const figure = page.locator('[data-metric-figure=residuals]');
    await figure.waitFor();
    await page.evaluate(() => document.fonts.ready);
    assert.equal(await figure.getByRole('slider').count(), 10);
    const a5 = figure.getByRole('slider', { name: 'A T5 forecast', exact: true });
    const b5 = figure.getByRole('slider', { name: 'B T5 forecast', exact: true });
    const summary = figure.getByRole('region', { name: 'Residual summary and actual fitted baseline' });
    await dragTo(page, a5, 8, 5);
    assert.equal(await value(b5), 8);
    let cells = await summary.locator('tbody tr').first().locator('td').allTextContents();
    assert.deepEqual(cells, ['0.4', '0.8', '0.894427', '0', '4', '0.92']);
    const aSquares = figure.locator('.metric-paired > section').first();
    assert.equal(await aSquares.getByLabel('Squared residual area 4', { exact: true }).count(), 1);
    const side = await aSquares.getByLabel('Squared residual area 4', { exact: true }).evaluate(node => node.getBoundingClientRect().width);
    assert.ok(Math.abs(side - 64 * 2 / 14) < .05);
    await dragTo(page, b5, 10);
    assert.equal(await value(a5), 8, 'predictor states remain independent');
    assert.match(await figure.innerText(), /T5.*observed 10/s);
    await snapshot(page, figure, 'desktop-edited');
    results.push('desktop pointer coordinates/grab offset, A/B independence, fixed scales, linked exact errors/areas/metrics');

    await a5.focus();
    await a5.press('ArrowRight'); assert.equal(await value(a5), 8.25);
    await a5.press('PageDown'); assert.equal(await value(a5), 7.25);
    await a5.press('Home'); assert.equal(await value(a5), -2);
    await a5.press('ArrowLeft'); assert.equal(await value(a5), -2);
    await a5.press('End'); assert.equal(await value(a5), 12);
    await a5.press('ArrowRight'); assert.equal(await value(a5), 12);
    assert.equal(await a5.evaluate(node => node === document.activeElement), true);
    await figure.getByRole('button', { name: 'Match all observations' }).click();
    cells = await summary.locator('tbody tr').first().locator('td').allTextContents();
    assert.deepEqual(cells, ['0', '0', '0', '0', '0', '1']);
    await figure.getByRole('button', { name: 'Reset original forecasts' }).click();
    assert.equal(await value(a5), 4); assert.equal(await value(b5), 8);
    assert.match(await figure.innerText(), /negative duration is physically invalid/);
    results.push('keyboard arrows/Page/Home/End, boundary clamping/focus, exact zero null, reset and physical-duration warning');

    await page.setViewportSize({ width: 390, height: 850 });
    await dragTo(page, a5, 9);
    await snapshot(page, figure, 'phone-edited');
    await a5.scrollIntoViewIfNeeded();
    const start = await screenPoint(a5, (9 + 2) / 14), end = await screenPoint(a5, (7 + 2) / 14);
    const cdp = await context.newCDPSession(page);
    await cdp.send('Input.dispatchTouchEvent', { type: 'touchStart', touchPoints: [{ x: start.x, y: start.y }] });
    await cdp.send('Input.dispatchTouchEvent', { type: 'touchMove', touchPoints: [{ x: end.x, y: end.y }] });
    await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
    await settle(page); assert.equal(await value(a5), 7, 'actual touch drag updates without scroll interception');
    await cdp.detach();
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1);
    assert.equal(overflow, false);
    await page.setViewportSize({ width: 320, height: 850 });
    await dragTo(page, b5, 6);
    for (const width of [320, 390, 1366]) {
      await page.setViewportSize({ width, height: 850 });
      const bounds = await a5.boundingBox();
      targetDimensions.push({ viewport: width, width: bounds.width, height: bounds.height });
    }
    console.log(JSON.stringify({ targetDimensions }));
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
    results.push('390px pointer and actual touch gestures; 320px pointer; responsive coordinate agreement and no page overflow');

    const lab = page.locator('[data-metric-lab=regression]');
    assert.match(await lab.innerText(), /Negative differences mean A has less error/, 'production must contain latest residual explanation');
    const labA5 = lab.getByRole('slider', { name: 'A T5 forecast', exact: true });
    await dragTo(page, labA5, 8);
    assert.equal(await lab.getByLabel('T5 A', { exact: true }).inputValue(), '8');
    await lab.getByLabel('T5 A', { exact: true }).fill('7.1');
    await labA5.press('ArrowRight');
    assert.equal(await lab.getByLabel('T5 A', { exact: true }).inputValue(), '7.35', 'keyboard increments preserve off-grid exact numeric edits');
    await lab.getByLabel('T5 A', { exact: true }).fill('8');
    await lab.getByLabel('Display unit', { exact: true }).selectOption('60');
    await labA5.press('ArrowLeft');
    assert.equal(await lab.getByLabel('T5 A', { exact: true }).inputValue(), '7.75');
    const labRow = lab.locator('.metric-residual').filter({ has: page.getByRole('slider', { name: 'A T5 forecast', exact: true }) });
    assert.match(await labRow.innerText(), /observed 600.*forecast 465/s);
    assert.match(await labRow.innerText(), /Residual 135.*squared error 18225/s);
    await lab.getByLabel('T5 A', { exact: true }).fill('');
    assert.equal(await lab.getByRole('alert').count(), 1);
    assert.equal(await lab.getByRole('slider').count(), 0);
    await lab.getByLabel('T5 A', { exact: true }).fill('8');
    assert.equal(await lab.getByRole('slider').count(), 10);
    await lab.getByRole('button', { name: 'All targets and predictions = 4', exact: true }).click();
    assert.match(await lab.innerText(), /R² undefined: the evaluation target has zero variance/);
    await lab.getByRole('button', { name: 'Reset residual investigation', exact: true }).click();
    assert.equal(await lab.getByLabel('Display unit', { exact: true }).inputValue(), '1');
    assert.equal(await value(labA5), 4);
    results.push('full lab pointer/numeric synchronization, seconds conversion, invalid-input suppression/recovery, constant-target undefined R² and reset');
    assert.deepEqual(errors, []);
    for (const path of sources) assert.equal(hash(path), sourceHashes[path], `source changed during verification: ${path}`);
    const receipt = { status: 'passed', checkedAt: new Date().toISOString(), base, topic, sourceHashes, checks: results, screenshots, targetDimensions, browserErrors: errors, limitations: 'This receipt covers residual ruler controls and linked results; it is not a new whole-topic numerical/data audit.' };
    fs.writeFileSync('docs/teaching/evidence/residual-ruler-controls-browser.json', JSON.stringify(receipt, null, 2) + '\n');
    console.log(JSON.stringify({ status: 'passed', checks: results.length, screenshots: screenshots.length }));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

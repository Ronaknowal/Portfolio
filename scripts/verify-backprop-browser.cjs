// Scoped live/browser evidence. Root supplies the production preview; an author
// run sets BACKPROP_EVIDENCE=backprop-browser-author and BACKPROP_PREVIEW=development.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');

const base = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4194';
const evidenceName = process.env.BACKPROP_EVIDENCE || 'backprop-browser';
const outputDirectory = `docs/teaching/evidence/${evidenceName}-screenshots`;
const route = `${base}/learn/path/full-curriculum/backpropagation-automatic-differentiation?module=deep-learning-fundamentals`;
const sources = [
  'src/learn/data/topics/backprop.jsx', 'src/learn/data/backprop-models.js', 'src/learn/data/backprop-training.js',
  'src/learn/components/lesson-labs/BackpropShared.jsx', 'src/learn/components/lesson-labs/BackpropLabs.jsx',
  'src/learn/components/lesson-labs/BackpropFigures.jsx', 'src/learn/components/lesson-labs/backprop-labs.css',
  'scripts/verify-backprop-browser.cjs',
];
const record = { preview: process.env.BACKPROP_PREVIEW || 'production', route, timestamp: new Date().toISOString(), assertions: 0, screenshots: [], errors: [], widths: [], sourceHashes: Object.fromEntries(sources.map(file => [file, createHash('sha256').update(fs.readFileSync(file)).digest('hex')])) };
function check(value, description) { assert.ok(value, description); record.assertions++; }
function near(actual, expected, description, tolerance = 1e-9) { check(Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)), `${description}: ${actual} != ${expected}`); }
async function readNumber(locator, attribute) { return Number(await locator.getAttribute(attribute)); }
async function screenshot(locator, name) {
  await locator.scrollIntoViewIfNeeded();
  await locator.page().waitForTimeout(80);
  const destination = `${outputDirectory}/${name}.png`;
  // Only global chrome is hidden during element capture: fixed navigation can
  // otherwise paint across the middle of a diagram taller than the viewport.
  const captureStyle = await locator.page().addStyleTag({ content: '.learn-nav, .reader-sidebar { visibility: hidden !important; }' });
  try { await locator.screenshot({ path: destination }); }
  finally { await captureStyle.evaluate(element => element.remove()); }
  record.screenshots.push(destination);
}
async function numberInput(page, name, value) {
  await page.getByRole('spinbutton', { name: `${name} — exact value`, exact: true }).fill(String(value));
}

(async () => {
  fs.mkdirSync(outputDirectory, { recursive: true });
  const browser = await chromium.launch({ channel: process.env.PLAYWRIGHT_CHANNEL || 'msedge', headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1366, height: 950 } });
    const fontManifestPath = process.env.LEARNING_FONT_FIXTURES || 'scratch/kmeans-revision-review/fonts/manifest.json';
    if (fs.existsSync(fontManifestPath)) {
      const fonts = JSON.parse(fs.readFileSync(fontManifestPath, 'utf8'));
      await page.route(url => url.href === fonts.stylesheetUrl, request => request.fulfill({ path: fonts.stylesheet, contentType: 'text/css' }));
      for (const [url, file] of Object.entries(fonts.files)) await page.route(url, request => request.fulfill({ path: file, contentType: 'font/ttf', headers: { 'access-control-allow-origin': '*' } }));
      record.fonts = { method: 'Unmodified retained fonts fulfilled locally; application styling unchanged.', manifest: fontManifestPath, hashes: Object.fromEntries([fonts.stylesheet, ...Object.values(fonts.files)].map(file => [file, createHash('sha256').update(fs.readFileSync(file)).digest('hex')])) };
    }
    page.on('pageerror', error => record.errors.push(error.message));
    page.on('console', message => { if (message.type() === 'error') record.errors.push(message.text()); });
    await page.goto(route, { waitUntil: 'domcontentloaded', timeout: 60000 });
    await page.locator('[data-backprop-lab="fit"]').waitFor({ timeout: 60000 });
    await page.evaluate(() => document.fonts.ready);
    const root = page.locator('.backprop-lesson');
    check(await root.locator('[data-backprop-lab]').count() === 3, 'Exactly the three planned live investigations');
    check(await root.locator('.katex-error').count() === 0, 'Mathematical expressions render');
    check(!/check prediction|submit prediction|your prediction|commit your|unlock answer/i.test(await root.innerText()), 'No learner prediction feature');
    check(await root.locator('details').count() === 16, 'Eight practice hints and solutions retained');
    check(await root.locator('.backprop-section-route a').count() === 8, 'Complete anchored section route');
    const routeTargets = await root.locator('.backprop-section-route a').evaluateAll(links => links.map(link => [...document.querySelectorAll('[id]')].filter(element => element.id === link.getAttribute('href').slice(1)).length));
    check(routeTargets.every(count => count === 1), 'Each section route resolves exactly once');
    check(await root.locator('.backprop-baseline').count() === 3, 'Persistent input-labelled starting references');
    const fitResult = page.locator('[data-fit-result]');
    near(await readNumber(fitResult, 'data-old-loss'), .5, 'Initial MSE');
    near(await readNumber(fitResult, 'data-new-loss'), .17, 'Initial result available');
    await numberInput(page, 'Learning rate η', 1);
    near(await readNumber(fitResult, 'data-new-loss'), 12.5, 'Overshoot updates immediately');
    await screenshot(page.locator('[data-backprop-lab="fit"]'), '1366-fit-overshoot');
    await numberInput(page, 'Learning rate η', 0);
    near(await readNumber(fitResult, 'data-new-loss'), .5, 'No-step invariant');
    await numberInput(page, 'Learning rate η', .373);
    near(await readNumber(fitResult, 'data-new-loss'), .5 - 5 * .373 + 17 * .373 ** 2, 'Non-preset exact edit');
    near(Number(await page.getByRole('slider', { name: 'Learning rate η — slider', exact: true }).inputValue()), .373, 'Slider/exact editor round-trip');
    await numberInput(page, 'Weight w', .731);
    await numberInput(page, 'Weight w', -.83);
    near(await readNumber(fitResult, 'data-weight'), -.83, 'Rapid edits keep current model');
    const weightInput = page.getByRole('spinbutton', { name: 'Weight w — exact value', exact: true });
    await weightInput.fill('');
    check(await weightInput.getAttribute('aria-invalid') === 'true', 'Invalid input identified');
    near(await readNumber(fitResult, 'data-weight'), -.83, 'Invalid edit retains stated last valid result');
    check((await page.locator('[data-backprop-lab="fit"]').innerText()).includes('last valid result'), 'Retained-result explanation is local');
    await page.getByRole('button', { name: 'Reset line fit', exact: true }).click();
    near(Number(await weightInput.inputValue()), 1, 'Reset clears invalid buffer');
    // Real keyboard endpoint and interior gestures for every initially visible slider.
    const sliders = root.getByRole('slider');
    for (let index = 0; index < await sliders.count(); index++) {
      const slider = sliders.nth(index);
      const min = Number(await slider.getAttribute('min'));
      const max = Number(await slider.getAttribute('max'));
      await slider.focus(); await slider.press('Home'); near(Number(await slider.inputValue()), min, `Slider ${index} native minimum`);
      await slider.press('End'); near(Number(await slider.inputValue()), max, `Slider ${index} native maximum`);
      await slider.press('ArrowLeft'); check(Number(await slider.inputValue()) < max, `Slider ${index} keyboard interior`);
      const id = await slider.getAttribute('id');
      const accessible = await page.locator(`label[for="${id}"]`).innerText();
      check(accessible.includes('slider'), `Slider ${index} explicit accessible label`);
    }
    await page.getByRole('button', { name: 'Reset line fit', exact: true }).click();
    const rateSlider = page.getByRole('slider', { name: 'Learning rate η — slider', exact: true });
    await rateSlider.scrollIntoViewIfNeeded(); await page.waitForTimeout(150);
    const rect = await rateSlider.boundingBox();
    await page.mouse.move(rect.x + rect.width * .1, rect.y + rect.height / 2);
    await page.mouse.down(); await page.mouse.move(rect.x + rect.width * .42, rect.y + rect.height / 2, { steps: 8 }); await page.mouse.up();
    const pointerRate = Number(await rateSlider.inputValue());
    check(pointerRate > .3 && pointerRate < .55, 'Real pointer gesture changes the rate');
    near(await readNumber(fitResult, 'data-new-loss'), .5 - 5 * pointerRate + 17 * pointerRate ** 2, 'Pointer gesture changes loss continuously');
    await page.getByRole('button', { name: 'Reset shared graph', exact: true }).click();
    await page.getByRole('button', { name: 'Next backward stage', exact: true }).click();
    await numberInput(page, 'Branch coefficient c', -1);
    const sharedResult = page.locator('[data-shared-result]');
    for (const point of [1.5, -2]) {
      await numberInput(page, 'Shared input x', point);
      near(await readNumber(sharedResult, 'data-gradient'), 0, 'Cancellation derivative invariant');
      near(await readNumber(sharedResult, 'data-loss'), 0, 'Cancellation loss invariant');
    }
    check(await sharedResult.getAttribute('data-stage') === '1', 'Inspection stage preserved during edits');
    check(await page.locator('[data-backprop-figure="shared-graph"] .bp-active').getAttribute('marker-start') === 'url(#backprop-shared-arrow)', 'Active reverse contributions point backward');
    await screenshot(page.locator('[data-backprop-lab="shared"]'), '1366-shared-cancellation');
    await numberInput(page, 'Shared input x', 0);
    await numberInput(page, 'Branch coefficient c', 2);
    near(await readNumber(sharedResult, 'data-gradient'), 0, 'Zero-input null');
    await numberInput(page, 'Shared input x', -2);
    await numberInput(page, 'Branch coefficient c', -2);
    near(await readNumber(sharedResult, 'data-gradient'), 4, 'Changed-sign case');
    await page.getByRole('button', { name: 'Reset numerical checks', exact: true }).click();
    const checkA = page.locator('[data-difference-case="A"]');
    const checkB = page.locator('[data-difference-case="B"]');
    check(await readNumber(checkB, 'data-error') < await readNumber(checkA, 'data-error'), 'Initial sine improvement');
    await numberInput(page, 'Check A exponent (h = 10ᵉ)', -5);
    await numberInput(page, 'Check B exponent (h = 10ᵉ)', -9);
    check(await readNumber(checkB, 'data-error') > await readNumber(checkA, 'data-error'), 'Smaller sine step worsens');
    await page.getByLabel('Function to differentiate', { exact: true }).selectOption('linear');
    await numberInput(page, 'Constant offset C', 1e12);
    near(await readNumber(checkA, 'data-estimate'), 0, 'Large-offset cancellation');
    near(await readNumber(checkA, 'data-analytic'), 1, 'Offset leaves analytic derivative unchanged');
    await screenshot(page.locator('[data-backprop-lab="difference"]'), '1366-offset-cancellation');
    const offsetSlider = page.getByRole('slider', { name: 'Constant offset C — slider', exact: true });
    await offsetSlider.focus(); await offsetSlider.press('Home'); near(Number(await offsetSlider.inputValue()), 0, 'Offset native minimum');
    await offsetSlider.press('End'); near(Number(await offsetSlider.inputValue()), 1e12, 'Offset native maximum');
    await page.getByLabel('Function to differentiate', { exact: true }).selectOption('square');
    await numberInput(page, 'Evaluation point x', 0);
    near(await readNumber(checkA, 'data-error'), 0, 'Quadratic zero absolute error');
    check(await checkA.getAttribute('data-relative') === 'undefined', 'Zero-reference relative error undefined');
    await screenshot(page.locator('[data-backprop-lab="difference"]'), '1366-zero-reference');
    for (const name of ['Reset line fit', 'Reset shared graph', 'Reset numerical checks']) await page.getByRole('button', { name, exact: true }).click();
    const urls = await root.locator('a[href^="/learn-assets/backpropagation/"]').evaluateAll(elements => [...new Set(elements.map(element => element.href))]);
    for (const url of urls) { const response = await page.request.get(url); check(response.status() === 200 && (await response.body()).length > 0, `Offline download ${url}`); }
    for (const width of [1366, 390, 320]) {
      await page.setViewportSize({ width, height: 950 });
      await page.evaluate(() => document.fonts.ready);
      const geometry = await root.evaluate(element => ({ overflow: document.documentElement.scrollWidth > innerWidth + 1, figures: [...element.querySelectorAll('.backprop-svg')].map(svg => ({ width: svg.getBoundingClientRect().width, height: svg.getBoundingClientRect().height })), katex: [...element.querySelectorAll('.katex svg')].map(svg => ({ height: svg.getBoundingClientRect().height })) }));
      check(!geometry.overflow, `${width}: page contained`);
      check(geometry.figures.every(figure => figure.width > 100 && figure.height > 50), `${width}: diagrams remain physically sized`);
      check(geometry.katex.every(svg => svg.height > 1), `${width}: KaTeX SVGs not collapsed`);
      const controls = await root.locator('button,input[type="number"],select').evaluateAll(elements => elements.map(element => ({ background: getComputedStyle(element).backgroundColor, color: getComputedStyle(element).color })));
      check(controls.every(control => control.background !== 'rgb(255, 255, 255)' && control.color !== 'rgb(0, 0, 0)'), `${width}: dark/amber control paint`);
      record.widths.push({ width, geometry, controls: controls.length });
      const figures = root.locator('[data-backprop-figure]');
      for (let index = 0; index < await figures.count(); index++) {
        const figure = figures.nth(index);
        await screenshot(figure, `${width}-${await figure.getAttribute('data-backprop-figure')}`);
      }
      if (width !== 1366) await screenshot(page.locator('[data-backprop-lab="difference"]'), `${width}-finite-differences`);
    }
    check(record.errors.length === 0, `No browser errors: ${record.errors.join('; ')}`);
    record.passed = true;
  } catch (error) { record.failure = error.stack; throw error; }
  finally {
    await browser.close();
    fs.writeFileSync(`docs/teaching/evidence/${evidenceName}.json`, JSON.stringify(record, null, 2) + '\n');
  }
  console.log(`Backprop browser: ${record.assertions} assertions; ${record.screenshots.length} screenshots; ${record.preview} preview.`);
})().catch(error => { console.error(error); process.exitCode = 1; });

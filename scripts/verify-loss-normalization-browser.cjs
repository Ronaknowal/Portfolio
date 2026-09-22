const fs = require('node:fs');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const base = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4197';
const production = process.env.LEARNING_PRODUCTION === '1';
const hash = file => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const read = file => JSON.parse(fs.readFileSync(file, 'utf8').replace(/^\uFEFF/, ''));
const fonts = read('scratch/kmeans-revision-review/fonts/manifest.json');
const receipt = `docs/teaching/evidence/loss-normalization-browser-${production ? 'production' : 'author'}.json`;
const report = { passed: false, checkedAt: new Date().toISOString(), base, checks: [], captures: [], errors: [], sourceHashes: {} };
if (production) report.manifestHash = hash('dist/.vite/manifest.json');
const files = ['NeuralLessonElements', 'NeuralProgram', 'LossFunctionsLabs', 'NormalizationLabs'].map(name => `src/learn/components/lesson-labs/${name}.jsx`);
files.push(...['neural-lesson-elements', 'loss-functions-labs', 'normalization-labs'].map(name => `src/learn/components/lesson-labs/${name}.css`));
files.push(...['loss-functions-models', 'normalization-models', 'loss-functions-measurements', 'normalization-measurements'].map(name => `src/learn/data/${name}.js`));
files.push(...['loss-functions-ce-mse-focal-contrastive-triplet', 'batch-layer-group-rms-normalization'].map(name => `src/learn/data/topics/${name}.jsx`));
report.sourceHashes = Object.fromEntries(files.map(file => [file, hash(file)]));
const record = name => report.checks.push(name);
async function type(lab, name, value) { await lab.getByRole('spinbutton', { name, exact: true }).fill(String(value)); }
async function drag(page, locator) {
  await locator.evaluate(node => node.scrollIntoView({ block: 'center', behavior: 'instant' }));
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  const before = await locator.inputValue(), box = await locator.boundingBox();
  await page.mouse.move(box.x + box.width * .5, box.y + box.height / 2);
  await page.mouse.down(); await page.mouse.move(box.x + box.width * .7, box.y + box.height / 2, { steps: 5 }); await page.mouse.up();
  assert.notEqual(await locator.inputValue(), before);
  await locator.press('Home'); assert.equal(Number(await locator.inputValue()), Number(await locator.getAttribute('min')));
  await locator.press('End'); assert.equal(Number(await locator.inputValue()), Number(await locator.getAttribute('max')));
}
async function layout(page, topic, width) {
  await page.setViewportSize({ width, height: 1000 });
  await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  const result = await page.evaluate(() => ({ width: innerWidth, page: document.documentElement.scrollWidth, scrollX: window.scrollX,
    errors: document.querySelectorAll('.katex-error,.lesson-load-error').length,
    unlabelled: [...document.querySelectorAll('.neural-lab input,.neural-lab select')].filter(node => !(node.getAttribute('aria-label') || [...node.labels || []].some(label => label.textContent.trim()))).map(node => node.outerHTML.slice(0,100)),
    nonfinite: [...document.querySelectorAll('.neural-result')].filter(node => /\bNaN\b|Infinity/.test(node.textContent)).map(node => node.textContent),
  }));
  if (result.page > width + 1) { result.overflow = await page.evaluate(() => [...document.querySelectorAll('body *')].filter(n => n.getBoundingClientRect().right + scrollX > innerWidth + 1 && !n.closest('.katex-mathml') && !n.closest('.neural-equation')).map(n => ({tag:n.tagName,cls:n.className,right:n.getBoundingClientRect().right,cls:String(n.className?.baseVal??n.className),parent:n.parentElement.className,width:n.getBoundingClientRect().width,text:n.textContent.slice(0,120)})).slice(0,30)); await page.screenshot({path:'scratch/deep-learning-core-implementation/normalization-overflow.png',fullPage:true}); }
  assert.ok(result.page <= width + 1, JSON.stringify(result)); assert.equal(result.errors, 0); assert.deepEqual(result.unlabelled, []); assert.deepEqual(result.nonfinite, []);
  const figures = page.locator('.neural-lesson figure, [data-lab="normalization-membership"], [data-lab="normalization-state"], [data-lab="normalization-placement"]');
  for (let i = 0; i < await figures.count(); i++) {
    const figure = figures.nth(i); if (!await figure.isVisible()) continue;
    const path = `docs/teaching/evidence/screenshots/${topic}-figure-${i}-${width}.png`;
    await figure.scrollIntoViewIfNeeded();
    const style = await page.addStyleTag({ content: '.learn-nav{visibility:hidden!important}' });
    await figure.screenshot({ path }); await style.evaluate(node => node.remove());
    report.captures.push({ path, hash: hash(path) });
  }
  record(`${topic} ${width}px: figure captures, contained page, labelled controls, finite results`);
}
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const context = await browser.newContext({ viewport: { width: 1366, height: 1000 }, reducedMotion: 'reduce' });
    await context.route('https://fonts.googleapis.com/**', route => route.fulfill({ path: fonts.stylesheet, contentType: 'text/css', headers: { 'access-control-allow-origin': '*' } }));
    await context.route('https://fonts.gstatic.com/**', route => fonts.files[route.request().url()] ? route.fulfill({ path: fonts.files[route.request().url()], contentType: 'font/ttf', headers: { 'access-control-allow-origin': '*' } }) : route.abort());
    const page = await context.newPage(); page.on('pageerror', error => report.errors.push(error.message));
    const lossId = 'loss-functions-ce-mse-focal-contrastive-triplet';
    await page.goto(`${base}/learn/path/full-curriculum/${lossId}?module=deep-learning-fundamentals`, { timeout: 120000 });
    await page.locator('[data-lab="loss-regression"]').waitFor(); await page.evaluate(() => document.fonts.ready);
    const regression = page.locator('[data-lab="loss-regression"]');
    await type(regression, 'Observation 7 value', 100); assert.match(await regression.innerText(), /Current fit 14\.28571/);
    await regression.getByRole('combobox', { name: 'Fitting objective' }).selectOption('huber'); assert.match(await regression.innerText(), /Current fit 0\.16667/);
    await regression.getByRole('button', { name: 'All measurements are 3' }).click(); assert.match(await regression.innerText(), /Current fit 3;/);
    await drag(page, regression.getByRole('slider')); await regression.getByRole('button', { name: 'Reset measurements' }).click();
    await type(regression, 'Observation 7 value', ''); assert.equal(await regression.getByRole('spinbutton').getAttribute('aria-invalid'), 'true');
    await regression.getByRole('button', { name: 'Reset measurements' }).click(); assert.equal(await regression.getByRole('spinbutton').inputValue(), '10');
    record('Regression: outlier/Huber/null, actual drag/endpoints and invalid/reset');
    const focal = page.locator('[data-lab="loss-focal"]');
    assert.match(await focal.locator('[data-result]').innerText(), /raises the bias/);
    await type(focal, 'Focusing gamma', 0); assert.match(await focal.locator('[data-result]').innerText(), /lowers the bias/);
    await type(focal, 'Number of negatives', 1.4); assert.equal(await focal.getByRole('spinbutton', { name: 'Number of negatives', exact: true }).getAttribute('aria-invalid'), 'true');
    await type(focal, 'Number of negatives', 90); assert.match(await focal.locator('[data-result]').innerText(), /no first-order bias update/);
    await focal.getByRole('button', { name: 'Reset focal comparison' }).click(); record('Focal: signed reversal and exact BCE cancellation');
    const decision = page.locator('[data-lab="loss-decisions"]');
    assert.match(await decision.locator('[data-result]').innerText(), /TP 10, FP 1, FN 2, TN 107/);
    const metrics = await decision.getByRole('region', { name: 'Fixed-probability measures: unchanged by threshold' }).innerText();
    await type(decision, 'Decision threshold', .1); assert.match(await decision.locator('[data-result]').innerText(), /TP 12, FP 2, FN 0, TN 106/);
    assert.equal(await decision.getByRole('region', { name: 'Fixed-probability measures: unchanged by threshold' }).innerText(), metrics);
    await decision.getByRole('combobox', { name: 'Inspect validation specimen' }).selectOption('4'); assert.match(await decision.innerText(), /Stored p\(nine\)/);
    record('Recorded decisions: correct threshold counts, invariant probability metrics, specimen identity');
    const triplet = page.locator('[data-lab="loss-triplet"]');
    assert.match(await triplet.locator('[data-result]').innerText(), /Selected negative 1, loss 0\.56/);
    await type(triplet, 'Negative 1 x', 2.5); assert.match(await triplet.locator('[data-result]').innerText(), /No eligible candidate/);
    await triplet.getByRole('button', { name: 'Reset geometry' }).click(); const geometry = await triplet.locator('[data-result]').innerText();
    await triplet.getByRole('button', { name: 'Translate every point up 0.5' }).click(); assert.equal(await triplet.locator('[data-result]').innerText(), geometry);
    record('Triplet: editable candidate, skip policy, translation invariance');
    const info = page.locator('[data-lab="loss-infonce"]');
    assert.match(await info.locator('[data-result]').innerText(), /0\.059113895/);
    await info.getByRole('button', { name: 'Equal similarities' }).click(); const uniform = await info.locator('[data-result]').innerText();
    await type(info, 'Temperature', 1.4); assert.equal(await info.locator('[data-result]').innerText(), uniform);
    await info.getByRole('button', { name: 'Reset competition' }).click(); record('InfoNCE: exact loss, live temperature and equal-score null');
    assert.equal(await page.locator('.neural-practice details > summary').filter({ hasText: 'Worked solution' }).count(), 7);
    for (const width of [1366, 390, 320]) await layout(page, 'loss-functions', width);
    const normId = 'batch-layer-group-rms-normalization';
    await page.goto(`${base}/learn/path/full-curriculum/${normId}?module=deep-learning-fundamentals`, { timeout: 120000 });
    await page.locator('[data-lab="normalization-membership"]').waitFor(); await page.evaluate(() => document.fonts.ready);
    const membership = page.locator('[data-lab="normalization-membership"]');
    await membership.getByRole('button', { name: 'Change example 1’s first value to 19' }).click(); assert.match(await membership.locator('[data-result]').innerText(), /0\.150220577/);
    await membership.getByRole('combobox', { name: 'Statistics rule' }).selectOption('layer'); assert.match(await membership.locator('[data-result]').innerText(), /Maximum change in example 0: 0\./);
    record('Membership: BatchNorm cross-example effect and LayerNorm null');
    const geometryLab = page.locator('[data-lab="normalization-geometry"]');
    await type(geometryLab, 'Common offset', 10); assert.match(await geometryLab.locator('[data-result]').innerText(), /0\.9135002/);
    await geometryLab.getByRole('button', { name: 'Zero-mean comparison' }).click(); assert.match(await geometryLab.locator('[data-result]').innerText(), /Zero mean/);
    record('Normalization geometry: offset sensitivity, equal-output null');
    const state = page.locator('[data-lab="normalization-state"]');
    assert.match(await state.locator('[data-result]').innerText(), /mean 0\.4, variance 1\.56666667/);
    await state.getByRole('button', { name: 'Advance one forward pass' }).click(); await state.getByRole('combobox', { name: 'Forward mode' }).selectOption('evaluation');
    const nextState = (await state.locator('[data-result]').innerText()).replace(/^\d+/, 'COUNT');
    await state.getByRole('button', { name: 'Advance one forward pass' }).click(); assert.equal((await state.locator('[data-result]').innerText()).replace(/^\d+/, 'COUNT'), nextState);
    record('BatchNorm: one genuine buffer update, evaluation forward leaves state unchanged');
    const gradients = page.locator('[data-lab="normalization-gradients"]');
    assert.match(await gradients.locator('[data-result]').innerText(), /0\.6399920001/);
    await type(gradients, 'Affine update rate', 0); assert.match(await gradients.locator('[data-result]').innerText(), /Zero learning rate/);
    await gradients.getByRole('button', { name: 'Reset gradient trace' }).click(); record('Gradient graph: checked affine step and zero-rate identity');
    const measured = page.locator('[data-lab="normalization-measured"]');
    await measured.getByRole('combobox', { name: 'Measured quantity' }).selectOption('validation_correct'); assert.match(await measured.innerText(), /117/);
    await measured.getByRole('combobox', { name: 'Recorded seed' }).selectOption('3'); assert.match(await measured.innerText(), /115/);
    const placement = page.locator('[data-lab="normalization-placement"]'); await drag(page, placement.getByRole('slider')); await placement.getByRole('button', { name: 'Reset zero branch' }).click();
    record('Measured runs, pre/post placement and real pointer endpoints');
    assert.equal(await page.locator('.neural-practice details > summary').filter({ hasText: 'Worked solution' }).count(), 8);
    for (const width of [1366, 390, 320]) await layout(page, 'normalization', width);
    assert.deepEqual(report.errors, []); report.passed = true;
  } catch (error) { report.failure = error.stack; process.exitCode = 1; }
  finally { await browser.close(); report.verifierHash = hash(__filename); fs.writeFileSync(receipt, JSON.stringify(report, null, 2) + '\n'); console.log(JSON.stringify({ passed: report.passed, checks: report.checks.length, failure: report.failure, receipt })); }
})();

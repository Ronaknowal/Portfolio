// Scoped regressions for the September 2026 math control-affordance repair.
const fs = require('node:fs');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const base = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4194';
const read = path => JSON.parse(fs.readFileSync(path, 'utf8').replace(/^\uFEFF/, ''));
const hash = path => createHash('sha256').update(fs.readFileSync(path)).digest('hex');
const fonts = read('scratch/kmeans-revision-review/fonts/manifest.json');
const sources = ['VectorTensorLabs', 'MatrixCalculusLabs', 'GradientVariantsLabs', 'LearningRateScheduleLabs', 'TopologyTdaLabs', 'NumericalPdeLabs'].map(name => `src/learn/components/lesson-labs/${name}.jsx`);
sources.push('src/learn/components/lesson-labs/matrix-calculus-labs.css', 'scripts/verify-math-control-affordances.cjs');
const records = [], captures = [], errors = [];
let active = 'startup';
const receipt = 'docs/teaching/evidence/math-control-affordances-browser.json';
const result = { status: 'running', capturedAt: new Date().toISOString(), base, sourceHashes: Object.fromEntries(sources.map(path => [path, hash(path)])), records, captures, errors };
fs.writeFileSync(receipt, JSON.stringify(result, null, 2));
async function open(page, id) {
  await page.goto(`${base}/learn/path/full-curriculum/${id}?module=math-foundations`, { waitUntil: 'domcontentloaded' });
  await page.locator('main h2, main h3').first().waitFor();
  await page.evaluate(() => document.fonts.ready);
}
async function capture(page, locator, name) {
  const path = `docs/teaching/evidence/screenshots/math-control-${name}.png`;
  await locator.scrollIntoViewIfNeeded();
  const style = await page.addStyleTag({ content: '.learn-nav { visibility: hidden !important; }' });
  try { await locator.screenshot({ path }); } finally { await style.evaluate(node => node.remove()); }
  captures.push({ path, sha256: hash(path) });
}
async function noOverflow(page) {
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false, page.url() + ' width=' + await page.evaluate(() => document.documentElement.scrollWidth));
}
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const context = await browser.newContext({ viewport: { width: 1366, height: 1000 } });
  await context.route('https://fonts.googleapis.com/**', route => route.fulfill({ path: fonts.stylesheet, contentType: 'text/css', headers: { 'access-control-allow-origin': '*' } }));
  await context.route('https://fonts.gstatic.com/**', route => {
    const path = fonts.files[route.request().url()];
    return path ? route.fulfill({ path, contentType: 'font/ttf', headers: { 'access-control-allow-origin': '*' } }) : route.continue();
  });
  const page = await context.newPage();
  page.on('pageerror', error => errors.push({ active, message: error.message }));
  try {
    active = 'Six coordinate sliders have real label associations and keyboard changes';
    for (const [id, names] of [
      ['vectors-matrices-tensor-operations', ['v horizontal coordinate', 'v vertical coordinate', 'x first coordinate', 'x second coordinate']],
      ['matrix-calculus-jacobians', ['Base x1', 'Base x2']],
    ]) {
      await open(page, id);
      for (const name of names) {
        const input = page.getByRole('slider', { name, exact: true });
        assert.equal(await input.evaluate(node => node.closest('label').control === node), true, name);
        const before = await input.inputValue();
        const container = input.locator('xpath=ancestor::section[1]');
        const outputBefore = await container.innerText();
        await input.focus();
        await page.keyboard.press(before === await input.getAttribute('max') ? 'ArrowLeft' : 'ArrowRight');
        assert.notEqual(await input.inputValue(), before, name);
        assert.notEqual(await container.innerText(), outputBefore, name);
      }
      await page.setViewportSize({ width: 320, height: 900 });
      await noOverflow(page);
      await page.setViewportSize({ width: 1366, height: 1000 });
    }
    records.push({ check: active, sliders: 6 });

    active = 'Momentum preview responds before stepping, retains cursor, and disables inapplicable controls';
    await open(page, 'gradient-descent-variants-sgd-adam-adagrad-rmsprop-lamb-lars');
    const momentum = page.getByRole('region', { name: 'Momentum and lookahead investigation' });
    const loss = momentum.getByRole('img', { name: 'Preview of quadratic loss over the recomputed trajectory' });
    const lossBefore = await loss.locator('path.optimizer-line').getAttribute('d');
    await momentum.getByLabel('Momentum beta', { exact: true }).fill('0.5');
    assert.notEqual(await loss.locator('path.optimizer-line').getAttribute('d'), lossBefore);
    await momentum.getByLabel('Inspect trajectory update').fill('2');
    assert.match(await momentum.locator('[role=status]').innerText(), /3\.23656.*update 2/);
    await momentum.getByLabel('Momentum beta', { exact: true }).fill('0.8');
    assert.equal(await momentum.getByLabel('Inspect trajectory update').inputValue(), '2');
    assert.match(await momentum.locator('[role=status]').innerText(), /9\.6883.*update 2/);
    await momentum.getByLabel('Trajectory method').selectOption('sgd');
    assert.equal(await momentum.getByLabel('Momentum beta', { exact: true }).isDisabled(), true);
    await momentum.getByLabel('Trajectory method').selectOption('momentum');
    assert.equal(await momentum.getByLabel('Momentum beta', { exact: true }).isEnabled(), true);
    const adaptive = page.getByRole('region', { name: 'Adaptive gradient history investigation' });
    await adaptive.getByLabel('Adaptive method').selectOption('adagrad');
    assert.equal(await adaptive.getByLabel('Recent-square decay beta2').isDisabled(), true);
    await adaptive.getByLabel('Adaptive method').selectOption('adam');
    assert.equal(await adaptive.getByLabel('Recent-square decay beta2').isEnabled(), true);
    await capture(page, momentum, 'momentum-desktop');
    records.push({ check: active, independentArithmetic: 'At update 2: beta .5 -> theta (1.6128,-.44), F=3.23656192; beta .8 -> theta (1.5648,-.92), F=9.68829952.' });

    active = 'Clock and plateau previews recompute while preserving inspection positions';
    await open(page, 'learning-rate-schedules-cosine-warmup-onecyclelr');
    const clock = page.getByRole('region', { name: 'Trace schedule event clocks' });
    const clockPreview = clock.getByRole('img', { name: 'Preview of parameter changes under the selected clock' });
    const initialClock = await clockPreview.locator('polyline').getAttribute('points');
    await clock.getByLabel('Schedule clock policy', { exact: true }).selectOption('microbatch');
    assert.notEqual(await clockPreview.locator('polyline').getAttribute('points'), initialClock);
    await clock.getByLabel('Inspect clock microbatch').fill('6');
    await clock.getByLabel('Schedule clock policy', { exact: true }).selectOption('committed');
    assert.equal(await clock.getByLabel('Inspect clock microbatch').inputValue(), '6');
    await clock.getByLabel('Reject attempt two', { exact: true }).uncheck();
    assert.equal(await clock.getByLabel('Inspect clock microbatch').inputValue(), '6');
    const plateau = page.getByRole('region', { name: 'Trace validation-triggered scheduling' });
    const preview = plateau.getByRole('img', { name: 'Preview of rates after every validation observation' });
    await plateau.getByLabel('Inspect validation observation').fill('5');
    for (const [name, value] of [['Patience', '3'], ['Absolute threshold δ', '0.1'], ['Cooldown observations', '3']]) {
      const input = plateau.getByLabel(name, { exact: true });
      const originalValue = await input.inputValue();
      const previous = await preview.locator('polyline').getAttribute('points');
      await input.fill(value);
      assert.notEqual(await preview.locator('polyline').getAttribute('points'), previous, name);
      assert.equal(await plateau.getByLabel('Inspect validation observation').inputValue(), '5', name);
      await input.fill(originalValue);
    }
    await page.setViewportSize({ width: 390, height: 1000 });
    await noOverflow(page);
    await capture(page, plateau, 'plateau-phone');
    await plateau.getByLabel('Validation losses (apply to use edits)').fill('.9');
    await plateau.getByRole('button', { name: 'Apply validation losses', exact: true }).click();
    assert.equal(await plateau.getByLabel('Inspect validation observation').isDisabled(), true);
    assert.equal(await preview.locator('circle').count(), 1);
    await page.setViewportSize({ width: 320, height: 1000 });
    await noOverflow(page);
    records.push({ check: active, singleton: 'One validation value renders a point and disables the zero-width inspection control.' });

    active = 'Rips slider endpoint equals the exact full complex for all offered samples and scales';
    await page.setViewportSize({ width: 1366, height: 1000 });
    await open(page, 'topology-topological-data-analysis-tda');
    const rips = page.getByRole('region', { name: 'Rips filtration and persistence investigation' });
    const threshold = rips.getByLabel('Rips edge threshold ε', { exact: true });
    for (const sample of ['square', 'rectangle', 'ring', 'grid']) for (const scale of ['0.5', '1', '2']) {
      await rips.getByLabel('Point sample', { exact: true }).selectOption(sample);
      await rips.getByLabel('Uniform coordinate scale', { exact: true }).selectOption(scale);
      await threshold.focus();
      await page.keyboard.press('End');
      assert.equal(await threshold.inputValue(), '1000');
      const endpoint = await rips.locator('.tda-metrics').innerText();
      await rips.getByRole('button', { name: 'Full 2-skeleton', exact: true }).click();
      assert.equal(await rips.locator('.tda-metrics').innerText(), endpoint, `${sample}/${scale}`);
      await threshold.focus();
      await page.keyboard.press('Home');
      assert.match(await rips.locator('.tda-metrics').innerText(), /\d+ \/ 0 \/ 0/);
    }
    await rips.getByRole('button', { name: 'Reset', exact: true }).click();
    assert.match(await rips.locator('.tda-metrics').innerText(), /4 \/ 4 \/ 0/);
    await threshold.focus();
    await page.keyboard.press('End');
    assert.match(await rips.locator('.tda-metrics').innerText(), /4 \/ 6 \/ 4/);
    assert.match(await rips.locator('.tda-result').innerText(), /already died/);
    await page.setViewportSize({ width: 320, height: 1000 });
    await noOverflow(page);
    await capture(page, rips, 'rips-phone');
    records.push({ check: active, fixtureScalePairs: 12 });

    active = 'Advection edits retain the transport step and move the real cell values';
    await page.setViewportSize({ width: 1366, height: 1000 });
    await open(page, 'numerical-pdes-grids-finite-elements-stability');
    const transport = page.getByRole('region', { name: 'Move cell averages through shared faces' });
    await transport.getByLabel('Transport step').fill('3');
    const originalCells = await transport.locator('.npde-cell-strip').innerText();
    await transport.getByLabel('Velocity', { exact: true }).selectOption('-1');
    assert.equal(await transport.getByLabel('Transport step').inputValue(), '3');
    assert.notEqual(await transport.locator('.npde-cell-strip').innerText(), originalCells);
    await transport.getByLabel('Transport scheme', { exact: true }).selectOption('centered');
    assert.equal(await transport.getByLabel('Transport step').inputValue(), '3');
    await transport.getByLabel('Courant number c').fill('1');
    assert.equal(await transport.getByLabel('Transport step').inputValue(), '3');
    await page.setViewportSize({ width: 320, height: 1000 });
    await noOverflow(page);
    await capture(page, transport, 'advection-phone');
    records.push({ check: active });
    assert.deepEqual(errors, []);
    result.status = 'passed';
  } catch (error) {
    result.status = 'failed';
    result.failure = { active, message: error.message, stack: error.stack };
    throw error;
  } finally {
    fs.writeFileSync(receipt, JSON.stringify(result, null, 2));
    await browser.close();
  }
  console.log(JSON.stringify({ status: result.status, groups: records.length, captures: captures.length }));
})();

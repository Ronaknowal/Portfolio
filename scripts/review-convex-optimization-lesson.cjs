const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const directory = 'scratch/convex-optimization-review';
fs.mkdirSync(directory, { recursive: true });
async function capture(page, locator, name) {
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = 'hidden'));
  await locator.screenshot({ path: `${directory}/${name}.png` });
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = ''));
}
async function range(lab, label, value) {
  await lab.getByRole('slider', { name: label }).fill(String(value));
}
const number = value => value === 0 || Object.is(value, -0) ? '0' : Math.abs(value) < 0.001 || Math.abs(value) >= 10000 ? value.toExponential(2) : String(Number(value.toFixed(3)));
async function readout(lab, label, expected) {
  const field = lab.locator('.convex-readout > div').filter({ has: lab.page().locator('dt', { hasText: label }) });
  assert.equal(await field.locator('dd').innerText(), expected);
}
(async () => {
  const models = await import(pathToFileURL(path.resolve('src/learn/data/convex-optimization-models.js')));
  const { convexOptimizationExamples: examples } = await import(pathToFileURL(path.resolve('src/learn/data/convex-optimization-examples.js')));
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/convex-optimization?module=math-foundations');
      const lesson = page.locator('.convex-optimization-lesson');
      await lesson.waitFor();
      assert.equal(await lesson.locator('.convex-lab').count(), 4);
      assert.equal(await lesson.locator('.python-example').count(), 9);
      for (const example of Object.values(examples)) {
        const block = lesson.locator('.python-example').filter({ has: page.locator('h3', { hasText: example.title }) });
        assert.equal(await block.count(), 1);
        assert.ok((await block.textContent()).includes(example.code), example.title);
        assert.ok((await block.textContent()).includes(example.expected), example.title);
      }
      const anchors = await lesson.locator('.lesson-intro a').evaluateAll(nodes => nodes.map(node => node.hash));
      for (const anchor of anchors) {
        assert.equal(await page.locator(`[id="${anchor.slice(1)}"]`).count(), 1, anchor);
        await lesson.locator(`.lesson-intro a[href="${anchor}"]`).click();
        assert.equal(new URL(page.url()).hash, anchor);
      }
      await capture(page, lesson.locator('.lesson-intro'), `intro-${width}`);
      const chord = lesson.getByRole('region', { name: 'Chord and supporting-line investigation' });
      for (const preset of Object.keys(models.convexCurves)) {
        await chord.getByLabel('Function', { exact: true }).selectOption(preset);
        for (const fraction of [0, 0.25, 0.5, 1]) {
          await range(chord, /Fraction θ/, fraction);
          const state = models.convexChordState(preset, -1.5, 1.5, fraction);
          await readout(chord, 'Chord minus function', number(state.gap));
          assert.equal(await chord.locator('svg .convex-curve').count(), 1);
        }
      }
      await range(chord, /Fraction θ/, 0.5);
      assert.match(await chord.locator('[role="status"]').innerText(), /negative gap/);
      await capture(page, chord, `chord-counterexample-${width}`);
      await chord.getByRole('button', { name: 'Reset chord' }).focus(); await page.keyboard.press('Enter');
      assert.equal(await chord.getByLabel('Function', { exact: true }).inputValue(), 'quadratic');
      await capture(page, chord, `chord-default-${width}`);

      const allocation = lesson.getByRole('region', { name: 'Feasible allocation and optimality certificate investigation' });
      for (const budget of [0, 0.5, 1, 4, 7, 8]) {
        await range(allocation, /Total budget/, budget);
        await allocation.getByRole('button', { name: 'Use exact optimum' }).click();
        assert.match(await allocation.locator('[role="status"]').innerText(), /certifying global optimality/);
        await readout(allocation, 'Certified gap', '0');
      }
      await range(allocation, /Total budget/, 4);
      await allocation.getByRole('button', { name: 'Try desired allocation' }).click();
      assert.match(await allocation.locator('[role="status"]').innerText(), /exceeds the shared budget/);
      await readout(allocation, 'Certified gap', 'No feasible upper bound');
      await capture(page, allocation, `allocation-infeasible-${width}`);
      await allocation.getByRole('button', { name: 'Use exact optimum' }).click();
      await allocation.locator('summary').click();
      assert.equal(await allocation.locator('tbody tr').count(), 3);
      await capture(page, allocation, `allocation-optimum-${width}`);

      const ridge = lesson.getByRole('region', { name: 'Ridge curvature and gradient trajectory investigation' });
      for (const preset of ['full', 'duplicate']) for (const penalty of [0, 0.5, 2]) for (const factor of [1, 2, 2.1]) {
        await ridge.getByLabel('Feature matrix', { exact: true }).selectOption(preset);
        await ridge.getByLabel('Ridge penalty λ', { exact: true }).selectOption(String(penalty));
        await ridge.getByLabel('Step times largest curvature ηL', { exact: true }).selectOption(String(factor));
        await ridge.getByRole('button', { name: 'Run 24 steps' }).click();
        const state = models.ridgeCurvatureState(preset, penalty, factor, 24);
        await readout(ridge, 'Current objective', number(state.current.value));
        assert.equal(await ridge.getByRole('button', { name: 'Next step', exact: true }).isDisabled(), true);
        await ridge.getByRole('button', { name: 'Previous step', exact: true }).click();
        await readout(ridge, 'Iteration', '23 / 24');
      }
      await capture(page, ridge, `ridge-unstable-${width}`);
      await ridge.getByRole('button', { name: 'Reset ridge' }).focus(); await page.keyboard.press('Space');
      await ridge.getByRole('button', { name: 'Next step', exact: true }).focus(); await page.keyboard.press('Enter');
      await readout(ridge, 'Iteration', '1 / 24');
      await ridge.getByRole('button', { name: 'Run 24 steps' }).click();
      await ridge.locator('summary').click();
      assert.equal(await ridge.locator('tbody tr').count(), 25);
      await ridge.locator('summary').click();
      await capture(page, ridge, `ridge-default-${width}`);
      await ridge.getByLabel('Feature matrix', { exact: true }).selectOption('duplicate');
      await ridge.getByLabel('Ridge penalty λ', { exact: true }).selectOption('0');
      await ridge.getByRole('button', { name: 'Next step', exact: true }).click();
      await capture(page, ridge, `ridge-flat-${width}`);

      const threshold = lesson.getByRole('region', { name: 'Soft thresholding and subgradient investigation' });
      for (const input of [-3, -1, 0, 1, 3]) for (const penalty of [0, 1, 4]) {
        await range(threshold, /Unregularized input/, input);
        await range(threshold, /Absolute-value penalty/, penalty);
        await threshold.getByRole('button', { name: 'Use proximal optimum' }).click();
        const state = models.softThresholdState(input, penalty, Math.sign(input) * Math.max(Math.abs(input) - penalty, 0));
        await readout(threshold, 'Optimal x', number(state.optimum));
        assert.match(await threshold.locator('[role="status"]').innerText(), /Zero belongs/);
      }
      await capture(page, threshold, `threshold-zero-${width}`);
      await threshold.getByRole('button', { name: 'Reset threshold' }).focus(); await page.keyboard.press('Enter');
      assert.match(await threshold.locator('[role="status"]').innerText(), /does not contain zero/);
      await capture(page, threshold, `threshold-default-${width}`);
      await threshold.getByRole('slider', { name: /Unregularized input/ }).focus();
      await page.keyboard.press('ArrowRight');
      assert.equal(await threshold.getByRole('slider', { name: /Unregularized input/ }).inputValue(), '3.25');

      for (let index = 0; index < 3; index += 1) await capture(page, lesson.locator('.convex-inline').nth(index), `inline-${index}-${width}`);
      await lesson.locator('details').evaluateAll(nodes => nodes.forEach(node => node.open = true));
      assert.equal(await lesson.locator('.lesson-exercise').count(), 6);
      const math = await lesson.locator('.katex-display').evaluateAll(nodes => nodes.map(node => ({ width: node.clientWidth, scroll: node.scrollWidth, formula: node.textContent })));
      const overflowMath = math.filter(item => item.scroll > item.width + 2);
      fs.writeFileSync(`${directory}/math-${width}.json`, JSON.stringify(math, null, 2));
      assert.deepEqual(overflowMath, [], 'A displayed formula overflows');
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 2);
      assert.equal(overflow, false, 'Page overflow');
      assert.equal(await page.locator('.katex-error').count(), 0);
      const svgs = await lesson.locator('svg.convex-graph').evaluateAll(nodes => nodes.map(node => ({ title: node.querySelector('title')?.textContent, description: node.querySelector('desc')?.textContent, invalid: /NaN|Infinity/.test(node.outerHTML) })));
      assert.ok(svgs.every(svg => svg.title && svg.description && !svg.invalid));
      await capture(page, lesson.locator('.lesson-exercise').first(), `practice-${width}`);
      await capture(page, lesson.locator('.lesson-sources'), `sources-${width}`);
      assert.deepEqual(errors, []);
      results.push({ width, anchors: anchors.length, examples: 9, investigations: 4, inlineFigures: 3, math: math.length, svg: svgs.length, errors, overflow });
      await page.close();
    }
    fs.writeFileSync(`${directory}/browser-results.json`, JSON.stringify({ checkedAt: new Date().toISOString(), results }, null, 2));
    console.log(JSON.stringify(results, null, 2));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

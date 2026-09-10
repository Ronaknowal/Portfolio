const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const directory = path.resolve(__dirname, '../scratch/multivariate-browser');
fs.mkdirSync(directory, { recursive: true });

async function capture(page, element, filename) {
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = 'hidden'));
  await element.screenshot({ path: path.join(directory, filename) });
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = ''));
}

async function setRange(locator, value) {
  await locator.fill(String(value));
}

(async () => {
  const models = await import(pathToFileURL(path.resolve(__dirname, '../src/learn/data/multivariate-calculus-models.js')));
  const { multivariateCalculusExamples: examples } = await import(pathToFileURL(path.resolve(__dirname, '../src/learn/data/multivariate-calculus-examples.js')));
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  const errors = [];
  for (const width of [1440, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
    page.on('pageerror', error => errors.push(error.message));
    await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/multivariate-calculus-gradients');
    await page.locator('[data-lab="local-gradient"]').waitFor();
    const lesson = page.locator('.multivariate-lesson');
    assert.equal(await lesson.locator('[data-lab]').count(), 5);
    assert.equal(await lesson.locator('.python-example').count(), 10);
    assert.equal(await lesson.locator('.multivariate-practice').count(), 6);
    assert.equal(await lesson.locator('.katex-error').count(), 0);
    const anchors = await page.locator('.lesson-intro nav a').evaluateAll(nodes => nodes.map(node => node.hash.slice(1)));
    assert.equal(anchors.length, 9);
    for (const anchor of anchors) assert.equal(await page.locator(`[id="${anchor}"]`).count(), 1, anchor);
    await page.locator('.lesson-intro nav a').nth(3).click();
    assert.ok(page.url().endsWith('#4-choose-a-direction-and-measure-its-rate'));

    const local = page.locator('[data-lab="local-gradient"]');
    let localStates = 0;
    for (const base of [[1, 1], [0, 0], [-2, 1], [2, -2]]) {
      await local.getByLabel('Base x', { exact: true }).fill(String(base[0]));
      await local.getByLabel('Base y', { exact: true }).fill(String(base[1]));
      await local.getByRole('button', { name: 'Apply base point', exact: true }).click();
      for (const angle of [0, 45, 90, 180, 270, 360]) {
        await setRange(local.getByLabel(/^Direction angle/), angle);
        for (const step of [-.5, 0, .2]) {
          await setRange(local.getByLabel(/^Signed step/), step);
          const expected = models.localChangeState(...base, angle, step);
          assert.equal(await local.locator('[data-result="rate"]').innerText(), models.formatCalculusNumber(expected.rate));
          assert.equal(await local.locator('[data-result="remainder"]').innerText(), models.formatCalculusNumber(expected.remainder));
          localStates++;
        }
      }
    }
    await local.getByRole('button', { name: 'Reset', exact: true }).click();
    await local.getByRole('button', { name: 'Tangent direction', exact: true }).click();
    assert.ok(Math.abs(Number(await local.locator('[data-result="rate"]').innerText())) < 1e-11);
    assert.equal(await local.locator('[data-result="remainder"]').innerText(), '0.048');
    await capture(page, local, `local-tangent-${width}.png`);
    await local.getByRole('button', { name: 'Along gradient', exact: true }).click();
    assert.equal(await local.locator('[data-result="rate"]').innerText(), models.formatCalculusNumber(Math.sqrt(20)));
    const savedRate = await local.locator('[data-result="rate"]').innerText();
    for (const bad of ['', 'abc', '3', 'Infinity']) {
      await local.getByLabel('Base x', { exact: true }).fill(bad);
      await local.getByRole('button', { name: 'Apply base point', exact: true }).click();
      assert.equal(await local.getByRole('alert').count(), 1);
      assert.equal(await local.locator('[data-result="rate"]').innerText(), savedRate);
    }
    await local.getByRole('button', { name: 'Reset', exact: true }).click();
    await local.getByLabel('Base x', { exact: true }).fill('0');
    await local.getByLabel('Base y', { exact: true }).fill('0');
    await local.getByRole('button', { name: 'Apply base point', exact: true }).click();
    assert.equal(await local.getByRole('button', { name: 'Along gradient', exact: true }).isDisabled(), true);
    assert.ok((await local.innerText()).includes('every first-order rate is zero'));
    await local.getByRole('button', { name: 'Reset', exact: true }).click();
    const directionControl = local.getByLabel(/^Direction angle/);
    await directionControl.focus(); await page.keyboard.press('ArrowRight');
    assert.ok(Number(await directionControl.inputValue()) > 0);

    const paths = page.locator('[data-lab="approach-paths"]');
    let pathStates = 0;
    for (const kind of ['axis', 'line', 'parabola']) {
      await paths.getByLabel(/^Approach path/).selectOption({ value: kind });
      if (kind === 'axis') assert.equal(await paths.getByLabel(/^Coefficient c/).isDisabled(), true);
      for (const coefficient of kind === 'axis' ? [1] : [-2, -1, -.5, 0, .5, 1, 2]) {
        if (kind !== 'axis') await paths.getByLabel(/^Coefficient c/).selectOption({ value: String(coefficient) });
        const expected = models.approachState(kind, coefficient);
        assert.equal(await paths.locator('[data-result="path-limit"]').innerText(), models.formatCalculusNumber(expected.limit));
        assert.equal(await paths.locator('tbody tr').count(), 7);
        assert.deepEqual(await paths.locator('tbody tr td:last-child').allTextContents(), expected.samples.map(row => models.formatCalculusNumber(row.value)));
        pathStates++;
      }
    }
    await paths.getByLabel(/^Coefficient c/).selectOption({ value: '1' });
    assert.equal(await paths.locator('tbody tr').last().locator('td').nth(1).innerText(), '1.000e-12');
    assert.equal(await paths.locator('tbody tr').last().locator('td').nth(2).innerText(), '0.5');
    await capture(page, paths, `parabolic-path-${width}.png`);
    await paths.getByRole('button', { name: 'Reset', exact: true }).click();
    assert.equal(await paths.locator('[data-result="path-limit"]').innerText(), '0');

    const circle = page.locator('[data-lab="circle-gradient"]');
    let circleStates = 0;
    for (const angle of [0, 30, 60, 90, 120, 180, 240, 270, 330, 360]) {
      await setRange(circle.getByLabel(/^Position angle/), angle);
      const expected = models.circleMotionState(angle);
      assert.equal(await circle.locator('[data-result="circle-value"]').innerText(), models.formatCalculusNumber(expected.value));
      assert.equal(await circle.locator('[data-result="circle-rate"]').innerText(), models.formatCalculusNumber(expected.rate));
      circleStates++;
    }
    for (const name of ['Maximum candidate', 'Minimum candidate']) {
      await circle.getByRole('button', { name, exact: true }).click();
      assert.ok(Math.abs(Number(await circle.locator('[data-result="circle-rate"]').innerText())) < 1e-11);
      const expected = (name.startsWith('Maximum') ? 1 : -1) * Math.sqrt(5);
      assert.equal(await circle.locator('[data-result="circle-value"]').innerText(), models.formatCalculusNumber(expected));
      circleStates++;
    }
    await circle.getByRole('button', { name: 'Maximum candidate', exact: true }).click();
    await capture(page, circle, `circle-maximum-${width}.png`);
    await circle.getByRole('button', { name: 'Reset', exact: true }).click();

    const curvature = page.locator('[data-lab="curvature-slices"]');
    let curvatureStates = 0;
    for (const preset of Object.keys(models.curvaturePresets)) {
      await curvature.getByLabel(/^Function at the origin/).selectOption({ value: preset });
      for (const angle of [0, 45, 90, 135, 180, 270, 360]) {
        await setRange(curvature.getByLabel(/^Slice direction/), angle);
        const expected = models.curvatureState(preset, angle);
        assert.equal(await curvature.locator('[data-result="curvature"]').innerText(), models.formatCalculusNumber(expected.curvature));
        assert.ok((await curvature.innerText()).includes(expected.verdict));
        curvatureStates++;
      }
    }
    await curvature.getByLabel(/^Function at the origin/).selectOption({ value: 'flatSaddle' });
    await setRange(curvature.getByLabel(/^Slice direction/), 90);
    await capture(page, curvature, `flat-saddle-${width}.png`);
    await curvature.getByRole('button', { name: 'Reset', exact: true }).click();

    const descent = page.locator('[data-lab="gradient-descent"]');
    let descentStates = 0;
    for (const rate of [.01, .1, .49, .5, .6]) {
      await descent.getByRole('button', { name: `η=${rate}`, exact: true }).click();
      assert.equal(await descent.getByRole('button', { name: 'Previous step', exact: true }).isDisabled(), true);
      for (let steps = 0; steps <= 12; steps++) {
        const expected = models.descentState(rate, steps);
        assert.equal(await descent.locator('tbody tr').count(), steps + 1);
        assert.equal(await descent.locator('[data-result="descent-point"]').innerText(), '(' + expected.current.point.map(models.formatCalculusNumber).join(', ') + ')');
        if (steps < 12) await descent.getByRole('button', { name: 'Next step', exact: true }).click();
        descentStates++;
      }
      assert.equal(await descent.getByRole('button', { name: 'Next step', exact: true }).isDisabled(), true);
      await descent.getByRole('button', { name: 'Previous step', exact: true }).click();
      assert.equal(await descent.locator('tbody tr').count(), 12);
      await descent.getByRole('button', { name: 'Run 12 steps', exact: true }).click();
    }
    await capture(page, descent, `diverging-descent-${width}.png`);
    const savedPoint = await descent.locator('[data-result="descent-point"]').innerText();
    for (const bad of ['', 'abc', '-1', '1']) {
      await descent.getByLabel('Learning rate η', { exact: true }).fill(bad);
      await descent.getByRole('button', { name: 'Apply rate', exact: true }).click();
      assert.equal(await descent.getByRole('alert').count(), 1);
      assert.equal(await descent.locator('[data-result="descent-point"]').innerText(), savedPoint);
    }
    await descent.getByRole('button', { name: 'Reset', exact: true }).click();
    await descent.getByLabel('Learning rate η', { exact: true }).fill('0.25');
    await descent.getByRole('button', { name: 'Apply rate', exact: true }).click();
    await descent.getByRole('button', { name: 'Next step', exact: true }).click();
    assert.equal(await descent.locator('[data-result="descent-point"]').innerText(), '(1.5, 0)');

    const firstHint = lesson.locator('.multivariate-practice').first().locator('summary').first();
    await firstHint.focus(); await page.keyboard.press('Enter');
    assert.equal(await firstHint.locator('..').getAttribute('open'), '');
    await page.keyboard.press('Space');
    assert.equal(await firstHint.locator('..').getAttribute('open'), null);
    await lesson.locator('details').evaluateAll(nodes => nodes.forEach(node => node.open = true));
    const rendered = await lesson.innerText();
    for (const example of Object.values(examples)) {
      assert.ok(rendered.includes(example.title), example.title);
      assert.ok(rendered.includes(example.code), `complete code: ${example.title}`);
      assert.ok(rendered.includes(example.expected), `complete output: ${example.title}`);
    }
    const references = await lesson.locator('.lesson-sources a').evaluateAll(nodes => nodes.map(node => ({ title: node.textContent, url: node.href })));
    assert.ok(references.length >= 10);
    const overflowingMath = await lesson.locator('.katex-display').evaluateAll(nodes => nodes.filter(node => node.scrollWidth > node.clientWidth + 2).map(node => ({ text: node.textContent.slice(0, 90), width: node.clientWidth, scroll: node.scrollWidth })));
    assert.deepEqual(overflowingMath, []);
    const controls = lesson.locator('.multivariate-lab button:not([disabled]), .multivariate-lab input:not([disabled]), .multivariate-lab select:not([disabled]), .multivariate-table, .multivariate-practice summary');
    const focusableControls = await controls.count();
    for (let index = 0; index < focusableControls; index++) {
      const control = controls.nth(index);
      await control.focus();
      assert.equal(await control.evaluate(node => node === document.activeElement), true);
    }
    const pageOverflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1);
    assert.equal(pageOverflow, false);
    assert.equal(await lesson.locator('.katex-error').count(), 0);
    results.push({ width, localStates, pathStates, circleStates, curvatureStates, descentStates,
      invalidUiGroups: 8, focusableControls, anchors: anchors.length, completePrograms: Object.keys(examples).length,
      references, overflowingMath, pageOverflow });
    await page.close();
  }
  await browser.close();
  assert.deepEqual(errors, []);
  fs.writeFileSync(path.join(directory, 'results.json'), JSON.stringify({ status: 'passed', checkedAt: new Date().toISOString(), results, errors }, null, 2));
  console.log(JSON.stringify(results.map(({ references, ...rest }) => ({ ...rest, references: references.length })), null, 2));
})().catch(error => { console.error(error); process.exitCode = 1; });

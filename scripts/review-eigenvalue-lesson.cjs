const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const repository = path.resolve(__dirname, '..');
const directory = path.join(repository, 'scratch/eigenvalue-lesson-review');
fs.mkdirSync(directory, { recursive: true });
const number = value => String(Number(value.toFixed(3)));
const pair = value => '[' + value.map(number).join(', ') + ']';

async function setAngle(locator, value) {
  await locator.focus();
  await locator.press('Home');
  for (let index = 0; index < value / 5; index++) await locator.press('ArrowRight');
  assert.equal(Number(await locator.inputValue()), value);
}
async function capture(page, locator, name) {
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = 'hidden'));
  await locator.screenshot({ path: path.join(directory, name) });
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = ''));
}
async function readings(lab) {
  return lab.locator('.eigen-readings dd').allTextContents();
}
async function checkMatrix(lab, expected) {
  const actual = await lab.locator('.eigen-matrix tbody tr').evaluateAll(rows => rows.map(row => Array.from(row.cells).map(cell => Number(cell.textContent))));
  assert.deepEqual(actual, expected.map(row => row.map(value => Number(number(value)))));
}
async function checkPlaneVector(plane, label, vector, range) {
  const line = plane.locator('g').filter({ has: plane.page().locator('title', { hasText: new RegExp(`^${label}:`) }) }).locator(':scope > line');
  assert.equal(await line.count(), 1);
  const x = Number(await line.getAttribute('x2')), y = Number(await line.getAttribute('y2'));
  assert.ok(Math.abs(x - (160 + vector[0] * 124 / range)) < 1e-9);
  assert.ok(Math.abs(y - (160 - vector[1] * 124 / range)) < 1e-9);
}

(async () => {
  const models = await import(pathToFileURL(path.join(repository, 'src/learn/data/eigenvalue-models.js')));
  const { eigenvalueExamples: examples } = await import(pathToFileURL(path.join(repository, 'src/learn/data/eigenvalue-examples.js')));
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [], errors = [];
  for (const width of [1440, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
    page.on('pageerror', error => errors.push(error.message));
    await page.routeWebSocket('**', socket => socket.close());
    await page.goto(`${process.env.LEARNING_BASE_URL || 'http://127.0.0.1:5173'}/learn/path/full-curriculum/eigenvalues-eigenvectors?module=math-foundations`);
    const lesson = page.locator('.eigenvalue-lesson');
    await lesson.waitFor();
    assert.equal(await lesson.locator('.eigen-lab').count(), 3);
    assert.equal(await lesson.locator('.python-example').count(), 10);
    for (const example of Object.values(examples)) {
      const block = lesson.locator('.python-example').filter({ has: page.locator('h3', { hasText: example.title }) });
      assert.equal(await block.count(), 1);
      assert.ok((await block.textContent()).includes(example.code), example.title);
      assert.ok((await block.textContent()).includes(example.expected), example.title + ' output');
    }
    assert.equal(await lesson.locator('.katex-error').count(), 0);
    assert.ok(await lesson.locator('.katex-display').count() >= 5);
    const anchors = await lesson.locator('.lesson-intro a[href^="#"]').evaluateAll(links => links.map(link => link.hash.slice(1)));
    assert.equal(anchors.length, 8);
    for (const anchor of anchors) assert.equal(await page.locator(`[id="${anchor}"]`).count(), 1, anchor);
    await capture(page, lesson.locator('.lesson-intro'), `intro-${width}.png`);

    const direction = lesson.getByRole('region', { name: 'Preserved direction investigation' });
    let directionStates = 0;
    for (const preset of Object.keys(models.eigenDirectionPresets)) {
      await direction.getByLabel('Transformation').selectOption(preset);
      for (const angle of [0, 30, 45, 90, 135, 180, 270, 360]) {
        await setAngle(direction.getByRole('slider'), angle);
        const state = models.eigenDirectionState(preset, angle);
        await checkMatrix(direction, state.matrix);
        assert.deepEqual(await readings(direction), [pair(state.input), pair(state.output), number(state.alongFactor), `${pair(state.perpendicular)}; length ${number(state.residualNorm)}`]);
        assert.match(await direction.locator('.eigen-feedback').innerText(), state.isZeroOutput ? /maps to zero/ : state.isEigenDirection ? /line is preserved/ : /line is not preserved/);
        await checkPlaneVector(direction.locator('.eigen-plane'), 'Output Av', state.output, 4);
        await checkPlaneVector(direction.locator('.eigen-plane'), 'Unit input v', state.input, 4);
        const projection = direction.locator('.eigen-plane circle[fill="none"]');
        assert.ok(Math.abs(Number(await projection.getAttribute('cx')) - (160 + 31 * state.along[0])) < 1e-9);
        assert.ok(Math.abs(Number(await projection.getAttribute('cy')) - (160 - 31 * state.along[1])) < 1e-9);
        directionStates++;
      }
      await setAngle(direction.getByRole('slider'), preset === 'diagonalStretch' ? 45 : 90);
      await capture(page, direction, `direction-${preset}-${width}.png`);
    }
    await direction.getByLabel('Transformation').selectOption('diagonalStretch');
    await setAngle(direction.getByRole('slider'), 30);
    await capture(page, direction, `direction-remainder-${width}.png`);
    await direction.getByRole('button', { name: 'Reset direction' }).click();
    assert.equal(await direction.getByLabel('Transformation').inputValue(), 'diagonalStretch');
    assert.equal(await direction.getByRole('slider').inputValue(), '30');

    const repetition = lesson.getByRole('region', { name: 'Repeated matrix update investigation' });
    let repeatedStates = 0;
    for (const preset of Object.keys(models.repeatedMapPresets)) {
      await repetition.getByLabel('Update rule').selectOption(preset);
      for (const start of Object.keys(models.repeatedMapStarts)) {
        await repetition.getByLabel('Starting vector').selectOption(start);
        const trace = models.repeatedMapTrace(preset, start);
        const range = Math.max(2, Math.ceil(Math.max(...trace.states.flatMap(item => item.vector.map(Math.abs))) / 2) * 2);
        const largest = Math.max(1, ...trace.states.map(state => state.norm));
        const circles = await repetition.locator('.eigen-history circle').evaluateAll(nodes => nodes.map(node => [Number(node.getAttribute('cx')), Number(node.getAttribute('cy'))]));
        assert.equal(circles.length, 13);
        trace.states.forEach((state, index) => {
          assert.ok(Math.abs(circles[index][0] - (44 + 294 * index / 12)) < 1e-9);
          assert.ok(Math.abs(circles[index][1] - (174 - 134 * state.norm / largest)) < 1e-9);
        });
        for (const state of trace.states) {
          assert.equal(await repetition.locator('.eigen-feedback p').first().innerText(), `x${state.step} = ${pair(state.vector)}`);
          assert.equal(await repetition.locator('.eigen-feedback p').last().innerText(), `Length = ${number(state.norm)}; eigenvalues: ${trace.eigenvalues}.`);
          await checkPlaneVector(repetition.locator('.eigen-plane'), 'Current state', state.vector, range);
          assert.equal(await repetition.locator('.eigen-history circle[r="5"]').count(), 1);
          if (state.step < 12) await repetition.getByRole('button', { name: 'Next update', exact: true }).click();
          repeatedStates++;
        }
        assert.equal(await repetition.getByRole('button', { name: 'Next update', exact: true }).isDisabled(), true);
        await repetition.getByRole('button', { name: 'Previous update', exact: true }).click();
      }
      await repetition.getByLabel('Starting vector').selectOption('vertical');
      for (let step = 0; step < 3; step++) await repetition.getByRole('button', { name: 'Next update', exact: true }).click();
      await capture(page, repetition, `updates-${preset}-${width}.png`);
    }
    await repetition.getByRole('button', { name: 'Reset updates' }).click();
    assert.equal(await repetition.getByRole('button', { name: 'Previous update', exact: true }).isDisabled(), true);

    const pca = lesson.getByRole('region', { name: 'Variance direction investigation' });
    let pcaStates = 0;
    for (const dataset of Object.keys(models.pcaDirectionDatasets)) {
      await pca.getByLabel('Dataset').selectOption(dataset);
      for (const angle of [0, 30, 45, 90, 135, 180]) {
        await setAngle(pca.getByRole('slider'), angle);
        const state = models.pcaDirectionState(dataset, angle);
        await checkMatrix(pca, state.covariance);
        assert.deepEqual(await readings(pca), [pair(state.direction), number(state.variance), number(state.totalVariance), number(100 * state.retainedFraction) + '%', number(state.squaredReconstructionError)]);
        const rows = await pca.locator('.eigen-data tbody tr').evaluateAll(nodes => nodes.map(row => Array.from(row.cells).map(cell => cell.textContent)));
        assert.deepEqual(rows, state.points.map((point, index) => [String.fromCharCode(65 + index), pair(point), number(state.scores[index]), pair(state.projections[index])]));
        const dots = await pca.locator('.eigen-plane circle').evaluateAll(nodes => nodes.map(node => [Number(node.getAttribute('cx')), Number(node.getAttribute('cy'))]));
        [...state.points, ...state.projections].forEach((point, index) => {
          assert.ok(Math.abs(dots[index][0] - (160 + 124 * point[0] / 3)) < 1e-9);
          assert.ok(Math.abs(dots[index][1] - (160 - 124 * point[1] / 3)) < 1e-9);
        });
        pcaStates++;
      }
      await setAngle(pca.getByRole('slider'), 45);
      await capture(page, pca, `variance-${dataset}-${width}.png`);
    }
    await pca.getByRole('button', { name: 'Reset variance' }).click();
    assert.equal(await pca.getByRole('slider').inputValue(), '0');
    for (let index = 0; index < 2; index++) await capture(page, lesson.locator('.eigen-inline').nth(index), `inline-${index}-${width}.png`);

    const solution = lesson.locator('details').filter({ has: page.locator('summary', { hasText: 'Solution and complete independent check' }) });
    const summary = solution.locator(':scope > summary');
    await summary.focus(); await page.keyboard.press('Shift+Tab'); await page.keyboard.press('Tab');
    assert.equal(await summary.evaluate(node => node === document.activeElement), true);
    assert.ok(await summary.evaluate(node => getComputedStyle(node).outlineStyle !== 'none'));
    await page.keyboard.press('Enter'); assert.equal(await solution.evaluate(node => node.open), true);
    await page.keyboard.press('Space'); assert.equal(await solution.evaluate(node => node.open), false);
    await lesson.locator('.lesson-intro a').last().click();
    assert.equal(new URL(page.url()).hash, '#8-practise-and-connect-to-local-change');
    assert.equal(await lesson.locator('a[href="./matrix-calculus-jacobians?module=math-foundations"]').count(), 1);
    const overflow = await page.evaluate(() => ({ page: document.documentElement.scrollWidth > innerWidth + 1, figures: Array.from(document.querySelectorAll('.eigen-lab, .eigen-inline')).filter(node => node.scrollWidth > node.clientWidth + 2).length }));
    assert.deepEqual(overflow, { page: false, figures: 0 });
    results.push({ width, programs: 10, anchors: 8, directionStates, repeatedStates, pcaStates, overflow });
    await page.close();
  }
  await browser.close();
  assert.deepEqual(errors, []);
  fs.writeFileSync(path.join(directory, 'results.json'), JSON.stringify({ reviewedAt: new Date().toISOString(), results, errors }, null, 2) + '\n');
  console.log(JSON.stringify({ results, errors }, null, 2));
})().catch(error => { console.error(error); process.exit(1); });

const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const repository = path.resolve(__dirname, '..');
const directory = path.join(repository, 'scratch/matrix-decomposition-lesson-review');
fs.mkdirSync(directory, { recursive: true });

async function setRange(locator, value) {
  const minimum = Number(await locator.getAttribute('min'));
  const step = Number(await locator.getAttribute('step'));
  await locator.focus();
  await locator.press('Home');
  for (let index = 0; index < Math.round((value - minimum) / step); index++) await locator.press('ArrowRight');
  assert.equal(Number(await locator.inputValue()), value);
}

async function capture(page, locator, filename) {
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = 'hidden'));
  await locator.screenshot({ path: path.join(directory, filename) });
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = ''));
}

async function assertMatrix(lab, label, expected) {
  const table = lab.locator('.decomposition-matrix table').filter({ has: lab.page().locator('caption', { hasText: label }) });
  const values = await table.locator('tr').evaluateAll(rows => rows.map(row => Array.from(row.cells).map(cell => Number(cell.textContent))));
  assert.equal(values.length, expected.length);
  for (let row = 0; row < expected.length; row++) {
    for (let column = 0; column < expected[row].length; column++) {
      assert.ok(Math.abs(values[row][column] - expected[row][column]) <= 0.000501, `${label}[${row},${column}]`);
    }
  }
}

(async () => {
  const models = await import(pathToFileURL(path.join(repository, 'src/learn/data/matrix-decomposition-models.js')));
  const { matrixDecompositionExamples: examples } = await import(pathToFileURL(path.join(repository, 'src/learn/data/matrix-decomposition-examples.js')));
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [], errors = [];
  for (const width of [1440, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
    page.on('pageerror', error => errors.push(error.message));
    // Keep this review at its initially loaded source snapshot while other
    // authors change unrelated Vite metadata. Production has no HMR socket.
    await page.routeWebSocket('**', socket => socket.close());
    await page.goto(`${process.env.LEARNING_BASE_URL || 'http://127.0.0.1:5173'}/learn/path/full-curriculum/matrix-decompositions-svd-qr-cholesky-lu?module=math-foundations`);
    const lesson = page.locator('.matrix-decomposition-lesson');
    await lesson.waitFor();
    assert.equal(await lesson.locator('.decomposition-lab').count(), 4);
    assert.equal(await lesson.locator('.python-example').count(), 10);
    for (const example of Object.values(examples)) {
      const block = lesson.locator('.python-example').filter({ has: page.locator('h3', { hasText: example.title }) });
      assert.equal(await block.count(), 1);
      assert.ok((await block.textContent()).includes(example.code), example.title);
      assert.ok((await block.textContent()).includes(example.expected), `${example.title} output`);
    }
    assert.equal(await lesson.locator('.katex-error').count(), 0);
    assert.ok(await lesson.locator('.katex-display').count() >= 5);
    const anchors = await lesson.locator('.lesson-intro a[href^="#"]').evaluateAll(links => links.map(link => link.hash.slice(1)));
    assert.equal(anchors.length, 8);
    for (const anchor of anchors) assert.equal(await page.locator(`[id="${anchor}"]`).count(), 1, anchor);
    await capture(page, lesson.locator('.lesson-intro'), `route-${width}.png`);
    await capture(page, lesson.locator('.decomposition-inline'), `triangular-dependency-${width}.png`);

    const elimination = lesson.getByRole('region', { name: 'Pivoted elimination investigation' });
    let eliminationStates = 0;
    for (const presetName of Object.keys(models.eliminationPresets)) {
      await elimination.locator('select').selectOption(presetName);
      const trace = models.eliminationTrace(presetName);
      for (let index = 0; index < trace.steps.length; index++) {
        await assertMatrix(elimination, 'Current equations', trace.steps[index].augmented);
        assert.equal(await elimination.locator('.decomposition-feedback').innerText(), trace.steps[index].message);
        if (index < trace.steps.length - 1) await elimination.getByRole('button', { name: 'Next operation', exact: true }).click();
        eliminationStates++;
      }
      assert.equal(await elimination.getByRole('button', { name: 'Next operation', exact: true }).isDisabled(), true);
      await capture(page, elimination, `elimination-${presetName}-${width}.png`);
      await elimination.getByRole('button', { name: 'Previous operation', exact: true }).click();
      await elimination.getByRole('button', { name: 'Reset trace', exact: true }).click();
      assert.equal(await elimination.getByRole('button', { name: 'Previous operation', exact: true }).isDisabled(), true);
    }
    const qr = lesson.getByRole('region', { name: 'QR column projection investigation' });
    let qrStates = 0;
    for (const column of [[1,0],[1,1],[0,0],[-2,2],[2,-2],[-1,-1],[2,1]]) {
      await setRange(qr.getByLabel('Second column first coordinate', { exact: true }), column[0]);
      await setRange(qr.getByLabel('Second column second coordinate', { exact: true }), column[1]);
      const model = models.qrGeometry(column);
      assert.equal(await qr.locator('.decomposition-matrix').count(), model.dependent ? 0 : 2);
      if (!model.dependent) {
        await assertMatrix(qr, /^Q$/, model.orthogonal);
        await assertMatrix(qr, 'R: Q coordinates of A', model.triangular);
      }
      assert.match(await qr.locator('.decomposition-feedback').innerText(), model.dependent ? /No new direction/ : /Normalize the remainder/);
      qrStates++;
    }
    await capture(page, qr, `qr-${width}.png`);
    await qr.getByRole('button', { name: 'Reset columns' }).click();
    assert.equal(await qr.getByLabel('Second column first coordinate', { exact: true }).inputValue(), '1');
    assert.equal(await qr.getByLabel('Second column second coordinate', { exact: true }).inputValue(), '0');
    const covariance = lesson.getByRole('region', { name: 'Covariance factor investigation' });
    for (const correlation of [-1, -0.5, 0, 0.5, 1]) {
      await setRange(covariance.getByLabel('Covariance correlation'), correlation);
      const model = models.covarianceGeometry(correlation);
      await assertMatrix(covariance, /^L$/, model.lower);
      await assertMatrix(covariance, 'C = LLᵀ', model.covariance);
      assert.match(await covariance.locator('.decomposition-feedback').innerText(), model.positiveDefinite ? /positive definite/ : /positive semidefinite/);
    }
    await capture(page, covariance, `covariance-singular-${width}.png`);
    await covariance.getByRole('button', { name: 'Reset correlation' }).click();
    assert.equal(await covariance.getByLabel('Covariance correlation').inputValue(), '0.5');
    await capture(page, covariance, `covariance-${width}.png`);
    const svd = lesson.getByRole('region', { name: 'Singular directions investigation' });
    let svdStates = 0;
    for (const smallerScale of [0, 1, 3]) {
      await setRange(svd.getByLabel('Smaller singular value', { exact: true }), smallerScale);
      for (const retained of [0,1,2]) {
        await svd.getByLabel('Singular directions kept').selectOption(String(retained));
        const model = models.svdGeometry({ smallerScale, retained });
        await assertMatrix(svd, 'Constructed A', model.matrix);
        await assertMatrix(svd, /^Aₖ$/, model.approximation);
        assert.match(await svd.locator('.decomposition-readout').innerText(), new RegExp(`Original rank: ${model.rank}`));
        svdStates++;
      }
    }
    await setRange(svd.getByLabel('Input basis angle', { exact: true }), -90);
    await setRange(svd.getByLabel('Output basis angle', { exact: true }), 90);
    await setRange(svd.getByLabel('Input vector angle', { exact: true }), 270);
    await assertMatrix(svd, 'Constructed A', models.svdGeometry({ inputAngle: -90, outputAngle: 90, vectorAngle: 270, smallerScale: 3 }).matrix);
    await svd.getByRole('button', { name: 'Reset SVD' }).click();
    await svd.getByLabel('Singular directions kept').selectOption('1');
    await capture(page, svd.locator('.decomposition-controls'), `svd-controls-${width}.png`);
    await capture(page, svd.locator('.decomposition-factor-chain'), `svd-chain-${width}.png`);
    await capture(page, svd.locator('.decomposition-planes'), `svd-geometry-${width}.png`);
    const solution = lesson.locator('details').filter({ has: page.locator('summary', { hasText: 'Solution and complete reproducible check' }) });
    await solution.locator(':scope > summary').focus();
    // Arrive with the keyboard so :focus-visible is tested in keyboard modality.
    await page.keyboard.press('Shift+Tab');
    await page.keyboard.press('Tab');
    assert.equal(await solution.locator(':scope > summary').evaluate(element => element === document.activeElement), true);
    assert.ok(await solution.locator(':scope > summary').evaluate(element => getComputedStyle(element).outlineStyle !== 'none'));
    await page.keyboard.press('Enter');
    assert.equal(await solution.evaluate(element => element.open), true);
    await page.keyboard.press('Space');
    assert.equal(await solution.evaluate(element => element.open), false);
    await lesson.locator('.lesson-intro a[href="#7-practise-and-explain-your-choice"]').click();
    assert.equal(new URL(page.url()).hash, '#7-practise-and-explain-your-choice');
    assert.ok(await lesson.locator('a[href="./eigenvalues-eigenvectors?module=math-foundations"]').count());
    const overflows = await page.evaluate(() => ({ page: document.documentElement.scrollWidth > innerWidth + 1,
      figures: Array.from(document.querySelectorAll('.decomposition-lab, .decomposition-inline')).filter(element => element.scrollWidth > element.clientWidth + 2).length }));
    assert.deepEqual(overflows, { page: false, figures: 0 });
    results.push({ width, programs: 10, anchors: anchors.length, eliminationStates, qrStates, covarianceStates: 5, svdStates: svdStates + 1, overflows });
    await page.close();
  }
  await browser.close();
  assert.deepEqual(errors, []);
  fs.writeFileSync(path.join(directory, 'results.json'), JSON.stringify({ results, errors }, null, 2) + '\n');
  console.log(JSON.stringify({ results, errors }, null, 2));
})().catch(error => { console.error(error); process.exit(1); });

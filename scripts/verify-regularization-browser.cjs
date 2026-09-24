// Production browser review of the regularization lesson: visible content, the
// four investigations, prediction/commit/retirement, figure geometry, narrow
// layouts, sequence, completion and load-failure recovery.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { gzipSync } = require('node:zlib');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const { inspectLessonVisualLayout } = require('./lib/lesson-visual-layout.cjs');

const distDir = process.env.DIST_DIR || 'dist';
const base = (process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4173').replace(/\/+$/, '');
const topicId = 'regularization-l1-l2-elastic-net-dropout';
const sourcePath = `src/learn/data/topics/${topicId}.jsx`;
const route = `${base}/learn/path/full-curriculum/${topicId}?module=classical-ml`;
const evidencePath = 'docs/teaching/evidence/regularization-browser.json';
const read = filename => JSON.parse(fs.readFileSync(filename, 'utf8'));
const hash = filename => createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
const normalize = text => text.replace(/\s+/g, ' ').trim();
const ownedFiles = [
  sourcePath,
  'src/learn/data/regularization-models.js',
  'src/learn/data/regularization-data.js',
  'src/learn/data/regularization-examples.js',
  'src/learn/components/lesson-labs/RegularizationShared.jsx',
  'src/learn/components/lesson-labs/RegularizationLabs.jsx',
  'src/learn/components/lesson-labs/RegularizationFigures.jsx',
  'src/learn/components/lesson-labs/regularization-labs.css',
  `src/learn/data/curriculum/blueprints/${topicId}.js`,
  'public/learn-assets/regularization/airfoil-self-noise.dat',
];

(async () => {
  const startedAt = new Date().toISOString();
  const sourceHashes = Object.fromEntries(ownedFiles.map(filename => [filename, hash(filename)]));
  const build = read(`${distDir}/.vite/manifest.json`);
  const buildHash = hash(`${distDir}/.vite/manifest.json`);
  const publications = read('src/learn/data/lesson-manifest.json');
  const { tracks } = await import('../src/learn/data/tracks.js');
  const { regularizationExamples } = await import('../src/learn/data/regularization-examples.js');
  const data = await import('../src/learn/data/regularization-data.js');
  const module = tracks.find(track => track.id === 'classical-ml');
  assert.equal(module.topicIds[21], topicId);
  assert.equal(module.topicIds.length, 39);
  const bodyFile = build[sourcePath].file;
  const bodyFiles = new Set(Object.values(publications).map(value => build[`src/learn/data/${value.replace(/^\.\//, '')}`].file));
  const allowedScripts = new Set();
  function addClosure(key) {
    if (allowedScripts.has(build[key].file)) return;
    allowedScripts.add(build[key].file);
    for (const child of build[key].imports || []) addClosure(child);
  }
  addClosure(Object.keys(build).find(key => build[key].isEntry && key.endsWith('index.html')));
  addClosure('src/learn/Reader.jsx');
  addClosure(sourcePath);
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [];
  const screenshotPaths = [];
  const checkText = async (locator, pattern) => assert.match(normalize(await locator.innerText()), pattern);
  const settle = page => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  const ready = async page => {
    await page.locator('.rg-lesson').waitFor();
    await page.waitForFunction(() => [...document.fonts].some(font => font.family.includes('Space Grotesk') && font.status === 'loaded'), null, { timeout: 20000 });
    await page.evaluate(() => document.fonts.ready);
    await settle(page);
  };
  try {
    const context = await browser.newContext({ viewport: { width: 1366, height: 1000 } });
    const page = await context.newPage();
    const requests = [], errors = [], failedAssets = [];
    page.on('request', request => requests.push(request.url()));
    page.on('pageerror', error => errors.push(error.message));
    page.on('requestfailed', request => failedAssets.push(request.url()));
    await page.goto(route, { waitUntil: 'domcontentloaded' });
    await ready(page);

    const screenshot = async (locator, filename) => {
      const destination = path.join('docs/teaching/evidence/screenshots', filename);
      fs.mkdirSync(path.dirname(destination), { recursive: true });
      const viewport = page.viewportSize();
      const bounds = await locator.boundingBox();
      if (bounds.height > viewport.height - 160) await page.setViewportSize({ width: viewport.width, height: Math.ceil(bounds.height) + 180 });
      await locator.scrollIntoViewIfNeeded();
      await locator.evaluate(element => window.scrollTo(0, element.getBoundingClientRect().top + window.scrollY - 100));
      await locator.screenshot({ path: destination });
      await page.setViewportSize(viewport);
      screenshotPaths.push(destination);
    };

    // ---------------------------------------------------------- 1. structure
    await checkText(page.locator('.reader-header h1'), /^Regularization \(L1, L2, Elastic Net, Dropout\)$/);
    await checkText(page.locator('.reader-header__meta'), /22 of 39 topics on this route/);
    await checkText(page.locator('.reader-footer__previous'), /Cross-Validation/);
    await checkText(page.locator('.reader-footer__next'), /Feature Selection/);
    assert.equal(await page.locator('.rg-investigation').count(), 4);
    assert.equal(await page.locator('.rg-figure').count(), 8);
    assert.equal(await page.locator('.rg-practice').count(), 10);
    assert.equal(await page.locator('.python-example').count(), 3);
    const rendered = normalize(await page.locator('.rg-lesson').textContent());
    for (const [key, example] of Object.entries(regularizationExamples)) {
      assert.ok(rendered.includes(normalize(example.code)), `Missing actual code for ${key}`);
      assert.ok(rendered.includes(normalize(example.expected)), `Missing actual expected output for ${key}`);
    }
    for (const anchor of await page.locator('.lesson-intro a[href^="#"]').evaluateAll(items => items.map(item => item.getAttribute('href')))) {
      assert.equal(await page.locator(`[id="${anchor.slice(1)}"]`).count(), 1, `Broken lesson anchor ${anchor}`);
    }
    assert.equal(await page.locator('.katex-error').count(), 0);
    for (const row of data.candidates) {
      assert.ok(rendered.includes(row[2].toFixed(6)), `candidate ${row[0]} λ=${row[1]} score is shown`);
    }
    assert.ok(rendered.includes(data.baselineMeanMse.toFixed(6)), 'the mean baseline is shown');
    assert.ok(rendered.includes(data.olsMeanMse.toFixed(6)), 'the OLS score is shown');
    assert.ok(rendered.includes('The reserved rows receive no prediction or score in this lesson'), 'the protected reserve is stated');
    assert.ok(rendered.includes('L1 can create sparsity; the selection objective and sample do not guarantee'), 'the honesty caveat survives');
    assert.ok(rendered.includes('Every investigation asks for a prediction before it shows an answer'), 'the intro promises what the page keeps');
    assert.ok(rendered.includes('CC BY 4.0'), 'the licence travels with the data');
    assert.equal((rendered.match(/Before running:/g) ?? []).length, 3, 'one Before running per program');
    const asset = await page.request.get(`${base}/learn-assets/regularization/airfoil-self-noise.dat`);
    assert.equal(asset.status(), 200);
    const bytes = await asset.body();
    assert.equal(bytes.length, data.provenance.bytes, 'the served file is byte-for-byte the packet file');
    assert.equal(createHash('sha256').update(bytes).digest('hex'), data.provenance.sha256, 'and matches its recorded hash');
    assert.ok(bytes.toString('latin1').split('\r\n')[0].split('\t').length === 6, 'six tab-separated columns');
    const attribution = await page.request.get(`${base}/learn-assets/regularization/ATTRIBUTION.txt`);
    assert.equal(attribution.status(), 200);
    records.push({ case: 'Complete visible code and output for three programs, eleven route anchors, eight figures, ten practice tasks, all eighteen candidate scores, both baselines, the served unchanged dataset with its hash and attribution, current metadata and module sequence' });

    // --------------------------------------------------- 2. threshold lab (I1)
    const threshold = page.locator('.rg-investigation').nth(0);
    assert.equal(await threshold.locator('input[type="radio"]:checked').count(), 0, 'no prediction is preselected');
    assert.ok(await threshold.getByRole('button', { name: 'Check prediction' }).isDisabled(), 'checking waits for a choice');
    assert.equal(await threshold.locator('.rg-verdict').count(), 0, 'no answer before a prediction');
    await threshold.getByLabel('Exactly zero', { exact: true }).check();
    await threshold.getByRole('button', { name: 'Check prediction' }).click();
    await checkText(threshold.locator('.rg-verdict'), /Your prediction matches: Exactly zero\./);
    await checkText(threshold, /The coefficient is exactly 0\./);
    await screenshot(threshold, 'regularization-threshold-desktop.png');
    await threshold.getByRole('button', { name: 'Contrast: z = 1.4' }).click();
    assert.equal(await threshold.locator('input[type="radio"]:checked').count(), 0, 'a preset retires the choice');
    await threshold.getByLabel('Positive', { exact: true }).check();
    await threshold.getByRole('button', { name: 'Check prediction' }).click();
    await checkText(threshold.locator('.rg-verdict'), /Your prediction matches: Positive\./);
    await checkText(threshold, /0\.4/);
    await threshold.getByRole('button', { name: 'Null: z = −0.6' }).click();
    await threshold.getByLabel('Exactly zero', { exact: true }).check();
    await threshold.getByRole('button', { name: 'Check prediction' }).click();
    await checkText(threshold.locator('.rg-verdict'), /Your prediction matches: Exactly zero\./);
    await threshold.getByRole('button', { name: 'Null: λ = 0' }).click();
    await threshold.getByLabel('Positive', { exact: true }).check();
    await threshold.getByRole('button', { name: 'Check prediction' }).click();
    await checkText(threshold.locator('.rg-readout'), /With λ = 0 the answer is z for every ρ/);
    await threshold.getByRole('button', { name: 'Family comparison at z = 3' }).click();
    await threshold.getByLabel('Positive', { exact: true }).check();
    await threshold.getByRole('button', { name: 'Check prediction' }).click();
    await checkText(threshold, /1\.5/);
    await checkText(threshold, /1\.666666667/);
    await checkText(threshold, /These are minima of three different objectives/);
    await screenshot(threshold, 'regularization-threshold-family-desktop.png');
    await threshold.getByRole('spinbutton', { name: /^Data preference z/ }).fill('2');
    await checkText(threshold.locator('.rg-pending'), /Inputs changed; record a new prediction/);
    assert.equal(await threshold.locator('input[type="radio"]:checked').count(), 0, 'the edit retired the choice');
    assert.equal(await threshold.locator('.rg-verdict').count(), 0, 'and hid the stale feedback');
    await threshold.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'Threshold lab: nothing preselected, the exact zero at z = 0.4, the 1.4 contrast crossing the threshold, the −0.6 null still exactly zero, the λ = 0 null returning z, the three-family comparison, and an edit that retires both the prediction and its feedback' });

    // -------------------------------------------------- 3. coordinate lab (I2)
    const coordinate = page.locator('.rg-investigation').nth(1);
    assert.equal(await coordinate.locator('input[type="radio"]:checked').count(), 0, 'the coordinate prediction opens unset');
    await coordinate.getByLabel('Output to predict').selectOption('coefficient2');
    await coordinate.getByLabel('Exactly zero', { exact: true }).check();
    await coordinate.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(coordinate.locator('.rg-verdict'), /Your prediction matches: Exactly zero\./);
    await checkText(coordinate, /Converged: the largest optimality violation is 0, at or below the tolerance 1e-10, after 1 sweep\./);
    await checkText(coordinate, /Pure-lasso λ threshold for all coefficients to be zero/);
    await checkText(coordinate, /coefficient 1 2/);
    await screenshot(coordinate, 'regularization-coordinate-desktop.png');
    await coordinate.getByRole('button', { name: 'Contrast: row 0 target 3.4 → 7.4' }).click();
    await coordinate.getByLabel('Positive', { exact: true }).check();
    await coordinate.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(coordinate.locator('.rg-verdict'), /Your prediction matches: Positive\./);
    await checkText(coordinate, /intercept 1/);
    await coordinate.getByRole('button', { name: 'Null: add 7 to every target' }).click();
    await coordinate.getByLabel('Exactly zero', { exact: true }).check();
    await coordinate.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(coordinate, /intercept 7/);
    await checkText(coordinate, /coefficient 1 2/);
    await coordinate.getByRole('button', { name: 'Null: the same rows in a different order' }).click();
    await coordinate.getByLabel('Exactly zero', { exact: true }).check();
    await coordinate.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(coordinate, /coefficient 1 2/);
    await checkText(coordinate, /intercept 0/);
    await coordinate.getByRole('button', { name: 'A constant second feature' }).click();
    await coordinate.getByLabel('Exactly zero', { exact: true }).check();
    await coordinate.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(coordinate, /positive penalty still minimizes at zero/);
    assert.doesNotMatch(await coordinate.innerText(), /declared 0 · flat/);
    await screenshot(coordinate, 'regularization-coordinate-constant-desktop.png');
    // The duplicate design, and the order contract recorded with the prediction.
    await coordinate.getByRole('button', { name: 'Two rows of duplicate sensors' }).click();
    await coordinate.getByLabel('Output to predict').selectOption('coefficient1');
    await coordinate.getByRole('button', { name: 'Two rows of duplicate sensors' }).click();
    await coordinate.getByLabel('Positive', { exact: true }).check();
    await coordinate.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(coordinate.locator('.rg-verdict'), /The recorded coordinate order was feature 1, then feature 2\./);
    await checkText(coordinate, /coefficient 1 1/);
    await coordinate.getByLabel('Coordinate order').selectOption('reversed');
    assert.equal(await coordinate.locator('input[type="radio"]:checked').count(), 0, 'changing the order retires the prediction');
    await coordinate.getByLabel('Exactly zero', { exact: true }).check();
    await coordinate.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(coordinate.locator('.rg-verdict'), /Your prediction matches: Exactly zero\..*feature 2, then feature 1/);
    await checkText(coordinate, /coefficient 2 1/);
    await screenshot(coordinate, 'regularization-coordinate-duplicate-desktop.png');
    // A numeric row prediction, which cannot be committed without a number.
    await coordinate.getByRole('button', { name: 'Four constructed rows' }).click();
    await coordinate.getByLabel('Output to predict').selectOption('row');
    await coordinate.getByLabel('Above that row’s target', { exact: true }).check();
    assert.ok(await coordinate.getByRole('button', { name: 'Apply and check' }).isDisabled(), 'a numeric prediction is required here');
    await coordinate.getByLabel('The fitted value, to two decimals').fill('2');
    await coordinate.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(coordinate.locator('.rg-verdict'), /You wrote 2 for the fitted value; the calculation gives 2/);
    // A multi-sweep fit keeps its recorded history and says whether it converged.
    await coordinate.getByRole('button', { name: 'Two rows of duplicate sensors' }).click();
    await coordinate.getByLabel('Output to predict').selectOption('coefficient1');
    await coordinate.getByRole('button', { name: 'Two rows of duplicate sensors' }).click();
    await coordinate.getByRole('spinbutton', { name: /^Mixing fraction ρ/ }).fill('0');
    await coordinate.getByLabel('Positive', { exact: true }).check();
    await coordinate.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(coordinate, /sweeps used 18 of 10000/);
    await checkText(coordinate, /Objective by sweep, from the actual computed history/);
    await screenshot(coordinate, 'regularization-coordinate-history-desktop.png');
    await coordinate.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'Coordinate lab: the exact zero second slope in one sweep with a zero optimality violation, the 7.4 contrast giving intercept 1, both nulls (target shift and row reorder), the constant column correctly minimized at zero under positive L1 penalty, the duplicate design returning (1, 0) or (0, 1) with the coordinate order recorded alongside the prediction, a required numeric row prediction, and an 18-sweep ridge history' });

    // ----------------------------------------------------- 4. airfoil lab (I3)
    const airfoil = page.locator('.rg-investigation').nth(2);
    assert.equal(await airfoil.locator('input[type="radio"]:checked').count(), 0, 'the airfoil prediction opens unset');
    await airfoil.getByLabel('Exactly the same', { exact: true }).check();
    await airfoil.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(airfoil.locator('.rg-verdict'), /Your prediction matches: Exactly the same\./);
    await checkText(airfoil.locator('.rg-readout'), /124\.465305401 dB/);
    await checkText(airfoil, /Nothing is edited, so no term is highlighted\./);
    await screenshot(airfoil, 'regularization-airfoil-desktop.png');
    await airfoil.getByRole('button', { name: 'Contrast: frequency 1,250 → 1,750 Hz' }).click();
    await airfoil.getByLabel('A lower predicted level', { exact: true }).check();
    await airfoil.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(airfoil.locator('.rg-verdict'), /Your prediction matches: A lower predicted level\./);
    await checkText(airfoil.locator('.rg-readout'), /123\.628646280 dB/);
    await checkText(airfoil, /moves 6 of the twenty terms together/);
    assert.equal(await airfoil.locator('.rg-chip.is-touched').count(), 6, 'exactly six terms are highlighted');
    await screenshot(airfoil, 'regularization-airfoil-changed-desktop.png');
    await airfoil.getByRole('button', { name: 'Null: change only the reference target' }).click();
    await airfoil.getByLabel('Exactly the same', { exact: true }).check();
    await airfoil.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(airfoil.locator('.rg-readout'), /124\.465305401 dB/);
    await checkText(airfoil.locator('.rg-readout'), /Against the reference target 130 dB the residual is/);
    assert.equal(await airfoil.locator('.rg-chip.is-touched').count(), 0, 'the target is not a feature');
    await airfoil.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'Airfoil trace: the recorded 124.465305 dB reproduced from five measurements, the 1,750 Hz contrast giving 123.628646 dB and moving exactly six polynomial terms, and the target-only null leaving the prediction untouched while the residual changes' });

    // ----------------------------------------------------- 5. dropout lab (I4)
    const dropout = page.locator('.rg-investigation').nth(3);
    assert.equal(await dropout.locator('input[type="radio"]:checked').count(), 0, 'the dropout prediction opens unset');
    await dropout.getByLabel('Higher than the clean loss', { exact: true }).check();
    assert.ok(await dropout.getByRole('button', { name: 'Check prediction' }).isDisabled(), 'the mean output is required too');
    await dropout.getByLabel('The mean noisy output').fill('1');
    await dropout.getByRole('button', { name: 'Check prediction' }).click();
    await checkText(dropout.locator('.rg-verdict'), /Your prediction matches: Higher than the clean loss\./);
    await checkText(dropout.locator('.rg-verdict'), /You wrote 1 for the mean output; the calculation gives 1, within 0\.000001/);
    await checkText(dropout, /expected noisy half-squared loss 2\.5/);
    await checkText(dropout, /\(1 − q\)\/\(2q\) × Σ \(wⱼxⱼ\)² 2\.5/);
    await checkText(dropout, /floating-point/);
    await screenshot(dropout, 'regularization-dropout-desktop.png');
    await dropout.getByRole('button', { name: 'Contrast: q = 1' }).click();
    await dropout.getByLabel('Exactly equal', { exact: true }).check();
    await dropout.getByLabel('The mean noisy output').fill('1');
    await dropout.getByRole('button', { name: 'Check prediction' }).click();
    await checkText(dropout.locator('.rg-verdict'), /Your prediction matches: Exactly equal\./);
    assert.equal(await dropout.locator('.rg-stage svg rect[stroke-dasharray]').count(), 3, 'three zero-probability branches are drawn as impossible');
    await checkText(dropout, /At q = 1 the zero-probability branches remain listed as impossible/);
    await screenshot(dropout, 'regularization-dropout-q1-desktop.png');
    await dropout.getByRole('button', { name: 'Practice 5: x = (1, 2), w = (2, 0), y = 1, q = 0.75' }).click();
    await dropout.getByLabel('Higher than the clean loss', { exact: true }).check();
    await dropout.getByLabel('The mean noisy output').fill('2');
    await dropout.getByRole('button', { name: 'Check prediction' }).click();
    await checkText(dropout, /expected noisy half-squared loss 1\.166666667/);
    await checkText(dropout, /A zero coefficient makes its mask irrelevant/);
    // The branch probabilities are not uniform once q leaves one half.
    // This is the state the contract insists on most, so it is photographed
    // here, before any later preset changes it.
    const probabilities = await dropout.locator('.rg-table-scroll tbody tr td:nth-child(2)').allTextContents();
    assert.deepEqual(probabilities.slice(0, 4), ['0.0625', '0.1875', '0.1875', '0.5625'],
      'the four branch probabilities at q = 0.75 are not a uniform quarter each');
    await screenshot(dropout, 'regularization-dropout-practice-desktop.png');
    await dropout.getByRole('button', { name: 'Null: change the target to 3' }).click();
    await dropout.getByLabel('Higher than the clean loss', { exact: true }).check();
    await dropout.getByLabel('The mean noisy output').fill('1');
    await dropout.getByRole('button', { name: 'Check prediction' }).click();
    await checkText(dropout.locator('.rg-readout'), /The analytic excess 2\.5 depends on x, w and q and not on the target/);
    await screenshot(dropout, 'regularization-dropout-target-null-desktop.png');
    await dropout.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'Dropout lab: both commitments required, the exact 2.5 expected loss matching the closed formula, the q = 1 contrast with labelled impossible branches, the practice fixture at 7/6 with a zero coefficient, non-uniform branch probabilities at q = 0.75, and the target null leaving the difference at 2.5' });

    // ------------------------------------------------------------- 6. figures
    // Sequential review: independently chosen legal boundary and tiny-change cases.
    const fillNumber = async (lab, name, value) => lab.getByRole('spinbutton', { name }).fill(String(value));
    for (const z of [-0.07, 0.07]) {
      await fillNumber(threshold, /^Data preference z/, z);
      await fillNumber(threshold, /^Strength λ/, 0.7);
      await fillNumber(threshold, /^Mixing fraction ρ/, 0.1);
      await threshold.getByLabel('Exactly zero', { exact: true }).check();
      await threshold.getByRole('button', { name: 'Check prediction' }).click();
      await checkText(threshold.locator('.rg-verdict'), /Your prediction matches: Exactly zero/);
      await checkText(threshold.locator('.rg-readout'), /unique minimizer/);
    }
    await threshold.getByRole('button', { name: 'Contrast: z = 1.4' }).click();
    await threshold.getByLabel('Positive', { exact: true }).check();
    await threshold.getByRole('button', { name: 'Check prediction' }).click();
    await checkText(threshold.locator('.rg-readout'), /derivative from the left is 0 and from the right 0/);
    for (const z of [-4, 4]) {
      await threshold.getByRole('button', { name: 'Null: λ = 0' }).click();
      await fillNumber(threshold, /^Data preference z/, z);
      await threshold.getByLabel(z > 0 ? 'Positive' : 'Negative', { exact: true }).check();
      await threshold.getByRole('button', { name: 'Check prediction' }).click();
      const marker = await threshold.locator('.rg-plot').nth(1).locator('circle.rg-mark').evaluate(node => ({
        x: Number(node.getAttribute('cx')), y: Number(node.getAttribute('cy')),
      }));
      assert(marker.x >= 42 && marker.x <= 288 && marker.y >= 14 && marker.y <= 140,
        `legal endpoint z=${z} stays inside the map axes`);
      await checkText(threshold, /corner at w = 0 exists when λρ > 0/);
    }
    await screenshot(threshold, 'regularization-threshold-boundary-desktop.png');
    // Inclusive numeric limits plus a visibly outside case; shared by all four labs.
    for (const [guess, accepted] of [[1.0001, true], [0.9999, true], [1.000101, false], [0.999899, false]]) {
      await threshold.getByRole('button', { name: 'Null: λ = 0' }).click();
      await fillNumber(threshold, /^Data preference z/, 1);
      await checkText(threshold.locator('.rg-numeric-guess'), /Answers within 0\.0001 are accepted/);
      await threshold.getByLabel('Positive', { exact: true }).check();
      await threshold.getByLabel('Optional: the coefficient itself', { exact: false }).fill(String(guess));
      await threshold.getByRole('button', { name: 'Check prediction' }).click();
      await checkText(threshold.locator('.rg-verdict'), accepted ? /within 0\.0001/ : /outside 0\.0001/);
    }
    await threshold.getByRole('button', { name: 'Reset', exact: true }).click();

    await fillNumber(coordinate, /^row 0 target/, 7.4);
    await fillNumber(coordinate, /^Strength λ/, 0.7);
    await fillNumber(coordinate, /^Mixing fraction ρ/, 0.5);
    await coordinate.getByLabel('Coordinate order').selectOption('reversed');
    await coordinate.getByLabel('Output to predict').selectOption('row');
    await coordinate.getByLabel('Which row').selectOption('2');
    assert.equal(await coordinate.getByRole('spinbutton', { name: /^row 0 target/ }).inputValue(), '7.4');
    assert.equal(await coordinate.getByRole('spinbutton', { name: /^Strength λ/ }).inputValue(), '0.7');
    assert.equal(await coordinate.getByRole('spinbutton', { name: /^Mixing fraction ρ/ }).inputValue(), '0.5');
    assert.equal(await coordinate.getByLabel('Coordinate order').inputValue(), 'reversed');
    assert.equal(await coordinate.locator('.rg-verdict').count(), 0);
    await coordinate.getByRole('button', { name: 'Reset', exact: true }).click();
    assert.equal(await coordinate.getByLabel('Output to predict').inputValue(), 'coefficient1');
    await coordinate.getByLabel('Output to predict').selectOption('row');
    assert.equal(await coordinate.getByLabel('Which row').inputValue(), '0');
    await coordinate.getByRole('button', { name: 'A constant second feature' }).click();
    await fillNumber(coordinate, /^Strength λ/, 0);
    await coordinate.getByLabel('Output to predict').selectOption('coefficient2');
    await coordinate.getByLabel('Exactly zero', { exact: true }).check();
    await coordinate.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(coordinate, /declared 0 · flat/);
    await coordinate.getByRole('button', { name: 'Reset', exact: true }).click();
    await fillNumber(coordinate, /^Mixing fraction ρ/, 0.5);
    await coordinate.getByLabel('Positive', { exact: true }).check();
    await coordinate.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(coordinate, /Pure-lasso λ threshold for all coefficients to be zero/);
    await coordinate.getByRole('button', { name: 'Reset', exact: true }).click();

    await fillNumber(dropout, /^Input x₁/, 0.0001);
    await fillNumber(dropout, /^Input x₂/, 0);
    await fillNumber(dropout, /^Coefficient w₁/, 0.0001);
    await fillNumber(dropout, /^Coefficient w₂/, 0);
    await dropout.getByLabel('Higher than the clean loss', { exact: true }).check();
    await dropout.getByLabel('The mean noisy output', { exact: false }).fill('0.00000001');
    await dropout.getByRole('button', { name: 'Check prediction' }).click();
    await checkText(dropout.locator('.rg-verdict'), /Your prediction matches: Higher than the clean loss/);
    await checkText(dropout.locator('.rg-verdict'), /analytic excess is 5\.000 × 10⁻¹⁷/);
    assert.doesNotMatch(await dropout.innerText(), /The enumeration and the formula agree exactly/);
    for (const width of [1366, 390]) {
      await page.setViewportSize({ width, height: 1000 });
      await settle(page);
      const dropoutPaint = await dropout.locator('svg').evaluateAll(svgs => svgs.map(svg => {
        const view = svg.viewBox.baseVal;
        const labels = [...svg.querySelectorAll('text')].filter(node => node.textContent.trim()).map(node => {
          const box = node.getBBox();
          return { text: node.textContent, x: box.x, y: box.y, right: box.x + box.width, bottom: box.y + box.height, fill: getComputedStyle(node).fill };
        });
        const outside = labels.filter(label => label.x < view.x - 0.5 || label.right > view.x + view.width + 0.5
          || label.y < view.y - 0.5 || label.bottom > view.y + view.height + 0.5);
        const collisions = labels.flatMap((a, index) => labels.slice(index + 1).filter(b =>
          Math.min(a.right, b.right) - Math.max(a.x, b.x) > 0.5
          && Math.min(a.bottom, b.bottom) - Math.max(a.y, b.y) > 0.5).map(b => [a.text, b.text]));
        return { count: labels.length, outside, collisions, fills: [...new Set(labels.map(label => label.fill))] };
      }));
      assert.equal(dropoutPaint.length, 2, 'the committed tiny case exposes tree and distribution for this check');
      for (const figure of dropoutPaint) {
        assert(figure.count > 0);
        assert.deepEqual(figure.outside, [], `tiny dropout labels stay in SVG at ${width}`);
        assert.deepEqual(figure.collisions, [], `tiny dropout labels remain separate at ${width}`);
        assert.deepEqual(figure.fills, ['rgb(207, 215, 210)'], 'every label has the intended visible fill');
      }
      if (width === 390) await screenshot(dropout, 'regularization-dropout-tiny-excess-390.png');
    }
    await page.setViewportSize({ width: 1366, height: 1000 });
    await screenshot(dropout, 'regularization-dropout-tiny-excess-desktop.png');
    await dropout.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'Sequential regressions: decimal threshold ties and correct smooth slopes, legal ±4 map endpoints, inclusive numeric tolerance versus just-outside answers, selectors preserving custom data and reset restoring selectors, penalized versus genuinely flat constant coordinate, pure-lasso threshold labeling, and a positive 5e-17 dropout excess' });

    const geometry = page.locator('.rg-figure').nth(1);
    await checkText(geometry, /Budget 2 · objective 2\.58/);
    await checkText(geometry, /attained penalty measure/);
    await checkText(geometry, /the second coordinate is exactly 0/);
    await checkText(geometry, /both coordinates survive/);
    const duplicate = page.locator('.rg-figure').nth(2);
    await checkText(duplicate, /lasso \(0\.5, 0\.5\)/);
    await checkText(duplicate, /ridge \(2\/3, 2\/3\)/);
    const paths = page.locator('.rg-figure').nth(3);
    await paths.getByLabel('Family').selectOption('lasso');
    await paths.getByLabel('Inspect λ').selectOption('0.1');
    await checkText(paths.locator('.rg-readout'), /lasso at λ = 0\.1: fold MSEs .* nonzero counts 9, 9, 9 of twenty/);
    await paths.getByLabel('Fold, for the per-fold lines and the coefficients').selectOption('1');
    await checkText(paths.locator('.rg-readout'), /Fold 2 keeps 9\./);
    await paths.getByLabel('Inspect λ').selectOption('10');
    await checkText(paths.locator('.rg-readout'), /nonzero counts 0, 0, 0 of twenty/);
    assert.equal(await paths.locator('tbody tr.is-zero').count(), 20, 'every term is marked as an exact zero');
    await paths.getByLabel('Family').selectOption('ridge');
    await paths.getByLabel('Inspect λ').selectOption('0.001');
    await checkText(paths.locator('.rg-readout'), /nonzero counts 20, 20, 20 of twenty/);
    await paths.getByLabel('Follow one term').selectOption('chord_m');
    await checkText(paths, /chord_m is drawn in gold at .* when λ = 0\.001 and .* when λ = 100/);
    assert.equal(await paths.locator('tbody tr.is-selected').count(), 1, 'the followed term is marked in the full table');
    const lineWidths = await paths.locator('polyline.rg-curve[style]').evaluateAll(nodes => nodes.map(node => Number.parseFloat(getComputedStyle(node).strokeWidth)));
    assert.equal(lineWidths.filter(width => width === 2.4).length, 1, 'one followed curve actually paints thicker');
    assert.equal(lineWidths.filter(width => width === 1.1).length, 19, 'nineteen other coefficient curves paint thinner');
    assert.equal(await duplicate.locator('polyline.rg-region').evaluate(node => getComputedStyle(node).strokeWidth), '4px', 'lasso minimizer segment actually paints thick');
    const smoothness = page.locator('.rg-figure').nth(6);
    assert.deepEqual(await smoothness.locator('polyline.rg-curve').evaluateAll(nodes => nodes.map(node => getComputedStyle(node).strokeWidth)), ['1.1px', '1.1px', '1.1px']);
    assert.deepEqual(await smoothness.locator('line.rg-stem:not(.is-data):not(.is-penalty)').evaluateAll(nodes => nodes.map(node => getComputedStyle(node).stroke)), ['rgb(231, 185, 74)', 'rgb(231, 185, 74)', 'rgb(231, 185, 74)'], 'difference stems paint gold instead of invisible default stroke');
    await screenshot(paths, 'regularization-figure-4-followed-term-desktop.png');
    await paths.getByLabel('Follow one term').selectOption('all');
    const complexity = page.locator('.rg-figure').nth(7);
    await checkText(complexity, /0101010101010101 → 1\|0101 = 5 bits/);
    await checkText(complexity, /0101010001010101 → 0\|0101010001010101 = 17 bits/);
    await checkText(complexity, /313\.8155/);
    await checkText(complexity, /315\.0259/);
    for (let index = 0; index < 8; index += 1) {
      await screenshot(page.locator('.rg-figure').nth(index), `regularization-figure-${index + 1}-desktop.png`);
    }
    records.push({ case: 'Figures carry their required content: the constraint budgets are the attained penalty measures, the duplicate segment lists its midpoint, the path figure reports actual per-fold nonzero counts including the twenty exact zeros at λ = 10, and the bit code prints both modes' });

    // ------------------------------------------------- 7. drawn geometry checks
    // Every investigation plot is on screen for this pass, not just the figures.
    // There is no ungraded path any more: each investigation is committed with a
    // real prediction, which is also the only way its plots reach the screen.
    assert.equal(await page.getByRole('button', { name: /without recording a prediction/ }).count(), 0,
      'no investigation offers a way round the prediction');
    await threshold.getByLabel('Exactly zero', { exact: true }).check();
    await threshold.getByRole('button', { name: 'Check prediction' }).click();
    await coordinate.getByRole('button', { name: 'Two rows of duplicate sensors' }).click();
    await coordinate.getByRole('spinbutton', { name: /^Mixing fraction ρ/ }).fill('0');
    await coordinate.getByLabel('Positive', { exact: true }).check();
    await coordinate.getByRole('button', { name: 'Apply and check' }).click();
    await airfoil.getByLabel('Exactly the same', { exact: true }).check();
    await airfoil.getByRole('button', { name: 'Apply and check' }).click();
    await dropout.getByLabel('Higher than the clean loss', { exact: true }).check();
    await dropout.getByLabel('The mean noisy output').fill('1');
    await dropout.getByRole('button', { name: 'Check prediction' }).click();
    await settle(page);
    assert.equal(await page.locator('.rg-investigation .rg-plot').count(), 3, 'the three investigation plots are rendered for this pass');
    for (const width of [1366, 1024, 768, 390, 320]) {
      await page.setViewportSize({ width, height: 1000 });
      await settle(page);
      const issues = await page.evaluate(inspectLessonVisualLayout, '.rg-lesson');
      assert.deepEqual(issues.flatMap(figure => figure.issues), [], `Figure layout collides at ${width}px`);
    }
    await page.setViewportSize({ width: 1366, height: 1000 });
    records.push({ case: 'No label leaves its own SVG, overlaps another label or is crossed by a foreground line, at five widths' });

    // ------------------------------------------------------ 8. loading closure
    const localScripts = [...new Set(requests.map(address => new URL(address)).filter(address => address.origin === new URL(base).origin && address.pathname.endsWith('.js')).map(address => address.pathname.slice(1)))];
    assert.deepEqual(localScripts.filter(filename => bodyFiles.has(filename)), [bodyFile]);
    assert.deepEqual(localScripts.filter(filename => !allowedScripts.has(filename)), []);
    assert.ok(!requests.some(address => address.includes('/outlines/')));
    assert.ok(!requests.some(address => address.includes('airfoil-self-noise.dat') && !address.includes('learn-assets')),
      'the page never fetches the dataset to render');
    records.push({
      case: 'Fresh route requests only the selected lesson and its shared closure',
      requestedScripts: localScripts,
      rawScriptBytes: localScripts.reduce((sum, filename) => sum + fs.statSync(path.join(distDir, filename)).size, 0),
      gzipEstimateBytes: localScripts.reduce((sum, filename) => sum + gzipSync(fs.readFileSync(path.join(distDir, filename))).length, 0),
      interpretation: 'Built requested file sizes; the gzip estimate is not measured network compression or latency.',
    });

    // -------------------------------------------------------- 9. narrow widths
    for (const width of [390, 320]) {
      await page.setViewportSize({ width, height: 844 });
      await page.locator('.rg-lesson details').evaluateAll(items => items.forEach(item => { item.open = true; }));
      await settle(page);
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `Document overflows at ${width}px`);
      const mathOverflow = await page.locator('.rg-lesson .katex-display').evaluateAll(items => items.filter(item => item.scrollWidth > item.clientWidth + 1).map(item => item.textContent));
      assert.deepEqual(mathOverflow, [], `Overflowing formulas at ${width}px`);
      assert.equal(await page.locator('.katex-error').count(), 0);
      if (width === 320) {
        await screenshot(page.locator('.rg-figure').nth(0), 'regularization-figure-1-320.png');
        await screenshot(page.locator('.rg-figure').nth(1), 'regularization-figure-2-320.png');
        await screenshot(page.locator('.rg-figure').nth(3), 'regularization-figure-4-320.png');
      }
      if (width === 390) {
        await screenshot(page.locator('.rg-investigation').nth(0), 'regularization-threshold-390.png');
        await screenshot(page.locator('.rg-investigation').nth(1), 'regularization-coordinate-390.png');
        await screenshot(page.locator('.rg-investigation').nth(2), 'regularization-airfoil-390.png');
        await screenshot(page.locator('.rg-investigation').nth(3), 'regularization-dropout-390.png');
        for (const index of [2, 4, 5, 6, 7]) {
          await screenshot(page.locator('.rg-figure').nth(index), `regularization-figure-${index + 1}-390.png`);
        }
      }
      records.push({ case: `Narrow ${width}px layout, readable formula grouping and every deeper branch rendered` });
    }
    await page.setViewportSize({ width: 1366, height: 1000 });

    // -------------------------------------------------------- 10. sequence
    await page.locator('.reader-complete').click();
    assert.equal(new URL(page.url()).pathname.split('/').at(-1), topicId);
    await page.reload({ waitUntil: 'domcontentloaded' });
    await ready(page);
    assert.equal(await page.evaluate(id => JSON.parse(localStorage.getItem('kd-progress'))[id], topicId), true);
    await page.locator('.reader-footer__next').click();
    await page.waitForURL('**/feature-selection-importance-shap-permutation-mutual-info?module=classical-ml');
    records.push({ case: 'Completion persists under the stable ID without auto-advance; Next opens the actual successor' });
    assert.deepEqual(errors, []);
    assert.deepEqual(failedAssets, []);
    await context.close();

    // -------------------------------------------------------- 11. recovery
    for (const failure of ['import', 'render']) {
      const isolated = await browser.newContext();
      const trial = await isolated.newPage();
      let inject = true;
      await trial.route(`**/${bodyFile}`, async intercepted => {
        if (!inject) return intercepted.continue();
        inject = false;
        if (failure === 'import') return intercepted.abort('failed');
        return intercepted.fulfill({ status: 200, contentType: 'text/javascript', body: 'export default {content(){throw new Error("Controlled review failure")}}' });
      });
      await trial.goto(route, { waitUntil: 'domcontentloaded' });
      await trial.locator('.lesson-load-error').waitFor();
      assert.ok(await trial.locator('.reader-complete').isDisabled());
      await trial.getByRole('button', { name: /reload/i }).click();
      await ready(trial);
      assert.ok(await trial.locator('.reader-complete').isEnabled());
      records.push({ case: `${failure} failure keeps completion disabled and recovers with an explicit reload` });
      await isolated.close();
    }
    for (const [filename, expected] of Object.entries(sourceHashes)) assert.equal(hash(filename), expected, `Source changed during check: ${filename}`);
    assert.equal(hash(`${distDir}/.vite/manifest.json`), buildHash);
    const report = {
      startedAt, completedAt: new Date().toISOString(), topicId, status: 'passed',
      browser: await browser.version(), distDir, buildManifestHash: buildHash, sourceHashes,
      moduleTopicCount: module.topicIds.length,
      dataset: { file: data.provenance.file, sha256: data.provenance.sha256, bytes: data.provenance.bytes, reservedPredictionsComputed: data.provenance.reservedPredictionsComputed },
      records, screenshots: screenshotPaths,
      visualInterpretation: 'The intended Space Grotesk font loaded before the checks. Screenshots capture informative states: a matched prediction, a retired one, the duplicate order contrast, the constant column uniquely minimized at zero by its positive L1 penalty, the changed airfoil row with its six moved terms, the q = 1 dropout contrast, the legal threshold-map endpoint, a tiny positive dropout excess and every inline figure at desktop and at 320 or 390 px. They require separate visual inspection.',
    };
    fs.writeFileSync(evidencePath, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify({ status: report.status, cases: records.length, evidencePath, sourceFiles: ownedFiles.length, screenshots: screenshotPaths.length }));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });

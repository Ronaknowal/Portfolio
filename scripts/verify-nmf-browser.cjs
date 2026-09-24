// Production browser review of the NMF lesson: visible content, the three
// investigations, prediction/commit/retire behaviour, the image panels and their
// scales, narrow layouts, sequence, completion and load-failure recovery.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { gzipSync } = require('node:zlib');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const { inspectLessonVisualLayout } = require('./lib/lesson-visual-layout.cjs');

const distDir = process.env.DIST_DIR || 'dist';
const base = (process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4173').replace(/\/+$/, '');
const topicId = 'non-negative-matrix-factorization-nmf';
const sourcePath = `src/learn/data/topics/${topicId}.jsx`;
const route = `${base}/learn/path/full-curriculum/${topicId}?module=classical-ml`;
const evidencePath = 'docs/teaching/evidence/nmf-browser.json';
const read = filename => JSON.parse(fs.readFileSync(filename, 'utf8'));
const hash = filename => createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
const normalize = text => text.replace(/\s+/g, ' ').trim();
const ownedFiles = [
  sourcePath,
  'src/learn/data/nmf-models.js',
  'src/learn/data/nmf-data.js',
  'src/learn/data/nmf-examples.js',
  'src/learn/components/lesson-labs/NmfShared.jsx',
  'src/learn/components/lesson-labs/NmfLabs.jsx',
  'src/learn/components/lesson-labs/NmfFigures.jsx',
  'src/learn/components/lesson-labs/nmf-labs.css',
  `src/learn/data/curriculum/blueprints/${topicId}.js`,
  'public/learn-assets/nmf/digits-300.csv',
  'public/learn-assets/nmf/data-provenance.md',
];

(async () => {
  const startedAt = new Date().toISOString();
  const sourceHashes = Object.fromEntries(ownedFiles.map(filename => [filename, hash(filename)]));
  const build = read(`${distDir}/.vite/manifest.json`);
  const buildHash = hash(`${distDir}/.vite/manifest.json`);
  const publications = read('src/learn/data/lesson-manifest.json');
  const { tracks } = await import('../src/learn/data/tracks.js');
  const { nmfExamples } = await import('../src/learn/data/nmf-examples.js');
  const { NMF_DIGITS, NMF_RECORDED } = await import('../src/learn/data/nmf-data.js');
  const module = tracks.find(track => track.id === 'classical-ml');
  assert.equal(module.topicIds[18], topicId);
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
    await page.locator('.nm-lesson').waitFor();
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

    // ---------------------------------------------------------- 1. the page
    await checkText(page.locator('.reader-header h1'), /^Non-Negative Matrix Factorization \(NMF\)$/);
    await checkText(page.locator('.reader-header__meta'), /19 of 39 topics on this route/);
    await checkText(page.locator('.reader-footer__previous'), /Independent Component Analysis/);
    await checkText(page.locator('.reader-footer__next'), /Feature Scaling, Encoding & Imputation/);
    assert.equal(await page.locator('.nm-investigation').count(), 3);
    assert.equal(await page.locator('.nm-figure').count(), 9);
    assert.equal(await page.locator('.nm-practice').count(), 8);
    assert.equal(await page.locator('.python-example').count(), 3);
    const rendered = normalize(await page.locator('.nm-lesson').textContent());
    for (const [key, example] of Object.entries(nmfExamples)) {
      assert.ok(rendered.includes(normalize(example.code)), `Missing actual code for ${key}`);
      assert.ok(rendered.includes(normalize(example.expected)), `Missing actual expected output for ${key}`);
    }
    for (const anchor of await page.locator('.lesson-intro a[href^="#"]').evaluateAll(items => items.map(item => item.getAttribute('href')))) {
      assert.equal(await page.locator(`[id="${anchor.slice(1)}"]`).count(), 1, `Broken lesson anchor ${anchor}`);
    }
    assert.equal(await page.locator('.katex-error').count(), 0);
    // Every recorded candidate is auditable on the page.
    for (const run of NMF_DIGITS.runs) {
      assert.ok(rendered.includes(run.validationMse.toFixed(6)), `candidate k=${run.k} seed=${run.seed} validation score is shown`);
      assert.ok(rendered.includes(run.trainMse.toFixed(6)), `candidate k=${run.k} seed=${run.seed} training score is shown`);
    }
    assert.ok(rendered.includes(NMF_DIGITS.baselines.pcaTestMse.toFixed(6)) && rendered.includes(NMF_DIGITS.baselines.nmfTestMse.toFixed(6))
      && rendered.includes(NMF_DIGITS.baselines.meanTestMse.toFixed(6)), 'all three reserved comparisons are printed');
    assert.ok(rendered.includes('PCA wins this reconstruction comparison'), 'the unflattering outcome is published as it came out');
    assert.ok(rendered.includes('Nonnegativity guarantees additive reconstruction'), 'the interpretation contract is on the page');
    assert.ok(rendered.includes('Every investigation asks for a prediction before it shows an answer'), 'the intro promises what the page keeps');
    assert.ok(rendered.includes('Writer identities are unavailable in this extract'), 'the split limit travels with the real data');
    assert.ok(rendered.includes('E. Alpaydin and C. Kaynak'), 'the data attribution is on the page');
    assert.equal((rendered.match(/Before running:/g) ?? []).length, 3, 'one Before running per program, not two');
    // The manuscript's exact worked values.
    for (const value of ['2.075472', '5.5', '2.65', '17.06', '0.03944744', '30/13', '[2/3,1/3]'.replace(/\s/g, '')]) {
      assert.ok(rendered.replace(/\s/g, '').includes(value.replace(/\s/g, '')), `the worked value ${value} survives to the page`);
    }
    const csv = await page.request.get(`${base}/learn-assets/nmf/digits-300.csv`);
    assert.equal(csv.status(), 200);
    const csvText = await csv.text();
    assert.equal(csvText.trim().split('\n').length, 301, 'the CSV serves a header and 300 rows');
    assert.ok(csvText.startsWith('source_row,digit,pixel_0,'));
    assert.equal(createHash('sha256').update(csvText.replace(/\r\n/g, '\n'), 'utf8').digest('hex').length, 64);
    const provenance = await page.request.get(`${base}/learn-assets/nmf/data-provenance.md`);
    assert.equal(provenance.status(), 200);
    assert.ok((await provenance.text()).includes('CC BY 4.0'), 'the served provenance states the licence');
    records.push({ case: 'Complete visible code and output for three programs, ten route anchors, nine figures, eight practice tasks, all eight candidate scores, the downloadable CSV and provenance, current metadata and module sequence' });

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

    // ------------------------------------------------ 2. the additive figure
    const build1 = page.locator('.nm-figure').nth(0);
    await checkText(build1, /2 × 1 = 2 and 1 × 1 = 1, and 2 \+ 1 = 3/);
    await build1.getByRole('button', { name: 'feature 2' }).click();
    await checkText(build1, /2 × 0 = 0 and 1 × 1 = 1, and 0 \+ 1 = 1/);
    await build1.getByRole('button', { name: 'feature 1' }).click();
    await checkText(build1, /2 × 1 = 2 and 1 × 0 = 0, and 2 \+ 0 = 2/);
    await screenshot(build1, 'nmf-build-row-desktop.png');
    records.push({ case: 'The additive figure expands every selected feature into its two exact products' });

    // ------------------------------------------------------ 3. mixture lab
    const mixture = page.locator('.nm-investigation').nth(0);
    assert.equal(await mixture.locator('.nm-prediction input[type="radio"][name^="«"]:checked, .nm-prediction input[type="radio"]:checked').count(), 1,
      'only the target-feature radio starts selected; the prediction itself starts unset');
    assert.ok(await mixture.getByRole('button', { name: 'Check prediction' }).isDisabled(), 'checking waits for a choice');
    await mixture.getByRole('spinbutton', { name: /^Amount a of component 1/ }).fill('3');
    await mixture.getByLabel('It increases', { exact: true }).check();
    await mixture.getByRole('button', { name: 'Check prediction' }).click();
    await checkText(mixture.locator('.nm-verdict'), /Your prediction matches: It increases/);
    await checkText(mixture, /contributes 3 × 1 = 3 and component 2 contributes 1 × 1 = 1, so feature 3 is now 4 against 3 before the edit/);
    // A graded verdict must not be destroyable by a second click that computes
    // nothing: both commit actions are inert until an input changes.
    assert.ok(await mixture.getByRole('button', { name: 'Check prediction' }).isDisabled(), 'checking is closed after grading');
    assert.ok(await mixture.getByRole('button', { name: 'Calculate without recording a prediction' }).isDisabled(),
      'exploring is closed after grading, so it cannot replace the verdict with a zero-difference result');
    await checkText(mixture, /All three strips share this scale/);
    await screenshot(mixture, 'nmf-mixture-graded-desktop.png');
    await screenshot(mixture, 'nmf-mixture-desktop.png');
    // The declared alternative states.
    await mixture.getByRole('button', { name: 'Drop component 2 (b = 0)' }).click();
    await mixture.getByRole('button', { name: 'Calculate without recording a prediction' }).click();
    await checkText(mixture, /feature 1 2 0 2 feature 2 0 0 0 feature 3 2 0 2/);
    await mixture.getByRole('button', { name: 'Reset' }).click();
    await mixture.getByRole('spinbutton', { name: /^H12$/ }).fill('0.5');
    await mixture.getByLabel('It increases', { exact: true }).check();
    await mixture.getByRole('button', { name: 'Check prediction' }).click();
    // Feature 3 is untouched by H12, so the graded answer must be "unchanged".
    await checkText(mixture.locator('.nm-verdict'), /You recorded It increases; the calculation gives It stays the same/);
    await mixture.getByRole('button', { name: 'Reset' }).click();
    // The null case: with a = 0 the first pattern is multiplied away entirely.
    await mixture.getByRole('button', { name: 'Null case: a = 0, then edit H₁₁' }).click();
    await mixture.getByRole('spinbutton', { name: /^H11$/ }).fill('2');
    await mixture.getByLabel('It stays the same', { exact: true }).check();
    await mixture.getByRole('button', { name: 'Check prediction' }).click();
    await checkText(mixture.locator('.nm-verdict'), /Your prediction matches: It stays the same/);
    await screenshot(mixture, 'nmf-mixture-null-desktop.png');
    // An edit retires a recorded answer rather than re-grading it.
    await mixture.getByRole('spinbutton', { name: /^Amount b of component 2/ }).fill('2');
    await checkText(mixture.locator('.nm-pending'), /Inputs changed; record a new prediction/);
    assert.equal(await mixture.locator('.nm-verdict').count(), 0, 'the edit retired the verdict');
    // Out-of-lattice input is refused with its range, and nothing is applied.
    await mixture.getByRole('spinbutton', { name: /^Amount b of component 2/ }).fill('2.1');
    await checkText(mixture.locator('.nm-field-error'), /Use steps of 0\.25/);
    await mixture.getByRole('spinbutton', { name: /^Amount b of component 2/ }).fill('9');
    await checkText(mixture.locator('.nm-field-error'), /Supported range 0 to 4/);
    await mixture.getByRole('button', { name: 'Reset' }).click();
    // The same-third-feature synthesis the investigation asks for.
    await mixture.getByRole('button', { name: 'Same third feature, different first (a = 1, b = 2)' }).click();
    await mixture.getByRole('button', { name: 'Calculate without recording a prediction' }).click();
    await checkText(mixture, /feature 1 1 0 1 feature 2 0 2 2 feature 3 1 2 3/);
    await mixture.getByRole('button', { name: 'Reset' }).click();
    records.push({ case: 'Mixture lab: an increase graded against the committed inputs, the H12 case where the selected feature does not move, the a = 0 null, an edit retiring the verdict, refused off-lattice and out-of-range input, and the (1,2)/(2,1) pair sharing a third feature of 3' });

    // -------------------------------------------------------- 4. update lab
    const update = page.locator('.nm-investigation').nth(1);
    await checkText(update, /Half squared Frobenius loss 17\.060000000/);
    await checkText(update, /numerator \(WᵀX\) 5\.5/);
    await checkText(update, /denominator \(\(WᵀW\)H\) 2\.65/);
    await checkText(update, /2\.075471698/);
    assert.equal(await update.locator('input[name^="«"][value="grows"]:checked').count(), 0, 'the prediction opens unset');
    await update.getByLabel('It grows', { exact: true }).check();
    await update.getByRole('button', { name: 'Record it' }).click();
    await update.getByRole('button', { name: 'Update H', exact: true }).click();
    await checkText(update.locator('.nm-verdict'), /At sweep 0, the H phase moved H\[1,1\] from 1\.000000000 to 2\.075471698, so it grows/);
    await checkText(update, /H updated and waiting for the W phase/);
    await update.getByRole('button', { name: 'Update W', exact: true }).click();
    await checkText(update, /Half squared Frobenius loss 0\.039447438/);
    await checkText(update, /0\.826888/);
    // The verdict survives into sweep 1, so it has to say which sweep it describes.
    await checkText(update, /Starting from sweep 1/);
    await checkText(update.locator('.nm-verdict'), /At sweep 0, which you have since stepped past/);
    await screenshot(update, 'nmf-update-desktop.png');
    await update.getByRole('button', { name: 'Back one phase' }).click();
    await checkText(update, /H updated and waiting for the W phase/);
    await update.getByRole('button', { name: 'Back one phase' }).click();
    await checkText(update, /Half squared Frobenius loss 17\.060000000/);
    assert.equal(await update.locator('.nm-verdict').count(), 0, 'an undone phase leaves no verdict behind');
    // The shrinking contrast, so the prediction is not always "grows".
    await update.getByRole('button', { name: 'Start H₁₁ at 3 instead of 1' }).click();
    await checkText(update, /denominator \(\(WᵀW\)H\) 7\.15/);
    await update.getByLabel('It grows', { exact: true }).check();
    await update.getByRole('button', { name: 'Record it' }).click();
    await update.getByRole('button', { name: 'Update H', exact: true }).click();
    await checkText(update.locator('.nm-verdict'), /At sweep 0, the H phase moved H\[1,1\] from 3\.000000000 to 2\.307692308, so it shrinks/);
    await checkText(update, /recomputed from the current factors/);
    assert.doesNotMatch(normalize(await update.innerText()), /ratio above describes the H phase that would follow it/);
    await checkText(update, /hypothetical.*H.*W.*fixed/i);
    await screenshot(update, 'nmf-update-shrinks-desktop.png');
    // The altered-measurement contrast.
    await update.getByRole('button', { name: /^Change X\[1,2\] from 1 to 2$/ }).click();
    await update.getByRole('button', { name: 'Update H', exact: true }).click();
    await checkText(update, /0\.489795918/);
    await checkText(update, /2\.264150943/);
    await update.getByRole('button', { name: 'Update W', exact: true }).click();
    await checkText(update, /Half squared Frobenius loss 0\.224942552/);
    // The exact-fit null: nothing moves and the loss stays exactly zero.
    await update.getByRole('button', { name: 'Exact-fit null: X = W₁H₁' }).click();
    await checkText(update, /Half squared Frobenius loss 0\.000000000/);
    await update.getByLabel('It stays the same', { exact: true }).check();
    await update.getByRole('button', { name: 'Record it' }).click();
    await update.getByRole('button', { name: 'Update H', exact: true }).click();
    await checkText(update.locator('.nm-verdict'), /At sweep 0, the H phase moved H\[1,1\] from 1\.000000000 to 1\.000000000, so it stayed the same/);
    await update.getByRole('button', { name: 'Update W', exact: true }).click();
    await checkText(update, /Half squared Frobenius loss 0\.000000000/);
    await checkText(update, /Exactly zero loss at every sweep/);
    await screenshot(update, 'nmf-update-exact-null-desktop.png');
    // A zero row in X is refused rather than quietly repaired.
    await update.getByRole('button', { name: 'The taught start' }).click();
    for (const cell of ['X\\[1,1\\]', 'X\\[1,2\\]', 'X\\[1,3\\]']) {
      await update.getByRole('spinbutton', { name: new RegExp(`^${cell}$`) }).fill('0');
    }
    await update.getByRole('button', { name: 'Apply setup and restart' }).click();
    await checkText(update.locator('.nm-problem'), /Every row and column of X must keep a positive total/);
    await checkText(update, /Half squared Frobenius loss 17\.060000000/);
    await update.getByRole('button', { name: 'The taught start' }).click();
    // The bounded forty-sweep history and its log-scale trace.
    for (let sweepIndex = 0; sweepIndex < 40; sweepIndex += 1) {
      await update.getByRole('button', { name: 'Update H', exact: true }).click();
      await update.getByRole('button', { name: 'Update W', exact: true }).click();
    }
    await checkText(update, /Sweep 40/);
    assert.ok(await update.getByRole('button', { name: 'Update H', exact: true }).isDisabled(), 'the bounded history stops at forty sweeps');
    const tracePlot = update.locator('.nm-plot').filter({ hasText: 'logarithmic scale' });
    assert.equal(await tracePlot.locator('circle.nm-mark').count(), 41, 'all recorded sweeps remain plotted');
    await screenshot(tracePlot, 'nmf-update-trace-desktop.png');
    await update.getByRole('button', { name: 'Reset' }).click();
    records.push({ case: 'Update lab: the exact 5.5 / 2.65 step graded as a growth, the H₁₁ = 3 case graded as a shrink, the altered X[1,2] contrast at 0.489795918 and 0.224942552, the exact-fit null with an exactly zero loss strip, a refused zero row, a working back step and a bounded 41-point logarithmic trace' });

    // -------------------------------------------------- 5. contribution lab
    const contribution = page.locator('.nm-investigation').nth(2);
    await checkText(contribution, /source row 242/);
    await checkText(contribution, /0\.021503643/);
    assert.equal(await contribution.locator('.nm-image').count(), 3, 'the change panel appears only after an apply');
    await contribution.getByRole('button', { name: /^Component 2 · included/ }).click();
    await contribution.getByLabel('It rises', { exact: true }).check();
    await contribution.getByRole('button', { name: 'Check prediction' }).click();
    await checkText(contribution.locator('.nm-verdict'), /went from 0\.021503643 to 0\.068409087.*so it rises/);
    await checkText(contribution.locator('.nm-verdict'), /pixels improved/);
    assert.equal(await contribution.locator('.nm-image').count(), 4, 'the signed change image is drawn after an apply');
    await screenshot(contribution, 'nmf-contribution-desktop.png');
    // The exact null: component 8 has zero activation on this image.
    await contribution.getByRole('button', { name: 'Reset' }).click();
    await contribution.getByRole('button', { name: /^Component 8 · included/ }).click();
    await contribution.getByLabel('It stays the same to within 1e−8', { exact: true }).check();
    await contribution.getByRole('button', { name: 'Check prediction' }).click();
    await checkText(contribution.locator('.nm-verdict'), /so it stayed the same to within 1e−8.*Every switched component has zero activation/);
    await screenshot(contribution, 'nmf-contribution-null-desktop.png');
    // A pixel can improve while the total gets worse.
    await contribution.getByRole('button', { name: 'Reset' }).click();
    const worsePixel = await contribution.locator('.nm-image').first().locator('button').nth(35);
    await worsePixel.click();
    await contribution.getByRole('button', { name: /^Component 2 · included/ }).click();
    await contribution.getByLabel('It rises', { exact: true }).check();
    await contribution.getByLabel('This pixel gets better', { exact: true }).check();
    await contribution.getByRole('button', { name: 'Check prediction' }).click();
    await checkText(contribution.locator('.nm-verdict'), /Your selected pixel's squared error (rises|falls|did not move)/);
    await screenshot(contribution, 'nmf-contribution-pixel-desktop.png');
    // The all-removed boundary.
    await contribution.getByRole('button', { name: 'Reset' }).click();
    for (let component = 1; component <= 8; component += 1) {
      await contribution.getByRole('button', { name: new RegExp(`^Component ${component} · included`) }).click();
    }
    await contribution.getByRole('button', { name: 'Apply without recording a prediction' }).click();
    await checkText(contribution.locator('.nm-verdict'), /^·\s*Calculated without a recorded prediction\./);
    const observedRow = NMF_DIGITS.test[0].pixels.map(value => value / NMF_DIGITS.scale);
    const emptyMse = observedRow.reduce((sum, value) => sum + value * value, 0) / 64;
    await checkText(contribution, new RegExp(emptyMse.toFixed(9).replace('.', '\\.')));
    await checkText(contribution, /Reconstruction from 0 of 8 contributions/);
    await screenshot(contribution, 'nmf-contribution-empty-desktop.png');
    // Switching image clears the mask and the recorded answer.
    await contribution.getByRole('combobox', { name: 'Reserved image' }).selectOption({ index: 1 });
    assert.equal(await contribution.locator('.nm-verdict').count(), 0, 'changing the image cleared the recorded answer');
    await checkText(contribution, /Reconstruction from 8 of 8 contributions/);
    await checkText(contribution, new RegExp(`source row ${NMF_DIGITS.test[1].sourceRow}`));
    await contribution.getByRole('button', { name: 'Reset' }).click();
    records.push({ case: 'Contribution lab: the recorded removal of component 2 at 0.068409087 graded as a rise, the zero-activation component 8 null, an individual pixel graded separately from the total, the all-removed boundary at the mean squared observation, and an image change clearing mask and prediction' });

    // Sequential audit: an ungraded exploration must not mark an optional
    // pixel choice wrong, and an untouched mask is not a zero-activation removal.
    await contribution.locator('.nm-image').first().locator('button').nth(35).click();
    await contribution.getByRole('button', { name: /^Component 2 · included/ }).click();
    await contribution.getByLabel('This pixel gets better', { exact: true }).check();
    await contribution.getByRole('button', { name: 'Apply without recording a prediction' }).click();
    const exploratoryPixel = normalize(await contribution.locator('.nm-verdict').innerText());
    assert.match(exploratoryPixel, /Calculated without a recorded prediction/);
    assert.match(exploratoryPixel, /This exploration was not graded/);
    assert.doesNotMatch(exploratoryPixel, /second prediction (?:matched|missed)/i);
    await screenshot(contribution.locator('.nm-prediction'), 'nmf-contribution-ungraded-pixel.png');
    await contribution.getByRole('button', { name: 'Reset' }).click();
    await contribution.getByLabel('It stays the same to within 1e−8', { exact: true }).check();
    await contribution.getByRole('button', { name: 'Check prediction' }).click();
    const untouchedMask = normalize(await contribution.locator('.nm-verdict').innerText());
    assert.match(untouchedMask, /The mask did not change, so the reconstruction is identical/);
    assert.doesNotMatch(untouchedMask, /zero activation/);
    await screenshot(contribution.locator('.nm-prediction'), 'nmf-contribution-untouched-mask.png');
    await contribution.getByRole('button', { name: 'Reset' }).click();
    records.push({ case: 'Sequential audit: optional pixel choice remains ungraded on exploration; an untouched mask has its own null explanation while the zero-activation switch retains its exact-null explanation' });

    // ---------------------------------------------- 6. figures carry content
    const ambiguity = page.locator('.nm-figure').nth(4);
    await checkText(ambiguity, /1\.25 0\.25 1\.5/);
    await ambiguity.getByRole('button', { name: 'Second dictionary only' }).click();
    await settle(page);
    await ambiguity.getByRole('button', { name: 'Both dictionaries' }).click();
    const support = page.locator('.nm-figure').nth(8); // §7, the last figure
    await checkText(support, /A and B/);
    for (const pair of ['A and B', 'A and C', 'A and D', 'B and C', 'B and D', 'C and D']) {
      await support.getByRole('button', { name: pair, exact: true }).click();
      await checkText(support.locator('.nm-readout'), /where S is 0/);
    }
    const dictionary = page.locator('.nm-figure').nth(7);
    await checkText(dictionary, /1\.017675|1\.0177/);
    await checkText(dictionary, /These are pattern weights, not image intensities/);
    // The caption must agree with itself about which panels share which scale.
    await checkText(dictionary, /The first two panels above, observed and reconstructed, share one intensity scale/);
    assert.doesNotMatch(normalize(await dictionary.innerText()), /three panels below share one intensity scale/,
      'the superseded self-contradicting caption must be gone');
    const fitTransform = page.locator('.nm-figure').nth(5);
    const flowSvg = fitTransform.locator('svg');
    assert.equal(await flowSvg.count(), 1, 'the fit-versus-transform figure draws its flow');
    assert.equal(await flowSvg.locator('rect.nm-lane').count(), 12, 'four boxes on each of three lanes');
    assert.equal(await flowSvg.locator('rect.nm-shared').count(), 1, 'H is drawn once, as one shared node');
    const flowEdges = await flowSvg.locator('[marker-end]').count();
    assert.equal(flowEdges, 12, 'nine within-lane arrows plus one write edge and two read edges');
    const updatePaint = await flowSvg.locator('line.nm-flow.is-update').evaluate(node => ({
      stroke: getComputedStyle(node).stroke,
      width: getComputedStyle(node).strokeWidth,
    }));
    assert.equal(updatePaint.stroke, 'rgb(231, 185, 74)', 'training update shaft actually paints the legend’s gold');
    assert.equal(updatePaint.width, '2px', 'update arrow keeps its intended weight');
    await checkText(fitTransform, /One gold arrow into H: the training lane is the only one that updates the dictionary/);
    await checkText(fitTransform, /Two green arrows out of H/);
    await dictionary.getByRole('button', { name: 'Patterns normalized to sum to 1' }).click();
    await checkText(dictionary, /Divided by its own total 9\.264877/);
    await checkText(dictionary, /every number elsewhere on this page still uses the actual values/);
    await dictionary.getByRole('button', { name: 'Contribution to this image' }).click();
    await checkText(dictionary, /These eight add up to the reconstruction exactly/);
    await checkText(dictionary, /Contribution total 5\.814577 = activation 0\.740161 × mass 7\.855831/);
    await dictionary.getByRole('button', { name: 'Actual pattern values' }).click();
    assert.equal(await dictionary.locator('.nm-image').count(), 11, 'three image panels plus eight component patterns');
    // Nothing is drawn above its own stated maximum in any of the three modes.
    for (const mode of ['Actual pattern values', 'Contribution to this image', 'Patterns normalized to sum to 1']) {
      await dictionary.getByRole('button', { name: mode }).click();
      await settle(page);
      assert.equal(await dictionary.locator('.nm-pixel.is-over').count(), 0, `values above the stated scale in mode: ${mode}`);
    }
    await dictionary.getByRole('button', { name: 'Actual pattern values' }).click();
    // Every image panel is drawn square.
    const aspects = await page.locator('.nm-image').evaluateAll(items => items.map(item => {
      const box = item.getBoundingClientRect();
      return box.width / box.height;
    }));
    assert.ok(aspects.every(ratio => Math.abs(ratio - 1) < 0.02), `image panels must stay square: ${aspects.join(', ')}`);
    records.push({ case: 'Figures carry their required content: both exact products, all six crossed-zero pairs, the labelled normalization divisors, the above-one reconstruction peak, and eleven square image panels' });
    for (const [index, name] of [[0, 'build-row'], [1, 'residual'], [2, 'update-phases'], [3, 'zero-lock'], [4, 'ambiguity'],
      [5, 'fit-transform'], [6, 'candidates'], [7, 'dictionary'], [8, 'support']]) {
      await screenshot(page.locator('.nm-figure').nth(index), `nmf-figure-${name}-desktop.png`);
    }

    // -------------------------------------------- 7. geometry and overflow
    for (const width of [1366, 1024, 768, 390, 320]) {
      await page.setViewportSize({ width, height: 1000 });
      await settle(page);
      const issues = await page.evaluate(inspectLessonVisualLayout, '.nm-lesson');
      assert.deepEqual(issues.flatMap(figure => figure.issues), [], `Lesson SVG layout collides at ${width}px`);
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), `Document overflows at ${width}px`);
      const tiny = await page.locator('.nm-lesson svg text').evaluateAll(items => items
        .filter(item => item.getBoundingClientRect().height > 0)
        .map(item => ({ text: item.textContent, size: Number(getComputedStyle(item).fontSize.replace('px', '')) * (item.ownerSVGElement.getBoundingClientRect().width / item.ownerSVGElement.viewBox.baseVal.width) }))
        .filter(item => item.size < 10));
      assert.deepEqual(tiny, [], `SVG type below 10 rendered pixels at ${width}px`);
    }
    await page.setViewportSize({ width: 320, height: 900 });
    await page.locator('.nm-lesson details').evaluateAll(items => items.forEach(item => { item.open = true; }));
    await settle(page);
    const mathOverflow = await page.locator('.nm-lesson .katex-display').evaluateAll(items => items.filter(item => item.scrollWidth > item.clientWidth + 1).map(item => item.textContent));
    assert.deepEqual(mathOverflow, [], 'Overflowing formulas at 320px');
    assert.equal(await page.locator('.katex-error').count(), 0);
    for (const [index, name] of [[0, 'build-row'], [1, 'residual'], [4, 'ambiguity'], [5, 'fit-transform'], [6, 'candidates'], [7, 'dictionary']]) {
      await screenshot(page.locator('.nm-figure').nth(index), `nmf-figure-${name}-320.png`);
    }
    await screenshot(page.locator('.nm-investigation').nth(2), 'nmf-contribution-320.png');
    await page.setViewportSize({ width: 390, height: 900 });
    await settle(page);
    await screenshot(page.locator('.nm-investigation').nth(0), 'nmf-mixture-390.png');
    await screenshot(page.locator('.nm-investigation').nth(1), 'nmf-update-390.png');
    records.push({ case: 'Narrow layouts at 1366, 1024, 768, 390 and 320px: no SVG label collisions, no document overflow, no formula overflow and no SVG type below 10 rendered pixels' });

    await page.setViewportSize({ width: 1366, height: 1000 });
    await page.evaluate(() => { document.documentElement.style.fontSize = '200%'; });
    await settle(page);
    assert.deepEqual((await page.evaluate(inspectLessonVisualLayout, '.nm-lesson')).flatMap(figure => figure.issues), [],
      'the lesson stays separated with enlarged root text');
    const enlarged = await page.locator('.nm-figure').nth(0).locator('.nm-cell').first()
      .evaluate(element => Number(getComputedStyle(element).fontSize.replace('px', '')));
    const enlargedMatrix = await page.locator('.nm-matrix-cell').first()
      .evaluate(element => Number(getComputedStyle(element).fontSize.replace('px', '')));
    await screenshot(page.locator('.nm-figure').nth(0), 'nmf-build-row-enlarged-text.png');
    await page.evaluate(() => { document.documentElement.style.fontSize = ''; });
    await settle(page);
    const normalCell = await page.locator('.nm-figure').nth(0).locator('.nm-cell').first()
      .evaluate(element => Number(getComputedStyle(element).fontSize.replace('px', '')));
    const normalMatrix = await page.locator('.nm-matrix-cell').first()
      .evaluate(element => Number(getComputedStyle(element).fontSize.replace('px', '')));
    // The number printed in a cell is the content, not a decoration: it has to
    // respond to the reader's text size like the prose around it.
    assert.ok(enlarged > normalCell * 1.8, `strip cell type must grow with root text: ${normalCell} to ${enlarged}`);
    assert.ok(enlargedMatrix > normalMatrix * 1.8, `matrix cell type must grow with root text: ${normalMatrix} to ${enlargedMatrix}`);
    records.push({ case: `Enlarged root text keeps every figure label separated, and the numbers inside strip and matrix cells grow with it (${normalCell}px to ${enlarged}px, ${normalMatrix}px to ${enlargedMatrix}px)` });

    // -------------------------------------------------- 8. loading contract
    const localScripts = [...new Set(requests.map(address => new URL(address)).filter(address => address.origin === new URL(base).origin && address.pathname.endsWith('.js')).map(address => address.pathname.slice(1)))];
    assert.deepEqual(localScripts.filter(filename => bodyFiles.has(filename)), [bodyFile]);
    assert.deepEqual(localScripts.filter(filename => !allowedScripts.has(filename)), []);
    assert.ok(!requests.some(address => address.includes('/outlines/')));
    assert.ok(!requests.some(address => address.includes('digits-300.csv') && !address.includes('learn-assets')), 'the page never fetches the data to render');
    assert.ok(!requests.some(address => address.includes('manifold-learning')), 'this lesson serves its own copy of the dataset');
    records.push({
      case: 'Fresh route requests only the selected lesson and its shared closure',
      requestedScripts: localScripts,
      rawScriptBytes: localScripts.reduce((sum, filename) => sum + fs.statSync(path.join(distDir, filename)).size, 0),
      gzipEstimateBytes: localScripts.reduce((sum, filename) => sum + gzipSync(fs.readFileSync(path.join(distDir, filename))).length, 0),
      interpretation: 'Built requested file sizes; the gzip estimate is not measured network compression or latency.',
    });

    // ---------------------------------------------------- 9. sequence, state
    await page.locator('.reader-complete').click();
    assert.equal(new URL(page.url()).pathname.split('/').at(-1), topicId);
    await page.reload({ waitUntil: 'domcontentloaded' });
    await ready(page);
    assert.equal(await page.evaluate(id => JSON.parse(localStorage.getItem('kd-progress'))[id], topicId), true);
    await page.locator('.reader-footer__next').click();
    await page.waitForURL('**/feature-scaling-encoding-imputation?module=classical-ml');
    records.push({ case: 'Completion persists under the stable ID without auto-advance; Next opens the actual successor' });
    assert.deepEqual(errors, []);
    assert.deepEqual(failedAssets, []);
    await context.close();

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
      reservedComparison: NMF_DIGITS.baselines,
      firstReservedImage: { sourceRow: NMF_DIGITS.test[0].sourceRow, rowMse: NMF_RECORDED.removals.baseMse },
      records, screenshots: screenshotPaths,
      visualInterpretation: 'The intended Space Grotesk font loaded before the checks. Screenshots capture informative states: a matched and a missed prediction, the shrinking update contrast, the exact-fit null, the zero-activation null, the all-removed boundary, the image panels at their stated scales, and every inline figure at desktop and 320px. They require separate visual inspection.',
    };
    fs.writeFileSync(evidencePath, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify({ status: report.status, cases: records.length, screenshots: screenshotPaths.length, evidencePath, sourceFiles: ownedFiles.length }));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });

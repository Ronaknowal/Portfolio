// Production browser review of the feature-selection lesson: visible content,
// the five investigations, prediction/commit/retirement, figure geometry,
// narrow layouts and stacked tables, route loading, sequence, completion and
// load-failure recovery.
//
// Run against a preview of a production build:
//   npx vite build --outDir dist-fsel
//   npx vite preview --outDir dist-fsel --host 127.0.0.1 --port 4186
//   PLAYWRIGHT_PACKAGE=... DIST_DIR=dist-fsel LEARNING_BASE_URL=http://127.0.0.1:4186 \
//     node scripts/verify-selection-browser.cjs
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { gzipSync } = require('node:zlib');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const { inspectLessonVisualLayout } = require('./lib/lesson-visual-layout.cjs');

const distDir = process.env.DIST_DIR || 'dist-fsel';
const base = (process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4186').replace(/\/+$/, '');
const topicId = 'feature-selection-importance-shap-permutation-mutual-info';
const previousId = 'regularization-l1-l2-elastic-net-dropout';
const nextId = 'bias-variance-tradeoff-learning-curves';
const sourcePath = `src/learn/data/topics/${topicId}.jsx`;
const route = `${base}/learn/path/full-curriculum/${topicId}?module=classical-ml`;
const evidencePath = 'docs/teaching/evidence/selection-browser.json';
const read = filename => JSON.parse(fs.readFileSync(filename, 'utf8'));
const hash = filename => createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
const normalize = text => text.replace(/\s+/g, ' ').trim();
const ownedFiles = [
  sourcePath,
  'src/learn/data/selection-models.js',
  'src/learn/data/selection-data.js',
  'src/learn/data/selection-examples.js',
  'src/learn/components/lesson-labs/SelectionShared.jsx',
  'src/learn/components/lesson-labs/SelectionLabs.jsx',
  'src/learn/components/lesson-labs/SelectionFigures.jsx',
  'src/learn/components/lesson-labs/selection-labs.css',
  `src/learn/data/curriculum/blueprints/${topicId}.js`,
  'public/learn-assets/feature-selection/wine.data',
  'public/learn-assets/feature-selection/ATTRIBUTION.txt',
];

(async () => {
  const startedAt = new Date().toISOString();
  const sourceHashes = Object.fromEntries(ownedFiles.map(filename => [filename, hash(filename)]));
  const build = read(`${distDir}/.vite/manifest.json`);
  const buildHash = hash(`${distDir}/.vite/manifest.json`);
  const publications = read('src/learn/data/lesson-manifest.json');
  const { tracks } = await import('../src/learn/data/tracks.js');
  const { selectionExamples } = await import('../src/learn/data/selection-examples.js');
  const data = await import('../src/learn/data/selection-data.js');
  const module = tracks.find(track => track.id === 'classical-ml');
  const position = module.topicIds.indexOf(topicId);
  assert.ok(position > 0, 'the topic is registered in the classical-ml route');
  assert.equal(module.topicIds[position - 1], previousId, 'it follows Regularization');
  assert.equal(module.topicIds[position + 1], nextId, 'and precedes Bias-Variance');
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
    await page.locator('.fs-lesson').waitFor();
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
    await checkText(page.locator('.reader-header h1'), /Feature Selection & Importance/);
    await checkText(page.locator('.reader-footer__previous'), /Regularization/);
    await checkText(page.locator('.reader-footer__next'), /Bias/);
    assert.equal(await page.locator('.fs-investigation').count(), 5);
    assert.equal(await page.locator('.fs-figure').count(), 5);
    assert.equal(await page.locator('.fs-practice').count(), 10);
    assert.equal(await page.locator('.python-example').count(), 4);
    const rendered = normalize(await page.locator('.fs-lesson').textContent());
    for (const [key, example] of Object.entries(selectionExamples)) {
      assert.ok(rendered.includes(normalize(example.code)), `Missing actual code for ${key}`);
      assert.ok(rendered.includes(normalize(example.expected)), `Missing actual expected output for ${key}`);
    }
    for (const anchor of await page.locator('.lesson-intro a[href^="#"]').evaluateAll(items => items.map(item => item.getAttribute('href')))) {
      assert.equal(await page.locator(`[id="${anchor.slice(1)}"]`).count(), 1, `Broken lesson anchor ${anchor}`);
    }
    assert.equal(await page.locator('.katex-error').count(), 0);
    for (const entry of data.candidates) {
      assert.ok(rendered.includes(entry.meanAccuracy.toFixed(6)), `the mean for k = ${entry.k} is shown`);
    }
    for (const entry of data.permutationRecords) {
      assert.ok(rendered.includes(entry.mean.toFixed(6)), `${entry.name}'s mean accuracy decrease is shown`);
      assert.ok(rendered.includes(entry.mdi.toFixed(6)), `${entry.name}'s impurity importance is shown`);
    }
    assert.ok(rendered.includes('reserved rows receive no prediction or score here'), 'the protected reserve is stated');
    assert.ok(rendered.includes('not a statement that malic acid has no association with cultivar'), 'the honesty caveat survives');
    assert.ok(rendered.includes('Every investigation asks for a prediction before it shows an answer'), 'the intro promises what the page keeps');
    assert.ok(rendered.includes(data.provenance.license), 'the licence travels with the data');
    assert.ok(rendered.includes(data.provenance.sha256), 'and so does the dataset hash');
    assert.equal((rendered.match(/Before running:/g) ?? []).length, 4, 'one Before running per program');
    const asset = await page.request.get(`${base}${data.provenance.file}`);
    assert.equal(asset.status(), 200);
    const bytes = await asset.body();
    assert.equal(bytes.length, data.provenance.bytes, 'the served file is byte-for-byte the packet file');
    assert.equal(createHash('sha256').update(bytes).digest('hex'), data.provenance.sha256, 'and matches its recorded hash');
    assert.equal(bytes.toString('latin1').split(/\r?\n/)[0].split(',').length, 14, 'fourteen comma-separated columns');
    const attribution = await page.request.get(`${base}${data.provenance.attribution}`);
    assert.equal(attribution.status(), 200);
    assert.ok((await attribution.text()).includes(data.provenance.sha256), 'the attribution carries the hash too');
    records.push({ case: 'Complete visible code and output for four programs, nine route anchors, five figures, ten practice tasks, every published mean accuracy and permutation value, the served unchanged dataset with its hash and attribution, and the module sequence' });

    // ------------------------------------------------ 2. count information (I1)
    const information = page.locator('.fs-investigation').nth(0);
    assert.equal(await information.locator('input[type="radio"]:checked').count(), 0, 'no prediction is preselected');
    assert.ok(await information.getByRole('button', { name: 'Apply and check' }).isDisabled(), 'checking waits for a choice');
    assert.equal(await information.locator('.fs-verdict').count(), 0, 'no answer before a prediction');
    assert.equal(await information.locator('svg').count(), 0, 'and nothing is drawn on first paint');
    await information.getByLabel('Some of it', { exact: true }).check();
    await information.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(information.locator('.fs-verdict'), /Your prediction matches: Some of it\./);
    await checkText(information.locator('.fs-readout'), /I\(X; Y\) = 0\.188721876 bits/);
    await checkText(information, /entropy equal within 10⁻¹² even though they prefer opposite labels/);
    await screenshot(information, 'selection-information-desktop.png');
    await information.getByRole('button', { name: 'Contrast: 4/0/0/4' }).click();
    assert.equal(await information.locator('input[type="radio"]:checked').count(), 0, 'a preset retires the choice');
    await information.getByLabel('All of it', { exact: true }).check();
    await information.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(information.locator('.fs-verdict'), /Your prediction matches: All of it\./);
    await checkText(information.locator('.fs-readout'), /H\(Y \| X\) = 0 bits/);
    await information.getByRole('button', { name: 'Null for association: 2/2/2/2' }).click();
    await information.getByLabel('None of it', { exact: true }).check();
    await information.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(information.locator('.fs-readout'), /A zero estimate on a sample is not a certificate of population independence/);
    await information.getByRole('button', { name: 'Null for the estimate: every count × 3' }).click();
    await information.getByLabel('Some of it', { exact: true }).check();
    await information.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(information.locator('.fs-readout'), /I\(X; Y\) = 0\.188721876 bits/);
    await checkText(information.locator('.fs-readout'), /24 observations and 72 observations support the same estimate very differently/);
    await screenshot(information, 'selection-information-scaled-desktop.png');
    await information.getByRole('button', { name: 'Practice 1: 2/0/0/6' }).click();
    await information.getByLabel('All of it', { exact: true }).check();
    await information.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(information.locator('.fs-readout'), /H\(Y\) = 0\.811278124 bits/);
    await information.getByRole('spinbutton', { name: /^X=0, Y=0/ }).fill('5');
    await checkText(information.locator('.fs-pending'), /Inputs changed; record a new prediction/);
    assert.equal(await information.locator('.fs-verdict').count(), 0, 'and the stale feedback is hidden');
    await information.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'Count investigation: nothing preselected and nothing drawn before a prediction, the worked 0.188722 bits, the perfect-copy and independent contrasts, the count-scaling null that changes the sample size but not the estimate, the practice fixture, and an edit that retires both the prediction and its feedback' });

    // ------------------------------------------------- 3. subset search (I2)
    const subset = page.locator('.fs-investigation').nth(1);
    assert.equal(await subset.locator('input[type="radio"]:checked').count(), 0, 'the lattice prediction opens unset');
    await checkText(subset, /the current subset is \{ \} with accuracy 0\.5/);
    await subset.getByLabel('At the empty subset', { exact: true }).check();
    await subset.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(subset.locator('.fs-verdict'), /Your prediction matches: At the empty subset\./);
    await checkText(subset.locator('.fs-readout'), /stopped because no strictly higher score was available/);
    await checkText(subset.locator('.fs-readout'), /The pair \{A, B\} scores 1, which this rule never evaluated/);
    await screenshot(subset, 'selection-subset-desktop.png');
    await subset.getByRole('button', { name: 'Policy change: XOR with two forced additions' }).click();
    await subset.getByLabel('At both inputs', { exact: true }).check();
    await subset.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(subset.locator('.fs-verdict'), /Your prediction matches: At both inputs\./);
    await checkText(subset.locator('.fs-readout'), /not a universally better one/);
    await screenshot(subset, 'selection-subset-forced-desktop.png');
    await subset.getByRole('button', { name: 'Contrast: Y = A, labels 0, 0, 1, 1' }).click();
    await subset.getByLabel('At one input', { exact: true }).check();
    await subset.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(subset.locator('.fs-readout'), /\{ \} → \{A\}/);
    await subset.getByRole('button', { name: 'Null: the same world displayed bottom-up' }).click();
    await subset.getByLabel('At the empty subset', { exact: true }).check();
    await subset.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(subset.locator('.fs-readout'), /reordering a complete world changes no computation/);
    await subset.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'Lattice investigation: the current subset score and candidate identities shown before committing, the strict rule stopping at the empty set with its announced reason, the forced policy reaching the pair, Y = A found, and the display-order null leaving every score identical' });

    // ---------------------------------------------- 4. donor permutation (I3)
    const donor = page.locator('.fs-investigation').nth(2);
    assert.equal(await donor.locator('input[type="radio"]:checked').count(), 0, 'the donor prediction opens unset');
    await donor.getByLabel('It increases', { exact: true }).check();
    await donor.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(donor.locator('.fs-verdict'), /Your prediction matches: It increases\./);
    await checkText(donor.locator('.fs-readout'), /the difference is \+4 in squared target units/);
    await screenshot(donor, 'selection-donor-desktop.png');
    await donor.getByRole('button', { name: 'Null: shuffle a column the model never reads' }).click();
    await donor.getByLabel('No change within floating-point precision', { exact: true }).check();
    await donor.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(donor.locator('.fs-readout'), /coefficient zero, so the prediction never reads it/);
    await checkText(donor.locator('.fs-readout'), /not about the sensor in the world/);
    await screenshot(donor, 'selection-donor-null-desktop.png');
    await donor.getByRole('button', { name: 'Null: the identity donor map' }).click();
    await donor.getByLabel('No change within floating-point precision', { exact: true }).check();
    await donor.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(donor.locator('.fs-readout'), /donated to themselves/);
    await donor.getByRole('button', { name: 'Practice 3: average model, donor 0, 2, 1, 3' }).click();
    await donor.getByLabel('It increases', { exact: true }).check();
    await donor.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(donor.locator('.fs-readout'), /the difference is \+0\.5 in squared target units/);
    await checkText(donor, /Rows 0, 3 donated to themselves/);
    await screenshot(donor, 'selection-donor-practice-desktop.png');
    // The three-model contrast is the section's point and must be on screen.
    await checkText(donor, /first sensor only/);
    await checkText(donor, /average of sensors/);
    // A donor map that is not a permutation is refused with a local reason.
    await donor.getByRole('button', { name: 'Baseline: the first-sensor model' }).click();
    await donor.getByLabel('row 0 donor').selectOption('1');
    await checkText(donor, /must appear exactly once/);
    assert.ok(await donor.getByRole('button', { name: 'Apply and check' }).isDisabled(), 'an invalid donor map blocks Apply');
    await screenshot(donor, 'selection-donor-invalid-desktop.png');
    await donor.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'Donor investigation: the +4 increase for the first-sensor model, the unused-column and identity-donor nulls, the practice donor with two fixed points giving +0.5, the three-model contrast table, and an invalid donor ordering refused with a local reason rather than silently repaired' });

    // ---------------------------------------------- 5. coalition game (I4)
    const coalition = page.locator('.fs-investigation').nth(3);
    assert.equal(await coalition.locator('input[type="radio"]:checked').count(), 0, 'the coalition prediction opens unset');
    await coalition.getByLabel('Feature B', { exact: true }).check();
    await coalition.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(coalition.locator('.fs-verdict'), /Your prediction matches: Feature B\./);
    await checkText(coalition.locator('.fs-readout'), /φ_A = 5; φ_B = 6/);
    await screenshot(coalition, 'selection-coalition-desktop.png');
    await coalition.getByRole('button', { name: 'Contrast: reference (0, 0) and (1, 1)' }).click();
    await coalition.getByLabel('Feature B', { exact: true }).check();
    await coalition.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(coalition.locator('.fs-readout'), /Baseline v\(∅\) = 1\.5/);
    await checkText(coalition.locator('.fs-readout'), /is 1\.25\. These quantities can differ/);
    await checkText(coalition.locator('.fs-readout'), /f\(x\) = 11/);
    await screenshot(coalition, 'selection-coalition-two-row-desktop.png');
    await coalition.getByRole('button', { name: 'Null for order: γ = 0' }).click();
    await coalition.getByLabel('Feature B', { exact: true }).check();
    await coalition.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(coalition, /each feature adds the same amount whenever it arrives/);
    await coalition.getByRole('button', { name: 'Null: change only the observed target' }).click();
    await coalition.getByLabel('Feature B', { exact: true }).check();
    await coalition.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(coalition.locator('.fs-readout'), /moved none of these numbers, because this game explains a model output and never reads a label/);
    await coalition.getByRole('button', { name: 'Practice 4: x = (1, 2), γ = 2, reference (0, 0)' }).click();
    await coalition.getByLabel('Feature B', { exact: true }).check();
    await coalition.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(coalition.locator('.fs-readout'), /φ_A = 3; φ_B = 4/);
    await coalition.getByRole('button', { name: 'The instance equals its only reference row' }).click();
    await coalition.getByLabel('Equal within 10⁻¹²', { exact: true }).check();
    await coalition.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(coalition.locator('.fs-readout'), /This instance matches every reference row, so both contributions are zero/);
    await screenshot(coalition, 'selection-coalition-zero-desktop.png');
    await coalition.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'Coalition investigation: the zero-reference allocation 5 and 6, the two-row reference changing the baseline to 1.5 while the prediction stays at 11 and the mean output differs from the output at the mean input, the γ = 0 order null, the observed-target null, practice 4, and an instance at its own reference receiving exactly zero' });

    // -------------------------------------------------- 6. wine inference (I5)
    const wine = page.locator('.fs-investigation').nth(4);
    assert.equal(await wine.locator('input[type="radio"]:checked').count(), 0, 'the wine prediction opens unset');
    await wine.getByLabel('Exactly zero', { exact: true }).check();
    await wine.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(wine.locator('.fs-verdict'), /Your prediction matches: Exactly zero\./);
    await checkText(wine, /Baseline 0\.33 over 100 reference rows/);
    await checkText(wine, /Malic acid receives exactly zero/);
    await checkText(wine, /12\.78000020980835/);
    await screenshot(wine, 'selection-wine-desktop.png');
    await wine.getByRole('button', { name: 'Contrast: alcohol 12.51 → 13.5' }).click();
    await wine.getByLabel('Exactly one', { exact: true }).check();
    await wine.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(wine.locator('.fs-verdict'), /Your prediction matches: Exactly one\./);
    await checkText(wine, /alcohol \+0\.216667/);
    await screenshot(wine, 'selection-wine-alcohol-desktop.png');
    await wine.getByRole('button', { name: 'Null: malic acid 1.73 → 4.1' }).click();
    await wine.getByLabel('Exactly zero', { exact: true }).check();
    await wine.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(wine, /Baseline 0\.33 over 100 reference rows/);
    await checkText(wine, /Malic acid receives exactly zero/);
    await screenshot(wine, 'selection-wine-malic-null-desktop.png');
    await wine.getByRole('button', { name: 'Reference contrast: the class-3 cohort' }).click();
    await wine.getByLabel('Exactly zero', { exact: true }).check();
    await wine.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(wine, /all-class-3 cohort/);
    await checkText(wine, /Baseline 0 over 12 reference rows/);
    await screenshot(wine, 'selection-wine-cohort-desktop.png');
    // An out-of-range edit is marked as extrapolation rather than refused.
    await wine.getByRole('button', { name: 'Recorded: source row 104' }).click();
    await wine.getByRole('spinbutton', { name: /^alcohol/ }).fill('16');
    await wine.getByLabel('Exactly one', { exact: true }).check();
    await wine.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(wine, /Extrapolation:/);
    await checkText(wine, /describes no observed specimen/);
    await screenshot(wine, 'selection-wine-extrapolation-desktop.png');
    await wine.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'Wine investigation: the recorded baseline 0.33 and class-1 probability zero with the exact float32 split trace, the alcohol contrast reaching probability one, the malic-acid null leaving every value alone, the class-3 cohort reference changing the baseline to zero without changing the prediction, and an out-of-range edit marked as extrapolation rather than refused' });

    // ------------------------------------------------------------- 7. figures
    const routes = page.locator('.fs-figure').nth(0);
    await checkText(routes, /refits a model/);
    await checkText(routes, /outside these tools/);
    const xorFigure = page.locator('.fs-figure').nth(1);
    await checkText(xorFigure, /I\(\(A, B\); Y\)/);
    await checkText(xorFigure, /their sum, if you added the two rankings 2/);
    const arrival = page.locator('.fs-figure').nth(2);
    await checkText(arrival, /both orders end at v = 11/);
    await arrival.getByLabel('Show the paths up to').selectOption('0');
    await arrival.getByLabel('Show the paths up to').selectOption('2');
    const procedure = page.locator('.fs-figure').nth(3);
    await checkText(procedure, /never predicted, never scored/);
    await checkText(procedure, /[Oo]nly k = 3, 6, 13 were fitted/);
    await checkText(procedure, /Normalized training-impurity decrease/);
    await procedure.getByLabel('Retained size to inspect').selectOption('13');
    await checkText(procedure, /At k = 13 the three folds kept/);
    await procedure.getByLabel('Retained size to inspect').selectOption('6');
    const treeFigure = page.locator('.fs-figure').nth(4);
    await checkText(treeFigure, /12\.78000020980835/);
    await treeFigure.getByLabel('Coalition to inspect').selectOption('4');
    await checkText(treeFigure, /v\(retain flavanoids\) = 0\.42/);
    // Anchor by role and a start-anchored name: a bare 'Field' also matches the
    // tree diagram's description and a table region.
    await treeFigure.getByRole('combobox', { name: /^Field/ }).selectOption('1');
    await checkText(treeFigure, /Every attribution for malic acid is exactly zero across all twelve/);
    await treeFigure.getByRole('combobox', { name: /^Field/ }).selectOption('0');
    for (let index = 0; index < 5; index += 1) {
      await screenshot(page.locator('.fs-figure').nth(index), `selection-figure-${index + 1}-desktop.png`);
    }
    records.push({ case: 'Figures carry their required content: the five routes badge where fitting happens, the exact-copy table shows the double count, both arrival orders end at the same value, the boundary diagram closes the unscored reserve and the two importance panels stay on separate axes, and the tree figure exposes the exact threshold and a coalition value built from a hundred hybrid rows' });

    // Sequential audit: small genuine effects and correct explanations of nulls.
    const fillNumber = async (lab, name, value) => lab.getByRole('spinbutton', { name }).fill(String(value));
    for (const [label, value] of [['X=0, Y=0', 9999], ['X=0, Y=1', 9998], ['X=1, Y=0', 10000], ['X=1, Y=1', 9999]]) {
      await fillNumber(information, label, value);
    }
    await information.getByLabel('Some of it', { exact: true }).check();
    await information.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(information.locator('.fs-verdict'), /Your prediction matches: Some of it/);
    await checkText(information.locator('.fs-readout'), /4\.510 × 10⁻¹⁸/);
    assert.doesNotMatch(await information.locator('.fs-readout').innerText(), /match what independence would predict exactly|prefer opposite labels/);
    await screenshot(information, 'selection-information-near-independent-desktop.png');
    for (const [guess, accepted] of [[1.0001, true], [.9999, true], [1.000101, false], [.999899, false]]) {
      await information.getByRole('button', { name: 'Contrast: 4/0/0/4' }).click();
      await checkText(information.locator('.fs-numeric-guess'), /Answers within 0\.0001 are accepted/);
      await information.getByLabel('All of it', { exact: true }).check();
      await information.getByLabel('Optional: the mutual information in bits').fill(String(guess));
      await information.getByRole('button', { name: 'Apply and check' }).click();
      await checkText(information.locator('.fs-verdict'), accepted ? /within 0\.0001/ : /outside 0\.0001/);
    }
    await information.getByRole('button', { name: 'Reset', exact: true }).click();

    for (let row = 0; row < 4; row += 1) {
      await fillNumber(donor, `row ${row} sensor 1`, row < 2 ? -.0001 : .0001);
      await fillNumber(donor, `row ${row} sensor 2`, row < 2 ? -.0001 : .0001);
      await fillNumber(donor, `row ${row} target`, 0);
    }
    await fillNumber(donor, 'Fixed coefficient w₁', .0001);
    await fillNumber(donor, 'Fixed coefficient w₂', -.0001);
    await donor.getByLabel('It increases', { exact: true }).check();
    await donor.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(donor.locator('.fs-verdict'), /Your prediction matches: It increases/);
    await checkText(donor.locator('.fs-verdict'), /4\.000 × 10⁻¹⁶/);
    await screenshot(donor, 'selection-donor-tiny-increase-desktop.png');
    await donor.getByRole('button', { name: 'Reset', exact: true }).click();

    await fillNumber(coalition, 'Explained a', 2);
    await fillNumber(coalition, 'Explained b', 2);
    await fillNumber(coalition, 'Interaction γ', -1);
    await coalition.getByLabel('Equal within 10⁻¹²', { exact: true }).check();
    await coalition.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(coalition.locator('.fs-readout'), /cancelling increments/);
    assert.doesNotMatch(await coalition.locator('.fs-readout').innerText(), /This instance matches every reference row/);
    // A zero level really spans zero distance; no decorative minimum width.
    const baseWidths = await coalition.locator('.fs-waterfall rect.is-base').evaluateAll(nodes => nodes.map(node => Number(node.getAttribute('width'))));
    assert.deepEqual(baseWidths, [0, 0]);
    await screenshot(coalition, 'selection-coalition-cancellation-desktop.png');
    await fillNumber(coalition, 'Explained a', 0);
    await fillNumber(coalition, 'Explained b', 3);
    await fillNumber(coalition, 'Interaction γ', 1);
    await coalition.getByLabel('Feature B', { exact: true }).check();
    await coalition.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(coalition, /For these particular coordinates and reference rows/);
    assert.doesNotMatch(await coalition.innerText(), /With no interaction term/);
    await fillNumber(coalition, 'Explained a', .0001);
    await fillNumber(coalition, 'Explained b', 0);
    await fillNumber(coalition, 'Interaction γ', .0001);
    await fillNumber(coalition, 'reference 0 b', -.0001);
    await coalition.getByLabel('Equal within 10⁻¹²', { exact: true }).check();
    await coalition.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(coalition.locator('.fs-verdict'), /Their difference is −1\.000 × 10⁻¹²; the tie tolerance is 10⁻¹²/);
    await coalition.getByRole('button', { name: 'Reset', exact: true }).click();
    const repeatCircles = await page.locator('.fs-figure').nth(3).locator('svg[aria-label^="Accuracy decrease"] circle').evaluateAll(nodes => nodes.map(node => Number(node.getAttribute('cx'))));
    assert.equal(repeatCircles.length, 80, 'all twenty repeats for four features are actually drawn');
    assert(repeatCircles.every(x => x >= 86 && x <= 292), 'all repeats lie inside the permutation axis, including drop0.5');
    records.push({ case: 'Sequential semantics: positive determinant-one MI versus true independence, inclusive numeric boundary, positive4e-16 permutation change, cancellation versus matching reference, order independence with a nonlinear model, declared small-contribution tie, zero-width levels and all80 donor-repeat points within their axis' });

    // The overflow cue must describe layout, never paint a gold magnitude behind data.
    for (const width of [1366, 390]) {
      await page.setViewportSize({ width, height: 1000 });
      await settle(page);
      await page.waitForFunction(() => [...document.querySelectorAll('.fs-scroll-content')].filter(node => node.getBoundingClientRect().width > 0).every(node => {
        const hint = node.parentElement.querySelector('.fs-scroll-hint');
        return !hint.hidden === (node.scrollWidth > node.clientWidth + 1);
      }));
      const cues = await page.locator('.fs-scroll-content').evaluateAll(nodes => nodes.filter(node => node.getBoundingClientRect().width > 0).map(node => ({
        overflowing: node.scrollWidth > node.clientWidth + 1,
        hint: !node.parentElement.querySelector('.fs-scroll-hint').hidden,
        background: getComputedStyle(node).backgroundImage,
        parentBackground: getComputedStyle(node.parentElement).backgroundImage,
        before: getComputedStyle(node, '::before').backgroundImage,
        after: getComputedStyle(node, '::after').backgroundImage,
        focusable: node.getAttribute('tabindex'),
        description: node.getAttribute('aria-describedby'),
      })));
      assert(cues.length > 10, 'check real SVG and table frames, not an empty selector');
      for (const cue of cues) {
        assert.equal(cue.hint, cue.overflowing, `hint matches actual overflow at ${width}`);
        assert.deepEqual([cue.background, cue.parentBackground, cue.before, cue.after], ['none', 'none', 'none', 'none']);
        if (cue.overflowing) { assert.equal(cue.focusable, '0'); assert(cue.description); }
      }
      assert(cues.some(cue => !cue.overflowing), 'fitting content does not advertise scrolling');
      if (width === 390) {
        const svgFrames = page.locator('.fs-scroll-content').filter({ has: page.locator('svg') });
        const index = await svgFrames.evaluateAll(nodes => nodes.findIndex(node => node.scrollWidth > node.clientWidth + 1 && node.getBoundingClientRect().width > 0));
        assert(index >= 0, 'a mobile SVG really overflows and can test keyboard movement');
        const frame = svgFrames.nth(index);
        await frame.scrollIntoViewIfNeeded();
        await frame.evaluate(node => { node.scrollLeft = 0; });
        await frame.focus();
        await page.keyboard.press('ArrowRight');
        await page.waitForFunction(() => document.activeElement?.classList.contains('fs-scroll-content') && document.activeElement.scrollLeft > 0);
        await frame.evaluate(node => { node.scrollLeft = 0; });
        await screenshot(page.locator('.fs-figure').nth(0), 'selection-scroll-cue-390.png');
      } else await screenshot(page.locator('.fs-figure').nth(0), 'selection-scroll-cue-desktop.png');
    }
    await page.setViewportSize({ width: 1366, height: 1000 });
    records.push({ case: 'Scroll regions have no background or pseudo-element gradients; measured overflow alone shows a plain cue and focusable named frame; a mobile SVG scrolls with the keyboard and fitting frames omit the cue' });

    // ------------------------------------------- 8. drawn geometry at five widths
    assert.equal(await page.getByRole('button', { name: /without recording a prediction/ }).count(), 0,
      'no investigation offers a way round the prediction');
    // Every investigation's drawing lives inside {state.result && …}. A sweep run
    // after the per-lab Reset clicks therefore inspects the FIGURES only and
    // silently passes over seven unmounted SVGs. Commit all five first, then
    // assert that the sweep actually had them in front of it.
    const commitAll = async () => {
      for (let index = 0; index < 5; index += 1) {
        const lab = page.locator('.fs-investigation').nth(index);
        if (await lab.locator('.fs-verdict').count()) continue;
        await lab.locator('input[type="radio"]').first().check();
        const guess = lab.locator('.fs-numeric-guess input');
        if (await guess.count() && await guess.isEditable()) await guess.fill('1');
        await lab.getByRole('button', { name: 'Apply and check' }).click();
      }
      await settle(page);
    };
    await commitAll();
    const drawnSubjects = () => page.evaluate(() => {
      const visible = element => {
        const box = element.getBoundingClientRect();
        return box.width > 0 && box.height > 0 && getComputedStyle(element).visibility !== 'hidden';
      };
      const all = [...document.querySelectorAll('.fs-lesson svg')].filter(visible);
      return {
        total: all.length,
        inInvestigations: all.filter(svg => svg.closest('.fs-investigation')).length,
        labels: all.reduce((sum, svg) => sum + [...svg.querySelectorAll('text')].filter(visible).length, 0),
      };
    });
    const sweptWidths = [1366, 1024, 768, 390, 320];
    const sweepCoverage = [];
    for (const width of sweptWidths) {
      await page.setViewportSize({ width, height: 1000 });
      await settle(page);
      const subjects = await drawnSubjects();
      // Fail loudly when the subject set is empty or has lost the investigations:
      // an inspector that finds nothing must never read as a clean pass.
      assert.ok(subjects.total >= 12,
        `the geometry sweep at ${width}px inspected only ${subjects.total} visible SVGs`);
      assert.ok(subjects.inInvestigations >= 7,
        `the geometry sweep at ${width}px saw only ${subjects.inInvestigations} investigation SVGs; the labs were not revealed`);
      assert.ok(subjects.labels >= 200,
        `the geometry sweep at ${width}px saw only ${subjects.labels} labels`);
      const issues = await page.evaluate(inspectLessonVisualLayout, '.fs-lesson');
      assert.deepEqual(issues.flatMap(figure => figure.issues), [], `Figure layout collides at ${width}px`);
      sweepCoverage.push({ width, ...subjects });
    }
    // The label floor: an SVG authored at viewBox 340 is scaled by CSS, so a
    // declared 10px label can render far smaller. Measure what a reader sees.
    const smallestLabel = async () => page.evaluate(() => {
      let smallest = Infinity;
      let where = null;
      document.querySelectorAll('.fs-lesson svg').forEach(svg => {
        const box = svg.getBoundingClientRect();
        const view = svg.viewBox.baseVal;
        if (!box.width || !view || !view.width) return;
        const scale = box.width / view.width;
        svg.querySelectorAll('text').forEach(text => {
          const rect = text.getBoundingClientRect();
          if (!rect.width || !rect.height) return;
          const effective = parseFloat(getComputedStyle(text).fontSize) * scale;
          if (effective < smallest) { smallest = effective; where = text.textContent.trim().slice(0, 40); }
        });
      });
      return { smallest, where };
    });
    const floors = [];
    for (const width of [390, 320]) {
      await page.setViewportSize({ width, height: 1000 });
      await settle(page);
      const floor = await smallestLabel();
      assert.ok(floor.smallest >= 9.4,
        `at ${width}px the smallest drawn label renders at ${floor.smallest.toFixed(1)} effective px ("${floor.where}")`);
      floors.push({ width, smallestEffectivePx: Number(floor.smallest.toFixed(2)), label: floor.where });
    }
    await page.setViewportSize({ width: 1366, height: 1000 });
    records.push({
      case: 'With all five investigations revealed, no label leaves its own SVG, overlaps another label or is crossed by a foreground line, at five widths; the sweep asserts it actually had the investigation diagrams in front of it, and every drawn label clears a 9.4 effective-px floor at 390 and 320',
      sweepCoverage, labelFloors: floors,
    });

    // ------------------------------- 8b. a WRONG prediction, and the contract in all five
    // Every other verdict assertion in this file matches "Your prediction
    // matches". A regression that made `correct` unconditionally true would pass
    // all of them, so the miss branch is exercised here explicitly.
    await page.locator('.fs-investigation').nth(0).getByRole('button', { name: 'Reset', exact: true }).click();
    const missLab = page.locator('.fs-investigation').nth(0);
    await missLab.getByLabel('None of it', { exact: true }).check();
    await missLab.getByLabel('Optional: the mutual information in bits').fill('0.9');
    await missLab.getByRole('button', { name: 'Apply and check' }).click();
    const missVerdict = missLab.locator('.fs-verdict');
    assert.ok(await missVerdict.evaluate(node => node.classList.contains('is-miss')),
      'a wrong prediction is marked as a miss');
    await checkText(missVerdict, /You recorded None of it; the calculation gives Some of it\./);
    await checkText(missVerdict, /≠/);
    await checkText(missVerdict, /You wrote 0\.9 for the information; the calculation gives 0\.188721876, outside 0\.0001/);
    await screenshot(missLab, 'selection-information-miss-desktop.png');
    // And the numeric guess inside tolerance reads as inside.
    await missLab.getByRole('button', { name: 'Reset', exact: true }).click();
    await missLab.getByLabel('Some of it', { exact: true }).check();
    await missLab.getByLabel('Optional: the mutual information in bits').fill('0.188722');
    await missLab.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(missLab.locator('.fs-verdict'), /within 0\.0001/);
    await missLab.getByRole('button', { name: 'Reset', exact: true }).click();
    // The "nothing before a prediction" and "retired on edit" contract, in all five.
    const firstEdit = [
      async lab => lab.getByRole('spinbutton', { name: /^X=0, Y=0/ }).fill('5'),
      async lab => lab.getByRole('spinbutton', { name: /^Target for A=0, B=0/ }).fill('1'),
      async lab => lab.getByRole('spinbutton', { name: /^row 0 sensor 1/ }).fill('2'),
      async lab => lab.getByRole('spinbutton', { name: /^Explained a/ }).fill('4'),
      async lab => lab.getByRole('spinbutton', { name: /^alcohol/ }).fill('13'),
    ];
    for (let index = 0; index < 5; index += 1) {
      const lab = page.locator('.fs-investigation').nth(index);
      await lab.getByRole('button', { name: 'Reset', exact: true }).click();
      assert.equal(await lab.locator('.fs-verdict').count(), 0, `investigation ${index + 1} shows no verdict before a prediction`);
      assert.equal(await lab.locator('svg').count(), 0, `investigation ${index + 1} draws nothing before a prediction`);
      assert.equal(await lab.locator('table').count(), 0, `investigation ${index + 1} shows no table before a prediction`);
      assert.equal(await lab.locator('input[type="radio"]:checked').count(), 0, `investigation ${index + 1} preselects nothing`);
      assert.ok(await lab.getByRole('button', { name: 'Apply and check' }).isDisabled(),
        `investigation ${index + 1} cannot be applied without a prediction`);
      await lab.locator('input[type="radio"]').first().check();
      assert.equal(await lab.locator('.fs-verdict').count(), 0, `investigation ${index + 1} reveals nothing on selecting a radio alone`);
      const guess = lab.locator('.fs-numeric-guess input');
      if (await guess.count() && await guess.isEditable()) await guess.fill('1');
      await lab.getByRole('button', { name: 'Apply and check' }).click();
      assert.equal(await lab.locator('.fs-verdict').count(), 1, `investigation ${index + 1} shows a verdict after Apply`);
      assert.ok(await lab.locator('svg').count() > 0, `investigation ${index + 1} draws its result after Apply`);
      await firstEdit[index](lab);
      await settle(page);
      assert.equal(await lab.locator('.fs-verdict').count(), 0, `investigation ${index + 1} retires the verdict on an edit`);
      assert.equal(await lab.locator('svg').count(), 0, `investigation ${index + 1} retires its drawing on an edit`);
      assert.equal(await lab.locator('input[type="radio"]:checked').count(), 0, `investigation ${index + 1} retires the recorded choice on an edit`);
      await checkText(lab.locator('.fs-pending'), /Inputs changed; record a new prediction/);
      await lab.getByRole('button', { name: 'Reset', exact: true }).click();
    }
    records.push({ case: 'A wrong prediction is marked as a miss with the ≠ mark, the recorded-versus-calculated sentence and an outside-tolerance numeric message; a correct numeric guess reads as within tolerance; and in all five investigations nothing is drawn or tabulated before a prediction, a radio alone reveals nothing, and a single edit retires the verdict, the drawing and the recorded choice' });

    // ------------------------------------------------------ 9. loading closure
    const localScripts = [...new Set(requests.map(address => new URL(address)).filter(address => address.origin === new URL(base).origin && address.pathname.endsWith('.js')).map(address => address.pathname.slice(1)))];
    assert.deepEqual(localScripts.filter(filename => bodyFiles.has(filename)), [bodyFile]);
    assert.deepEqual(localScripts.filter(filename => !allowedScripts.has(filename)), []);
    assert.ok(!requests.some(address => address.includes('/outlines/')));
    assert.ok(!requests.some(address => address.includes('wine.data') && !address.includes('learn-assets')),
      'the page never fetches the dataset to render');
    records.push({
      case: 'Fresh route requests only the selected lesson and its shared closure',
      requestedScripts: localScripts,
      rawScriptBytes: localScripts.reduce((sum, filename) => sum + fs.statSync(path.join(distDir, filename)).size, 0),
      gzipEstimateBytes: localScripts.reduce((sum, filename) => sum + gzipSync(fs.readFileSync(path.join(distDir, filename))).length, 0),
      interpretation: 'Built requested file sizes; the gzip estimate is not measured network compression or latency.',
    });

    // --------------------------------------------- 10. narrow widths and tables
    // The narrow-width captures must show the labs REVEALED: an unrevealed lab
    // photographs its controls, not the drawing whose layout is in question.
    await commitAll();
    for (const width of [390, 320]) {
      await page.setViewportSize({ width, height: 844 });
      await page.locator('.fs-lesson details').evaluateAll(items => items.forEach(item => { item.open = true; }));
      await settle(page);
      assert.ok((await drawnSubjects()).inInvestigations >= 7,
        `the ${width}px captures are taken with every investigation revealed`);
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `Document overflows at ${width}px`);
      const mathOverflow = await page.locator('.fs-lesson .katex-display').evaluateAll(items => items.filter(item => item.scrollWidth > item.clientWidth + 1).map(item => item.textContent));
      assert.deepEqual(mathOverflow, [], `Overflowing formulas at ${width}px`);
      assert.equal(await page.locator('.katex-error').count(), 0);
      // Tables stack below the breakpoint rather than scrolling sideways.
      const stacking = await page.locator('.fs-table-scroll.is-stackable').evaluateAll(items => items.map(item => {
        const row = item.querySelector('tbody tr');
        const cell = row && row.querySelector('td');
        return {
          headStacked: getComputedStyle(item.querySelector('thead')).position === 'absolute',
          cellBlockish: cell ? getComputedStyle(cell).display !== 'table-cell' : true,
          overflows: item.scrollWidth > item.clientWidth + 1,
        };
      }));
      assert.ok(stacking.length > 0, `stackable tables are present at ${width}px`);
      assert.deepEqual(stacking.filter(entry => !entry.headStacked || !entry.cellBlockish || entry.overflows), [],
        `Every stackable table stacks without horizontal overflow at ${width}px`);
      if (width === 320) {
        await screenshot(page.locator('.fs-figure').nth(0), 'selection-figure-1-320.png');
        await screenshot(page.locator('.fs-figure').nth(3), 'selection-figure-4-320.png');
        await screenshot(page.locator('.fs-investigation').nth(0), 'selection-information-320.png');
        await screenshot(page.locator('.fs-investigation').nth(4), 'selection-wine-320.png');
      }
      if (width === 390) {
        for (let index = 0; index < 5; index += 1) {
          await screenshot(page.locator('.fs-investigation').nth(index), `selection-investigation-${index + 1}-390.png`);
          await screenshot(page.locator('.fs-figure').nth(index), `selection-figure-${index + 1}-390.png`);
        }
      }
      records.push({ case: `Narrow ${width}px layout: no document overflow, no overflowing formula, and every data table stacked into labelled rows` });
    }
    await page.setViewportSize({ width: 1366, height: 1000 });

    // -------------------------------------------------------- 11. sequence
    await page.locator('.reader-complete').click();
    assert.equal(new URL(page.url()).pathname.split('/').at(-1), topicId);
    await page.reload({ waitUntil: 'domcontentloaded' });
    await ready(page);
    assert.equal(await page.evaluate(id => JSON.parse(localStorage.getItem('kd-progress'))[id], topicId), true);
    await page.locator('.reader-footer__next').click();
    await page.waitForURL(`**/${nextId}?module=classical-ml`);
    records.push({ case: 'Completion persists under the stable ID without auto-advance; Next opens the actual successor' });
    assert.deepEqual(errors, []);
    assert.deepEqual(failedAssets, []);
    await context.close();

    // -------------------------------------------------------- 12. recovery
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
      routePosition: position + 1,
      moduleTopicCount: module.topicIds.length,
      dataset: { file: data.provenance.file, sha256: data.provenance.sha256, bytes: data.provenance.bytes, reservedScored: data.split.reservedScored },
      records, screenshots: screenshotPaths,
      visualInterpretation: 'The intended Space Grotesk font loaded before the checks. Screenshots capture informative states: a matched prediction, a retired one, the count-scaling null, the forced search policy, an invalid donor ordering refused, an instance at its own reference, the alcohol contrast and the malic-acid null, the class-3 cohort, a marked extrapolation, and every inline figure at desktop and at 320 or 390 px. They require separate visual inspection.',
    };
    fs.writeFileSync(evidencePath, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify({ status: report.status, cases: records.length, evidencePath, sourceFiles: ownedFiles.length, screenshots: screenshotPaths.length }));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });

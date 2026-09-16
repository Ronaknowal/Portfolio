// Production browser review of the imbalanced-learning lesson: visible content,
// all five investigations with their nulls, figure geometry read back from the
// drawn coordinates, label collisions including curve-shaped ones, SVG type size
// against its own viewBox, narrow layouts and stacked tables, formula width,
// sequence, completion and load-failure recovery.
//
// Run against a preview of a production build:
//   npx vite build --outDir dist-imb
//   npx vite preview --outDir dist-imb --host 127.0.0.1 --port 4188
//   PLAYWRIGHT_PACKAGE=... DIST_DIR=dist-imb LEARNING_BASE_URL=http://127.0.0.1:4188 \
//     node scripts/verify-imbalance-browser.cjs
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { gzipSync } = require('node:zlib');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const { inspectLessonVisualLayout } = require('./lib/lesson-visual-layout.cjs');

const distDir = process.env.DIST_DIR || 'dist-imb';
const base = (process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4188').replace(/\/+$/, '');
const topicId = 'imbalanced-learning-smote-cost-sensitive-learning';
const sourcePath = `src/learn/data/topics/${topicId}.jsx`;
const route = `${base}/learn/path/full-curriculum/${topicId}?module=classical-ml`;
const evidencePath = 'docs/teaching/evidence/imbalance-browser.json';
const read = filename => JSON.parse(fs.readFileSync(filename, 'utf8'));
const hash = filename => createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
const normalize = text => text.replace(/\s+/g, ' ').trim();
const ownedFiles = [
  sourcePath,
  'src/learn/data/imbalance-models.js',
  'src/learn/data/imbalance-data.js',
  'src/learn/data/imbalance-examples.js',
  'src/learn/components/lesson-labs/ImbalanceShared.jsx',
  'src/learn/components/lesson-labs/ImbalanceLabs.jsx',
  'src/learn/components/lesson-labs/ImbalanceFigures.jsx',
  'src/learn/components/lesson-labs/imbalance-labs.css',
  `src/learn/data/curriculum/blueprints/${topicId}.js`,
  'public/learn-assets/imbalanced-learning/yeast.data',
];

/** Runs in the page. Samples every curve along its own geometry and reports the
 * labels it passes through.
 *
 * Copied from scripts/verify-bias-variance-browser.cjs, with this lesson's grid
 * and backplate class names. `scripts/lib/lesson-visual-layout.cjs` iterates
 * `svg.querySelectorAll('line')`, so <path>, <polyline> and <polygon> are
 * invisible to it — which is how four data curves crossed value labels behind a
 * fully green run. This lesson draws population flows as <path> arcs, threshold
 * ladders and risk lines as <polyline>, so that whole class matters here.
 *
 * A label carrying an opaque backplate may be crossed briefly, because the
 * backplate is what a crossing is mitigated with. It may not be travelled along:
 * a backplate hides a line passing behind glyphs, not a curve running the width
 * of the text. Grid lines are background and are skipped.
 */
function sampleCurvesThroughLabels(root) {
  const SAMPLES = 320;
  const HALO_ALLOWANCE = 0.02;
  const findings = [];
  for (const svg of document.querySelectorAll(`${root} svg`)) {
    const labels = [...svg.querySelectorAll('text')]
      .filter(text => text.textContent.trim() && text.getClientRects().length)
      .map(text => ({
        text: text.textContent.trim().slice(0, 40),
        halo: text.classList.contains('imb-halo'),
        box: text.getBoundingClientRect(),
      }));
    if (!labels.length) continue;
    for (const shape of svg.querySelectorAll('path, polyline, polygon')) {
      if (shape.classList.contains('imb-grid')) continue;
      if (typeof shape.getTotalLength !== 'function') continue;
      const length = shape.getTotalLength();
      if (!(length > 0)) continue;
      const matrix = shape.getScreenCTM();
      if (!matrix) continue;
      const counts = labels.map(() => 0);
      for (let step = 0; step <= SAMPLES; step += 1) {
        const local = shape.getPointAtLength(length * step / SAMPLES);
        const x = matrix.a * local.x + matrix.c * local.y + matrix.e;
        const y = matrix.b * local.x + matrix.d * local.y + matrix.f;
        labels.forEach((label, index) => {
          const { box } = label;
          if (x >= box.left && x <= box.right && y >= box.top && y <= box.bottom) counts[index] += 1;
        });
      }
      counts.forEach((count, index) => {
        if (!count) return;
        const fraction = count / (SAMPLES + 1);
        const allowed = labels[index].halo ? HALO_ALLOWANCE : 0;
        if (fraction > allowed) {
          findings.push({
            label: labels[index].text,
            backplate: labels[index].halo,
            shape: shape.getAttribute('class') || shape.tagName,
            fractionOfCurveInsideLabel: Number(fraction.toFixed(4)),
          });
        }
      });
    }
  }
  return findings;
}

/** Runs in the page. Reports the rendered size of every SVG label in CSS pixels.
 *
 * A viewBox wider than the element scales its type down; a viewBox narrower
 * scales it up. The declared 12px inside a ~340-unit box is only legible if the
 * element is roughly that wide, so the measured size is what matters, not the
 * stylesheet's number. Last round four class selectors missed every diagram in
 * an investigation stage wrapper and they rendered at the browser default.
 */
function measureSvgTypeSizes(root) {
  const readings = [];
  for (const svg of document.querySelectorAll(`${root} svg`)) {
    const box = svg.getBoundingClientRect();
    if (!box.width) continue;
    const viewBox = svg.viewBox.baseVal;
    const scale = viewBox && viewBox.width ? box.width / viewBox.width : 1;
    for (const text of svg.querySelectorAll('text')) {
      if (!text.textContent.trim() || !text.getClientRects().length) continue;
      const declared = parseFloat(getComputedStyle(text).fontSize);
      readings.push({
        text: text.textContent.trim().slice(0, 30),
        secondary: text.classList.contains('imb-small'),
        declaredPx: Number(declared.toFixed(2)),
        renderedPx: Number((declared * scale).toFixed(2)),
        scale: Number(scale.toFixed(3)),
        figure: (svg.closest('.imb-figure, .imb-investigation') || {}).className || 'unscoped',
      });
    }
  }
  return readings;
}

(async () => {
  const startedAt = new Date().toISOString();
  const sourceHashes = Object.fromEntries(ownedFiles.map(filename => [filename, hash(filename)]));
  const build = read(`${distDir}/.vite/manifest.json`);
  const buildHash = hash(`${distDir}/.vite/manifest.json`);
  const publications = read('src/learn/data/lesson-manifest.json');
  const { tracks } = await import('../src/learn/data/tracks.js');
  const { imbalanceExamples } = await import('../src/learn/data/imbalance-examples.js');
  const data = await import('../src/learn/data/imbalance-data.js');
  const models = await import('../src/learn/data/imbalance-models.js');
  const module_ = tracks.find(track => track.id === 'classical-ml');
  assert.equal(module_.topicIds[24], topicId, 'the lesson sits at position 25 of its module');
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

  const original = data.methods.find(entry => entry.name === 'original');
  const smoteMethod = data.methods.find(entry => entry.name === 'smote');
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [];
  const screenshotPaths = [];
  const typeReadings = [];
  const checkText = async (locator, pattern) => assert.match(normalize(await locator.innerText()), pattern);
  const settle = page => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  const ready = async page => {
    await page.locator('.imb-lesson').waitFor();
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
    await checkText(page.locator('.reader-header h1'), /Imbalanced Learning/);
    await checkText(page.locator('.reader-header__meta'), /25 of 39 topics on this route/);
    await checkText(page.locator('.reader-footer__previous'), /Bias-Variance/);
    await checkText(page.locator('.reader-footer__next'), /AutoML/);
    assert.equal(await page.locator('.imb-investigation').count(), 5, 'five investigations');
    assert.equal(await page.locator('.imb-figure').count(), 6, 'six inline figures');
    assert.equal(await page.locator('.imb-practice').count(), 10, 'ten practice tasks');
    assert.equal(await page.locator('.python-example').count(), 3, 'three displayed programs');
    const rendered = normalize(await page.locator('.imb-lesson').textContent());
    for (const [key, example] of Object.entries(imbalanceExamples)) {
      assert.ok(rendered.includes(normalize(example.code)), `Missing actual code for ${key}`);
      assert.ok(rendered.includes(normalize(example.expected)), `Missing actual expected output for ${key}`);
    }
    for (const anchor of await page.locator('.lesson-intro a[href^="#"]').evaluateAll(items => items.map(item => item.getAttribute('href')))) {
      assert.equal(await page.locator(`[id="${anchor.slice(1)}"]`).count(), 1, `Broken lesson anchor ${anchor}`);
    }
    assert.equal(await page.locator('.katex-error').count(), 0, 'no formula failed to render');
    // Every recorded outcome the manuscript states is visible on the page.
    for (const method of data.methods) {
      const tuned = models.countsAt(data.inspectionRecords.labels, method.inspectionScores, method.chosenThreshold);
      const cost = models.costOf(tuned, data.study.costFalsePositive, data.study.costFalseNegative);
      assert.ok(rendered.includes(`${tuned.tp} / ${tuned.fp} / ${tuned.fn} / ${tuned.tn}`),
        `${method.name}: its tuned counts are shown`);
      assert.ok(rendered.includes(String(cost)), `${method.name}: its realised cost ${cost} is shown`);
      assert.ok(rendered.includes(method.averagePrecision.toFixed(6)), `${method.name}: its AP is shown`);
    }
    assert.ok(rendered.includes('reserved proteins remain unscored')
      || rendered.includes('No predictions or scores in this lesson'), 'the protected reserve is stated');
    assert.ok(rendered.includes('CC BY 4.0'), 'the licence travels with the data');
    assert.ok(rendered.includes(data.provenance.sha256), 'and so does the file hash');
    assert.equal((rendered.match(/Before running:/g) ?? []).length, 3, 'one Before running per program');
    const asset = await page.request.get(`${base}${data.provenance.file}`);
    assert.equal(asset.status(), 200);
    const bytes = await asset.body();
    assert.equal(bytes.length, data.provenance.bytes, 'the served file is byte-for-byte the packet file');
    assert.equal(createHash('sha256').update(bytes).digest('hex'), data.provenance.sha256, 'and matches its recorded hash');
    assert.equal(bytes.toString('latin1').split('\n')[0].trim().split(/\s+/).length, 10,
      'ten whitespace-separated source fields');
    const attribution = await page.request.get(`${base}${data.provenance.attribution}`);
    assert.equal(attribution.status(), 200);
    const attributionText = await attribution.text();
    assert.ok(attributionText.includes('CC BY 4.0') && attributionText.includes(data.provenance.sha256));
    assert.ok(attributionText.includes('ME2'), 'the attribution records the class definition');
    records.push({ case: 'Complete visible code and output for three programs, every route anchor, six figures, five investigations, ten practice tasks, all five recorded outcome rows with their counts, costs and average precisions, the served unchanged dataset with its hash and attribution, current metadata and module sequence' });

    // ------------------------------------------------- 2. the score queue (I1)
    const queue = page.locator('.imb-investigation').nth(0);
    assert.equal(await queue.locator('input[type="radio"]:checked').count(), 0, 'no prediction is preselected');
    assert.ok(await queue.getByRole('button', { name: 'Apply and check' }).isDisabled(), 'checking waits for a choice');
    assert.equal(await queue.locator('.imb-verdict').count(), 0, 'no answer before a prediction');
    assert.equal(await queue.locator('.imb-plot').count(), 0, 'and no step chart on first paint');
    assert.equal(await queue.locator('.imb-card.is-tp, .imb-card.is-fp, .imb-card.is-fn, .imb-card.is-tn').count(), 0,
      'no record is binned before the gate is applied');
    await checkText(queue.locator('.imb-rail'), /awaiting the gate/);
    await screenshot(queue, 'imbalance-queue-initial-desktop.png');

    // The declared counterexample: raising the gate makes precision worse.
    await queue.getByLabel('It falls', { exact: true }).check();
    await queue.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(queue.locator('.imb-verdict'), /Your prediction matches: It falls\./);
    await checkText(queue, /A0\.9|A \(0\.9|Selected: A/);
    await checkText(queue, /undefined \(TP\+FP = 0\)/);
    assert.equal(await queue.locator('.imb-plot').count(), 1, 'the step chart appears only after the prediction');
    await screenshot(queue, 'imbalance-queue-counterexample-desktop.png');

    // Correcting the highest record's label makes precision one everywhere.
    await queue.getByRole('button', { name: /Correct A/ }).click();
    assert.equal(await queue.locator('input[type="radio"]:checked').count(), 0, 'a setup retires the choice');
    assert.equal(await queue.locator('.imb-verdict').count(), 0, 'and hides the stale verdict');
    // A retired verdict must NOT name an outcome while the next prediction is
    // being recorded: for a null the previous outcome IS the current answer.
    // Only the neutral held-attempt notice may appear here.
    assert.equal(await queue.locator('.imb-history:not(.is-pending)').count(), 0,
      'no outcome-bearing history is shown while the gate is open');
    await checkText(queue.locator('.imb-history.is-pending'), /stays hidden until you apply/);
    const openHistory = normalize(await queue.locator('.imb-history.is-pending').innerText());
    for (const [, optionLabel] of [['decrease', 'It falls'], ['unchanged', 'Exactly unchanged'], ['increase', 'It rises'], ['undefined', 'It becomes undefined']]) {
      assert.ok(!openHistory.includes(optionLabel),
        `the held-attempt notice names no outcome, but it contains "${optionLabel}"`);
    }
    // With every record positive, precision is 1 at both gates: the contrast the
    // manuscript names is that it stops falling, not that it rises.
    await queue.getByLabel('Exactly unchanged', { exact: true }).check();
    await queue.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(queue.locator('.imb-verdict'), /Your prediction matches: Exactly unchanged\./);
    await checkText(queue.locator('.imb-verdict'), /Recall moved decrease, from 1 to 0\.333333/);
    await screenshot(queue, 'imbalance-queue-corrected-desktop.png');

    // The tie null: the same records displayed in the other order.
    await queue.getByRole('button', { name: /Setup for the tie null/ }).click();
    await queue.getByLabel('It falls', { exact: true }).check();
    await queue.getByRole('button', { name: 'Apply and check' }).click();
    const ladderRows = () => queue.locator('.imb-table').first().locator('tbody tr')
      .evaluateAll(rows => rows.map(row => [...row.querySelectorAll('th, td')].map(cell => cell.textContent.trim())));
    const beforeRows = await ladderRows();
    await queue.getByRole('button', { name: /Null: the same tied records, displayed in the other order/ }).click();
    await queue.getByLabel('Exactly unchanged', { exact: true }).check();
    await queue.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(queue.locator('.imb-verdict'), /Your prediction matches: Exactly unchanged\./);
    const afterRows = await ladderRows();
    // The null has to be non-vacuous: the display must really have changed, while
    // every decision the gate makes must not. Column 1 is the selected-ID list,
    // which follows display order; columns 2 onward are the decision itself.
    assert.equal(afterRows.length, beforeRows.length, 'the ladder has the same operating points');
    beforeRows.forEach((row, index) => {
      assert.deepEqual(afterRows[index].filter((_, column) => column !== 1),
        row.filter((_, column) => column !== 1),
        'reordering the display leaves every count, precision and recall identical');
      assert.deepEqual(afterRows[index][1].split(' ').sort(), row[1].split(' ').sort(),
        'and selects exactly the same identities');
    });
    assert.ok(beforeRows.some((row, index) => afterRows[index][1] !== row[1]),
      'the reordered display really did change the listed order, so this null is not vacuous');
    await screenshot(queue, 'imbalance-queue-tie-null-desktop.png');

    // Practice 2's four records, and the average-precision breakdown.
    await queue.getByRole('button', { name: /Practice 2/ }).click();
    await queue.getByLabel('It falls', { exact: true }).check();
    await queue.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(queue, /1\/3/);
    await checkText(queue, /Average precision/);
    await screenshot(queue, 'imbalance-queue-practice-desktop.png');

    // A queue whose gate selects nothing reports undefined, not zero.
    await queue.getByRole('button', { name: /Above every score/ }).click();
    await queue.getByLabel('It becomes undefined', { exact: true }).check();
    // The numeric guess is filled DELIBERATELY here, against an operating point
    // whose precision is undefined. This is the combination that lets an
    // undefined value reach the verdict as a formatted dash.
    await queue.getByLabel('Optional: precision at the new gate').fill('0.5');
    await queue.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(queue.locator('.imb-verdict'), /Your prediction matches: It becomes undefined\./);
    await checkText(queue.locator('.imb-verdict'), /there is no number to compare it against here/);
    await checkText(queue.locator('.imb-verdict'), /TP \+ FP is zero and precision has no value/);
    await checkText(queue.locator('.imb-verdict'), /not a near miss, and it is not zero/);
    const undefinedVerdict = normalize(await queue.locator('.imb-verdict').innerText());
    assert.ok(!undefinedVerdict.includes('—'),
      'an undefined graded value is named, never rendered as an em dash');
    assert.ok(!/outside \d/.test(undefinedVerdict),
      'and a guess against it is not graded as though a number existed');
    await checkText(queue.locator('.imb-counts'), /precision undefined/);
    await screenshot(queue, 'imbalance-queue-undefined-desktop.png');

    // An edit retires the recorded prediction and hides its feedback.
    await queue.getByRole('button', { name: 'Reset', exact: true }).click();
    await queue.getByLabel('It falls', { exact: true }).check();
    await queue.getByRole('button', { name: 'Apply and check' }).click();
    await queue.getByRole('spinbutton', { name: /^A — score/ }).fill('0.55');
    assert.equal(await queue.locator('input[type="radio"]:checked').count(), 0, 'the edit retired the choice');
    assert.equal(await queue.locator('.imb-verdict').count(), 0, 'and hid the stale feedback');
    await checkText(queue.locator('.imb-pending'), /Draft inputs differ from the applied ones/);
    await queue.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'Score queue: nothing preselected, no bins and no step chart on first paint, the declared counterexample falling from 2/3 to 0 with the no-alert state reported undefined, a corrected label reversing the verdict, the tie null leaving every operating point byte-identical under a reordered display, practice 2\'s 1/3, and an edit that retires both the prediction and its feedback while keeping it as labelled history' });

    // ------------------------------------------------- 3. action costs (I2)
    const cost = page.locator('.imb-investigation').nth(1);
    assert.equal(await cost.locator('input[type="radio"]:checked').count(), 0, 'the cost prediction opens unset');
    assert.equal(await cost.locator('.imb-plot').count(), 0, 'no crossing is drawn before a prediction');
    await cost.getByLabel('Selecting', { exact: true }).check();
    await cost.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(cost.locator('.imb-verdict'), /Your prediction matches: Selecting\./);
    await checkText(cost, /0\.9/);
    await checkText(cost, /1\.2/);
    await checkText(cost, /1\/13/);
    await screenshot(cost, 'imbalance-cost-default-desktop.png');

    await cost.getByRole('button', { name: /Contrast: lower the posterior/ }).click();
    await cost.getByLabel('Skipping', { exact: true }).check();
    await cost.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(cost.locator('.imb-verdict'), /Your prediction matches: Skipping\./);
    await checkText(cost, /0\.95/);
    await screenshot(cost, 'imbalance-cost-contrast-desktop.png');

    // The common-factor null: the action and the cutoff do not move.
    await cost.getByRole('button', { name: /Null: multiply both costs by 3/ }).click();
    await cost.getByLabel('Selecting', { exact: true }).check();
    await cost.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(cost.locator('.imb-verdict'), /Your prediction matches: Selecting\./);
    await checkText(cost, /2\.7/);
    await checkText(cost, /3\.6/);
    await checkText(cost, /1\/13/);
    await screenshot(cost, 'imbalance-cost-null-desktop.png');

    // Both costs zero: no cutoff exists, and the interface says why.
    await cost.getByRole('button', { name: /Corner: both costs zero/ }).click();
    await cost.getByLabel('They are exactly equal', { exact: true }).check();
    await cost.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(cost.locator('.imb-counts'), /cutoff undefined/);
    await checkText(cost, /both costs are zero/);
    await screenshot(cost, 'imbalance-cost-zero-desktop.png');
    await cost.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'Action costs: nothing drawn before a prediction, risks .9 against 1.2 with the 1/13 cutoff, the posterior contrast flipping the action to skipping, the common-factor null tripling both risks while the cutoff and the action stay put, and both-costs-zero reported as having no cutoff at all with its reason named' });

    // --------------------------------------------- 4. the weighted score (I3)
    const weighted = page.locator('.imb-investigation').nth(2);
    assert.equal(await weighted.locator('.imb-plot').count(), 0, 'no loss curve before a prediction');
    await weighted.getByLabel('Exactly one half', { exact: true }).check();
    await weighted.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(weighted.locator('.imb-verdict'), /Your prediction matches: Exactly one half\./);
    await checkText(weighted, /0\.1/);
    assert.equal(await weighted.locator('.imb-plot').count(), 1, 'the loss curve appears after the prediction');
    await screenshot(weighted, 'imbalance-weighted-default-desktop.png');

    // The doubling null: the minimiser holds while the printed loss doubles.
    await weighted.getByRole('button', { name: /Null: double both weights/ }).click();
    await weighted.getByLabel('Exactly one half', { exact: true }).check();
    await weighted.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(weighted.locator('.imb-verdict'), /Your prediction matches: Exactly one half\./);
    const weightedText = normalize(await weighted.innerText());
    // A trailing sentence period must not be swallowed into the captured number.
    const NUMBER = String.raw`(\d+(?:\.\d+)?)`;
    const lossPair = weightedText.match(
      new RegExp(`loss at that optimum ${NUMBER}\. The current loss at its optimum is ${NUMBER}`));
    assert.ok(lossPair, 'the reveal prints the previous and the current loss at the optimum side by side');
    assert.ok(Math.abs(Number(lossPair[2]) / Number(lossPair[1]) - 2) < 1e-6,
      `doubling both weights doubles the printed loss: ${lossPair[1]} to ${lossPair[2]}`);
    await screenshot(weighted, 'imbalance-weighted-doubled-desktop.png');

    // Inverse mode recovers the original probability.
    await weighted.getByRole('button', { name: /Inverse mode/ }).click();
    await weighted.getByLabel('Below one half', { exact: true }).check();
    await weighted.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(weighted.locator('.imb-verdict'), /Your prediction matches: Below one half\./);
    await checkText(weighted, /Recovered population probability/);
    await screenshot(weighted, 'imbalance-weighted-inverse-desktop.png');
    await weighted.getByRole('button', { name: 'Reset', exact: true }).click();
    // An undefined value renders as an em dash, which reads as a formatted
    // number rather than as a missing one: "the formula gives —" passed every
    // text assertion while naming nothing. Nothing revealed may have that shape.
    const emptySlots = await page.evaluate(() => {
      const findings = [];
      for (const lab of document.querySelectorAll('.imb-investigation, .imb-figure')) {
        const text = lab.textContent.replace(/\s+/g, ' ');
        for (const match of text.matchAll(/\b(gives|is|are|equals|of)\s+—/g)) {
          findings.push({ context: text.slice(Math.max(0, match.index - 60), match.index + 20) });
        }
      }
      return findings;
    });
    assert.deepEqual(emptySlots, [],
      'a numeric slot rendered as an em dash, so an undefined value is being printed as if it were a number');
    records.push({ case: 'No revealed numeric slot renders as an em dash: every stated value names a number or an explicit undefined state' });

    records.push({ case: 'Weighted score: no curve before a prediction, the optimum .5 at p = .1 with weights 9 and 1, the doubling null holding the minimiser while the printed loss at the optimum doubles exactly, and inverse mode recovering the original probability' });

    // ------------------------------------------------ 5. SMOTE geometry (I4)
    const smote = page.locator('.imb-investigation').nth(3);
    assert.equal(await smote.locator('.imb-mark.is-generated').count(), 0,
      'the generated point is not drawn before a prediction');
    assert.equal(await smote.locator('.imb-table').count(), 0, 'and neither is the distance table');
    await screenshot(smote, 'imbalance-smote-initial-desktop.png');
    await smote.getByLabel('Yes, it collides with a majority point', { exact: true }).check();
    await smote.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(smote.locator('.imb-verdict'), /Your prediction matches: Yes, it collides/);
    await checkText(smote, /exactly on M/);
    assert.equal(await smote.locator('.imb-mark.is-generated').count(), 1, 'the generated point is drawn after reveal');
    await screenshot(smote, 'imbalance-smote-collision-desktop.png');

    // The exact null: moving only the majority point changes nothing.
    await smote.getByRole('button', { name: /Exact null: move only the majority point/ }).click();
    await smote.getByLabel('No, it stays exactly where it was', { exact: true }).check();
    await smote.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(smote.locator('.imb-verdict'), /Your prediction matches: No, it stays exactly where it was\./);
    const generated = await smote.locator('.imb-mark.is-generated').getAttribute('cx');
    await screenshot(smote, 'imbalance-smote-null-desktop.png');

    // The geometric contrast: a changed divisor changes the nearest neighbour.
    await smote.getByRole('button', { name: /Geometric contrast/ }).click();
    await smote.getByLabel('Yes, a different neighbour is chosen', { exact: true }).check();
    await smote.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(smote.locator('.imb-verdict'), /Your prediction matches: Yes, a different neighbour/);
    await checkText(smote, /1\.25/);
    assert.notEqual(await smote.locator('.imb-mark.is-generated').getAttribute('cx'), generated,
      'the metric contrast really does move the generated point');
    await screenshot(smote, 'imbalance-smote-rescaled-desktop.png');

    // Practice 5's geometry, and the refusal when k exceeds the minority count.
    await smote.getByRole('button', { name: /Practice 5/ }).click();
    await smote.getByLabel('No majority-point collision', { exact: true }).check();
    await smote.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(smote, /2\.5/);
    await screenshot(smote, 'imbalance-smote-practice-desktop.png');
    await smote.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'SMOTE geometry: nothing generated before a prediction, the declared collision on majority point M, the exact majority-move null leaving the generated coordinate byte-identical, the scale-divisor contrast switching the nearest neighbour and moving the point to (0, 1.25), and practice 5\'s (2, 2.5)' });

    // ----------------------------------------------- 6. the tuning queue (I5)
    const tuning = page.locator('.imb-investigation').nth(4);
    await checkText(tuning.locator('.imb-role'), /Role: tuning/);
    assert.equal(await tuning.locator('.imb-table').count(), 0, 'no record table before a prediction');
    await screenshot(tuning, 'imbalance-tuning-initial-desktop.png');
    await tuning.getByRole('button', { name: /Lower the trial gate from \.5 to \.15/ }).click();
    await tuning.getByLabel('It falls', { exact: true }).check();
    await tuning.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(tuning.locator('.imb-verdict'), /Your prediction matches: It falls\./);
    await checkText(tuning, /TP 5, FP 4, FN 2, TN 189/);
    await checkText(tuning, /28/);
    await screenshot(tuning, 'imbalance-tuning-lowered-desktop.png');

    // The labels are hidden until the learner asks for them.
    await checkText(tuning, /hidden/);
    await tuning.getByRole('button', { name: /Show the actual labels in the queue/ }).click();
    await settle(page);
    await checkText(tuning, /actual ME2\?/);
    await screenshot(tuning, 'imbalance-tuning-labels-desktop.png');

    // The cost-scaling null, on the quantity that is actually invariant. Doubling
    // both costs doubles the TOTAL cost, so a preset that grades the total and
    // calls itself a null marks correct reasoning wrong; it grades the selected
    // gate instead, which cannot move when every candidate's cost scales alike.
    await tuning.getByRole('button', { name: /Exact null: double both costs/ }).click();
    await tuning.getByLabel('The selected gate does not move', { exact: true }).check();
    await tuning.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(tuning.locator('.imb-verdict'), /Your prediction matches: The selected gate does not move\./);
    await checkText(tuning, /Scaling both costs by a common factor scales every candidate/);
    await screenshot(tuning, 'imbalance-tuning-gate-null-desktop.png');
    // The same doubling, asked about the total cost, genuinely does rise — and
    // the page says so rather than calling that a null. Reset first, so the
    // doubling is measured against the declared costs rather than against the
    // already-doubled state the previous case left applied.
    await tuning.getByRole('button', { name: 'Reset', exact: true }).click();
    await tuning.getByRole('button', { name: /The same doubling, asked about the total cost instead/ }).click();
    await tuning.getByLabel('It rises', { exact: true }).check();
    await tuning.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(tuning.locator('.imb-verdict'), /Your prediction matches: It rises\./);
    await screenshot(tuning, 'imbalance-tuning-cost-doubles-desktop.png');

    // Exploratory mode changes the claim the locked panel makes.
    await checkText(tuning.locator('.imb-role').first(), /Role: tuning/);
    await tuning.getByRole('button', { name: 'Enter exploratory mode' }).click();
    await tuning.getByLabel('Exactly unchanged', { exact: true }).check();
    await tuning.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(tuning.locator('.imb-role').first(), /Exploratory mode/);
    await checkText(tuning, /exploratory/);
    await screenshot(tuning, 'imbalance-tuning-exploratory-desktop.png');
    // The reserve is never predicted, in either mode.
    const tuningText = normalize(await tuning.innerText());
    assert.ok(tuningText.includes(`${data.roles.reserve.records} reserved proteins receive no prediction`),
      'the reserve is stated as unscored inside the investigation');
    await tuning.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'Tuning queue: a tuning role badge, no records before a prediction, the saved .5 to .15 move detecting five positives instead of one and falling from 72 to 28, labels hidden until asked for, the exact cost-scaling null, and exploratory mode restating the inspection panel as exploratory rather than untouched' });

    // Sequential review: the specific legal edge states found in the source pass.
    for (const id of ['A', 'B', 'C']) await queue.getByLabel(`${id} — actual truth`).selectOption('0');
    await queue.getByLabel('Exactly unchanged', { exact: true }).check();
    await queue.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(queue, /recall is undefined and has no line or markers/);
    assert.equal(await queue.locator('.imb-plot .is-recall, .imb-plot .is-hollow').count(), 0,
      'undefined recall is not painted as zero');
    await screenshot(queue, 'imbalance-sequential-no-positives.png');
    await queue.getByRole('button', { name: 'Reset', exact: true }).click();

    for (const [guess, accepted] of [[.9, true], [.9 + 1e-6, true], [.9 - 1e-6, true],
      [.9 + 1.01e-6, false], [.9 - 1.01e-6, false], [-.3, false]]) {
      await cost.getByRole('button', { name: 'Reset', exact: true }).click();
      await checkText(cost.locator('.imb-numeric-guess'), /Answers within 0.000001 are accepted/);
      await cost.getByLabel('Selecting', { exact: true }).check();
      await cost.getByLabel('Optional: the expected cost of selecting').fill(String(guess));
      await cost.getByRole('button', { name: 'Apply and check' }).click();
      await checkText(cost.locator('.imb-verdict'), accepted
        ? /calculation gives 0\.9, within 0\.000001/ : /calculation gives 0\.9, outside 0\.000001/);
    }
    await cost.getByRole('button', { name: 'Reset', exact: true }).click();

    await weighted.getByRole('spinbutton', { name: 'Population probability p — exact value', exact: true }).fill('.01');
    await weighted.getByRole('spinbutton', { name: 'Positive class weight w₊ — exact value', exact: true }).fill('.1');
    await weighted.getByRole('spinbutton', { name: 'Negative class weight w₋ — exact value', exact: true }).fill('20');
    await weighted.getByLabel('Below one half', { exact: true }).check();
    await weighted.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(weighted, /Inverting returns p = 0\.01/);
    const extremeMarker = Number(await weighted.locator('.imb-plot circle.imb-mark').getAttribute('cx'));
    assert.ok(extremeMarker >= 48 && extremeMarker <= 326, 'extreme optimum remains inside the plotted q axis');
    const minimumVertex = await weighted.locator('.imb-plot svg').evaluate(svg => {
      const marker = svg.querySelector('circle.imb-mark');
      const x = Number(marker.getAttribute('cx')), y = Number(marker.getAttribute('cy'));
      return [...svg.querySelector('polyline.is-loss').points].some(point =>
        Math.abs(point.x - x) <= .011 && Math.abs(point.y - y) <= .011);
    });
    assert.ok(minimumVertex, 'the drawn loss curve includes the exact extreme minimum marked on the plot');
    await screenshot(weighted, 'imbalance-sequential-extreme-weight.png');
    await weighted.getByRole('button', { name: /Inverse mode/ }).click();
    await weighted.getByRole('spinbutton', { name: 'Weighted-loss optimum q* — exact value', exact: true }).fill('.2');
    await weighted.getByLabel('Below one half', { exact: true }).check();
    await weighted.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(weighted.locator('.imb-history'), /answered a different question/);
    const inverseMarker = Number(await weighted.locator('.imb-plot circle.imb-mark').getAttribute('cx'));
    assert.ok(Math.abs(inverseMarker - (48 + .2 * 278)) < .02,
      'inverse-mode loss minimum is the entered q=.2, not stale forward p=.1 with q=.5');
    assert.match(await weighted.locator('.imb-plot svg').getAttribute('aria-label'), /Its minimum is at q = 0\.2/);
    await screenshot(weighted, 'imbalance-sequential-inverse-context.png');
    await weighted.getByRole('button', { name: 'Reset', exact: true }).click();

    await smote.getByRole('spinbutton', { name: 'Interpolation fraction u — exact value', exact: true }).fill('0');
    await smote.getByLabel('No majority-point collision', { exact: true }).check();
    await smote.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(smote, /coincides with minority point\(s\) A, but no majority point/);
    await smote.getByRole('button', { name: 'Reset', exact: true }).click();
    for (const [id, x] of [['A', '.1'], ['B', '-.1'], ['C', '.3']]) {
      await smote.getByRole('spinbutton', { name: `${id} — x`, exact: true }).fill(x);
      await smote.getByRole('spinbutton', { name: `${id} — y`, exact: true }).fill('0');
    }
    await smote.getByLabel('No majority-point collision', { exact: true }).check();
    await smote.getByRole('button', { name: 'Apply and check' }).click();
    assert.equal(await smote.locator('.imb-table tbody tr').first().locator('th, td').first().textContent(), 'B',
      'the decimal-distance tie selects B before C');
    await screenshot(smote, 'imbalance-sequential-smote-tie.png');
    await smote.getByRole('button', { name: 'Reset', exact: true }).click();

    await tuning.getByRole('button', { name: /Lower the trial gate/ }).click();
    await tuning.getByLabel('It falls', { exact: true }).check();
    await tuning.getByRole('button', { name: 'Apply and check' }).click();
    await tuning.getByRole('button', { name: /Show the actual labels in the queue/ }).click();
    await tuning.getByRole('button', { name: 'Reset', exact: true }).click();
    await tuning.getByRole('button', { name: /Lower the trial gate/ }).click();
    await tuning.getByLabel('It falls', { exact: true }).check();
    await tuning.getByRole('button', { name: 'Apply and check' }).click();
    assert.equal(await tuning.getByRole('button', { name: /Show the actual labels in the queue/ }).count(), 1,
      'reset restores the hidden-truth gate');
    await checkText(tuning, /Displayed candidates: near-optimal gates plus a regular sample/);
    await tuning.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'Sequential edge review: no-positive recall omitted from paint; selecting-risk target and six numeric boundary/incorrect-answer cases; extreme weighted-score round trip and in-axis marker; inverse q=.2 curve and mode-specific history; minority endpoint collision; decimal-distance SMOTE ID tie; tuning truth reset and sampled candidate caption' });

    // ------------------------------- 6b. structural guards over the whole page
    //
    // Four checks that test a claim against the thing it claims about, rather
    // than testing a value. Each closes a class an independent review found:
    // a preset whose label promises a null the graded question does not
    // provide; a retired verdict shown while it would be a hint; a figure
    // caption describing a drawing it does not match; and a question that
    // grades a change without one.

    // (i) Every preset whose label says "null" must leave the graded outcome
    //     where it was. The outcome BEFORE the preset is read off the page, not
    //     assumed, so this cannot be satisfied by a coincidence of wording.
    const nullPresets = [
      // `holds: 'same'` — the question is unchanged, so the null must grade to
      // the outcome that held a moment ago. `holds: <label>` — the null also
      // switches to the question it is a null ABOUT, so it must grade to that
      // question's explicit no-change branch.
      { lab: queue, before: /Setup for the tie null/, preset: /Null: the same tied records/, name: 'I1 reorder', holds: 'same' },
      { lab: cost, before: /Declared setup/, preset: /Null: multiply both costs by 3/, name: 'I2 common factor', holds: 'same' },
      { lab: weighted, before: /Declared setup/, preset: /Null: double both weights/, name: 'I3 doubled weights', holds: 'same' },
      { lab: smote, before: /Declared setup/, preset: /Exact null: move only the majority point/, name: 'I4 majority move', holds: 'No, it stays exactly where it was' },
      { lab: tuning, before: /Lower the trial gate/, preset: /Exact null: double both costs/, name: 'I5 selected gate', holds: 'The selected gate does not move' },
    ];
    // No preset labelled "null" may escape the table: a new one added later
    // must be given the state it is a null FROM, not left unchecked.
    const labelledNull = await page.evaluate(() => [...document.querySelectorAll('.imb-investigation')]
      .flatMap(lab => [...lab.querySelectorAll('button')]
        .map(button => button.textContent.replace(/\s+/g, ' ').trim())
        .filter(text => /\bnull\b/i.test(text) && !/^Setup for/i.test(text))));
    assert.equal(labelledNull.length, nullPresets.length,
      `${labelledNull.length} presets are labelled "null" but ${nullPresets.length} are covered: ${labelledNull.join(' | ')}`);
    for (const text of labelledNull) {
      assert.ok(nullPresets.some(entry => entry.preset.test(text)),
        `a preset labelled "null" is not covered by the null guard: ${text}`);
    }
    const outcomeOf = async lab => {
      const text = normalize(await lab.locator('.imb-verdict').innerText());
      // Up to the FIRST period: an option label never contains one, while the
      // explanatory sentence that follows the verdict does.
      const matched = text.match(/Your prediction matches: ([^.]+)\./)
        ?? text.match(/the calculation gives ([^.]+)\./);
      assert.ok(matched, `could not read a graded outcome from: ${text.slice(0, 120)}`);
      return matched[1];
    };
    for (const entry of nullPresets) {
      await entry.lab.getByRole('button', { name: 'Reset', exact: true }).click();
      await entry.lab.getByRole('button', { name: entry.before }).click();
      // Any choice: the outcome is read from the calculation, not the choice.
      await entry.lab.locator('input[type="radio"]').first().check();
      await entry.lab.getByRole('button', { name: 'Apply and check' }).click();
      const before = await outcomeOf(entry.lab);
      await entry.lab.getByRole('button', { name: entry.preset }).click();
      // Record the outcome the null promises: the one that held a moment ago,
      // or the no-change branch of the question the null switches to.
      const expected = entry.holds === 'same' ? before : entry.holds;
      await entry.lab.getByLabel(expected, { exact: true }).check();
      await entry.lab.getByRole('button', { name: 'Apply and check' }).click();
      const verdict = entry.lab.locator('.imb-verdict');
      assert.equal(await verdict.evaluate(node => node.classList.contains('is-miss')), false,
        `${entry.name}: a preset labelled "null" graded the no-change answer as wrong`);
      assert.equal(await outcomeOf(entry.lab), expected,
        `${entry.name}: the graded outcome is not the no-change branch across a preset labelled "null"`);
      // And the comparison the retained history exists for is now on screen.
      // Where the null switches question, the two verdicts are answers to
      // different questions, so only the same-question nulls report no movement.
      if (entry.holds === 'same') {
        await checkText(entry.lab.locator('.imb-history'), /The answer did not move between the two applied states/);
      } else {
        await checkText(entry.lab.locator('.imb-history'), /answered a different question/);
      }
      await entry.lab.getByRole('button', { name: 'Reset', exact: true }).click();
    }
    records.push({ case: `Every one of the ${nullPresets.length} presets whose label says "null" leaves the graded outcome exactly where it was, with the prior outcome read off the page rather than assumed, and each one's retired verdict appears beside the new one reporting that the answer did not move` });

    // (ii) A figure that draws no arrowheads must not describe its own drawing
    //      in terms of arrows, and its visible text must agree with its
    //      aria-label about the reserve.
    const figureClaims = await page.evaluate(() => [...document.querySelectorAll('.imb-figure')].map(figure => {
      const svgs = [...figure.querySelectorAll('svg')];
      const markers = svgs.reduce((total, svg) => total
        + svg.querySelectorAll('marker').length
        + [...svg.querySelectorAll('*')].filter(node =>
          node.getAttribute('marker-end') || node.getAttribute('marker-start')).length, 0);
      return {
        caption: (figure.querySelector('figcaption')?.textContent ?? '').replace(/\s+/g, ' ').trim(),
        visible: figure.textContent.replace(/\s+/g, ' ').trim(),
        aria: svgs.map(svg => svg.getAttribute('aria-label') ?? '').join(' ').replace(/\s+/g, ' ').trim(),
        markers,
      };
    }));
    assert.equal(figureClaims.length, 6, 'six figures were inspected for the claims they make');
    for (const figure of figureClaims) {
      if (figure.markers === 0) {
        assert.ok(!/\barrows?\b/i.test(figure.visible),
          `a figure with no arrowheads speaks of arrows: ${figure.caption.slice(0, 90)}`);
      }
    }
    const pipelineClaim = figureClaims[3];
    assert.equal(pipelineClaim.markers, 0, 'the role diagram draws no arrowheads');
    // Both the visible text and the aria description must make the same claim:
    // nothing leaves the reserve onward, while the split itself does reach it.
    for (const [surface, text] of [['visible', pipelineClaim.visible], ['aria', pipelineClaim.aria]]) {
      assert.ok(/reserve/i.test(text), `the role diagram's ${surface} surface mentions the reserve`);
      assert.ok(!/no arrow reaches the reserve/i.test(text),
        `the role diagram's ${surface} surface still claims no arrow reaches the reserve`);
    }
    assert.ok(/no connector leaves the reserved proteins/i.test(pipelineClaim.visible),
      'the caption states the true property: nothing leaves the reserve onward');
    assert.ok(/no connector onward into any transform or predict step/i.test(pipelineClaim.aria),
      'and the aria description states the same property');
    records.push({ case: 'No figure describes its own drawing in terms of arrowheads it does not draw, and the role diagram makes the same claim about the reserved proteins in its caption, in the diagram and in its aria description' });

    // (iii) A question that grades a CHANGE must require one.
    await smote.getByRole('button', { name: 'Reset', exact: true }).click();
    await smote.getByLabel('Question to answer').selectOption('majorityMove');
    await smote.getByLabel('No, it stays exactly where it was', { exact: true }).check();
    assert.ok(await smote.getByRole('button', { name: 'Apply and check' }).isDisabled(),
      'a before-and-after question cannot be applied with nothing changed');
    await checkText(smote.locator('.imb-note').last(), /needs a move to compare/);
    await smote.getByRole('button', { name: /Exact null: move only the majority point/ }).click();
    // A preset retires the recorded choice, so the radio is set again before
    // asking whether the button is enabled: otherwise this would measure the
    // missing choice rather than the missing change.
    await smote.getByLabel('No, it stays exactly where it was', { exact: true }).check();
    assert.equal(await smote.getByRole('button', { name: 'Apply and check' }).isDisabled(), false,
      'and becomes applicable once something has actually moved');
    await smote.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'Investigation 4’s two before-and-after questions refuse to grade until an input has actually changed, so the degenerate "nothing moved, so nothing changed" path cannot be mistaken for the null' });

    // (iv) Practice 5's preset loads the geometry its own solution describes.
    await smote.getByRole('button', { name: /Practice 5/ }).click();
    const practiceCloud = await smote.evaluate(node => {
      const classes = [...node.querySelectorAll('select')]
        .filter(select => /class/.test(select.closest('label')?.textContent ?? ''))
        .map(select => select.value);
      const k = node.querySelector('input[type="number"][max]');
      return { minority: classes.filter(value => value === 'minority').length, classes };
    });
    assert.equal(practiceCloud.minority, 3,
      'practice 5 loads the three minority points its question and solution both describe');
    const kField = smote.getByRole('spinbutton', { name: /^k — neighbours considered/ });
    assert.equal(await kField.getAttribute('max'), '2',
      'so the fold offers at most two other neighbours, as the solution states');
    await kField.fill('5');
    await checkText(smote.locator('.imb-field-error').first(), /Keep it between 1 and 2/);
    await smote.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'Practice 5’s preset loads three minority points, so the lab offers at most two other neighbours and refuses k = 5 with the bound its solution quotes' });

    // ------------------------------------------- 7. the grading contract's misses
    await queue.getByLabel('It rises', { exact: true }).check();
    await queue.getByLabel('Optional: precision at the new gate').fill('0.9');
    await queue.getByRole('button', { name: 'Apply and check' }).click();
    const miss = queue.locator('.imb-verdict');
    assert.equal(await miss.evaluate(node => node.classList.contains('is-miss')), true,
      'a wrong prediction renders the mismatch branch');
    await checkText(miss, /You recorded It rises; the calculation gives It falls\./);
    await checkText(miss, /You wrote 0\.9 for the new precision; the calculation gives 0, outside/);
    await checkText(queue, /Graded against the committed state/);
    await screenshot(queue, 'imbalance-queue-mismatch-desktop.png');
    await queue.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'The grading contract off its happy path: a wrong direction renders the mismatch branch naming both the recorded and the calculated outcome, the numeric guess is reported outside its tolerance, and the committed state is echoed' });

    // ------------------------------------------------------------- 8. figures
    const caseFlow = page.locator('.imb-figure').nth(0);
    await checkText(caseFlow, /separately normalised|normalised within its own class/i);
    await checkText(caseFlow, /0\.4375/);
    await checkText(caseFlow, /undefined/);
    const prevalence = page.locator('.imb-figure').nth(1);
    await checkText(prevalence, /99\.9/);
    await checkText(prevalence, /0\.446927/);
    await checkText(prevalence, /0\.074143/);
    const stepFigure = page.locator('.imb-figure').nth(2);
    await checkText(stepFigure, /−0\.25/);
    await checkText(stepFigure, /−0\.75/);
    await checkText(stepFigure, /0\.524979/);
    const pipeline = page.locator('.imb-figure').nth(3);
    // The figure draws no arrowheads, so it must not claim a property of arrows.
    await checkText(pipeline, /no connector leaves the reserved proteins into a transform or predict step/i);
    await checkText(pipeline, /The split itself does reach the reserve/i);
    await checkText(pipeline, /462/);
    const outcomes = page.locator('.imb-figure').nth(4);
    await checkText(outcomes, /49/);
    await checkText(outcomes, /84/);
    await checkText(outcomes, /three different winners|different winners/i);
    const lossMass = page.locator('.imb-figure').nth(5);
    await checkText(lossMass, /1053\.6/);
    await checkText(lossMass, /10\.53/);
    await checkText(lossMass, /product-rule|(1 − p)/i);
    for (let index = 0; index < 6; index += 1) {
      await screenshot(page.locator('.imb-figure').nth(index), `imbalance-figure-${index + 1}-desktop.png`);
    }
    records.push({ case: 'Figures carry their required content: the bands say they are separately normalised and the baseline precision is undefined, the two flows keep their fractional expected counts, the weighted step shows both signed gradients and the updated scores, the role diagram says no arrow reaches the reserve, the outcome panel keeps its inconvenient disagreement, and the loss-mass panels name the product-rule term' });

    // ------------------------------------------- 8b. plotted coordinates
    // Every other figure assertion is a regex over rendered text, so a band drawn
    // at the wrong proportion, or a cost bar whose segments were swapped, would
    // pass. Read the geometry back and test it against the model layer.
    const bands = models.classBands(models.fixtures.modelCounts);
    const drawnBands = await caseFlow.locator('svg').first().evaluate(svg => ({
      width: svg.viewBox.baseVal.width,
      rects: [...svg.querySelectorAll('rect.imb-band')].map(rect => ({
        cls: rect.getAttribute('class'),
        x: Number(rect.getAttribute('x')),
        width: Number(rect.getAttribute('width')),
      })),
    }));
    assert.equal(drawnBands.rects.length, 4, 'four band parts are drawn');
    const span = 336 - 4;
    for (const band of bands.bands) {
      for (const part of band.parts) {
        const drawn = drawnBands.rects.find(rect => rect.cls.includes(`is-${part.key}`));
        assert.ok(drawn, `a band part is drawn for ${part.key}`);
        assert.ok(Math.abs(drawn.width - span * part.withinClass) < 0.75,
          `${part.key} is drawn at its within-class share, not its share of the page`);
        assert.ok(Math.abs(drawn.width - span * part.ofPopulation) > 0.75
          || Math.abs(part.withinClass - part.ofPopulation) < 0.01,
          `${part.key} is not drawn at its share of the whole population`);
      }
    }
    // The cost bars: each segment's length is its own contribution to the cost.
    const drawnBars = await outcomes.locator('svg').first().evaluate(svg => ({
      fp: [...svg.querySelectorAll('rect.is-cost-fp')].map(rect => Number(rect.getAttribute('width'))),
      fn: [...svg.querySelectorAll('rect.is-cost-fn')].map(rect => Number(rect.getAttribute('width'))),
      originX: [...svg.querySelectorAll('rect.is-cost-fp')].map(rect => Number(rect.getAttribute('x'))),
    }));
    assert.equal(drawnBars.fp.length, 10, 'two bars per procedure');
    assert.ok(drawnBars.originX.every(x => Math.abs(x - drawnBars.originX[0]) < 1e-9),
      'every cost bar starts at the same zero');
    const worstCost = Math.max(data.study.baselineCost, ...data.methods.flatMap(method => {
      const tuned = models.countsAt(data.inspectionRecords.labels, method.inspectionScores, method.chosenThreshold);
      const half = models.countsAt(data.inspectionRecords.labels, method.inspectionScores, 0.5);
      return [models.costOf(tuned, 1, 12), models.costOf(half, 1, 12)];
    }));
    const barSpan = 340 - 96 - 92;  // the reserved right-hand label column
    data.methods.forEach((method, index) => {
      const half = models.countsAt(data.inspectionRecords.labels, method.inspectionScores, 0.5);
      const tuned = models.countsAt(data.inspectionRecords.labels, method.inspectionScores, method.chosenThreshold);
      [[half, index * 2], [tuned, index * 2 + 1]].forEach(([counts, slot]) => {
        assert.ok(Math.abs(drawnBars.fp[slot] - barSpan * counts.fp / worstCost) < 0.75,
          `${method.name}: the false-alarm segment is its own contribution`);
        assert.ok(Math.abs(drawnBars.fn[slot] - barSpan * 12 * counts.fn / worstCost) < 0.75,
          `${method.name}: the missed-positive segment is twelve times its count`);
      });
    });
    // A presentation attribute loses to the stylesheet, so a shape written with
    // fill="none" over a class that sets a fill is painted solid and covers
    // whatever it was meant to frame. That is invisible to every text and
    // coordinate assertion: the geometry is right and the picture is wrong.
    const paintedOver = await page.evaluate(() => {
      const findings = [];
      for (const svg of document.querySelectorAll('.imb-lesson svg')) {
        for (const shape of svg.querySelectorAll('[fill], [stroke]')) {
          const declared = shape.getAttribute('fill');
          const computed = getComputedStyle(shape).fill;
          if (declared === 'none' && computed !== 'none') {
            findings.push({ tag: shape.tagName, cls: shape.getAttribute('class'), declared, computed });
          }
        }
      }
      return findings;
    });
    assert.deepEqual(paintedOver, [],
      'a fill="none" attribute is overridden by the stylesheet and paints over what it frames');
    // And the bands really are painted: each part has its own opaque colour.
    const bandPaint = await caseFlow.locator('svg').first().evaluate(svg =>
      [...svg.querySelectorAll('rect.imb-band')].map(rect => ({
        cls: rect.getAttribute('class'), fill: getComputedStyle(rect).fill,
      })).concat([...svg.querySelectorAll('rect.imb-lane')].map(rect => ({
        cls: rect.getAttribute('class'), fill: getComputedStyle(rect).fill,
      }))));
    assert.equal(bandPaint.filter(entry => entry.cls.includes('imb-band')).length, 4, 'four painted band parts');
    assert.deepEqual(bandPaint.filter(entry => entry.cls.includes('imb-band') && entry.fill === 'none'), [],
      'every band part is painted');
    assert.equal(new Set(bandPaint.filter(entry => entry.cls.includes('imb-band')).map(entry => entry.fill)).size, 4,
      'and each of the four carries its own distinct colour');
    assert.deepEqual(bandPaint.filter(entry => entry.cls.includes('is-outline') && entry.fill !== 'none'), [],
      'the band outline is unfilled, so it frames the bands instead of covering them');
    // The in-band labels exist and are on top of their own part.
    const bandLabels = await caseFlow.locator('svg').first()
      .evaluateAll(svgs => [...svgs[0].querySelectorAll('text')].map(text => text.textContent.trim()));
    for (const expected of ['detected 14', 'missed 6', 'cleared 962']) {
      assert.ok(bandLabels.some(label => label.includes(expected)), `the band carries its "${expected}" label`);
    }
    records.push({ case: 'No shape in the lesson declares fill="none" and computes to a solid fill, the four class-band parts each paint their own distinct colour, the band outline is unfilled, and every in-band count label is present' });

    records.push({ case: 'Plotted coordinates read back from two figures: each of the four class-band parts is drawn at its within-class share rather than its share of the page, and every cost bar starts at a common zero with its false-alarm and missed-positive segments drawn at their own contributions' });

    // ------------------------------- 9. drawn geometry, labels and type size
    // Commit every investigation so the reveal-gated drawings are on the page.
    await queue.getByLabel('It falls', { exact: true }).check();
    await queue.getByRole('button', { name: 'Apply and check' }).click();
    await cost.getByLabel('Selecting', { exact: true }).check();
    await cost.getByRole('button', { name: 'Apply and check' }).click();
    await weighted.getByLabel('Exactly one half', { exact: true }).check();
    await weighted.getByRole('button', { name: 'Apply and check' }).click();
    await smote.getByLabel('Yes, it collides with a majority point', { exact: true }).check();
    await smote.getByRole('button', { name: 'Apply and check' }).click();
    await tuning.getByLabel('Exactly unchanged', { exact: true }).check();
    await tuning.getByRole('button', { name: 'Apply and check' }).click();
    await settle(page);
    assert.ok(await page.locator('.imb-investigation .imb-plot').count() >= 3,
      'the reveal-gated plots are on the page for the geometry sweep');
    const sweptSvgCounts = {};
    for (const width of [1366, 1024, 768, 390, 320]) {
      await page.setViewportSize({ width, height: 1000 });
      await settle(page);
      const svgCount = await page.locator('.imb-lesson svg').count();
      assert.ok(svgCount >= 12, `the sweep at ${width}px inspected ${svgCount} diagrams, which must not collapse to none`);
      sweptSvgCounts[width] = svgCount;
      const issues = await page.evaluate(inspectLessonVisualLayout, '.imb-lesson');
      assert.deepEqual(issues.flatMap(figure => figure.issues), [], `Figure layout collides at ${width}px`);
      // The shared inspector samples straight <line> elements only, so a flow
      // arc, a risk line or a threshold ladder can run through a label and still
      // report clean. Sample the curve geometry itself against every label box.
      const curveHits = await page.evaluate(sampleCurvesThroughLabels, '.imb-lesson');
      assert.deepEqual(curveHits, [], `A curve runs through a label at ${width}px`);
      // Type that is legible in the stylesheet can still render at 7px if its
      // viewBox is wider than its element.
      const sizes = await page.evaluate(measureSvgTypeSizes, '.imb-lesson');
      assert.ok(sizes.length >= 60, `the type sweep at ${width}px read ${sizes.length} labels`);
      // Two tiers, because the lesson has two: a primary label carries a value
      // or an identity and must stay at 10px or more; a secondary caption
      // repeats something the prose beside it also states, and 8.4px is its
      // floor. Neither tier may fall to the illegible sizes a mis-scoped rule
      // produces, which is what this check exists to catch.
      const tooSmall = sizes.filter(entry => entry.renderedPx < (entry.secondary ? 8.4 : 10));
      assert.deepEqual(tooSmall.slice(0, 6), [], `SVG type renders below its tier's floor at ${width}px`);
      const tooLarge = sizes.filter(entry => entry.renderedPx > 26);
      assert.deepEqual(tooLarge.slice(0, 6), [], `SVG type renders above 26px at ${width}px`);
      typeReadings.push({
        width, labels: sizes.length,
        smallestPrimaryRenderedPx: Math.min(...sizes.filter(entry => !entry.secondary).map(entry => entry.renderedPx)),
        smallestSecondaryRenderedPx: Math.min(...sizes.filter(entry => entry.secondary).map(entry => entry.renderedPx)),
        largestRenderedPx: Math.max(...sizes.map(entry => entry.renderedPx)),
      });
    }
    await page.setViewportSize({ width: 1366, height: 1000 });
    records.push({
      case: 'No label leaves its own SVG, overlaps another label, is crossed by a straight foreground line, or is run through by a curve, at five widths with all five investigations committed; and every SVG label renders between 9 and 26 CSS pixels at every one of those widths',
      sweptSvgCounts,
      curveSamplingNote: 'Curves, flow arcs and threshold ladders are sampled along their own geometry and tested against every label box, because the shared inspector reads straight lines only. A label with an opaque backplate may be crossed by up to 2% of a curve length; a label without one may not be crossed at all.',
      typeSizeNote: 'Rendered size is the computed font-size scaled by element width over viewBox width, which is what a reader actually sees.',
    });

    // ------------------------------------------------------ 10. loading closure
    const localScripts = [...new Set(requests.map(address => new URL(address)).filter(address => address.origin === new URL(base).origin && address.pathname.endsWith('.js')).map(address => address.pathname.slice(1)))];
    assert.deepEqual(localScripts.filter(filename => bodyFiles.has(filename)), [bodyFile],
      'only this lesson body is fetched');
    assert.deepEqual(localScripts.filter(filename => !allowedScripts.has(filename)), []);
    assert.ok(!requests.some(address => address.includes('/outlines/')));
    assert.ok(!requests.some(address => address.includes('yeast.data')),
      'the page never fetches the dataset to render');
    assert.ok(!requests.some(address => address.includes('learn-assets/bias-variance')),
      'and never borrows another lesson\'s asset');
    records.push({
      case: 'Fresh route requests only the selected lesson and its shared closure, and never fetches the dataset or another lesson\'s asset to render',
      requestedScripts: localScripts,
      rawScriptBytes: localScripts.reduce((sum, filename) => sum + fs.statSync(path.join(distDir, filename)).size, 0),
      gzipEstimateBytes: localScripts.reduce((sum, filename) => sum + gzipSync(fs.readFileSync(path.join(distDir, filename))).length, 0),
      interpretation: 'Built requested file sizes; the gzip estimate is not measured network compression or latency.',
    });

    // -------------------------------------------------------- 11. narrow widths
    for (const width of [390, 320]) {
      await page.setViewportSize({ width, height: 844 });
      await page.locator('.imb-lesson details').evaluateAll(items => items.forEach(item => { item.open = true; }));
      await settle(page);
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
        `Document overflows at ${width}px`);
      const mathOverflow = await page.locator('.imb-lesson .katex-display').evaluateAll(items =>
        items.filter(item => item.scrollWidth > item.clientWidth + 1)
          .map(item => ({ text: item.textContent.trim().slice(0, 70), scrollWidth: item.scrollWidth, clientWidth: item.clientWidth })));
      assert.deepEqual(mathOverflow, [], `Overflowing formulas at ${width}px`);
      assert.equal(await page.locator('.katex-error').count(), 0);
      const mathCount = await page.locator('.imb-lesson .katex-display').count();
      assert.ok(mathCount >= 12, `${mathCount} display formulas were measured at ${width}px`);
      // Below the breakpoint a wide numeric table stacks instead of clipping.
      const stacking = await page.locator('.imb-table-scroll tbody tr').first()
        .evaluate(element => ({
          row: getComputedStyle(element).display,
          cell: getComputedStyle(element.querySelector('td, th')).display,
          label: getComputedStyle(element.querySelector('td, th'), '::before').content,
          columns: getComputedStyle(element.querySelector('td, th')).gridTemplateColumns,
        }));
      assert.equal(stacking.row, 'block', `Table rows do not stack at ${width}px`);
      assert.equal(stacking.cell, 'grid', `Stacked cells are not laid out as label and value at ${width}px`);
      assert.ok(stacking.label && stacking.label !== 'none', `Stacked cells carry no column label at ${width}px`);
      assert.ok(stacking.columns.split(' ').length === 2, `Stacked cells do not have a label column at ${width}px`);
      const clipped = await page.locator('.imb-table-scroll').evaluateAll(items =>
        items.filter(item => item.scrollWidth > item.clientWidth + 1).length);
      assert.equal(clipped, 0, `A table still scrolls sideways at ${width}px`);
      // A caption inside a blockified table collapses to its longest word. Ours
      // is a sibling paragraph, so it must be outside the scroll box and wide.
      const captions = await page.locator('.imb-table').evaluateAll(items => items.map(item => {
        const caption = item.querySelector(':scope > .imb-caption');
        const scroll = item.querySelector(':scope > .imb-table-scroll');
        return {
          hasCaption: Boolean(caption),
          insideScrollBox: Boolean(caption && scroll && scroll.contains(caption)),
          nativeCaption: Boolean(item.querySelector('caption')),
          width: caption ? Math.round(caption.getBoundingClientRect().width) : 0,
          parentWidth: Math.round(item.getBoundingClientRect().width),
        };
      }));
      assert.ok(captions.length >= 8, `${captions.length} captioned tables were inspected at ${width}px`);
      assert.deepEqual(captions.filter(entry => !entry.hasCaption), [], `A table has no caption at ${width}px`);
      assert.deepEqual(captions.filter(entry => entry.insideScrollBox), [],
        `A caption sits inside the scroll box at ${width}px`);
      assert.deepEqual(captions.filter(entry => entry.nativeCaption), [],
        `A native <caption> survives inside a blockified table at ${width}px`);
      assert.deepEqual(captions.filter(entry => entry.width < entry.parentWidth * 0.6), [],
        `A caption collapsed to less than 60% of its table width at ${width}px`);
      if (width === 320) {
        await screenshot(page.locator('.imb-investigation').nth(0), 'imbalance-queue-320.png');
        await screenshot(page.locator('.imb-investigation').nth(3), 'imbalance-smote-320.png');
        await screenshot(page.locator('.imb-figure').nth(0), 'imbalance-figure-1-320.png');
        await screenshot(page.locator('.imb-figure').nth(1), 'imbalance-figure-2-320.png');
        await screenshot(page.locator('.imb-figure').nth(4), 'imbalance-figure-5-320.png');
      }
      if (width === 390) {
        for (let index = 0; index < 5; index += 1) {
          await screenshot(page.locator('.imb-investigation').nth(index), `imbalance-investigation-${index + 1}-390.png`);
        }
        for (let index = 0; index < 6; index += 1) {
          await screenshot(page.locator('.imb-figure').nth(index), `imbalance-figure-${index + 1}-390.png`);
        }
      }
      records.push({ case: `Narrow ${width}px layout: no horizontal overflow, ${mathCount} display formulas each inside its own width, every numeric table stacked into labelled rows, and every table caption outside the scroll box at full width` });
    }
    await page.setViewportSize({ width: 1366, height: 1000 });

    // -------------------------------------- 12. keyboard reach and raw code
    // The geometry sweep left every investigation committed, which disables its
    // radios by design. Keyboard reach is measured on the state a learner meets.
    for (const lab of [queue, cost, weighted, smote, tuning]) {
      await lab.getByRole('button', { name: 'Reset', exact: true }).click();
    }
    await settle(page);
    const reach = await page.evaluate(() => {
      const labs = [...document.querySelectorAll('.imb-investigation')];
      return labs.map(lab => {
        const reachable = [...lab.querySelectorAll('button, input, select, textarea')].filter(node => !node.disabled);
        return { controls: reachable.length, radios: reachable.filter(node => node.type === 'radio').length };
      });
    });
    assert.equal(reach.length, 5, 'five investigations were measured for keyboard reach');
    reach.forEach((entry, index) => {
      assert.ok(entry.controls > 8, `investigation ${index + 1} exposes ${entry.controls} enabled focusable controls`);
      assert.ok(entry.radios >= 2, `investigation ${index + 1} exposes its prediction as radios`);
    });
    await queue.getByRole('spinbutton', { name: /^A — score/ }).focus();
    assert.equal(await page.evaluate(() => document.activeElement.getAttribute('type')), 'number',
      'a control takes keyboard focus');
    await page.keyboard.press('Tab');
    assert.ok(await page.evaluate(() => document.activeElement !== document.body),
      'focus moves on rather than being dropped');
    // Every control the prose asks a learner to use must accept the value it names.
    // Controls that feed the model must declare their bounds. The optional
    // prediction guess must NOT: a learner has to be able to record a wrong
    // number, including one outside the range the model would accept.
    const numericFields = await page.evaluate(() => [...document.querySelectorAll('.imb-investigation input[type="number"]')]
      .map(node => ({
        label: (node.closest('label')?.querySelector('span')?.textContent || '').trim().slice(0, 40),
        guess: Boolean(node.closest('.imb-numeric-guess')),
        min: node.min, max: node.max, step: node.step,
      })));
    const controlRanges = numericFields.filter(entry => !entry.guess);
    const guessFields = numericFields.filter(entry => entry.guess);
    assert.ok(controlRanges.length >= 15, `${controlRanges.length} model controls carry an explicit range`);
    assert.deepEqual(controlRanges.filter(entry => entry.min === '' || entry.max === ''), [],
      'every model control declares both bounds');
    assert.equal(guessFields.length, 5, 'one optional numeric guess per investigation');
    assert.deepEqual(guessFields.filter(entry => entry.min !== '' || entry.max !== ''), [],
      'and no prediction guess is bounded, so a wrong answer can be recorded');
    // The code-on-page check normalizes whitespace, so Python indentation would
    // survive being destroyed. Read it raw.
    const rawProgram = await page.locator('.python-example').first().textContent();
    assert.ok(/\n {4}features = np\.asarray/.test(rawProgram),
      'the displayed program keeps the indentation that makes it valid Python');
    records.push({ case: 'Keyboard reach into all five investigations, focus moving on rather than being dropped, every numeric control declaring both bounds, and a displayed program whose Python indentation survives on the page' });

    // -------------------------------------------------------- 13. sequence
    await page.locator('.reader-complete').click();
    assert.equal(new URL(page.url()).pathname.split('/').at(-1), topicId);
    await page.reload({ waitUntil: 'domcontentloaded' });
    await ready(page);
    assert.equal(await page.evaluate(id => JSON.parse(localStorage.getItem('kd-progress'))[id], topicId), true);
    await page.locator('.reader-footer__next').click();
    await page.waitForURL('**/automl-neural-architecture-search-nas?module=classical-ml');
    records.push({ case: 'Completion persists under the stable ID without auto-advance; Next opens the actual successor' });
    assert.deepEqual(errors, [], 'no uncaught page error');
    assert.deepEqual(failedAssets, [], 'no failed asset request');
    await context.close();

    // -------------------------------------------------------- 14. recovery
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

    // -------------------------------------- 15. screenshots are distinguishable
    const digests = screenshotPaths.map(file => ({ file, digest: hash(file), bytes: fs.statSync(file).size }));
    assert.equal(new Set(screenshotPaths).size, screenshotPaths.length, 'every capture has its own path');
    const byDigest = new Map();
    for (const entry of digests) {
      if (byDigest.has(entry.digest)) {
        assert.fail(`Two captures hold byte-identical images: ${byDigest.get(entry.digest)} and ${entry.file}`);
      }
      byDigest.set(entry.digest, entry.file);
    }
    records.push({ case: `${digests.length} screenshots, each with its own path and its own content hash` });

    for (const [filename, expected] of Object.entries(sourceHashes)) assert.equal(hash(filename), expected, `Source changed during check: ${filename}`);
    assert.equal(hash(`${distDir}/.vite/manifest.json`), buildHash);
    const report = {
      startedAt, completedAt: new Date().toISOString(), topicId, status: 'passed',
      browser: await browser.version(), distDir, buildManifestHash: buildHash, sourceHashes,
      moduleTopicCount: module_.topicIds.length,
      dataset: {
        file: data.provenance.file, sha256: data.provenance.sha256, bytes: data.provenance.bytes,
        reserveScored: data.provenance.reserveScored,
      },
      recordedThresholds: { original: original.chosenThreshold, smote: smoteMethod.chosenThreshold },
      typeReadings,
      records,
      screenshots: digests,
      visualInterpretation: 'The intended Space Grotesk font loaded before the checks. Screenshots capture informative states, not defaults only: each investigation before any commitment, each declared contrast, each exact null, the mismatch branch of the grading contract, every inline figure at desktop and at 390 px, and the widest figures and investigations at 320 px. Every capture has a distinct path and a distinct content hash. They require separate visual inspection, which was performed.',
    };
    fs.writeFileSync(evidencePath, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify({ status: report.status, cases: records.length, evidencePath, sourceFiles: ownedFiles.length, screenshots: screenshotPaths.length }));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });

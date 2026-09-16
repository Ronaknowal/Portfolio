// Production browser review of the cross-validation lesson: visible content, the
// four investigations and their predict/commit/apply contract, the contrast and
// null cases the specification names, narrow layouts, formula overflow,
// sequence, completion and load-failure recovery.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { gzipSync } = require('node:zlib');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const { inspectLessonVisualLayout } = require('./lib/lesson-visual-layout.cjs');

const distDir = process.env.DIST_DIR || 'dist';
const base = (process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4184').replace(/\/+$/, '');
const topicId = 'cross-validation-hyperparameter-tuning';
const sourcePath = `src/learn/data/topics/${topicId}.jsx`;
const route = `${base}/learn/path/full-curriculum/${topicId}?module=classical-ml`;
const evidencePath = 'docs/teaching/evidence/validation-browser.json';
const read = filename => JSON.parse(fs.readFileSync(filename, 'utf8'));
const hash = filename => createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
const normalize = text => text.replace(/\s+/g, ' ').trim();
const ownedFiles = [
  sourcePath,
  'src/learn/data/validation-models.js',
  'src/learn/data/validation-data.js',
  'src/learn/data/validation-examples.js',
  'src/learn/components/lesson-labs/ValidationShared.jsx',
  'src/learn/components/lesson-labs/ValidationLabs.jsx',
  'src/learn/components/lesson-labs/ValidationFigures.jsx',
  'src/learn/components/lesson-labs/validation-labs.css',
  `src/learn/data/curriculum/blueprints/${topicId}.js`,
  'public/learn-assets/cross-validation/penguins.csv',
  'public/learn-assets/cross-validation/data-provenance.md',
  'public/learn-assets/cross-validation/nested-experiment.json',
];

(async () => {
  const startedAt = new Date().toISOString();
  const sourceHashes = Object.fromEntries(ownedFiles.map(filename => [filename, hash(filename)]));
  const build = read(`${distDir}/.vite/manifest.json`);
  const buildHash = hash(`${distDir}/.vite/manifest.json`);
  const publications = read('src/learn/data/lesson-manifest.json');
  const { tracks } = await import('../src/learn/data/tracks.js');
  const { validationExamples } = await import('../src/learn/data/validation-examples.js');
  const { NESTED_EXPERIMENT } = await import('../src/learn/data/validation-data.js');
  const module = tracks.find(track => track.id === 'classical-ml');
  assert.equal(module.topicIds[20], topicId);
  assert.equal(module.topicIds.length, 39);
  const bodyFile = build[sourcePath].file;
  const bodyFiles = new Set(Object.values(publications).map(value => build[`src/learn/data/${value.replace(/^\.\//, '')}`].file));

  const records = [];
  const screenshots = [];
  const layouts = [];
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const checkText = async (locator, pattern) => assert.match(normalize(await locator.innerText()), pattern);
  const settle = page => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));

  try {
    const context = await browser.newContext({ viewport: { width: 1366, height: 1000 } });
    const page = await context.newPage();
    const requests = [], errors = [], failedAssets = [];
    page.on('request', request => requests.push(request.url()));
    page.on('pageerror', error => errors.push(error.message));
    page.on('requestfailed', request => failedAssets.push(request.url()));
    const ready = async target => {
      await target.locator('.cv-lesson').waitFor();
      await target.waitForFunction(() => [...document.fonts].some(font => font.family.includes('Space Grotesk') && font.status === 'loaded'), null, { timeout: 20000 });
      await target.evaluate(() => document.fonts.ready);
      await settle(target);
    };
    const lab = kind => page.locator(`[data-cv-lab="${kind}"]`);
    const button = (target, name) => target.getByRole('button', { name, exact: true });
    const capture = async (locator, filename) => {
      const destination = path.join('docs/teaching/evidence/screenshots', filename);
      fs.mkdirSync(path.dirname(destination), { recursive: true });
      const viewport = page.viewportSize();
      const bounds = await locator.boundingBox();
      if (bounds && bounds.height > viewport.height - 160) {
        await page.setViewportSize({ width: viewport.width, height: Math.min(4000, Math.ceil(bounds.height) + 180) });
      }
      await locator.scrollIntoViewIfNeeded();
      await settle(page);
      await locator.screenshot({ path: destination });
      await page.setViewportSize(viewport);
      screenshots.push(destination);
    };
    /** Predict, commit, apply — the contract every investigation in this lesson keeps. */
    const predictApply = async (target, answers, applyLabel) => {
      for (let index = 0; index < answers.length; index += 1) {
        await target.locator('.cv-prediction fieldset').nth(index)
          .getByRole('radio', { name: answers[index], exact: true }).check();
      }
      await button(target, 'Commit prediction').click();
      await button(target, applyLabel).click();
    };

    await page.goto(route, { waitUntil: 'domcontentloaded' });
    await ready(page);
    await checkText(page.locator('.reader-header h1'), /^Cross-Validation & Hyperparameter Tuning$/);
    await checkText(page.locator('.reader-header__meta'), /21 of 39 topics on this route/);
    await checkText(page.locator('.reader-footer__previous'), /Feature Scaling, Encoding & Imputation/);
    await checkText(page.locator('.reader-footer__next'), /Regularization/);
    assert.equal(await page.locator('.cv-lesson > h2').count(), 11);
    assert.equal(await page.locator('[data-cv-lab]').count(), 4);
    assert.equal(await page.locator('[data-cv-figure]').count(), 7);
    assert.equal(await page.locator('.cv-practice').count(), 10);
    assert.equal(await page.locator('.lesson-check').count(), 0, 'no checkpoint duplicates a practice task');
    assert.equal(await page.locator('.python-example').count(), 3);
    assert.equal(await page.locator('.cv-practice details[open]').count(), 0, 'every hint and solution starts closed');
    assert.equal(await page.locator('.katex-error').count(), 0);
    const rendered = normalize(await page.locator('.cv-lesson').textContent());
    for (const [key, example] of Object.entries(validationExamples)) {
      assert.ok(rendered.includes(normalize(example.code)), `Missing complete displayed code for ${key}`);
      assert.ok(rendered.includes(normalize(example.expected)), `Missing executed output for ${key}`);
    }
    for (const anchor of await page.locator('.lesson-intro a[href^="#"]').evaluateAll(items => items.map(item => item.getAttribute('href')))) {
      assert.equal(await page.locator(`[id="${anchor.slice(1)}"]`).count(), 1, `Broken lesson anchor ${anchor}`);
    }
    // Every claim the manuscript prints must be on the page.
    for (const phrase of [
      '0.42857142857142855', '0.9883549', '0.9883721', '0.9913043', '340/344',
      '0.6875', 'duplicate of an existing rule', 'no universal fixed percentage of optimism',
      'A splitter name is not a substitute', 'not a universal statistical correction',
      'not a guarantee about an oracle-best setting', 'Do not vote among outer-fold settings',
      'not a theorem that sixty trials reach within 5%', 'does not make random search universally dominate',
      'not newly independent validation', 'not automatically a valid standard error',
      'no universal trial count', 'monotonic improvement alone did not make the ranking safe',
      'remains experimental', 'no universal threshold', 'keep that failure visible',
      'not executed during the content phase', 'CC0',
      // Added after the independent review.
      'contains', 'Twenty trials of this sequential sampler did not reach a strictly better point',
      'settled by the order the candidates', 'split record',
      'pins for reproducibility', 'standard error',
    ]) assert.ok(rendered.includes(phrase), `Missing manuscript claim: ${phrase}`);
    assert.ok(rendered.includes('3/StandardScaler and 3/RobustScaler'), 'section 6 names the final two-way tie');
    assert.equal(await page.locator('a[href="/learn-assets/cross-validation/nested-experiment.json"]').count(), 1,
      'the served split record is linked from the page');
    assert.ok(rendered.includes('0.9539302') && rendered.includes('0.4528434'), 'both hit probabilities are printed');
    assert.ok(rendered.includes('149 draws suffice'), 'practice 7 prints the computed draw count');
    const served = await page.request.get(`${base}/learn-assets/cross-validation/penguins.csv`);
    assert.equal(served.status(), 200);
    const csvText = await served.text();
    assert.equal(csvText.trim().split('\n').length, 345, 'the CSV serves a header and 344 rows');
    assert.ok(csvText.startsWith('species,island,bill_length_mm'));
    for (const asset of ['data-provenance.md', 'nested-experiment.json']) {  // both are linked from the page
      const response = await page.request.get(`${base}/learn-assets/cross-validation/${asset}`);
      assert.equal(response.status(), 200, `${asset} is served`);
    }
    records.push({ case: 'Eleven sections, seven figures, four investigations, three complete executed programs, ten closed practice tasks, every manuscript claim, the served CSV and provenance, and the module sequence' });

    // Every investigation starts unset and cannot compute before a prediction.
    for (const kind of ['folds', 'selection', 'nested', 'halving']) {
      assert.equal(await lab(kind).locator('input[type="radio"]:checked').count(), 0, `${kind}: nothing preselected`);
      assert.ok(await button(lab(kind), 'Commit prediction').isDisabled(), `${kind}: commit waits for an answer`);
      assert.equal(await lab(kind).locator('[data-cv-result]').count(), 0, `${kind}: no answer before a prediction`);
      assert.equal(await lab(kind).locator('[data-cv-status]').count(), 1, `${kind}: says nothing is computed yet`);
    }
    records.push({ case: 'All four investigations open unset: no preselected prediction, no computed answer, and commit disabled until a choice exists' });

    // ---- I1: fold building -------------------------------------------
    const folds = lab('folds');
    await predictApply(folds, ['label 1'], 'Apply the fold plan');
    await checkText(folds.locator('.cv-verdict'), /Predicted label: your prediction matches — label 1\./);
    await checkText(folds.locator('[data-cv-mechanism]'), /nearest eligible training row is 3/);
    await checkText(folds, /0\.4285714286/);
    await checkText(folds, /Unweighted fold mean 0\.5/);
    // The reason printed beside the two summaries must be the actual reason.
    await checkText(folds, /They differ here because the folds are unequal in size/);
    await capture(folds, 'validation-folds-desktop.png');
    // Contrast: relabelling training row 3 changes the held-out prediction.
    await button(folds, 'Contrast: training row 3 relabelled').click();
    assert.equal(await folds.locator('[data-cv-result]').count(), 0, 'a new setup retires the previous answer');
    await predictApply(folds, ['label 0'], 'Apply the fold plan');
    await checkText(folds.locator('.cv-verdict'), /matches — label 0/);
    // Null: the held-out row's own label cannot reach its own fit.
    await button(folds, 'Null: row 0’s own label changed').click();
    await predictApply(folds, ['label 1'], 'Apply the fold plan');
    await checkText(folds.locator('.cv-verdict'), /matches — label 1/);
    await checkText(folds.locator('[data-cv-mechanism]'), /the true label 1 decides only whether it counts as correct/);
    // Null: translating every feature leaves the whole table alone.
    await button(folds, 'Null: every x shifted by \\+10').click().catch(async () => {
      await folds.getByRole('button', { name: /every x shifted/ }).click();
    });
    await predictApply(folds, ['label 1'], 'Apply the fold plan');
    await checkText(folds, /Unweighted fold mean 0\.5/);
    await checkText(folds, /0\.4285714286/);
    // An edit retires the recorded answer and keeps it as a labelled comparison.
    await button(folds, 'Baseline: three consecutive folds').click();
    await predictApply(folds, ['label 1'], 'Apply the fold plan');
    assert.equal(await folds.locator('[data-cv-previous]').count(), 0, 'no comparison before a second trial exists');
    await folds.getByLabel('label y of row 3', { exact: true }).selectOption('0');
    await checkText(folds.locator('[data-cv-previous]'), /Previous trial, kept as a labelled comparison\. Predicted label: label 1\./);
    await predictApply(folds, ['label 0'], 'Apply the fold plan');
    await checkText(folds.locator('[data-cv-previous]'), /Predicted label: label 1\./);
    await checkText(folds.locator('.cv-verdict'), /matches — label 0/);
    await checkText(folds, /They agree here despite the unequal fold sizes, because every fold reaches the same accuracy/);
    await capture(folds, 'validation-folds-previous-trial.png');
    await button(folds, 'Reset').click();
    assert.equal(await folds.locator('[data-cv-previous]').count(), 0, 'Reset clears the comparison');
    await folds.getByRole('radio', { name: 'label 1', exact: true }).first().check();
    await button(folds, 'Commit prediction').click();
    await checkText(folds.locator('[data-cv-committed]'), /Prediction recorded against these inputs/);
    await folds.getByLabel('feature x of row 3', { exact: true }).fill('9');
    assert.equal(await folds.locator('[data-cv-committed]').count(), 0, 'editing an input retires the commitment');
    assert.equal(await folds.locator('input[type="radio"]:checked').count(), 0, 'and clears the recorded choice');
    // An invalid fold assignment is refused rather than silently repaired.
    await button(folds, 'Reset').click();
    for (const row of [0, 1, 2]) await folds.getByLabel(`validation fold of row ${row}`, { exact: true }).selectOption('1');
    await checkText(folds.locator('[data-cv-error]'), /Fold 1 is empty/);
    assert.ok(await button(folds, 'Commit prediction').isDisabled(), 'no prediction can be committed against an invalid plan');
    await button(folds, 'Reset').click();
    records.push({ case: 'I1 baseline trace, training-label contrast, own-label null, translation null, arbitrary edit retires the prediction, empty fold refused' });

    // ---- I2: selection on a criterion with no signal -----------------
    const selection = lab('selection');
    await predictApply(selection, ['0.75 (3 of 4)'], 'Apply and score the rules');
    await checkText(selection.locator('.cv-verdict'), /matches — 0\.75/);
    await checkText(selection, /Average selected validation accuracy 0\.6875/);
    await checkText(selection, /Expected accuracy on fresh fair labels 0\.5/);
    // The winning rule is marked by a word, not by a background colour alone.
    await checkText(selection.locator('[data-cv-table="candidate-scores"]'), /candidate B · selected/);
    // The manuscript's table: 2 patterns with best 4, 8 with 3, 6 with 2.
    await checkText(selection.locator('[data-cv-table="enumeration"]'), /0 1 1\/16 4 1 4 4\/16 3 2 6 6\/16 2 3 4 4\/16 3 4 1 1\/16 4/);
    await capture(selection, 'validation-selection-desktop.png');
    // Null: a duplicate candidate raises nothing.
    await button(selection, 'Null: add a duplicate rule').click();
    await predictApply(selection, ['0.75 (3 of 4)'], 'Apply and score the rules');
    await checkText(selection, /Average selected validation accuracy 0\.6875/);
    // Sixteen rules reach 1 on every possible validation set, and still predict half.
    await button(selection, 'All sixteen prediction patterns').click();
    await predictApply(selection, ['1 (4 of 4)'], 'Apply and score the rules');
    await checkText(selection, /Average selected validation accuracy 1/);
    await checkText(selection, /Expected accuracy on fresh fair labels 0\.5/);
    // Practice 5's fixture, before and after adding the matching rule.
    await button(selection, 'Fixture: labels 0,1,0,1').click();
    await predictApply(selection, ['0.5 (2 of 4)'], 'Apply and score the rules');
    await checkText(selection.locator('.cv-verdict'), /matches — 0\.5/);
    await button(selection, 'Fixture plus a matching rule').click();
    await predictApply(selection, ['1 (4 of 4)'], 'Apply and score the rules');
    await checkText(selection.locator('.cv-verdict'), /matches — 1 \(4 of 4\)/);
    await checkText(selection.locator('[data-cv-mechanism]'), /expected accuracy on fresh fair labels is 0\.5/);
    await button(selection, 'Reset').click();
    records.push({ case: 'I2 0.6875 enumeration, duplicate-candidate null, sixteen-pattern saturation at 1 with future 0.5, and the 0,1,0,1 fixture before and after a matching rule' });

    // ---- I3: one nested fold -----------------------------------------
    const nested = lab('nested');
    await predictApply(nested, ['1 neighbour'], 'Apply and run every fit');
    await checkText(nested.locator('.cv-verdict'), /Selected neighbour count: your prediction matches — 1 neighbour/);
    await checkText(nested.locator('[data-cv-mechanism]'), /They tie, so the declared rule takes the smaller count: 1/);
    await checkText(nested, /Selected neighbour count 1 \(tie, smaller count\)/);
    await checkText(nested, /Correct on 7 of 8/);
    await capture(nested, 'validation-nested-desktop.png');
    // Contrast: an outer-training label can change the selection.
    await button(nested, 'Contrast: row 3 relabelled').click();
    await predictApply(nested, ['3 neighbours'], 'Apply and run every fit');
    await checkText(nested.locator('.cv-verdict'), /matches — 3 neighbours/);
    await checkText(nested, /k = 1 inner mean 0\.5/);
    await checkText(nested, /k = 3 inner mean 0\.625/);
    await checkText(nested, /Correct on 7 of 8/, 'the protected outcome is unchanged by the selection change');
    // Null: a protected label cannot touch the fitting path.
    await button(nested, 'Null: row 2 relabelled').click();
    await predictApply(nested, ['1 neighbour'], 'Apply and run every fit');
    await checkText(nested, /k = 1 inner mean 0\.875/);
    await checkText(nested, /Correct on 6 of 8/, 'only the correctness moves');
    // Role swap: the same row trains one fold and is protected in the other.
    await button(nested, 'Role swap: row 7 relabelled').click();
    await predictApply(nested, ['label 1'], 'Apply and run every fit');
    await checkText(nested.locator('.cv-verdict'), /matches — label 1/);
    await checkText(nested, /Correct on 8 of 8/);
    await nested.getByLabel('Outer fold to open', { exact: true }).selectOption('1');
    assert.equal(await nested.locator('[data-cv-result]').count(), 0, 'changing the inspected fold retires the answer');
    // In the other outer fold row 7 is protected, so the fitting path is untouched
    // and only its own correctness moves: 8 of 8 becomes 7 of 8.
    await predictApply(nested, ['label 0'], 'Apply and run every fit');
    await checkText(nested.locator('.cv-verdict'), /matches — label 0/);
    await checkText(nested, /k = 1 inner mean 0\.875/);
    await checkText(nested, /Correct on 7 of 8/);
    await button(nested, 'Reset').click();
    records.push({ case: 'I3 tie-resolved baseline 7/8 and 8/8, outer-training-label selection contrast, protected-label null, and the role swap where one relabelled row changes the fit in one fold and only correctness in the other' });

    // ---- I4: resource allocation -------------------------------------
    const halving = lab('halving');
    assert.equal(await halving.locator('[data-cv-hindsight]').count(), 0, 'no hindsight before a run');
    await predictApply(halving, ['candidate A', 'no'], 'Run the schedule');
    await checkText(halving.locator('.cv-verdict').first(), /Survivor: your prediction matches — candidate A/);
    await checkText(halving.locator('.cv-verdict').nth(1), /Matches the full-budget best: your prediction matches — no/);
    await checkText(halving, /Nominal resource, retraining from scratch 150 units/);
    assert.equal(await halving.locator('[data-cv-hindsight]').count(), 0, 'hindsight stays closed until asked for');
    const beforeHindsight = normalize(await halving.textContent());
    assert.ok(!beforeHindsight.includes('Hindsight regret'), 'the regret is not in the DOM before the reveal');
    // The weaker fact is that the block is absent; the one that matters is that
    // no surface names the full-budget winner or its loss before the control.
    const mechanism = normalize(await halving.locator('[data-cv-mechanism]').innerText());
    assert.doesNotMatch(mechanism, /would have finished lower|candidate B at|Candidate B would/,
      'the post-Apply caption does not name the hindsight winner');
    assert.match(mechanism, /Reveal hindsight names it/, 'it points at the control instead');
    await button(halving, 'Reveal hindsight').click();
    await checkText(halving.locator('[data-cv-hindsight]'), /candidate B at 0\.1/);
    await checkText(halving.locator('[data-cv-hindsight]'), /Hindsight regret 0\.14/);
    await capture(halving, 'validation-halving-desktop.png');
    // Deciding one stage later reaches the eventual winner.
    await button(halving, 'Decide at budget 30 instead').click();
    await predictApply(halving, ['candidate B', 'yes'], 'Run the schedule');
    await checkText(halving.locator('.cv-verdict').first(), /matches — candidate B/);
    await checkText(halving, /Nominal resource, retraining from scratch 180 units/);
    // Null: a value nobody paid for cannot change the decision.
    await button(halving, 'Null: change a value nobody paid for').click();
    await predictApply(halving, ['candidate A', 'no'], 'Run the schedule');
    await checkText(halving.locator('.cv-verdict').first(), /matches — candidate A/);
    await button(halving, 'Reveal hindsight').click();
    await checkText(halving.locator('[data-cv-hindsight]'), /Hindsight regret 0\.22/);
    // Contrast: a value it did pay for changes the survivor.
    await button(halving, 'Contrast: change a value it did pay for').click();
    await predictApply(halving, ['candidate B', 'yes'], 'Run the schedule');
    await checkText(halving.locator('.cv-verdict').first(), /matches — candidate B/);
    // The nine-candidate resource ledger from the manuscript.
    await button(halving, 'Nine candidates, factor 3').click();
    await predictApply(halving, ['candidate I', 'yes'], 'Run the schedule');
    await checkText(halving, /Nominal resource, retraining from scratch 270 units/);
    await checkText(halving, /Giving every candidate the largest budget 810 units/);
    await checkText(halving, /Incremental resource, if training genuinely resumes 210 units/);
    await button(halving, 'Reset').click();
    records.push({ case: 'I4 early-ranking baseline with its 150-unit ledger, later-start contrast, unobserved-value null, observed-value contrast, and the nine-candidate 270/810/210 ledger; before the Reveal control neither the hindsight block nor the post-Apply caption names the full-budget winner or its loss' });

    // ---- Sequential topic-21 review ---------------------------------
    // Retained history must keep the original output, even when a new question
    // happens to reuse the same answer key and a different meaning for "1".
    await predictApply(nested, ['1 neighbour'], 'Apply and run every fit');
    await nested.getByLabel('What to predict', { exact: true }).selectOption('row');
    await checkText(nested.locator('[data-cv-previous]'), /Selected neighbour count: 1 neighbour/);
    assert.doesNotMatch(await nested.locator('[data-cv-previous]').innerText(), /Prediction for row 8: label 1/);
    await predictApply(nested, ['label 0'], 'Apply and run every fit');
    await nested.getByLabel('Protected row to predict', { exact: true }).selectOption('0');
    await checkText(nested.locator('[data-cv-previous]'), /Prediction for row 8: label 0/);
    assert.doesNotMatch(await nested.locator('[data-cv-previous]').innerText(), /Prediction for row 0/);
    await capture(nested.locator('[data-cv-prediction]'), 'validation-previous-output-identity.png');
    await button(nested, 'Reset').click();
    assert.equal(await nested.locator('[data-cv-previous]').count(), 0);

    // Exact two-decimal symmetry must not be broken by binary subtraction dust.
    await nested.getByText('Edit the sixteen rows', { exact: true }).click();
    for (const [id, value] of [[0, '0.3'], [1, '0.5'], [3, '0.1']]) {
      await nested.getByLabel(`feature x of row ${id}`, { exact: true }).fill(value);
    }
    await nested.getByLabel('label y of row 3', { exact: true }).selectOption('1');
    await nested.getByLabel('What to predict', { exact: true }).selectOption('row');
    await nested.getByLabel('Protected row to predict', { exact: true }).selectOption('0');
    await predictApply(nested, ['label 0'], 'Apply and run every fit');
    await checkText(nested.locator('.cv-verdict'), /matches — label 0/);
    const rowZero = nested.locator('[data-cv-table="protected"] tbody tr').first();
    assert.deepEqual(await rowZero.locator('th,td').allTextContents(), ['0', '1', '0', '0', 'correct']);
    await capture(nested.locator('[data-cv-prediction]'), 'validation-decimal-distance-tie.png');
    await button(nested, 'Reset').click();

    for (const [candidate, losses] of [['A', [.4, .2, .1]], ['B', [.3, .2, .1]], ['C', [.5, .3, .2]]]) {
      for (let stage = 0; stage < 3; stage += 1) {
        await halving.getByLabel(`candidate ${candidate} loss at budget ${[10, 30, 90][stage]}`, { exact: true }).fill(String(losses[stage]));
      }
    }
    await predictApply(halving, ['candidate B', 'yes'], 'Run the schedule');
    assert.equal(await halving.locator('.cv-verdict.is-miss').count(), 0, 'a tied minimum attains the lowest loss');
    await checkText(halving.locator('[data-cv-mechanism]'), /possibly tied with another candidate/);
    await button(halving, 'Reveal hindsight').click();
    await checkText(halving.locator('[data-cv-hindsight]'), /Hindsight regret 0/);
    await capture(halving.locator('[data-cv-prediction]'), 'validation-halving-tied-minimum.png');
    await button(halving, 'Nine candidates, factor 3').click();
    await halving.getByLabel('Survival factor', { exact: true }).selectOption('2');
    assert.doesNotMatch(await halving.innerText(), /stage counts 9, 3, 1/, 'editing the factor retires the preset explanation');
    for (const value of await halving.locator('input[type="number"]').evaluateAll(inputs => inputs.map(input => input.value))) {
      assert.ok(!/\.\d{3,}/.test(value), 'declared two-decimal preset values show no binary tails');
    }
    await checkText(halving.locator('[data-cv-prediction]'), /selected among those assessed at the largest budget/);
    await predictApply(halving, ['candidate I', 'yes'], 'Run the schedule');
    await checkText(halving.locator('[data-cv-mechanism]'), /3 candidate\(s\) reached budget 90/);
    const finalists = halving.locator('[data-cv-table="stage-2"] tbody tr');
    assert.equal(await finalists.count(), 3);
    assert.deepEqual(await finalists.locator('td:last-child').allTextContents(), ['selected', 'assessed at final budget', 'assessed at final budget']);
    assert.match(await halving.locator('svg').getAttribute('aria-label'), /3 candidates reach budget 90/);
    assert.equal(await halving.locator('svg polyline').evaluateAll(lines => lines.filter(line => line.points.getItem(line.points.numberOfItems - 1).x === 330).length), 3, 'all three finalists paint a point at budget 90');
    assert.equal(await halving.locator('svg text').evaluateAll(labels => labels.filter(label => /^[A-I]$/.test(label.textContent.trim())).length), 0, 'candidate identity is in the endpoint table rather than overlapping curves');
    const endpointRows = await halving.locator('[data-cv-table="schedule-endpoints"] tbody tr').evaluateAll(rows => rows.map(row => [...row.querySelectorAll('th,td')].map(cell => cell.textContent)));
    assert.equal(endpointRows.length, 9, 'every trajectory retains an accessible named endpoint');
    assert.equal(endpointRows.filter(row => row[1] === '90').length, 3, 'the table identifies every final-budget trajectory');
    await capture(halving, 'validation-halving-multiple-finalists.png');
    await button(halving, 'Reset').click();

    // All16 pattern rows retain multiplicity, so their mean is independently
    // reconstructible even for an asymmetric edited pair of rules.
    await selection.getByRole('button', { name: 'candidate B, case 1, prediction 1', exact: true }).click();
    await selection.getByRole('button', { name: 'candidate B, case 2, prediction 1', exact: true }).click();
    await selection.getByRole('button', { name: 'candidate B, case 3, prediction 1', exact: true }).click();
    await predictApply(selection, ['0.25 (1 of 4)'], 'Apply and score the rules');
    const enumerationDetails = selection.locator('details');
    assert.equal(await enumerationDetails.count(), 1);
    assert.equal(await enumerationDetails.getAttribute('open'), null, 'full enumeration begins closed');
    await enumerationDetails.locator('summary').click();
    const enumerated = await enumerationDetails.locator('tbody tr').evaluateAll(rows => rows.map(row => [...row.querySelectorAll('th,td')].map(cell => cell.textContent)));
    assert.equal(enumerated.length, 16);
    let correctSum = 0;
    for (const row of enumerated) {
      const labels = row[0].split(' ').map(Number);
      const counts = [[0, 0, 0, 0], [0, 0, 0, 1]].map(pattern => pattern.filter((bit, index) => bit === labels[index]).length);
      const best = Math.max(...counts);
      assert.equal(Number(row[2]), best);
      assert.equal(row[1], counts[0] >= counts[1] ? 'candidate A' : 'candidate B');
      assert.equal(row[4], '1/16');
      correctSum += best;
    }
    await checkText(selection, new RegExp(`Average selected validation accuracy ${correctSum / 64}`));
    await button(selection, 'Reset').click();

    const risk = page.locator('[data-cv-figure="risk"]');
    for (const [selector, count, expected] of [
      ['rect.is-variance', 5, 'rgb(36, 48, 41)'], ['rect.is-covariance', 20, 'rgb(20, 26, 24)'],
      ['text.is-variance', 5, 'rgb(231, 185, 74)'], ['text.is-covariance', 20, 'rgb(125, 139, 133)'],
    ]) {
      const nodes = risk.locator(selector);
      assert.equal(await nodes.count(), count);
      assert.deepEqual(await nodes.evaluateAll(items => [...new Set(items.map(item => getComputedStyle(item).fill))]), [expected], 'covariance emphasis survives CSS');
    }
    records.push({ case: 'Sequential review: original question/output identity survives history edits; decimal symmetric distances use source-ID ties; tied minimum losses grade correctly; all three factor-2 finalists remain visible and distinct from the selected winner; all sixteen label patterns expose the exact mean; computed covariance fills retain diagonal and off-diagonal emphasis' });

    // ---- Read-only real evidence -------------------------------------
    const explorer = page.locator('[data-cv-figure="real-experiment"]');
    for (const fold of NESTED_EXPERIMENT.folds) {
      await button(explorer, `Outer fold ${fold.fold + 1}`).click();
      await settle(page);
      const text = normalize(await explorer.textContent());
      for (const candidate of fold.candidates) {
        assert.ok(text.includes(candidate.meanScore.toFixed(6)), `fold ${fold.fold + 1}: candidate mean ${candidate.meanScore.toFixed(6)} is visible`);
      }
      assert.ok(text.includes(`${fold.correct} / ${fold.testRows.length}`), 'the protected result is visible');
      assert.ok(text.includes(`${fold.baselineCorrect} / ${fold.testRows.length}`), 'the majority baseline is visible');
      assert.ok(text.includes('Selection score (chose the setting)'), 'the two scores are separately labelled');
      assert.equal(await explorer.locator('input, select').count(), 0, 'the real evidence has no editable control');
    }
    await capture(explorer, 'validation-real-experiment-desktop.png');
    await button(explorer, 'Final search · all 344 rows').click();
    await settle(page);
    const finalText = normalize(await explorer.textContent());
    for (const candidate of NESTED_EXPERIMENT.finalSelected.candidates) {
      assert.ok(finalText.includes(candidate.meanScore.toFixed(6)), `the final search shows candidate mean ${candidate.meanScore.toFixed(6)}`);
    }
    assert.ok(finalText.includes('3/StandardScaler and 3/RobustScaler'), 'and names both tied candidates');
    assert.ok(finalText.includes('none — every row was used to choose'), 'and says it has no protected result of its own');
    assert.equal(await explorer.locator('input, select').count(), 0, 'the final view is read-only too');
    await capture(explorer, 'validation-final-search-desktop.png');
    await button(explorer, 'Outer fold 1').click();
    records.push({ case: 'All eighteen real candidate means plus the final search\u2019s six, the three protected results and baselines, all read-only, with the selection and assessment scores separately labelled and the final two-way tie surfaced' });

    // ---- Figures -----------------------------------------------------
    const rooms = page.locator('[data-cv-figure="nested-rooms"]');
    const roomsText = normalize(await rooms.textContent());
    for (const split of NESTED_EXPERIMENT.folds[0].innerSplits) {
      assert.ok(roomsText.includes(split.validationRows.slice(0, 6).join(', ')),
        `figure 3 prints the first inner row IDs of a ${split.validationRows.length}-row fold`);
    }
    assert.ok(roomsText.includes('Original row IDs stay visible at both levels'), 'and the caption it now keeps');
    records.push({ case: 'Figure 3 prints original row IDs at the outer and the inner level, not sizes alone' });

    for (const [id, filename] of [
      ['loops', 'validation-loops-desktop.png'],
      ['split-question', 'validation-split-question-desktop.png'],
      ['nested-rooms', 'validation-nested-rooms-desktop.png'],
      ['coverage', 'validation-coverage-desktop.png'],
      ['risk', 'validation-risk-desktop.png'],
      ['improvement', 'validation-improvement-desktop.png'],
    ]) await capture(page.locator(`[data-cv-figure="${id}"]`), filename);

    for (const width of [1366, 1024, 768, 390, 320]) {
      await page.setViewportSize({ width, height: 1000 });
      await settle(page);
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), `Document overflows at ${width}px`);
      const report = await page.evaluate(inspectLessonVisualLayout, '.cv-lesson');
      layouts.push({ width, figures: report.length, issues: report.flatMap(figure => figure.issues.map(issue => ({ ...issue, figure: figure.description.slice(0, 60) }))) });
      const tiny = await page.locator('.cv-lesson svg text').evaluateAll(items => items
        .map(item => ({ text: item.textContent.trim().slice(0, 40), size: Number(getComputedStyle(item).fontSize.replace('px', '')) * (item.ownerSVGElement.getBoundingClientRect().width / item.ownerSVGElement.viewBox.baseVal.width) }))
        .filter(item => item.text && item.size < 9.5));
      assert.deepEqual(tiny, [], `SVG text below 9.5 effective pixels at ${width}px`);
    }
    await page.setViewportSize({ width: 320, height: 900 });
    await page.locator('.cv-lesson details').evaluateAll(items => items.forEach(item => { item.open = true; }));
    await settle(page);
    const mathOverflow = await page.locator('.cv-lesson .katex-display').evaluateAll(items => items
      .filter(item => item.scrollWidth > item.clientWidth + 1).map(item => item.textContent.slice(0, 80)));
    assert.deepEqual(mathOverflow, [], 'Overflowing display formulas at 320px');
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'Document overflows at 320px with every disclosure open');
    for (const [selector, filename] of [
      ['[data-cv-figure="loops"]', 'validation-loops-320.png'],
      ['[data-cv-figure="coverage"]', 'validation-coverage-320.png'],
      ['[data-cv-figure="risk"]', 'validation-risk-320.png'],
      ['[data-cv-figure="improvement"]', 'validation-improvement-320.png'],
      ['[data-cv-figure="split-question"]', 'validation-split-question-320.png'],
    ]) await capture(page.locator(selector), filename);
    await page.setViewportSize({ width: 390, height: 900 });
    await settle(page);
    await capture(lab('folds'), 'validation-folds-390.png');
    await capture(lab('halving'), 'validation-halving-390.png');
    records.push({ case: 'Layout inspected at 1366/1024/768/390/320: no document overflow, no display-formula overflow with every disclosure open, no SVG label below 9.5 effective pixels, and narrow captures of every diagram' });

    await page.setViewportSize({ width: 1366, height: 1000 });
    await page.evaluate(() => { document.documentElement.style.fontSize = '200%'; });
    await settle(page);
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'No overflow with enlarged root text');
    await page.evaluate(() => { document.documentElement.style.fontSize = ''; });

    const localScripts = [...new Set(requests.map(address => new URL(address))
      .filter(address => address.origin === new URL(base).origin && address.pathname.endsWith('.js'))
      .map(address => address.pathname.slice(1)))];
    assert.deepEqual(localScripts.filter(filename => bodyFiles.has(filename)), [bodyFile], 'Only this lesson body is loaded');
    assert.ok(!requests.some(address => address.includes('/outlines/')));
    assert.ok(!requests.some(address => address.includes('penguins.csv')), 'the page never fetches the dataset to render');
    records.push({
      case: 'Fresh route requests only the selected lesson body',
      requestedScripts: localScripts.length,
      rawScriptBytes: localScripts.reduce((sum, filename) => sum + fs.statSync(path.join(distDir, filename)).size, 0),
      gzipEstimateBytes: localScripts.reduce((sum, filename) => sum + gzipSync(fs.readFileSync(path.join(distDir, filename))).length, 0),
      interpretation: 'Built file sizes; the gzip figure is an estimate, not a measured network transfer or latency.',
    });

    await page.locator('.reader-complete').click();
    await page.reload({ waitUntil: 'domcontentloaded' });
    await ready(page);
    assert.equal(await page.evaluate(id => JSON.parse(localStorage.getItem('kd-progress'))[id], topicId), true);
    await page.locator('.reader-footer__next').click();
    await page.waitForURL('**/regularization-l1-l2-elastic-net-dropout?module=classical-ml');
    await page.waitForFunction(() => document.querySelector('.reader-header h1')?.textContent.includes('Regularization'), null, { timeout: 20000 });
    assert.deepEqual(errors, []);
    assert.deepEqual(failedAssets, []);
    records.push({ case: 'Completion persists under the stable ID without auto-advance, Next opens the actual successor, and no render error or failed asset appeared' });
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
      await trial.locator('.cv-lesson').waitFor();
      assert.ok(await trial.locator('.reader-complete').isEnabled());
      records.push({ case: `${failure} failure keeps completion disabled and recovers with an explicit reload` });
      await isolated.close();
    }

    for (const [filename, expected] of Object.entries(sourceHashes)) {
      assert.equal(hash(filename), expected, `Source changed during the check: ${filename}`);
    }
    assert.equal(hash(`${distDir}/.vite/manifest.json`), buildHash);
    const report = {
      startedAt, completedAt: new Date().toISOString(), topicId, status: 'passed',
      browser: await browser.version(), distDir, base, buildManifestHash: buildHash, sourceHashes,
      moduleTopicCount: module.topicIds.length, modulePosition: module.topicIds.indexOf(topicId) + 1,
      records, screenshots, layouts,
      realResult: {
        selections: NESTED_EXPERIMENT.folds.map(fold => `${fold.selected.k}/${fold.selected.scaler}`),
        correct: NESTED_EXPERIMENT.folds.map(fold => fold.correct),
        pooledAccuracy: NESTED_EXPERIMENT.pooledAccuracy,
        foldMean: NESTED_EXPERIMENT.foldMean,
        finalSelectionScore: NESTED_EXPERIMENT.finalSelected.selectionScore,
      },
      visualInterpretation: 'Assertions cover geometry, effective label size and overflow. They cannot tell whether a diagram is the right diagram: every screenshot listed here was opened and looked at separately, and the design record names what that pass changed.',
      limits: [
        'A passing layout assertion is not a legible figure; the screenshots were inspected by eye.',
        'One browser channel, one build. No screen-reader pass and no reduced-motion pass were run.',
      ],
    };
    fs.mkdirSync(path.dirname(evidencePath), { recursive: true });
    fs.writeFileSync(evidencePath, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify({ status: report.status, cases: records.length, screenshots: screenshots.length, evidencePath }));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });

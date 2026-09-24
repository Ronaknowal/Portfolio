// Production browser review of the bias-variance lesson: visible content, the
// two investigations, prediction/commit/retirement, figure geometry, narrow
// layouts and stacked tables, sequence, completion and load-failure recovery.
//
// Run against a preview of a production build:
//   npx vite build --outDir dist-bv
//   npx vite preview --outDir dist-bv --host 127.0.0.1 --port 4187
//   PLAYWRIGHT_PACKAGE=... DIST_DIR=dist-bv LEARNING_BASE_URL=http://127.0.0.1:4187 \
//     node scripts/verify-bias-variance-browser.cjs
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { gzipSync } = require('node:zlib');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const { inspectLessonVisualLayout } = require('./lib/lesson-visual-layout.cjs');

const distDir = process.env.DIST_DIR || 'dist-bv';
const base = (process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4187').replace(/\/+$/, '');
const topicId = 'bias-variance-tradeoff-learning-curves';
const sourcePath = `src/learn/data/topics/${topicId}.jsx`;
const route = `${base}/learn/path/full-curriculum/${topicId}?module=classical-ml`;
const evidencePath = 'docs/teaching/evidence/bias-variance-browser.json';
const read = filename => JSON.parse(fs.readFileSync(filename, 'utf8'));
const hash = filename => createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
const normalize = text => text.replace(/\s+/g, ' ').trim();
const ownedFiles = [
  sourcePath,
  'src/learn/data/bias-variance-models.js',
  'src/learn/data/bias-variance-data.js',
  'src/learn/data/bias-variance-examples.js',
  'src/learn/components/lesson-labs/BiasVarianceShared.jsx',
  'src/learn/components/lesson-labs/BiasVarianceLabs.jsx',
  'src/learn/components/lesson-labs/BiasVarianceFigures.jsx',
  'src/learn/components/lesson-labs/bias-variance-labs.css',
  `src/learn/data/curriculum/blueprints/${topicId}.js`,
  'public/learn-assets/bias-variance/airfoil-self-noise.dat',
];

/** Runs in the page. Samples every curve along its own geometry and reports the
 * labels it passes through.
 *
 * `scripts/lib/lesson-visual-layout.cjs` iterates `svg.querySelectorAll('line')`,
 * so <path> and <polyline> are invisible to it — which is how four data curves
 * crossed value labels behind a fully green run. This closes that class for this
 * lesson; the same routine would serve the shared inspector.
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
        halo: text.classList.contains('bv-halo'),
        box: text.getBoundingClientRect(),
      }));
    if (!labels.length) continue;
    for (const shape of svg.querySelectorAll('path, polyline, polygon')) {
      if (shape.classList.contains('bv-grid')) continue;
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

(async () => {
  const startedAt = new Date().toISOString();
  const sourceHashes = Object.fromEntries(ownedFiles.map(filename => [filename, hash(filename)]));
  const build = read(`${distDir}/.vite/manifest.json`);
  const buildHash = hash(`${distDir}/.vite/manifest.json`);
  const publications = read('src/learn/data/lesson-manifest.json');
  const { tracks } = await import('../src/learn/data/tracks.js');
  const { biasVarianceExamples } = await import('../src/learn/data/bias-variance-examples.js');
  const data = await import('../src/learn/data/bias-variance-data.js');
  const models = await import('../src/learn/data/bias-variance-models.js');
  const series = Object.fromEntries(data.procedures.map(entry => [entry.model, models.learningCurveSeries(entry)]));
  const trace = models.trajectorySeries(data.boostingTrajectory);
  const leaf = models.validationCurveSeries(data.validationCurveRecord);
  const module = tracks.find(track => track.id === 'classical-ml');
  assert.equal(module.topicIds[23], topicId, 'the lesson sits at position 24 of its module');
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
    await page.locator('.bv-lesson').waitFor();
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
    await checkText(page.locator('.reader-header h1'), /^Bias-Variance Tradeoff & Learning Curves$/);
    await checkText(page.locator('.reader-header__meta'), /24 of 39 topics on this route/);
    await checkText(page.locator('.reader-footer__previous'), /Feature Selection/);
    await checkText(page.locator('.reader-footer__next'), /Imbalanced Learning/);
    assert.equal(await page.locator('.bv-investigation').count(), 2);
    assert.equal(await page.locator('.bv-figure').count(), 7);
    assert.equal(await page.locator('.bv-practice').count(), 8);
    assert.equal(await page.locator('.python-example').count(), 3);
    const rendered = normalize(await page.locator('.bv-lesson').textContent());
    for (const [key, example] of Object.entries(biasVarianceExamples)) {
      assert.ok(rendered.includes(normalize(example.code)), `Missing actual code for ${key}`);
      assert.ok(rendered.includes(normalize(example.expected)), `Missing actual expected output for ${key}`);
    }
    for (const anchor of await page.locator('.lesson-intro a[href^="#"]').evaluateAll(items => items.map(item => item.getAttribute('href')))) {
      assert.equal(await page.locator(`[id="${anchor.slice(1)}"]`).count(), 1, `Broken lesson anchor ${anchor}`);
    }
    assert.equal(await page.locator('.katex-error').count(), 0);
    for (const entry of data.procedures) {
      series[entry.model].validationMeans.forEach((value, index) => {
        assert.ok(rendered.includes(value.toFixed(4)),
          `${entry.model} at ${data.trainSizes[index]} fitted rows is shown`);
      });
    }
    assert.ok(rendered.includes('The 303 reserved rows were not used for any reported selection or score'),
      'the protected reserve is stated');
    assert.ok(rendered.includes('Both investigations ask for a recorded prediction before they calculate anything'),
      'the intro promises what the page keeps');
    assert.ok(rendered.includes('CC BY 4.0'), 'the licence travels with the data');
    assert.ok(rendered.includes(data.provenance.sha256), 'and so does the file hash');
    assert.equal((rendered.match(/Before running:/g) ?? []).length, 3, 'one Before running per program');
    const asset = await page.request.get(`${base}${data.provenance.file}`);
    assert.equal(asset.status(), 200);
    const bytes = await asset.body();
    assert.equal(bytes.length, data.provenance.bytes, 'the served file is byte-for-byte the packet file');
    assert.equal(createHash('sha256').update(bytes).digest('hex'), data.provenance.sha256, 'and matches its recorded hash');
    assert.ok(bytes.toString('latin1').split('\r\n')[0].split('\t').length === 6, 'six tab-separated columns');
    const attribution = await page.request.get(`${base}${data.provenance.attribution}`);
    assert.equal(attribution.status(), 200);
    assert.ok((await attribution.text()).includes('CC BY 4.0'));
    records.push({ case: 'Complete visible code and output for three programs, eleven route anchors, seven figures, eight practice tasks, all twenty recorded validation means, the served unchanged dataset with its hash and attribution, current metadata and module sequence' });

    // ------------------------------------------------- 2. prediction ruler (I1)
    const spread = page.locator('.bv-investigation').nth(0);
    assert.equal(await spread.locator('input[type="radio"]:checked').count(), 0, 'no prediction is preselected');
    assert.ok(await spread.getByRole('button', { name: 'Apply and check' }).isDisabled(), 'checking waits for a choice');
    assert.equal(await spread.locator('.bv-verdict').count(), 0, 'no answer before a prediction');
    assert.equal(await spread.locator('.bv-waterfall').count(), 0, 'and no decomposition on first paint');
    // Both rulers are on screen before any commitment: they are the entities.
    assert.equal(await spread.locator('.bv-panel svg').count(), 2, 'two separate rulers');
    await screenshot(spread, 'bias-variance-spread-initial-desktop.png');

    await spread.getByRole('button', { name: 'Move the dots inward: 9, 10, 11' }).click();
    await checkText(spread.locator('.bv-pending'), /Draft inputs differ from the applied ones/);
    await spread.getByLabel('It falls', { exact: true }).check();
    await spread.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(spread.locator('.bv-verdict'), /Your prediction matches: It falls\./);
    await checkText(spread, /Total moved from 3\.666666667 to 1\.666666667/);
    await checkText(spread, /the change comes from prediction variance/);
    await screenshot(spread, 'bias-variance-spread-inward-desktop.png');

    // A permutation is an exact null in every term.
    await spread.getByRole('button', { name: 'Baseline: predictions 8, 10, 12' }).click();
    await spread.getByLabel('It rises', { exact: true }).check();
    await spread.getByRole('button', { name: 'Apply and check' }).click();
    await spread.getByRole('button', { name: 'Null: the same dots reordered, 12, 10, 8' }).click();
    await spread.getByLabel('Unchanged within tolerance', { exact: true }).check();
    await spread.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(spread.locator('.bv-verdict'), /Your prediction matches: Unchanged within tolerance\./);
    await checkText(spread.locator('.bv-readout'), /Squared bias, prediction variance and target noise are unchanged/);
    await screenshot(spread, 'bias-variance-spread-null-desktop.png');

    // The translation null: every prediction and the true mean move together.
    await spread.getByRole('button', { name: 'Null: move everything up 2' }).click();
    await spread.getByLabel('Unchanged within tolerance', { exact: true }).check();
    await spread.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(spread.locator('.bv-verdict'), /Your prediction matches: Unchanged within tolerance\./);

    await checkText(spread.locator('.bv-readout'), /average prediction itself can still move with the target mean/);
    assert.doesNotMatch(await spread.locator('.bv-readout').innerText(), /left the average prediction.*exactly where/);

    // Same average, no spread; and the enumerated pair table.
    await spread.getByRole('button', { name: 'Same average, no spread: 10, 10, 10' }).click();
    await spread.getByLabel('It falls', { exact: true }).check();
    await spread.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(spread, /exactly 0/);
    await spread.getByRole('group', { name: /Enumerate|enumerate/ }).count().catch(() => 0);
    await spread.locator('details').first().evaluate(element => { element.open = true; });
    await checkText(spread, /All 6 equally weighted pairs/);
    await screenshot(spread, 'bias-variance-spread-enumerated-desktop.png');

    // An edit retires the recorded prediction and hides its feedback.
    await spread.getByRole('spinbutton', { name: /^Prediction D1/ }).fill('12');
    assert.equal(await spread.locator('input[type="radio"]:checked').count(), 0, 'the edit retired the choice');
    assert.equal(await spread.locator('.bv-verdict').count(), 0, 'and hid the stale feedback');
    await checkText(spread.locator('.bv-pending'), /the comparison will be against the state currently applied/);
    await spread.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'Prediction ruler: nothing preselected and no decomposition on first paint, a fall from 11/3 to 5/3 attributed to the spread alone, both exact nulls (a permutation and a translation) reporting no moved term, the collapsed setup showing an exact zero variance, the six-pair enumeration, and an edit that retires both the prediction and its feedback' });

    // -------------------------------------------------- 3. finite worlds (I2)
    const worlds = page.locator('.bv-investigation').nth(1);
    assert.equal(await worlds.locator('input[type="radio"]:checked').count(), 0, 'the worlds prediction opens unset');
    assert.equal(await worlds.locator('.bv-plot').count(), 0, 'no fitted curves before a prediction');
    await worlds.getByLabel('Candidate lower', { exact: true }).check();
    await worlds.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(worlds.locator('.bv-verdict'), /Your prediction matches: Candidate lower\./);
    await checkText(worlds, /Reference 0\.538194444 against candidate 0\.4296875/);
    await checkText(worlds, /There are 8 training worlds/);
    assert.equal(await worlds.locator('.bv-plot').count(), 2, 'a reference panel and a candidate panel');
    await screenshot(worlds, 'bias-variance-worlds-default-desktop.png');

    // The flat-truth contrast reverses the verdict.
    await worlds.getByRole('button', { name: 'Flat truth: curvature 0' }).click();
    assert.equal(await worlds.locator('input[type="radio"]:checked').count(), 0, 'a preset retires the choice');
    await worlds.getByLabel('Candidate higher', { exact: true }).check();
    await worlds.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(worlds.locator('.bv-verdict'), /Your prediction matches: Candidate higher\./);
    await screenshot(worlds, 'bias-variance-worlds-flat-desktop.png');

    // Two nulls: identical predictions in every world, and a noiseless tie.
    await worlds.getByRole('button', { name: 'Null: probe 0, constant against line' }).click();
    await worlds.getByLabel('Equal within tolerance', { exact: true }).check();
    await worlds.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(worlds.locator('.bv-readout'), /Every training world gives predictions within 10⁻¹²/);
    await screenshot(worlds, 'bias-variance-worlds-null-desktop.png');
    await worlds.getByRole('button', { name: 'Null: no noise and no curvature' }).click();
    await worlds.getByLabel('Equal within tolerance', { exact: true }).check();
    await worlds.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(worlds.locator('.bv-readout'), /With σ = 0 every world has the same targets/);

    // Practice 2's fixture, and the interpolation weights behind it.
    await worlds.getByRole('button', { name: 'Louder noise: σ = 1' }).click();
    await worlds.getByLabel('Candidate higher', { exact: true }).check();
    await worlds.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(worlds, /0\.71875/);
    await checkText(worlds, /1\.71875/);
    await worlds.locator('details').first().evaluate(element => { element.open = true; });
    await checkText(worlds, /−0\.125/);
    await screenshot(worlds, 'bias-variance-worlds-louder-desktop.png');

    // The five-input design: thirty-two worlds and the pointwise reversal.
    await worlds.getByRole('button', { name: 'Five training inputs, constant against quadratic' }).click();
    await worlds.getByLabel('Candidate lower', { exact: true }).check();
    await worlds.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(worlds, /There are 32 training worlds/);
    await checkText(worlds, /0\.3625/);
    await checkText(worlds, /0\.340277778/);
    await screenshot(worlds, 'bias-variance-worlds-five-desktop.png');
    // Selecting a world is inspection: it must not move any error term.
    const beforeSelection = normalize(await worlds.locator('.bv-readout').innerText());
    await worlds.getByLabel('Inspect one training world').selectOption({ index: 17 });
    await settle(page);
    assert.equal(normalize(await worlds.locator('.bv-readout').innerText()), beforeSelection,
      'choosing a world changes nothing about the averaged terms');
    await screenshot(worlds, 'bias-variance-worlds-selected-desktop.png');
    await worlds.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'Finite worlds: no curves before a prediction, the default line-against-quadratic verdict, the flat-truth reversal, two exact nulls with their own explanations, practice 2\'s louder-noise fixture at 23/32 and 55/32 with its interpolation weights, the thirty-two-world design showing the 0.340278 → 0.3625 reversal, and a world selector that leaves every averaged term alone' });

    // ------------------------------------------------------------- 4. figures
    const deviation = page.locator('.bv-figure').nth(0);
    await checkText(deviation, /expectation zero|mean zero/);
    await checkText(deviation, /1\.25/);
    await checkText(deviation, /0\.25/);
    const split = page.locator('.bv-figure').nth(1);
    await checkText(split, /1,503/);
    await checkText(split, /reserved: no fit, no score/);
    await checkText(split, /960/);
    const curves = page.locator('.bv-figure').nth(2);
    await checkText(curves, new RegExp(`tree better from ${models.firstCrossing(series.tree_leaf1, series.ridge)}`));
    await checkText(curves, /The segments between fitted sizes are joins, not measurements/);
    await checkText(curves, /exactly zero at every inspected size/);
    await curves.getByRole('button', { name: /fold values/ }).click();
    await checkText(curves, /Show the individual fold values/);
    await curves.getByRole('button', { name: /fold values/ }).click();
    const axes = page.locator('.bv-figure').nth(3);
    await checkText(axes, new RegExp(`round ${trace.bestRound}, which is the last one`));
    await checkText(axes, new RegExp(`the least restricted, ${leaf.bestSetting}`));
    await axes.getByLabel('Inspect one recorded round').selectOption('30');
    await checkText(axes.locator('.bv-readout'), /Round 30: training MSE 17\.5512, monitoring MSE 20\.0168/);
    await screenshot(axes, 'bias-variance-figure-4-round30-desktop.png');
    const sameInputs = page.locator('.bv-figure').nth(4);
    await checkText(sameInputs, /2\.666667/);
    await checkText(sameInputs, /5\.333333/);
    await checkText(sameInputs, /1\.333333/);
    const loss = page.locator('.bv-figure').nth(5);
    await checkText(loss, /0\.22/);
    await checkText(loss, /0\.26/);
    const descent = page.locator('.bv-figure').nth(6);
    await checkText(descent, /undefined/);
    await checkText(descent, /4\.01/);
    await checkText(descent, /0\.944444/);
    await axes.getByLabel('Inspect one recorded round').selectOption(String(trace.bestRound));
    await settle(page);
    for (let index = 0; index < 7; index += 1) {
      await screenshot(page.locator('.bv-figure').nth(index), `bias-variance-figure-${index + 1}-desktop.png`);
    }
    assert.equal(new Set(screenshotPaths).size, screenshotPaths.length, 'every capture has its own path');
    records.push({ case: 'Figures carry their required content: the cross terms are named rather than dropped, the reserved block has a blocked arrow, the crossover size is drawn from the model layer, the trajectory names its own last-round minimum, the optimism triple appears with its doubled gap, and the asymptotic curve says where it is undefined' });

    // ------------------------------------------- 4b. plotted coordinates
    // Every other figure assertion is a regex over rendered text, so swapping a
    // training series for a validation one, or drawing the crossover stem at the
    // wrong x while its label stayed right, would pass. Read the geometry.
    const plotted = await curves.locator('.bv-panel').first().locator('svg').first().evaluate(svg => {
      const box = svg.viewBox.baseVal;
      return {
        width: box.width,
        height: box.height,
        series: [...svg.querySelectorAll('polyline.bv-curve')].map(item => ({
          stroke: item.getAttribute('stroke'),
          points: item.getAttribute('points').split(' ').map(pair => pair.split(',').map(Number)),
        })),
        stem: (() => {
          const line = svg.querySelector('line.bv-stem');
          return line && { x1: Number(line.getAttribute('x1')), x2: Number(line.getAttribute('x2')) };
        })(),
      };
    });
    const sizeDomain = [30, 930];
    // The axis must contain every displayed fold, not only the means. This
    // dataset's maximum 51.363045026 requires a padded upper tick of 60.
    const errorRange = [0, 60];
    const padding = { left: 42, right: 14, top: 16, bottom: 34 };
    const expectX = value => padding.left + (plotted.width - padding.left - padding.right)
      * (value - sizeDomain[0]) / (sizeDomain[1] - sizeDomain[0]);
    const expectY = value => plotted.height - padding.bottom
      - (plotted.height - padding.top - padding.bottom) * (value - errorRange[0]) / (errorRange[1] - errorRange[0]);
    assert.equal(plotted.series.length, 4, 'the combined plot draws one polyline per procedure');
    const strokeOf = { mean: '#a08a5a', ridge: '#91aecf', tree_leaf1: '#e7b94a', tree_leaf20: '#8eb9a5' };
    for (const entry of data.procedures) {
      const drawn = plotted.series.find(item => item.stroke === strokeOf[entry.model]);
      assert.ok(drawn, `a polyline is drawn for ${entry.model}`);
      assert.equal(drawn.points.length, data.trainSizes.length, `${entry.model} plots every fitted size`);
      drawn.points.forEach((point, index) => {
        // The drawn height must be the validation mean, not the training mean.
        assert.ok(Math.abs(point[0] - expectX(data.trainSizes[index])) < 0.75,
          `${entry.model} point ${index} sits at its fitted size`);
        assert.ok(Math.abs(point[1] - expectY(series[entry.model].validationMeans[index])) < 0.75,
          `${entry.model} point ${index} is drawn at its validation mean`);
        assert.ok(Math.abs(point[1] - expectY(series[entry.model].trainingMeans[index])) > 0.75
          || Math.abs(series[entry.model].validationMeans[index] - series[entry.model].trainingMeans[index]) < 0.3,
          `${entry.model} point ${index} is not the training mean`);
      });
    }
    assert.ok(Math.abs(plotted.stem.x1 - expectX(models.firstCrossing(series.tree_leaf1, series.ridge))) < 0.75,
      'the crossover stem is drawn at the size its label names');
    records.push({ case: 'Plotted coordinates read back from the combined learning-curve plot: four polylines, each point at its own fitted size and at its validation mean rather than its training mean, and the crossover stem at the size its label names' });

    const paintedWidths = await curves.locator('.bv-panel').first().locator('polyline.bv-curve').evaluateAll(nodes =>
      nodes.map(node => Number.parseFloat(getComputedStyle(node).strokeWidth)));
    assert.deepEqual(paintedWidths, [2.2, 2.2, 2.2, 2.2], 'four combined curves paint the intended widths');
    const foldGeometry = await curves.locator('.bv-panels .bv-panel svg').evaluateAll(svgs => svgs.map(svg => ({
      points: [...svg.querySelectorAll('circle:not(.bv-mark)')].map(node => ({
        x: Number(node.getAttribute('cx')), y: Number(node.getAttribute('cy')), r: Number(node.getAttribute('r')),
      })),
      ticks: [...svg.querySelectorAll('text')].filter(node => node.getAttribute('text-anchor') === 'end').map(node => node.textContent),
    })));
    assert.equal(foldGeometry.length, 4, 'all four fold panels are checked');
    for (const panel of foldGeometry) {
      assert.equal(panel.points.length, 50, 'twenty-five training and twenty-five validation fold dots');
      assert.deepEqual(panel.ticks, ['0', '30', '60'], 'all fold panels share the full-data scale');
      for (const point of panel.points) {
        assert(point.x >= 40 && point.x <= 288 && point.y >= 14 && point.y <= 158,
          `fold point ${JSON.stringify(point)} lies within the plotted axes`);
      }
    }
    const subsetWidths = await split.locator('svg').nth(1).locator('rect.bv-lane:not(.is-fold):not(.is-reserved)').evaluateAll(nodes =>
      nodes.map(node => Number(node.getAttribute('width'))));
    assert.deepEqual(subsetWidths, [9, 18, 36, 72, 135], 'subset bar lengths are proportional to 60/120/240/480/900 rows');
    await checkText(sameInputs, /Averaging prediction variance.*leverage/s);
    await checkText(deviation, /Each distinct mixed product appears in two off-diagonal cells/);
    const arrow = descent.locator('polygon.bv-flow');
    assert.equal(await arrow.count(), 1, 'one continuation arrow is drawn');
    assert.equal(await arrow.evaluate(node => getComputedStyle(node).fill), 'rgb(142, 185, 165)', 'the continuation arrow paints green');
    assert.doesNotMatch(await descent.locator('svg').first().getAttribute('aria-label'), /two arrows/);
    records.push({ case: 'Sequential figure regressions: all 200 fold points inside their shared 0–60 axes, four painted 2.2px curves, proportional subset bars without a minimum-width floor, corrected leverage and mixed-product prose, and one filled continuation arrow' });

    // ------------------------------------------- 4c. the grading contract's unhappy paths
    // Every other verdict assertion matches on a correct prediction, so an
    // inverted comparison would pass. Commit a deliberately wrong one.
    await spread.getByRole('button', { name: 'Reset', exact: true }).click();
    await spread.getByRole('button', { name: 'Move them outward: 6, 10, 14' }).click();
    await spread.getByLabel('It falls', { exact: true }).check();
    await spread.getByLabel('Optional: the new total expected squared error').fill('1');
    await spread.getByRole('button', { name: 'Apply and check' }).click();
    const miss = spread.locator('.bv-verdict');
    assert.equal(await miss.evaluate(node => node.classList.contains('is-miss')), true,
      'a wrong prediction renders the mismatch branch');
    await checkText(miss, /You recorded It falls; the calculation gives It rises\./);
    await checkText(miss, /You wrote 1 for the new total; the calculation gives 11\.666666667, outside 0\.0001/);
    await checkText(spread, /Graded against the committed state: \[8, 10, 12\], f = 10, σ = 1 → \[6, 10, 14\]/);
    await screenshot(spread, 'bias-variance-spread-mismatch-desktop.png');
    // And a numeric guess inside the tolerance is reported as inside.
    await spread.getByRole('button', { name: 'Reset', exact: true }).click();
    await spread.getByRole('button', { name: 'Procedure B: 9, 9, 9' }).click();
    await spread.getByLabel('It falls', { exact: true }).check();
    await spread.getByLabel('Optional: the new total expected squared error').fill('2');
    await spread.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(spread.locator('.bv-verdict'), /the calculation gives 2, within 0\.0001/);
    assert.equal(await spread.locator('.bv-verdict').evaluate(node => node.classList.contains('is-miss')), false,
      'a correct prediction does not render the mismatch branch');
    await spread.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'The grading contract off its happy path: a wrong direction renders the mismatch branch naming both the recorded and the calculated outcome, a numeric guess is reported outside its tolerance, the committed state is echoed, and a guess inside tolerance is reported as inside' });

    // Sequential audit: decimal tolerance endpoints and multiple changed causes.
    for (const [guess, accepted] of [['2.0001', true], ['1.9999', true], ['2.000101', false], ['1.999899', false]]) {
      await spread.getByRole('button', { name: 'Reset', exact: true }).click();
      await spread.getByRole('button', { name: 'Procedure B: 9, 9, 9' }).click();
      await checkText(spread.locator('.bv-prediction'), /Answers within 0\.0001 are accepted/);
      await spread.getByLabel('It falls', { exact: true }).check();
      await spread.getByLabel('Optional: the new total expected squared error', { exact: false }).fill(guess);
      await spread.getByRole('button', { name: 'Apply and check' }).click();
      await checkText(spread.locator('.bv-verdict'), accepted ? /calculation gives 2, within 0\.0001/ : /calculation gives 2, outside 0\.0001/);
    }
    for (const preset of ['Move the dots inward: 9, 10, 11', 'Same spread, shifted up: 9, 11, 13']) {
      await spread.getByRole('button', { name: 'Reset', exact: true }).click();
      await spread.getByRole('button', { name: preset }).click();
      await spread.getByRole('spinbutton', { name: /^Fresh-outcome spread/ }).fill('2');
      await spread.getByLabel('It rises', { exact: true }).check();
      await spread.getByRole('button', { name: 'Apply and check' }).click();
      await checkText(spread.locator('.bv-verdict'), /Your prediction matches/);
      await checkText(spread.locator('.bv-readout'), /Changed operands:.*target noise/);
      assert.doesNotMatch(await spread.locator('.bv-readout').innerText(), /so the change comes from|whole change came from/,
        'a simultaneous noise change cannot be attributed to the other operand alone');
    }
    await screenshot(spread, 'bias-variance-spread-multiple-causes-desktop.png');
    await spread.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'Sequential numeric boundary and causal feedback: inclusive decimal tolerance endpoints pass, just-outside answers fail, translation does not claim the average stayed fixed, and simultaneous variance/noise or bias/noise changes are not attributed to a single term' });

    // ------------------------------------------- 4d. keyboard reach and displayed code
    const focusOrder = await page.evaluate(() => {
      const lab = document.querySelectorAll('.bv-investigation')[0];
      const reachable = [...lab.querySelectorAll('button, input, select, textarea')]
        .filter(node => !node.disabled);
      return { count: reachable.length, firstRadio: reachable.some(node => node.type === 'radio') };
    });
    assert.ok(focusOrder.count > 10 && focusOrder.firstRadio, 'the investigation exposes enabled focusable controls');
    await spread.getByRole('spinbutton', { name: /^Prediction D1/ }).focus();
    assert.equal(await page.evaluate(() => document.activeElement.getAttribute('type')), 'number',
      'a control takes keyboard focus');
    await page.keyboard.press('Tab');
    assert.ok(await page.evaluate(() => document.activeElement !== document.body),
      'focus moves on to the next control rather than being dropped');
    // The code-on-page check normalizes whitespace, so Python indentation would
    // survive being destroyed. Read it raw.
    // CodeBlock renders a white-space:pre <div>, not a <pre>.
    const rawProgram = await page.locator('.python-example').first().textContent();
    assert.ok(/\n {4}design = np\.vander/.test(rawProgram),
      'the displayed program keeps the indentation that makes it valid Python');
    records.push({ case: 'Keyboard reach into the investigation, focus moving on rather than being dropped, and a displayed program whose Python indentation survives on the page' });

    // ------------------------------------------------- 5. drawn geometry checks
    assert.equal(await page.getByRole('button', { name: /without recording a prediction/ }).count(), 0,
      'no investigation offers a way round the prediction');
    await spread.getByLabel('Unchanged within tolerance', { exact: true }).check();
    await spread.getByRole('button', { name: 'Apply and check' }).click();
    await worlds.getByLabel('Candidate lower', { exact: true }).check();
    await worlds.getByRole('button', { name: 'Apply and check' }).click();
    await settle(page);
    assert.ok(await page.locator('.bv-investigation .bv-plot').count() >= 2, 'both fitted panels are rendered for this pass');
    for (const width of [1366, 1024, 768, 390, 320]) {
      await page.setViewportSize({ width, height: 1000 });
      await settle(page);
      const issues = await page.evaluate(inspectLessonVisualLayout, '.bv-lesson');
      assert.deepEqual(issues.flatMap(figure => figure.issues), [], `Figure layout collides at ${width}px`);
      // The shared inspector samples straight <line> elements only, so a data
      // curve or a flow arrow can run through a label and still report clean.
      // Sample the curve geometry itself and test it against every label box.
      const curveHits = await page.evaluate(sampleCurvesThroughLabels, '.bv-lesson');
      assert.deepEqual(curveHits, [], `A curve runs through a label at ${width}px`);
    }
    await page.setViewportSize({ width: 1366, height: 1000 });
    records.push({
      case: 'No label leaves its own SVG, overlaps another label, is crossed by a straight foreground line, or is run through by a curve, at five widths, with both investigations committed',
      curveSamplingNote: 'Curves and flow arrows are sampled along their own geometry and tested against every label box, because the shared inspector reads straight lines only. A label with an opaque backplate may be crossed by up to 2% of a curve length; a label without one may not be crossed at all.',
    });

    // ------------------------------------------------------ 6. loading closure
    const localScripts = [...new Set(requests.map(address => new URL(address)).filter(address => address.origin === new URL(base).origin && address.pathname.endsWith('.js')).map(address => address.pathname.slice(1)))];
    assert.deepEqual(localScripts.filter(filename => bodyFiles.has(filename)), [bodyFile]);
    assert.deepEqual(localScripts.filter(filename => !allowedScripts.has(filename)), []);
    assert.ok(!requests.some(address => address.includes('/outlines/')));
    assert.ok(!requests.some(address => address.includes('airfoil-self-noise.dat') && !address.includes('learn-assets')),
      'the page never fetches the dataset to render');
    assert.ok(!requests.some(address => address.includes('learn-assets/regularization')),
      'and never borrows another lesson\'s copy of it');
    records.push({
      case: 'Fresh route requests only the selected lesson and its shared closure, and never another lesson\'s dataset',
      requestedScripts: localScripts,
      rawScriptBytes: localScripts.reduce((sum, filename) => sum + fs.statSync(path.join(distDir, filename)).size, 0),
      gzipEstimateBytes: localScripts.reduce((sum, filename) => sum + gzipSync(fs.readFileSync(path.join(distDir, filename))).length, 0),
      interpretation: 'Built requested file sizes; the gzip estimate is not measured network compression or latency.',
    });

    // -------------------------------------------------------- 7. narrow widths
    for (const width of [390, 320]) {
      await page.setViewportSize({ width, height: 844 });
      await page.locator('.bv-lesson details').evaluateAll(items => items.forEach(item => { item.open = true; }));
      await settle(page);
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `Document overflows at ${width}px`);
      const mathOverflow = await page.locator('.bv-lesson .katex-display').evaluateAll(items => items.filter(item => item.scrollWidth > item.clientWidth + 1).map(item => item.textContent));
      assert.deepEqual(mathOverflow, [], `Overflowing formulas at ${width}px`);
      assert.equal(await page.locator('.katex-error').count(), 0);
      // Below the breakpoint a wide numeric table stacks instead of clipping:
      // each row becomes its own block and each cell a two-column grid whose
      // first column is the heading it lost.
      const stacking = await page.locator('.bv-table-scroll tbody tr').first()
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
      const clipped = await page.locator('.bv-table-scroll').evaluateAll(items =>
        items.filter(item => item.scrollWidth > item.clientWidth + 1).length);
      assert.equal(clipped, 0, `A table still scrolls sideways at ${width}px`);
      if (width === 320) {
        await screenshot(page.locator('.bv-investigation').nth(0), 'bias-variance-spread-320.png');
        await screenshot(page.locator('.bv-figure').nth(2), 'bias-variance-figure-3-320.png');
        await screenshot(page.locator('.bv-figure').nth(6), 'bias-variance-figure-7-320.png');
      }
      if (width === 390) {
        await screenshot(page.locator('.bv-investigation').nth(0), 'bias-variance-spread-390.png');
        await screenshot(page.locator('.bv-investigation').nth(1), 'bias-variance-worlds-390.png');
        for (const index of [0, 1, 3, 4, 5, 6]) {
          await screenshot(page.locator('.bv-figure').nth(index), `bias-variance-figure-${index + 1}-390.png`);
        }
      }
      records.push({ case: `Narrow ${width}px layout: no horizontal overflow, readable formula grouping, every deeper branch rendered, and every numeric table stacked into labelled rows` });
    }
    await page.setViewportSize({ width: 1366, height: 1000 });

    // -------------------------------------------------------- 8. sequence
    await page.locator('.reader-complete').click();
    assert.equal(new URL(page.url()).pathname.split('/').at(-1), topicId);
    await page.reload({ waitUntil: 'domcontentloaded' });
    await ready(page);
    assert.equal(await page.evaluate(id => JSON.parse(localStorage.getItem('kd-progress'))[id], topicId), true);
    await page.locator('.reader-footer__next').click();
    await page.waitForURL('**/imbalanced-learning-smote-cost-sensitive-learning?module=classical-ml');
    records.push({ case: 'Completion persists under the stable ID without auto-advance; Next opens the actual successor' });
    assert.deepEqual(errors, []);
    assert.deepEqual(failedAssets, []);
    await context.close();

    // -------------------------------------------------------- 9. recovery
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
      dataset: {
        file: data.provenance.file, sha256: data.provenance.sha256, bytes: data.provenance.bytes,
        reservedPredictionsComputed: data.provenance.reservedPredictionsComputed,
      },
      records, screenshots: screenshotPaths,
      visualInterpretation: 'The intended Space Grotesk font loaded before the checks. Screenshots capture informative states, not defaults only: the ruler before any commitment, a matched fall attributed to the spread alone, both exact nulls, the enumerated pair table, the fitted-world bundles under four different mechanisms including the thirty-two-world design, an inspected boosting round, and every inline figure at desktop and at 320 or 390 px. They require separate visual inspection.',
    };
    fs.writeFileSync(evidencePath, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify({ status: report.status, cases: records.length, evidencePath, sourceFiles: ownedFiles.length, screenshots: screenshotPaths.length }));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });

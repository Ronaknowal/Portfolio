// Production browser review of the Gaussian mixture lesson: visible content, the
// three investigations, prediction/commit/recovery, narrow layouts, sequence,
// completion and load-failure recovery.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { gzipSync } = require('node:zlib');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const { inspectLessonVisualLayout } = require('./lib/lesson-visual-layout.cjs');

const distDir = process.env.DIST_DIR || 'dist';
const base = (process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4173').replace(/\/+$/, '');
const topicId = 'gaussian-mixture-models-gmm-em-algorithm';
const sourcePath = `src/learn/data/topics/${topicId}.jsx`;
const route = `${base}/learn/path/full-curriculum/${topicId}?module=classical-ml`;
const evidencePath = 'docs/teaching/evidence/gmm-browser.json';
const read = filename => JSON.parse(fs.readFileSync(filename, 'utf8'));
const hash = filename => createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
const normalize = text => text.replace(/\s+/g, ' ').trim();
const ownedFiles = [
  sourcePath,
  'src/learn/data/gmm-models.js',
  'src/learn/data/gmm-iris-data.js',
  'src/learn/data/gmm-examples.js',
  'src/learn/components/lesson-labs/GmmShared.jsx',
  'src/learn/components/lesson-labs/GmmLabs.jsx',
  'src/learn/components/lesson-labs/GmmFigures.jsx',
  'src/learn/components/lesson-labs/gmm-labs.css',
  `src/learn/data/curriculum/blueprints/${topicId}.js`,
  'public/learn-assets/gmm/iris.csv',
];

(async () => {
  const startedAt = new Date().toISOString();
  const sourceHashes = Object.fromEntries(ownedFiles.map(filename => [filename, hash(filename)]));
  const build = read(`${distDir}/.vite/manifest.json`);
  const buildHash = hash(`${distDir}/.vite/manifest.json`);
  const publications = read('src/learn/data/lesson-manifest.json');
  const { tracks } = await import('../src/learn/data/tracks.js');
  const { gmmExamples } = await import('../src/learn/data/gmm-examples.js');
  const { selection, candidates } = await import('../src/learn/data/gmm-iris-data.js');
  const module = tracks.find(track => track.id === 'classical-ml');
  assert.equal(module.topicIds[15], topicId);
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
    await page.locator('.gm-lesson').waitFor();
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
    await checkText(page.locator('.reader-header h1'), /^Gaussian Mixture Models \(GMM\) & EM Algorithm$/);
    await checkText(page.locator('.reader-header__meta'), /16 of 39 topics on this route/);
    await checkText(page.locator('.reader-footer__previous'), /Anomaly & Outlier Detection/);
    await checkText(page.locator('.reader-footer__next'), /t-SNE, UMAP/);
    assert.equal(await page.locator('.gm-investigation').count(), 3);
    assert.equal(await page.locator('.gm-figure').count(), 6);
    assert.equal(await page.locator('.gm-practice').count(), 8);
    assert.equal(await page.locator('.python-example').count(), 3);
    const rendered = normalize(await page.locator('.gm-lesson').textContent());
    for (const [key, example] of Object.entries(gmmExamples)) {
      assert.ok(rendered.includes(normalize(example.code)), `Missing actual code for ${key}`);
      assert.ok(rendered.includes(normalize(example.expected)), `Missing actual expected output for ${key}`);
    }
    for (const anchor of await page.locator('.lesson-intro a[href^="#"]').evaluateAll(items => items.map(item => item.getAttribute('href')))) {
      assert.equal(await page.locator(`[id="${anchor.slice(1)}"]`).count(), 1, `Broken lesson anchor ${anchor}`);
    }
    assert.equal(await page.locator('.katex-error').count(), 0);
    const csv = await page.request.get(`${base}/learn-assets/gmm/iris.csv`);
    assert.equal(csv.status(), 200);
    const csvText = await csv.text();
    assert.equal(csvText.trim().split('\n').length, 151, 'the CSV serves a header and 150 rows');
    assert.ok(csvText.startsWith('observation_id,sepal_length_cm,sepal_width_cm,petal_length_cm,petal_width_cm,species'));
    // Every candidate row is visible, so the selection is auditable on the page.
    for (const row of candidates) assert.ok(rendered.includes(row[2].toFixed(6)), `candidate ${row[0]} K=${row[1]} score is shown`);
    assert.ok(rendered.includes('Every investigation asks for a prediction before it shows an answer'), 'the intro promises what the page keeps');
    assert.ok(rendered.includes('Numerical fitting can vary across versions'), 'the version caveat travels with the candidate table');
    assert.ok(rendered.includes('np.random.default_rng(16).permutation(150)'), 'the split is reproducible from the prose');
    assert.equal((rendered.match(/Before running:/g) ?? []).length, 3, 'one Before running per program, not two');
    records.push({ case: 'Complete visible code and output for three programs, twelve route anchors, six figures, eight practice tasks, all sixteen candidate scores, the downloadable CSV, current metadata and module sequence' });

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

    // 1. Responsibility against density.
    const responsibility = page.locator('.gm-investigation').nth(0);
    assert.equal(await responsibility.locator('input[type="radio"]:checked').count(), 0, 'no prediction is preselected');
    assert.ok(await responsibility.getByRole('button', { name: 'Check prediction' }).isDisabled(), 'checking waits for a choice');
    assert.doesNotMatch(await responsibility.locator('.gm-plot svg').getAttribute('aria-label'), /responsibilities are [\d.]+ and/, 'accessible text also withholds the answer before prediction');
    await responsibility.getByLabel('Component B', { exact: true }).check();
    await responsibility.getByRole('button', { name: 'Check prediction' }).click();
    await checkText(responsibility.locator('.gm-verdict'), /Your prediction matches: Component B\./);
    await checkText(responsibility, /0\.241970725|0\.24197/);
    await responsibility.getByLabel(/^Measurement x/).fill('8');
    await checkText(responsibility.locator('.gm-pending'), /Inputs changed; record a new prediction/);
    assert.equal(await responsibility.locator('input[type="radio"]:checked').count(), 0, 'the edit retired the choice');
    await responsibility.getByLabel('Component B', { exact: true }).check();
    await responsibility.getByRole('button', { name: 'Check prediction' }).click();
    await checkText(responsibility.locator('.gm-readout'), /negative log-density 19\.612086/);
    await responsibility.getByRole('button', { name: 'Save this point for the comparison' }).click();
    await responsibility.getByLabel(/^Measurement x/).fill('2');
    await responsibility.getByLabel('Component B', { exact: true }).check();
    await responsibility.getByRole('button', { name: 'Check prediction' }).click();
    await responsibility.getByRole('button', { name: 'Save this point for the comparison' }).click();
    await checkText(responsibility, /Both points give B a responsibility above 0\.99, and their negative log-densities differ by 18\.0003/);
    await screenshot(responsibility, 'gmm-responsibility-desktop.png');
    await responsibility.getByRole('button', { name: 'Identical components' }).click();
    await responsibility.getByLabel('They are equal', { exact: true }).check();
    await responsibility.getByRole('button', { name: 'Check prediction' }).click();
    await checkText(responsibility.locator('.gm-verdict'), /You recorded They are equal; the calculation gives Component B\..*responsibilities are exactly the mixing weights 0\.2 and 0\.8/);
    await responsibility.getByRole('button', { name: 'Reset', exact: true }).click();
    // Practice 1 sends the learner here with means 0 and 2 and weights 0.25/0.75.
    await responsibility.getByRole('spinbutton', { name: /^Mean of A/ }).fill('0');
    await responsibility.getByRole('spinbutton', { name: /^Mean of B/ }).fill('2');
    await responsibility.getByRole('spinbutton', { name: /^Weight of A/ }).fill('0.25');
    await responsibility.getByRole('spinbutton', { name: /^Measurement x/ }).fill('1');
    await responsibility.getByLabel('Component B', { exact: true }).check();
    await responsibility.getByRole('button', { name: 'Check prediction' }).click();
    await checkText(responsibility, /tie at x = 0\.450693856/);
    await checkText(responsibility.locator('.gm-verdict'), /Your prediction matches: Component B/);
    await checkText(responsibility, /0\.241971/);
    await screenshot(responsibility, 'gmm-responsibility-boundary-desktop.png');
    assert.ok(await responsibility.getByLabel('Component A', { exact: true }).isDisabled(), 'a graded choice is frozen');
    await responsibility.getByRole('button', { name: 'Use tie 1 as measurement' }).click();
    await responsibility.getByLabel('They are equal', { exact: true }).check();
    await responsibility.getByRole('button', { name: 'Check prediction' }).click();
    await checkText(responsibility.locator('.gm-verdict'), /Your prediction matches: They are equal/);
    await responsibility.getByRole('button', { name: 'Identical components' }).click();
    await responsibility.getByRole('spinbutton', { name: /^Weight of A/ }).fill('0.5');
    await responsibility.getByRole('spinbutton', { name: /^Variance of A/ }).fill('4');
    await responsibility.getByRole('button', { name: 'Calculate without recording a prediction' }).click();
    assert.equal(await responsibility.getByRole('button', { name: /^Use tie/ }).count(), 2, 'unequal variances can create two crossings');
    await responsibility.getByRole('spinbutton', { name: /^Weight of A/ }).fill('0.05');
    await responsibility.getByRole('spinbutton', { name: /^Variance of A/ }).fill('0.25');
    await responsibility.getByRole('button', { name: 'Calculate without recording a prediction' }).click();
    await checkText(responsibility, /There is no responsibility tie inside/);
    assert.doesNotMatch(await responsibility.locator('.gm-caption').first().innerText(), /With identical components/);
    await responsibility.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'Responsibility crossing regression: exact practice tie is selectable and graded, two roots are both offered, a no-root unequal model is not called identical, committed choices cannot be edited, and accessible answers wait for reveal' });
    records.push({ case: 'Responsibility lab: no preselected prediction, a retired choice on an edited measurement, the 19.612086 contrast at 8, the saved two-point comparison differing by 18.000336, and the identical-component null returning the mixing weights' });

    // 2. One exact EM cycle.
    const em = page.locator('.gm-investigation').nth(1);
    assert.equal(await em.locator('input[type="radio"]:checked').count(), 0, 'the EM prediction opens unset');
    await checkText(em, /The allocation these starting parameters already imply, before any step/);
    await checkText(em, /0\.98201379/);
    await em.getByLabel('It rises', { exact: true }).check();
    await em.getByRole('button', { name: 'Record it' }).click();
    await em.getByRole('button', { name: 'Compute E-step' }).click();
    await checkText(em, /Responsibilities just computed, and about to be used for the M-step/);
    await checkText(em, /0\.98201379/);
    await checkText(em.locator('.gm-readout'), /Total log-likelihood -7\.158186977/);
    await em.getByRole('button', { name: 'Apply M-step' }).click();
    await checkText(em.locator('.gm-verdict'), /moved the log-likelihood by \+0\.696330676, so it rose/);
    await checkText(em.locator('.gm-readout'), /Total log-likelihood -6\.461856301/);
    await checkText(em, /-1\.344824658/);
    await checkText(em, /0\.691446639/);
    assert.equal(await em.locator('input[type="radio"]:checked').count(), 0, 'a graded cycle leaves the next question unset');
    await checkText(em, /mean before/);
    await checkText(em, /weighted scatter/);
    await em.getByRole('button', { name: 'Back one half-step' }).click();
    await checkText(em.locator('.gm-readout'), /Total log-likelihood -7\.158186977/);
    assert.equal(await em.locator('.gm-verdict').count(), 0, 'an undone cycle leaves no verdict behind');
    await screenshot(em, 'gmm-em-step-desktop.png');
    await em.getByRole('button', { name: 'Identical components' }).click();
    await em.getByLabel('It rises', { exact: true }).check();
    await em.getByRole('button', { name: 'Record it' }).click();
    await em.getByRole('button', { name: 'Compute E-step' }).click();
    await em.getByRole('button', { name: 'Apply M-step' }).click();
    await checkText(em.locator('.gm-verdict'), /stayed the same to within 1e-9.*Identical components give every row the same allocation/);
    await screenshot(em, 'gmm-em-null-desktop.png');
    await em.getByRole('button', { name: 'Repeated measurements' }).click();
    await em.getByLabel('It rises', { exact: true }).check();
    await em.locator('textarea').fill('The two duplicated positions leave almost no scatter, so the floor should bind.');
    await em.getByRole('button', { name: 'Record it' }).click();
    await em.getByRole('button', { name: 'Compute E-step' }).click();
    await em.getByRole('button', { name: 'Apply M-step' }).click();
    await checkText(em, /0\.25 \(floor\)/);
    await checkText(em, /0\.005363803/);
    await checkText(em, /Your reason, kept as you wrote it/);
    await screenshot(em, 'gmm-em-floor-desktop.png');
    await em.getByRole('button', { name: 'Standard data' }).click();
    await em.getByLabel(/^Measurement D/).fill('3');
    await checkText(em.locator('.gm-pending'), /Edited values are not in the model yet/);
    await em.getByRole('button', { name: 'Apply setup' }).click();
    await em.getByLabel('It rises', { exact: true }).check();
    await em.getByRole('button', { name: 'Record it' }).click();
    await em.getByRole('button', { name: 'Compute E-step' }).click();
    await em.getByRole('button', { name: 'Apply M-step' }).click();
    await checkText(em, /1\.844792261/);
    await checkText(em, /1\.582910448/);
    await em.getByRole('button', { name: 'Reset', exact: true }).click();
    for (let cycle = 0; cycle < 7; cycle += 1) {
      if (cycle === 6) {
        await em.getByLabel('It stays the same to within 1e-9', { exact: true }).check();
        await em.locator('textarea').fill('The separated fit has reached a numerical plateau.');
        await em.getByRole('button', { name: 'Record it' }).click();
        assert.ok(await em.locator('textarea').isDisabled(), 'the recorded reason cannot be overwritten');
      }
      await em.getByRole('button', { name: 'Compute E-step' }).click();
      await em.getByRole('button', { name: 'Apply M-step' }).click();
    }
    await checkText(em.locator('.gm-verdict'), /numerical plateau.*components differ/);
    await screenshot(em, 'gmm-em-plateau-desktop.png');
    for (let cycle = 7; cycle < 50; cycle += 1) {
      await em.getByRole('button', { name: 'Compute E-step' }).click();
      await em.getByRole('button', { name: 'Apply M-step' }).click();
    }
    const historyPlot = em.locator('.gm-plot').filter({ hasText: 'Total log-likelihood by iteration' });
    assert.equal(await historyPlot.locator('circle.gm-mark').count(), 51, 'all recorded iterations remain plotted');
    assert.deepEqual(await historyPlot.locator('svg > g > text').allTextContents(), ['0', '10', '20', '30', '40', '50'], 'late-cycle tick density stays readable without dropping history');
    assert.ok(await em.getByRole('button', { name: 'Compute E-step' }).isDisabled(), 'the bounded 50-cycle history stops advancing');
    await screenshot(historyPlot, 'gmm-em-history-desktop.png');
    await page.setViewportSize({ width: 320, height: 1000 });
    await screenshot(historyPlot, 'gmm-em-history-320.png');
    await page.setViewportSize({ width: 1366, height: 1000 });
    await em.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'EM lab: a recorded prediction graded only after the M-step, the exact first cycle including −1.344824658 and 0.691446639, a working back half-step, the identical-start null, the active variance floor, and the edited D = 3 setup giving 1.844792261' });

    // Coincident/nearby means must not superimpose component names. The key
    // reflows outside the numeric plot, before and after an actual EM cycle.
    for (const width of [1366, 320]) {
      await page.setViewportSize({ width, height: 1000 });
      for (const preset of ['Standard data', 'Asymmetric start', 'Identical components', 'Repeated measurements']) {
        await em.getByRole('button', { name: preset, exact: true }).click();
        for (const stage of ['initial', 'after one cycle']) {
          if (stage !== 'initial') {
            await em.getByRole('button', { name: 'Compute E-step', exact: true }).click();
            await em.getByRole('button', { name: 'Apply M-step', exact: true }).click();
          }
          await settle(page);
          const key = em.getByRole('list', { name: 'EM component curve key' });
          assert.equal(await key.getByRole('listitem').count(), 2);
          await checkText(key, /Left component · green, long dashes.*Right component · blue, short dashes/);
          assert.ok(await key.evaluate(element => element.scrollWidth <= element.clientWidth + 1), `EM key overflows: ${preset}, ${stage}, ${width}px`);
          const positionedNames = await em.locator('.gm-em-curves svg text').allTextContents();
          assert.ok(!positionedNames.some(text => text === 'Left' || text === 'Right'), 'component names must not be positioned at possibly coincident means');
          const issues = await page.evaluate(inspectLessonVisualLayout, '.gm-em-curves');
          assert.deepEqual(issues.flatMap(figure => figure.issues), [], `EM plot collides: ${preset}, ${stage}, ${width}px`);
          if (preset === 'Identical components' && stage !== 'initial') await screenshot(em.locator('.gm-em-curves'), `gmm-em-identical-key-${width}.png`);
        }
      }
    }
    await page.setViewportSize({ width: 1366, height: 1000 });
    await em.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'EM annotation regression: wrapping color/dash key remains separate from plotted means in all four presets, before and after a full cycle, at desktop and 320px; coincident component names never overlap' });

    // 3. Covariance geometry.
    const covariance = page.locator('.gm-investigation').nth(2);
    assert.equal(await covariance.getByRole('region', { name: 'Both panels, exactly' }).count(), 0, 'quadratic-form answers are not revealed before prediction');
    assert.doesNotMatch(await covariance.locator('svg').first().getAttribute('aria-label'), /Squared distances are/);
    assert.deepEqual(await covariance.locator('svg text').evaluateAll(items => items.filter(item => getComputedStyle(item).fill === 'rgb(0, 0, 0)').map(item => item.textContent)), [], 'covariance axes and mean labels need visible text fill on the dark plot');
    await covariance.getByLabel('Q has the higher density', { exact: true }).check();
    await covariance.getByRole('button', { name: 'Check prediction' }).click();
    await checkText(covariance.locator('.gm-verdict'), /You recorded Q has the higher density; the calculation gives P has the higher density.*1\.142857.*8/);
    await checkText(covariance, /0\.135882/);
    await checkText(covariance, /0\.004407/);
    await checkText(covariance, /39\.3469/);
    await covariance.getByRole('combobox', { name: 'Inspect projections of' }).selectOption('Q');
    await checkText(covariance.getByRole('region', { name: 'Resolve the selected point along the covariance axes' }), /0\.25/);
    await screenshot(covariance, 'gmm-covariance-desktop.png');
    await covariance.getByRole('spinbutton', { name: /^Correlation/ }).fill('-0.75');
    await checkText(covariance.locator('.gm-pending'), /Inputs changed; record a new prediction/);
    await covariance.getByLabel('Q has the higher density', { exact: true }).check();
    await covariance.getByRole('button', { name: 'Check prediction' }).click();
    await checkText(covariance.locator('.gm-verdict'), /Your prediction matches: Q has the higher density/);
    await covariance.getByRole('spinbutton', { name: /^Correlation/ }).fill('0');
    await covariance.getByLabel('They are equal', { exact: true }).check();
    await covariance.getByRole('button', { name: 'Check prediction' }).click();
    await checkText(covariance.locator('.gm-verdict'), /Your prediction matches: They are equal/);
    await covariance.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'Covariance lab: the 8/7 against 8 ordering with its densities, the reversed order at negative correlation, the equal-density null at zero correlation, and the 39.3469% unit contour stated instead of 68%' });

    const gallery = page.locator('.gm-figure').nth(3);
    await checkText(gallery, /shared by both: \[1, 0\.5; 0\.5, 1\]/);
    await checkText(gallery, /left: 1\.5 I/);
    await checkText(gallery, /left: \[2, 0\.75; 0\.75, 1\]/);
    const collapse = page.locator('.gm-figure').nth(2);
    await checkText(collapse, /Three standard deviations on one window/);
    await checkText(collapse, /The objective, against a logarithmic σ/);
    await checkText(collapse, /-8\.122121377/);
    await checkText(collapse, /-2\.369725898/);
    const clippedCollapsePoints = await collapse.locator('svg').first().locator('polyline').evaluateAll(lines => lines.flatMap(line => [...line.points].filter(point => point.y < 0 || point.y > 168).map(point => ({ x: point.x, y: point.y }))));
    assert.deepEqual(clippedCollapsePoints, [], 'all shrinking-component peaks must remain inside the drawn scale');
    const candidateFigure = page.locator('.gm-figure').nth(4);
    await checkText(candidateFigure, /-2\.668800/);
    await candidateFigure.getByText('Inspect all 150 observation IDs, measurements and split memberships', { exact: true }).click();
    assert.equal(await candidateFigure.getByRole('region', { name: 'The observed sepal measurements and their fixed evaluation roles' }).locator('tbody tr').count(), 150);
    await candidateFigure.getByText('Inspect all 150 observation IDs, measurements and split memberships', { exact: true }).click();
    records.push({ case: 'Figures carry their required content: the gallery prints every covariance, the collapse figure adds a shared window and a logarithmic objective strip, and the candidate table keeps six decimal places so the selected score matches the prose' });
    await screenshot(page.locator('.gm-figure').nth(0), 'gmm-selector-desktop.png');
    await screenshot(page.locator('.gm-figure').nth(1), 'gmm-allocation-desktop.png');
    await screenshot(page.locator('.gm-figure').nth(2), 'gmm-collapse-desktop.png');
    await screenshot(page.locator('.gm-figure').nth(3), 'gmm-covariance-gallery-desktop.png');
    await screenshot(page.locator('.gm-figure').nth(4), 'gmm-candidates-desktop.png');
    await screenshot(page.locator('.gm-figure').nth(5), 'gmm-bound-chain-desktop.png');

    // The reported allocation defect was inside a valid viewBox, so document
    // overflow alone could not catch it. Check glyph/mark geometry and reflow.
    const allocation = page.locator('.gm-allocation-figure');
    for (const width of [1366, 1024, 768, 390, 320]) {
      await page.setViewportSize({ width, height: 1000 });
      await settle(page);
      const issues = await page.evaluate(inspectLessonVisualLayout, '.gm-allocation-figure');
      assert.deepEqual(issues.flatMap(figure => figure.issues), [], `Allocation layout collides at ${width}px`);
      const components = await allocation.locator('.gm-allocation-components > section').evaluateAll(items => items.map(item => {
        const box = item.getBoundingClientRect(); return { width: box.width, left: box.left, top: box.top, bottom: box.bottom };
      }));
      assert.ok(Math.abs(components[0].width - components[1].width) < 1, 'component strips retain equal physical scale');
      if (width <= 390) assert.ok(components[1].top >= components[0].bottom, 'component explanations stack on phones');
      assert.equal(await allocation.locator('.gm-allocation-step').count(), 4, 'both components retain their before and after values');
      const columns = await allocation.locator('.gm-allocation-measurements g').evaluateAll(items => items.map(item => [...item.querySelectorAll('rect')].map(rect => Number(rect.getAttribute('height')))));
      assert.ok(columns.every(heights => heights.length === 2 && Math.abs(heights[0] + heights[1] - 54) < 1e-9), 'every allocation column still represents one observation');
      if (width === 320) await screenshot(allocation, 'gmm-allocation-320.png');
    }
    await page.setViewportSize({ width: 1366, height: 1000 });
    await page.evaluate(() => { document.documentElement.style.fontSize = '200%'; });
    await settle(page);
    assert.deepEqual((await page.evaluate(inspectLessonVisualLayout, '.gm-allocation-figure')).flatMap(figure => figure.issues), [], 'allocation remains separated with enlarged root text');
    await screenshot(allocation, 'gmm-allocation-enlarged-text.png');
    await page.evaluate(() => { document.documentElement.style.fontSize = ''; });
    records.push({ case: 'Allocation layout regression: distinct stages, four exact before/after strips, equal component scales, unit-mass columns, glyph/line separation at five widths and enlarged root text' });

    const localScripts = [...new Set(requests.map(address => new URL(address)).filter(address => address.origin === new URL(base).origin && address.pathname.endsWith('.js')).map(address => address.pathname.slice(1)))];
    assert.deepEqual(localScripts.filter(filename => bodyFiles.has(filename)), [bodyFile]);
    assert.deepEqual(localScripts.filter(filename => !allowedScripts.has(filename)), []);
    assert.ok(!requests.some(address => address.includes('/outlines/')));
    assert.ok(!requests.some(address => address.includes('iris.csv') && !address.includes('learn-assets')), 'the page never fetches the data to render');
    records.push({
      case: 'Fresh route requests only the selected lesson and its shared closure',
      requestedScripts: localScripts,
      rawScriptBytes: localScripts.reduce((sum, filename) => sum + fs.statSync(path.join(distDir, filename)).size, 0),
      gzipEstimateBytes: localScripts.reduce((sum, filename) => sum + gzipSync(fs.readFileSync(path.join(distDir, filename))).length, 0),
      interpretation: 'Built requested file sizes; the gzip estimate is not measured network compression or latency.',
    });

    for (const width of [390, 320]) {
      await page.setViewportSize({ width, height: 844 });
      await page.locator('.gm-lesson details').evaluateAll(items => items.forEach(item => { item.open = true; }));
      await settle(page);
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `Document overflows at ${width}px`);
      const mathOverflow = await page.locator('.gm-lesson .katex-display').evaluateAll(items => items.filter(item => item.scrollWidth > item.clientWidth + 1).map(item => item.textContent));
      assert.deepEqual(mathOverflow, [], `Overflowing formulas at ${width}px`);
      assert.equal(await page.locator('.katex-error').count(), 0);
      if (width === 390) {
        await screenshot(page.locator('.gm-figure').nth(1), 'gmm-allocation-mobile.png');
        await screenshot(page.locator('.gm-figure').nth(4), 'gmm-candidates-mobile.png');
        await screenshot(covariance, 'gmm-covariance-mobile.png');
        await screenshot(em, 'gmm-em-step-mobile.png');
      }
      records.push({ case: `Narrow ${width}px layout, readable formula grouping and every deeper branch rendered` });
    }
    await page.setViewportSize({ width: 1366, height: 1000 });
    await page.locator('.reader-complete').click();
    assert.equal(new URL(page.url()).pathname.split('/').at(-1), topicId);
    await page.reload({ waitUntil: 'domcontentloaded' });
    await ready(page);
    assert.equal(await page.evaluate(id => JSON.parse(localStorage.getItem('kd-progress'))[id], topicId), true);
    await page.locator('.reader-footer__next').click();
    await page.waitForURL('**/t-sne-umap-manifold-learning?module=classical-ml');
    await page.waitForFunction(() => document.querySelector('.reader-header h1')?.textContent.includes('t-SNE'), null, { timeout: 20000 });
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
      selection: { rule: selection.rule, selected: selection.selected, testMeanLogDensity: selection.testMeanLogDensity },
      records, screenshots: screenshotPaths,
      visualInterpretation: 'The intended Space Grotesk font loaded before the checks. Screenshots capture informative states: a matched and a missed prediction, the saved two-point contrast, the symmetric null, the active variance floor, and every inline figure. They require separate visual inspection.',
    };
    fs.writeFileSync(evidencePath, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify({ status: report.status, cases: records.length, evidencePath, sourceFiles: ownedFiles.length }));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });

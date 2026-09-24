// Production browser review of the anomaly-detection lesson: visible content,
// the six investigations, prediction/commit/reveal and stale recovery, narrow
// layouts, sequence, completion and load-failure recovery.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { gzipSync } = require('node:zlib');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');

const distDir = process.env.DIST_DIR || 'dist';
const base = (process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4173').replace(/\/+$/, '');
const topicId = 'anomaly-outlier-detection-isolation-forest-one-class-svm-lof';
const sourcePath = `src/learn/data/topics/${topicId}.jsx`;
const route = `${base}/learn/path/full-curriculum/${topicId}?module=classical-ml`;
const evidencePath = 'docs/teaching/evidence/anomaly-detection-browser.json';
const read = filename => JSON.parse(fs.readFileSync(filename, 'utf8'));
const hash = filename => createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
const normalize = text => text.replace(/\s+/g, ' ').trim();
const ownedFiles = [
  sourcePath,
  'src/learn/data/anomaly-detection-models.js',
  'src/learn/data/anomaly-temperature-data.js',
  'src/learn/data/anomaly-detection-examples.js',
  'src/learn/components/lesson-labs/AnomalyDetectionLabs.jsx',
  'src/learn/components/lesson-labs/AnomalyTemperatureLab.jsx',
  'src/learn/components/lesson-labs/AnomalyDetectionShared.jsx',
  'src/learn/components/lesson-labs/AnomalyDetectionFigures.jsx',
  'src/learn/components/lesson-labs/anomaly-detection-labs.css',
  `src/learn/data/curriculum/blueprints/${topicId}.js`,
  'public/learn-assets/anomaly-detection/machine_temperature_system_failure.csv',
  'public/learn-assets/anomaly-detection/nab-event-windows.json',
  'public/learn-assets/anomaly-detection/NAB-LICENSE.txt',
];

(async () => {
  const startedAt = new Date().toISOString();
  const sourceHashes = Object.fromEntries(ownedFiles.map(filename => [filename, hash(filename)]));
  const build = read(`${distDir}/.vite/manifest.json`);
  const buildHash = hash(`${distDir}/.vite/manifest.json`);
  const publications = read('src/learn/data/lesson-manifest.json');
  const { tracks } = await import('../src/learn/data/tracks.js');
  const { anomalyExamples } = await import('../src/learn/data/anomaly-detection-examples.js');
  const { rowCounts, publishedOutcomes } = await import('../src/learn/data/anomaly-temperature-data.js');
  const module = tracks.find(track => track.id === 'classical-ml');
  assert.equal(module.topicIds[14], topicId);
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
    await page.locator('.anomaly-lesson').waitFor();
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
    await checkText(page.locator('.reader-header h1'), /^Anomaly & Outlier Detection \(Isolation Forest, One-Class SVM, LOF\)$/);
    await checkText(page.locator('.reader-header__meta'), /15 of 39 topics on this route/);
    await checkText(page.locator('.reader-footer__previous'), /DBSCAN & Density-Based Clustering/);
    await checkText(page.locator('.reader-footer__next'), /Gaussian Mixture Models/);
    assert.equal(await page.locator('.ad-investigation').count(), 6);
    assert.equal(await page.locator('.ad-figure').count(), 4);
    assert.equal(await page.locator('.ad-practice').count(), 10);
    assert.equal(await page.locator('.python-example').count(), 5);
    const rendered = normalize(await page.locator('.anomaly-lesson').textContent());
    for (const [key, example] of Object.entries(anomalyExamples)) {
      assert.ok(rendered.includes(normalize(example.code)), `Missing actual code for ${key}`);
      assert.ok(rendered.includes(normalize(example.expected)), `Missing actual expected output for ${key}`);
    }
    for (const anchor of await page.locator('.lesson-intro a[href^="#"]').evaluateAll(items => items.map(item => item.getAttribute('href')))) {
      assert.equal(await page.locator(`[id="${anchor.slice(1)}"]`).count(), 1, `Broken lesson anchor ${anchor}`);
    }
    assert.equal(await page.locator('.katex-error').count(), 0);
    const csv = await page.request.get(`${base}/learn-assets/anomaly-detection/machine_temperature_system_failure.csv`);
    assert.equal(csv.status(), 200);
    assert.equal((await csv.text()).trim().split('\n').length, 22696, 'CSV serves the header and all 22,695 raw rows');
    const windows = await page.request.get(`${base}/learn-assets/anomaly-detection/nab-event-windows.json`);
    assert.equal(windows.status(), 200);
    assert.equal((await windows.json())['realKnownCause/machine_temperature_system_failure.csv'].length, 4);
    assert.equal((await page.request.get(`${base}/learn-assets/anomaly-detection/NAB-LICENSE.txt`)).status(), 200);
    records.push({ case: 'Complete visible code and output for five programs, thirteen route anchors, four figures, ten practice tasks, the three offline source files, current metadata and module sequence' });

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

    // 1. Isolation: expectation over every cut sequence.
    const isolation = page.locator('.ad-investigation').nth(0);
    await isolation.getByLabel('Predict first', { exact: false }).selectOption('P4');
    await isolation.getByRole('button', { name: 'Commit prediction' }).click();
    await isolation.getByRole('button', { name: 'Reveal the calculation' }).click();
    await checkText(isolation.locator('.ad-verdict'), /Your prediction matches: P4 at 12\..*normalizer is c\(5\) = 77\/30/);
    await checkText(isolation, /841\/660 ≈ 1\.2742/);
    await checkText(isolation, /0\.708845/);
    await isolation.getByLabel(/^P4 position/).fill('4');
    await checkText(isolation.locator('.ad-stale'), /inputs changed after your last prediction, so that answer is retired/);
    await isolation.getByRole('button', { name: 'Regular spacing' }).click();
    await isolation.getByLabel('Predict first', { exact: false }).selectOption('tie');
    await isolation.getByRole('button', { name: 'Commit prediction' }).click();
    await isolation.getByRole('button', { name: 'Reveal the calculation' }).click();
    await checkText(isolation.locator('.ad-verdict'), /matches: Two or more tie for shortest/);
    await isolation.getByRole('button', { name: 'Identical records' }).click();
    await isolation.getByLabel('Predict first', { exact: false }).selectOption('tie');
    await isolation.getByRole('button', { name: 'Commit prediction' }).click();
    await isolation.getByRole('button', { name: 'Reveal the calculation' }).click();
    await checkText(isolation.locator('.ad-verdict'), /no cut can separate anything.*all five corrected paths equal c\(5\).*scores are all 0\.5/);
    await isolation.getByRole('button', { name: 'One distant position' }).click();
    await isolation.locator('summary').click();
    await checkText(isolation.locator('.ad-readout'), /Corrected path .* normalized score/);
    await isolation.getByLabel(/^Depth cap/).selectOption('5');
    assert.equal(await isolation.getByLabel(/^Cut [1-5]$/).count(), 5, 'every offered depth has an editable cut');
    await isolation.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'Isolation lab: exact 77/30 normalizer and 841/660 expectation, stale invalidation on an edited position, the symmetric tie, the identical-record case where every score is one half, and one followed cut sequence' });

    // 2. LOF neighbourhoods: the farther query with the lower factor.
    const neighbourhood = page.locator('.ad-investigation').nth(1);
    await neighbourhood.getByLabel(/higher local outlier factor/).selectOption('four');
    await neighbourhood.getByRole('button', { name: 'Commit prediction' }).first().click();
    await neighbourhood.getByRole('button', { name: 'Reveal the calculation' }).first().click();
    await checkText(neighbourhood.locator('.ad-verdict').first(), /LOF\(4\) = 35\/24 and LOF\(17\) = 35\/32/);
    await checkText(neighbourhood, /max\(3, 8\)|8/);
    await neighbourhood.getByLabel(/will its factor be above/).selectOption('above');
    await neighbourhood.getByRole('button', { name: 'Commit prediction' }).nth(1).click();
    await neighbourhood.getByRole('button', { name: 'Reveal the calculation' }).nth(1).click();
    await checkText(neighbourhood.locator('.ad-verdict').nth(1), /LOF = 35\/24/);
    await neighbourhood.getByLabel(/^Neighbours k/).selectOption('3');
    assert.ok(await neighbourhood.locator('.ad-stale').count() >= 1, 'changing k retires both committed answers');
    await neighbourhood.getByLabel(/higher local outlier factor/).selectOption('equal');
    await neighbourhood.getByRole('button', { name: 'Commit prediction' }).first().click();
    await neighbourhood.getByRole('button', { name: 'Reveal the calculation' }).first().click();
    await checkText(neighbourhood.locator('.ad-verdict').first(), /matches: They tie.*mean-neighbour-density to query-density ratios agree/);
    await checkText(neighbourhood, /factor equal to query 4/);
    await screenshot(neighbourhood, 'anomaly-lof-tie-desktop.png');
    await neighbourhood.getByLabel(/^Reference P1/).fill('2');
    await checkText(neighbourhood.locator('.ad-note'), /requires distinct coordinates.*can have zero mean reach/);
    await neighbourhood.getByRole('button', { name: 'Reset', exact: true }).click();
    await neighbourhood.getByLabel(/^Reference P3/).fill('16');
    await checkText(neighbourhood.locator('.ad-prediction').first(), /Query 4 sits 2 units.*query 17 sits 1 units/);
    assert.equal((await neighbourhood.locator('select').first().innerText()).includes('closer'), false, 'choice labels do not keep the original distance ordering after edits');
    await neighbourhood.getByLabel(/higher local outlier factor/).selectOption('four');
    await neighbourhood.getByRole('button', { name: 'Commit prediction' }).first().click();
    await neighbourhood.getByRole('button', { name: 'Reveal the calculation' }).first().click();
    await checkText(neighbourhood.locator('.ad-verdict').first(), /LOF\(4\) = 35\/24 and LOF\(17\) = 11\/12/);
    await checkText(neighbourhood, /Query 17: nearest-reference distance 1, and a factor lower than query 4/);
    await neighbourhood.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'LOF lab: the closer query taking the higher factor with 35/24 against 35/32, the second prediction on the live query, staleness when k changes, and the refused duplicate coordinate' });

    // 3. Fitting mode at one coordinate.
    const fitting = page.locator('.ad-investigation').nth(2);
    await fitting.getByLabel('Predict first', { exact: false }).selectOption('different');
    await fitting.getByRole('button', { name: 'Commit prediction' }).click();
    await fitting.getByRole('button', { name: 'Reveal the calculation' }).click();
    await checkText(fitting.locator('.ad-verdict'), /reference row gives 4\/3 and the new query gives 7\/8/);
    await checkText(fitting.locator('.ad-readout'), /Reference factor 4\/3; new-query score 7\/8/);
    assert.equal(await fitting.locator('svg[role="img"]').count(), 2, 'both neighbourhood contracts have linked number-line strips');
    await fitting.getByLabel(/^Coordinate to compare/).selectOption('24');
    await checkText(fitting.locator('.ad-stale'), /answer is retired/);
    records.push({ case: 'Fitting-mode lab: 4/3 for the reference row against 7/8 for a new query at the same coordinate, retired when the coordinate changes' });

    // 4. Kernel boundary: a midpoint outside a region whose anchors are on it.
    const kernel = page.locator('.ad-investigation').nth(3);
    assert.equal(await kernel.locator('.ad-readout').count(), 0, 'kernel arithmetic is absent before prediction');
    assert.equal((await kernel.innerText()).includes('-0.141278'), false, 'kernel result does not leak through text');
    assert.equal((await kernel.locator('svg').getAttribute('aria-label')).includes('0.510'), false, 'kernel result does not leak through the accessible description');
    await kernel.getByLabel('Predict first', { exact: false }).selectOption('outside');
    await kernel.getByRole('button', { name: 'Commit prediction' }).click();
    await kernel.getByRole('button', { name: 'Reveal the calculation' }).click();
    await checkText(kernel.locator('.ad-verdict'), /matches: Outside: the decision is negative.*g\(0\) = -0\.141278/);
    await checkText(kernel.locator('.ad-readout'), /2 separated pieces/);
    await kernel.getByLabel(/^Kernel width gamma/).fill('0.1');
    await checkText(kernel.locator('.ad-stale'), /answer is retired/);
    assert.equal(await kernel.locator('.ad-readout').count(), 0, 'editing gamma hides the previous answer');
    await kernel.getByLabel('Predict first', { exact: false }).selectOption('inside');
    await kernel.getByRole('button', { name: 'Commit prediction' }).click();
    await kernel.getByRole('button', { name: 'Reveal the calculation' }).click();
    await checkText(kernel.locator('.ad-verdict'), /matches: Inside: the decision is positive.*g\(0\) = 0\.069677/);
    await checkText(kernel.locator('.ad-readout'), /accepted region is the single interval/);
    await kernel.getByRole('button', { name: 'Put the query on a reference point' }).click();
    await kernel.getByLabel('Predict first', { exact: false }).selectOption('boundary');
    await kernel.getByRole('button', { name: 'Commit prediction' }).click();
    await kernel.getByRole('button', { name: 'Reveal the calculation' }).click();
    await checkText(kernel.locator('.ad-verdict'), /matches: Exactly on the boundary/);
    await kernel.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'Kernel lab: the midpoint outside at gamma 1 with a disconnected accepted region, inside at gamma 0.1 with one interval, and a reference anchor exactly on the boundary at both' });

    // 5. The review queue.
    const queue = page.locator('.ad-investigation').nth(4);
    await queue.getByLabel(/What fraction of the alerts will be faults/).selectOption('low');
    await queue.getByRole('button', { name: 'Commit prediction' }).first().click();
    assert.ok(await queue.getByRole('button', { name: 'Reveal the calculation' }).isDisabled(), 'precision cannot disclose the still-uncommitted workload answer');
    assert.equal(await queue.locator('.ad-bars').count(), 0, 'no counts before both predictions');
    await queue.getByLabel(/And will the review budget/).selectOption('short');
    await queue.getByRole('button', { name: 'Commit prediction' }).nth(1).click();
    await queue.getByRole('button', { name: 'Reveal the calculation' }).click();
    await checkText(queue.locator('.ad-verdict').first(), /Expected alerts 1079 of which 80 are faults: 7\.41%/);
    await queue.getByRole('button', { name: 'Reveal the workload' }).click();
    await checkText(queue.locator('.ad-verdict').nth(1), /matches: There are more alerts than slots.*short by 879/);
    await checkText(queue, /7\.4143%/);
    await queue.getByLabel(/^Fault prevalence/).fill('10');
    await checkText(queue.locator('.ad-stale').first(), /answer is retired/);
    await queue.getByLabel(/What fraction of the alerts will be faults/).selectOption('high');
    await queue.getByRole('button', { name: 'Commit prediction' }).first().click();
    await queue.getByLabel(/And will the review budget/).selectOption('short');
    await queue.getByRole('button', { name: 'Commit prediction' }).nth(1).click();
    await queue.getByRole('button', { name: 'Reveal the calculation' }).click();
    await checkText(queue.locator('.ad-verdict').first(), /matches: More than half the alerts are faults/);
    await queue.getByLabel(/^Sensitivity: faults flagged/).fill('0');
    await queue.getByLabel(/^False-positive rate/).fill('0');
    await queue.getByLabel(/What fraction of the alerts will be faults/).selectOption('none');
    await queue.getByRole('button', { name: 'Commit prediction' }).first().click();
    await queue.getByLabel(/And will the review budget/).selectOption('covers');
    await queue.getByRole('button', { name: 'Commit prediction' }).nth(1).click();
    await queue.getByRole('button', { name: 'Reveal the calculation' }).click();
    await checkText(queue.locator('.ad-verdict').first(), /nothing is flagged, so precision is undefined rather than zero or one/);
    await queue.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'Review-queue lab: 1,079 alerts at 7.4143% precision, precision rising with prevalence alone, and an undefined precision when nobody is flagged' });

    // 6. The real series.
    const temperature = page.locator('.ad-investigation').nth(5);
    const isolation975 = publishedOutcomes.isolation['0.975'];
    await temperature.locator('summary').first().click();
    await checkText(temperature, /Rows in this block: 711, 60 shown on this page/);
    await checkText(temperature, /Rows 1–60 of 711/);
    const firstTimestamp = await temperature.locator('.ad-rows tbody tr').first().locator('td').first().innerText();
    await temperature.getByRole('button', { name: 'Next rows', exact: true }).click();
    await checkText(temperature, /Rows 61–120 of 711/);
    assert.notEqual(await temperature.locator('.ad-rows tbody tr').first().locator('td').first().innerText(), firstTimestamp);
    for (let pageIndex = 1; pageIndex < 11; pageIndex += 1) await temperature.getByRole('button', { name: 'Next rows', exact: true }).click();
    await checkText(temperature, /Rows 661–711 of 711/);
    assert.ok(await temperature.getByRole('button', { name: 'Next rows', exact: true }).isDisabled());
    await temperature.getByRole('button', { name: 'Previous rows', exact: true }).focus();
    await page.keyboard.press('Enter');
    await checkText(temperature, /Rows 601–660 of 711/);
    assert.equal(await temperature.getByRole('option', { name: 'No unmatched alerts', exact: true }).count(), 1);
    assert.ok(await temperature.getByRole('button', { name: 'Show every row in this block' }).isDisabled(), 'the alert filter waits for the reveal');
    assert.equal((await temperature.innerText()).includes('Alerting rows in this block'), false, 'no alert count before the prediction');
    await screenshot(temperature, 'anomaly-temperature-uncommitted-desktop.png');
    await temperature.locator('summary').first().click();
    await temperature.getByLabel(/^How many of the four annotated windows/).selectOption('4');
    await temperature.getByLabel(/^How many alerts land outside/).selectOption('large');
    await temperature.getByRole('button', { name: 'Commit both predictions' }).click();
    await temperature.getByRole('button', { name: "Reveal this threshold's result" }).click();
    await checkText(temperature.locator('.ad-verdict'), /Windows with an alert: 4 of 4, you said 4\. Alerts outside every window: 4,263/);
    await checkText(temperature.locator('.ad-readout'), new RegExp(`leaves 28 calibration alerts out of 1,152, and ${isolation975[2].toLocaleString('en-US')} alerts out of 20,634 later rows`));
    await checkText(temperature, /One-Class SVM/);
    await temperature.getByLabel(/^Method/).selectOption('baseline');
    await checkText(temperature.locator('.ad-stale'), /method or threshold changed after your last prediction/);
    await temperature.getByLabel(/^How many of the four annotated windows/).selectOption('4');
    await temperature.getByLabel(/^How many alerts land outside/).selectOption('medium');
    await temperature.getByRole('button', { name: 'Commit both predictions' }).click();
    await temperature.getByRole('button', { name: "Reveal this threshold's result" }).click();
    await checkText(temperature.locator('.ad-verdict'), /Alerts outside every window: 1,036/);
    await temperature.getByLabel(/^How the threshold is set/).selectOption('direct');
    await checkText(temperature.locator('.ad-caption').first(), /Typed values snap to the nearest of 242 precomputed levels/);
    await temperature.getByLabel(/^Threshold on the score scale/).fill('6');
    await temperature.getByLabel(/^How many of the four annotated windows/).selectOption('4');
    await temperature.getByLabel(/^How many alerts land outside/).selectOption('small');
    await temperature.getByRole('button', { name: 'Commit both predictions' }).click();
    await temperature.getByRole('button', { name: "Reveal this threshold's result" }).click();
    await checkText(temperature.locator('.ad-readout'), /snapped to the nearest available level/);
    await temperature.locator('summary').first().click();
    await checkText(temperature, /Window 1 of 4: 2013-12-10 06:25 to 2013-12-12 05:35/);
    await temperature.getByRole('button', { name: 'Next window' }).click();
    await checkText(temperature, /Window 2 of 4: 2013-12-15 17:50 to 2013-12-17 17:00/);
    await temperature.getByRole('button', { name: 'Show every row in this block' }).click();
    await temperature.locator('summary').nth(1).click();
    await checkText(temperature, /All four windows are hit in all eight rows/);
    await checkText(temperature, /7,913/);
    await temperature.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'Temperature lab: the 0.975 isolation threshold with 4,263 unmatched alerts and four window hits, the baseline at 1,036, a typed threshold snapped to a precomputed level, window paging against the annotation timestamps, and the published eight-row table' });

    await isolation.getByLabel('Predict first', { exact: false }).selectOption('P4');
    await isolation.getByRole('button', { name: 'Commit prediction' }).click();
    await isolation.getByRole('button', { name: 'Reveal the calculation' }).click();
    await screenshot(isolation, 'anomaly-isolation-desktop.png');
    await neighbourhood.getByLabel(/higher local outlier factor/).selectOption('seventeen');
    await neighbourhood.getByRole('button', { name: 'Commit prediction' }).first().click();
    await neighbourhood.getByRole('button', { name: 'Reveal the calculation' }).first().click();
    await screenshot(neighbourhood, 'anomaly-lof-desktop.png');
    await kernel.getByLabel('Predict first', { exact: false }).selectOption('outside');
    await kernel.getByRole('button', { name: 'Commit prediction' }).click();
    await kernel.getByRole('button', { name: 'Reveal the calculation' }).click();
    await screenshot(kernel, 'anomaly-kernel-desktop.png');
    await queue.getByLabel(/What fraction of the alerts will be faults/).selectOption('low');
    await queue.getByRole('button', { name: 'Commit prediction' }).first().click();
    await queue.getByLabel(/And will the review budget/).selectOption('short');
    await queue.getByRole('button', { name: 'Commit prediction' }).nth(1).click();
    await queue.getByRole('button', { name: 'Reveal the calculation' }).click();
    await queue.getByRole('button', { name: 'Reveal the workload' }).click();
    await screenshot(queue, 'anomaly-queue-desktop.png');
    await temperature.getByLabel(/^How many of the four annotated windows/).selectOption('4');
    await temperature.getByLabel(/^How many alerts land outside/).selectOption('large');
    await temperature.getByRole('button', { name: 'Commit both predictions' }).click();
    await temperature.getByRole('button', { name: "Reveal this threshold's result" }).click();
    await screenshot(temperature, 'anomaly-temperature-desktop.png');
    await screenshot(page.locator('.ad-figure').nth(0), 'anomaly-provenance-desktop.png');
    await checkText(page.locator('.ad-figure').nth(0), /annotations enter evaluation only/);
    await screenshot(page.locator('.ad-figure').nth(1), 'anomaly-first-cut-desktop.png');
    await screenshot(page.locator('.ad-figure').nth(2), 'anomaly-reach-floor-desktop.png');
    await screenshot(page.locator('.ad-figure').nth(3), 'anomaly-threshold-ruler-desktop.png');
    await fitting.getByRole('button', { name: 'Reset', exact: true }).click();
    await fitting.getByLabel('Predict first', { exact: false }).selectOption('different');
    await fitting.getByRole('button', { name: 'Commit prediction' }).click();
    await fitting.getByRole('button', { name: 'Reveal the calculation' }).click();
    await screenshot(fitting, 'anomaly-fitting-mode-desktop.png');
    await page.setViewportSize({ width: 320, height: 1000 });
    await screenshot(fitting, 'anomaly-fitting-mode-revealed-320.png');
    await page.setViewportSize({ width: 1366, height: 1000 });

    const localScripts = [...new Set(requests.map(address => new URL(address)).filter(address => address.origin === new URL(base).origin && address.pathname.endsWith('.js')).map(address => address.pathname.slice(1)))];
    assert.deepEqual(localScripts.filter(filename => bodyFiles.has(filename)), [bodyFile]);
    assert.deepEqual(localScripts.filter(filename => !allowedScripts.has(filename)), []);
    assert.ok(!requests.some(address => address.includes('/outlines/')));
    assert.ok(!requests.some(address => address.includes('machine_temperature_system_failure.csv') && !address.includes('learn-assets')), 'the lesson never fetches the raw series to render');
    records.push({
      case: 'Fresh route requests only the selected lesson and its shared closure',
      requestedScripts: localScripts,
      rawScriptBytes: localScripts.reduce((sum, filename) => sum + fs.statSync(path.join(distDir, filename)).size, 0),
      gzipEstimateBytes: localScripts.reduce((sum, filename) => sum + gzipSync(fs.readFileSync(path.join(distDir, filename))).length, 0),
      interpretation: 'Built requested file sizes; the gzip estimate is not measured network compression or latency.',
    });

    for (const width of [390, 320]) {
      await page.setViewportSize({ width, height: 844 });
      await page.locator('.anomaly-lesson details').evaluateAll(items => items.forEach(item => { item.open = true; }));
      await settle(page);
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `Document overflows at ${width}px`);
      const mathOverflow = await page.locator('.anomaly-lesson .katex-display').evaluateAll(items => items.filter(item => item.scrollWidth > item.clientWidth + 1).map(item => item.textContent));
      assert.deepEqual(mathOverflow, [], `Overflowing formulas at ${width}px`);
      assert.equal(await page.locator('.katex-error').count(), 0);
      if (width === 390) {
        await screenshot(page.locator('.ad-figure').nth(0), 'anomaly-provenance-mobile.png');
        await screenshot(isolation, 'anomaly-isolation-mobile.png');
        await screenshot(neighbourhood, 'anomaly-lof-mobile.png');
        await screenshot(page.locator('.ad-figure').nth(1), 'anomaly-first-cut-mobile.png');
        await screenshot(page.locator('.ad-figure').nth(3), 'anomaly-threshold-ruler-mobile.png');
        await screenshot(temperature, 'anomaly-temperature-mobile.png');
        await screenshot(kernel, 'anomaly-kernel-mobile.png');
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
    await page.waitForURL('**/gaussian-mixture-models-gmm-em-algorithm?module=classical-ml');
    await page.waitForFunction(() => document.querySelector('.reader-header h1')?.textContent.includes('Gaussian Mixture'), null, { timeout: 20000 });
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
      moduleTopicCount: module.topicIds.length, seriesRows: rowCounts.total, records,
      screenshots: screenshotPaths,
      visualInterpretation: 'The intended Space Grotesk font loaded before the checks. Screenshots capture informative states: a revealed isolation expectation, a missed LOF prediction, the disconnected kernel region, the review queue and the real-series threshold with its window bands. They require separate visual inspection.',
    };
    fs.writeFileSync(evidencePath, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify({ status: report.status, cases: records.length, evidencePath, sourceFiles: ownedFiles.length }));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });

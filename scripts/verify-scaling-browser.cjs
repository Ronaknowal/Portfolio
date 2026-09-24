// Production browser review of the feature-preparation lesson: visible content,
// the four investigations, prediction/commit/retirement, the frozen fitted
// bundle, narrow layouts, sequence, completion and load-failure recovery.
//
// Read-only by default, so an independent reviewer can re-run it inside a
// no-write boundary. Pass --write to refresh the evidence file and the
// screenshots.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { gzipSync } = require('node:zlib');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const { inspectLessonVisualLayout } = require('./lib/lesson-visual-layout.cjs');

const distDir = process.env.DIST_DIR || 'dist';
const base = (process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4173').replace(/\/+$/, '');
const topicId = 'feature-scaling-encoding-imputation';
const sourcePath = `src/learn/data/topics/${topicId}.jsx`;
const route = `${base}/learn/path/full-curriculum/${topicId}?module=classical-ml`;
const evidencePath = 'docs/teaching/evidence/scaling-browser.json';
const write = process.argv.includes('--write');
const read = filename => JSON.parse(fs.readFileSync(filename, 'utf8'));
const hash = filename => createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
const normalize = text => text.replace(/\s+/g, ' ').trim();
const ownedFiles = [
  sourcePath,
  'src/learn/data/scaling-models.js',
  'src/learn/data/scaling-data.js',
  'src/learn/data/scaling-examples.js',
  'src/learn/components/lesson-labs/ScalingShared.jsx',
  'src/learn/components/lesson-labs/ScalingLabs.jsx',
  'src/learn/components/lesson-labs/ScalingFigures.jsx',
  'src/learn/components/lesson-labs/scaling-labs.css',
  `src/learn/data/curriculum/blueprints/${topicId}.js`,
  'public/learn-assets/feature-scaling/penguins.csv',
];

(async () => {
  const startedAt = new Date().toISOString();
  const sourceHashes = Object.fromEntries(ownedFiles.map(filename => [filename, hash(filename)]));
  const build = read(`${distDir}/.vite/manifest.json`);
  const buildHash = hash(`${distDir}/.vite/manifest.json`);
  const publications = read('src/learn/data/lesson-manifest.json');
  const { tracks } = await import('../src/learn/data/tracks.js');
  const { scalingExamples } = await import('../src/learn/data/scaling-examples.js');
  const { comparison, fitted, provenance, scaleFixture, split } = await import('../src/learn/data/scaling-data.js');
  const module = tracks.find(track => track.id === 'classical-ml');
  assert.equal(module.topicIds[19], topicId);
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
    await page.locator('.sc-lesson').waitFor();
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
    await checkText(page.locator('.reader-header h1'), /^Feature Scaling, Encoding & Imputation$/);
    await checkText(page.locator('.reader-header__meta'), /20 of 39 topics on this route/);
    await checkText(page.locator('.reader-footer__previous'), /Non-Negative Matrix Factorization/);
    await checkText(page.locator('.reader-footer__next'), /Cross-Validation & Hyperparameter Tuning/);
    assert.equal(await page.locator('.sc-investigation').count(), 4);
    assert.equal(await page.locator('.sc-figure').count(), 9);
    assert.equal(await page.locator('.sc-practice').count(), 9);
    assert.equal(await page.locator('.python-example').count(), 2);
    const rendered = normalize(await page.locator('.sc-lesson').textContent());
    for (const [key, example] of Object.entries(scalingExamples)) {
      assert.ok(rendered.includes(normalize(example.code)), `Missing actual code for ${key}`);
      assert.ok(rendered.includes(normalize(example.expected)), `Missing actual expected output for ${key}`);
    }
    for (const anchor of await page.locator('.lesson-intro a[href^="#"]').evaluateAll(items => items.map(item => item.getAttribute('href')))) {
      assert.equal(await page.locator(`[id="${anchor.slice(1)}"]`).count(), 1, `Broken lesson anchor ${anchor}`);
    }
    assert.equal(await page.locator('.katex-error').count(), 0);
    const csv = await page.request.get(`${base}/learn-assets/feature-scaling/penguins.csv`);
    assert.equal(csv.status(), 200);
    const csvBody = await csv.body();
    assert.equal(createHash('sha256').update(csvBody).digest('hex'), provenance.sha256, 'the served CSV is the checkpointed file');
    const csvText = csvBody.toString('utf8');
    assert.equal(csvText.trim().split('\n').length, 345, 'the CSV serves a header and 344 rows');
    assert.ok(csvText.startsWith('species,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex,year'));
    assert.ok(csvText.includes(',NA,'), 'missing values are served as NA, not as zeros');
    // Every recorded number the prose quotes is on the page.
    for (const row of comparison) assert.ok(rendered.includes(`${row.correct} / 86`), `${row.method} count is shown`);
    for (const kind of ['standard', 'minmax', 'robust']) {
      assert.ok(rendered.includes(scaleFixture[kind].new150.toFixed(4)), `the later 150 under ${kind}`);
    }
    fitted.standard.center.forEach(value => assert.ok(rendered.includes(value.toFixed(4)), `fitted centre ${value}`));
    fitted.standard.scale.forEach(value => assert.ok(rendered.includes(value.toFixed(4)), `fitted scale ${value}`));
    assert.ok(rendered.includes('[45.0,17.3,197.0,4000.0]'), 'the fitted medians');
    assert.ok(rendered.includes('[51.0,18.8,203.0,4100.0,"male"]'), 'the first held-out record');
    assert.ok(rendered.includes(provenance.sha256), 'the dataset hash travels with the lesson');
    assert.ok(rendered.includes('CC0') || rendered.includes('Palmer Penguins'), 'the dataset is attributed');
    assert.ok(rendered.includes('Two rows lack each of the four measurements; eleven lack recorded sex'), 'the stated missing counts');
    assert.ok(rendered.includes('One extra correct row does not establish'), 'the no-winner caveat survives');
    assert.ok(rendered.includes('it does not certify that an estimated value was physically measured'), 'the imputation honesty caveat survives');
    assert.ok(rendered.includes('does not turn a skewed distribution into a Gaussian distribution'), 'the standardization caveat survives');
    assert.ok(rendered.includes('Adding an indicator does not, by itself, solve MNAR'), 'the MNAR caveat survives');
    assert.ok(rendered.includes('Every investigation asks for a prediction before it shows an answer'), 'the intro promises what the page keeps');
    assert.ok(!/accepts these exact coordinates/.test(rendered), 'no practice claims a lab accepts inputs it refuses');
    assert.ok(rendered.includes('its fields are bounded to plausible penguin measurements'), 'practice 1 says what the lab actually does');
    assert.equal((rendered.match(/Before running:/g) ?? []).length, 2, 'one Before running per program');
    records.push({ case: 'Complete visible code and output for two programs, eleven route anchors, nine figures, nine practice tasks, every recorded count and fitted statistic, the served CSV with its checkpointed hash, current metadata and module sequence' });

    const screenshot = async (locator, filename) => {
      const destination = path.join('docs/teaching/evidence/screenshots', filename);
      if (!write) { screenshotPaths.push(destination); return; }
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

    // 1. Choose the ruler.
    const ruler = page.locator('.sc-investigation').nth(0);
    assert.equal(await ruler.locator('input[type="radio"]:checked').count(), 0, 'no prediction is preselected');
    assert.ok(await ruler.getByRole('button', { name: 'Check prediction' }).isDisabled(), 'checking waits for a choice');
    assert.equal(await ruler.locator('.sc-table-scroll').count(), 0, 'no contribution table before a committed prediction');
    await ruler.getByLabel('B is nearer', { exact: true }).check();
    await ruler.getByRole('button', { name: 'Check prediction' }).click();
    await checkText(ruler.locator('.sc-verdict'), /Your prediction matches: B is nearer\./);
    await checkText(ruler, /10001/);
    await checkText(ruler, /\b10\b/);
    await screenshot(ruler, 'scaling-ruler-raw-desktop.png');
    await ruler.getByRole('button', { name: 'Declared: 1 mm, 100 g' }).click();
    await checkText(ruler.locator('.sc-pending'), /Inputs changed; record a new prediction/);
    assert.equal(await ruler.locator('input[type="radio"]:checked').count(), 0, 'the edit retired the choice');
    await checkText(ruler.locator('.sc-previous'), /Previous trial, kept for comparison and not current evidence/);
    await ruler.getByLabel('A is nearer', { exact: true }).check();
    await ruler.getByRole('button', { name: 'Check prediction' }).click();
    await checkText(ruler.locator('.sc-verdict'), /Your prediction matches: A is nearer\./);
    await checkText(ruler, /9\.0001/);
    await screenshot(ruler, 'scaling-ruler-scaled-desktop.png');
    // Contrast: A's mass moves to 4,400 g and B wins again.
    await ruler.getByRole('spinbutton', { name: /^A body mass/ }).fill('4400');
    await ruler.getByLabel('A is nearer', { exact: true }).check();
    await ruler.getByRole('button', { name: 'Check prediction' }).click();
    await checkText(ruler.locator('.sc-verdict'), /You recorded A is nearer; the calculation gives B is nearer/);
    await checkText(ruler, /\b17\b/);
    // Null: doubling both divisors changes every total by a common factor only.
    await ruler.getByRole('button', { name: 'Reset', exact: true }).click();
    await ruler.getByRole('button', { name: 'Both divisors doubled: 2 mm, 200 g' }).click();
    await ruler.getByLabel('A is nearer', { exact: true }).check();
    await ruler.getByRole('button', { name: 'Check prediction' }).click();
    await checkText(ruler.locator('.sc-verdict'), /Your prediction matches: A is nearer/);
    await checkText(ruler, /2\.250025/);
    await checkText(ruler, /0\.5/);
    // Null: translate everything by one vector.
    await ruler.getByRole('button', { name: 'Reset', exact: true }).click();
    await ruler.getByRole('button', { name: 'Declared: 1 mm, 100 g' }).click();
    await ruler.getByRole('button', { name: /^Translate everything/ }).click();
    await ruler.getByLabel('A is nearer', { exact: true }).check();
    await ruler.getByRole('button', { name: 'Check prediction' }).click();
    await checkText(ruler.locator('.sc-verdict'), /Your prediction matches: A is nearer/);
    await checkText(ruler, /9\.0001/);
    await ruler.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'Ruler lab: unset prediction, the 10,001 versus 10 raw totals, the 2 versus 9.0001 flip, the 4,400 g contrast giving 17, the doubled-divisor null at 0.5 and 2.250025, a translation null, a retired choice and a kept previous trial' });

    // A graded verdict cannot be destroyed by the ungraded path in any lab.
    for (const [index, name] of [[0, 'ruler'], [1, 'donor'], [2, 'pipeline'], [3, 'encoding']]) {
      const lab = page.locator('.sc-investigation').nth(index);
      assert.ok(await lab.getByRole('button', { name: 'Calculate without recording a prediction' }).isDisabled() === false,
        `${name}: the ungraded path is open before a result stands`);
    }
    await ruler.getByLabel('B is nearer', { exact: true }).check();
    await ruler.getByRole('button', { name: 'Check prediction' }).click();
    assert.ok(await ruler.getByRole('button', { name: 'Calculate without recording a prediction' }).isDisabled(),
      'the ungraded path closes once a verdict stands');
    await ruler.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'The ungraded calculate path is open before a commit and closed after one in all four investigations, so a graded verdict cannot be silently replaced' });

    // 2. Donor eligibility.
    const donor = page.locator('.sc-investigation').nth(1);
    assert.equal(await donor.locator('.sc-verdict').count(), 0, 'no result before a prediction');
    await donor.getByRole('checkbox', { name: 'D1 donates' }).check();
    await donor.getByRole('checkbox', { name: 'D2 donates' }).check();
    await donor.getByRole('spinbutton', { name: 'Predicted estimate' }).fill('200');
    await donor.getByRole('button', { name: 'Check prediction' }).click();
    await checkText(donor.locator('.sc-verdict'), /uses D1 and D2 and gives 200/);
    await checkText(donor, /7\.5/);
    await checkText(donor, /\b12\b/);
    await screenshot(donor, 'scaling-donor-desktop.png');
    // Contrast: D2 loses its target measurement.
    await donor.getByRole('spinbutton', { name: 'D2 c' }).fill('');
    await donor.getByRole('checkbox', { name: 'D1 donates' }).check();
    await donor.getByRole('checkbox', { name: 'D3 donates' }).check();
    await donor.getByRole('spinbutton', { name: 'Predicted estimate' }).fill('300');
    await donor.getByRole('button', { name: 'Check prediction' }).click();
    await checkText(donor.locator('.sc-verdict'), /uses D1 and D3 and gives 300/);
    await screenshot(donor, 'scaling-donor-removed-desktop.png');
    // Separate contrast: D1's target moves to 140.
    await donor.getByRole('button', { name: 'Reset', exact: true }).click();
    await donor.getByRole('spinbutton', { name: 'D1 c' }).fill('140');
    await donor.getByRole('checkbox', { name: 'D1 donates' }).check();
    await donor.getByRole('checkbox', { name: 'D2 donates' }).check();
    await donor.getByRole('spinbutton', { name: 'Predicted estimate' }).fill('220');
    await donor.getByRole('button', { name: 'Check prediction' }).click();
    await checkText(donor.locator('.sc-verdict'), /uses D1 and D2 and gives 220/);
    // Null: an unselected donor's target changes nothing.
    await donor.getByRole('button', { name: 'Reset', exact: true }).click();
    await donor.getByRole('spinbutton', { name: 'D3 c' }).fill('900');
    await donor.getByRole('checkbox', { name: 'D1 donates' }).check();
    await donor.getByRole('checkbox', { name: 'D2 donates' }).check();
    await donor.getByRole('spinbutton', { name: 'Predicted estimate' }).fill('200');
    await donor.getByRole('button', { name: 'Check prediction' }).click();
    await checkText(donor.locator('.sc-verdict'), /uses D1 and D2 and gives 200/);
    // Fixture: no overlap at all falls back to the observed column mean, labelled.
    await donor.getByRole('button', { name: 'Reset', exact: true }).click();
    await donor.getByRole('spinbutton', { name: 'query a' }).fill('');
    await donor.getByRole('spinbutton', { name: 'query b' }).fill('');
    await donor.getByRole('button', { name: 'Calculate without recording a prediction' }).click();
    await checkText(donor, /fallback is the mean of the observed values/);
    await checkText(donor, /\b300\b/);
    await screenshot(donor, 'scaling-donor-fallback-desktop.png');
    // Fixture: an entirely absent target column is reported, never filled.
    await donor.getByRole('button', { name: 'Reset', exact: true }).click();
    for (const name of ['D1 c', 'D2 c', 'D3 c']) await donor.getByRole('spinbutton', { name }).fill('');
    await donor.getByRole('checkbox', { name: 'No eligible data' }).check();
    await donor.getByRole('button', { name: 'Check prediction' }).click();
    await checkText(donor, /cannot estimate/);
    await screenshot(donor, 'scaling-donor-empty-column-desktop.png');
    await donor.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'Donor lab: the 7.5 / 3 / 12 overlap distances giving 200, the removed-target contrast at 300, the moved-target contrast at 220, an unselected-donor null, the labelled no-overlap fallback 300 and the refusal to invent a value for an entirely absent target column' });

    // 3. One held-out record through the frozen bundle.
    const pipeline = page.locator('.sc-investigation').nth(2);
    await checkText(pipeline, /observed source row 309/);
    await pipeline.getByRole('spinbutton', { name: 'Predicted coordinate value' }).fill('1.309');
    await pipeline.getByRole('button', { name: 'Check prediction' }).click();
    await checkText(pipeline.locator('.sc-verdict'), /the calculation gives 1\.3091367505/);
    await checkText(pipeline, /45 − 43\.93062|43\.93062/);
    await screenshot(pipeline, 'scaling-pipeline-desktop.png');
    // Contrast: delete the mass and the fitted median fills it.
    await pipeline.getByRole('combobox', { name: 'Output coordinate to predict' }).selectOption('3');
    await pipeline.getByRole('spinbutton', { name: 'body mass (g)' }).fill('');
    await checkText(pipeline, /edited copy — not a measured specimen/);
    await pipeline.getByRole('spinbutton', { name: 'Predicted coordinate value' }).fill('-0.2368');
    await pipeline.getByRole('button', { name: 'Check prediction' }).click();
    await checkText(pipeline.locator('.sc-verdict'), /Your prediction matches|the calculation gives -0\.2367563058/);
    await checkText(pipeline, /4000 \(training median\)/);
    await screenshot(pipeline, 'scaling-pipeline-imputed-desktop.png');
    // Null: editing the mass leaves the bill-length coordinate where it was.
    await pipeline.getByRole('combobox', { name: 'Output coordinate to predict' }).selectOption('0');
    await pipeline.getByRole('spinbutton', { name: 'Predicted coordinate value' }).fill('1.3091');
    await pipeline.getByRole('button', { name: 'Check prediction' }).click();
    await checkText(pipeline.locator('.sc-verdict'), /the calculation gives 1\.3091367505/);
    // Category contrasts, including an unknown value and an absent one.
    await pipeline.getByRole('button', { name: 'Reset', exact: true }).click();
    await pipeline.getByRole('combobox', { name: 'Output coordinate to predict' }).selectOption('4');
    await pipeline.getByRole('combobox', { name: 'Recorded sex' }).selectOption('female');
    await pipeline.getByLabel('[1, 0, 0]', { exact: true }).check();
    await pipeline.getByRole('button', { name: 'Check prediction' }).click();
    await checkText(pipeline.locator('.sc-verdict'), /Your prediction matches/);
    await pipeline.getByRole('combobox', { name: 'Recorded sex' }).selectOption('juvenile — a value the fit never saw');
    await pipeline.getByLabel('[0, 0, 0]', { exact: true }).check();
    await pipeline.getByRole('button', { name: 'Check prediction' }).click();
    await checkText(pipeline.locator('.sc-verdict'), /Your prediction matches/);
    await checkText(pipeline, /not in the fitted vocabulary/);
    await pipeline.getByRole('combobox', { name: 'Recorded sex' }).selectOption('absent (not recorded)');
    await pipeline.getByLabel('[0, 0, 1]', { exact: true }).check();
    await pipeline.getByRole('button', { name: 'Check prediction' }).click();
    await checkText(pipeline.locator('.sc-verdict'), /Your prediction matches/);
    await checkText(pipeline, /the fitted not_recorded category absorbs it/);
    await screenshot(pipeline, 'scaling-pipeline-category-desktop.png');
    // The fitted statistics never move, however the record is edited.
    const frozen = await pipeline.locator('.sc-table-scroll').first().innerText();
    await pipeline.getByRole('button', { name: 'Reset', exact: true }).click();
    await pipeline.getByRole('combobox', { name: /^Held-out record/ }).selectOption('142');
    await pipeline.getByRole('button', { name: 'Calculate without recording a prediction' }).click();
    const afterSwitch = await pipeline.locator('.sc-table-scroll').first().innerText();
    // The lab prints six decimals with trailing zeros trimmed, so compare the
    // string the page actually renders rather than a fixed-width copy.
    const asShown = value => value.toFixed(6).replace(/0+$/, '').replace(/\.$/, '');
    for (const value of [...fitted.standard.center, ...fitted.standard.scale]) {
      assert.ok(frozen.includes(asShown(value)), `fitted statistic ${value} shown before the record changed`);
      assert.ok(afterSwitch.includes(asShown(value)), `fitted statistic ${value} unchanged after the record changed`);
    }
    await pipeline.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'Pipeline lab: source row 309 giving 1.3091367505, the deleted mass taking the fitted median 4000 and -0.2367563058, an unchanged bill-length null, all three category states including an unknown all-zero block, and frozen fitted statistics across a record change' });

    // 4. The target-dependency graph.
    const encoding = page.locator('.sc-investigation').nth(3);
    await encoding.getByRole('spinbutton', { name: 'Predicted encoded value' }).fill('0.5556');
    await encoding.getByRole('button', { name: 'Check prediction' }).click();
    await checkText(encoding.locator('.sc-verdict'), /the calculation gives 5\/9/);
    await checkText(encoding, /1\/3/);
    await checkText(encoding, /7\/9/);
    await screenshot(encoding, 'scaling-encoding-desktop.png');
    // Null: row 0's own target cannot reach row 0's own value.
    await encoding.getByRole('combobox', { name: 'row 0 target' }).selectOption('0');
    await checkText(encoding.locator('.sc-pending'), /record a new prediction, even if the result turns out to be unchanged/);
    await encoding.getByRole('spinbutton', { name: 'Predicted encoded value' }).fill('0.5556');
    await encoding.getByRole('button', { name: 'Check prediction' }).click();
    await checkText(encoding.locator('.sc-verdict'), /the calculation gives 5\/9/);
    // Contrast: inspecting row 1 under the same edit does move, and so does B.
    await encoding.getByRole('combobox', { name: 'Inspect row' }).selectOption('1');
    await encoding.getByRole('spinbutton', { name: 'Predicted encoded value' }).fill('0.2222');
    await encoding.getByRole('button', { name: 'Check prediction' }).click();
    await checkText(encoding.locator('.sc-verdict'), /the calculation gives 2\/9/);
    await screenshot(encoding, 'scaling-encoding-null-desktop.png');
    await encoding.getByRole('button', { name: 'Reset', exact: true }).click();
    // Practice 7: row 3's target moves row 0 through the prior, and row 3 stays.
    await encoding.getByRole('combobox', { name: 'row 3 target' }).selectOption('1');
    await encoding.getByRole('spinbutton', { name: 'Predicted encoded value' }).fill('0.7778');
    await encoding.getByRole('button', { name: 'Check prediction' }).click();
    await checkText(encoding.locator('.sc-verdict'), /the calculation gives 7\/9/);
    await encoding.getByRole('combobox', { name: 'Inspect row' }).selectOption('3');
    await encoding.getByRole('spinbutton', { name: 'Predicted encoded value' }).fill('0.4444');
    await encoding.getByRole('button', { name: 'Check prediction' }).click();
    await checkText(encoding.locator('.sc-verdict'), /the calculation gives 4\/9/);
    // Fixture: a fold emptied by editing is refused rather than divided by zero.
    await encoding.getByRole('button', { name: 'Reset', exact: true }).click();
    for (const row of [1, 3, 5]) await encoding.getByRole('combobox', { name: `row ${row} fold` }).selectOption('0');
    await checkText(encoding, /Both internal folds need at least one row/);
    assert.ok(await encoding.getByRole('button', { name: 'Check prediction' }).isDisabled(), 'an empty fold blocks the calculation');
    await screenshot(encoding, 'scaling-encoding-empty-fold-desktop.png');
    await encoding.getByRole('button', { name: 'Reset', exact: true }).click();
    // Fixture: zero smoothing with no same-category donor uses the declared prior.
    await encoding.getByRole('combobox', { name: 'row 1 category' }).selectOption('D');
    await encoding.getByRole('combobox', { name: 'Inspect row' }).selectOption('1');
    await encoding.getByRole('spinbutton', { name: 'Smoothing α' }).fill('0');
    await encoding.getByRole('button', { name: 'Calculate without recording a prediction' }).click();
    await checkText(encoding, /declared policy here is the fold's own|fold's own prior/);
    await encoding.getByRole('button', { name: 'Reset', exact: true }).click();
    // S2: the graph must draw an edge from every donor into the prior, not only
    // from the donors whose category differs.
    await encoding.getByRole('button', { name: 'Reset', exact: true }).click();
    await encoding.getByRole('button', { name: 'Calculate without recording a prediction' }).click();
    const priorEdges = await encoding.locator('svg path.sc-flow.is-target').count();
    assert.equal(priorEdges, 3, 'all three donors feed the fold prior');
    assert.equal(await encoding.locator('svg path.sc-flow:not(.is-target)').count(), 1, 'one same-category donor feeds the numerator');
    await checkText(encoding, /dashed: prior \(every donor\)/);
    await encoding.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'Target-encoding graph: three dashed prior edges and one solid numerator edge, so tracing the drawing reproduces the 1/3 prior the arithmetic states' });

    records.push({ case: 'Target-encoding lab: the 5/9 base value with its 1/3 fold prior, the own-target null against the row 1 contrast, practice 7 moving row 0 to 7/9 while row 3 stays at 4/9, a refused empty fold and the declared zero-smoothing prior fallback' });

    // Sequential topic-20 review: supported edits must remain answerable, and
    // accepted precision must agree with the advertised inclusive tolerance.
    const translate = ruler.getByRole('button', { name: /^Translate everything/ });
    for (let step = 0; step < 5; step += 1) await translate.click();
    assert.ok(await translate.isDisabled(), 'translation stops before A mass would exceed 7000');
    await checkText(ruler, /Translation has reached a measurement bound/);
    assert.equal(await ruler.getByRole('spinbutton', { name: /^A body mass/ }).inputValue(), '6600');
    await ruler.getByRole('button', { name: 'Reset', exact: true }).click();
    await ruler.getByRole('spinbutton', { name: /^B bill length/ }).fill('75');
    await translate.click();
    assert.equal(await ruler.getByRole('spinbutton', { name: /^B bill length/ }).inputValue(), '80');
    assert.ok(await translate.isDisabled(), 'the bill-length boundary independently stops translation');
    await ruler.getByRole('button', { name: 'Reset', exact: true }).click();
    for (const candidate of ['A', 'B']) {
      await ruler.getByRole('spinbutton', { name: new RegExp(`^${candidate} bill length`) }).fill('40');
      await ruler.getByRole('spinbutton', { name: new RegExp(`^${candidate} body mass`) }).fill('4000');
    }
    await ruler.getByLabel('They tie', { exact: true }).check();
    await ruler.getByRole('button', { name: 'Check prediction' }).click();
    assert.equal(await ruler.locator('.sc-verdict.is-miss').count(), 0);
    for (const width of await ruler.locator('.sc-bar-fill').evaluateAll(items => items.map(item => item.getBoundingClientRect().width))) {
      assert.equal(width, 0, 'coincident points paint no nonzero contribution');
    }
    await ruler.getByRole('button', { name: 'Reset', exact: true }).click();

    for (const [targets, k, prediction, matches] of [
      [[0, 0, 1], 3, '0.3333', true], [[0, 0, 1], 3, '0.3334', false],
      [[0, 0.0001, 1], 2, '0', true], [[0, 0.0001, 1], 2, '0.0001', true],
      [[0, 0.0001, 1], 2, '0.0002', false],
    ]) {
      await donor.getByRole('button', { name: 'Reset', exact: true }).click();
      for (let index = 0; index < 3; index += 1) await donor.getByRole('spinbutton', { name: `D${index + 1} c` }).fill(String(targets[index]));
      await donor.getByRole('spinbutton', { name: 'Neighbours k' }).fill(String(k));
      for (let index = 0; index < k; index += 1) await donor.getByRole('checkbox', { name: `D${index + 1} donates` }).check();
      await donor.getByRole('spinbutton', { name: 'Predicted estimate' }).fill(prediction);
      await checkText(donor, /four decimal places; answers within 0\.00005 are accepted/);
      await donor.getByRole('button', { name: 'Check prediction' }).click();
      assert.equal(await donor.locator('.sc-verdict.is-miss').count(), matches ? 0 : 1, `donor rounding ${prediction}`);
      if (k === 3 && matches) await screenshot(donor, 'scaling-donor-repeating-mean-desktop.png');
    }
    await donor.getByRole('button', { name: 'Reset', exact: true }).click();

    const predictionField = pipeline.getByRole('spinbutton', { name: 'Predicted coordinate value' });
    const predictionMaximum = await predictionField.getAttribute('max');
    const measurementNames = ['bill length (mm)', 'bill depth (mm)', 'flipper length (mm)', 'body mass (g)'];
    for (let coordinate = 0; coordinate < 4; coordinate += 1) {
      for (const measurement of coordinate === 3 ? [1500, 8000] : [5, 300]) {
        await pipeline.getByRole('button', { name: 'Reset', exact: true }).click();
        await pipeline.getByRole('combobox', { name: 'Output coordinate to predict' }).selectOption(String(coordinate));
        await pipeline.getByRole('spinbutton', { name: measurementNames[coordinate], exact: true }).fill(String(measurement));
        const expected = (measurement - fitted.standard.center[coordinate]) / fitted.standard.scale[coordinate];
        await predictionField.fill(expected.toFixed(4));
        assert.equal(await predictionField.getAttribute('max'), predictionMaximum, 'prediction bounds cannot reveal the current answer');
        assert.equal(await pipeline.locator('.sc-field-error').count(), 0, 'every endpoint has an enterable prediction');
        await pipeline.getByRole('button', { name: 'Check prediction' }).click();
        await checkText(pipeline.locator('.sc-verdict'), /Your prediction matches/);
        if (coordinate === 1 && measurement === 300) await screenshot(pipeline, 'scaling-pipeline-large-coordinate-desktop.png');
      }
    }
    await pipeline.getByRole('button', { name: 'Reset', exact: true }).click();

    for (const [prediction, matches] of [['0.495', true], ['0.505', true], ['0.4949', false], ['0.5051', false]]) {
      await encoding.getByRole('button', { name: 'Reset', exact: true }).click();
      await encoding.getByRole('spinbutton', { name: 'Smoothing α' }).fill('0');
      await encoding.getByRole('combobox', { name: 'row 3 category' }).selectOption('A');
      // Row 0 now has A donors row 1 (target 1) and row 3 (target 0).
      await encoding.getByRole('spinbutton', { name: 'Predicted encoded value' }).fill(prediction);
      await checkText(encoding, /Answers within 0\.005 are accepted/);
      await encoding.getByRole('button', { name: 'Check prediction' }).click();
      assert.equal(await encoding.locator('.sc-verdict.is-miss').count(), matches ? 0 : 1, `encoding inclusive boundary ${prediction}`);
      if (prediction === '0.505') await screenshot(encoding, 'scaling-encoding-inclusive-boundary-desktop.png');
    }
    await encoding.getByRole('button', { name: 'Reset', exact: true }).click();
    const absentMeasurementStage = page.locator('.sc-figure').first().locator('.sc-stage').last();
    await checkText(absentMeasurementStage, /recorded sex female/);
    await checkText(absentMeasurementStage, /female own indicator/);
    assert.ok(!(await absentMeasurementStage.innerText()).includes('not_recorded'), 'missing mass does not erase observed sex');
    records.push({ case: 'Sequential review: translation respects both measurement bounds; coincident points paint zero bars; rounded repeating donor means and inclusive half-unit boundaries grade correctly; all eight pipeline input endpoints are answerable with fixed prediction bounds; target-encoding ±0.005 boundaries pass and ±0.0051 fail; Figure 1 preserves recorded female when only mass is absent' });

    // Figures: numbers, geometry and layout.
    for (let index = 0; index < 9; index += 1) {
      const figure = page.locator('.sc-figure').nth(index);
      const issues = await page.evaluate(inspectLessonVisualLayout, `.sc-figure:nth-of-type(${index + 1})`);
      assert.ok(Array.isArray(issues));
    }
    const rulerFigure = page.locator('.sc-figure').nth(1);
    await checkText(rulerFigure, /48\.5/);
    await checkText(rulerFigure, /Magnified: the four small values/);
    const categoryFigure = page.locator('.sc-figure').nth(2);
    await checkText(categoryFigure, /schematic of three points in a three-coordinate representation/);
    await checkText(categoryFigure, /1\.4142/);
    const comparisonFigure = page.locator('.sc-figure').nth(4);
    await checkText(comparisonFigure, /only the numerical ruler changes/);
    await checkText(comparisonFigure, /The axis starts at zero/);
    const rankFigure = page.locator('.sc-figure').nth(6);
    await checkText(rankFigure, /null for the rank/);
    await checkText(rankFigure, /6\.907755/);
    const poolingFigure = page.locator('.sc-figure').nth(8);
    await checkText(poolingFigure, /16\/3/);
    await checkText(poolingFigure, /2\.3094/);
    for (let index = 0; index < 9; index += 1) {
      await screenshot(page.locator('.sc-figure').nth(index), `scaling-figure-${index + 1}-desktop.png`);
    }
    assert.ok(await page.locator('.sc-lesson marker#sc-arrow').count() >= 1, 'flows carry a shared arrowhead definition');
    assert.ok(await page.locator('.sc-lesson path[marker-end]').count() >= 10, 'flows actually reference it');
    const identifierStyle = await page.locator('.sc-lesson text.sc-point-id').first().evaluate(element => {
      const style = getComputedStyle(element);
      return { weight: style.fontWeight, fill: style.fill };
    });
    assert.equal(identifierStyle.weight, '700', 'observation identifiers keep their weight');
    assert.equal(identifierStyle.fill, 'rgb(240, 230, 200)', 'observation identifiers keep their colour');
    records.push({ case: 'Direction and emphasis survive the stylesheet: a shared arrowhead is defined and referenced by every flow, and the observation identifiers keep the weight and colour their own rule declares' });

    records.push({ case: 'Figures carry their required content: the magnified ruler inset with 48.5 kept on the full line, the labelled category schematic beside a true coordinate plane, the zero-origin comparison axis, the rank null against the log contrast, and the pooled 16/3 with its separate standard error' });

    // Geometry triage across widths, including the equal-aspect requirement.
    for (const width of [1366, 1024, 768, 390, 320]) {
      await page.setViewportSize({ width, height: 1000 });
      await settle(page);
      const issues = await page.evaluate(inspectLessonVisualLayout, '.sc-lesson');
      const flagged = issues.flatMap(figure => figure.issues.map(issue => ({ ...issue, description: figure.description })));
      assert.deepEqual(flagged, [], `Figure geometry collides at ${width}px: ${JSON.stringify(flagged).slice(0, 900)}`);
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `Document overflows at ${width}px`);
    }
    await page.setViewportSize({ width: 1366, height: 1000 });
    records.push({ case: 'Label/line geometry and document width clear at 1366, 1024, 768, 390 and 320 px' });

    const localScripts = [...new Set(requests.map(address => new URL(address)).filter(address => address.origin === new URL(base).origin && address.pathname.endsWith('.js')).map(address => address.pathname.slice(1)))];
    assert.deepEqual(localScripts.filter(filename => bodyFiles.has(filename)), [bodyFile]);
    assert.deepEqual(localScripts.filter(filename => !allowedScripts.has(filename)), []);
    assert.ok(!requests.some(address => address.includes('/outlines/')));
    assert.ok(!requests.some(address => address.includes('penguins.csv')), 'the page never fetches the data to render');
    records.push({
      case: 'Fresh route requests only the selected lesson and its shared closure',
      requestedScripts: localScripts,
      rawScriptBytes: localScripts.reduce((sum, filename) => sum + fs.statSync(path.join(distDir, filename)).size, 0),
      gzipEstimateBytes: localScripts.reduce((sum, filename) => sum + gzipSync(fs.readFileSync(path.join(distDir, filename))).length, 0),
      interpretation: 'Built requested file sizes; the gzip estimate is not measured network compression or latency.',
    });

    for (const width of [390, 320]) {
      await page.setViewportSize({ width, height: 844 });
      await page.locator('.sc-lesson details').evaluateAll(items => items.forEach(item => { item.open = true; }));
      await settle(page);
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `Document overflows at ${width}px`);
      const mathOverflow = await page.locator('.sc-lesson .katex-display').evaluateAll(items => items.filter(item => item.scrollWidth > item.clientWidth + 1).map(item => item.textContent));
      assert.deepEqual(mathOverflow, [], `Overflowing formulas at ${width}px`);
      assert.equal(await page.locator('.katex-error').count(), 0);
      const tableOverflow = await page.locator('.sc-lesson .sc-table-scroll').evaluateAll(items => items.filter(item => item.getBoundingClientRect().width > innerWidth + 1).length);
      assert.equal(tableOverflow, 0, `A table exceeds the viewport at ${width}px`);
      if (width === 390) {
        await screenshot(page.locator('.sc-figure').nth(0), 'scaling-record-mobile.png');
        await screenshot(page.locator('.sc-figure').nth(1), 'scaling-ruler-figure-mobile.png');
        await screenshot(page.locator('.sc-figure').nth(2), 'scaling-category-mobile.png');
        await screenshot(page.locator('.sc-investigation').nth(0), 'scaling-ruler-mobile.png');
        await screenshot(page.locator('.sc-investigation').nth(2), 'scaling-pipeline-mobile.png');
      }
      if (width === 320) {
        await screenshot(page.locator('.sc-figure').nth(3), 'scaling-boundary-320.png');
        await screenshot(page.locator('.sc-investigation').nth(1), 'scaling-donor-320.png');
      }
      const typeSizes = await page.locator('.sc-figure svg').evaluateAll(items => items.flatMap(svg => {
        const box = svg.getBoundingClientRect();
        const viewBox = svg.viewBox.baseVal.width || box.width;
        const scale = box.width / viewBox;
        return [...svg.querySelectorAll('text')].map(node => Number(getComputedStyle(node).fontSize.replace('px', '')) * scale);
      }));
      records.push({
        case: `Narrow ${width}px layout, readable formula grouping, contained tables and every deeper branch rendered`,
        figureTextNodes: typeSizes.length,
        smallestRenderedPx: Number(Math.min(...typeSizes).toFixed(2)),
        below9px: typeSizes.filter(size => size < 9).length,
        below8px: typeSizes.filter(size => size < 8).length,
        interpretation: 'Rendered size is the computed font-size scaled by the SVG width over its viewBox width; these are measurements, not a target.',
      });
    }
    await page.setViewportSize({ width: 1366, height: 1000 });
    await page.evaluate(() => { document.documentElement.style.fontSize = '200%'; });
    await settle(page);
    assert.deepEqual((await page.evaluate(inspectLessonVisualLayout, '.sc-lesson')).flatMap(figure => figure.issues), [], 'figures stay separated with enlarged root text');
    await screenshot(page.locator('.sc-figure').nth(3), 'scaling-boundary-enlarged-text.png');
    await page.evaluate(() => { document.documentElement.style.fontSize = ''; });
    records.push({ case: 'Enlarged root text leaves every figure label separated' });

    await page.locator('.reader-complete').click();
    assert.equal(new URL(page.url()).pathname.split('/').at(-1), topicId);
    await page.reload({ waitUntil: 'domcontentloaded' });
    await ready(page);
    assert.equal(await page.evaluate(id => JSON.parse(localStorage.getItem('kd-progress'))[id], topicId), true);
    await page.locator('.reader-footer__next').click();
    await page.waitForURL('**/cross-validation-hyperparameter-tuning?module=classical-ml');
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
      dataset: { name: provenance.name, sha256: provenance.sha256, rows: provenance.rows, licence: provenance.licence },
      split: { training: split.training, heldOut: split.heldOut },
      correctCounts: Object.fromEntries(comparison.map(row => [row.method, row.correct])),
      records, screenshots: screenshotPaths,
      visualInterpretation: 'The intended Space Grotesk font loaded before the checks. Screenshots capture informative states: matched and missed predictions, the two donor contrasts and both fixture cases, an imputed coordinate, an unknown category block, an empty internal fold, every inline figure, and mobile captures at 390 and 320 px. They require separate visual inspection.',
    };
    if (write) fs.writeFileSync(evidencePath, JSON.stringify(report, null, 2) + '\n');
    else if (fs.existsSync(evidencePath)) {
      const recorded = read(evidencePath);
      assert.deepEqual(recorded.sourceHashes, sourceHashes, 'recorded evidence describes different sources; re-run with --write');
      assert.equal(recorded.records.length, records.length, 'recorded case count differs; re-run with --write');
    }
    console.log(JSON.stringify({
      status: report.status, cases: records.length, evidencePath, sourceFiles: ownedFiles.length,
      mode: write ? 'evidence and screenshots rewritten' : 'read-only',
    }));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });

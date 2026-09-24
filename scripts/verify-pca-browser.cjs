// Production browser review of the PCA lesson: visible content, the four
// investigations, narrow layouts, module sequence, completion and recovery.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { gzipSync } = require('node:zlib');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');

const base = (process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4173').replace(/\/+$/, '');
const topicId = 'pca-dimensionality-reduction';
const sourcePath = `src/learn/data/topics/${topicId}.jsx`;
const route = `${base}/learn/path/full-curriculum/${topicId}?module=classical-ml`;
const evidencePath = 'docs/teaching/evidence/pca-browser.json';
const read = filename => JSON.parse(fs.readFileSync(filename, 'utf8'));
const hash = filename => createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
const normalize = text => text.replace(/\s+/g, ' ').trim();
const ownedFiles = [sourcePath, 'src/learn/data/pca-models.js', 'src/learn/data/pca-wine-data.js', 'src/learn/data/pca-examples.js', 'src/learn/components/lesson-labs/PcaLabs.jsx', 'src/learn/components/lesson-labs/PcaFigures.jsx', 'src/learn/components/lesson-labs/pca-labs.css', `src/learn/data/curriculum/blueprints/${topicId}.js`, 'public/learn-assets/pca/wine.csv'];

(async () => {
  const startedAt = new Date().toISOString();
  const sourceHashes = Object.fromEntries(ownedFiles.map(filename => [filename, hash(filename)]));
  const build = read('dist/.vite/manifest.json');
  const buildHash = hash('dist/.vite/manifest.json');
  const publications = read('src/learn/data/lesson-manifest.json');
  const { tracks } = await import('../src/learn/data/tracks.js');
  const { pcaExamples } = await import('../src/learn/data/pca-examples.js');
  const module = tracks.find(track => track.id === 'classical-ml');
  assert.equal(module.topicIds[11], topicId);
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
  const checkText = async (locator, pattern) => assert.match(await locator.innerText(), pattern);
  const settle = page => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  const ready = async page => {
    await page.locator('.pca-lesson').waitFor();
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
    await checkText(page.locator('.reader-header h1'), /^PCA & Dimensionality Reduction$/);
    await checkText(page.locator('.reader-header__meta'), /12 of 39 topics on this route/);
    await checkText(page.locator('.reader-footer__previous'), /K-Means & Hierarchical Clustering/);
    await checkText(page.locator('.reader-footer__next'), /Clustering Evaluation & Validation/);
    assert.equal(await page.locator('.pca-investigation').count(), 4);
    assert.equal(await page.locator('.pca-figure').count(), 7);
    assert.equal(await page.locator('.pca-practice').count(), 8);
    assert.equal(await page.locator('.python-example').count(), 8);
    const rendered = normalize(await page.locator('.pca-lesson').textContent());
    for (const [key, example] of Object.entries(pcaExamples)) {
      assert.ok(rendered.includes(normalize(example.code)), `Missing actual code for ${key}`);
      assert.ok(rendered.includes(normalize(example.expected)), `Missing actual expected output for ${key}`);
    }
    for (const anchor of await page.locator('.lesson-intro a[href^="#"]').evaluateAll(items => items.map(item => item.getAttribute('href')))) {
      assert.equal(await page.locator(`[id="${anchor.slice(1)}"]`).count(), 1, `Broken lesson anchor ${anchor}`);
    }
    assert.equal(await page.locator('.katex-error').count(), 0);
    const csv = await page.request.get(`${base}/learn-assets/pca/wine.csv`);
    assert.equal(csv.status(), 200);
    assert.equal((await csv.text()).trim().split('\n').length, 179, 'CSV serves header plus 178 rows');
    records.push({ case: 'Complete visible code/output for eight programs, eleven route anchors, seven figures, eight practice tasks, downloadable CSV, current metadata and module sequence' });

    const projection = page.locator('.pca-investigation').nth(0);
    await checkText(projection.locator('.pca-readout'), /hidden until you compare/);
    await projection.getByLabel('Predict first', { exact: false }).selectOption('less');
    await projection.getByRole('button', { name: 'Compare prediction' }).click();
    await checkText(projection.locator('.pca-feedback'), /Your prediction matches: Less error/);
    await checkText(projection.locator('.pca-feedback'), /reference loses 10 units²; the proposed direction loses 2 units²/);
    await checkText(projection.locator('.pca-readout'), /retained 18 \+ residual 2/);
    const proposedAngle = projection.getByLabel('Proposed angle', { exact: false });
    await proposedAngle.press('ArrowRight');
    await checkText(projection.locator('.pca-feedback'), /inputs changed after your last comparison/);
    await projection.getByRole('button', { name: 'Fit best direction' }).click();
    await checkText(projection.locator('.pca-field output').nth(1), /^45°$/);
    await projection.getByLabel('A first reading', { exact: false }).fill('3');
    await checkText(projection.locator('.pca-caption'), /exact best direction for these points is/);
    await projection.getByRole('button', { name: 'Back', exact: true }).click();
    await projection.getByRole('button', { name: 'Apply translation' }).click();
    await checkText(projection.locator('.pca-readout'), /Mean \(5, 1\)/);
    await projection.getByRole('button', { name: 'Reset', exact: true }).click();
    await checkText(projection.locator('.pca-readout'), /Mean \(3, 2\)/);
    records.push({ case: 'Projection workbench: hidden residual until comparison, correct SSE feedback, stale invalidation, fit-best-direction, point editing with Back, translation and reset' });

    // Complementary regression: changed data must fit at full precision, and
    // undo must restore the selected point as well as the point collection.
    await projection.getByRole('button', { name: 'Add a point at the mean' }).click();
    assert.equal(await projection.getByLabel('Selected observation', { exact: false }).inputValue(), '4');
    await projection.getByRole('button', { name: 'Back', exact: true }).click();
    assert.equal(await projection.getByLabel('Selected observation', { exact: false }).locator('option').count(), 4);
    assert.equal(await projection.getByLabel('Selected observation', { exact: false }).inputValue(), '0');
    assert.equal(await projection.getByLabel('A first reading', { exact: false }).inputValue(), '1');
    await projection.getByLabel('Selected observation', { exact: false }).selectOption('3');
    await projection.getByLabel('D second reading', { exact: false }).fill('2.5');
    await projection.getByRole('button', { name: 'Fit best direction' }).click();
    assert.ok(Math.abs(Number(await proposedAngle.inputValue()) - 43.339262118326985) < 1e-9, 'Fitted angle retains the exact direction rather than whole-degree rounding');
    await projection.getByLabel('Predict first', { exact: false }).selectOption('less');
    await projection.getByRole('button', { name: 'Compare prediction' }).click();
    await checkText(projection.locator('.pca-feedback'), /proposed direction loses 2\.582 units²/);
    assert.ok(await projection.getByLabel('Predict first', { exact: false }).isDisabled(), 'Compared prediction is frozen');
    await projection.getByLabel('D second reading', { exact: false }).fill('3');
    assert.ok(await projection.getByLabel('Predict first', { exact: false }).isEnabled(), 'Changed data permits a new prediction');
    await projection.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'Projection regression: Add→Back restores selection, noninteger fitted optimum retains precision, compared prediction freezes and changed inputs release it' });

    const metric = page.locator('.pca-investigation').nth(1);
    await checkText(metric.locator('.pca-readout'), /Apply your settings/);
    await metric.getByLabel('Same multiplier, typed', { exact: false }).fill('10');
    await metric.getByLabel('Predict first', { exact: false }).selectOption('second');
    await metric.getByRole('button', { name: 'Apply and compare' }).click();
    await checkText(metric.locator('.pca-feedback'), /Your prediction matches: The second axis/);
    await checkText(metric.locator('.pca-readout'), /retaining 96\.15%/);
    await metric.getByLabel('Same multiplier, typed', { exact: false }).fill('2.67');
    await metric.getByLabel('Rectangle half-height b', { exact: false }).fill('0.75');
    await metric.getByLabel('Same multiplier, typed', { exact: false }).fill('2.6667');
    await metric.getByLabel('Predict first', { exact: false }).selectOption('tie');
    await metric.getByRole('button', { name: 'Apply and compare' }).click();
    await checkText(metric.locator('.pca-feedback'), /Your prediction matches: No preferred axis/);
    await metric.getByLabel('Coordinate geometry').selectOption('standardized');
    await metric.getByLabel('Same multiplier, typed', { exact: false }).fill('10');
    await metric.getByLabel('Predict first', { exact: false }).selectOption('tie');
    await metric.getByRole('button', { name: 'Apply and compare' }).click();
    await checkText(metric.locator('.pca-feedback'), /Your prediction matches: No preferred axis/);
    await metric.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'Metric lab: pending versus applied state, second-axis flip at m = 10 with 96.15%, exact tie at m = a/b, standardized tie at every multiplier, reset' });

    await metric.getByLabel('Rectangle half-height b', { exact: false }).fill('4');
    await metric.getByLabel('Same multiplier, typed', { exact: false }).fill('10');
    await metric.getByLabel('Predict first', { exact: false }).selectOption('second');
    await metric.getByRole('button', { name: 'Apply and compare' }).click();
    await checkText(metric.locator('.pca-feedback'), /Your prediction matches: The second axis/);
    await checkText(metric.locator('.pca-readout'), /Axis variances 5\.3333 and 2133\.3333/);
    await checkText(metric.locator('.pca-readout'), /retaining 99\.75%/);
    assert.ok(await metric.getByLabel('Predict first', { exact: false }).isDisabled());
    await metric.getByLabel('Coordinate geometry').selectOption('standardized');
    await metric.getByLabel('Predict first', { exact: false }).selectOption('tie');
    await metric.getByRole('button', { name: 'Apply and compare' }).click();
    await checkText(metric.locator('.pca-feedback'), /Your prediction matches: No preferred axis/);
    await metric.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'Metric regression: legal b=4, multiplier=10 renders and applies raw ±40 coordinates; independent axis variances and 99.75% agree; standardized alternative ties' });

    const budget = page.locator('.pca-investigation').nth(2);
    assert.equal(await budget.locator('.pca-hidden-curve').count(), 1, 'Curve hidden before commitment');
    await budget.getByLabel('Predict first', { exact: false }).selectOption('8');
    await budget.getByRole('button', { name: 'Reveal the crossing' }).click();
    await checkText(budget.locator('.pca-feedback'), /Your prediction matches: 8 components/);
    await checkText(budget.locator('.pca-feedback'), /7 components leave 0\.1265/);
    assert.equal(await budget.locator('.pca-curve').count(), 1);
    assert.ok(await budget.getByLabel('Predict first', { exact: false }).isDisabled(), 'Revealed budget prediction is frozen');
    await budget.getByLabel('Same budget, typed').fill('0.06');
    await checkText(budget.locator('.pca-feedback'), /inputs changed/);
    assert.equal(await budget.locator('.pca-curve').count(), 0, 'Changing the budget hides the newly computed crossing until another commitment');
    assert.equal(await budget.locator('.pca-hidden-curve').count(), 1);
    assert.ok(await budget.getByLabel('Predict first', { exact: false }).isEnabled());
    await budget.getByLabel('Predict first', { exact: false }).selectOption('10');
    await budget.getByRole('button', { name: 'Reveal the crossing' }).click();
    await checkText(budget.locator('.pca-feedback'), /Your prediction matches: 10 components/);
    await checkText(budget.locator('.pca-feedback'), /9 components leave 0\.0724/);
    await budget.getByLabel('Explore a different k', { exact: false }).fill('13');
    await checkText(budget.locator('.pca-readout'), /with 13 components/);
    await budget.getByLabel('Validation wine').selectOption({ index: 3 });
    assert.equal(await budget.locator('.pca-table-scroll').first().locator('tbody tr').count(), 13, 'Thirteen feature rows for the selected wine');
    await budget.getByRole('button', { name: 'Reset', exact: true }).click();
    assert.equal(await budget.locator('.pca-hidden-curve').count(), 1);
    records.push({ case: 'Budget lab: curve hidden until commitment, 10% selects 8 and 6% selects 10 with bracketing ratios, stale invalidation on budget change, per-wine residual table at explored k, reset' });

    const labels = page.locator('.pca-investigation').nth(3);
    await labels.getByLabel('Predict first', { exact: false }).selectOption('collide');
    await labels.getByRole('button', { name: 'Check', exact: true }).click();
    await checkText(labels.locator('.pca-feedback'), /Your prediction matches/);
    await checkText(labels.locator('.pca-feedback'), /Collisions: A \(A\) and B \(B\)/);
    await checkText(labels.locator('.pca-readout'), /keeps 99\.01% of the variance/);
    await labels.getByLabel('Label rule').selectOption('x');
    await labels.getByLabel('Predict first', { exact: false }).selectOption('distinct');
    await labels.getByRole('button', { name: 'Check', exact: true }).click();
    await checkText(labels.locator('.pca-feedback'), /Your prediction matches: Yes/);
    await labels.getByLabel('Retained components').selectOption('pc2');
    await labels.getByLabel('Predict first', { exact: false }).selectOption('collide');
    await labels.getByRole('button', { name: 'Check', exact: true }).click();
    await checkText(labels.locator('.pca-feedback'), /Your prediction matches: No/);
    await labels.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'Label lab: PC1 collision at 99.01%, label-rule switch reverses usefulness without changing the fit, PC2 collision for x labels, reset' });

    await labels.getByLabel('Retained components').selectOption('pc2');
    await labels.getByLabel('Predict first', { exact: false }).selectOption('distinct');
    await labels.getByRole('button', { name: 'Check', exact: true }).click();
    await checkText(labels.locator('.pca-feedback'), /The 2 retained locations each contain only one class/);
    await checkText(labels.locator('.pca-readout'), /Labels remain distinguishable/);
    assert.ok(await labels.getByLabel('Predict first', { exact: false }).isDisabled());
    await labels.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'Class-information regression: PC2 and y labels produce two class-pure locations, not four distinct observations, with a frozen checked prediction' });

    const firstHint = page.locator('.pca-practice').first().getByText('Get a hint', { exact: true });
    await firstHint.focus();
    await page.keyboard.press('Enter');
    assert.ok(await firstHint.evaluate(element => element.parentElement.open));
    await page.locator('.pca-practice').first().getByText('Show the explained solution', { exact: true }).click();
    await checkText(page.locator('.pca-practice').first(), /\(6\.5, 5\.5\)/);
    await page.locator('.pca-figure').nth(3).getByRole('button', { name: /Overlay the cultivar labels/ }).click();
    await checkText(page.locator('.pca-figure').nth(3), /circle = cultivar 1/);
    const screenshot = async (locator, filename) => {
      const destination = `docs/teaching/evidence/${filename}`;
      const viewport = page.viewportSize();
      const bounds = await locator.boundingBox();
      if (bounds.height > viewport.height - 160) await page.setViewportSize({ width: viewport.width, height: Math.ceil(bounds.height) + 180 });
      await locator.scrollIntoViewIfNeeded();
      await locator.evaluate(element => window.scrollTo(0, element.getBoundingClientRect().top + window.scrollY - 100));
      await locator.screenshot({ path: destination });
      await page.setViewportSize(viewport);
      screenshotPaths.push(destination);
    };
    await projection.getByLabel('Predict first', { exact: false }).selectOption('less');
    await projection.getByRole('button', { name: 'Compare prediction' }).click();
    await screenshot(projection, 'pca-projection-desktop.png');
    await screenshot(page.locator('.pca-figure').nth(0), 'pca-shadow-desktop.png');
    await screenshot(page.locator('.pca-figure').nth(3), 'pca-wine-overview-desktop.png');
    await budget.getByLabel('Predict first', { exact: false }).selectOption('8');
    await budget.getByRole('button', { name: 'Reveal the crossing' }).click();
    await screenshot(budget, 'pca-budget-desktop.png');
    await metric.getByLabel('Same multiplier, typed', { exact: false }).fill('2');
    await metric.getByLabel('Predict first', { exact: false }).selectOption('tie');
    await metric.getByRole('button', { name: 'Apply and compare' }).click();
    await screenshot(metric, 'pca-metric-tie-desktop.png');
    await labels.getByLabel('Predict first', { exact: false }).selectOption('collide');
    await labels.getByRole('button', { name: 'Check', exact: true }).click();
    await screenshot(labels, 'pca-labels-collision-desktop.png');
    await screenshot(page.locator('.pca-figure').nth(4), 'pca-biplot-desktop.png');
    await screenshot(page.locator('.pca-figure').nth(6), 'pca-residual-alarm-desktop.png');
    const localScripts = [...new Set(requests.map(address => new URL(address)).filter(address => address.origin === new URL(base).origin && address.pathname.endsWith('.js')).map(address => address.pathname.slice(1)))];
    assert.deepEqual(localScripts.filter(filename => bodyFiles.has(filename)), [bodyFile]);
    assert.deepEqual(localScripts.filter(filename => !allowedScripts.has(filename)), []);
    assert.ok(!requests.some(address => address.includes('/outlines/')));
    records.push({ case: 'Fresh route requests only the selected lesson and its shared closure', requestedScripts: localScripts, rawScriptBytes: localScripts.reduce((sum, filename) => sum + fs.statSync(path.join('dist', filename)).size, 0), gzipEstimateBytes: localScripts.reduce((sum, filename) => sum + gzipSync(fs.readFileSync(path.join('dist', filename))).length, 0), interpretation: 'Built requested file sizes; gzip estimate is not measured network compression or latency.' });

    for (const width of [390, 320]) {
      await page.setViewportSize({ width, height: 844 });
      await page.locator('.pca-lesson details').evaluateAll(items => items.forEach(item => { item.open = true; }));
      await settle(page);
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `Document overflows at ${width}px`);
      const mathOverflow = await page.locator('.pca-lesson .katex-display').evaluateAll(items => items.filter(item => item.scrollWidth > item.clientWidth + 1).map(item => item.textContent));
      assert.deepEqual(mathOverflow, [], `Overflowing formulas at ${width}px`);
      assert.equal(await page.locator('.katex-error').count(), 0);
      if (width === 390) {
        await screenshot(page.locator('.pca-figure').nth(1), 'pca-conservation-mobile.png');
        await screenshot(page.locator('.pca-figure').nth(2), 'pca-shapes-mobile.png');
        await screenshot(labels, 'pca-labels-mobile.png');
        await screenshot(page.locator('.pca-figure').nth(5), 'pca-gaussian-mobile.png');
        await screenshot(metric, 'pca-metric-mobile.png');
      }
      records.push({ case: `Narrow ${width}px layout, readable formula grouping and all deeper branches rendered` });
    }
    await page.setViewportSize({ width: 1366, height: 1000 });
    await page.locator('.reader-complete').click();
    assert.equal(new URL(page.url()).pathname.split('/').at(-1), topicId);
    await page.reload({ waitUntil: 'domcontentloaded' });
    await ready(page);
    assert.equal(await page.evaluate(id => JSON.parse(localStorage.getItem('kd-progress'))[id], topicId), true);
    await page.locator('.reader-footer__next').click();
    await page.waitForURL('**/clustering-evaluation-validation-silhouette-ari-nmi?module=classical-ml');
    await page.waitForFunction(() => document.querySelector('.reader-header h1')?.textContent.includes('Clustering Evaluation'), null, { timeout: 20000 });
    await checkText(page.locator('.reader-header h1'), /Clustering Evaluation & Validation/);
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
        return intercepted.fulfill({ status: 200, contentType: 'text/javascript', body: 'export default {content(){throw new Error("Controlled PCA review failure")}}' });
      });
      await trial.goto(route, { waitUntil: 'domcontentloaded' });
      await trial.locator('.lesson-load-error').waitFor();
      assert.ok(await trial.locator('.reader-complete').isDisabled());
      await trial.getByRole('button', { name: /reload/i }).click();
      await ready(trial);
      assert.ok(await trial.locator('.reader-complete').isEnabled());
      records.push({ case: `${failure} failure keeps completion disabled and recovers with explicit reload` });
      await isolated.close();
    }
    for (const [filename, expected] of Object.entries(sourceHashes)) assert.equal(hash(filename), expected, `Source changed during check: ${filename}`);
    assert.equal(hash('dist/.vite/manifest.json'), buildHash);
    const report = { startedAt, completedAt: new Date().toISOString(), topicId, status: 'passed', browser: await browser.version(), buildManifestHash: buildHash, sourceHashes, moduleTopicCount: module.topicIds.length, records, screenshots: screenshotPaths, visualInterpretation: 'Intended Space Grotesk font loaded before checks. Screenshots capture informative states: a compared prediction, a revealed budget crossing, the labeled overview. They require separate visual inspection; structural checks alone do not establish teaching quality.' };
    fs.writeFileSync(evidencePath, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify({ status: report.status, cases: records.length, evidencePath, sourceFiles: ownedFiles.length }));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });

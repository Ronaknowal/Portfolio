// Scoped production review: topic interactions, visible content and lazy integration.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { gzipSync } = require('node:zlib');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');

const base = (process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4173').replace(/\/+$/, '');
const topicId = 'k-means-hierarchical-clustering';
const sourcePath = `src/learn/data/topics/${topicId}.jsx`;
const route = `${base}/learn/path/full-curriculum/${topicId}?module=classical-ml`;
const evidencePath = 'docs/teaching/evidence/k-means-hierarchical-browser.json';
const read = filename => JSON.parse(fs.readFileSync(filename, 'utf8'));
const hash = filename => createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
const normalize = text => text.replace(/\s+/g, ' ').trim();
const ownedFiles = [sourcePath, 'src/learn/data/k-means-hierarchical-models.js', 'src/learn/data/k-means-hierarchical-faithful.js', 'src/learn/data/k-means-hierarchical-examples.js', 'src/learn/components/lesson-labs/KMeansHierarchicalLabs.jsx', 'src/learn/components/lesson-labs/k-means-hierarchical-labs.css', 'src/learn/components/lesson-labs/KMeansHierarchicalFigures.jsx', 'src/learn/components/lesson-labs/k-means-hierarchical-figures.css', `src/learn/data/curriculum/blueprints/${topicId}.js`];

(async () => {
  const startedAt = new Date().toISOString();
  const sourceHashes = Object.fromEntries(ownedFiles.map(filename => [filename, hash(filename)]));
  const build = read('dist/.vite/manifest.json');
  const buildHash = hash('dist/.vite/manifest.json');
  const publications = read('src/learn/data/lesson-manifest.json');
  const baseline = read('docs/teaching/evidence/classical-ml-supervised-baseline.json');
  const { topicCatalogue } = await import('../src/learn/data/curriculum/topic-catalogue.js');
  const { tracks } = await import('../src/learn/data/tracks.js');
  const { clusteringExamples } = await import('../src/learn/data/k-means-hierarchical-examples.js');
  const module = tracks.find(track => track.id === 'classical-ml');
  assert.deepEqual(Object.keys(topicCatalogue), baseline.catalogueIds);
  assert.deepEqual(publications, baseline.publicationMappings);
  assert.deepEqual(module.topicIds, baseline.moduleOrder);
  assert.equal(module.topicIds[10], topicId);
  const bodyFile = build[sourcePath].file;
  const bodyFiles = new Set(Object.values(publications).map(value => build[`src/learn/data/${value.replace(/^\.\//, '')}`].file));
  assert.equal(bodyFiles.size, 228);
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
    await page.locator('.clustering-lesson').waitFor();
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
    await checkText(page.locator('.reader-header h1'), /^K-Means & Hierarchical Clustering$/);
    await checkText(page.locator('.reader-header__meta'), /11 of 39 topics on this route/);
    await checkText(page.locator('.reader-footer__previous'), /Survival Analysis/);
    await checkText(page.locator('.reader-footer__next'), /PCA & Dimensionality Reduction/);
    const moduleGroup = page.locator('[data-module-id="classical-ml"]');
    assert.deepEqual(await moduleGroup.locator('[data-topic-id]').evaluateAll(items => items.map(item => item.dataset.topicId)), baseline.moduleOrder);
    assert.equal(await page.locator('.kh-investigation').count(), 5);
    assert.equal(await page.locator('.cluster-figure').count(), 7);
    assert.equal(await page.locator('.cluster-practice').count(), 10);
    assert.equal(await page.locator('.python-example').count(), 6);
    const rendered = normalize(await page.locator('.clustering-lesson').textContent());
    for (const [key, example] of Object.entries(clusteringExamples)) {
      assert.ok(rendered.includes(normalize(example.code)), `Missing actual code for ${key}`);
      assert.ok(rendered.includes(normalize(example.expected)), `Missing actual expected output for ${key}`);
    }
    for (const anchor of await page.locator('.lesson-intro a[href^="#"]').evaluateAll(items => items.map(item => item.getAttribute('href')))) {
      assert.equal(await page.locator(`[id="${anchor.slice(1)}"]`).count(), 1, `Broken lesson anchor ${anchor}`);
    }
    assert.equal(await page.locator('.katex-error').count(), 0);
    records.push({ case: 'Complete visible code/output, 14 route anchors, practice, figures, current metadata and actual module sequence' });

    const lloyd = page.locator('.kh-investigation').nth(0);
    assert.ok(await lloyd.getByRole('button', { name: 'Back', exact: true }).isDisabled());
    const next = lloyd.getByRole('button', { name: 'Next phase' });
    await next.focus();
    await page.keyboard.press('Enter');
    await checkText(lloyd.locator('.kh-readout'), /SSE = 19\.25/);
    await next.click();
    await checkText(lloyd.locator('.kh-readout'), /SSE = 7\.6875/);
    const assigned = await lloyd.locator('.kh-readout').innerText();
    await lloyd.getByRole('button', { name: 'Back', exact: true }).click();
    await next.click();
    assert.equal(await lloyd.locator('.kh-readout').innerText(), assigned);
    await next.click();
    await checkText(lloyd.locator('.kh-readout'), /fixed point/);
    assert.ok(await next.isDisabled());
    await lloyd.getByRole('button', { name: 'Reset', exact: true }).click();
    await lloyd.getByLabel('Predict first', { exact: false }).selectOption('1');
    await lloyd.getByRole('button', { name: 'Run to fixed point' }).click();
    await checkText(lloyd.locator('.kh-feedback'), /Your prediction matches: One update/);
    assert.equal(await lloyd.locator('.kh-boundary').count(), 1, 'Nearest-center boundary is drawn');
    for (const [secondRow, expected] of [['2', 1], ['1', 9]]) {
      await lloyd.getByLabel('Point configuration').selectOption('rectangle');
      await lloyd.getByLabel('Center 0 starts at row').selectOption('0');
      await lloyd.getByLabel('Center 1 starts at row').selectOption(secondRow);
      for (let count = 0; await next.isEnabled() && count < 20; count += 1) await next.click();
      await checkText(lloyd.locator('.kh-readout'), new RegExp(`SSE = ${expected} using`));
    }
    await lloyd.getByLabel('Center 1 starts at row').selectOption('0');
    await next.click();
    await next.click();
    await checkText(lloyd.locator('.kh-note'), /Empty center/);
    await lloyd.getByRole('button', { name: 'Reset', exact: true }).click();
    await checkText(lloyd.locator('.kh-readout'), /no assignment yet/);
    records.push({ case: 'Lloyd keyboard, prediction feedback, boundary line, free seed rows, back/reset, fixed-point disable, nonoptimal rectangle and duplicate center' });

    const geometry = page.locator('.kh-investigation').nth(1);
    await checkText(geometry.locator('.kh-readout'), /splits: SSE 1\./);
    assert.equal(await geometry.locator('.kh-table-scroll').first().locator('tbody tr').count(), 7, 'All seven two-group splits are listed');
    await geometry.getByLabel('Predict first', { exact: false }).selectOption('bottom-top');
    await geometry.getByLabel('Vertical measurement unit').selectOption('10');
    await checkText(geometry.locator('.kh-readout'), /splits: SSE 9\./);
    await geometry.getByRole('button', { name: 'Check', exact: true }).click();
    await checkText(geometry.locator('.kh-feedback'), /Your prediction matches/);
    await geometry.getByLabel('Weight on squared vertical differences').selectOption('0.01');
    await checkText(geometry.locator('.kh-readout'), /splits: SSE 1\./);
    await geometry.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'Exhaustive seven-split table, prediction check, unadjusted unit change alters geometry and compensating weight restores it' });

    const seeding = page.locator('.kh-investigation').nth(2);
    assert.equal(await seeding.getByLabel('Draw position in the cumulative probability line').evaluate(element => element.tagName), 'INPUT', 'The range label must target the input, not its output');
    await seeding.getByLabel('Draw position in the cumulative probability line').press('Home');
    await checkText(seeding.locator('.kh-readout'), /selects P1/);
    await seeding.getByLabel('Draw position in the cumulative probability line').press('End');
    await checkText(seeding.locator('.kh-readout'), /selects P5/);
    await seeding.getByLabel('Draw position in the cumulative probability line').fill('0.5');
    await checkText(seeding.locator('.kh-readout'), /selects P4 as center C1/);
    await seeding.getByRole('button', { name: 'Accept this draw and pick the next center' }).click();
    await checkText(seeding.locator('.kh-step-controls'), /Centers so far: P0, P4/);
    assert.equal(await seeding.locator('.kh-frequency li').count(), 6, 'Frequency comparison lists every row');
    await seeding.getByRole('button', { name: 'Undo last draw' }).click();
    await checkText(seeding.locator('.kh-step-controls'), /Centers so far: P0$/);
    await seeding.getByLabel('Seeding data').selectOption('duplicates');
    await checkText(seeding.locator('.kh-readout'), /Every D² is zero/);
    await checkText(seeding.locator('.kh-scatter figcaption'), /C0/);
    assert.ok(!(await seeding.locator('.kh-scatter figcaption').innerText()).includes('C1'));
    await seeding.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'D² first/last positive intervals, keyboard draw, sequential draws with frequency bars, zero-mass state and reset' });

    const hierarchy = page.locator('.kh-investigation').nth(3);
    for (const method of ['single', 'complete', 'average', 'ward']) {
      await hierarchy.getByLabel('Linkage definition').selectOption(method);
      await hierarchy.getByLabel('Partition rule').selectOption('count');
      await hierarchy.getByLabel('Requested cluster count k').press('Home');
      await checkText(hierarchy.locator('.kh-readout'), /1 clusters after 5 merges/);
      await hierarchy.getByLabel('Requested cluster count k').press('End');
      await checkText(hierarchy.locator('.kh-readout'), /6 clusters after 0 merges/);
      await hierarchy.getByLabel('Requested cluster count k').press('ArrowLeft');
      await hierarchy.getByLabel('Requested cluster count k').press('ArrowLeft');
      await checkText(hierarchy.locator('.kh-readout'), /4 clusters after 2 merges/);
      await hierarchy.getByRole('button', { name: 'Set the tied height' }).click();
      await checkText(hierarchy.locator('.kh-readout'), /3 clusters after 3 merges/);
    }
    await hierarchy.getByLabel('Cut height').press('Home');
    await checkText(hierarchy.locator('.kh-readout'), /6 clusters/);
    await hierarchy.getByRole('button', { name: 'Next merge' }).click();
    await checkText(hierarchy.locator('.kh-readout'), /5 clusters after 1 merges/);
    await hierarchy.getByRole('button', { name: 'Undo merge' }).click();
    await checkText(hierarchy.locator('.kh-readout'), /6 clusters after 0 merges/);
    await hierarchy.getByLabel('Point configuration').selectOption('chain');
    await checkText(hierarchy.locator('.kh-readout'), /2 clusters after 7 merges/);
    await hierarchy.getByLabel('Linkage definition').selectOption('complete');
    await checkText(hierarchy.locator('.kh-table-scroll').nth(0), /P0, P1, P2, P3/);
    await hierarchy.getByLabel('Predict first', { exact: false }).selectOption('not-complete');
    await hierarchy.getByRole('button', { name: 'Check', exact: true }).click();
    await checkText(hierarchy.locator('.kh-feedback'), /Your prediction matches/);
    await hierarchy.getByRole('button', { name: 'Reset', exact: true }).click();
    await checkText(hierarchy.locator('.kh-readout'), /3 clusters after 3 merges/);
    records.push({ case: 'All four linkages, exact count prefixes, whole tied-height cuts, merge stepping, chain fixture contrast with prediction, endpoints and reset' });

    const palette = page.locator('.kh-investigation').nth(4);
    await palette.getByLabel('Requested palette size').press('End');
    await checkText(palette.locator('.kh-readout'), /floating-center SSE: 0/);
    await checkText(palette.locator('.kh-readout'), /integer-RGB SSE: 0/);
    await palette.getByLabel('Requested palette size').press('Home');
    await checkText(palette.locator('.kh-readout'), /692609/);
    await palette.getByLabel('Image to quantize').selectOption('gradient');
    await checkText(palette.locator('.kh-readout'), /160 unique colors in 160 pixels/);
    await palette.getByLabel('Requested palette size').press('End');
    await checkText(palette.locator('.kh-readout'), /integer-RGB SSE: 165886/);
    await palette.getByLabel('Image to quantize').selectOption('sky');
    await checkText(palette.locator('.kh-readout'), /37 unique colors/);
    await palette.getByRole('button', { name: 'Reset', exact: true }).click();
    await checkText(palette.locator('.kh-readout'), /integer-RGB SSE: 44910/);
    records.push({ case: 'Palette keyboard endpoints, three images, actual byte-color error and reset' });

    const firstHint = page.locator('.cluster-practice').first().getByText('Get a hint', { exact: true });
    await firstHint.focus();
    await page.keyboard.press('Enter');
    assert.ok(await firstHint.evaluate(element => element.parentElement.open));
    await page.locator('.cluster-practice').first().getByText('Show the explained solution', { exact: true }).click();
    await checkText(page.locator('.cluster-practice').first(), /86\/3/);
    const screenshot = async (locator, filename) => {
      const destination = `docs/teaching/evidence/${filename}`;
      const viewport = page.viewportSize();
      const bounds = await locator.boundingBox();
      // Keep the fixed site header outside the selected evidence region.
      if (bounds.height > viewport.height - 160) await page.setViewportSize({ width: viewport.width, height: Math.ceil(bounds.height) + 180 });
      await locator.scrollIntoViewIfNeeded();
      await locator.evaluate(element => window.scrollTo(0, element.getBoundingClientRect().top + window.scrollY - 100));
      await locator.screenshot({ path: destination });
      await page.setViewportSize(viewport);
      screenshotPaths.push(destination);
    };
    await next.click();
    await next.click();
    await screenshot(lloyd.locator('.kh-scatter'), 'k-means-lloyd-desktop.png');
    await screenshot(seeding, 'k-means-seeding-desktop.png');
    const localScripts = [...new Set(requests.map(address => new URL(address)).filter(address => address.origin === new URL(base).origin && address.pathname.endsWith('.js')).map(address => address.pathname.slice(1)))];
    assert.deepEqual(localScripts.filter(filename => bodyFiles.has(filename)), [bodyFile]);
    assert.deepEqual(localScripts.filter(filename => !allowedScripts.has(filename)), []);
    assert.ok(!requests.some(address => address.includes('/outlines/')));
    records.push({ case: 'Fresh route requests only selected lesson and actual shared closure', requestedScripts: localScripts, rawScriptBytes: localScripts.reduce((sum, filename) => sum + fs.statSync(path.join('dist', filename)).size, 0), gzipEstimateBytes: localScripts.reduce((sum, filename) => sum + gzipSync(fs.readFileSync(path.join('dist', filename))).length, 0), interpretation: 'Built requested file sizes; gzip estimate is not measured network compression or latency.' });

    for (const width of [390, 320]) {
      await page.setViewportSize({ width, height: 844 });
      await page.locator('.clustering-lesson details').evaluateAll(items => items.forEach(item => { item.open = true; }));
      await settle(page);
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `Document overflows at ${width}px`);
      const mathOverflow = await page.locator('.clustering-lesson .katex-display').evaluateAll(items => items.filter(item => item.scrollWidth > item.clientWidth + 1).map(item => item.textContent));
      assert.deepEqual(mathOverflow, [], `Unnecessarily overflowing formulas at ${width}px`);
      assert.equal(await page.locator('.katex-error').count(), 0);
      if (width === 390) {
        await screenshot(hierarchy.locator('.kh-dendrogram'), 'k-means-tree-mobile.png');
        await screenshot(page.locator('.cluster-figure').nth(1), 'k-means-mean-mobile.png');
        await screenshot(page.locator('.cluster-figure').nth(0), 'k-means-faithful-scatter-mobile.png');
        await screenshot(page.locator('.cluster-figure').nth(5), 'k-means-diagnostics-mobile.png');
        await screenshot(page.locator('.cluster-figure').nth(3), 'k-means-faithful-tree-mobile.png');
        await screenshot(hierarchy, 'k-means-hierarchy-chain-mobile.png');
        await screenshot(palette.locator('.kh-image-pair'), 'k-means-palette-mobile.png');
      }
      records.push({ case: `Narrow ${width}px layout, readable formula grouping and all deeper branches rendered` });
    }
    await page.setViewportSize({ width: 1366, height: 1000 });
    await page.locator('.reader-complete').click();
    assert.equal(new URL(page.url()).pathname.split('/').at(-1), topicId);
    await page.reload({ waitUntil: 'domcontentloaded' });
    await ready(page);
    assert.equal(await page.evaluate(id => JSON.parse(localStorage.getItem('kd-progress'))[id], topicId), true);
    await checkText(moduleGroup.locator('.reader-group__completed'), /1 completed/);
    await page.locator('.reader-footer__next').click();
    await page.waitForURL(`**/pca-dimensionality-reduction?module=classical-ml`);
    await checkText(page.locator('.reader-header h1'), /PCA & Dimensionality Reduction/);
    records.push({ case: 'Completion persists under stable ID without auto-advance; Next opens actual PCA successor' });
    assert.deepEqual(errors, []);
    assert.deepEqual(failedAssets, []);
    await context.close();

    // New lesson failure stays visibly published and recoverable. Shared stale-import
    // behavior is unchanged and retains its previously recorded integration evidence.
    for (const failure of ['import', 'render']) {
      const isolated = await browser.newContext();
      const trial = await isolated.newPage();
      let inject = true;
      await trial.route(`**/${bodyFile}`, async intercepted => {
        if (!inject) return intercepted.continue();
        inject = false;
        if (failure === 'import') return intercepted.abort('failed');
        return intercepted.fulfill({ status: 200, contentType: 'text/javascript', body: 'export default {content(){throw new Error("Controlled clustering review failure")}}' });
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
    const report = { startedAt, completedAt: new Date().toISOString(), topicId, status: 'passed', browser: await browser.version(), buildManifestHash: buildHash, sourceHashes, catalogueIds: Object.keys(topicCatalogue).length, publicationMappings: Object.keys(publications).length, moduleTopicCount: module.topicIds.length, records, screenshots: screenshotPaths, visualInterpretation: 'Intended Space Grotesk font loaded before checks. Tall element captures temporarily extend viewport height to keep the fixed site header outside the selected region. Screenshots require the author’s separate visual inspection; structural checks alone do not establish pedagogical quality.' };
    fs.writeFileSync(evidencePath, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify({ status: report.status, cases: records.length, evidencePath, sourceFiles: ownedFiles.length }));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });

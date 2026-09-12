// Production browser review of the DBSCAN lesson: visible content, the four
// investigations, narrow layouts, module sequence, completion and recovery.
// DIST_DIR and LEARNING_BASE_URL allow a per-topic build/preview.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { gzipSync } = require('node:zlib');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');

const distDir = process.env.DIST_DIR || 'dist';
const base = (process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4173').replace(/\/+$/, '');
const topicId = 'dbscan-density-based-clustering';
const sourcePath = `src/learn/data/topics/${topicId}.jsx`;
const route = `${base}/learn/path/full-curriculum/${topicId}?module=classical-ml`;
const evidencePath = 'docs/teaching/evidence/dbscan-browser.json';
const read = filename => JSON.parse(fs.readFileSync(filename, 'utf8'));
const hash = filename => createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
const normalize = text => text.replace(/\s+/g, ' ').trim();
const ownedFiles = [sourcePath, 'src/learn/data/dbscan-models.js', 'src/learn/data/dbscan-iris-data.js', 'src/learn/data/dbscan-examples.js', 'src/learn/components/lesson-labs/DbscanLabs.jsx', 'src/learn/components/lesson-labs/DbscanFigures.jsx', 'src/learn/components/lesson-labs/dbscan-labs.css', `src/learn/data/curriculum/blueprints/${topicId}.js`, 'public/learn-assets/dbscan/iris.csv'];

(async () => {
  const startedAt = new Date().toISOString();
  const sourceHashes = Object.fromEntries(ownedFiles.map(filename => [filename, hash(filename)]));
  const manifestPath = path.join(distDir, '.vite/manifest.json');
  const build = read(manifestPath);
  const buildHash = hash(manifestPath);
  const publications = read('src/learn/data/lesson-manifest.json');
  const { tracks } = await import('../src/learn/data/tracks.js');
  const { dbscanExamples } = await import('../src/learn/data/dbscan-examples.js');
  const module = tracks.find(track => track.id === 'classical-ml');
  assert.equal(module.topicIds[13], topicId);
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
    await page.locator('.dbscan-lesson').waitFor();
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
    await checkText(page.locator('.reader-header h1'), /^DBSCAN & Density-Based Clustering$/);
    await checkText(page.locator('.reader-header__meta'), /14 of 39 topics on this route/);
    await checkText(page.locator('.reader-footer__previous'), /Clustering Evaluation & Validation/);
    await checkText(page.locator('.reader-footer__next'), /Anomaly & Outlier Detection/);
    assert.equal(await page.locator('.db-investigation').count(), 4);
    assert.equal(await page.locator('.db-figure').count(), 7);
    assert.equal(await page.locator('.db-practice').count(), 12);
    assert.equal(await page.locator('.python-example').count(), 9);
    await page.locator('.dbscan-lesson details').evaluateAll(items => items.forEach(item => { item.open = true; }));
    const rendered = normalize(await page.locator('.dbscan-lesson').textContent());
    for (const [key, example] of Object.entries(dbscanExamples)) {
      assert.ok(rendered.includes(normalize(example.code)), `Missing actual code for ${key}`);
      assert.ok(rendered.includes(normalize(example.expected)), `Missing actual expected output for ${key}`);
    }
    for (const anchor of await page.locator('.lesson-intro a[href^="#"]').evaluateAll(items => items.map(item => item.getAttribute('href')))) {
      assert.equal(await page.locator(`[id="${anchor.slice(1)}"]`).count(), 1, `Broken lesson anchor ${anchor}`);
    }
    assert.equal(await page.locator('.katex-error').count(), 0);
    const csv = await page.request.get(`${base}/learn-assets/dbscan/iris.csv`);
    assert.equal(csv.status(), 200);
    assert.equal((await csv.text()).trim().split('\n').length, 151, 'CSV serves header plus 150 rows');
    await page.locator('.dbscan-lesson details').evaluateAll(items => items.forEach(item => { item.open = false; }));
    records.push({ case: 'Complete visible code/output for nine programs, fourteen route anchors, seven figures, twelve practice tasks, downloadable CSV, current metadata and module sequence' });

    const trail = page.locator('.db-investigation').nth(0);
    assert.equal(await trail.locator('.db-hidden').count(), 1, 'Trail result hidden before commitment');
    await trail.getByLabel('Number of core components').selectOption('2');
    await trail.getByLabel('Type of I').selectOption('border');
    await trail.getByRole('button', { name: 'Commit prediction and apply' }).click();
    await checkText(trail.locator('.db-feedback'), /Your prediction matches/);
    await checkText(trail.locator('.db-readout'), /2 core components.*\{A, B, C, D\} and \{E, F, G, H\}/);
    await checkText(trail.locator('.db-readout'), /Core 8, border 1, noise 1/);
    await trail.getByLabel('Radius ε, meters', { exact: false }).fill('1.25');
    await checkText(trail.locator('.db-feedback'), /inputs changed after your last comparison/);
    await trail.getByLabel('Number of core components').selectOption('1');
    await trail.getByLabel('Type of I').selectOption('core');
    await trail.getByRole('button', { name: 'Commit prediction and apply' }).click();
    await checkText(trail.locator('.db-feedback'), /Your prediction matches/);
    await checkText(trail.locator('.db-readout'), /1 core component:/);
    await trail.getByLabel('Radius ε, meters', { exact: false }).fill('1');
    await trail.getByLabel('Visiting order').selectOption('reversed');
    await trail.getByLabel('Number of core components').selectOption('2');
    await trail.getByLabel('Type of I').selectOption('border');
    await trail.getByRole('button', { name: 'Commit prediction and apply' }).click();
    await checkText(trail.locator('.db-readout'), /Visit order: J I H G F E D C B A/);
    await trail.getByLabel('I y, meters', { exact: false }).fill('0.125');
    await trail.getByLabel('Type of I').selectOption('noise');
    await trail.getByRole('button', { name: 'Commit prediction and apply' }).click();
    await checkText(trail.locator('.db-feedback'), /Your prediction matches/);
    await trail.getByLabel('Point configuration').selectOption('duplicates');
    await trail.getByLabel('Number of core components').selectOption('1');
    await trail.getByLabel('Type of P').selectOption('core');
    await trail.getByRole('button', { name: 'Commit prediction and apply' }).click();
    await checkText(trail.locator('.db-feedback'), /Your prediction matches/);
    await trail.getByRole('button', { name: 'Reset', exact: true }).click();
    assert.equal(await trail.locator('.db-hidden').count(), 1);
    records.push({ case: 'Trail lab: hidden until commitment, ε 1 and 1.25 contrasts, stale invalidation, reversed visiting order, moved I null, duplicate rows, reset' });

    const metric = page.locator('.db-investigation').nth(1);
    await metric.getByRole('button', { name: /Convert only y and ε by 100/ }).click();
    await metric.getByLabel('Same neighbourhoods?').selectOption('yes');
    await metric.getByRole('button', { name: 'Commit prediction and apply' }).click();
    await checkText(metric.locator('.db-feedback'), /Your prediction matches/);
    await metric.getByLabel('Which corners').selectOption('five');
    await metric.getByLabel('Same neighbourhoods?').selectOption('no');
    await metric.getByRole('button', { name: 'Commit prediction and apply' }).click();
    await checkText(metric.locator('.db-feedback'), /Your prediction matches.*neighbour sets changed: P, Q, T/);
    await metric.getByRole('button', { name: /Convert both coordinates and ε by 100/ }).click();
    await metric.getByLabel('Same neighbourhoods?').selectOption('yes');
    await metric.getByRole('button', { name: 'Commit prediction and apply' }).click();
    await checkText(metric.locator('.db-feedback'), /Your prediction matches/);
    await metric.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'Metric lab: four-corner accidental null, five-row faulty conversion changes T, uniform conversion preserves the graph, reset' });

    const iris = page.locator('.db-investigation').nth(2);
    await iris.getByLabel('Returned groups').selectOption('2');
    await iris.getByLabel('Coverage', { exact: false }).selectOption('gt75');
    await iris.getByRole('button', { name: 'Commit and reveal the report' }).click();
    await checkText(iris.locator('.db-feedback'), /Your prediction matches/);
    await checkText(iris.locator('.db-table-scroll').first(), /116 \/ 150 = 77\.3%/);
    await checkText(iris.locator('.db-table-scroll').first(), /hidden until species are revealed/);
    await iris.getByRole('button', { name: 'Save this report as snapshot A' }).click();
    await iris.getByRole('button', { name: /Reveal species/ }).click();
    await checkText(iris.locator('.db-table-scroll').first(), /0\.442/);
    await iris.getByLabel('Count m, rows including self', { exact: false }).fill('10');
    await checkText(iris.locator('.db-feedback'), /inputs changed/);
    await iris.getByLabel('Returned groups').selectOption('3');
    await iris.getByLabel('Coverage', { exact: false }).selectOption('25to50');
    await iris.getByRole('button', { name: 'Commit and reveal the report' }).click();
    await checkText(iris.locator('.db-feedback'), /Your prediction matches/);
    await checkText(iris.locator('.db-table-scroll').first(), /61 \/ 150 = 40\.7%/);
    await iris.getByRole('button', { name: /Reveal species/ }).click();
    await checkText(iris.locator('.db-table-scroll').first(), /0\.279/);
    await checkText(iris.locator('.db-table-scroll').nth(1), /61 rows assigned by both/);
    await checkText(iris.locator('.db-table-scroll').nth(1), /ARI vs species on the common rows only/);
    await checkText(iris.locator('.db-table-scroll').nth(1), /0\.848/);
    await checkText(iris.locator('.db-table-scroll').nth(1), /61 rows, same population for both/);
    assert.equal(await iris.getByLabel('Radius ε in standardized units', { exact: false }).count(), 1, 'standardized slider label');
    await iris.getByLabel('Feature scaling').selectOption('raw');
    assert.equal(await iris.getByLabel('Radius ε in centimetres', { exact: false }).count(), 1, 'raw slider label names centimetres');
    await iris.getByLabel('Count m, rows including self', { exact: false }).fill('5');
    await iris.getByLabel('Returned groups').selectOption('2');
    await iris.getByLabel('Coverage', { exact: false }).selectOption('gt75');
    await iris.getByRole('button', { name: 'Commit and reveal the report' }).click();
    await checkText(iris.locator('.db-table-scroll').first(), /133 \/ 150 = 88\.7%/);
    await iris.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'Iris lab: m = 5 versus m = 10 with species hidden until revealed, snapshot comparison on common rows including species ARI on the 61 common rows, slider label follows the representation, raw-feature practice K values, reset' });

    const interval = page.locator('.db-investigation').nth(3);
    await interval.getByLabel('Does a radius recover all three groups?').selectOption('inspect');
    await interval.getByLabel('Does ε = 0.25 recover all three?').selectOption('no');
    await interval.getByRole('button', { name: 'Commit prediction and apply' }).click();
    await checkText(interval.locator('.db-feedback'), /Here is the answer you chose to inspect/);
    assert.equal(await interval.locator('.db-feedback').filter({ hasText: 'Not this time' }).count(), 0, 'inspect option is not graded as a miss');
    assert.match(await interval.locator('svg.db-svg-wide').getAttribute('aria-label'), /Magnified: the two dense groups from −0\.125 to 1\.5 meters.*L1 at 0 is core.*M4 at 1\.125 is core/, 'magnified strip names every dense row');
    await interval.getByRole('button', { name: 'Reset', exact: true }).click();
    await interval.getByLabel('Does a radius recover all three groups?').selectOption('no');
    await interval.getByLabel('Does ε = 0.25 recover all three?').selectOption('no');
    await interval.getByRole('button', { name: 'Commit prediction and apply' }).click();
    await checkText(interval.locator('.db-feedback'), /Your prediction matches/);
    await checkText(interval.locator('.db-readout'), /No radius satisfies both requirements/);
    await interval.getByLabel('Spacing inside the right group', { exact: false }).fill('0.25');
    await interval.getByLabel('Does a radius recover all three groups?').selectOption('yes');
    await interval.getByLabel('Does ε = 0.25 recover all three?').selectOption('yes');
    await interval.getByRole('button', { name: 'Commit prediction and apply' }).click();
    await checkText(interval.locator('.db-readout'), /Usable interval \[0\.25, 0\.375\)/);
    await interval.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'Interval lab: inspect option is neutral, baseline has no radius, magnified dense-group strip, practice F spacing gives [0.25, 0.375), reset' });

    const firstHint = page.locator('.db-practice').first().getByText('Get a hint', { exact: true });
    await firstHint.focus();
    await page.keyboard.press('Enter');
    assert.ok(await firstHint.evaluate(element => element.parentElement.open));
    await page.locator('.db-practice').first().getByText('Show the explained solution', { exact: true }).click();
    await checkText(page.locator('.db-practice').first(), /Counts are 2, 3, 3, 2, 1/);
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
    await trail.getByLabel('Number of core components').selectOption('2');
    await trail.getByLabel('Type of I').selectOption('border');
    await trail.getByRole('button', { name: 'Commit prediction and apply' }).click();
    await screenshot(trail, 'dbscan-trail-desktop.png');
    await screenshot(page.locator('.db-figure').nth(1), 'dbscan-border-figure-desktop.png');
    await screenshot(page.locator('.db-figure').nth(2), 'dbscan-core-radius-desktop.png');
    await iris.getByLabel('Returned groups').selectOption('2');
    await iris.getByLabel('Coverage', { exact: false }).selectOption('gt75');
    await iris.getByRole('button', { name: 'Commit and reveal the report' }).click();
    await iris.getByRole('button', { name: /Reveal species/ }).click();
    await screenshot(iris, 'dbscan-iris-desktop.png');
    await interval.getByLabel('Does a radius recover all three groups?').selectOption('no');
    await interval.getByLabel('Does ε = 0.25 recover all three?').selectOption('no');
    await interval.getByRole('button', { name: 'Commit prediction and apply' }).click();
    await screenshot(interval, 'dbscan-interval-desktop.png');
    await screenshot(page.locator('.db-figure').nth(5), 'dbscan-optics-desktop.png');
    await page.locator('.dbscan-lesson details.db-deeper').evaluateAll(items => items.forEach(item => { item.open = true; }));
    await settle(page);
    await screenshot(page.locator('.db-figure').nth(4), 'dbscan-rings-desktop.png');
    const localScripts = [...new Set(requests.map(address => new URL(address)).filter(address => address.origin === new URL(base).origin && address.pathname.endsWith('.js')).map(address => address.pathname.slice(1)))];
    assert.deepEqual(localScripts.filter(filename => bodyFiles.has(filename)), [bodyFile]);
    assert.deepEqual(localScripts.filter(filename => !allowedScripts.has(filename)), []);
    assert.ok(!requests.some(address => address.includes('/outlines/')));
    records.push({ case: 'Fresh route requests only the selected lesson and its shared closure', requestedScripts: localScripts, rawScriptBytes: localScripts.reduce((sum, filename) => sum + fs.statSync(path.join(distDir, filename)).size, 0), gzipEstimateBytes: localScripts.reduce((sum, filename) => sum + gzipSync(fs.readFileSync(path.join(distDir, filename))).length, 0), interpretation: 'Built requested file sizes; gzip estimate is not measured network compression or latency.' });

    for (const width of [390, 320]) {
      await page.setViewportSize({ width, height: 844 });
      await page.locator('.dbscan-lesson details').evaluateAll(items => items.forEach(item => { item.open = true; }));
      await settle(page);
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `Document overflows at ${width}px`);
      const mathOverflow = await page.locator('.dbscan-lesson .katex-display').evaluateAll(items => items.filter(item => item.scrollWidth > item.clientWidth + 1).map(item => item.textContent));
      assert.deepEqual(mathOverflow, [], `Overflowing formulas at ${width}px`);
      assert.equal(await page.locator('.katex-error').count(), 0);
      const smallText = await page.locator('.db-figure svg text, .db-investigation svg text').evaluateAll(items => items.filter(item => item.getClientRects().length > 0 && item.getBoundingClientRect().height < 11.5).map(item => `${item.textContent} ${item.getBoundingClientRect().height.toFixed(1)}px`));
      assert.deepEqual(smallText, [], `SVG text rendered below 11.5 px at ${width}px`);
      if (width === 390) {
        await screenshot(page.locator('.db-figure').nth(0), 'dbscan-roster-mobile.png');
        await screenshot(page.locator('.db-figure').nth(3), 'dbscan-interval-figure-mobile.png');
        await screenshot(page.locator('.db-figure').nth(6), 'dbscan-stability-mobile.png');
        await screenshot(metric, 'dbscan-metric-mobile.png');
      }
      if (width === 320) {
        await screenshot(page.locator('.db-figure').nth(2), 'dbscan-core-radius-320.png');
        await screenshot(page.locator('.db-figure').nth(3), 'dbscan-interval-figure-320.png');
        await screenshot(page.locator('.db-figure').nth(5), 'dbscan-optics-320.png');
        await screenshot(page.locator('.db-figure').nth(6), 'dbscan-stability-320.png');
        await screenshot(interval, 'dbscan-interval-lab-320.png');
        await screenshot(page.locator('.db-figure').nth(0), 'dbscan-roster-320.png');
      }
      records.push({ case: `Narrow ${width}px layout, readable formula grouping, no SVG text below 11.5 px, all deeper branches rendered` });
    }
    await page.setViewportSize({ width: 1366, height: 1000 });
    await page.locator('.reader-complete').click();
    assert.equal(new URL(page.url()).pathname.split('/').at(-1), topicId);
    await page.reload({ waitUntil: 'domcontentloaded' });
    await ready(page);
    assert.equal(await page.evaluate(id => JSON.parse(localStorage.getItem('kd-progress'))[id], topicId), true);
    await page.locator('.reader-footer__next').click();
    await page.waitForURL('**/anomaly-outlier-detection-isolation-forest-one-class-svm-lof?module=classical-ml');
    await page.waitForFunction(() => document.querySelector('.reader-header h1')?.textContent.includes('Anomaly'), null, { timeout: 20000 });
    await checkText(page.locator('.reader-header h1'), /Anomaly & Outlier Detection/);
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
        return intercepted.fulfill({ status: 200, contentType: 'text/javascript', body: 'export default {content(){throw new Error("Controlled DBSCAN review failure")}}' });
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
    assert.equal(hash(manifestPath), buildHash);
    const report = { startedAt, completedAt: new Date().toISOString(), topicId, status: 'passed', browser: await browser.version(), distDir, buildManifestHash: buildHash, sourceHashes, moduleTopicCount: module.topicIds.length, records, screenshots: screenshotPaths, visualInterpretation: 'Intended Space Grotesk font loaded before checks. Screenshots capture informative states: compared predictions, the revealed Iris report with species, the incompatible-interval baseline, the OPTICS and rings figures. They require separate visual inspection; structural checks alone do not establish teaching quality.' };
    fs.writeFileSync(evidencePath, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify({ status: report.status, cases: records.length, evidencePath, sourceFiles: ownedFiles.length }));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });

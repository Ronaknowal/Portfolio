// Production browser review of the clustering-evaluation lesson: visible
// content, the five investigations, narrow layouts, sequence, completion, recovery.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { gzipSync } = require('node:zlib');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');

const distDir = process.env.DIST_DIR || 'dist';
const base = (process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4173').replace(/\/+$/, '');
const topicId = 'clustering-evaluation-validation-silhouette-ari-nmi';
const sourcePath = `src/learn/data/topics/${topicId}.jsx`;
const route = `${base}/learn/path/full-curriculum/${topicId}?module=classical-ml`;
const evidencePath = 'docs/teaching/evidence/clustering-evaluation-browser.json';
const read = filename => JSON.parse(fs.readFileSync(filename, 'utf8'));
const hash = filename => createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
const normalize = text => text.replace(/\s+/g, ' ').trim();
const ownedFiles = [sourcePath, 'src/learn/data/clustering-evaluation-models.js', 'src/learn/data/clustering-evaluation-data.js', 'src/learn/data/clustering-evaluation-examples.js', 'src/learn/components/lesson-labs/ClusteringEvaluationLabs.jsx', 'src/learn/components/lesson-labs/ClusteringEvaluationFigures.jsx', 'src/learn/components/lesson-labs/clustering-evaluation-labs.css', `src/learn/data/curriculum/blueprints/${topicId}.js`, 'public/learn-assets/clustering-evaluation/iris.csv'];

(async () => {
  const startedAt = new Date().toISOString();
  const sourceHashes = Object.fromEntries(ownedFiles.map(filename => [filename, hash(filename)]));
  const build = read(`${distDir}/.vite/manifest.json`);
  const buildHash = hash(`${distDir}/.vite/manifest.json`);
  const publications = read('src/learn/data/lesson-manifest.json');
  const { tracks } = await import('../src/learn/data/tracks.js');
  const { clusteringEvaluationExamples } = await import('../src/learn/data/clustering-evaluation-examples.js');
  const module = tracks.find(track => track.id === 'classical-ml');
  assert.equal(module.topicIds[12], topicId);
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
    await page.locator('.ce-lesson').waitFor();
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
    await checkText(page.locator('.reader-header h1'), /^Clustering Evaluation & Validation \(Silhouette, ARI, NMI\)$/);
    await checkText(page.locator('.reader-header__meta'), /13 of 39 topics on this route/);
    await checkText(page.locator('.reader-footer__previous'), /PCA & Dimensionality Reduction/);
    await checkText(page.locator('.reader-footer__next'), /DBSCAN & Density-Based Clustering/);
    assert.equal(await page.locator('.ce-investigation').count(), 5);
    assert.equal(await page.locator('.ce-figure').count(), 7);
    assert.equal(await page.locator('.ce-practice').count(), 8);
    assert.equal(await page.locator('.python-example').count(), 6);
    const rendered = normalize(await page.locator('.ce-lesson').textContent());
    for (const [key, example] of Object.entries(clusteringEvaluationExamples)) {
      assert.ok(rendered.includes(normalize(example.code)), `Missing actual code for ${key}`);
      assert.ok(rendered.includes(normalize(example.expected)), `Missing actual expected output for ${key}`);
    }
    for (const anchor of await page.locator('.lesson-intro a[href^="#"]').evaluateAll(items => items.map(item => item.getAttribute('href')))) {
      assert.equal(await page.locator(`[id="${anchor.slice(1)}"]`).count(), 1, `Broken lesson anchor ${anchor}`);
    }
    assert.equal(await page.locator('.katex-error').count(), 0);
    const csv = await page.request.get(`${base}/learn-assets/clustering-evaluation/iris.csv`);
    assert.equal(csv.status(), 200);
    assert.equal((await csv.text()).trim().split('\n').length, 151, 'CSV serves header plus 150 rows');
    records.push({ case: 'Complete visible code/output for six programs, twelve route anchors, seven figures, eight practice tasks, downloadable CSV, current metadata and module sequence' });

    const silhouette = page.locator('.ce-investigation').nth(0);
    await checkText(silhouette.locator('.ce-readout'), /C has a = 1\.5, b = 6 .* s = 0\.75/);
    await silhouette.getByRole('button', { name: 'Move C to the other group' }).click();
    await silhouette.getByLabel('Predict first', { exact: false }).selectOption('sign:negative');
    await silhouette.getByRole('button', { name: 'Apply and compare' }).click();
    await checkText(silhouette.locator('.ce-feedback'), /Your prediction matches.*Mean moved from 0\.8065 to 0\.4594; s\(C\) is now -0\.75/);
    await silhouette.getByLabel(/^Selected observation/).selectOption('3');
    await checkText(silhouette.locator('.ce-feedback'), /inputs changed after your last comparison/);
    await silhouette.getByLabel(/^Selected observation/).selectOption('2');
    await checkText(silhouette.locator('.ce-feedback'), /s\(C\) is now -0\.75/);
    await silhouette.getByLabel('C location', { exact: false }).fill('3');
    await checkText(silhouette.locator('.ce-feedback'), /inputs changed after your last comparison/);
    await silhouette.getByRole('button', { name: 'Move C to the other group' }).click();
    await silhouette.getByLabel('Predict first', { exact: false }).selectOption('sign:positive');
    await silhouette.getByRole('button', { name: 'Apply and compare' }).click();
    await checkText(silhouette.locator('.ce-readout'), /a = 2\.5, b = 5 .* s = 0\.5/);
    await silhouette.getByRole('button', { name: 'Double every draft coordinate' }).click();
    await silhouette.getByLabel('Predict first', { exact: false }).selectOption('mean:same');
    await silhouette.getByRole('button', { name: 'Apply and compare' }).click();
    await checkText(silhouette.locator('.ce-feedback'), /Your prediction matches: overall mean the same/);
    for (const name of ['A', 'B', 'D', 'E', 'F']) await silhouette.getByLabel(`${name} group`, { exact: false }).selectOption('L');
    await silhouette.getByLabel('C group', { exact: false }).selectOption('L');
    await silhouette.getByLabel('Predict first', { exact: false }).selectOption('sign:undefined');
    await silhouette.getByRole('button', { name: 'Apply and compare' }).click();
    await checkText(silhouette.locator('.ce-readout'), /Silhouette is undefined for the applied state: fewer than two groups/);
    await silhouette.getByRole('button', { name: 'Reset', exact: true }).click();
    await checkText(silhouette.locator('.ce-readout'), /s = 0\.75/);
    records.push({ case: 'Silhouette lab: membership change to −0.75 with recomputed mean, stale invalidation on a changed selection and on a learner-entered coordinate, geometry change to 0.5, common-scale invariance, undefined one-group state, reset' });

    const pairs = page.locator('.ce-investigation').nth(1);
    await checkText(pairs.locator('.ce-readout'), /S = 6, A = 12, B = 12, M = 28 .* RI = 0\.5714.* ARI = 0\.125/);
    await pairs.getByRole('button', { name: /Rename every candidate label/ }).click();
    await pairs.getByLabel('Predict first', { exact: false }).selectOption('ari:same');
    await pairs.getByRole('button', { name: 'Apply and compare' }).click();
    await checkText(pairs.locator('.ce-feedback'), /Your prediction matches: ARI the same/);
    await pairs.getByLabel('Selected pair').selectOption('1-2');
    await checkText(pairs.locator('.ce-feedback'), /inputs changed after your last comparison/);
    await pairs.getByRole('button', { name: 'Cross the labels' }).click();
    await pairs.getByLabel('Predict first', { exact: false }).selectOption('ari:lower');
    await pairs.getByRole('button', { name: 'Apply and compare' }).click();
    await checkText(pairs.locator('.ce-readout'), /RI = 0\.4286.* ARI = -0\.1667/);
    await pairs.getByRole('button', { name: 'Refine into four pairs' }).click();
    await pairs.getByLabel('Predict first', { exact: false }).selectOption('ari:higher');
    await pairs.getByRole('button', { name: 'Apply and compare' }).click();
    await checkText(pairs.locator('.ce-readout'), /RI = 0\.7143.* ARI = 0\.3636/);
    await pairs.getByLabel('H in V', { exact: false }).selectOption('7');
    await checkText(pairs.locator('.ce-feedback'), /inputs changed/);
    assert.equal(await pairs.locator('.ce-tile:not(.is-blank)').count(), 28);
    await pairs.locator('.ce-tile:not(.is-blank)').nth(3).click();
    await checkText(pairs.locator('.ce-readout'), /Selected pair A–E/);
    await pairs.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'Pair lab: rename leaves ARI fixed, cross gives −1/6, refinement gives 4/11, stale invalidation, 28 selectable pair tiles, reset' });

    const chance = page.locator('.ce-investigation').nth(2);
    await checkText(chance.locator('.ce-readout'), /70 distinct assignments.* mean NMI 0\.1148/);
    await chance.getByLabel('Predict first', { exact: false }).selectOption('ami:negative');
    await chance.getByRole('button', { name: 'Apply and compare' }).click();
    await checkText(chance.locator('.ce-feedback'), /Your prediction matches: Observed AMI negative/);
    assert.ok(await chance.getByLabel('Predict first', { exact: false }).isDisabled(), 'Compared predictions cannot be rewritten after the result is visible');
    await chance.getByRole('button', { name: 'Unbalanced 3/5 candidate' }).click();
    await chance.getByLabel('Predict first', { exact: false }).selectOption('null:positive');
    await chance.getByRole('button', { name: 'Apply and compare' }).click();
    await checkText(chance.locator('.ce-readout'), /56 distinct assignments/);
    await chance.getByRole('button', { name: 'Constant candidate' }).click();
    await chance.getByLabel('Predict first', { exact: false }).selectOption('ami:degenerate');
    await chance.getByRole('button', { name: 'Apply and compare' }).click();
    await checkText(chance.locator('.ce-feedback'), /Your prediction matches: AMI has a degenerate null/);
    assert.equal(await chance.locator('.ce-matrix').evaluate(element => getComputedStyle(element).gridTemplateColumns.split(' ').length), 3, 'A constant candidate has one group column plus row labels and totals');
    await chance.getByRole('button', { name: 'Reset', exact: true }).click();
    await chance.getByLabel('A in U', { exact: false }).selectOption('1');
    await chance.getByRole('button', { name: 'Match U exactly', exact: true }).click();
    for (const name of ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']) {
      assert.equal(await chance.getByLabel(`${name} in V`, { exact: false }).inputValue(), await chance.getByLabel(`${name} in U`, { exact: false }).inputValue(), 'Match copies the edited reference');
    }
    await chance.getByLabel('Predict first', { exact: false }).selectOption('ami:positive');
    await chance.getByRole('button', { name: 'Apply and compare' }).click();
    await checkText(chance.locator('.ce-feedback'), /Your prediction matches: Observed AMI positive/);
    await checkText(chance.locator('.ce-readout'), /NMI \(arithmetic\) = 1; AMI = 1/);
    assert.equal(await chance.locator('.ce-matrix .is-highlight').getAttribute('aria-label'), 'U1 and V0: 0 observations', 'Highlighted overlap uses the first U group and smaller V label, even when V first appears as 1');
    await chance.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'Chance regression: immutable compared prediction, constant-candidate table alignment, edited-reference exact copy, highlighted overlap agrees with the null definition' });
    records.push({ case: 'Chance lab: negative AMI at overlap 2, unbalanced margins recompute 56 assignments, constant candidate reports a degenerate null, reset' });

    const iris = page.locator('.ce-investigation').nth(3);
    await checkText(iris.locator('.ce-readout'), /raw measurements \(cm\), k = 2: mean silhouette 0\.681.* species ARI 0\.5399/);
    await iris.getByLabel('Number of groups k').selectOption('3');
    await checkText(iris.locator('.ce-readout'), /k = 3: mean silhouette 0\.5528.* species ARI 0\.7302/);
    await iris.getByLabel('Mode').selectOption('rescore');
    await checkText(iris.locator('.ce-readout'), /Frozen standardized k = 3 labels .* mean silhouette 0\.4599/);
    await iris.getByRole('button', { name: 'Petal width 4' }).click();
    await iris.getByLabel('Predict first', { exact: false }).selectOption('mean:higher');
    await iris.getByRole('button', { name: 'Apply and compare' }).click();
    await checkText(iris.locator('.ce-feedback'), /Your prediction matches: Mean higher.*from 0\.4599 to 0\.4712/);
    await checkText(iris.locator('.ce-readout'), /Species ARI 0\.6201 and AMI 0\.6552: same memberships/);
    await iris.getByRole('button', { name: 'Common weight 4 on every feature' }).click();
    await iris.getByLabel('Predict first', { exact: false }).selectOption('mean:lower');
    await iris.getByRole('button', { name: 'Apply and compare' }).click();
    await checkText(iris.locator('.ce-feedback'), /Your prediction matches: Mean lower. Mean moved from 0\.4712 to 0\.4599/);
    await iris.getByRole('button', { name: 'Reset', exact: true }).click();
    await iris.getByLabel('Mode').selectOption('rescore');
    await iris.getByRole('button', { name: 'Common weight 4 on every feature' }).click();
    await iris.getByLabel('Predict first', { exact: false }).selectOption('mean:same');
    await iris.getByRole('button', { name: 'Apply and compare' }).click();
    await checkText(iris.locator('.ce-feedback'), /Your prediction matches: Mean the same. Mean moved from 0\.4599 to 0\.4599/);
    await iris.getByLabel('Selected specimen', { exact: false }).selectOption('106');
    await checkText(iris.locator('.ce-readout'), /Row 106 \(virginica/);
    await checkText(iris.locator('.ce-feedback'), /inputs changed after your last comparison/);
    await iris.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'Iris lab: declared fits reproduce the raw k2/k3 disagreement, frozen rescoring moves silhouette while ARI/AMI stay fixed, common weight invariance, specimen selection, reset' });

    const resample = page.locator('.ce-investigation').nth(4);
    await checkText(resample.locator('.ce-readout'), /Fit A: centers 0\.5, 6\.5.* Fit B: centers 2\.5, 8\.5.* ARI = -0\.0714/);
    await resample.getByLabel('Predict first', { exact: false }).selectOption('pair:together');
    await resample.getByLabel('C × in fit B', { exact: false }).selectOption('5');
    await resample.getByRole('button', { name: 'Apply and compare' }).click();
    assert.ok(await resample.locator('.ce-feedback').count());
    await resample.getByLabel('Selected probe pair').selectOption('0-1');
    await checkText(resample.locator('.ce-feedback'), /inputs changed after your last comparison/);
    await resample.getByLabel('Groups k in both fits').selectOption('1');
    await resample.getByLabel('Predict first', { exact: false }).selectOption('ari:higher');
    await resample.getByRole('button', { name: 'Apply and compare' }).click();
    await checkText(resample.locator('.ce-readout'), /ARI = 1 by convention/);
    await resample.locator('summary').click();
    await resample.getByRole('button', { name: 'Two tight runs, uniform weights' }).click();
    await resample.getByLabel('Groups k in both fits').selectOption('2');
    await resample.getByLabel('Predict first', { exact: false }).selectOption('ari:same');
    await resample.getByRole('button', { name: 'Apply and compare' }).click();
    await checkText(resample.locator('.ce-readout'), /Fit A: centers 1, 9, cost 4/);
    await resample.getByRole('button', { name: 'Triple every location' }).click();
    await resample.getByLabel('Predict first', { exact: false }).selectOption('ari:same');
    await resample.getByRole('button', { name: 'Apply and compare' }).click();
    await checkText(resample.locator('.ce-readout'), /Fit A: centers 3, 27, cost 36/);
    await resample.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'Resample lab: default left/right emphasis with ARI −1/14, learner weight edit, k = 1 null, changed locations 1/9 cost 4 and tripled 3/27 cost 36, reset' });

    const firstHint = page.locator('.ce-practice').first().getByText('Get a hint', { exact: true });
    await firstHint.focus();
    await page.keyboard.press('Enter');
    assert.ok(await firstHint.evaluate(element => element.parentElement.open));
    await page.locator('.ce-practice').first().getByText('Show the explained solution', { exact: true }).click();
    await checkText(page.locator('.ce-practice').first(), /b = 4 and s = 0\.5/);
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
    await silhouette.getByRole('button', { name: 'Move C to the other group' }).click();
    await silhouette.getByLabel('Predict first', { exact: false }).selectOption('sign:negative');
    await silhouette.getByRole('button', { name: 'Apply and compare' }).click();
    await screenshot(silhouette, 'clustering-evaluation-silhouette-desktop.png');
    await pairs.getByRole('button', { name: 'Cross the labels' }).click();
    await pairs.getByLabel('Predict first', { exact: false }).selectOption('ari:lower');
    await pairs.getByRole('button', { name: 'Apply and compare' }).click();
    await screenshot(pairs, 'clustering-evaluation-pairs-desktop.png');
    await screenshot(chance, 'clustering-evaluation-chance-desktop.png');
    await iris.getByLabel('Mode').selectOption('rescore');
    await iris.getByRole('button', { name: 'Petal width 4' }).click();
    await iris.getByLabel('Predict first', { exact: false }).selectOption('mean:higher');
    await iris.getByRole('button', { name: 'Apply and compare' }).click();
    await screenshot(iris, 'clustering-evaluation-iris-desktop.png');
    await screenshot(resample, 'clustering-evaluation-resample-desktop.png');
    await screenshot(page.locator('.ce-figure').nth(2), 'clustering-evaluation-rings-desktop.png');
    await screenshot(page.locator('.ce-figure').nth(6), 'clustering-evaluation-report-flow-desktop.png');
    const localScripts = [...new Set(requests.map(address => new URL(address)).filter(address => address.origin === new URL(base).origin && address.pathname.endsWith('.js')).map(address => address.pathname.slice(1)))];
    assert.deepEqual(localScripts.filter(filename => bodyFiles.has(filename)), [bodyFile]);
    assert.deepEqual(localScripts.filter(filename => !allowedScripts.has(filename)), []);
    assert.ok(!requests.some(address => address.includes('/outlines/')));
    records.push({ case: 'Fresh route requests only the selected lesson and its shared closure', requestedScripts: localScripts, rawScriptBytes: localScripts.reduce((sum, filename) => sum + fs.statSync(path.join(distDir, filename)).size, 0), gzipEstimateBytes: localScripts.reduce((sum, filename) => sum + gzipSync(fs.readFileSync(path.join(distDir, filename))).length, 0), interpretation: 'Built requested file sizes; gzip estimate is not measured network compression or latency.' });

    for (const width of [390, 320]) {
      await page.setViewportSize({ width, height: 844 });
      await page.locator('.ce-lesson details').evaluateAll(items => items.forEach(item => { item.open = true; }));
      await settle(page);
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `Document overflows at ${width}px`);
      const mathOverflow = await page.locator('.ce-lesson .katex-display').evaluateAll(items => items.filter(item => item.scrollWidth > item.clientWidth + 1).map(item => item.textContent));
      assert.deepEqual(mathOverflow, [], `Overflowing formulas at ${width}px`);
      assert.equal(await page.locator('.katex-error').count(), 0);
      if (width === 390) {
        await screenshot(page.locator('.ce-figure').nth(1), 'clustering-evaluation-fan-mobile.png');
        await screenshot(page.locator('.ce-figure').nth(5), 'clustering-evaluation-rejection-mobile.png');
        await screenshot(pairs, 'clustering-evaluation-pairs-mobile.png');
        await screenshot(resample, 'clustering-evaluation-resample-mobile.png');
        await screenshot(page.locator('.ce-figure').nth(3), 'clustering-evaluation-information-mobile.png');
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
    await page.waitForURL('**/dbscan-density-based-clustering?module=classical-ml');
    await page.waitForFunction(() => document.querySelector('.reader-header h1')?.textContent.includes('DBSCAN'), null, { timeout: 20000 });
    await checkText(page.locator('.reader-header h1'), /DBSCAN & Density-Based Clustering/);
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
      records.push({ case: `${failure} failure keeps completion disabled and recovers with explicit reload` });
      await isolated.close();
    }
    for (const [filename, expected] of Object.entries(sourceHashes)) assert.equal(hash(filename), expected, `Source changed during check: ${filename}`);
    assert.equal(hash(`${distDir}/.vite/manifest.json`), buildHash);
    const report = { startedAt, completedAt: new Date().toISOString(), topicId, status: 'passed', browser: await browser.version(), distDir, buildManifestHash: buildHash, sourceHashes, moduleTopicCount: module.topicIds.length, records, screenshots: screenshotPaths, visualInterpretation: 'Intended Space Grotesk font loaded before checks. Screenshots capture informative states: compared predictions, the crossed partition, the ring contrast, the rescored Iris workspace. They require separate visual inspection.' };
    fs.writeFileSync(evidencePath, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify({ status: report.status, cases: records.length, evidencePath, sourceFiles: ownedFiles.length }));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });

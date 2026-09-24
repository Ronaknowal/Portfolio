// Review the authorized K-Means revision without rewriting historical receipts.
const fs = require('node:fs');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { gzipSync } = require('node:zlib');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const base = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4184';
const production = process.argv.includes('--production');
const capture = !process.argv.includes('--no-captures');
const id = 'k-means-hierarchical-clustering';
const route = `${base}/learn/path/full-curriculum/${id}?module=classical-ml`;
const read = file => JSON.parse(fs.readFileSync(file, 'utf8').replace(/^\uFEFF/, ''));
const hash = file => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const baseline = read('docs/teaching/evidence/k-means-revision-review-baseline.json');
const owned = Object.keys(baseline.sourceHashes).filter(file => file.startsWith('src/'));
const sourceHashes = Object.fromEntries(owned.map(file => [file, hash(file)]));
const fontManifest = process.env.LEARNING_FONT_FIXTURES ? read(process.env.LEARNING_FONT_FIXTURES) : null;
const fontEvidence = fontManifest ? { stylesheetUrl: fontManifest.stylesheetUrl, files: Object.fromEntries([fontManifest.stylesheet, ...Object.values(fontManifest.files)].map(file => [file, hash(file)])), method: 'Unmodified existing Google Fonts assets fulfilled from retained snapshots because the browser sandbox blocks their network requests; application styles are unchanged.' } : null;
const receipt = `docs/teaching/evidence/k-means-revision-${production ? 'production' : 'browser'}.json`;
const records = [], screenshots = [];
let activeCheck = 'open reader';
const normalize = value => value.replace(/\s+/g, ' ').trim();
const check = (name, evidence = {}) => records.push({ check: name, ...evidence });
const snapshot = async (page, locator, name) => {
  if (!capture) return;
  // Final production captures focus on the two amended figures and tree-label
  // repair; preserve the earlier inspected images for unchanged representations.
  if (production && !/^(figure-(4|6)-|chain-)/.test(name)) return;
  const file = `docs/teaching/evidence/screenshots/k-means-revision-${production ? 'final-' : ''}${name}.png`;
  const style = await page.addStyleTag({ content: '.learn-nav { visibility: hidden !important; }' });
  try { await locator.screenshot({ path: file }); } finally { await style.evaluate(node => node.remove()); }
  screenshots.push({ path: file, sha256: hash(file) });
};
const settle = page => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
async function inspect(page) {
  const result = await page.locator('.clustering-lesson').evaluate(root => {
    const visible = node => node.getClientRects().length > 0;
    const ids = [...document.querySelectorAll('[id]')].map(node => node.id);
    return {
      pageOverflow: document.documentElement.scrollWidth > innerWidth + 1,
      duplicateIds: ids.filter((id, index) => ids.indexOf(id) !== index),
      anchors: [...root.querySelectorAll('a[href^="#"]')].filter(a => !document.getElementById(a.hash.slice(1))).map(a => a.hash),
      unlabeled: [...root.querySelectorAll('input, select, textarea')].filter(visible).filter(node => !node.labels?.length && !node.getAttribute('aria-label') && !node.getAttribute('aria-labelledby')).map(node => node.outerHTML),
      blueLinks: [...root.querySelectorAll('a')].filter(visible).filter(node => ['rgb(0, 0, 238)', 'rgb(85, 26, 139)'].includes(getComputedStyle(node).color)).map(node => node.textContent),
      collapsedMath: [...root.querySelectorAll('.katex svg')].filter(visible).filter(node => node.getBoundingClientRect().height < 1).map(node => node.closest('.katex').textContent),
      mathErrors: root.querySelectorAll('.katex-error').length,
      leafCollisions: [...root.querySelectorAll('.kh-tree-plot')].flatMap(plot => {
        const leaves = [...plot.querySelectorAll('.kh-leaf')].map(node => ({ text: node.textContent, box: node.getBoundingClientRect() })).sort((a, b) => a.box.left - b.box.left);
        return leaves.slice(1).filter((leaf, index) => leaf.box.left < leaves[index].box.right - 0.5).map(leaf => leaf.text);
      }),
    };
  });
  assert.equal(result.pageOverflow, false, JSON.stringify(result));
  assert.equal(result.mathErrors, 0);
  for (const key of ['duplicateIds', 'anchors', 'unlabeled', 'blueLinks', 'collapsedMath', 'leafCollisions']) assert.deepEqual(result[key], [], `${key}: ${page.viewportSize().width}`);
  return result;
}

(async () => {
  fs.writeFileSync(receipt, JSON.stringify({ status: 'in-progress', sourceHashes }, null, 2) + '\n');
  const { tracks } = await import('../src/learn/data/generated/navigation.js');
  const { clusteringExamples } = await import('../src/learn/data/k-means-hierarchical-examples.js');
  const publication = read('src/learn/data/lesson-manifest.json');
  assert.deepEqual(publication, baseline.publication);
  const module = tracks.find(track => track.id === 'classical-ml');
  assert.equal(module.topicIds[10], id);
  const build = production ? read('dist/.vite/manifest.json') : null;
  const manifestHash = production ? hash('dist/.vite/manifest.json') : null;
  if (production) for (const file of owned) assert.ok(fs.statSync(file).mtimeMs <= fs.statSync('dist/.vite/manifest.json').mtimeMs, `Build predates ${file}`);
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const freshContext = async options => {
    const context = await browser.newContext(options);
    if (fontManifest) {
      await context.route(url => url.href === fontManifest.stylesheetUrl, request => request.fulfill({ path: fontManifest.stylesheet, contentType: 'text/css' }));
      for (const [url, file] of Object.entries(fontManifest.files)) await context.route(url, request => request.fulfill({ path: file, contentType: 'font/ttf', headers: { 'access-control-allow-origin': '*' } }));
    }
    return context;
  };
  const ready = async page => {
    await page.locator('.clustering-lesson').waitFor();
    await page.waitForFunction(() => ['Space Grotesk', 'JetBrains Mono'].every(family => [...document.fonts].some(font => font.family.replaceAll('"', '') === family && font.status === 'loaded')), null, { timeout: 20000 });
    await page.evaluate(() => document.fonts.ready); await settle(page);
  };
  try {
    const context = await freshContext({ viewport: { width: 1366, height: 950 }, reducedMotion: 'reduce' });
    const page = await context.newPage(), requests = [], errors = [];
    page.on('request', request => requests.push(request.url()));
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(route); await ready(page); await inspect(page);
    const labs = page.locator('.kh-investigation');
    assert.equal(await labs.count(), 5); assert.equal(await page.locator('.cluster-figure').count(), 7);
    assert.equal(await page.locator('.cluster-practice').count(), 10); assert.equal(await page.locator('.python-example').count(), 6);
    const text = normalize(await page.locator('.clustering-lesson').textContent());
    for (const [name, example] of Object.entries(clusteringExamples)) {
      assert.ok(text.includes(normalize(example.code)), `Missing complete ${name} code`);
      assert.ok(text.includes(normalize(example.expected)), `Missing complete ${name} output`);
    }
    assert.deepEqual(await page.locator('[data-module-id="classical-ml"] [data-topic-id]').evaluateAll(nodes => nodes.map(node => node.dataset.topicId)), module.topicIds);
    assert.match(await page.locator('.reader-footer__previous').innerText(), /Survival Analysis/);
    assert.match(await page.locator('.reader-footer__next').innerText(), /PCA/);
    check('Full lesson, programs/output, current theme, formula geometry and module sequence');
    const [lloyd, geometry, seeding, hierarchy, palette] = [0, 1, 2, 3, 4].map(index => labs.nth(index));
    const prediction = lab => lab.getByLabel('Predict first', { exact: false });
    const reset = lab => lab.getByRole('button', { name: 'Reset', exact: true }).click();
    const commit = async (lab, answer) => { await prediction(lab).selectOption(answer); await lab.getByRole('button', { name: 'Check', exact: true }).click(); };
    const matches = async lab => assert.match(await lab.locator('.kh-feedback').innerText(), /Your prediction matches/);
    activeCheck = 'Lloyd prediction and trace';
    const next = lloyd.getByRole('button', { name: 'Next phase', exact: true });
    assert.ok(await next.isDisabled());
    assert.doesNotMatch(await lloyd.locator('.kh-step-controls').innerText(), /of \d/);
    await prediction(lloyd).selectOption('1'); await next.focus(); await page.keyboard.press('Enter');
    assert.match(await lloyd.locator('.kh-readout').innerText(), /SSE = 19\.25/);
    await next.click(); assert.match(await lloyd.locator('.kh-readout').innerText(), /SSE = 7\.6875/);
    await snapshot(page, lloyd.locator('.kh-scatter'), 'lloyd-moved-1366');
    await next.click(); await matches(lloyd); assert.ok(await prediction(lloyd).isDisabled());
    assert.ok(await next.isDisabled());
    for (const [second, error] of [['2', 1], ['1', 9]]) {
      await lloyd.getByLabel('Point configuration').selectOption('rectangle');
      await lloyd.getByLabel('Center 1 starts at row').selectOption(second);
      assert.equal(await prediction(lloyd).inputValue(), '');
      await prediction(lloyd).selectOption('1'); await lloyd.getByRole('button', { name: 'Run to fixed point' }).click();
      assert.match(await lloyd.locator('.kh-readout').innerText(), new RegExp(`SSE = ${error} using`));
    }
    await reset(lloyd); await lloyd.getByLabel('Center 1 starts at row').selectOption('0');
    await prediction(lloyd).selectOption('3+'); await next.click(); assert.match(await lloyd.locator('.kh-note').innerText(), /Empty center/);
    await reset(lloyd); check('Lloyd gate, keyboard, fixed point, changed seed optima, duplicate center and reset');
    activeCheck = 'Feature geometry reveal';
    assert.equal(await geometry.locator('.kh-center-label').count(), 0);
    assert.equal(await geometry.getByRole('region', { name: 'Every two-group split of the four records, in this metric', exact: true }).count(), 0);
    await geometry.getByLabel('Vertical measurement unit').selectOption('10'); await commit(geometry, 'bottom-top'); await matches(geometry);
    assert.match(await geometry.locator('.kh-readout').innerText(), /SSE 9/);
    assert.equal(await geometry.locator('table').first().locator('tbody tr').count(), 7);
    assert.ok(await prediction(geometry).isDisabled());
    await geometry.getByLabel('Weight on squared vertical differences').selectOption('0.01');
    assert.equal(await prediction(geometry).inputValue(), ''); assert.equal(await geometry.locator('.kh-feedback').count(), 0);
    assert.equal(await geometry.locator('.kh-center-label').count(), 0);
    await commit(geometry, 'left-right'); await matches(geometry); assert.match(await geometry.locator('.kh-readout').innerText(), /SSE 1/);
    check('Geometry input-only view, exhaustive reveal, changed-unit optimum and restored metric');
    activeCheck = 'Seeding exact probabilities and finite draws';
    assert.equal(await seeding.locator('.kh-frequency').count(), 0);
    assert.ok(await seeding.getByLabel('Draw position in the cumulative probability line').isDisabled());
    await commit(seeding, 'less'); await matches(seeding);
    await seeding.getByLabel('Draw position in the cumulative probability line').press('Home');
    assert.match(await seeding.locator('.kh-readout').innerText(), /selects P1/);
    await seeding.getByLabel('Draw position in the cumulative probability line').press('End');
    assert.match(await seeding.locator('.kh-readout').innerText(), /selects P5/);
    await seeding.getByLabel('Draw position in the cumulative probability line').fill('0.5');
    await seeding.getByRole('button', { name: 'Accept this draw and pick the next center' }).click();
    assert.equal(await prediction(seeding).inputValue(), ''); assert.equal(await seeding.locator('.kh-frequency').count(), 0);
    await commit(seeding, 'more'); await matches(seeding);
    assert.match(await seeding.locator('.kh-feedback').innerText(), /53\.2468%/);
    assert.match(await seeding.locator('.kh-frequency li').nth(3).innerText(), /90\/200/);
    assert.equal(await seeding.locator('.kh-frequency li').nth(0).locator('.kh-frequency-uniform').count(), 0);
    assert.equal(await seeding.locator('.kh-frequency li').nth(4).locator('.kh-frequency-uniform').count(), 0);
    await snapshot(page, seeding.locator('.kh-frequency'), 'seeding-probability-frequency-1366');
    await seeding.getByLabel('Seeding data').selectOption('duplicates'); await commit(seeding, 'none'); await matches(seeding);
    assert.match(await seeding.locator('.kh-readout').innerText(), /Every D² is zero/);
    assert.ok(await seeding.getByRole('button', { name: 'Accept this draw and pick the next center' }).isDisabled());
    await reset(seeding);
    // Repeated smallest-positive draws eventually represent all six distinct rows.
    for (let remaining = 5; remaining > 0; remaining -= 1) {
      await commit(seeding, 'less');
      await seeding.getByLabel('Draw position in the cumulative probability line').press('Home');
      await seeding.getByRole('button', { name: 'Accept this draw and pick the next center' }).click();
    }
    await commit(seeding, 'none'); await matches(seeding);
    assert.doesNotMatch(await seeding.locator('.kh-scatter figcaption').innerText(), /C0 already/);
    assert.equal(await seeding.locator('.kh-frequency').count(), 0);
    await snapshot(page, seeding.locator('.kh-scatter'), 'seeding-all-centers-1366');
    check('Seeding theory versus finite frequencies, keyboard interval endpoints, sequential draws, zero-mass cases and uniform baseline');
    activeCheck = 'Hierarchy prediction and exploration';
    assert.equal(await hierarchy.locator('.kh-dendrogram').count(), 0);
    assert.equal(await hierarchy.locator('.kh-readout').count(), 0);
    await commit(hierarchy, '3'); await matches(hierarchy);
    for (const method of ['single', 'complete', 'average', 'ward']) {
      await hierarchy.getByLabel('Linkage definition').selectOption(method);
      await hierarchy.getByLabel('Partition rule').selectOption('count');
      await hierarchy.getByLabel('Requested cluster count k').fill('4');
      assert.match(await hierarchy.locator('.kh-readout').innerText(), /4 clusters after 2 merges/);
      await hierarchy.getByRole('button', { name: /Set the tied height/ }).click();
      assert.match(await hierarchy.locator('.kh-readout').innerText(), /3 clusters after 3 merges/);
    }
    await hierarchy.getByLabel('Cut height').press('Home'); await hierarchy.getByRole('button', { name: 'Next merge', exact: true }).click();
    assert.match(await hierarchy.locator('.kh-readout').innerText(), /5 clusters after 1 merges/);
    await hierarchy.getByRole('button', { name: 'Undo merge', exact: true }).click();
    assert.match(await hierarchy.locator('.kh-readout').innerText(), /6 clusters after 0 merges/);
    await hierarchy.getByLabel('Point configuration').selectOption('chain');
    assert.equal(await hierarchy.locator('.kh-dendrogram').count(), 0);
    await commit(hierarchy, 'not-complete'); await matches(hierarchy);
    await hierarchy.getByLabel('Linkage definition').selectOption('complete');
    assert.match(await hierarchy.locator('.kh-readout').innerText(), /2 clusters after 7 merges/);
    const labels = await hierarchy.locator('.kh-scatter .kh-point-label').allTextContents();
    assert.deepEqual(labels, ['P0·0', 'P1·0', 'P2·0', 'P3·0', 'P4·1', 'P5·1', 'P6·1', 'P7·1', 'P8·1']);
    await snapshot(page, hierarchy, 'chain-complete-1366');
    check('Hierarchy gate, all four tied-height/count cuts, merge/undo and independent 4-plus-2 chain split');
    activeCheck = 'Palette caps and prediction commitment';
    for (const [image, unique, error] of [['sky', 37, 35945], ['gradient', 160, 165886], ['mosaic', 6, 0]]) {
      await palette.getByLabel('Image to quantize').selectOption(image); await palette.getByLabel('Requested palette size').press('End');
      await commit(palette, 'none'); await matches(palette);
      assert.match(await palette.locator('.kh-readout').innerText(), new RegExp(`integer-RGB SSE: ${error}`));
      assert.match(await palette.locator('.kh-feedback').innerText(), image === 'mosaic' ? /reconstruction is exact/ : new RegExp(`stops at 8 palette entries.*${unique} unique colors.*still has error ${error}`));
      assert.ok(await prediction(palette).isDisabled());
    }
    await palette.getByLabel('Image to quantize').selectOption('sky'); await palette.getByLabel('Requested palette size').fill('4');
    await commit(palette, 'no'); await snapshot(page, palette.locator('.kh-image-pair'), 'sky-palette-1366');
    check('Palette all three image endpoints, true lossless versus teaching cap, sealed feedback and changed request');
    activeCheck = 'Visuals, keyboard practice and responsive layouts';
    const hint = page.locator('.cluster-practice').first().getByText('Get a hint', { exact: true });
    await hint.focus(); await page.keyboard.press('Enter'); assert.ok(await hint.evaluate(node => node.parentElement.open));
    // Review the advanced mathematics and solutions too, not only initially
    // visible formulas. This changes disclosure state, never lesson source.
    await page.locator('.clustering-lesson details').evaluateAll(nodes => nodes.forEach(node => { node.open = true; }));
    for (const width of [1366, 820, 390, 320]) {
      await page.setViewportSize({ width, height: 950 }); await settle(page);
      await inspect(page);
      if (width === 1366 || width === 390) for (let figure = 0; figure < 7; figure += 1) await snapshot(page, page.locator('.cluster-figure').nth(figure), `figure-${figure + 1}-${width}`);
      if (width <= 390) {
        await snapshot(page, hierarchy, `chain-complete-${width}`);
        await snapshot(page, seeding.locator('.kh-scatter'), `seeding-all-centers-${width}`);
        await snapshot(page, geometry.locator('.kh-scatter'), `geometry-restored-${width}`);
        await snapshot(page, palette.locator('.kh-image-pair'), `sky-palette-${width}`);
      }
      check('Viewport, controls, links and math', { width });
    }
    await page.setViewportSize({ width: 390, height: 950 });
    const enlarged = await page.addStyleTag({ content: 'html { font-size: 125% !important; }' });
    await settle(page); await inspect(page); await snapshot(page, hierarchy, 'chain-enlarged-text-390');
    await enlarged.evaluate(node => node.remove());
    check('Increased text size and keyboard practice');
    assert.deepEqual(errors, []);
    if (production) {
      activeCheck = 'Production loading, completion and recovery';
      const source = `src/learn/data/topics/${id}.jsx`, body = build[source].file;
      const bodies = new Set(Object.values(publication).map(value => build[`src/learn/data/${value.replace(/^\.\//, '')}`].file));
      const outlines = new Set(Object.entries(build).filter(([key]) => key.includes('/generated/outlines/')).map(([,value]) => value.file));
      const allowed = new Set(), visit = key => { if (allowed.has(build[key].file)) return; allowed.add(build[key].file); (build[key].imports || []).forEach(visit); };
      [Object.keys(build).find(key => build[key].isEntry && key.endsWith('index.html')), 'src/learn/Reader.jsx', source].forEach(visit);
      const local = [...new Set(requests.filter(url => url.startsWith(`${base}/`)).map(url => new URL(url).pathname.slice(1)))];
      assert.deepEqual(local.filter(file => bodies.has(file)), [body]); assert.deepEqual(local.filter(file => outlines.has(file)), []);
      const scripts = local.filter(file => file.endsWith('.js')); assert.deepEqual(scripts.filter(file => !allowed.has(file)), []);
      check('Selected production lesson and shared dependencies only', { decodedJsBytes: scripts.reduce((n, f) => n + fs.statSync(`dist/${f}`).size, 0), gzipEstimateBytes: scripts.reduce((n, f) => n + gzipSync(fs.readFileSync(`dist/${f}`)).length, 0) });
      await page.locator('.reader-complete').click(); await page.reload(); await ready(page);
      assert.equal(await page.evaluate(id => JSON.parse(localStorage.getItem('kd-progress'))[id], id), true);
      await page.locator('.reader-footer__next').click(); await page.waitForURL('**/pca-dimensionality-reduction?module=classical-ml');
      check('Completion survives reload and Next opens the actual PCA successor');
      for (const failure of ['import', 'render']) {
        const isolated = await freshContext(); const trial = await isolated.newPage(); let attempts = 0;
        await trial.route(`**/${body}*`, request => {
          attempts += 1; if (attempts > 1) return request.continue();
          return failure === 'import' ? request.abort('failed') : request.fulfill({ status: 200, contentType: 'text/javascript', body: 'export default {content(){throw new Error("Controlled K-Means review render failure")}};' });
        });
        await trial.goto(route); const alert = trial.locator('.lesson-load-error'); await alert.waitFor();
        assert.ok(await trial.locator('.reader-complete').isDisabled()); assert.equal(await trial.locator('.planned-lesson').count(), 0);
        await Promise.all([trial.waitForEvent('domcontentloaded'), alert.getByRole('button', { name: 'Reload page', exact: true }).click()]);
        await ready(trial); assert.ok(attempts > 1); assert.ok(await trial.locator('.reader-complete').isEnabled());
        check('Published lesson error recovers after reload', { failure, attempts }); await isolated.close();
      }
      assert.equal(hash('dist/.vite/manifest.json'), manifestHash);
    }
    for (const [file, digest] of Object.entries(sourceHashes)) assert.equal(hash(file), digest, `Source changed during check: ${file}`);
    fs.writeFileSync(receipt, JSON.stringify({ status: 'passed', checkedAt: new Date().toISOString(), production, browser: browser.version(), sourceHashes, verifierSha256: hash(__filename), manifestHash, fontEvidence, records, screenshots, limitations: 'Single Chromium-family engine; screenshots require a separate actual image review. Source-bound native checks are recorded separately.' }, null, 2) + '\n');
    console.log(`PASS: ${records.length} K-Means review groups; ${screenshots.length} captures.`);
  } finally { await browser.close(); }
})().catch(error => {
  fs.writeFileSync(receipt, JSON.stringify({ status: 'failed', checkedAt: new Date().toISOString(), activeCheck, sourceHashes, records, screenshots, error: error.stack }, null, 2) + '\n');
  console.error(error); process.exitCode = 1;
});

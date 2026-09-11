// Focused closure after the complete production pass: label layout changed,
// while numerical models, examples, lesson text and reader behavior stayed fixed.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { gzipSync } = require('node:zlib');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const base = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4173';
const previousPath = 'docs/teaching/archive/k-means-hierarchical-before-label-layout/k-means-hierarchical-browser.json';
const previous = JSON.parse(fs.readFileSync(previousPath, 'utf8'));
const hash = filename => createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
const changedFiles = ['src/learn/components/lesson-labs/KMeansHierarchicalLabs.jsx', 'src/learn/components/lesson-labs/k-means-hierarchical-labs.css'];

(async () => {
  assert.equal(previous.status, 'passed');
  const sourceHashes = Object.fromEntries(Object.keys(previous.sourceHashes).map(filename => [filename, hash(filename)]));
  for (const [filename, expected] of Object.entries(previous.sourceHashes)) {
    if (!changedFiles.includes(filename)) assert.equal(sourceHashes[filename], expected, `An unrelated source changed: ${filename}`);
  }
  const build = JSON.parse(fs.readFileSync('dist/.vite/manifest.json', 'utf8'));
  const manifestHash = hash('dist/.vite/manifest.json');
  const source = 'src/learn/data/topics/k-means-hierarchical-clustering.jsx';
  const allowed = new Set();
  const addClosure = key => {
    if (allowed.has(build[key].file)) return;
    allowed.add(build[key].file);
    for (const dependency of build[key].imports || []) addClosure(dependency);
  };
  addClosure(Object.keys(build).find(key => build[key].isEntry && key.endsWith('index.html')));
  addClosure('src/learn/Reader.jsx');
  addClosure(source);
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const cases = [], screenshots = [];
  try {
    const context = await browser.newContext({ viewport: { width: 1366, height: 1000 } });
    const page = await context.newPage();
    const errors = [], failures = [], requests = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('requestfailed', request => failures.push({ url: request.url(), reason: request.failure() }));
    page.on('request', request => requests.push(request.url()));
    await page.goto(`${base}/learn/path/full-curriculum/k-means-hierarchical-clustering?module=classical-ml`);
    await page.locator('.clustering-lesson').waitFor();
    await page.waitForFunction(() => [...document.fonts].some(font => font.family.includes('Space Grotesk') && font.status === 'loaded'), null, { timeout: 20000 });
    await page.evaluate(() => document.fonts.ready);
    const labs = page.locator('.kh-investigation');
    const lloyd = labs.nth(0);
    const screenshot = async (locator, filename) => {
      const viewport = page.viewportSize();
      const bounds = await locator.boundingBox();
      if (bounds.height > viewport.height - 160) await page.setViewportSize({ width: viewport.width, height: Math.ceil(bounds.height) + 180 });
      await locator.scrollIntoViewIfNeeded();
      await locator.evaluate(element => window.scrollTo(0, element.getBoundingClientRect().top + scrollY - 100));
      const destination = `docs/teaching/evidence/${filename}`;
      await locator.screenshot({ path: destination });
      screenshots.push(destination);
      await page.setViewportSize(viewport);
    };
    const checkLabels = async (lab, state) => {
      const overlaps = await lab.locator('.kh-point-label, .kh-center-label').evaluateAll(items => {
        const boxes = items.map(element => ({ text: element.textContent, box: element.getBoundingClientRect() }));
        const collisions = [];
        for (let i = 0; i < boxes.length; i += 1) for (let j = i + 1; j < boxes.length; j += 1) {
          const a = boxes[i].box, b = boxes[j].box;
          if (Math.min(a.right, b.right) - Math.max(a.left, b.left) > 1 && Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top) > 1) collisions.push([boxes[i].text, boxes[j].text]);
        }
        return collisions;
      });
      assert.deepEqual(overlaps, [], `Annotation overlap: ${state}`);
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), `Document overflow: ${state}`);
      cases.push({ state, viewport: page.viewportSize(), labelOverlap: false });
    };
    for (const width of [1366, 390, 320]) {
      await page.setViewportSize({ width, height: 1000 });
      await lloyd.getByRole('button', { name: 'Reset', exact: true }).click();
      await checkLabels(lloyd, 'six points: initial');
      await lloyd.getByRole('button', { name: 'Next phase' }).click();
      await checkLabels(lloyd, 'six points: assigned');
      assert.match(await lloyd.locator('.kh-readout').innerText(), /SSE = 19\.25/);
      await lloyd.getByRole('button', { name: 'Next phase' }).click();
      await checkLabels(lloyd, 'six points: moved means');
      assert.match(await lloyd.locator('.kh-readout').innerText(), /SSE = 7\.6875/);
      if (width !== 320) await screenshot(lloyd.locator('.kh-scatter'), width === 1366 ? 'k-means-lloyd-desktop.png' : 'k-means-lloyd-mobile.png');
      await lloyd.getByLabel('Initial center rows').selectOption('duplicate');
      await checkLabels(lloyd, 'duplicate centers: initial');
      await lloyd.getByRole('button', { name: 'Next phase' }).click();
      await checkLabels(lloyd, 'duplicate centers: assigned');
      await lloyd.getByRole('button', { name: 'Next phase' }).click();
      await checkLabels(lloyd, 'duplicate centers: moved');
      await lloyd.getByLabel('Initial center rows').selectOption('nearby');
      await checkLabels(lloyd, 'nearby centers: initial');
      await lloyd.getByRole('button', { name: 'Next phase' }).click();
      await checkLabels(lloyd, 'nearby centers: assigned');
      await lloyd.getByRole('button', { name: 'Next phase' }).click();
      await checkLabels(lloyd, 'nearby centers: moved');
      await lloyd.getByLabel('Point configuration').selectOption('rectangle');
      for (const initialization of ['separated', 'nearby']) {
        await lloyd.getByLabel('Initial center rows').selectOption(initialization);
        await checkLabels(lloyd, `rectangle ${initialization}: initial`);
        await lloyd.getByRole('button', { name: 'Next phase' }).click();
        await checkLabels(lloyd, `rectangle ${initialization}: assigned`);
        await lloyd.getByRole('button', { name: 'Next phase' }).click();
        await checkLabels(lloyd, `rectangle ${initialization}: moved`);
      }
      const geometry = labs.nth(1);
      await geometry.getByLabel('Vertical measurement unit').selectOption('10');
      await geometry.getByLabel('Weight on squared vertical differences').selectOption('4');
      await checkLabels(geometry, 'tall transformed rectangle');
      const seeding = labs.nth(2);
      await seeding.getByRole('button', { name: 'Reset', exact: true }).click();
      await checkLabels(seeding, 'second seed selected');
      if (width === 1366) await screenshot(seeding, 'k-means-seeding-desktop.png');
      await seeding.getByLabel('Seeding data').selectOption('duplicates');
      await checkLabels(seeding, 'identical rows: stopped seeding');
    }
    assert.deepEqual(errors, []);
    assert.deepEqual(failures, []);
    const scripts = [...new Set(requests.map(address => new URL(address)).filter(address => address.origin === new URL(base).origin && address.pathname.endsWith('.js')).map(address => address.pathname.slice(1)))];
    assert.deepEqual(scripts.filter(filename => !allowed.has(filename)), []);
    assert.equal(scripts.filter(filename => /k-means-hierarchical-clustering.*\.js$/.test(filename)).length, 1);
    for (const [filename, expected] of Object.entries(sourceHashes)) assert.equal(hash(filename), expected);
    assert.equal(hash('dist/.vite/manifest.json'), manifestHash);
    const result = {
      checkedAt: new Date().toISOString(), topicId: previous.topicId, status: 'passed',
      sourceHashes, buildManifestHash: manifestHash,
      previousVerification: { path: previousPath, sha256: hash(previousPath), cases: previous.records.length, scope: 'All 12 complete production cases passed; unchanged numerical/code/sequence/progress/recovery checks retained. This amendment targets only revised scatter annotation placement and the resulting production chunk.' },
      labelLayout: { cases, screenshots, interpretation: 'Actual intended fonts; DOM checks detect label-to-label overlap. Selected images must additionally be opened to assess point/center-marker clearance and legibility.' },
      network: { requestedScripts: scripts, rawScriptBytes: scripts.reduce((sum, filename) => sum + fs.statSync(path.join('dist', filename)).size, 0), gzipEstimateBytes: scripts.reduce((sum, filename) => sum + gzipSync(fs.readFileSync(path.join('dist', filename))).length, 0) },
      retainedScreenshots: previous.screenshots.filter(filename => !screenshots.includes(filename)),
      catalogueIds: previous.catalogueIds, publicationMappings: previous.publicationMappings, moduleTopicCount: previous.moduleTopicCount
    };
    fs.writeFileSync('docs/teaching/evidence/k-means-hierarchical-browser.json', JSON.stringify(result, null, 2) + '\n');
    console.log(JSON.stringify({ status: result.status, focusedStates: cases.length, reusedCases: previous.records.length }));
    await context.close();
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });

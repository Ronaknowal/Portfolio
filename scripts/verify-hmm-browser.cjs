// Production browser review of the hidden Markov models lesson: visible content,
// all six investigations with their nulls and construction tasks, figure geometry
// read back from the drawn coordinates, label collisions including curve-shaped
// ones, SVG type size measured in RENDERED pixels against each viewBox, the two
// objective windows checked as a perceptible separation rather than as a model
// claim, narrow layouts, stacked tables, formula width, sequence, completion and
// load-failure recovery.
//
// Run against a preview of a production build:
//   npx vite build --outDir dist-hmm
//   npx vite preview --outDir dist-hmm --host 127.0.0.1 --port 4190
//   PLAYWRIGHT_PACKAGE=... DIST_DIR=dist-hmm LEARNING_BASE_URL=http://127.0.0.1:4190 \
//     node scripts/verify-hmm-browser.cjs
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { gzipSync } = require('node:zlib');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const { inspectLessonVisualLayout } = require('./lib/lesson-visual-layout.cjs');

const distDir = process.env.DIST_DIR || 'dist-hmm';
const base = (process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4190').replace(/\/+$/, '');
const topicId = 'hidden-markov-models-hmm';
const sourcePath = `src/learn/data/topics/${topicId}.jsx`;
const route = `${base}/learn/path/full-curriculum/${topicId}?module=classical-ml`;
const evidencePath = 'docs/teaching/evidence/hmm-browser.json';
const read = filename => JSON.parse(fs.readFileSync(filename, 'utf8'));
const hash = filename => createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
const normalize = text => text.replace(/\s+/g, ' ').trim();
const ownedFiles = [
  sourcePath,
  'src/learn/data/hmm-models.js',
  'src/learn/data/hmm-data.js',
  'src/learn/data/hmm-examples.js',
  'src/learn/components/lesson-labs/HmmShared.jsx',
  'src/learn/components/lesson-labs/HmmLabs.jsx',
  'src/learn/components/lesson-labs/HmmFigures.jsx',
  'src/learn/components/lesson-labs/hmm-labs.css',
  `src/learn/data/curriculum/blueprints/${topicId}.js`,
  'public/learn-assets/hmm/ewt-sequences.json',
  'public/learn-assets/hmm/hmm-experiments.py',
  'public/learn-assets/hmm/hmmlearn-examples.py',
  'public/learn-assets/hmm/ATTRIBUTION.txt',
];

/** Runs in the page. Samples every curve along its own geometry and reports the
 * labels it passes through.
 *
 * Copied from scripts/verify-bias-variance-browser.cjs with this lesson's class
 * names. `scripts/lib/lesson-visual-layout.cjs` iterates `svg.querySelectorAll('line')`,
 * so <path>, <polyline> and <polygon> are invisible to it. This lesson draws the
 * state graph's transitions and self-loops as <path> arcs and every objective
 * track as a <polyline>, so that whole class matters here.
 *
 * A label carrying an opaque backplate may be crossed briefly, because the
 * backplate is what a crossing is mitigated with. It may not be travelled along.
 * Grid lines are background and are skipped.
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
        halo: text.classList.contains('hmm-halo'),
        box: text.getBoundingClientRect(),
      }));
    if (!labels.length) continue;
    for (const shape of svg.querySelectorAll('path, polyline, polygon')) {
      if (shape.classList.contains('hmm-grid')) continue;
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

/** Runs in the page. Reports the rendered size of every SVG label in CSS pixels.
 *
 * A viewBox wider than the element scales its type down. The declared 12px
 * inside a 420-unit box is only legible if the element is roughly that wide, so
 * the measured size is what matters, not the stylesheet's number. The reading is
 * the computed font size times the element-to-viewBox scale, which is what the
 * reader's eye actually receives.
 */
function measureSvgTypeSizes(root) {
  const readings = [];
  for (const svg of document.querySelectorAll(`${root} svg`)) {
    const box = svg.getBoundingClientRect();
    if (!box.width) continue;
    const viewBox = svg.viewBox.baseVal;
    const scale = viewBox && viewBox.width ? box.width / viewBox.width : 1;
    for (const text of svg.querySelectorAll('text')) {
      if (!text.textContent.trim() || !text.getClientRects().length) continue;
      const declared = parseFloat(getComputedStyle(text).fontSize);
      readings.push({
        text: text.textContent.trim().slice(0, 30),
        secondary: text.classList.contains('hmm-small'),
        declaredPx: Number(declared.toFixed(2)),
        renderedPx: Number((declared * scale).toFixed(2)),
        elementWidth: Number(box.width.toFixed(1)),
        viewBoxWidth: viewBox ? Number(viewBox.width.toFixed(1)) : null,
        scale: Number(scale.toFixed(3)),
        figure: (svg.closest('.hmm-figure, .hmm-investigation') || {}).className || 'unscoped',
      });
    }
  }
  return readings;
}

/** Runs in the page. Reads back what the trellis actually drew: node centres,
 * edge endpoints, stroke widths and the observation cards.
 *
 * The point is that a drawn edge is a mathematical claim. Reading the stroke
 * width back out of the rendered geometry is the only way to know that the
 * width the model computed is the width the reader sees, rather than one a
 * stylesheet overrode.
 */
function readTrellisGeometry(selector) {
  const svg = document.querySelector(selector);
  if (!svg) return null;
  const toScreen = element => {
    const box = element.getBoundingClientRect();
    return { x: box.left + box.width / 2, y: box.top + box.height / 2, width: box.width, height: box.height };
  };
  return {
    nodes: [...svg.querySelectorAll('rect.hmm-node')].map(toScreen),
    cards: [...svg.querySelectorAll('rect.hmm-card')].map(toScreen),
    edges: [...svg.querySelectorAll('line.hmm-edge')].map(line => ({
      classes: line.getAttribute('class'),
      strokeWidth: Number(parseFloat(getComputedStyle(line).strokeWidth).toFixed(3)),
      opacity: Number(getComputedStyle(line).opacity),
      x1: Number(line.x1.baseVal.value.toFixed(2)), y1: Number(line.y1.baseVal.value.toFixed(2)),
      x2: Number(line.x2.baseVal.value.toFixed(2)), y2: Number(line.y2.baseVal.value.toFixed(2)),
    })),
    emissionConnectors: svg.querySelectorAll('line.hmm-emission').length,
    valueLabels: [...svg.querySelectorAll('text.hmm-muted')].map(text => text.textContent.trim()),
  };
}

/** Runs in the page. Reads back the rendered fill of every bar, so a bar that is
 * present in the DOM but invisible on screen is caught.
 *
 * One lesson shipped a figure whose bands were entirely invisible — an outline
 * rect's fill="none" lost to a stylesheet rule — while every assertion passed.
 * This reads computed fill, opacity and painted area rather than attributes.
 */
function readPaintedRects({ root, selector }) {
  const results = [];
  for (const svg of document.querySelectorAll(`${root} svg`)) {
    for (const rect of svg.querySelectorAll(selector)) {
      const style = getComputedStyle(rect);
      const box = rect.getBoundingClientRect();
      results.push({
        classes: rect.getAttribute('class'),
        fill: style.fill,
        stroke: style.stroke,
        fillOpacity: Number(style.fillOpacity),
        opacity: Number(style.opacity),
        widthPx: Number(box.width.toFixed(2)),
        heightPx: Number(box.height.toFixed(2)),
      });
    }
  }
  return results;
}

/** Runs in the page. The rendered vertical separation of the final point of each
 * objective track, panel by panel.
 *
 * The lesson claims one window cannot resolve what the other can, and quotes
 * 0.20 px against 5.57 px. That is a claim about what a reader can see, so it is
 * measured here in rendered CSS pixels rather than trusted from the model.
 */
function measureObjectiveSeparation() {
  const panels = [...document.querySelectorAll('.hmm-figure .hmm-panel')]
    .filter(panel => panel.querySelector('polyline.hmm-track'));
  return panels.map(panel => {
    const heading = (panel.querySelector('h4') || {}).textContent || '';
    const finals = [...panel.querySelectorAll('polyline.hmm-track')].map(track => {
      const length = track.getTotalLength();
      const point = track.getPointAtLength(length);
      const matrix = track.getScreenCTM();
      return {
        cls: track.getAttribute('class'),
        y: matrix.b * point.x + matrix.d * point.y + matrix.f,
      };
    }).sort((left, right) => left.y - right.y);
    let smallest = Infinity;
    for (let index = 1; index < finals.length; index += 1) {
      smallest = Math.min(smallest, Math.abs(finals[index].y - finals[index - 1].y));
    }
    return { heading: heading.trim(), tracks: finals.length, smallestSeparationPx: Number(smallest.toFixed(2)) };
  });
}

(async () => {
  const startedAt = new Date().toISOString();
  const sourceHashes = Object.fromEntries(ownedFiles.map(filename => [filename, hash(filename)]));
  const build = read(`${distDir}/.vite/manifest.json`);
  const buildHash = hash(`${distDir}/.vite/manifest.json`);
  const publications = read('src/learn/data/lesson-manifest.json');
  const { tracks } = await import('../src/learn/data/tracks.js');
  const { hmmExamples } = await import('../src/learn/data/hmm-examples.js');
  const data = await import('../src/learn/data/hmm-data.js');
  const models = await import('../src/learn/data/hmm-models.js');
  const module_ = tracks.find(track => track.id === 'classical-ml');
  assert.equal(module_.topicIds[26], topicId, 'the lesson sits at position 27 of its module');
  const bodyFile = build[sourcePath].file;
  const bodyFiles = new Set(Object.values(publications)
    .map(value => build[`src/learn/data/${value.replace(/^\.\//, '')}`].file));
  const allowedScripts = new Set();
  function addClosure(key) {
    if (allowedScripts.has(build[key].file)) return;
    allowedScripts.add(build[key].file);
    for (const child of build[key].imports || []) addClosure(child);
  }
  addClosure(Object.keys(build).find(key => build[key].isEntry && key.endsWith('index.html')));
  addClosure('src/learn/Reader.jsx');
  addClosure(sourcePath);

  /* Fifteen tables are present with every investigation at its fixture; more
     appear once predictions are committed, and both states are swept. */
  const tableFloor = 15;
  const weather = models.fixtures.weather;
  const reports = models.fixtures.reports;
  const main = models.infer(weather, reports);
  const forward = models.trellis(weather, reports, 'sum');
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [];
  const screenshotPaths = [];
  const typeReadings = [];
  const checkText = async (locator, pattern) => assert.match(normalize(await locator.innerText()), pattern);
  const settle = page => page.evaluate(() =>
    new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  const ready = async page => {
    await page.locator('.hmm-lesson').waitFor();
    await page.waitForFunction(
      () => [...document.fonts].some(font => font.family.includes('Space Grotesk') && font.status === 'loaded'),
      null, { timeout: 20000 });
    await page.evaluate(() => document.fonts.ready);
    await settle(page);
  };
  try {
    const context = await browser.newContext({ viewport: { width: 1366, height: 1000 } });
    const page = await context.newPage();
    /* Investigations are found by their own heading, never by position. The
       tagging lab belongs to section 8 and the dwell-time lab to section 9, so an
       index would encode that ordering in the verifier as well as in the page. */
    const labNamed = title => page.locator('.hmm-investigation').filter({ hasText: title });
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
      if (bounds.height > viewport.height - 160) {
        await page.setViewportSize({ width: viewport.width, height: Math.ceil(bounds.height) + 180 });
      }
      await locator.scrollIntoViewIfNeeded();
      await locator.evaluate(element =>
        window.scrollTo(0, element.getBoundingClientRect().top + window.scrollY - 100));
      /* The site navigation is position: fixed, so on a narrow screen it paints
         across whatever is being captured. Hiding it for the shot changes no
         layout, because a fixed element occupies no flow space, and it is what
         makes the captured image show the figure rather than the chrome. */
      await page.evaluate(() => {
        const nav = document.querySelector('.learn-nav');
        if (nav) nav.style.visibility = 'hidden';
      });
      await settle(page);
      await locator.screenshot({ path: destination });
      await page.evaluate(() => {
        const nav = document.querySelector('.learn-nav');
        if (nav) nav.style.visibility = '';
      });
      await page.setViewportSize(viewport);
      screenshotPaths.push(destination);
    };

    // ----------------------------------------------------------- 1. structure
    await checkText(page.locator('.reader-header h1'), /Hidden Markov Models/);
    await checkText(page.locator('.reader-header__meta'), /27 of 39 topics on this route/);
    await checkText(page.locator('.reader-footer__previous'), /AutoML/);
    await checkText(page.locator('.reader-footer__next'), /Bayesian Networks/);
    assert.equal(await page.locator('.hmm-investigation').count(), 6, 'six investigations');
    assert.equal(await page.locator('.hmm-figure').count(), 9, 'nine inline figures');
    assert.equal(await page.locator('.hmm-practice').count(), 10, 'ten practice tasks');
    assert.equal(await page.locator('.python-example').count(), 2, 'two runnable programs');
    assert.equal(await page.locator('.hmm-excerpt').count(), 4, 'four displayed excerpts of the complete program');
    const rendered = normalize(await page.locator('.hmm-lesson').textContent());
    for (const excerpt of hmmExamples.experiments.excerpts) {
      assert.ok(rendered.includes(normalize(excerpt.code)),
        `Missing the actual ${excerpt.key} excerpt on the page`);
    }
    assert.ok(rendered.includes(normalize(hmmExamples.hmmlearn.code)), 'the optional program is shown in full');
    assert.ok(rendered.includes(normalize(hmmExamples.hmmlearn.expected)),
      'together with the output it actually produced');
    assert.ok(rendered.includes(normalize(hmmExamples.experiments.expected)),
      'and the complete program prints what it printed');
    assert.ok(rendered.includes(hmmExamples.hmmlearn.environment.hmmlearn)
      && rendered.includes(hmmExamples.hmmlearn.environment.numpy),
    'the isolated environment is named beside the output it produced');
    for (const anchor of await page.locator('.lesson-intro a[href^="#"]')
      .evaluateAll(items => items.map(item => item.getAttribute('href')))) {
      assert.equal(await page.locator(`[id="${anchor.slice(1)}"]`).count(), 1, `Broken lesson anchor ${anchor}`);
    }
    assert.equal(await page.locator('.katex-error').count(), 0, 'no formula failed to render');
    // Every recorded outcome the lesson states is visible.
    for (const configuration of data.configurations) {
      assert.ok(rendered.includes(String(configuration.correct)),
        `${configuration.key}: its token count is shown`);
    }
    assert.ok(rendered.includes(String(data.majority.correct)), 'the majority baseline is shown');
    assert.ok(rendered.includes('269') && rendered.includes('272'),
      'the tie band is stated on the page, not only in the evidence');
    /* The split the introduction names must be the split the computation used.
       It said "three of the forty test sentences" while the page says three
       separate times that the reserved forty are never scored, so the opening
       paragraph asserted a result on the held-out data the rest of the lesson
       promises never to touch. */
    const intro = normalize(await page.locator('.lesson-intro').textContent());
    const tiedSentences = data.tieAudit.configurations[1].tiedSentences;
    assert.ok(intro.includes(`${tiedSentences.length} of the ${data.provenance.sentences.development} development sentences`),
      `the introduction names the development split and its size: ${intro.slice(0, 400)}`);
    for (const wrong of ['test sentences have', 'reserved sentences have', 'held-out sentences have']) {
      assert.ok(!intro.includes(wrong),
        `the introduction never attributes the tie finding to another split, but it contains "${wrong}"`);
    }
    for (const tie of tiedSentences) {
      assert.ok(data.developmentSentences.some(sentence => sentence.id === tie.id),
        `tied sentence ${tie.sentence} is one of the development sentences the page serves`);
    }
    assert.equal(data.provenance.reservedScored, false, 'and the reserved split is recorded as unscored');
    for (const phrase of ['not decoded anywhere in this lesson', 'stay unscored', 'never predicted or scored']) {
      assert.ok(rendered.includes(phrase), `the page still states "${phrase}"`);
    }
    // The permitted-edge count is stated in a figure heading, in the prose and in
    // an investigation note; all three must agree with the model.
    const permitted = models.fixtures.constrained.transition.flat().filter(value => value > 0).length;
    assert.equal(permitted, 4, 'the constrained graph permits four transitions');
    assert.ok(rendered.includes('Only four transitions exist'),
      'and the figure heading says four');
    assert.ok(rendered.includes('A→B, A→C, B→A and C→A'), 'the prose lists all four');
    assert.ok(!/Only three transitions/.test(rendered), 'and nothing on the page still says three');
    assert.ok(rendered.includes('CC BY-SA 4.0'), 'the licence travels with the data');
    assert.ok(rendered.includes(data.provenance.sha256), 'and so does the file hash');
    assert.ok(/reserved sentences (are )?(never|not)/i.test(rendered)
      || rendered.includes('remain unscored'), 'the protected reserve is stated');
    const asset = await page.request.get(`${base}${data.provenance.file}`);
    assert.equal(asset.status(), 200);
    const bytes = await asset.body();
    assert.equal(bytes.length, data.provenance.bytes, 'the served extract is byte-for-byte the packet file');
    assert.equal(createHash('sha256').update(bytes).digest('hex'), data.provenance.sha256,
      'and matches its recorded hash');
    const attribution = await page.request.get(`${base}${data.provenance.attribution}`);
    assert.equal(attribution.status(), 200);
    const attributionText = await attribution.text();
    assert.ok(attributionText.includes('CC BY-SA 4.0') && attributionText.includes(data.provenance.sha256));
    assert.ok(attributionText.includes('r2.16'), 'the attribution records the release');
    for (const program of ['hmm-experiments.py', 'hmmlearn-examples.py']) {
      const served = await page.request.get(`${base}/learn-assets/hmm/${program}`);
      assert.equal(served.status(), 200, `${program} is downloadable`);
      assert.equal(createHash('sha256').update(await served.body()).digest('hex'),
        hash(`public/learn-assets/hmm/${program}`), `${program} is served unchanged`);
    }
    records.push({
      case: 'Complete visible code for four verbatim excerpts and the optional program with its real output and '
        + 'named environment, every route anchor, nine figures, six investigations, ten practice tasks, the four '
        + 'recorded token counts and the tie band, the served unchanged extract and both programs with their '
        + 'hashes and attribution, current metadata and module sequence',
    });

    // --------------------------------------------- 2. one path's product (I1)
    const pathLab = labNamed('Follow one complete story through the model');
    assert.equal(await pathLab.locator('input[type="radio"]:checked').count(), 0, 'no prediction is preselected');
    assert.ok(await pathLab.getByRole('button', { name: 'Apply and check' }).isDisabled(),
      'checking waits for a choice');
    assert.equal(await pathLab.locator('.hmm-verdict').count(), 0, 'no answer before a prediction');
    assert.equal(await pathLab.locator('.hmm-table').count(), 0, 'and no factor or path table on first paint');
    await screenshot(pathLab, 'hmm-path-initial-desktop.png');
    await pathLab.getByRole('button', { name: 'Sunny, Sunny, Sunny, Rainy' }).click();
    await pathLab.getByLabel('the most probable of them', { exact: true }).check();
    await pathLab.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(pathLab.locator('.hmm-verdict'), /your prediction matches/i);
    assert.equal(await pathLab.locator('.hmm-table').count(), 2, 'the two tables appear only after the prediction');
    await checkText(pathLab, /0\.0031104/);
    await checkText(pathLab, /one start, four emissions and three transitions/);
    await screenshot(pathLab, 'hmm-path-best-desktop.png');
    // An impossible story is called impossible, not merely unlikely.
    await pathLab.getByRole('button', { name: 'Reset', exact: true }).click();
    await pathLab.getByRole('button', { name: 'All Rainy' }).click();
    await pathLab.getByLabel('less probable than the best one', { exact: true }).check();
    await pathLab.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(pathLab.locator('.hmm-verdict'), /your prediction matches/i);
    await screenshot(pathLab, 'hmm-path-other-desktop.png');
    records.push({ case: 'Path investigation: nothing computed before the prediction, the eight named factors and '
      + 'all sixteen stories after it, and the best path reported at its exact joint mass' });

    // ------------------------------------- 3. evidence arriving later (I2)
    const laterLab = labNamed('Change the future; watch which belief moves');
    assert.equal(await laterLab.locator('.hmm-verdict').count(), 0, 'no verdict on first paint');
    // The drafted trellis must carry its structure and NOT its cell values,
    // because the graded quantity is derivable from them.
    const hiddenValues = await laterLab.evaluate(node =>
      [...node.querySelectorAll('svg text.hmm-muted')].map(text => text.textContent.trim()));
    assert.ok(hiddenValues.length >= 4,
      `${hiddenValues.length} muted labels were inspected, so a class rename cannot make this pass on nothing`);
    assert.ok(hiddenValues.every(text => /^t = \d+$/.test(text) || text === ''),
      `the drafted trellis shows no cell values before a prediction, but shows ${JSON.stringify(hiddenValues.slice(0, 6))}`);
    assert.ok(await laterLab.locator('svg rect.hmm-node').count() >= 8,
      'while still drawing its nodes, so the learner can see what they are editing');
    await screenshot(laterLab, 'hmm-later-initial-desktop.png');
    // The declared contrast: correcting the last report holds filtering exactly
    // and moves smoothing.
    await laterLab.getByRole('button', { name: /Correct the final report to Walk/ }).click();
    await laterLab.locator('.hmm-group').nth(0).getByLabel('does not move', { exact: true }).check();
    await laterLab.locator('.hmm-group').nth(1).getByLabel('falls', { exact: true }).check();
    await laterLab.getByRole('button', { name: 'Apply and check' }).click();
    const laterVerdict = normalize(await laterLab.locator('.hmm-verdict').innerText());
    assert.ok(/filtering at t = 1: your prediction matches/i.test(laterVerdict), laterVerdict);
    assert.ok(/smoothing at t = 1: your prediction matches/i.test(laterVerdict), laterVerdict);
    // The two filtered values printed beside "does not move" must be identical
    // strings: a verdict that says nothing moved while showing two different
    // numbers is the defect this contract exists to prevent.
    const printedPair = laterVerdict.match(/Filtering: (-?[\d.]+) → (-?[\d.]+)\./);
    assert.ok(printedPair, `the verdict prints both filtered values: ${laterVerdict}`);
    assert.equal(printedPair[1], printedPair[2],
      'and they are the same string, to the twelve places the verdict prints');
    assert.equal(printedPair[1].split('.')[1].length, 12, 'which really is twelve decimal places');
    const smoothedPair = laterVerdict.match(/Smoothing: (-?[\d.]+) → (-?[\d.]+)\./);
    assert.notEqual(smoothedPair[1], smoothedPair[2], 'while the smoothed pair genuinely differs');
    await screenshot(laterLab, 'hmm-later-corrected-desktop.png');
    // The construction task starts unsolved and is graded from the computation.
    const construction = laterLab.locator('.hmm-construction');
    assert.equal(await construction.locator('.hmm-verdict').count(), 0, 'the construction starts unsolved');
    await checkText(construction, /starts unsolved/);
    await construction.getByRole('button', { name: 'Submit this attempt' }).click();
    await checkText(construction.locator('.hmm-verdict'), /Solved\./);
    await screenshot(laterLab, 'hmm-later-construction-solved-desktop.png');
    // And the original report fails it, on a computed condition rather than a name.
    await laterLab.getByRole('button', { name: /The recording as given/ }).click();
    await construction.getByRole('button', { name: 'Submit this attempt' }).click();
    await checkText(construction.locator('.hmm-verdict'), /Not yet\./);
    await checkText(construction.locator('.hmm-verdict'), /which is not below 0\.41/);
    await screenshot(laterLab, 'hmm-later-construction-unsolved-desktop.png');
    /* S4 IN THE PAINT, in the state the reviewer reached through the interface.
       Everything above runs on a page where every edge carries some mass, so a
       zero-width regression is invisible to it: the whole point of the defect
       was that a zero-mass edge is only reachable by editing the model. The
       free exploration panel accepts any Rainy emission row that is a
       distribution, and a zero in the Walk column empties every edge out of
       Rainy while leaving the Sunny ones carrying. */
    await laterLab.locator('details', { hasText: 'Free exploration' }).locator('summary').click();
    await laterLab.getByRole('button', { name: 'Unlock the model' }).click();
    await laterLab.locator('.hmm-row-editor input').first().fill('0, 0.5, 0.5');
    await laterLab.getByRole('button', { name: 'Validate and apply this row' }).click();
    const zeroMassPaint = await laterLab.evaluate(node =>
      [...node.querySelectorAll('svg line.hmm-edge')].map(line => ({
        classes: line.getAttribute('class') || '',
        attribute: line.getAttribute('stroke-width'),
        painted: Number(parseFloat(getComputedStyle(line).strokeWidth).toFixed(4)),
      })));
    assert.ok(zeroMassPaint.length >= 8,
      `${zeroMassPaint.length} edges are drawn once the Rainy row has a zero in it`);
    const emptyEdges = zeroMassPaint.filter(edge => edge.classes.includes('is-empty'));
    const carryingEdges = zeroMassPaint.filter(edge => !edge.classes.includes('is-empty')
      && !edge.classes.includes('is-forbidden'));
    assert.ok(emptyEdges.length >= 2,
      `the edited row really does produce edges that carry nothing (${emptyEdges.length} of them), `
      + 'so what follows is not a check on an empty set');
    assert.ok(carryingEdges.length >= 2,
      `while ${carryingEdges.length} edges still carry mass, so there is something to compare against`);
    for (const edge of zeroMassPaint) {
      assert.ok(edge.attribute !== null,
        `every edge in the reachable zero-mass state sets a stroke-width attribute (${edge.classes} does not)`);
    }
    const widestEmpty = Math.max(...emptyEdges.map(edge => edge.painted));
    const thinnestCarrying = Math.min(...carryingEdges.map(edge => edge.painted));
    assert.ok(widestEmpty < thinnestCarrying,
      `an edge carrying nothing paints at ${widestEmpty}px, and the thinnest edge carrying anything at `
      + `${thinnestCarrying}px, so nothing is drawn as though it carried more than it does`);
    await screenshot(laterLab, 'hmm-later-zero-mass-edges-desktop.png');
    await laterLab.getByRole('button', { name: 'Restore the original Rainy row' }).click();
    await laterLab.getByRole('button', { name: 'Lock the model again' }).click();
    // The impossible branch: a sequence with zero evidence has no posterior.
    await laterLab.getByRole('button', { name: 'Reset', exact: true }).click();
    records.push({ case: 'Evidence investigation: cell values withheld until commitment, separate filtering and '
      + 'smoothing verdicts, an unchanged verdict showing two identical twelve-place values, a construction '
      + 'task graded from inference that the original report fails and the corrected one solves, and an edited '
      + 'emission row driving the trellis into its zero-mass state, where every edge still sets a width and '
      + 'none carrying nothing is painted wider than one carrying something' });

    // ------------------------------------------- 4. repair a legal path (I3)
    const pathRepair = labNamed('Connect the most probable cells, then check whether you have a path');
    assert.equal(await pathRepair.locator('.hmm-verdict').count(), 0, 'no verdict on first paint');
    const hiddenScores = await pathRepair.evaluate(node =>
      [...node.querySelectorAll('svg text.hmm-muted')].map(text => text.textContent.trim()));
    assert.ok(hiddenScores.length >= 2,
      `${hiddenScores.length} muted labels were inspected, so a class rename cannot make this pass on nothing`);
    assert.ok(hiddenScores.every(text => /^t = \d+$/.test(text) || text === ''),
      'no path score is shown before the prediction is recorded');
    // A forbidden edge is drawn in its own vocabulary, not as a thin permitted one.
    const forbidden = await pathRepair.locator('svg line.hmm-edge.is-forbidden').count();
    assert.ok(forbidden >= 4, `forbidden edges are drawn distinctly (${forbidden} found)`);
    await screenshot(pathRepair, 'hmm-legal-initial-desktop.png');
    await pathRepair.getByLabel('no — it does not, or is not even a path', { exact: true }).check();
    await pathRepair.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(pathRepair.locator('.hmm-verdict'), /your prediction matches/i);
    await checkText(pathRepair, /A → A with joint probability exactly 0/);
    await screenshot(pathRepair, 'hmm-legal-modes-desktop.png');
    // Under the changed prior both decoders agree, which is the declared contrast.
    await pathRepair.getByRole('button', { name: 'Reset', exact: true }).click();
    await pathRepair.getByRole('button', { name: /The changed prior/ }).click();
    await pathRepair.getByLabel('yes, it does here', { exact: true }).check();
    await pathRepair.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(pathRepair.locator('.hmm-verdict'), /your prediction matches/i);
    await screenshot(pathRepair, 'hmm-legal-changed-prior-desktop.png');
    // Construction: the pointwise route fails, B to A solves.
    const repairTask = pathRepair.locator('.hmm-construction');
    await pathRepair.getByRole('button', { name: 'Reset', exact: true }).click();
    await pathRepair.getByRole('button', { name: 'Select A → A' }).click();
    await repairTask.getByRole('button', { name: 'Submit this attempt' }).click();
    await checkText(repairTask.locator('.hmm-verdict'), /Not yet\./);
    await checkText(repairTask.locator('.hmm-verdict'), /uses an edge the model forbids/);
    await pathRepair.getByRole('button', { name: 'Select B → A' }).click();
    await repairTask.getByRole('button', { name: 'Submit this attempt' }).click();
    await checkText(repairTask.locator('.hmm-verdict'), /Solved\./);
    await screenshot(pathRepair, 'hmm-legal-construction-desktop.png');
    records.push({ case: 'Legal-path investigation: path scores withheld until commitment, forbidden edges drawn '
      + 'in their own vocabulary, a pointwise route reported at exactly zero, the changed prior under which both '
      + 'decoders agree, and a construction that refuses the forbidden selection and accepts B to A' });

    // ------------------------------------------- 5. boundaries and counts (I4)
    const boundary = labNamed('Move the boundary and watch the event totals follow');
    assert.equal(await boundary.locator('.hmm-verdict').count(), 0, 'no verdict on first paint');
    assert.equal(await boundary.locator('.hmm-table').count(), 0, 'and no count table');
    await screenshot(boundary, 'hmm-boundary-initial-desktop.png');
    // The concatenated default: one start, three transitions, four emissions.
    await boundary.locator('.hmm-group').nth(0).getByLabel('1', { exact: true }).check();
    await boundary.locator('.hmm-group').nth(1).getByLabel('3', { exact: true }).check();
    await boundary.locator('.hmm-group').nth(2).getByLabel('4', { exact: true }).check();
    await boundary.getByRole('button', { name: 'Apply and check' }).click();
    const boundaryVerdict = normalize(await boundary.locator('.hmm-verdict').innerText());
    assert.equal((boundaryVerdict.match(/your prediction matches/gi) ?? []).length, 3,
      `all three event totals are graded separately: ${boundaryVerdict}`);
    assert.ok(await boundary.locator('.hmm-table').count() >= 2, 'the flow and update tables appear after');
    await screenshot(boundary, 'hmm-boundary-concatenated-desktop.png');
    // Restoring the boundary changes the counts rather than storing them.
    await boundary.getByRole('button', { name: 'Two two-step sessions' }).click();
    await boundary.locator('.hmm-group').nth(0).getByLabel('2', { exact: true }).check();
    await boundary.locator('.hmm-group').nth(1).getByLabel('2', { exact: true }).check();
    await boundary.locator('.hmm-group').nth(2).getByLabel('4', { exact: true }).check();
    await boundary.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(boundary.locator('.hmm-history'), /response category changed|response categories agree/);
    const splitEmissionTable = boundary.locator('.hmm-table').filter({ hasText: 'The M-step: updated emission rows' });
    const splitEmissionExpected = JSON.parse(fs.readFileSync(
      'docs/teaching/drafts/hidden-markov-models-hmm/calculated-inputs.json', 'utf8')).mechanisms.split_update.emission;
    const splitEmissionCells = await splitEmissionTable.locator('tbody tr').evaluateAll(rows =>
      rows.map(row => [...row.querySelectorAll('td')].map(cell => cell.textContent.trim())));
    assert.deepEqual(splitEmissionCells, splitEmissionExpected.map(row => row.map(value => value.toFixed(6))),
      'both updated emission rows match the independently recorded complete split update');
    await screenshot(boundary, 'hmm-boundary-split-desktop.png');
    // The discriminating construction: same totals, different structure.
    const boundaryTask = boundary.locator('.hmm-construction');
    await boundaryTask.getByRole('button', { name: 'Submit this attempt' }).click();
    await checkText(boundaryTask.locator('.hmm-verdict'), /Solved\./);
    await boundary.getByRole('button', { name: /A one-step and a three-step session/ }).click();
    await boundaryTask.getByRole('button', { name: 'Submit this attempt' }).click();
    await checkText(boundaryTask.locator('.hmm-verdict'), /Not yet\./);
    await checkText(boundaryTask.locator('.hmm-verdict'), /reaches the same three totals/);
    await screenshot(boundary, 'hmm-boundary-construction-desktop.png');
    // Several tables exist only once a prediction is committed, so the
    // reset-state sweep below cannot see them. The emission row of the
    // fractional-flow table was cut mid-entry here while that sweep was green.
    const revealedHidden = await page.evaluate(() => [...document.querySelectorAll('.hmm-table')].map(block => {
      const scroll = block.querySelector('.hmm-table-scroll');
      return {
        caption: block.querySelector('.hmm-caption').textContent.trim().slice(0, 60),
        hiddenPx: scroll.scrollWidth - scroll.clientWidth,
      };
    }).filter(entry => entry.hiddenPx > 1));
    const revealedCount = await page.locator('.hmm-table').count();
    assert.ok(revealedCount >= 17, `${revealedCount} tables are visible with investigations committed`);
    assert.ok(revealedCount > tableFloor,
      'and committing an investigation really does add tables the reset-state sweep cannot see');
    assert.deepEqual(revealedHidden, [], 'and none of them hides a column at desktop width');
    records.push({
      case: `All ${revealedCount} tables visible with investigations committed show every column at 1366px`,
    });
    records.push({ case: 'Boundary investigation: three event totals graded separately, the fractional flow and '
      + 'one-step update revealed only after, and a construction that rejects a one-and-three split which reaches '
      + 'exactly the same three totals' });

    // ------------------------------------------------------- 6. dwell time (I5)
    const duration = labNamed('A self-transition is a duration assumption you can look at');
    await checkText(duration.locator('.hmm-state-strip').first(), /recorded after your prediction/);
    assert.ok(!(normalize(await duration.locator('.hmm-state-strip').first().innerText()).includes('3.333333')),
      'the mean duration is not printed before it is predicted');
    assert.ok(await duration.locator('svg rect.hmm-bar.is-duration').count() === 8,
      'while the eight duration bars are visible, because looking at the shape is the work');
    assert.equal(await duration.locator('svg rect.hmm-bar.is-tail').count(), 1, 'plus one tail bar');
    await screenshot(duration, 'hmm-duration-initial-desktop.png');
    await duration.getByLabel('exactly the same', { exact: true }).check();
    await duration.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(duration.locator('.hmm-verdict'), /your prediction matches/i);
    await checkText(duration.locator('.hmm-verdict'), /constant hazard/);
    await screenshot(duration, 'hmm-duration-checked-desktop.png');
    // The absorbing case has no finite mean rather than a very large one.
    await duration.getByRole('button', { name: 'Reset', exact: true }).click();
    await duration.getByRole('button', { name: 'a = 1', exact: true }).click();
    await duration.getByLabel('exactly the same', { exact: true }).check();
    await duration.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(duration, /no finite mean/);
    assert.ok(!normalize(await duration.locator('.hmm-verdict').innerText()).includes('Infinity'),
      'and it is never reported as infinity');
    await screenshot(duration, 'hmm-duration-absorbing-desktop.png');
    records.push({ case: 'Dwell-time investigation: the mean and the hazard withheld until the prediction is '
      + 'recorded while the bars stay visible, the constant hazard confirmed, and an absorbing state reported as '
      + 'having no finite mean rather than a very large one' });

    // --------------------------------------------------- 7. real sentences (I6)
    const tagging = labNamed('Two decision rules, the same fitted counts, real sentences');
    assert.equal(await tagging.locator('.hmm-token').count(), 0, 'no token chips before the prediction');
    assert.equal(await tagging.locator('.hmm-verdict').count(), 0, 'and no verdict');
    await checkText(tagging.locator('.hmm-role'), /recorded results/);
    await screenshot(tagging, 'hmm-tagging-initial-desktop.png');
    await tagging.getByLabel('the HMM', { exact: true }).check();
    await tagging.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(tagging.locator('.hmm-verdict'), /your prediction matches/i);
    const chips = await tagging.locator('.hmm-token').count();
    assert.equal(chips, data.developmentSentences[27].tokens.length, 'one chip per token of the chosen sentence');
    await checkText(tagging, /ref concealed/);
    await screenshot(tagging, 'hmm-tagging-concealed-desktop.png');
    await tagging.getByRole('button', { name: /Reveal the reference labels/ }).click();
    await checkText(tagging, /ref Other · PUNCT|ref Noun · PROPN/);
    assert.ok(await tagging.locator('.hmm-token.is-repair').count() >= 1,
      'the repair in this specimen is marked once the reference is revealed');
    await screenshot(tagging, 'hmm-tagging-revealed-desktop.png');
    // The aggregate stays locked until a repair and a break are both named.
    assert.ok(!normalize(await tagging.innerText()).includes('Only now'),
      'the aggregate is locked before the two positions are identified');
    const identify = tagging.locator('.hmm-construction');
    // It starts unsolved: neither default names a repair or a break.
    await identify.getByRole('button', { name: 'Check both positions' }).click();
    await checkText(identify.locator('.hmm-verdict'), /Not yet\./);
    assert.ok(!normalize(await tagging.innerText()).includes('Only now'),
      'and a failed identification does not unlock the aggregate');
    const repair = data.decisionChanges.repairs[8];
    const broken = data.decisionChanges.breaks[0];
    await tagging.getByLabel('Repair — sentence index').fill(String(repair.sentence));
    await tagging.getByLabel('Repair — token position').fill(String(repair.position));
    await tagging.getByLabel('Break — sentence index').fill(String(broken.sentence));
    await tagging.getByLabel('Break — token position').fill(String(broken.position));
    await identify.getByRole('button', { name: 'Check both positions' }).click();
    await checkText(identify.locator('.hmm-verdict'), /Solved\./);
    await checkText(tagging, /net gain of 4 tokens/);
    await screenshot(tagging, 'hmm-tagging-aggregate-desktop.png');
    await tagging.getByLabel('Repair — sentence index').fill('0');
    assert.equal(await identify.locator('.hmm-verdict').count(), 0, 'editing a submitted construction retires its verdict');
    assert.doesNotMatch(await tagging.innerText(), /Only now: the aggregate/, 'an edited identification relocks its aggregate');
    // A tied sentence says so.
    await tagging.getByRole('button', { name: 'Reset', exact: true }).click();
    for (const label of ['Repair — sentence index', 'Repair — token position', 'Break — sentence index', 'Break — token position']) {
      assert.equal(await tagging.getByLabel(label).inputValue(), '0', `Reset clears ${label}`);
    }
    const tied = data.tieAudit.configurations[1].tiedSentences[1].sentence;
    await tagging.getByLabel(/Development sentence/).selectOption(String(tied));
    await tagging.getByLabel('the HMM', { exact: true }).check();
    await tagging.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(tagging.locator('.hmm-note').last(), /exactly.{0,3} equal probability/);
    await screenshot(tagging, 'hmm-tagging-tied-desktop.png');
    records.push({ case: 'Tagging investigation: no chips or labels before the prediction, the reference concealed '
      + 'until revealed, the aggregate locked until a repair and a break are both named against the recorded '
      + 'outcomes, and a tied sentence declaring its exact tie' });

    // -------------------------------------- sequential audit: legal edge cases
    for (const [offset, accepted] of [[1e-9, true], [-1e-9, true], [1.01e-9, false], [-1.01e-9, false]]) {
      await pathLab.getByRole('button', { name: 'Reset', exact: true }).click();
      await pathLab.getByRole('button', { name: 'Sunny, Sunny, Sunny, Rainy' }).click();
      await checkText(pathLab.locator('.hmm-numeric-guess'), /Answers within 1e-9 are accepted/);
      await pathLab.getByLabel('the most probable of them', { exact: true }).check();
      await pathLab.locator('.hmm-numeric-guess input').fill(String(0.0031104 + offset));
      await pathLab.getByRole('button', { name: 'Apply and check' }).click();
      await checkText(pathLab.locator('.hmm-verdict'), accepted ? /within 1e-9/ : /outside 1e-9/);
    }
    await pathLab.getByRole('button', { name: 'Reset', exact: true }).click();
    for (const name of ['All Rainy', 'All Sunny']) {
      await pathLab.getByRole('button', { name, exact: true }).click();
      await pathLab.getByLabel('less probable than the best one', { exact: true }).check();
      await pathLab.getByRole('button', { name: 'Apply and check' }).click();
    }
    await checkText(pathLab.locator('.hmm-history'), /response categories agree.*does not mean the numerical values or inputs stayed fixed/);

    await pathRepair.getByRole('button', { name: 'Reset', exact: true }).click();
    const priorEditor = pathRepair.locator('.hmm-row-editor');
    for (const row of [[1/3, 1/3, 1/3], [1e-7, 0.5, 0.4999999]]) {
      await priorEditor.locator('input').fill(row.join(', '));
      await priorEditor.getByRole('button', { name: 'Validate and apply this row' }).click();
      assert.equal(await priorEditor.locator('input').getAttribute('aria-invalid'), 'false');
      assert.deepEqual((await priorEditor.locator('input').inputValue()).split(',').map(Number), row,
        'applied probability row round-trips thirds and tiny entries exactly');
      assert.equal(await priorEditor.locator('.hmm-field-error').count(), 0);
    }

    await laterLab.getByRole('button', { name: 'Reset', exact: true }).click();
    await laterLab.getByRole('button', { name: /One report only/ }).click();
    await laterLab.locator('.hmm-construction').getByRole('button', { name: 'Submit this attempt' }).click();
    await checkText(laterLab.locator('.hmm-construction .hmm-verdict'), /Not yet.*Time 1 filtering has no value/s);

    await boundary.getByRole('button', { name: 'Reset', exact: true }).click();
    for (let count = 0; count < 4; count += 1) {
      await boundary.getByRole('button', { name: 'Add a step to the last recording' }).click();
    }
    for (const length of [8, 9]) {
      if (length === 9) await boundary.getByRole('button', { name: 'Add a step to the last recording' }).click();
      await boundary.locator('.hmm-group').nth(0).getByLabel('1', { exact: true }).check();
      await boundary.locator('.hmm-group').nth(1).getByLabel('more than 6', { exact: true }).check();
      await boundary.locator('.hmm-group').nth(2).getByLabel(length === 8 ? '8' : 'more than 8', { exact: true }).check();
      await boundary.getByRole('button', { name: 'Apply and check' }).click();
      const verdict = await boundary.locator('.hmm-prediction .hmm-verdict').innerText();
      assert.equal((verdict.match(/your prediction matches/gi) ?? []).length, 3,
        `${length-1} transitions have an available, accepted response: ${verdict}`);
    }
    await screenshot(boundary, 'hmm-sequential-long-recording-desktop.png');

    await duration.getByRole('button', { name: 'Reset', exact: true }).click();
    await duration.getByRole('button', { name: 'a = 0', exact: true }).click();
    await duration.getByLabel('undefined — that elapsed history is impossible', { exact: true }).check();
    await duration.locator('.hmm-numeric-guess input').fill('1');
    await duration.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(duration.locator('.hmm-verdict'), /your prediction matches.*Surviving ten steps has probability zero/s);
    await checkText(duration.locator('.hmm-verdict'), /within 0\.000001/);
    await screenshot(duration, 'hmm-sequential-zero-survival-desktop.png');

    const spellingDetails = tagging.locator('details').filter({ hasText: 'Change one spelling and see what the encoding does with it' });
    if (!(await spellingDetails.getAttribute('open'))) await spellingDetails.locator('summary').click();
    for (const [fit, sentenceIndex] of [[0, 2], [1, 2], [1, 39]]) {
      await tagging.getByLabel(/Development sentence/).selectOption(String(sentenceIndex));
      await tagging.getByLabel('Smoothing strength, held fixed while you switch decoders').selectOption(String(fit));
      await checkText(spellingDetails.locator('.hmm-state-strip'), /symbols moved: no.*path moved: no/);
      const unknownPosition = data.developmentSentences[sentenceIndex].symbols.indexOf(0);
      assert(unknownPosition >= 0, 'the tie fixture contains a real unknown token');
      await tagging.getByLabel('Token position to respell').fill(String(unknownPosition));
      await tagging.getByLabel('Token position to respell').press('Tab');
      assert.equal(await tagging.getByLabel('Token position to respell').inputValue(), String(unknownPosition));
      for (const spelling of ['zzqwx', 'qxvblorp']) {
        await tagging.getByLabel('A replacement spelling').selectOption(spelling);
        await checkText(spellingDetails.locator('.hmm-state-strip'), /symbols moved: no.*path moved: no/);
      }
    }
    await tagging.getByLabel('Token position to respell').fill('10');
    await tagging.getByLabel('Token position to respell').press('Tab');
    await tagging.getByLabel(/Development sentence/).selectOption('27');
    assert.equal(await tagging.getByLabel('Token position to respell').inputValue(), '0', 'shorter specimen resets the spelling position');
    assert.equal(await tagging.getByLabel('A replacement spelling').inputValue(), '', 'shorter specimen clears the old replacement');
    records.push({ case: 'Sequential audit: four numeric tolerance boundaries, honest category history, exact probability-row round trips, one-report construction rejection, seven/eight transition responses, zero-survival conditional hazard, three recorded-tie spelling nulls, construction invalidation and complete reset' });

    // --------------------------------- 8. figure geometry read back from the DOM
    // Every investigation is returned to its named fixture first, so the sweep
    // reads the state a reader meets rather than whatever the last case left.
    for (let index = 0; index < 6; index += 1) {
      await page.locator('.hmm-investigation').nth(index)
        .getByRole('button', { name: 'Reset', exact: true }).click();
    }
    await settle(page);
    const trellisGeometry = await page.evaluate(readTrellisGeometry, '.hmm-figure svg');
    assert.ok(trellisGeometry, 'the first figure drew an SVG');
    // Edge widths must be the widths the model computed, read back from the paint.
    const edgeRows = models.trellisEdges(weather, reports, 'sum');
    const drawnEdges = await page.evaluate(() => {
      const figures = [...document.querySelectorAll('.hmm-figure')];
      const target = figures.find(figure => (figure.querySelector('h4') || {}).textContent
        && figure.textContent.includes('Forward trellis for Walk, Shop, Walk, Clean'));
      if (!target) return null;
      const svg = target.querySelector('.hmm-panel svg');
      // DOM order, so each drawn edge can be paired with the model edge at the
      // same position rather than compared as an unordered multiset.
      return [...svg.querySelectorAll('line.hmm-edge')].map(line => ({
        classes: line.getAttribute('class'),
        attribute: line.getAttribute('stroke-width'),
        strokeWidth: Number(parseFloat(getComputedStyle(line).strokeWidth).toFixed(4)),
        x1: Number(line.x1.baseVal.value.toFixed(2)), y1: Number(line.y1.baseVal.value.toFixed(2)),
        x2: Number(line.x2.baseVal.value.toFixed(2)), y2: Number(line.y2.baseVal.value.toFixed(2)),
      }));
    });
    assert.ok(drawnEdges && drawnEdges.length === 12, `the forward trellis drew its twelve edges (${drawnEdges && drawnEdges.length})`);
    // PAIRED, not sorted. Comparing sorted multisets establishes that the twelve
    // widths are present, not that each edge received its own, so any
    // permutation across the twelve passed.
    const modelEdges = edgeRows.flatMap(row => row.edges);
    assert.equal(drawnEdges.length, modelEdges.length, 'one drawn edge per model edge');
    drawnEdges.forEach((drawn, index) => {
      const expected = models.edgeWidth(modelEdges[index].share);
      // The browser reports a computed length at five decimal places, so the
      // comparison is by tolerance; 1e-3 of a CSS pixel is far below anything a
      // stylesheet override could hide in.
      assert.ok(Math.abs(drawn.strokeWidth - expected) < 1e-3,
        `drawn edge ${index} (${modelEdges[index].from}->${modelEdges[index].to}) paints at `
        + `${drawn.strokeWidth}, and its own computed width is ${expected}`);
      assert.ok(drawn.attribute !== null,
        `drawn edge ${index} carries a stroke-width attribute, so it is inside the override guard's domain`);
      assert.ok(Math.abs(Number(drawn.attribute) - expected) < 1e-9,
        `and that attribute is the model's width, not a rounded one`);
    });
    const actualWidths = drawnEdges.map(edge => edge.strokeWidth);
    assert.equal(new Set(actualWidths).size, 12,
      `all twelve widths differ from one another (${new Set(actualWidths).size} distinct), so the encoding is visible`);
    // No bar is invisible: a rect present in the DOM but unpainted is the defect
    // that survived a fully green run in an earlier lesson.
    const bars = await page.evaluate(readPaintedRects, { root: '.hmm-lesson', selector: 'rect.hmm-bar' });
    // Eight paired belief bars, eight duration bars plus one tail bar, and
    // twelve count-flow bars: twenty-nine in all. An exact count rather than a
    // floor, so a figure that stops drawing its bars is caught rather than
    // merely dropping below a threshold nobody revisits.
    assert.equal(bars.length, 29, `twenty-nine bars were inspected, not ${bars.length}`);
    for (const bar of bars) {
      assert.notEqual(bar.fill, 'none', `a bar is painted rather than fill:none (${bar.classes})`);
      assert.ok(bar.fillOpacity > 0 && bar.opacity > 0, `a bar is not transparent (${bar.classes})`);
      // Belief and duration bars encode with height; count-flow bars are
      // horizontal and encode with width. Both dimensions must be real for the
      // rect to be on screen at all.
      assert.ok(bar.widthPx > 0.5 && bar.heightPx > 0.5,
        `a bar has real extent in both dimensions (${bar.classes}: ${bar.widthPx} by ${bar.heightPx})`);
    }
    const encodingExtents = new Set(bars.map(bar => Math.round(Math.max(bar.widthPx, bar.heightPx) * 10)));
    assert.ok(encodingExtents.size >= 12,
      `the bars genuinely differ in their encoding dimension (${encodingExtents.size} distinct extents)`);
    // The nodes and observation cards of a trellis are painted too.
    const nodes = await page.evaluate(readPaintedRects,
      { root: '.hmm-lesson', selector: 'rect.hmm-node, rect.hmm-card' });
    assert.equal(nodes.length, 71, `exactly 71 trellis nodes and observation cards were inspected, not ${nodes.length}`);
    for (const node of nodes) {
      assert.ok(node.widthPx > 4 && node.heightPx > 4, `a trellis node has real extent (${node.classes})`);
      assert.notEqual(node.stroke, 'none', `and a visible outline (${node.classes})`);
    }
    // The general form of the defect this check exists for: a CSS declaration
    // outranks a presentation attribute, so an encoding can be correct in the
    // DOM and absent from the paint. Every element that sets one is compared.
    const attributeOverrides = await page.evaluate(() => {
      const findings = [];
      for (const node of document.querySelectorAll('.hmm-lesson svg [stroke-width]')) {
        const declared = parseFloat(node.getAttribute('stroke-width'));
        const painted = parseFloat(getComputedStyle(node).strokeWidth);
        if (!Number.isFinite(declared) || !Number.isFinite(painted)) continue;
        if (Math.abs(declared - painted) > 0.01) {
          findings.push({
            element: node.tagName, classes: node.getAttribute('class'),
            attribute: declared, painted,
          });
        }
      }
      return findings;
    });
    assert.deepEqual(attributeOverrides, [],
      'no stylesheet rule overrides a stroke width a component computed');
    const attributedElements = await page.evaluate(() =>
      document.querySelectorAll('.hmm-lesson svg [stroke-width]').length);
    assert.ok(attributedElements >= 40,
      `${attributedElements} elements set a stroke width attribute, so this check had a real subject set`);
    // Every edge must be INSIDE that domain. The guard compares attributes with
    // painted values, so an element with no attribute is not a passing case, it
    // is an unexamined one — which is how a zero-mass edge came to paint wider
    // than a nonzero one without anything noticing.
    const edgesWithoutAttribute = await page.evaluate(() =>
      [...document.querySelectorAll('.hmm-lesson svg line.hmm-edge')]
        .filter(edge => edge.getAttribute('stroke-width') === null)
        .map(edge => edge.getAttribute('class')));
    assert.deepEqual(edgesWithoutAttribute, [],
      'no drawn edge sits outside the override guard by carrying no width attribute');
    const totalEdges = await page.evaluate(() =>
      document.querySelectorAll('.hmm-lesson svg line.hmm-edge').length);
    assert.ok(totalEdges >= 40, `${totalEdges} drawn edges were checked for a width attribute`);

    records.push({
      case: 'Figure geometry read back from the paint: twelve forward-trellis edges whose rendered stroke widths '
        + 'are exactly the widths the model computed and genuinely differ across edges, every bar painted with '
        + 'nonzero fill and width, and every trellis node and observation card with real extent and a visible outline',
      distinctEdgeWidths: new Set(actualWidths).size,
      barsInspected: bars.length,
      distinctBarExtents: encodingExtents.size,
      nodesInspected: nodes.length,
      strokeWidthAttributesHonouredInThePaint: attributedElements,
    });

    // ------------------------- 9. the two objective windows, measured on screen
    const separations = await page.evaluate(measureObjectiveSeparation);
    assert.equal(separations.length, 2, 'the objective figure drew its two panels');
    const overviewPanel = separations.find(panel => /Overview/i.test(panel.heading));
    const detailPanel = separations.find(panel => /separates them/i.test(panel.heading));
    assert.ok(overviewPanel && detailPanel, `both panels were identified: ${JSON.stringify(separations)}`);
    assert.equal(overviewPanel.tracks, 4, 'the overview draws all four tracks');
    assert.equal(detailPanel.tracks, 4, 'and so does the detail panel');
    assert.ok(overviewPanel.smallestSeparationPx < 1,
      `the overview lands the finals less than a pixel apart, as its caption says `
      + `(${overviewPanel.smallestSeparationPx}px)`);
    assert.ok(detailPanel.smallestSeparationPx > 4,
      `while the detail window separates them by several pixels (${detailPanel.smallestSeparationPx}px)`);
    assert.ok(detailPanel.smallestSeparationPx > overviewPanel.smallestSeparationPx * 3,
      'and by a margin that justifies having two panels rather than one');
    records.push({
      case: 'The two objective windows measured as rendered separation on screen, not as a model claim',
      overviewSmallestSeparationPx: overviewPanel.smallestSeparationPx,
      detailSmallestSeparationPx: detailPanel.smallestSeparationPx,
    });

    // -------------------------------------------- 10. label geometry and type
    for (const width of [1366, 390, 320]) {
      await page.setViewportSize({ width, height: 1200 });
      await settle(page);
      const inspected = await page.evaluate(inspectLessonVisualLayout, '.hmm-lesson');
      const svgCount = await page.locator('.hmm-lesson svg').count();
      assert.ok(svgCount >= 25,
        `the sweep at ${width}px inspected ${svgCount} diagrams, which must not collapse to none`);
      // The inspector reports only diagrams that carry text, so the legend
      // swatches are legitimately absent from its list. The floor is what keeps
      // a silently empty sweep from passing.
      assert.ok(inspected.length >= 14,
        `and the inspector examined ${inspected.length} labelled diagrams of those ${svgCount}`);
      assert.ok(inspected.reduce((total, figure) => total + figure.labelCount, 0) > 200,
        'covering more than two hundred labels');
      const candidates = inspected.flatMap(figure => figure.issues)
        .filter(finding => !/hmm-grid|hmm-axis/.test(JSON.stringify(finding)));
      assert.deepEqual(candidates, [], `Straight-line or label collision at ${width}px`);
      const curveSubjects = await page.evaluate(() => {
        let curves = 0;
        let labels = 0;
        for (const svg of document.querySelectorAll('.hmm-lesson svg')) {
          curves += svg.querySelectorAll('path, polyline, polygon').length;
          labels += svg.querySelectorAll('text').length;
        }
        return { curves, labels };
      });
      assert.ok(curveSubjects.curves >= 15 && curveSubjects.labels >= 150,
        `the curve sweep at ${width}px had ${curveSubjects.curves} curves and ${curveSubjects.labels} labels to `
        + 'compare, so a clean result is not a result on nothing');
      const curveHits = await page.evaluate(sampleCurvesThroughLabels, '.hmm-lesson');
      assert.deepEqual(curveHits, [], `A curve runs through a label at ${width}px`);
      const readings = await page.evaluate(measureSvgTypeSizes, '.hmm-lesson');
      assert.ok(readings.length > 200, `${readings.length} SVG labels were measured at ${width}px`);
      const smallest = readings.reduce((low, entry) => (entry.renderedPx < low.renderedPx ? entry : low));
      assert.ok(smallest.renderedPx >= 9,
        `the smallest rendered label at ${width}px is ${smallest.renderedPx}px in "${smallest.text}" (${smallest.figure})`);
      for (const entry of readings) {
        assert.ok(entry.renderedPx <= 26,
          `no label is magnified past 26px at ${width}px: "${entry.text}" is ${entry.renderedPx}px`);
      }
      typeReadings.push({
        width, labels: readings.length,
        smallestRenderedPx: smallest.renderedPx,
        largestRenderedPx: readings.reduce((top, entry) => Math.max(top, entry.renderedPx), 0),
        distinctScales: new Set(readings.map(entry => entry.scale)).size,
      });
    }
    // The measurement must actually differ between widths. A size floor that
    // reported an identical figure at two widths was measuring user units rather
    // than rendered pixels, and also passed on an empty set.
    const desktop = typeReadings.find(entry => entry.width === 1366);
    const phone = typeReadings.find(entry => entry.width === 320);
    assert.notEqual(desktop.smallestRenderedPx, phone.smallestRenderedPx,
      'the rendered type size genuinely differs between 1366px and 320px, so this is measuring pixels');
    assert.ok(desktop.labels > 0 && phone.labels > 0, 'and it measured a non-empty set at both');
    records.push({
      case: 'No label leaves its own SVG, overlaps another label, is crossed by a straight foreground line, or is '
        + 'run through by a curve, at three widths; and every label’s RENDERED size stays between 9 and 26 '
        + 'pixels, measured as computed font size times the element-to-viewBox scale',
      curveSamplingNote: 'Curves are sampled along their own geometry and tested against every label box, because '
        + 'the shared inspector reads straight lines only. This lesson draws the state graph as <path> arcs and '
        + 'every objective track as a <polyline>.',
      typeReadings,
    });
    await page.setViewportSize({ width: 1366, height: 1000 });
    await settle(page);

    // ----------------------------- 10b. no table hides a column at desktop width
    // A scroll region is the right behaviour on a phone, where a wide table
    // stacks anyway. At desktop it hides a comparison column behind an overlay
    // scrollbar nobody sees, which is how the third column of a two-model
    // comparison sat outside its own panel while every assertion passed.
    for (const width of [1366, 900]) {
      await page.setViewportSize({ width, height: 1200 });
      await settle(page);
      const hidden = await page.evaluate(() => [...document.querySelectorAll('.hmm-table')].map(block => {
        const scroll = block.querySelector('.hmm-table-scroll');
        return {
          caption: block.querySelector('.hmm-caption').textContent.trim().slice(0, 60),
          hiddenPx: scroll.scrollWidth - scroll.clientWidth,
        };
      }).filter(entry => entry.hiddenPx > 1));
      if (width === 1366) {
        assert.deepEqual(hidden, [], 'no table hides a column at desktop width');
      } else {
        assert.equal(hidden.length, 1,
        `exactly one table scrolls at the intermediate ${width}px: ${JSON.stringify(hidden)}`);
      assert.ok(hidden[0].hiddenPx <= 45,
        `and by 42px, the measured value, not merely under some loose ceiling (${hidden[0].hiddenPx}px)`);
        records.push({
          case: `Intermediate ${width}px width inspected between the endpoints: `
            + `${hidden.length} table(s) scroll horizontally, none by more than 60px`,
          tablesScrollingAt900: hidden,
        });
      }
    }
    // Fifteen tables are present with every investigation at its fixture; the
    // rest appear only once a prediction is committed, and those are measured
    // in their own cases above. The floor is what stops an empty sweep passing.
    const tableCount = await page.locator('.hmm-table').count();
    assert.equal(tableCount, tableFloor, `${tableCount} tables are present with every investigation at its fixture`);
    records.push({ case: `All ${tableCount} tables show every column at 1366px without a horizontal scroll` });
    await page.setViewportSize({ width: 1366, height: 1000 });
    await settle(page);

    // ------------------------------------------- 11. figures at desktop width
    for (let index = 0; index < 9; index += 1) {
      await screenshot(page.locator('.hmm-figure').nth(index), `hmm-figure-${index + 1}-desktop.png`);
    }

    // -------------------------------------------------------- 12. loading closure
    const localScripts = [...new Set(requests.map(address => new URL(address))
      .filter(address => address.origin === new URL(base).origin && address.pathname.endsWith('.js'))
      .map(address => address.pathname.slice(1)))];
    assert.deepEqual(localScripts.filter(filename => bodyFiles.has(filename)), [bodyFile]);
    assert.deepEqual(localScripts.filter(filename => !allowedScripts.has(filename)), []);
    assert.ok(!requests.some(address => address.includes('/outlines/')));
    assert.ok(!requests.some(address => address.includes('ewt-sequences.json')),
      'the page never fetches the extract to render: the recorded results are in the module');
    assert.ok(!requests.some(address => /learn-assets\/(?!hmm\/)/.test(address)),
      'and never borrows another lesson’s assets');
    records.push({
      case: 'Fresh route requests only the selected lesson and its shared closure, never fetches the dataset to '
        + 'render, and never borrows another lesson’s assets',
      requestedScripts: localScripts,
      rawScriptBytes: localScripts.reduce((sum, filename) => sum + fs.statSync(path.join(distDir, filename)).size, 0),
      gzipEstimateBytes: localScripts.reduce(
        (sum, filename) => sum + gzipSync(fs.readFileSync(path.join(distDir, filename))).length, 0),
      interpretation: 'Built requested file sizes; the gzip estimate is not measured network compression or latency.',
    });

    // ----------------------------------------------------- 13. narrow widths
    for (const width of [390, 320]) {
      await page.setViewportSize({ width, height: 1400 });
      await settle(page);
      const overflow = await page.evaluate(() => ({
        document: document.documentElement.scrollWidth - document.documentElement.clientWidth,
        widest: [...document.querySelectorAll('.hmm-lesson *')]
          .map(node => ({ cls: node.className, right: node.getBoundingClientRect().right }))
          .reduce((top, entry) => (entry.right > top.right ? entry : top), { right: 0, cls: '' }),
      }));
      assert.ok(overflow.document <= 1,
        `no horizontal overflow at ${width}px (widest ${JSON.stringify(overflow.widest)})`);
      // KaTeX display math must fit, measured on the real page.
      const maths = await page.evaluate(widthLimit => [...document.querySelectorAll('.katex-display')]
        .map(node => ({
          width: Number(node.getBoundingClientRect().width.toFixed(1)),
          scroll: node.scrollWidth,
          client: node.clientWidth,
          text: node.textContent.trim().slice(0, 44),
        }))
        .filter(entry => entry.scroll > entry.client + 1 || entry.width > widthLimit), width);
      assert.deepEqual(maths, [], `Display formulas overflow at ${width}px`);
      const mathCount = await page.locator('.katex-display').count();
      assert.ok(mathCount >= 18, `${mathCount} display formulas were measured at ${width}px`);
      // Tables become labelled blocks, and captions stay outside the scroll box.
      const tables = await page.evaluate(() => [...document.querySelectorAll('.hmm-table')].map(block => ({
        captionWidth: Number(block.querySelector('.hmm-caption').getBoundingClientRect().width.toFixed(1)),
        blockWidth: Number(block.getBoundingClientRect().width.toFixed(1)),
        display: getComputedStyle(block.querySelector('table')).display,
        overflow: block.querySelector('.hmm-table-scroll').scrollWidth
          - block.querySelector('.hmm-table-scroll').clientWidth,
      })));
      assert.ok(tables.length >= 15, `${tables.length} tables were measured at ${width}px`);
      for (const table of tables) {
        assert.equal(table.display, 'block', 'every numeric table stacks into labelled rows');
        assert.ok(table.captionWidth > table.blockWidth * 0.8,
          `a caption keeps its full width rather than collapsing to its longest word (${table.captionWidth} of ${table.blockWidth})`);
        assert.ok(table.overflow <= 1, 'and a stacked table does not scroll sideways');
      }
      if (width === 390) {
        for (let index = 0; index < 6; index += 1) {
          await screenshot(page.locator('.hmm-investigation').nth(index), `hmm-investigation-${index + 1}-390.png`);
        }
        for (let index = 0; index < 9; index += 1) {
          await screenshot(page.locator('.hmm-figure').nth(index), `hmm-figure-${index + 1}-390.png`);
        }
      } else {
        for (const index of [0, 1, 3, 4, 6]) {
          await screenshot(page.locator('.hmm-figure').nth(index), `hmm-figure-${index + 1}-320.png`);
        }
        for (const index of [1, 2, 5]) {
          await screenshot(page.locator('.hmm-investigation').nth(index), `hmm-investigation-${index + 1}-320.png`);
        }
      }
      records.push({
        case: `Narrow ${width}px layout: no horizontal overflow, ${mathCount} display formulas each inside its own `
          + `width, ${tables.length} numeric tables stacked into labelled rows with their captions at full width`,
      });
    }
    await page.setViewportSize({ width: 1366, height: 1000 });
    await settle(page);

    // -------------------------- 14. keyboard reach, control ranges, raw code
    for (let index = 0; index < 6; index += 1) {
      await page.locator('.hmm-investigation').nth(index)
        .getByRole('button', { name: 'Reset', exact: true }).click();
    }
    await settle(page);
    const reach = await page.evaluate(() => [...document.querySelectorAll('.hmm-investigation')].map(lab => {
      const reachable = [...lab.querySelectorAll('button, input, select, textarea')].filter(node => !node.disabled);
      return { controls: reachable.length, radios: reachable.filter(node => node.type === 'radio').length };
    }));
    assert.equal(reach.length, 6, 'six investigations were measured for keyboard reach');
    reach.forEach((entry, index) => {
      assert.ok(entry.controls > 6, `investigation ${index + 1} exposes ${entry.controls} enabled focusable controls`);
      assert.ok(entry.radios >= 2, `investigation ${index + 1} exposes its prediction as radios`);
    });
    await duration.getByRole('slider').first().focus();
    assert.equal(await page.evaluate(() => document.activeElement.getAttribute('type')), 'range',
      'a control takes keyboard focus');
    await page.keyboard.press('Tab');
    assert.ok(await page.evaluate(() => document.activeElement !== document.body),
      'focus moves on rather than being dropped');
    // Every value the prose names must be reachable on the control that sets it.
    const numericFields = await page.evaluate(() =>
      [...document.querySelectorAll('.hmm-investigation input[type="number"], .hmm-investigation input[type="range"]')]
        .map(node => ({
          label: (node.closest('label')?.querySelector('span')?.textContent || '').trim().slice(0, 44),
          guess: Boolean(node.closest('.hmm-numeric-guess')),
          type: node.type, min: node.min, max: node.max, step: node.step,
        })));
    const controlRanges = numericFields.filter(entry => !entry.guess);
    const guessFields = numericFields.filter(entry => entry.guess);
    assert.ok(controlRanges.length >= 8, `${controlRanges.length} model controls carry an explicit range`);
    assert.deepEqual(controlRanges.filter(entry => entry.min === '' || entry.max === ''), [],
      'every model control declares both bounds');
    assert.equal(guessFields.length, 6, 'one optional numeric guess per investigation');
    assert.deepEqual(guessFields.filter(entry => entry.min !== '' || entry.max !== ''), [],
      'and no prediction guess is bounded, so a wrong answer can be recorded');
    // Practice 7 asks for a = 0.8; practice 5 names three recording lengths.
    const stay = controlRanges.find(entry => /Self-transition probability/.test(entry.label));
    assert.ok(stay, 'the dwell-time control was found');
    const step = Number(stay.step);
    for (const named of [0, 0.7, 0.8, 0.95, 1]) {
      assert.ok(Math.abs(named / step - Math.round(named / step)) < 1e-9,
        `the prose names a = ${named}, which lands on the control's ${step} step`);
      assert.ok(named >= Number(stay.min) && named <= Number(stay.max),
        `and inside its ${stay.min} to ${stay.max} range`);
    }
    const queryField = controlRanges.find(entry => /Which time the two questions/.test(entry.label));
    assert.ok(queryField && Number(queryField.min) === 0, 'the queried time can reach the first step');
    // The code-on-page check normalizes whitespace, so Python indentation would
    // survive being destroyed. Read it raw.
    const rawProgram = await page.locator('.hmm-excerpt').first().textContent();
    assert.ok(/\n {4}observations = np\.asarray/.test(rawProgram),
      'the displayed excerpt keeps the indentation that makes it valid Python');
    records.push({
      case: 'Keyboard reach into all six investigations, focus moving on rather than being dropped, every model '
        + 'control declaring both bounds while no prediction guess is bounded, every self-transition value the '
        + 'prose names landing on its control’s step and inside its range, and a displayed excerpt whose '
        + 'Python indentation survives on the page',
      controlRanges: controlRanges.length,
    });

    // ------------------------------------------------------------ 15. sequence
    await page.locator('.reader-complete').click();
    assert.equal(new URL(page.url()).pathname.split('/').at(-1), topicId);
    await page.reload({ waitUntil: 'domcontentloaded' });
    await ready(page);
    assert.equal(await page.evaluate(id => JSON.parse(localStorage.getItem('kd-progress'))[id], topicId), true);
    await page.locator('.reader-footer__next').click();
    await page.waitForURL('**/bayesian-networks-causal-graphical-models?module=classical-ml');
    records.push({ case: 'Completion persists under the stable ID without auto-advance; Next opens the actual successor' });
    assert.deepEqual(errors, [], 'no uncaught page error');
    assert.deepEqual(failedAssets, [], 'no failed asset request');
    await context.close();

    // ------------------------------------------------------------ 16. recovery
    for (const failure of ['import', 'render']) {
      const isolated = await browser.newContext();
      const trial = await isolated.newPage();
      let inject = true;
      await trial.route(`**/${bodyFile}`, async intercepted => {
        if (!inject) return intercepted.continue();
        inject = false;
        if (failure === 'import') return intercepted.abort('failed');
        return intercepted.fulfill({
          status: 200, contentType: 'text/javascript',
          body: 'export default {content(){throw new Error("Controlled review failure")}}',
        });
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

    // ------------------------------------- 17. screenshots are distinguishable
    const digests = screenshotPaths.map(file => ({ file, digest: hash(file), bytes: fs.statSync(file).size }));
    assert.equal(new Set(screenshotPaths).size, screenshotPaths.length, 'every capture has its own path');
    const byDigest = new Map();
    for (const entry of digests) {
      if (byDigest.has(entry.digest)) {
        assert.fail(`Two captures hold byte-identical images: ${byDigest.get(entry.digest)} and ${entry.file}`);
      }
      byDigest.set(entry.digest, entry.file);
    }
    records.push({ case: `${digests.length} screenshots, each with its own path and its own content hash` });

    for (const [filename, expected] of Object.entries(sourceHashes)) {
      assert.equal(hash(filename), expected, `Source changed during check: ${filename}`);
    }
    assert.equal(hash(`${distDir}/.vite/manifest.json`), buildHash);
    const report = {
      startedAt, completedAt: new Date().toISOString(), topicId, status: 'passed',
      browser: await browser.version(), distDir, buildManifestHash: buildHash, sourceHashes,
      moduleTopicCount: module_.topicIds.length,
      dataset: {
        file: data.provenance.file, sha256: data.provenance.sha256, bytes: data.provenance.bytes,
        reservedScored: data.provenance.reservedScored,
      },
      nativeEnvironment: hmmExamples.hmmlearn.environment,
      recordedTokenCounts: data.configurations.map(entry => ({ key: entry.key, correct: entry.correct })),
      tieBands: data.tieAudit.configurations.map(entry => ({ smoothing: entry.smoothing, ...entry.tokenTotals })),
      typeReadings,
      records,
      screenshots: digests,
      visualInterpretation: 'The intended Space Grotesk font loaded before the checks. Screenshots capture '
        + 'informative states, not defaults only: every investigation before any commitment, every declared '
        + 'contrast, both branches of each construction grader, the concealed and revealed states of the real '
        + 'sentences, a tied sentence, an absorbing duration, all nine inline figures at desktop and at 390px, and '
        + 'the drawing-dense figures and investigations at 320px. Every capture has a distinct path and a distinct '
        + 'content hash. They require separate visual inspection, which was performed.',
    };
    fs.writeFileSync(evidencePath, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify({
      status: report.status, cases: records.length, evidencePath,
      sourceFiles: ownedFiles.length, screenshots: screenshotPaths.length,
    }));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });

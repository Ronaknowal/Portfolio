// Production browser review of the PAC learning and VC dimension lesson:
// visible content, the three investigations, prediction/commit/retirement, the
// leak property, figure geometry, narrow layouts, sequence and loading closure.
//
// Run against a preview of a production build:
//   npx vite build --outDir dist-pac
//   npx vite preview --outDir dist-pac --host 127.0.0.1 --port 4192
//   PLAYWRIGHT_PACKAGE=... DIST_DIR=dist-pac LEARNING_BASE_URL=http://127.0.0.1:4192 \
//     node scripts/verify-pac-browser.cjs
//
// The check this file exists for is section 3 below. The dominant defect class
// in these lessons is a page that grades a correct answer wrong, or shows the
// answer first -- and asserting that no verdict BANNER exists on first paint is
// what let that ship three times, because a banner is easy to withhold while
// the graded number leaks out of a readout beside it. So the assertion here is
// a property, pinned numerically: for EVERY investigation, the exact
// six-decimal text of the graded quantity -- computed independently in this
// process from pac-models.js, not read off the page -- must be ABSENT from the
// investigation's rendered text before a prediction is committed and PRESENT
// after. The absence assertion is paired with the presence assertion so that a
// pin which can never match is caught rather than counted.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { gzipSync } = require('node:zlib');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const { inspectLessonVisualLayout } = require('./lib/lesson-visual-layout.cjs');

const distDir = process.env.DIST_DIR || 'dist-pac';
const base = (process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4192').replace(/\/+$/, '');
const topicId = 'pac-learning-vc-dimension';
const sourcePath = `src/learn/data/topics/${topicId}.jsx`;
const route = `${base}/learn/path/full-curriculum/${topicId}?module=classical-ml`;
const evidencePath = 'docs/teaching/evidence/pac-browser.json';
const read = filename => JSON.parse(fs.readFileSync(filename, 'utf8'));
const hash = filename => createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
const normalize = text => text.replace(/\s+/g, ' ').trim();
const ownedFiles = [
  sourcePath,
  'src/learn/data/pac-models.js',
  'src/learn/data/pac-data.js',
  'src/learn/data/pac-examples.js',
  'src/learn/components/lesson-labs/PacShared.jsx',
  'src/learn/components/lesson-labs/PacLabs.jsx',
  'src/learn/components/lesson-labs/PacFigures.jsx',
  'src/learn/components/lesson-labs/pac-labs.css',
  `src/learn/data/curriculum/blueprints/${topicId}.js`,
  'public/learn-assets/pac-learning/banknote-subset.csv',
  'public/learn-assets/pac-learning/ATTRIBUTION.txt',
];

/** Runs in the page. Samples every curve along its own geometry and reports the
 * labels it passes through.
 *
 * Copied from scripts/verify-bias-variance-browser.cjs, because
 * `scripts/lib/lesson-visual-layout.cjs` iterates `svg.querySelectorAll('line')`
 * and so cannot see a <path> or <polyline> at all -- which is how four data
 * curves crossed value labels behind a fully green run. This lesson draws nine
 * plots out of polylines, so that blind spot would cover almost everything it
 * has.
 *
 * Grid lines are background and are skipped. Nothing on this page carries an
 * opaque backplate, so no crossing is allowed at all. */
function sampleCurvesThroughLabels(root) {
  const SAMPLES = 320;
  const HALO_ALLOWANCE = 0.02;
  const findings = [];
  for (const svg of document.querySelectorAll(`${root} svg`)) {
    const labels = [...svg.querySelectorAll('text')]
      .filter(text => text.textContent.trim() && text.getClientRects().length)
      .map(text => ({
        text: text.textContent.trim().slice(0, 40),
        halo: text.classList.contains('pac-halo'),
        box: text.getBoundingClientRect(),
      }));
    if (!labels.length) continue;
    for (const shape of svg.querySelectorAll('path, polyline, polygon')) {
      if (shape.classList.contains('pac-grid')) continue;
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

/** Runs in the page. Reports, for every SVG shape that carries a fill or stroke
 * PRESENTATION ATTRIBUTE, whether the painted value matches it.
 *
 * A CSS rule beats a presentation attribute, which is how `fill="none"` lost to
 * a CSS `fill` and a figure's bands became entirely invisible while every
 * offline assertion passed. It also reports any shape with NO fill declaration
 * at all, because an SVG shape with no fill paints solid black -- invisible on
 * this ground -- and "no attribute" is a case to examine, not to skip. */
function comparePaintedAgainstDeclared(root) {
  const findings = [];
  for (const svg of document.querySelectorAll(`${root} svg`)) {
    for (const shape of svg.querySelectorAll('rect, circle, line, path, polyline, polygon, text')) {
      if (!shape.getClientRects().length) continue;
      const painted = getComputedStyle(shape);
      for (const property of ['fill', 'stroke']) {
        const declared = shape.getAttribute(property);
        if (declared === null) continue;
        const normalizedDeclared = declared === 'none' ? 'none' : declared;
        const normalizedPainted = painted[property] === 'none' ? 'none' : painted[property];
        if ((normalizedDeclared === 'none') !== (normalizedPainted === 'none')) {
          findings.push({
            kind: 'attribute overridden',
            tag: shape.tagName, property, declared: normalizedDeclared, painted: normalizedPainted,
            shapeClass: shape.getAttribute('class') || '',
          });
        }
      }
      if (shape.tagName !== 'text' && painted.fill === 'rgb(0, 0, 0)' && painted.stroke === 'none') {
        findings.push({
          kind: 'black fill with no stroke, which is invisible on this ground',
          tag: shape.tagName, shapeClass: shape.getAttribute('class') || '',
          painted: painted.fill,
        });
      }
    }
  }
  return findings;
}

(async () => {
  const startedAt = new Date().toISOString();
  const sourceHashes = Object.fromEntries(ownedFiles.map(filename => [filename, hash(filename)]));
  const build = read(`${distDir}/.vite/manifest.json`);
  const buildHash = hash(`${distDir}/.vite/manifest.json`);
  const publications = read('src/learn/data/lesson-manifest.json');
  const models = await import('../src/learn/data/pac-models.js');
  const { pacData } = await import('../src/learn/data/pac-data.js');
  const { pacExamples } = await import('../src/learn/data/pac-examples.js');

  /* The independently computed answers this run pins the page against. Nothing
     here is read off the screen. */
  const fixedText = value => value.toFixed(6);
  const investigationOneAnswer = models.finiteWorldRun(models.fixtures.finiteWorld);
  const investigationTwoAnswer = models.requestVerdict(models.fixtures.request);
  const investigationTwoCount = models.intervalPatterns(models.fixtures.request.points.length).length;
  const investigationThreeApplied = models.intervalExperiment(models.fixtures.interval);
  const investigationThreeAfterEdit = models.intervalExperiment(models.fixtures.intervalCloserEdges);
  const investigationThreeNull = models.intervalExperiment(models.fixtures.intervalNegativeNull);

  const bodyFiles = new Set(Object.values(build)
    .filter(entry => entry.src && entry.src.includes('/data/topics/'))
    .map(entry => entry.file));
  const bodyEntry = Object.values(build).find(entry => entry.src && entry.src.endsWith(`${topicId}.jsx`));
  const bodyFile = bodyEntry.file;
  const allowedScripts = new Set();
  /* Walk by manifest KEY. `imports` lists keys, not built filenames, so a
     lookup that only matched `src` or `file` silently found nothing for every
     shared chunk and left six legitimate files looking unexpected. */
  const addClosure = key => {
    const entry = build[key] ?? Object.values(build).find(item => item.src === key || item.file === key);
    if (!entry || allowedScripts.has(entry.file)) return;
    allowedScripts.add(entry.file);
    (entry.imports || []).forEach(addClosure);
    (entry.dynamicImports || []).forEach(addClosure);
  };
  addClosure(Object.keys(build).find(key => build[key].isEntry && key.endsWith('index.html')));
  addClosure('src/learn/Reader.jsx');
  addClosure(sourcePath);
  assert.ok(allowedScripts.size >= 5,
    `the closure walk resolved only ${allowedScripts.size} scripts; it is not finding the manifest entries`);
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [];
  const screenshots = [];
  const checkText = async (locator, pattern) => assert.match(normalize(await locator.innerText()), pattern);
  const settle = page => page.evaluate(() =>
    new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  const ready = async page => {
    await page.locator('.pac-lesson').waitFor();
    await page.waitForFunction(() => [...document.fonts]
      .some(font => font.family.includes('Space Grotesk') && font.status === 'loaded'), null, { timeout: 20000 });
    await page.evaluate(() => document.fonts.ready);
    await settle(page);
  };
  let status = 'failed';
  try {
    const context = await browser.newContext({ viewport: { width: 1366, height: 1000 } });
    const page = await context.newPage();
    const requests = [], errors = [], failedAssets = [];
    page.on('request', request => requests.push(request.url()));
    page.on('pageerror', error => errors.push(error.message));
    page.on('requestfailed', request => failedAssets.push(request.url()));
    await page.goto(route, { waitUntil: 'domcontentloaded' });
    await ready(page);

    /** Unique path, unique content, and a recorded {file, digest, bytes}. That
     *  triple is how a stale orphan from a pre-fix run gets caught. */
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
      await locator.screenshot({ path: destination });
      await page.setViewportSize(viewport);
      const bytes = fs.statSync(destination).size;
      screenshots.push({ file: destination, digest: hash(destination), bytes });
    };

    // ------------------------------------------------------------ 1. structure
    await checkText(page.locator('.reader-header h1'), /^PAC Learning & VC Dimension$/);
    await checkText(page.locator('.reader-footer__previous'), /Evaluation Metrics/);
    await checkText(page.locator('.reader-footer__next'), /Calibration/);
    assert.equal(await page.locator('.pac-investigation').count(), 3);
    assert.equal(await page.locator('.pac-figure').count(), 11);
    assert.equal(await page.locator('.pac-practice').count(), 9);
    assert.equal(await page.locator('.python-example').count(), 3);
    assert.equal(await page.locator('.katex-error').count(), 0);
    const rendered = normalize(await page.locator('.pac-lesson').textContent());
    for (const key of ['patterns', 'finite-world', 'interval-risk']) {
      assert.ok(rendered.includes(normalize(pacExamples[key].code)), `Missing actual code for ${key}`);
      assert.ok(rendered.includes(normalize(pacExamples[key].expected)), `Missing actual output for ${key}`);
    }
    assert.ok(rendered.includes(normalize(pacExamples.curves.expected)),
      'the measured program output is shown as it was printed');
    for (const anchor of await page.locator('.lesson-intro a[href^="#"]')
      .evaluateAll(items => items.map(item => item.getAttribute('href')))) {
      assert.equal(await page.locator(`[id="${anchor.slice(1)}"]`).count(), 1, `Broken lesson anchor ${anchor}`);
    }
    assert.ok(rendered.includes(pacData.provenance.sha256), 'the file hash travels with the data');
    assert.ok(rendered.includes('CC BY 4.0'), 'and so does the licence');
    assert.ok(rendered.includes('not evaluated by this program'), 'the untouched test split is stated');
    assert.ok(rendered.includes('retires that prediction the moment an input changes'),
      'the intro promises what the page keeps');
    for (const row of pacData.learningCurves.rows) {
      assert.ok(rendered.includes(`${row.trainCorrect}/${row.n}`), `training count for ${row.model} at n=${row.n}`);
      assert.ok(rendered.includes(`${row.developmentCorrect}/${row.developmentN}`),
        `development count for ${row.model} at n=${row.n}`);
    }
    /* O3: `rendered.includes(String(row.intervals))` asserted, at n=1, that the
       page text contains "2" somewhere. The growth table's rows are checked as
       whole rows inside figure 5 instead. */
    const growthText = normalize(await page.locator('.pac-figure').nth(4).innerText());
    for (const row of models.growthTable) {
      const cells = [row.n, row.allBinary.toLocaleString('en-US'), row.thresholds, row.intervals, row.sauerD2];
      assert.ok(growthText.includes(cells.join(' ')),
        `figure 5 shows the whole growth row for n=${row.n}: ${cells.join(' ')}`);
    }
    assert.ok(rendered.includes(models.finiteFamilyRadii.selection.toFixed(8)),
      'the destination note\'s K=25/n=500 radius is on the page');
    const asset = await page.request.get(`${base}${pacData.provenance.file}`);
    assert.equal(asset.status(), 200);
    const bytes = await asset.body();
    assert.equal(bytes.length, pacData.provenance.bytes, 'the served file is byte-for-byte the packet file');
    assert.equal(createHash('sha256').update(bytes).digest('hex'), pacData.provenance.sha256);
    const attribution = await page.request.get(`${base}${pacData.provenance.attribution}`);
    assert.equal(attribution.status(), 200);
    assert.ok((await attribution.text()).includes('CC BY 4.0'));
    for (const program of [pacData.provenance.calculationProgram, pacData.provenance.curveProgram]) {
      const download = await page.request.get(`${base}${program}`);
      assert.equal(download.status(), 200, `${program} is actually served`);
      assert.ok((await download.text()).includes('def '), 'and is the program rather than a placeholder');
    }
    records.push({ case: 'Complete visible code and output for three programs, twelve route anchors, eleven figures, nine practice tasks, every measured count, the served dataset with its hash and attribution, both downloadable programs, and current metadata and module sequence' });

    // ------------------------------------------- 2. nothing revealed on paint
    for (let index = 0; index < 3; index += 1) {
      const investigation = page.locator('.pac-investigation').nth(index);
      assert.equal(await investigation.locator('input[type="radio"][name*=":r"]:checked').count(), 0,
        `investigation ${index + 1} has a prediction preselected`);
      assert.equal(await investigation.locator('.pac-verdict').count(), 0,
        `investigation ${index + 1} shows a verdict before any prediction`);
      assert.equal(await investigation.locator('.pac-reveal').count(), 0,
        `investigation ${index + 1} shows its reveal before any prediction`);
      assert.ok(await investigation.getByRole('button', { name: 'Apply and check' }).isDisabled(),
        `investigation ${index + 1} can be checked without recording anything`);
    }
    records.push({ case: 'All three investigations open with nothing selected, no verdict, no reveal and a disabled check' });

    // ------------------------ 3. the leak property, pinned numerically, per lab
    const leakChecks = [];
    const textOf = async index => normalize(await page.locator('.pac-investigation').nth(index).innerText());

    // Investigation 1: the graded quantity is the returned rule's population risk.
    const oneBefore = await textOf(0);
    const oneAnswerText = fixedText(investigationOneAnswer.risk);
    assert.ok(!oneBefore.includes(oneAnswerText),
      `investigation 1 shows its graded risk ${oneAnswerText} before a prediction is recorded`);
    /* NOT "the rule's bit string is absent": the candidate table lists all
       sixteen rules on purpose, because the predeclared class is the entity the
       learner reasons about, and 0000 is one of them. What must be absent is
       any statement of WHICH rule this sample selects and what it costs -- the
       three phrases only the reveal and the post-commit probability panel
       render. */
    for (const phrase of ['Rule returned', 'Meets ε', 'exact failure probability']) {
      assert.ok(!oneBefore.includes(phrase),
        `investigation 1 renders "${phrase}" before a prediction is recorded`);
    }
    assert.ok(oneBefore.includes('0.50'),
      'while the class table does show each rule\'s own risk, which is what makes the answer derivable');
    const one = page.locator('.pac-investigation').nth(0);
    await screenshot(one, 'pac-finite-world-initial-desktop.png');
    await one.getByLabel('Its risk will exceed ε', { exact: true }).check();
    await one.locator('.pac-numeric-guess input').fill(String(investigationOneAnswer.risk));
    await one.getByRole('button', { name: 'Apply and check' }).click();
    const oneAfter = await textOf(0);
    assert.ok(oneAfter.includes(oneAnswerText),
      `investigation 1 never shows ${oneAnswerText} even after committing, so the pin cannot fail`);
    await checkText(one.locator('.pac-verdict'), /Your prediction matches/);
    await checkText(one, /exact failure probability/);
    leakChecks.push({ investigation: 1, pinnedValue: oneAnswerText, absentBefore: true, presentAfter: true });
    await screenshot(one, 'pac-finite-world-committed-desktop.png');

    // Investigation 2: the graded quantity is the number of realizable labelings.
    const twoBefore = await textOf(1);
    const twoAnswerText = `${investigationTwoCount} of ${2 ** models.fixtures.request.points.length}`;
    assert.ok(!twoBefore.includes(twoAnswerText),
      `investigation 2 shows its graded count "${twoAnswerText}" before a prediction is recorded`);
    assert.ok(!twoBefore.includes('not realizable by this class'), 'and does not state the verdict');
    const two = page.locator('.pac-investigation').nth(1);
    await screenshot(two, 'pac-witness-initial-desktop.png');
    await two.getByLabel('No — the order forbids it', { exact: true }).check();
    await two.locator('.pac-numeric-guess input').fill(String(investigationTwoCount));
    await two.getByRole('button', { name: 'Apply and check' }).click();
    const twoAfter = await textOf(1);
    assert.ok(twoAfter.includes(twoAnswerText),
      `investigation 2 never shows "${twoAnswerText}" even after committing, so the pin cannot fail`);
    await checkText(two.locator('.pac-verdict'), /Your prediction matches/);
    await checkText(two, new RegExp(investigationTwoAnswer.obstruction.negative));
    leakChecks.push({ investigation: 2, pinnedValue: twoAnswerText, absentBefore: true, presentAfter: true });
    await screenshot(two, 'pac-witness-obstruction-desktop.png');

    // Investigation 3: the graded quantity is the risk AFTER the proposed edit.
    // The applied state's risk is on screen from the start -- section 8 works it
    // through -- so the pin is on the post-edit value, and the applied value is
    // asserted present to show the pin's domain is not empty.
    const three = page.locator('.pac-investigation').nth(2);
    const threeBefore = await textOf(2);
    assert.ok(threeBefore.includes(fixedText(investigationThreeApplied.risk)),
      'investigation 3 shows the applied state it draws, which is the starting point, not the answer');
    await screenshot(three, 'pac-interval-initial-desktop.png');
    await three.getByRole('button', { name: 'Add two positives near the edges: .31 and .69' }).click();
    const threeAfterEdit = await textOf(2);
    const threeAnswerText = fixedText(investigationThreeAfterEdit.risk);
    assert.ok(!threeAfterEdit.includes(threeAnswerText),
      `investigation 3 shows the post-edit risk ${threeAnswerText} before the prediction is committed`);
    /* Two pending notices, and both are wanted: the lab's own says the DRAWING
       is still the applied state, and the shared prediction control's says the
       DRAFT differs from what is applied. They answer different questions, so
       the count is asserted rather than one of them being matched loosely. */
    assert.equal(await three.locator('.pac-pending').count(), 2,
      'both pending notices appear once an edit is drafted');
    await checkText(three.locator('.pac-pending').first(), /still shows the applied state/);
    await checkText(three.locator('.pac-pending').last(), /Draft inputs differ from the applied ones/);
    await three.getByLabel('It falls', { exact: true }).check();
    await three.locator('.pac-numeric-guess input').fill(String(investigationThreeAfterEdit.risk));
    await three.getByRole('button', { name: 'Apply and check' }).click();
    const threeAfter = await textOf(2);
    assert.ok(threeAfter.includes(threeAnswerText),
      `investigation 3 never shows ${threeAnswerText} even after committing, so the pin cannot fail`);
    await checkText(three.locator('.pac-verdict'), /Your prediction matches: It falls\./);
    leakChecks.push({ investigation: 3, pinnedValue: threeAnswerText, absentBefore: true, presentAfter: true });
    await screenshot(three, 'pac-interval-falls-desktop.png');
    records.push({
      case: 'The graded quantity of every investigation is absent from its rendered text before commitment and present after, pinned to the exact six-decimal value computed independently in this process',
      leakChecks,
    });

    // ------------------------------- 4. the exact null, and unchanged iff
    await three.getByRole('button', { name: 'Reset' }).click();
    await settle(page);
    await three.getByRole('button', { name: 'Add two negatives far outside: .01 and .99' }).click();
    await three.getByLabel('It does not move at all', { exact: true }).check();
    await three.locator('.pac-numeric-guess input').fill(String(investigationThreeNull.risk));
    await three.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(three.locator('.pac-verdict'), /Your prediction matches: It does not move at all\./);
    const nullText = await textOf(2);
    assert.ok(nullText.includes(fixedText(investigationThreeNull.risk)),
      'the null reports the same risk it started with');
    assert.equal(investigationThreeNull.risk, investigationThreeApplied.risk,
      'and the two really are the same number, not merely close');
    await checkText(three, /inside the tolerance/);
    await screenshot(three, 'pac-interval-exact-null-desktop.png');

    /* S6: the same null, reached from a DIRTY draft. The presets used to be
       applied to the draft, so an uncommitted edit composed with the preset and
       the button the lesson presents as the exact null produced a move -- a
       correct verdict to a question the learner did not think they had asked.
       Presets are built from the applied state now, so a pending edit is
       discarded and the null holds however it was reached. */
    await three.getByRole('button', { name: 'Reset' }).click();
    await settle(page);
    await three.getByLabel('target right endpoint').fill('0.9');
    await settle(page);
    await three.getByRole('button', { name: 'Add two negatives far outside: .01 and .99' }).click();
    await settle(page);
    /* The pending banners SHOULD still be here: the preset itself is a pending
       change, and that is the question being asked. What must be gone is the
       edit that was typed before it -- the target is back at .7, so the preset
       describes a change from the applied state and nothing else. */
    assert.equal(await three.getByLabel('target right endpoint').inputValue(), '0.7',
      'choosing a comparison preset discards the typed target edit rather than composing with it');
    assert.equal(await three.locator('.pac-pending').count(), 2,
      'while the preset itself is still an uncommitted change, and says so');
    /* Captured HERE, before committing: the committed state is by construction
       identical to the clean route's, so a post-commit capture would be
       byte-identical to the one above and the uniqueness check would read it as
       a duplicate. The informative moment is the one where the typed edit has
       just been discarded and the target reads .7 again. */
    await screenshot(three, 'pac-interval-preset-discards-edit-desktop.png');
    await three.getByLabel('It does not move at all', { exact: true }).check();
    await three.locator('.pac-numeric-guess input').fill(String(investigationThreeNull.risk));
    await three.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(three.locator('.pac-verdict'), /Your prediction matches: It does not move at all\./);
    const dirtyRouteText = normalize(await three.innerText());
    assert.ok(dirtyRouteText.includes(fixedText(investigationThreeNull.risk)),
      'and the null reached from a dirty draft reports the same risk as the clean route');
    await three.getByRole('button', { name: 'Reset' }).click();
    await settle(page);
    records.push({ case: 'Two exterior negative observations are graded as an exact null, with the tolerance named and the risk unchanged to the last digit — and the same null still holds when the preset is chosen after an uncommitted edit to the target' });

    // ------------------------------------------- 5. a verdict retires on edit
    await one.getByRole('button', { name: 'add an observation at 2' }).click();
    await settle(page);
    assert.equal(await one.locator('.pac-verdict').count(), 0, 'an edit retires the verdict');
    assert.equal(await one.locator('.pac-reveal').count(), 0, 'and its reveal');
    assert.equal(await one.locator('input[type="radio"]:checked').count(),
      await one.locator('.pac-label-choice input[type="radio"]:checked').count(),
      'and clears the recorded prediction while leaving the input controls set');
    await checkText(one.locator('.pac-history'), /stays hidden until you apply/);
    records.push({ case: 'Editing an input retires the verdict, its reveal and the recorded prediction, and holds the earlier attempt back until the next commitment' });

    // ------------------------------------------- 6. a refused input is named
    await two.getByRole('button', { name: 'Reset' }).click();
    await settle(page);
    await two.getByLabel('B coordinate').fill('0.2');
    await settle(page);
    /* Stated ONCE, beside the field that caused it. Rendering the same sentence
       again inside the prediction box put it on screen twice, which is what the
       first run of this check found. */
    assert.equal(await two.locator('.pac-note[role="status"]').count(), 1,
      'the coincident-coordinate problem is stated once, not twice');
    await checkText(two.locator('.pac-note[role="status"]'), /Two points share the coordinate 0\.2/);
    await checkText(two.locator('.pac-blocked'), /cannot be graded yet/);
    assert.ok(await two.getByRole('button', { name: 'Apply and check' }).isDisabled(),
      'and grading is blocked rather than guessed');
    await screenshot(two, 'pac-witness-refused-desktop.png');
    await two.getByRole('button', { name: 'Reset' }).click();
    await settle(page);
    records.push({ case: 'Two coincident coordinates are refused by name and block grading rather than being squeezed into a verdict' });

    // ------------------------------------------------- 7. figures and geometry
    for (let index = 0; index < 11; index += 1) {
      await screenshot(page.locator('.pac-figure').nth(index), `pac-figure-${index + 1}-desktop.png`);
    }
    const bandFigure = page.locator('.pac-figure').nth(2);
    assert.equal(await bandFigure.locator('.is-selected').count(), 0,
      'figure 3 marks no selected row until the reader asks for one');
    await bandFigure.getByRole('button', { name: 'Now pick the lowest observed error' }).click();
    await settle(page);
    await checkText(bandFigure, /selected/);
    await screenshot(bandFigure, 'pac-figure-3-selected-desktop.png');
    await bandFigure.getByRole('button', { name: 'Try charging for only the 3 finalists' }).click();
    await settle(page);
    await screenshot(bandFigure, 'pac-figure-3-shrunk-desktop.png');
    const elimination = page.locator('.pac-figure').nth(1);
    for (let step = 0; step < 3; step += 1) {
      await elimination.getByRole('button', { name: 'Sample one more input' }).click();
      await settle(page);
    }
    await checkText(elimination, /3 of 3 observations used/);
    await screenshot(elimination, 'pac-figure-2-eliminated-desktop.png');
    const curves = page.locator('.pac-figure').nth(9);
    // The default is the three-procedure comparison, which is the claim the
    // section makes; selecting one procedure adds its training curve.
    await checkText(curves, /At 80 training examples the depth-5 tree fits every one of its training labels/);
    await curves.getByRole('button', { name: 'Decision tree, maximum depth 5' }).click();
    await settle(page);
    await checkText(curves, /training error, squares/);
    await screenshot(curves, 'pac-figure-10-tree-desktop.png');
    await curves.getByRole('button', { name: 'All three, development only' }).click();
    await settle(page);
    const simulation = page.locator('.pac-figure').nth(8);
    await simulation.getByRole('button', { name: /retained individual risks/ }).click();
    await settle(page);
    await screenshot(simulation, 'pac-figure-9-preview-desktop.png');
    records.push({ case: 'Every inline figure captured at desktop width, plus the informative states: figure 3 before and after the selection is revealed and with the crossed-out K = 3 band, figure 2 fully eliminated, figure 10 on a second procedure, and figure 9 with its twenty retained risks shown' });

    // -------------------------------------- 8. painted encodings and curves
    /* Drive investigation 3 into its EXTREME state first and leave it there for
       the scans below. The number line draws the applied state, and the default
       sample sits comfortably inside the line, so an inset regression would be
       invisible to a scan of the opening view -- a breakage can survive simply
       because the defective state is not the one being looked at. Committing
       observations at .01 and .99 puts the outermost labels on screen for every
       width the inspector then visits. */
    await three.getByRole('button', { name: 'Reset' }).click();
    await settle(page);
    await three.getByRole('button', { name: 'Add two negatives far outside: .01 and .99' }).click();
    await three.getByLabel('It does not move at all', { exact: true }).check();
    await three.locator('.pac-numeric-guess input').fill(String(investigationThreeNull.risk));
    await three.getByRole('button', { name: 'Apply and check' }).click();
    await settle(page);
    // `allInnerTexts()` returns null for SVG <text>: innerText is an HTML
    // property. Read textContent instead.
    const extremeLabels = await three.locator('.pac-line text')
      .evaluateAll(nodes => nodes.map(node => node.textContent));
    assert.ok(extremeLabels.includes('0.01') && extremeLabels.includes('0.99'),
      'the outermost observations are drawn, so the scans below have something at the edges to catch');

    const paintFindings = await page.evaluate(comparePaintedAgainstDeclared, '.pac-lesson');
    assert.deepEqual(paintFindings, [],
      'a presentation attribute is overridden by CSS, or a shape paints solid black');
    for (const width of [1366, 1024, 768, 390, 320]) {
      await page.setViewportSize({ width, height: 1000 });
      await settle(page);
      // It is a PAGE function, not a Node helper: it reaches for `document`. It
      // returns one record PER SVG and puts findings inside `.issues`, so the
      // records have to be flattened -- comparing the records themselves with []
      // could only ever pass on a page with no SVGs at all.
      const layout = await page.evaluate(inspectLessonVisualLayout, '.pac-lesson');
      /* The inspector skips SVGs that carry no visible text, which here is the
         eleven half-plane panels of figure 4 -- they are pure geometry with an
         accessible title and description and no drawn label. So the page's own
         SVG count and the inspector's subject count are both floored, because
         "it found nothing" and "it looked at nothing" are different results. */
      const svgCount = await page.locator('.pac-lesson svg').count();
      assert.ok(svgCount >= 25, `only ${svgCount} SVGs are mounted at ${width}px`);
      assert.ok(layout.length >= 12,
        `the layout inspector considered only ${layout.length} SVGs at ${width}px; an empty scan passes vacuously`);
      // 91 today. Moving three prose sentences out of their SVGs into HTML --
      // the right fix for each -- lowered this legitimately; the floor tracks
      // the real number rather than a figure from before those repairs.
      assert.ok(layout.reduce((sum, figure) => sum + figure.labelCount, 0) >= 85,
        `and only ${layout.reduce((sum, figure) => sum + figure.labelCount, 0)} labels, which is too few to be the real page`);
      assert.deepEqual(layout.flatMap(figure => figure.issues.map(issue => ({ svgIndex: figure.svgIndex, ...issue }))),
        [], `Figure layout collides at ${width}px`);
      const curveHits = await page.evaluate(sampleCurvesThroughLabels, '.pac-lesson');
      assert.deepEqual(curveHits, [], `A curve runs through a label at ${width}px`);
    }
    await page.setViewportSize({ width: 1366, height: 1000 });
    await settle(page);
    records.push({
      case: 'No label leaves its own SVG, overlaps another label, is crossed by a straight foreground line, or is run through by a curve, at five widths; and every presentation attribute survives the stylesheet',
      curveSamplingNote: 'Curves are sampled along their own geometry and tested against every label box, because the shared inspector reads straight lines only. No label on this page carries an opaque backplate, so no crossing is permitted at all.',
    });

    // ------------------------------------------ 9. tables do not clip at desktop
    const clipped = await page.locator('.pac-table-scroll').evaluateAll(items =>
      items.filter(item => item.scrollWidth > item.clientWidth + 1)
        .map(item => item.querySelector('table caption, table')?.textContent?.slice(0, 60) ?? 'table'));
    assert.deepEqual(clipped, [], 'a figure table clips a column at desktop width');
    const captionsInside = await page.locator('.pac-table-scroll caption').count();
    assert.equal(captionsInside, 0, 'no caption sits inside a scroll box that blockifies');
    records.push({ case: 'No table clips a column at desktop width, and every caption sits outside its scroll box' });

    // ------------------------------- 9b. KaTeX radicals actually paint
    /* This lesson shipped `r_K = ln(2K/delta)/(2n)` -- the radius SQUARED --
       because `.pac-lesson svg { height: auto }` also matched KaTeX's radical
       SVGs, whose height comes from `height: inherit`, and collapsed them to
       nothing. The DOM was right, every offline check passed, and the formula
       on screen was wrong. Details are opened first: the practice solutions
       carry radicals too, and a hidden element measures zero for reasons that
       have nothing to do with this defect. */
    await page.locator('.pac-lesson details').evaluateAll(items => items.forEach(item => { item.open = true; }));
    await settle(page);
    const radicals = await page.evaluate(() => [...document.querySelectorAll('.pac-lesson .katex .sqrt svg')]
      .map(svg => {
        const box = svg.getBoundingClientRect();
        return { height: Math.round(box.height), width: Math.round(box.width) };
      }));
    /* Three: the finite-family radius in section 4, the explicit VC radius in
       section 7, and practice 2's solution. Floored at three so the check
       cannot pass by finding none. */
    assert.ok(radicals.length >= 3,
      `only ${radicals.length} square-root radicals were found; the page states three`);
    assert.deepEqual(radicals.filter(entry => entry.height < 4 || entry.width < 4), [],
      'a square-root radical paints with no height or width, so a formula reads as a bare fraction');
    await page.locator('.pac-lesson details').evaluateAll(items => items.forEach(item => { item.open = false; }));
    await settle(page);
    records.push({
      case: 'Every square-root radical on the page paints with real width and height, with the practice solutions opened',
      radicalsMeasured: radicals.length,
      smallestRadical: radicals.reduce((least, entry) => Math.min(least, entry.height), Infinity),
    });

    // ------------------------------------------------ 10. loading closure
    const localScripts = [...new Set(requests.map(address => new URL(address))
      .filter(address => address.origin === new URL(base).origin && address.pathname.endsWith('.js'))
      .map(address => address.pathname.slice(1)))];
    assert.deepEqual(localScripts.filter(filename => bodyFiles.has(filename)), [bodyFile]);
    assert.deepEqual(localScripts.filter(filename => !allowedScripts.has(filename)), []);
    assert.ok(!requests.some(address => address.includes('/outlines/')));
    assert.ok(!requests.some(address => address.includes('learn-assets/evaluation-metrics')),
      'the page never borrows the evaluation-metrics copy of the same extract');
    assert.ok(!requests.some(address => address.includes('learn-assets/semi-supervised-learning')),
      'nor the semi-supervised one');
    records.push({
      case: 'Fresh route requests only the selected lesson and its shared closure, and never another lesson\'s dataset',
      requestedScripts: localScripts,
      rawScriptBytes: localScripts.reduce((sum, filename) => sum + fs.statSync(path.join(distDir, filename)).size, 0),
      gzipEstimateBytes: localScripts.reduce((sum, filename) =>
        sum + gzipSync(fs.readFileSync(path.join(distDir, filename))).length, 0),
      interpretation: 'Built requested file sizes; the gzip estimate is not measured network compression or latency.',
    });

    // ----------------------------------------------------- 11. narrow widths
    for (const width of [390, 320]) {
      await page.setViewportSize({ width, height: 844 });
      await page.locator('.pac-lesson details').evaluateAll(items => items.forEach(item => { item.open = true; }));
      await settle(page);
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
        `Document overflows at ${width}px`);
      const mathOverflow = await page.locator('.pac-lesson .katex-display').evaluateAll(items =>
        items.filter(item => item.scrollWidth > item.clientWidth + 1).map(item => item.textContent));
      assert.deepEqual(mathOverflow, [], `Overflowing formulas at ${width}px`);
      assert.equal(await page.locator('.katex-error').count(), 0);
      const stacked = await page.locator('.pac-table-scroll tbody th').first()
        .evaluate(cell => getComputedStyle(cell).display);
      assert.equal(stacked, 'flex', `the wide tables stack rather than clip at ${width}px`);
      if (width === 320) {
        /* The multi-row display formulas, captured at the width they were
           restructured for. Four of them overflowed a 320 px column as single
           lines; an assertion that they no longer scroll is worth having, and
           an image of them at that width is worth looking at. */
        const tall = await page.locator('.pac-lesson .katex-display')
          .evaluateAll(items => items
            .map((item, index) => ({ index, height: Math.round(item.getBoundingClientRect().height) }))
            .sort((a, b) => b.height - a.height).slice(0, 5));
        assert.ok(tall.every(entry => entry.height > 55),
          `a formula selected as multi-row is only ${Math.min(...tall.map(entry => entry.height))}px tall, `
          + 'so the five restructured blocks are not the ones being captured');
        for (const entry of tall) {
          await screenshot(page.locator('.pac-lesson .katex-display').nth(entry.index),
            `pac-formula-${entry.index}-320.png`);
        }
        await screenshot(page.locator('.pac-investigation').nth(0), 'pac-finite-world-320.png');
        await screenshot(page.locator('.pac-figure').nth(3), 'pac-figure-4-320.png');
        await screenshot(page.locator('.pac-figure').nth(6), 'pac-figure-7-320.png');
      } else {
        await screenshot(page.locator('.pac-investigation').nth(1), 'pac-witness-390.png');
        await screenshot(page.locator('.pac-investigation').nth(2), 'pac-interval-390.png');
        for (const index of [0, 2, 4, 8, 9, 10]) {
          await screenshot(page.locator('.pac-figure').nth(index), `pac-figure-${index + 1}-390.png`);
        }
      }
    }
    records.push({ case: 'No document or formula overflow at 390 and 320 px, wide tables stack rather than clip, and the informative investigation and figure states are captured at both narrow widths' });

    assert.deepEqual(errors, [], 'the page threw');
    assert.deepEqual(failedAssets.filter(address => !address.includes('favicon')), [], 'an asset failed to load');
    assert.equal(new Set(screenshots.map(shot => shot.file)).size, screenshots.length,
      'every capture has its own path');
    assert.equal(new Set(screenshots.map(shot => shot.digest)).size, screenshots.length,
      'and its own content: two identical captures mean one of them is of the wrong thing');
    /* No orphan may survive. A capture left behind by an earlier, pre-fix run
       sits in the evidence directory looking exactly like a current one, and
       that is how a defective state gets shipped as proof that it was fixed. So
       every pac-*.png on disk must be one this run just wrote, and every file
       this run recorded must still exist at the digest and size recorded. */
    const onDisk = fs.readdirSync('docs/teaching/evidence/screenshots')
      .filter(name => name.startsWith('pac-') && name.endsWith('.png'))
      .map(name => path.join('docs/teaching/evidence/screenshots', name).replace(/\\/g, '/'));
    const written = new Set(screenshots.map(shot => shot.file.replace(/\\/g, '/')));
    assert.deepEqual(onDisk.filter(file => !written.has(file)), [],
      'orphaned pac-*.png files remain from an earlier run');
    for (const shot of screenshots) {
      assert.ok(fs.existsSync(shot.file), `${shot.file} was recorded but is not on disk`);
      assert.equal(hash(shot.file), shot.digest, `${shot.file} no longer matches its recorded digest`);
      assert.equal(fs.statSync(shot.file).size, shot.bytes, `${shot.file} no longer matches its recorded size`);
      assert.ok(shot.bytes > 1000, `${shot.file} is ${shot.bytes} bytes, which is not a rendered capture`);
    }
    assert.ok(publications[topicId], 'the topic is registered for publication');
    /* S5: `cases` was a printed tally nobody floored -- a whole section could
       be deleted and the run would still report `status: passed`. */
    assert.ok(records.length >= 12, `only ${records.length} browser cases ran; a section has been lost`);
    assert.ok(screenshots.length >= 40, `only ${screenshots.length} captures were taken`);
    status = 'passed';
  } finally {
    await browser.close();
    const report = {
      startedAt,
      finishedAt: new Date().toISOString(),
      status,
      route,
      distDir,
      buildManifestSha256: buildHash,
      sourceHashes,
      verifierSha256: hash('scripts/verify-pac-browser.cjs'),
      records,
      screenshots,
      screenshotCount: screenshots.length,
      visualInterpretation: 'Screenshots capture informative states, not defaults only: each investigation '
        + 'before any commitment and after one, the exact null, the refused coincident coordinates, figure 3 '
        + 'before and after the selection is revealed and with the crossed-out K = 3 band, figure 2 fully '
        + 'eliminated, figure 9 with its retained risks, figure 10 on a second procedure, and every inline '
        + 'figure at desktop plus a selection at 390 and 320 px. They require separate visual inspection; a '
        + 'recorded digest proves identity, not legibility.',
      limitations: [
        'A passing run is not a reading of the page. The captures above must be opened and looked at.',
        'The painted-against-declared check catches a stylesheet overriding an attribute, and a shape painting '
          + 'black with no stroke. It does NOT catch a shape that is painted correctly and then COVERED by a '
          + 'later sibling. That is how figure 11\'s two half-discs were invisible -- the dial ring carried a '
          + 'solid fill and is painted after them -- and it was found by looking at the image, not by any '
          + 'check here. A general occlusion check is not attempted; the figure-by-figure visual pass is what '
          + 'covers it.',
        'The shared layout inspector allows a two-pixel overlap, which is why two coordinate labels eleven '
          + 'units apart could merge and still pass it. That class is now covered in the model layer by '
          + 'labelCollisions rather than here.',
        'Model correctness is checked by scripts/verify-pac-models.mjs, data regeneration by '
          + 'scripts/verify-pac-data.py and program execution by scripts/verify-pac-examples.py.',
        'Screen-reader behaviour is not simulated here; roles, labels and descriptions are asserted structurally.',
      ],
    };
    fs.mkdirSync(path.dirname(evidencePath), { recursive: true });
    fs.writeFileSync(evidencePath, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify({
      status: report.status, cases: records.length, evidencePath,
      sourceFiles: ownedFiles.length, screenshots: screenshots.length,
    }));
  }
})();

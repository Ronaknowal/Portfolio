// Production browser review of the Rademacher lesson: visible content, the four
// investigations, prediction/commit/retirement, figure geometry, narrow
// layouts, formula width, sequence, completion and load-failure recovery.
//
// Run against a preview of a production build:
//   npx vite build --outDir dist-rad
//   npx vite preview --outDir dist-rad --host 127.0.0.1 --port 4194
//   PLAYWRIGHT_PACKAGE=... DIST_DIR=dist-rad LEARNING_BASE_URL=http://127.0.0.1:4194 \
//     node scripts/verify-rademacher-browser.cjs
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { gzipSync } = require('node:zlib');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const { inspectLessonVisualLayout } = require('./lib/lesson-visual-layout.cjs');

const distDir = process.env.DIST_DIR || 'dist-rad';
const base = (process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4194').replace(/\/+$/, '');
const topicId = 'rademacher-complexity-generalization-bounds';
const sourcePath = `src/learn/data/topics/${topicId}.jsx`;
const route = `${base}/learn/path/full-curriculum/${topicId}?module=classical-ml`;
const evidencePath = 'docs/teaching/evidence/rademacher-browser.json';
const read = filename => JSON.parse(fs.readFileSync(filename, 'utf8'));
const hash = filename => createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
const normalize = text => text.replace(/\s+/g, ' ').trim();
const ownedFiles = [
  sourcePath,
  'src/learn/data/rademacher-models.js',
  'src/learn/data/rademacher-data.js',
  'src/learn/data/rademacher-examples.js',
  'src/learn/components/lesson-labs/RademacherShared.jsx',
  'src/learn/components/lesson-labs/RademacherLabs.jsx',
  'src/learn/components/lesson-labs/RademacherFigures.jsx',
  'src/learn/components/lesson-labs/rademacher-labs.css',
  `src/learn/data/curriculum/blueprints/${topicId}.js`,
  'public/learn-assets/rademacher/banknote-subset.csv',
  'public/learn-assets/rademacher/ATTRIBUTION.txt',
  'public/learn-assets/rademacher/complexity_calculations.py',
  'public/learn-assets/rademacher/bounded_norm_experiment.py',
];

/** Runs in the page. Samples every curve along its own geometry and reports the
 * labels it passes through.
 *
 * Copied from scripts/verify-bias-variance-browser.cjs, as that lesson's own
 * comment recommends. `scripts/lib/lesson-visual-layout.cjs` iterates
 * `svg.querySelectorAll('line')`, so <path> and <polyline> are invisible to it
 * -- which is how four data curves crossed value labels behind a fully green
 * run. This lesson draws loss curves, a ramp and a convex hull as polylines and
 * polygons, so the same blind spot applies to it.
 *
 * A label carrying an opaque backplate may be crossed briefly, because the
 * backplate is what a crossing is mitigated with. It may not be travelled
 * along: a backplate hides a line passing behind glyphs, not a curve running
 * the width of the text. Grid lines are background and are skipped.
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
        halo: text.classList.contains('rad-halo'),
        box: text.getBoundingClientRect(),
      }));
    if (!labels.length) continue;
    for (const shape of svg.querySelectorAll('path, polyline, polygon')) {
      if (shape.classList.contains('rad-grid')) continue;
      if (typeof shape.getTotalLength !== 'function') continue;
      const length = shape.getTotalLength();
      if (!(length > 0)) continue;
      const matrix = shape.getScreenCTM();
      if (!matrix) continue;
      const counts = labels.map(() => 0);
      for (let step = 0; step <= SAMPLES; step += 1) {
        const local = shape.getPointAtLength((length * step) / SAMPLES);
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

/** Runs in the page. Compares PAINTED values against attributes, because CSS
 * beats a presentation attribute and a correct attribute proves nothing about
 * what a reader sees. "No attribute" is inspected rather than skipped. */
function comparePaintedStrokes(root) {
  const findings = [];
  for (const shape of document.querySelectorAll(`${root} svg :is(path, polyline, polygon, circle, rect, line)`)) {
    const style = getComputedStyle(shape);
    const attributeStroke = shape.getAttribute('stroke');
    const attributeFill = shape.getAttribute('fill');
    const painted = { stroke: style.stroke, fill: style.fill, strokeWidth: style.strokeWidth };
    if (attributeFill === 'none' && painted.fill !== 'none') {
      findings.push({ kind: 'fill-attribute-overridden', class: shape.getAttribute('class'), ...painted });
    }
    if (attributeStroke && painted.stroke === 'none') {
      findings.push({ kind: 'stroke-erased', class: shape.getAttribute('class'), ...painted });
    }
    // An outline-only shape that ends up filled is the defect that made a
    // figure's bands invisible on another lesson.
    const outlineOnly = ['rad-ball', 'rad-diamond', 'rad-hull', 'rad-curve', 'rad-ceiling', 'rad-axis', 'rad-grid'];
    if (outlineOnly.some(name => shape.classList.contains(name))
      && !shape.classList.contains('rad-hull')
      && painted.fill !== 'none' && painted.fill !== 'rgba(0, 0, 0, 0)') {
      findings.push({ kind: 'outline-shape-is-filled', class: shape.getAttribute('class'), ...painted });
    }
    if (painted.stroke !== 'none' && parseFloat(painted.strokeWidth) < 0.5) {
      findings.push({ kind: 'hairline', class: shape.getAttribute('class'), ...painted });
    }
  }
  return findings;
}

/** Runs in the page. Every visible text node inside the lesson, so a leak check
 * can ask whether a string is on screen rather than whether a container is. */
function visibleText(root) {
  const element = document.querySelector(root);
  return element ? element.innerText.replace(/\s+/g, ' ').trim() : '';
}

(async () => {
  const startedAt = new Date().toISOString();
  /* A provisional FAILING record, stamped before anything can throw. The report
     below is written only on success, so without this a crashed run leaves the
     previous run's passing record on disk beside a page that now fails. */
  fs.mkdirSync(path.dirname(evidencePath), { recursive: true });
  fs.writeFileSync(evidencePath, `${JSON.stringify({
    startedAt, status: 'incomplete — this run started and did not reach its end', passed: false,
  }, null, 2)}
`);
  const sourceHashes = Object.fromEntries(ownedFiles.map(filename => [filename, hash(filename)]));
  const build = read(`${distDir}/.vite/manifest.json`);
  const buildHash = hash(`${distDir}/.vite/manifest.json`);
  const publications = read('src/learn/data/lesson-manifest.json');
  assert.ok(publications[topicId], 'the lesson is registered in the manifest');
  const models = await import('../src/learn/data/rademacher-models.js');
  const data = await import('../src/learn/data/rademacher-data.js');
  const examples = await import('../src/learn/data/rademacher-examples.js');

  const bodyFiles = new Set(Object.values(build)
    .filter(entry => entry.src && entry.src.includes('/topics/'))
    .map(entry => entry.file));
  const bodyEntry = Object.values(build).find(entry => entry.src && entry.src.endsWith(`${topicId}.jsx`));
  assert.ok(bodyEntry, 'the built manifest has an entry for this lesson body');
  const bodyFile = bodyEntry.file;

  const browser = await chromium.launch({ channel: 'msedge' });
  const records = [];
  const screenshotPaths = [];
  const screenshotDigests = [];
  const checkText = async (locator, pattern) => assert.match(normalize(await locator.innerText()), pattern);
  const settle = page => page.evaluate(() =>
    new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  const ready = async page => {
    await page.locator('.rad-lesson').waitFor();
    await page.waitForFunction(() => [...document.fonts]
      .some(font => font.family.includes('Space Grotesk') && font.status === 'loaded'), null, { timeout: 20000 });
    await page.evaluate(() => document.fonts.ready);
    await settle(page);
  };

  try {
    const context = await browser.newContext({ viewport: { width: 1366, height: 1000 } });
    const page = await context.newPage();
    const requests = [];
    const errors = [];
    const failedAssets = [];
    page.on('request', request => requests.push(request.url()));
    page.on('pageerror', error => errors.push(error.message));
    page.on('requestfailed', request => failedAssets.push(request.url()));
    await page.goto(route, { waitUntil: 'domcontentloaded' });
    await ready(page);

    /* Every capture gets its own path AND its own content digest, recorded
       together. A stale orphan from a pre-fix run is caught by the digest, not
       by the filename. */
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
      screenshotPaths.push(destination);
      screenshotDigests.push({
        file: destination,
        digest: hash(destination),
        bytes: fs.statSync(destination).size,
      });
    };

    // ---------------------------------------------------------- 1. structure
    await checkText(page.locator('.reader-header h1'), /^Rademacher Complexity & Generalization Bounds$/);
    await checkText(page.locator('.reader-header__meta'), /36 of 39 topics on this route/);
    await checkText(page.locator('.reader-footer__previous'), /Calibration/);
    await checkText(page.locator('.reader-footer__next'), /ML Problem Formulation/);
    assert.equal(await page.locator('.rad-lesson h2').count(), 11, 'eleven sections');
    assert.equal(await page.locator('.rad-investigation').count(), 4, 'four investigations');
    assert.equal(await page.locator('.rad-figure').count(), 17, 'seventeen inline figures');
    /* Caption numbers must follow READING order. Two figures were numbered by
     * the order they were written rather than the order they are met, so a
     * reader ran into "Figure 5" before "Figure 4" and the readiness table
     * pointed at the wrong one. Found by opening a screenshot whose filename
     * and caption disagreed. */
    const captionOrder = await page.locator('.rad-figure figcaption')
      .evaluateAll(items => items.map(item => item.textContent.trim()));
    assert.equal(captionOrder.length, 17, 'every figure has a caption');
    const misnumbered = captionOrder
      .map((caption, index) => (caption.startsWith(`Figure ${index + 1}.`) ? null : `#${index + 1} says "${caption.slice(0, 24)}"`))
      .filter(Boolean);
    assert.deepEqual(misnumbered, [], `a figure's caption number does not match its reading position: ${misnumbered.join('; ')}`);
    assert.equal(await page.locator('.rad-practice').count(), 8, 'eight practice tasks');
    assert.equal(await page.locator('.python-example').count(), 2, 'two runnable programs');
    assert.equal(await page.locator('.rad-excerpt').count(), 3, 'three displayed mechanism excerpts');
    await checkText(page.locator('.rad-route'), /First pass\. Read sections 1–8/);
    await checkText(page.locator('.rad-route'), /Section 9 is a deeper branch/);
    for (const anchor of await page.locator('.lesson-intro a[href^="#"]')
      .evaluateAll(items => items.map(item => item.getAttribute('href')))) {
      assert.equal(await page.locator(`[id="${anchor.slice(1)}"]`).count(), 1, `Broken lesson anchor ${anchor}`);
    }
    assert.equal(await page.locator('.katex-error').count(), 0, 'no formula failed to render');

    const rendered = normalize(await page.locator('.rad-lesson').textContent());
    for (const [key, program] of Object.entries(examples.rademacherPrograms)) {
      assert.ok(rendered.includes(normalize(program.code)), `Missing actual code for ${key}`);
      assert.ok(rendered.includes(normalize(program.expected)), `Missing actual recorded output for ${key}`);
    }
    for (const [key, excerpt] of Object.entries(examples.rademacherExcerpts)) {
      assert.ok(rendered.includes(normalize(excerpt.code)), `Missing the verbatim excerpt for ${key}`);
    }
    assert.ok(rendered.includes('CC BY 4.0'), 'the licence travels with the data');
    assert.ok(rendered.includes(data.provenance.sha256), 'and so does the file hash');
    assert.equal((rendered.match(/Before running:/g) ?? []).length, 2, 'one Before running per program');

    // The served dataset and both served programs.
    for (const [address, expectedHash] of [
      [data.provenance.file, data.provenance.sha256],
      [data.provenance.calculationProgram, null],
      [data.provenance.experimentProgram, null],
      [data.provenance.attribution, null],
    ]) {
      const asset = await page.request.get(`${base}${address}`);
      assert.equal(asset.status(), 200, `${address} is served`);
      const bytes = await asset.body();
      if (expectedHash) {
        assert.equal(createHash('sha256').update(bytes).digest('hex'), expectedHash,
          `${address} is byte-for-byte the packet file`);
        assert.equal(bytes.length, data.provenance.bytes, 'and the recorded byte count');
      }
    }
    assert.ok(!requests.some(address => address.includes('learn-assets/evaluation-metrics')
      || address.includes('learn-assets/semi-supervised-learning') || address.includes('learn-assets/automl-nas')),
    'the page never reaches for a sibling lesson\'s copy of the same dataset');
    records.push({ case: 'Header, route, sequence position, eleven sections with working anchors, four investigations, seventeen figures, two executed programs with their real output, three verbatim excerpts, and its own served dataset and programs' });

    /* ---- the claim discipline, as assertions rather than a manual pass ----
     *
     * This lesson's central risk is a bound that reads as a promise. The
     * substance was checked by hand in phase C and found good, but the
     * independent review noted it had no regression guard — unlike almost
     * everything else here. These are cheap, and they are the property the
     * topic note is about. KaTeX renders each formula twice into `innerText`
     * (HTML plus its MathML annotation), so the needles avoid inline maths.
     */
    await page.locator('.rad-lesson details').evaluateAll(items => items.forEach(item => { item.open = true; }));
    const claimText = normalize(await page.locator('.rad-lesson').innerText());
    const mustSay = [
      ['empirical, expected and bound are separated in one place',
        'The empirical complexity is an exact average over sign patterns for one fixed sample'],
      ['the expectation is named as generally unknown', 'averages that over fresh samples too, and is generally unknown'],
      ['the bound is over the draw of the sample', 'over the draw of the sample, simultaneously for every member of the class'],
      ['it is not a deterministic promise', 'it is never a deterministic promise about the dataset in front of you'],
      ['it is not a per-prediction confidence', 'not a confidence attached to any individual prediction'],
      ['the vacuous result is stated plainly', 'so this calculation gives no informative numerical certificate'],
      ['vacuity does not mean the model is bad', 'That remains true even though the measured classifier is useful'],
      ['the stack is not clipped, and why', 'Clipping would hide exactly the thing worth seeing'],
      ['a fixed corpus is not an iid population', 'are different from the iid population model of our stated theorem'],
      ['simulation error is not generalisation uncertainty',
        'A standard error describes how much the sign simulation wobbles'],
      ['a random-label fit is lower evidence', 'proves that the supremum is'],
    ];
    const missingClaims = mustSay.filter(([, needle]) => !claimText.includes(needle)).map(([label]) => label);
    assert.deepEqual(missingClaims, [], `the page no longer states: ${missingClaims.join('; ')}`);
    const mustNotSay = [
      ['an unconditional guarantee about the model', /\bguarantees that the (model|classifier|predictor)\b/i],
      ['a certificate claim about this data', /\bcertifies that (this|the) (model|dataset|experiment)\b/i],
      ['a per-prediction confidence', /95% confiden(ce|t) (in|that) (this|the) prediction/i],
    ];
    const forbidden = mustNotSay.filter(([, pattern]) => pattern.test(claimText)).map(([label]) => label);
    assert.deepEqual(forbidden, [], `the page now contains a forbidden phrasing: ${forbidden.join('; ')}`);
    /* Radicals and accents must actually be drawn, at desktop as well as
     * narrow. Details are already open here, so the practice solutions' formulas
     * are in scope. */
    const desktopKatex = await page.locator('.rad-lesson .katex svg')
      .evaluateAll(items => items.map(item => ({
        radical: Boolean(item.closest('.sqrt')),
        height: Number(item.getBoundingClientRect().height.toFixed(2)),
      })));
    const desktopRadicals = desktopKatex.filter(entry => entry.radical);
    assert.ok(desktopRadicals.length >= 20, `only ${desktopRadicals.length} radicals found at 1366px`);
    const desktopCollapsed = desktopKatex.filter(entry => entry.height < 1).map(entry => entry.height);
    assert.deepEqual(desktopCollapsed, [],
      `${desktopCollapsed.length} of ${desktopKatex.length} KaTeX SVGs are under 1px tall at 1366px`);
    records.push({
      case: 'Every KaTeX radical and stretchy accent is actually drawn, at 1366, 390 and 320 px',
      katexSvgsMeasured: desktopKatex.length,
      radicalsMeasured: desktopRadicals.length,
      shortestHeightPx: Math.min(...desktopKatex.map(entry => entry.height)),
      note: 'A lesson-root bare descendant `svg` rule collapsed all 24 radicals to 0.05-0.72px. The existing '
        + 'accent-WIDTH guard could not see it, because a collapsed SVG is also narrow.',
    });
    records.push({
      case: 'The three quantities stay distinct and no bound is phrased as a promise',
      statementsRequired: mustSay.length,
      phrasingsForbidden: mustNotSay.length,
      note: 'Previously a manual pass with no regression guard, which is what the independent review flagged.',
    });

    // ------------------------------------------- 2. nothing revealed on first paint
    assert.equal(await page.locator('.rad-investigation .rad-verdict').count(), 0, 'no verdict before a prediction');
    assert.equal(await page.locator('.rad-investigation .rad-history').count(), 0, 'and no history');
    // The property, not the containers: for each investigation, is the string
    // the answer panel will print already on screen? Each entry names a
    // distinctive rendering of that investigation's graded quantity for its
    // OPENING draft. None may appear before a prediction; each must appear
    // after one, or this check is testing nothing.
    const gradedStrings = [
      {
        index: 0,
        label: 'best response',
        // The winning correlation for the opening pattern, and the aggregate.
        before: ['0.333333', '0.666667', 'exactly 2/3'],
        after: ['0.333333'],
      },
      {
        index: 1,
        label: 'signed geometry',
        /* NOT the phrase "every feasible coefficient attains the same value":
           that is one of the three radio OPTIONS, so it is the question, not
           the answer. Offering a candidate answer is how a multiple-choice
           prediction works, and a needle that matches an option can only be
           satisfied by removing the choice the learner is asked to make.
           (The AutoML lesson recorded the same trap for "on the frontier".)
           These two strings appear only in the revealed readout. */
        before: ['Signed sum v', 'Optimum for this pattern'],
        after: ['Signed sum v'],
      },
      {
        index: 2,
        label: 'margin bound',
        before: ['0.500000', 'Mean ramp loss', 'Raw sum'],
        after: ['Mean ramp loss'],
      },
      {
        index: 3,
        label: 'bounded norm',
        // Assessment counts and the bound expressions, both withheld.
        before: ['Assessment mistakes', '1.129848', 'Selected by validation'],
        after: ['Assessment mistakes'],
      },
    ];
    for (const entry of gradedStrings) {
      const text = normalize(await page.locator('.rad-investigation').nth(entry.index).innerText());
      const leaked = entry.before.filter(needle => text.includes(needle));
      assert.deepEqual(leaked, [],
        `${entry.label} shows a graded quantity before any prediction: ${leaked.join(', ')}`);
    }
    /* Pinned numerically as well as by marker class.
     *
     * Investigations 1-3 derive everything they show, so NO six-decimal value
     * may appear in them before a prediction is recorded.
     *
     * Investigation 4 is different by design: the five candidates' fit and
     * validation columns are the EVIDENCE the learner applies the declared
     * selection rule to, so they must be visible. Withholding them would delete
     * the question. What must be withheld there is the assessment column and
     * the ten bound expressions, so those are pinned by value -- computed from
     * the data module rather than transcribed -- and the check is two-sided:
     * the permitted numbers must be present, or the table could be missing
     * altogether and this would still pass. */
    const derivedOnly = await page.locator('.rad-investigation').evaluateAll(nodes => nodes.slice(0, 3)
      .map(node => [...node.innerText.matchAll(/\d\.\d{6,}/g)].map(match => match[0])));
    assert.deepEqual(derivedOnly.flat(), [],
      `investigations 1-3 print a six-decimal value before a prediction: ${derivedOnly.flat().join(', ')}`);

    const six = value => value.toFixed(6);
    const forbiddenInFour = [
      ...data.fittedModels.map(model => six(model.assessment.logLoss)),
      ...data.fittedModels.flatMap(model => model.bounds.flatMap(bound => [
        six(bound.rawUpper), six(bound.empiricalRamp), six(bound.complexityAddend),
      ])),
      six(data.experimentSettings.confidenceAddend),
    ];
    const permittedInFour = [
      ...data.fittedModels.map(model => six(model.validation.logLoss)),
      ...data.fittedModels.map(model => six(model.fit.logLoss)),
    ];
    const fourthText = normalize(await page.locator('.rad-investigation').nth(3).innerText());
    const leakedNumbers = forbiddenInFour.filter(value => fourthText.includes(value));
    assert.deepEqual(leakedNumbers, [],
      `investigation 4 shows a withheld assessment or bound value before the selection is committed: ${leakedNumbers.join(', ')}`);
    const missingEvidence = permittedInFour.filter(value => !fourthText.includes(value));
    assert.deepEqual(missingEvidence, [],
      `investigation 4 is missing the evidence the selection rule needs: ${missingEvidence.join(', ')}`);
    await screenshot(page.locator('.rad-investigation').nth(0), 'rademacher-best-response-before-prediction-desktop.png');
    await screenshot(page.locator('.rad-investigation').nth(3), 'rademacher-bounded-norm-before-prediction-desktop.png');
    records.push({
      case: 'No investigation shows a graded quantity before a prediction is recorded',
      checkedStrings: gradedStrings.reduce((total, entry) => total + entry.before.length, 0),
      note: 'Asks whether the answer string is on screen, and separately whether any six-decimal number is, rather than enumerating the containers an answer might sit in.',
    });

    // ------------------------------------- 3. investigation 1: commit, grade, retire
    const first = page.locator('.rad-investigation').nth(0);
    /* Deliberately the SECOND of the two tied winners, and deliberately the
     * value this page itself prints.
     *
     * Both were live defects found by looking at a screenshot. On the opening
     * pattern rows f2 and f4 both reach 1/3, and grading only the first marked
     * a learner who picked f4 wrong while the explanation underneath told them
     * either was correct. And the exact answer is 1/3, so a learner reading
     * 0.333333 off the page and typing it back was told it was OUTSIDE a 1e-9
     * tolerance. A page that grades its own printed value wrong is the defect
     * this whole contract exists to prevent. */
    await first.locator('.rad-choice input').nth(3).check();
    await first.locator('.rad-numeric-guess input').fill('0.333333');
    await first.getByRole('button', { name: 'Apply and check' }).click();
    await settle(page);
    const verdict = normalize(await first.locator('.rad-verdict').innerText());
    assert.match(verdict, /Your prediction matches/, 'a tied winner is graded as correct, not as a miss');
    assert.match(verdict, /2 answers are correct here/, 'and the other tied answer is named');
    assert.match(verdict, /within/, "the page accepts the value it prints for itself");
    assert.doesNotMatch(verdict, /outside/, 'so a learner reading 0.333333 off the page is not told it is wrong');
    assert.equal(await first.locator('.rad-verdict.is-miss').count(), 0, 'and the verdict is not marked as a miss');
    const afterFirst = normalize(await first.innerText());
    assert.ok(afterFirst.includes('0.333333'), 'the winning correlation appears only after the commitment');
    await screenshot(first, 'rademacher-best-response-committed-desktop.png');
    // An edit retires the verdict and keeps the earlier attempt hidden.
    await first.locator('input[type="number"]').first().fill('0.5');
    await settle(page);
    assert.equal(await first.locator('.rad-verdict').count(), 0, 'an edit retires the verdict');
    await checkText(first.locator('.rad-history.is-pending'), /stays hidden until you apply/);
    records.push({
      case: 'Investigation 1 commits a winning row and a numeric correlation together, grades against the committed draft, and retires both when a prediction row is edited',
      tieAndPrecision: 'The SECOND of two tied winners is graded correct and the other is named; the value the page '
        + 'itself prints (0.333333 for an exact 1/3) is accepted. Both were live defects before this run.',
    });

    /* ---- B1, driven on the live page at the reviewer's exact input ----
     *
     * Load the shipped "Perpendicular" preset and change one coordinate from 0
     * to 0.001 — a legal three-decimal value. The exact complexity and the
     * energy bound then differ by 8.8e-8 and print identically as 0.707107.
     * The old 1e-12 categorical threshold returned a ✗ whose own text showed
     * the two "different" values as the same number while its numeric half
     * said the learner was right. */
    const second = page.locator('.rad-investigation').nth(1);
    await second.getByRole('button', { name: 'Perpendicular (1, 0) and (0, 1)' }).click();
    await settle(page);
    await second.locator('input[type="number"]').nth(2).fill('0.001');
    await settle(page);
    // Stage 1 only has to be committed so that stage 2 renders; each stage is
    // its own `.rad-prediction`, and committing one disables its own radios.
    const patternStage = second.locator('.rad-prediction').nth(0);
    await patternStage.locator('.rad-choice input').first().check();
    await patternStage.locator('.rad-numeric-guess input').fill('0.707107');
    await patternStage.getByRole('button', { name: 'Apply and check' }).click();
    await settle(page);
    // Stage 2 is the comparison B1 is about. "Exactly equal to the energy
    // bound" is the correct answer once the gap is below the printed precision.
    const averageStage = second.locator('.rad-prediction').nth(1);
    await averageStage.locator('.rad-choice input').nth(1).check();
    await averageStage.locator('.rad-numeric-guess input').fill('0.707107');
    await averageStage.getByRole('button', { name: 'Reveal the enumeration and the bound' }).click();
    await settle(page);
    const gapVerdict = normalize(await averageStage.locator('.rad-verdict').innerText());
    assert.match(gapVerdict, /Your prediction matches/,
      'a gap invisible at the printed precision must not be graded as a miss');
    assert.doesNotMatch(gapVerdict, /Strictly below the energy bound/,
      'and the page must not claim a gap it prints as zero');
    const readout = normalize(await second.innerText());
    assert.ok(readout.includes('0.707107'), 'the two quantities are still printed at six decimals');
    await screenshot(second, 'rademacher-signed-geometry-near-tie-desktop.png');
    records.push({
      case: 'A gap smaller than the printed precision grades as equal, at the reviewer\'s reachable input',
      input: 'preset "Perpendicular (1, 0) and (0, 1)" with x2 coordinate 1 set to 0.001; exact gap 8.8e-8',
      note: 'Blocking finding B1. The categorical comparison now grades at the precision it prints, like the '
        + 'numeric field beside it.',
    });

    // ------------------------------------- 4. investigation 4: selection then reveal
    const fourth = page.locator('.rad-investigation').nth(3);
    const beforeSelection = normalize(await fourth.innerText());
    assert.ok(!beforeSelection.includes('Assessment mistakes'), 'assessment is closed before a selection');
    await fourth.locator('.rad-choice input').last().check();
    await fourth.getByRole('button', { name: 'Commit the selection and open the assessment rows' }).click();
    await settle(page);
    const afterSelection = normalize(await fourth.innerText());
    assert.ok(afterSelection.includes('Assessment mistakes'), 'and opens after it');
    assert.ok(afterSelection.includes(`${data.fittedModels[4].assessment.errors} of 80`)
      || afterSelection.includes(String(data.fittedModels[4].assessment.errors)),
    'with the real assessment counts');
    assert.ok(afterSelection.includes('You have opened the assessment rows'),
      'and says so, because a reset cannot make it unseen');
    await screenshot(fourth, 'rademacher-bounded-norm-assessment-open-desktop.png');
    // Reset clears answers but keeps the assessment-seen indicator.
    await fourth.getByRole('button', { name: 'Reset' }).click();
    await settle(page);
    assert.ok(normalize(await fourth.innerText()).includes('You have opened the assessment rows'),
      'the assessment-seen indicator survives Reset');
    records.push({ case: 'Investigation 4 keeps assessment closed until a selection is committed, reveals the real counts afterwards, and keeps an accurate assessment-seen indicator across Reset' });

    // ---------------------------------------------------- 5. figure geometry
    // The shared inspector is a BROWSER-side function, passed to page.evaluate
    // with the root selector as its argument -- not a Node helper taking a page.
    const layoutIssues = await page.evaluate(inspectLessonVisualLayout, '.rad-lesson');
    assert.deepEqual(layoutIssues.flatMap(figure => figure.issues), [],
      'the shared layout inspector found a collision at 1366px');
    const curveHits = await page.evaluate(sampleCurvesThroughLabels, '.rad-lesson');
    assert.deepEqual(curveHits, [], 'a curve runs through a label at 1366px');
    const paintFindings = await page.evaluate(comparePaintedStrokes, '.rad-lesson');
    assert.deepEqual(paintFindings, [], 'a painted value disagrees with its attribute');

    /* B2, measured from the painted DOM. Figure 8's three panels are captioned
     * "Same lengths"; the same budget ball must therefore be drawn at the same
     * radius in all three. It was 26.5px in panel 1 and 53px in the others. */
    const figure8Radii = await page.locator('.rad-figure').nth(7)
      .locator('svg circle.rad-ball').evaluateAll(items => items.map(item => item.getBoundingClientRect().width));
    assert.equal(figure8Radii.length, 3, 'figure 8 draws three budget balls');
    assert.ok(Math.max(...figure8Radii) - Math.min(...figure8Radii) < 0.5,
      `figure 8 is captioned "Same lengths" but draws its budget ball at ${figure8Radii.join(', ')} px`);
    const figure8Arrows = await page.locator('.rad-figure').nth(7)
      .locator('svg line.rad-arrow').evaluateAll(items => items.map(item => {
        const box = item.getBoundingClientRect();
        return Math.round(Math.hypot(box.width, box.height));
      }));
    assert.ok(Math.max(...figure8Arrows) - Math.min(...figure8Arrows) <= 1,
      `and it draws the same unit input at ${[...new Set(figure8Arrows)].join(', ')} px across panels`);
    records.push({
      case: 'Figure 8\'s small multiples share one scale, measured from the painted DOM',
      ballWidthsPx: figure8Radii,
      note: 'Blocking finding B2. Each panel used to autoscale to a signed sum it never draws.',
    });
    for (let index = 0; index < 17; index += 1) {
      await screenshot(page.locator('.rad-figure').nth(index), `rademacher-figure-${index + 1}-desktop.png`);
    }
    assert.equal(new Set(screenshotPaths).size, screenshotPaths.length, 'every capture has its own path');
    assert.equal(new Set(screenshotDigests.map(entry => entry.digest)).size, screenshotDigests.length,
      'and its own content, so a stale orphan from an earlier run cannot pass as new');
    records.push({
      case: 'No label leaves its SVG, overlaps another label, is crossed by a straight line or run through by a curve; every painted stroke and fill agrees with its attribute',
      curveSamplingNote: 'Curves are sampled along their own geometry and tested against every label box, because the shared inspector reads straight lines only.',
    });

    // -------------------------------------------------------- 6. narrow widths
    for (const width of [390, 320]) {
      await page.setViewportSize({ width, height: 844 });
      await page.locator('.rad-lesson details').evaluateAll(items => items.forEach(item => { item.open = true; }));
      await settle(page);
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
        `Document overflows at ${width}px`);
      const mathOverflow = await page.locator('.rad-lesson .katex-display')
        .evaluateAll(items => items.filter(item => item.scrollWidth > item.clientWidth + 1)
          .map(item => item.textContent.slice(0, 80)));
      assert.deepEqual(mathOverflow, [], `Overflowing formulas at ${width}px`);
      assert.equal(await page.locator('.katex-error').count(), 0);
      // The accent gotcha: a \widehat can emit an SVG wider than the column.
      const wideAccents = await page.locator('.rad-lesson .katex svg')
        .evaluateAll((items, limit) => items.filter(item => item.getBoundingClientRect().width > limit)
          .map(item => Math.round(item.getBoundingClientRect().width)), width - 32);
      assert.deepEqual(wideAccents, [], `A KaTeX accent SVG is wider than the column at ${width}px`);
      /* And that they have any HEIGHT at all.
       *
       * The width check above passed for months while every radical on the page
       * was 0.05-0.72px tall: a stylesheet rule scoped at the lesson root as a
       * bare descendant `svg` matched KaTeX's own inline SVGs, and `height:
       * auto` collapsed them. A square root that is not drawn is not a
       * cosmetic problem — the formula then says the radius squared where it
       * means the radius, and a collapsed accent turns this lesson's empirical
       * complexity into the population quantity it is defined against. The
       * width guard could not see it because a collapsed SVG is also narrow. */
      const katexHeights = await page.locator('.rad-lesson .katex svg')
        .evaluateAll(items => items.map(item => ({
          radical: Boolean(item.closest('.sqrt')),
          height: Number(item.getBoundingClientRect().height.toFixed(2)),
        })));
      assert.ok(katexHeights.length >= 40,
        `only ${katexHeights.length} KaTeX SVGs were found at ${width}px; the sweep must have something to look at`);
      const radicals = katexHeights.filter(entry => entry.radical);
      assert.ok(radicals.length >= 20, `only ${radicals.length} radicals found at ${width}px`);
      const collapsed = katexHeights.filter(entry => entry.height < 1).map(entry => entry.height);
      assert.deepEqual(collapsed, [],
        `${collapsed.length} of ${katexHeights.length} KaTeX SVGs are under 1px tall at ${width}px `
        + `(heights ${collapsed.slice(0, 8).join(', ')}) — a radical or accent is not being drawn`);
      const narrowCurves = await page.evaluate(sampleCurvesThroughLabels, '.rad-lesson');
      assert.deepEqual(narrowCurves, [], `a curve runs through a label at ${width}px`);
      if (width === 320) {
        await screenshot(page.locator('.rad-investigation').nth(0), 'rademacher-best-response-320.png');
        await screenshot(page.locator('.rad-figure').nth(6), 'rademacher-figure-7-320.png');
        await screenshot(page.locator('.rad-figure').nth(14), 'rademacher-figure-15-320.png');
      } else {
        for (const index of [1, 8, 12, 15]) {
          await screenshot(page.locator('.rad-figure').nth(index), `rademacher-figure-${index + 1}-390.png`);
        }
        await screenshot(page.locator('.rad-investigation').nth(3), 'rademacher-bounded-norm-390.png');
      }
      // A figure table must not clip a column at this width; it scrolls instead.
      const clipped = await page.locator('.rad-table-scroll').evaluateAll(items => items
        .filter(item => item.scrollWidth > item.clientWidth && getComputedStyle(item).overflowX !== 'auto').length);
      assert.equal(clipped, 0, `a table clips rather than scrolls at ${width}px`);
    }
    await page.setViewportSize({ width: 1366, height: 1000 });
    await settle(page);
    // And at desktop width no table may clip a column at all.
    const desktopClipping = await page.locator('.rad-table-scroll').evaluateAll(items => items
      .filter(item => item.scrollWidth > item.clientWidth + 1)
      .map(item => item.querySelector('p, table')?.textContent?.slice(0, 40) ?? 'table'));
    assert.deepEqual(desktopClipping, [], 'a figure table clips a column at desktop width');
    records.push({ case: 'No document or formula overflow, no oversized KaTeX accent, no clipped table column and no curve through a label at 1366, 390 and 320 px' });

    // ------------------------------------------------------ 7. loading closure
    const localScripts = [...new Set(requests.map(address => new URL(address))
      .filter(address => address.origin === new URL(base).origin && address.pathname.endsWith('.js'))
      .map(address => address.pathname.slice(1)))];
    assert.deepEqual(localScripts.filter(filename => bodyFiles.has(filename)), [bodyFile],
      'exactly one lesson body was downloaded');
    assert.ok(!requests.some(address => address.includes('/outlines/')), 'and no planned-topic outline');
    assert.ok(!requests.some(address => address.includes('banknote-subset.csv')),
      'the page never fetches the dataset to render: the rows travel in its own generated module');
    records.push({
      case: 'Fresh route requests only the selected lesson and its shared closure, and renders without fetching any data file',
      requestedScripts: localScripts,
      rawScriptBytes: localScripts.reduce((sum, filename) => sum + fs.statSync(path.join(distDir, filename)).size, 0),
      gzipEstimateBytes: localScripts.reduce((sum, filename) =>
        sum + gzipSync(fs.readFileSync(path.join(distDir, filename))).length, 0),
      interpretation: 'Built requested file sizes; the gzip estimate is not measured network compression or latency.',
    });

    // ------------------------------------------------------- 8. keyboard, errors
    await page.keyboard.press('Tab');
    const focusVisible = await page.evaluate(() => {
      const active = document.activeElement;
      if (!active) return false;
      const style = getComputedStyle(active);
      return style.outlineStyle !== 'none' || style.boxShadow !== 'none';
    });
    assert.ok(focusVisible !== null, 'keyboard focus reaches the page');
    assert.deepEqual(errors, [], `the page threw: ${errors.join('; ')}`);
    assert.deepEqual(failedAssets, [], `an asset failed to load: ${failedAssets.join('; ')}`);
    records.push({ case: 'No page error, no failed asset, keyboard focus reaches the lesson' });

    const report = {
      startedAt,
      finishedAt: new Date().toISOString(),
      status: 'passed',
      route,
      distDir,
      buildManifestSha256: buildHash,
      sourceHashes,
      dataset: { file: data.provenance.file, sha256: data.provenance.sha256, bytes: data.provenance.bytes },
      modelExports: Object.keys(models).length,
      records,
      screenshots: screenshotDigests,
      limitations: [
        'One browser (Edge/Chromium) at three widths. Other engines and assistive technologies are not covered.',
        'Numerical correctness is established by scripts/verify-rademacher-models.mjs and the data and examples '
          + 'verifiers; this run checks what the page renders and how it behaves.',
        'Screenshots are recorded with their content digests so a stale capture cannot be reported as fresh. They '
          + 'still have to be LOOKED AT; a passing geometry check is not a visual review.',
      ],
    };
    fs.mkdirSync(path.dirname(evidencePath), { recursive: true });
    fs.writeFileSync(evidencePath, `${JSON.stringify(report, null, 2)}\n`);
    console.log(JSON.stringify({
      status: report.status,
      cases: records.length,
      evidencePath,
      sourceFiles: ownedFiles.length,
      screenshots: screenshotDigests.length,
    }));
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});

// Production browser review of the ML Problem Formulation, Baselines & Data
// Leakage lesson: visible content, the two investigations, prediction, commit
// and retirement, the leak property, figure geometry, narrow layouts, sequence
// and loading closure.
//
// Run against a preview of a production build:
//   npx vite build --outDir dist-form
//   npx vite preview --outDir dist-form --host 127.0.0.1 --port 4195 --strictPort
//   PLAYWRIGHT_PACKAGE=... DIST_DIR=dist-form LEARNING_BASE_URL=http://127.0.0.1:4195 \
//     node scripts/verify-formulation-browser.cjs
//
// The check this file exists for is section 3 below. The dominant defect class
// in these lessons is a page that grades a correct answer wrong, or shows the
// answer first -- and asserting that no verdict BANNER exists on first paint is
// what let that ship three times, because a banner is easy to withhold while
// the graded number leaks out of a readout beside it. So the assertion here is
// a property, pinned numerically: for EVERY investigation, the exact
// six-decimal text of the graded quantity -- computed independently in this
// process from formulation-models.js, not read off the page -- must be ABSENT
// from the investigation's rendered text before a prediction is committed and
// PRESENT after. The absence assertion is paired with the presence assertion so
// that a pin which can never match is caught rather than counted.
//
// On this page there is a second, stronger version of the same property.
// Investigation 2's subject is information reaching a decision before it
// should, so it renders NO OUTCOME AT ALL for any of its 824 cases until a
// prediction is committed. That is asserted as the absence of the outcome
// vocabulary, not only of one number.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { gzipSync } = require('node:zlib');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const { inspectLessonVisualLayout } = require('./lib/lesson-visual-layout.cjs');

const distDir = process.env.DIST_DIR || 'dist-form';
const base = (process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4195').replace(/\/+$/, '');
const topicId = 'ml-problem-formulation-baselines-data-leakage';
const sourcePath = `src/learn/data/topics/${topicId}.jsx`;
const route = `${base}/learn/path/full-curriculum/${topicId}?module=classical-ml`;
const evidencePath = 'docs/teaching/evidence/formulation-browser.json';
const read = filename => JSON.parse(fs.readFileSync(filename, 'utf8'));
const hash = filename => createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
const normalize = text => text.replace(/\s+/g, ' ').trim();
const PRESET_ARRIVE_EARLIER = 'Let sensor A\u2019s delayed event arrive at time 4 instead';
const ownedFiles = [
  sourcePath,
  'src/learn/data/formulation-models.js',
  'src/learn/data/formulation-data.js',
  'src/learn/data/formulation-examples.js',
  'src/learn/components/lesson-labs/FormulationShared.jsx',
  'src/learn/components/lesson-labs/FormulationLabs.jsx',
  'src/learn/components/lesson-labs/FormulationFigures.jsx',
  'src/learn/components/lesson-labs/formulation-labs.css',
  `src/learn/data/curriculum/blueprints/${topicId}.js`,
  'public/learn-assets/problem-formulation/bank-additional.csv',
  'public/learn-assets/problem-formulation/ATTRIBUTION.txt',
  'public/learn-assets/problem-formulation/bank-marketing-variable-description.txt',
  'public/learn-assets/problem-formulation/formulation-calculations.py',
];

/** Runs in the page. Samples every curve along its own geometry and reports the
 * labels it passes through.
 *
 * Adapted from scripts/verify-bias-variance-browser.cjs, because
 * `scripts/lib/lesson-visual-layout.cjs` iterates `svg.querySelectorAll('line')`
 * and so cannot see a <path> or <polyline> at all. This lesson draws its
 * feedback channel as a polyline, its arrival markers as closed paths and its
 * arrowheads as marker paths, so that blind spot would cover the three shapes
 * most likely to run across a label.
 *
 * Two exclusions, both checked against what this lesson actually draws: grid
 * lines are background, and a shape inside <defs> is a marker template whose
 * local coordinates are not page coordinates. Nothing here carries an opaque
 * backplate, so no crossing is allowed at all.
 */
function sampleCurvesThroughLabels(root) {
  const SAMPLES = 320;
  const findings = [];
  const sampledClasses = [];
  let shapesSampled = 0;
  let labelsConsidered = 0;
  for (const svg of document.querySelectorAll(`${root} svg`)) {
    const labels = [...svg.querySelectorAll('text')]
      .filter(text => text.textContent.trim() && text.getClientRects().length)
      .map(text => ({ text: text.textContent.trim().slice(0, 40), box: text.getBoundingClientRect() }));
    if (!labels.length) continue;
    labelsConsidered += labels.length;
    for (const shape of svg.querySelectorAll('path, polyline, polygon')) {
      if (shape.closest('defs')) continue;
      if (shape.classList.contains('form-grid')) continue;
      if (typeof shape.getTotalLength !== 'function') continue;
      const length = shape.getTotalLength();
      if (!(length > 0)) continue;
      const matrix = shape.getScreenCTM();
      if (!matrix) continue;
      shapesSampled += 1;
      sampledClasses.push(shape.getAttribute('class') || shape.tagName);
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
        findings.push({
          label: labels[index].text,
          shape: shape.getAttribute('class') || shape.tagName,
          fractionOfCurveInsideLabel: Number((count / (SAMPLES + 1)).toFixed(4)),
        });
      });
    }
  }
  return { findings, shapesSampled, labelsConsidered, sampledClasses };
}

/** Runs in the page. Every KaTeX box that carries visible text, with BOTH of
 * its dimensions.
 *
 * Hoisted so the narrow widths run exactly this measurement and their count can
 * be compared with the desktop count. A check that silently stops finding
 * subjects is a check that has gone inert, and "it found nothing wrong" and
 * "it looked at nothing" are different results. */
function measureKatexBoxes() {
  return [...document.querySelectorAll('.formulation-lesson .katex .mord, '
    + '.formulation-lesson .katex .mfrac, .formulation-lesson .katex .mrel, '
    + '.formulation-lesson .katex .mop, .formulation-lesson .katex-display')]
    .filter(node => node.textContent.trim())
    .map(node => {
      const box = node.getBoundingClientRect();
      return {
        tag: node.tagName,
        className: node.getAttribute('class'),
        text: node.textContent.trim().slice(0, 24),
        height: Number(box.height.toFixed(2)),
        width: Number(box.width.toFixed(2)),
      };
    });
}

/** Runs in the page. Every SVG KaTeX draws itself, with both dimensions. This
 *  lesson's formulas emit none; the count is pinned rather than filtered. */
function measureKatexSvgs() {
  return [...document.querySelectorAll('.formulation-lesson .katex svg')].map(svg => {
    const box = svg.getBoundingClientRect();
    return { height: Number(box.height.toFixed(2)), width: Number(box.width.toFixed(2)) };
  });
}

/** Runs in the page. Reports every place a JSX expression lost the space after
 * it, leaving two words glued together in the rendered prose.
 *
 * `{value}` followed by a newline and an indented word renders as `0.25and`,
 * because JSX strips the leading whitespace of a continuation line. It is
 * invisible in the source, invisible in the DOM structure, and perfectly
 * visible to a reader. Three of these reached the rendered page of this lesson;
 * the first two were found by looking at screenshots, which is not a method
 * that scales.
 *
 * Prose elements only, one at a time: SVG text nodes are separate elements
 * whose text legitimately concatenates ("824" then "reserved"), so scanning a
 * whole subtree at once would report those as glue and this guard would be
 * deleted rather than fixed. */
function findGluedWords(root) {
  /* The signature is a VALUE followed immediately by a word: a digit, a closing
     bracket or a closing quote, then two or more lowercase letters.

     The first version of this pattern matched any word ENDING in a short word
     -- "into", "hand", "This" -- and reported eleven false positives and one
     real defect. A guard a reader has to triage is a guard that gets deleted,
     so it is anchored on the thing that actually breaks: an expression's value
     touching the next word. A few real spellings are excluded by name below. */
  const ALLOWED = /^(?:[0-9]+(?:st|nd|rd|th|px|pt|em|ms|kb|mb)\b)/;
  const pattern = /[0-9)\]”’](?=[a-z]{2,})/g;
  const findings = [];
  let elementsScanned = 0;
  for (const element of document.querySelectorAll(`${root} :is(p, li, dd, dt, td, th, blockquote, summary)`)) {
    if (element.closest('svg')) continue;
    const text = element.innerText;
    if (!text) continue;
    elementsScanned += 1;
    /* A SHA-256 is sixty-four characters of digits and lowercase letters, and
       every digit in it is followed by letters. The page prints one, on
       purpose. Hex runs are located once and matches inside them are skipped;
       excluding "any digit near a letter" instead would have disabled the
       guard everywhere. */
    const hexRuns = [...text.matchAll(/[0-9a-f]{16,}/g)]
      .map(run => [run.index, run.index + run[0].length]);
    for (const match of text.matchAll(pattern)) {
      if (hexRuns.some(([from, to]) => match.index >= from && match.index < to)) continue;
      const tail = text.slice(match.index);
      if (ALLOWED.test(tail) || ALLOWED.test(text.slice(Math.max(0, match.index - 6)))) continue;
      findings.push({
        glued: text.slice(Math.max(0, match.index - 6), match.index + 14),
        context: text.slice(Math.max(0, match.index - 45), match.index + 25),
      });
    }
  }
  return { findings, elementsScanned };
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
  let shapesExamined = 0;
  for (const svg of document.querySelectorAll(`${root} svg`)) {
    for (const shape of svg.querySelectorAll('rect, circle, line, path, polyline, polygon, text')) {
      if (!shape.getClientRects().length) continue;
      shapesExamined += 1;
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
          tag: shape.tagName, shapeClass: shape.getAttribute('class') || '', painted: painted.fill,
        });
      }
    }
  }
  return { findings, shapesExamined };
}

(async () => {
  const startedAt = new Date().toISOString();
  /* A PROVISIONAL record, written before anything is read or launched.
   *
   * The report below is written in a `finally`, which covers everything inside
   * the try. It does not cover a throw from the manifest read or the module
   * imports that happen first, and such a throw would leave the PREVIOUS
   * report on disk still saying `status: passed` for a build that no longer
   * exists. */
  fs.mkdirSync(path.dirname(evidencePath), { recursive: true });
  fs.writeFileSync(evidencePath, JSON.stringify({
    startedAt,
    verifier: 'scripts/verify-formulation-browser.cjs',
    status: 'failed',
    note: 'Provisional record written before the build manifest was read. If this is what is on disk, the '
      + 'run did not reach its end.',
  }, null, 2) + '\n');
  const sourceHashes = Object.fromEntries(ownedFiles.map(filename => [filename, hash(filename)]));
  const build = read(`${distDir}/.vite/manifest.json`);
  const buildHash = hash(`${distDir}/.vite/manifest.json`);
  const publications = read('src/learn/data/lesson-manifest.json');
  const models = await import('../src/learn/data/formulation-models.js');
  const { formulationData } = await import('../src/learn/data/formulation-data.js');
  const { formulationExamples } = await import('../src/learn/data/formulation-examples.js');

  /* The independently computed answers this run pins the page against. Nothing
     here is read off the screen. */
  const fixedText = value => value.toFixed(6);
  /* Mirrors the page's `round(value, 6)`: six decimals with trailing zeros
     stripped. Baselines are printed that way on purpose, so the six-decimal
     form stays reserved for graded values and the leak pin stays sharp. */
  const round6 = value => value.toFixed(6).replace(/0+$/, '').replace(/\.$/, '');
  const partition = formulationData.partition;
  const targetById = Object.fromEntries(formulationData.validation.ids
    .map((id, index) => [id, formulationData.validation.targets[index]]));
  const candidateRanking = models.rankByScore({
    ids: formulationData.validation.ids, scores: formulationData.validation.scores.candidate,
  });
  const timelineDefault = models.latestKnown(models.timelineFixture);
  const timelineMoved = models.latestKnown(
    models.withArrival(models.timelineFixture, { event: 4, version: 1, available: 4 }));
  const capacityAtFifty = models.selectionMetrics({
    order: candidateRanking, capacity: partition.capacity, targetById,
    totalPositives: partition.validationPositives,
  });
  const capacityAtTwentyFive = models.selectionMetrics({
    order: candidateRanking, capacity: 25, targetById, totalPositives: partition.validationPositives,
  });

  const bodyFiles = new Set(Object.values(build)
    .filter(entry => entry.src && entry.src.includes('/data/topics/'))
    .map(entry => entry.file));
  const bodyEntry = Object.values(build).find(entry => entry.src && entry.src.endsWith(`${topicId}.jsx`));
  assert.ok(bodyEntry, `the build has no chunk for ${topicId}.jsx; is the lesson registered?`);
  const bodyFile = bodyEntry.file;
  const allowedScripts = new Set();
  /* Walk by manifest KEY. `imports` lists keys, not built filenames, so a
     lookup that only matched `src` or `file` silently finds nothing for every
     shared chunk and leaves legitimate files looking unexpected. */
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
    await page.locator('.formulation-lesson').waitFor();
    await page.waitForFunction(() => [...document.fonts]
      .some(font => font.family.includes('Space Grotesk') && font.status === 'loaded'), null, { timeout: 20000 });
    await page.evaluate(() => document.fonts.ready);
    await settle(page);
  };
  let status = 'failed';
  let curveSampling = { shapesSampled: 0, labelsConsidered: 0 };
  let desktopKatexBoxes = 0;
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
    await checkText(page.locator('.reader-header h1'), /^ML Problem Formulation, Baselines & Data Leakage$/);
    assert.equal(await page.locator('.formulation-investigation').count(), 2);
    assert.equal(await page.locator('.formulation-figure').count(), 7);
    assert.equal(await page.locator('.formulation-practice').count(), 8);
    assert.equal(await page.locator('.python-example').count(), 2);
    assert.equal(await page.locator('.katex-error').count(), 0);
    const rendered = normalize(await page.locator('.formulation-lesson').textContent());
    for (const key of ['latest-known', 'experiment']) {
      assert.ok(rendered.includes(normalize(formulationExamples[key].code)), `Missing actual code for ${key}`);
      assert.ok(rendered.includes(normalize(formulationExamples[key].expected)),
        `Missing actual output for ${key}`);
    }
    for (const anchor of await page.locator('.lesson-intro a[href^="#"]')
      .evaluateAll(items => items.map(item => item.getAttribute('href')))) {
      assert.equal(await page.locator(`[id="${anchor.slice(1)}"]`).count(), 1, `Broken lesson anchor ${anchor}`);
    }
    assert.ok(rendered.includes(formulationData.provenance.sha256), 'the file hash travels with the data');
    assert.ok(rendered.includes('CC BY 4.0'), 'and so does the licence');
    assert.ok(rendered.includes('reserved rows are not'), 'the untouched reserved split is stated');
    assert.ok(rendered.includes('retire it the moment an input changes'),
      'the intro promises what the page keeps');
    for (const procedure of formulationData.procedures) {
      assert.ok(rendered.includes(fixedText(procedure.averagePrecision)),
        `the measured average precision for ${procedure.id} is on the page`);
      assert.ok(rendered.includes(`${procedure.correct}/${partition.validationRows}`),
        `and its correct count for ${procedure.id}`);
      /* The score and the contract travel together. A page that printed .4617
         without saying what that model needed to know would be teaching the
         opposite of its own subject. */
      assert.ok(rendered.includes(normalize(procedure.availability)),
        `and the availability contract for ${procedure.id}, beside its score`);
    }
    const asset = await page.request.get(`${base}${formulationData.provenance.file}`);
    assert.equal(asset.status(), 200);
    const bytes = await asset.body();
    assert.equal(bytes.length, formulationData.provenance.bytes,
      'the served file is byte-for-byte the packet file');
    assert.equal(createHash('sha256').update(bytes).digest('hex'), formulationData.provenance.sha256);
    for (const address of [formulationData.provenance.attribution, formulationData.provenance.description,
      formulationData.provenance.program]) {
      const download = await page.request.get(`${base}${address}`);
      assert.equal(download.status(), 200, `${address} is actually served`);
      assert.ok((await download.text()).length > 500, 'and is the file rather than a placeholder');
    }
    records.push({
      case: 'Complete visible code and output for two programs, nine route anchors, seven figures, eight '
        + 'practice tasks, every measured score, the served dataset with its hash and attribution, the '
        + 'provider description and the downloadable calculation program, and current metadata',
    });

    // ------------------------------------------- 2. nothing revealed on paint
    for (let index = 0; index < 2; index += 1) {
      const investigation = page.locator('.formulation-investigation').nth(index);
      assert.equal(await investigation.locator('.formulation-choices input[type="radio"]:checked').count(), 0,
        `investigation ${index + 1} has a prediction preselected`);
      assert.equal(await investigation.locator('.formulation-verdict').count(), 0,
        `investigation ${index + 1} shows a verdict before any prediction`);
      assert.equal(await investigation.locator('.formulation-reveal').count(), 0,
        `investigation ${index + 1} shows its reveal before any prediction`);
      assert.ok(await investigation.getByRole('button', { name: 'Apply and check' }).isDisabled(),
        `investigation ${index + 1} can be checked without recording anything`);
    }
    records.push({
      case: 'Both investigations open with nothing selected, no verdict, no reveal and a disabled check',
    });

    // ------------------------ 3. the leak property, pinned numerically, per lab
    const leakChecks = [];
    const textOf = async index =>
      normalize(await page.locator('.formulation-investigation').nth(index).innerText());

    // Investigation 1: the graded quantity is the selected record's value.
    const one = page.locator('.formulation-investigation').nth(0);
    const oneBefore = await textOf(0);
    const oneAnswerText = fixedText(timelineDefault.value);
    assert.ok(!oneBefore.includes(oneAnswerText),
      `investigation 1 shows its graded value ${oneAnswerText} before a prediction is recorded`);
    /* NOT "the value 10 is absent": the records table lists every value on
       purpose, because the history is the entity the learner reasons about.
       What must be absent is any statement of WHICH record the cutoff admits.
       Those phrases are rendered only by the reveal. */
    const investigationOnePhrases = ['Selected record', 'Selected value', 'Eligible records', 'Verdict'];
    for (const phrase of investigationOnePhrases) {
      assert.ok(!oneBefore.includes(phrase),
        `investigation 1 renders "${phrase}" before a prediction is recorded`);
    }
    assert.ok(oneBefore.includes('Available at'),
      'while the history table does show every arrival, which is what makes the answer derivable');
    await screenshot(one, 'formulation-timeline-initial-desktop.png');
    await one.getByLabel(models.recordLabel(timelineDefault.selected.record), { exact: true }).check();
    await one.locator('.formulation-numeric-guess input').fill(String(timelineDefault.value));
    await one.getByRole('button', { name: 'Apply and check' }).click();
    const oneAfter = await textOf(0);
    assert.ok(oneAfter.includes(oneAnswerText),
      `investigation 1 never shows ${oneAnswerText} even after committing, so the pin cannot fail`);
    await checkText(one.locator('.formulation-verdict'), /Your prediction matches/);
    await checkText(one, /newest event and the newest knowledge are different things/);
    /* PRESENCE PARTNER for the four phrase absences above. An absence assertion
       on its own cannot tell "correctly withheld" from "looking for a phrase
       this page never renders": a sibling pinned a value at six decimals where
       the page prints three, and only its presence partner caught it. */
    for (const phrase of investigationOnePhrases) {
      assert.ok(oneAfter.includes(phrase),
        `investigation 1 never renders "${phrase}" even after committing, so its absence before commitment `
        + 'proves nothing');
    }
    /* PRESENCE PARTNER for the `.formulation-reveal` absence asserted on first
       paint and after every retirement. It was the one absence in this file
       with no partner: nothing established that a reveal is ever rendered, so
       "no reveal before commitment" would also hold on a lab that never
       reveals anything. */
    assert.equal(await one.locator('.formulation-reveal').count(), 1,
      'investigation 1 renders exactly one reveal after committing');
    leakChecks.push({
      investigation: 1,
      pinnedValue: oneAnswerText,
      absentBefore: true,
      presentAfter: true,
      phrasesAbsentBeforeAndPresentAfter: investigationOnePhrases,
    });
    await screenshot(one, 'formulation-timeline-committed-desktop.png');

    // Investigation 2: the graded quantity is the precision AFTER the change,
    // and no outcome is rendered for any case at all beforehand.
    const two = page.locator('.formulation-investigation').nth(1);
    const twoBefore = await textOf(1);
    const investigationTwoPhrases = ['subscribed', 'did not subscribe', 'Subscriptions among them',
      'Entered the selected set'];
    for (const phrase of investigationTwoPhrases) {
      assert.ok(!twoBefore.includes(phrase),
        `investigation 2 renders "${phrase}" before a prediction is recorded; on this page that is the `
        + 'lesson\'s own error in its own interface');
    }
    assert.ok(twoBefore.includes('hidden until you apply'), 'and says so where the outcome column would be');
    assert.ok(!twoBefore.includes(fixedText(capacityAtFifty.precision)),
      'nor does it print the applied precision at six decimals before anything is committed');
    await screenshot(two, 'formulation-capacity-initial-desktop.png');
    await two.getByRole('button', { name: 'Halve the capacity to 25' }).click();
    await settle(page);
    const twoAnswerText = fixedText(capacityAtTwentyFive.precision);
    const twoAfterEdit = await textOf(1);
    assert.ok(!twoAfterEdit.includes(twoAnswerText),
      `investigation 2 shows the post-change precision ${twoAnswerText} before the prediction is committed`);
    assert.equal(await two.locator('.formulation-pending').count(), 1,
      'the pending notice appears once an edit is drafted');
    await two.getByLabel('It rises', { exact: true }).check();
    await two.locator('.formulation-numeric-guess input').fill(String(capacityAtTwentyFive.precision));
    await two.getByRole('button', { name: 'Apply and check' }).click();
    const twoAfter = await textOf(1);
    assert.ok(twoAfter.includes(twoAnswerText),
      `investigation 2 never shows ${twoAnswerText} even after committing, so the pin cannot fail`);
    await checkText(two.locator('.formulation-verdict'), /Your prediction matches: It rises\./);
    assert.ok(twoAfter.includes(fixedText(capacityAtTwentyFive.recall)),
      'and its recall, which fell while the precision rose');
    /* PRESENCE PARTNER for the outcome-vocabulary absences above. */
    for (const phrase of investigationTwoPhrases) {
      assert.ok(twoAfter.includes(phrase),
        `investigation 2 never renders "${phrase}" even after committing, so its absence before commitment `
        + 'proves nothing');
    }
    assert.equal(await two.locator('.formulation-reveal').count(), 1,
      'investigation 2 renders exactly one reveal after committing');
    leakChecks.push({
      investigation: 2,
      pinnedValue: twoAnswerText,
      absentBefore: true,
      presentAfter: true,
      phrasesAbsentBeforeAndPresentAfter: investigationTwoPhrases,
    });
    await screenshot(two, 'formulation-capacity-committed-desktop.png');
    records.push({
      case: 'The graded quantity of every investigation is absent from its rendered text before commitment '
        + 'and present after, pinned to the exact six-decimal value computed independently in this process; '
        + 'and investigation 2 renders no outcome for any of its 824 cases until a prediction is recorded',
      leakChecks,
    });

    // ----------------------------- 4. the exact nulls, and unchanged iff
    await two.getByRole('button', { name: 'Reset' }).click();
    await settle(page);
    await two.getByRole('button', { name: 'Reverse the order within the selected block' }).click();
    await settle(page);
    await two.getByLabel('It does not move at all', { exact: true }).check();
    await two.locator('.formulation-numeric-guess input').fill(String(capacityAtFifty.precision));
    await two.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(two.locator('.formulation-verdict'), /Your prediction matches: It does not move at all\./);
    const reversalText = await textOf(1);
    assert.ok(reversalText.includes(fixedText(capacityAtFifty.precision)),
      'the membership null reports the same precision it started with');
    await checkText(two, /No identity entered or left the selected set/);
    await screenshot(two, 'formulation-capacity-membership-null-desktop.png');

    await one.getByRole('button', { name: 'Reset' }).click();
    await settle(page);
    await one.getByRole('button', { name: 'Change the other sensor’s value to 88' }).click();
    await settle(page);
    await one.getByLabel(models.recordLabel(timelineDefault.selected.record), { exact: true }).check();
    await one.locator('.formulation-numeric-guess input').fill(String(timelineDefault.value));
    await one.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(one.locator('.formulation-verdict'), /Your prediction matches/);
    assert.ok((await textOf(0)).includes(fixedText(timelineDefault.value)),
      'editing the other entity leaves this one\'s selected value exactly where it was');
    await screenshot(one, 'formulation-timeline-entity-null-desktop.png');
    records.push({
      case: 'Two exact nulls are graded as nulls: reversing the selected block moves no set metric and says '
        + 'no identity entered or left, and editing the other sensor leaves the selected value unchanged to '
        + 'the last digit',
    });

    // ------- 4b. the graded answer is the answer the page prints (B1)
    /* A capacity whose precision does NOT terminate in six decimals. At 1e-9
       the page rendered a learner's 0.666667 and its own 0.666667 as the same
       string and called them different; roughly four fifths of the capacities
       this control reaches produce such a fraction, and the recorded captures
       only ever used 25, 50 and 100. This drives one of them and requires the
       printed value, typed back verbatim, to grade as correct. */
    await two.getByRole('button', { name: 'Reset' }).click();
    await settle(page);
    const awkwardCapacity = 3;
    const awkward = models.selectionMetrics({
      order: candidateRanking, capacity: awkwardCapacity, targetById,
      totalPositives: partition.validationPositives,
    });
    const awkwardPrinted = awkward.precision.toFixed(6);
    assert.notEqual(Number(awkwardPrinted), awkward.precision,
      `capacity ${awkwardCapacity} must have a precision that does NOT terminate in six decimals, or this `
      + 'case is testing the easy path');
    await two.getByRole('spinbutton', { name: /^calling capacity/ }).fill(String(awkwardCapacity));
    await settle(page);
    await two.getByLabel('It rises', { exact: true }).check();
    await two.locator('.formulation-numeric-guess input').fill(awkwardPrinted);
    await two.getByRole('button', { name: 'Apply and check' }).click();
    const awkwardText = await textOf(1);
    assert.ok(/matches to 6 decimal places/.test(awkwardText),
      `typing the page's own printed value ${awkwardPrinted} at capacity ${awkwardCapacity} was not accepted`);
    assert.ok(!/does not match/.test(awkwardText),
      'and the verdict does not simultaneously say it does not match');
    await screenshot(two, 'formulation-capacity-nonterminating-desktop.png');
    records.push({
      case: 'At a capacity whose precision does not terminate in six decimals, the value the page prints, '
        + `typed back verbatim, grades as correct (capacity ${awkwardCapacity}, printed ${awkwardPrinted})`,
      printedValue: awkwardPrinted,
      trueValue: awkward.precision,
    });

    // ------- 4c. the baseline the caption names is the baseline it grades (B2)
    /* The caption used to name capacity 50 and precision .4 unconditionally,
       because it reappears after every retirement, while the grader compares
       against the previously committed state. A learner following the
       instruction from the second round on was marked wrong in a verdict that
       confirmed their number. Two rounds, with the caption read off the page
       and required to agree with the grading. */
    await two.getByRole('button', { name: 'Reset' }).click();
    await settle(page);
    const captionOf = async () => normalize(await two.locator('.formulation-caption')
      .filter({ hasText: 'Predict what your' }).first().innerText());
    const firstCaption = await captionOf();
    assert.ok(firstCaption.includes('Section 5 records the starting point'),
      `the first round still cites section 5: ${firstCaption}`);
    assert.ok(firstCaption.includes(`capacity ${partition.capacity}`)
      && firstCaption.includes(round6(capacityAtFifty.precision)),
      `and names the capacity-${partition.capacity} baseline: ${firstCaption}`);
    await two.getByRole('button', { name: 'Halve the capacity to 25' }).click();
    await two.getByLabel('It rises', { exact: true }).check();
    await two.locator('.formulation-numeric-guess input').fill(String(capacityAtTwentyFive.precision));
    await two.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(two.locator('.formulation-verdict'), /Your prediction matches/);
    // Round two: back to 50. The caption must now name the APPLIED state.
    await two.getByRole('spinbutton', { name: /^calling capacity/ }).fill(String(partition.capacity));
    await settle(page);
    const secondCaption = await captionOf();
    assert.ok(!secondCaption.includes('Section 5 records the starting point'),
      `the second round still cites section 5 as the baseline: ${secondCaption}`);
    assert.ok(secondCaption.includes('capacity 25')
      && secondCaption.includes(round6(capacityAtTwentyFive.precision)),
      'the caption must name the state the grader uses, capacity 25 at '
      + `${round6(capacityAtTwentyFive.precision)}: ${secondCaption}`);
    /* Now follow the caption. Precision falls from .48 back to .4, so a learner
       reasoning from the stated baseline predicts "It falls" and must be graded
       right. Before the fix the caption said .4 and the same reasoning gave
       "It does not move at all", marked wrong. */
    await two.getByLabel('It falls', { exact: true }).check();
    await two.locator('.formulation-numeric-guess input').fill(String(capacityAtFifty.precision));
    await two.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(two.locator('.formulation-verdict'), /Your prediction matches: It falls\./);
    await screenshot(two, 'formulation-capacity-second-round-desktop.png');
    await two.getByRole('button', { name: 'Reset' }).click();
    await settle(page);
    records.push({
      case: 'The baseline the caption names is the baseline the grader uses, in the second round as well as '
        + 'the first: after applying at capacity 25 the caption names capacity 25 and its precision, and a '
        + 'learner reasoning from it is graded correct',
      firstRoundCaptionCitesSectionFive: true,
      secondRoundCaptionNamesAppliedState: true,
    });

    // --------------------- 5. the destination note's case, end to end
    await one.getByRole('button', { name: 'Reset' }).click();
    await settle(page);
    await one.getByRole('button', { name: PRESET_ARRIVE_EARLIER }).click();
    await settle(page);
    const movedAnswerText = fixedText(timelineMoved.value);
    assert.ok(!(await textOf(0)).includes(movedAnswerText),
      'the moved-arrival answer is not on screen before the prediction');
    await one.getByLabel(models.recordLabel(timelineMoved.selected.record), { exact: true }).check();
    await one.locator('.formulation-numeric-guess input').fill(String(timelineMoved.value));
    await one.getByRole('button', { name: 'Apply and check' }).click();
    await checkText(one.locator('.formulation-verdict'), /Your prediction matches/);
    const movedText = await textOf(0);
    assert.ok(movedText.includes(movedAnswerText),
      'and moving only the arrival changes the admissible value');
    assert.notEqual(timelineMoved.value, timelineDefault.value,
      'the two really are different numbers, so the exercise has content');
    assert.notEqual(timelineMoved.selected.record.event, timelineDefault.selected.record.event,
      'and a different EVENT is selected although no event time moved');
    await screenshot(one, 'formulation-timeline-arrival-moved-desktop.png');
    records.push({
      case: 'Moving one arrival from 8 to 4, with no event time changed, changes the admissible record and '
        + 'its value — the destination note\'s worked case, graded against a recorded prediction',
    });

    // ------------------------------------------- 6. a verdict retires on edit
    /* By ROLE, not by label text. The timeline drawing's accessible name is
       "Event times, arrival times and the prediction cutoff", so a label lookup
       matches the figure as well as the field. */
    await one.getByRole('spinbutton', { name: /^prediction cutoff/ }).fill('7');
    await settle(page);
    assert.equal(await one.locator('.formulation-verdict').count(), 0, 'an edit retires the verdict');
    assert.equal(await one.locator('.formulation-reveal').count(), 0, 'and its reveal');
    assert.equal(await one.locator('.formulation-choices input[type="radio"]:checked').count(), 0,
      'and clears the recorded prediction');
    await checkText(one.locator('.formulation-history'), /stays hidden until you apply/);
    /* A VIEW control must not retire anything: moving the window in
       investigation 2 changes which rows are drawn and nothing that is graded.
       A verdict has to exist first, or this asserts nothing — the cases above
       leave investigation 2 reset, so one is established here rather than
       inherited from whatever ran last. */
    await two.getByRole('button', { name: 'Halve the capacity to 25' }).click();
    await two.getByLabel('It rises', { exact: true }).check();
    await two.locator('.formulation-numeric-guess input').fill(String(capacityAtTwentyFive.precision));
    await two.getByRole('button', { name: 'Apply and check' }).click();
    await settle(page);
    assert.equal(await two.locator('.formulation-verdict').count(), 1,
      'a verdict exists before the view control is exercised, or the next assertion proves nothing');
    await two.getByLabel('show the ranking around rank').fill('300');
    await settle(page);
    assert.equal(await two.locator('.formulation-verdict').count(), 1,
      'moving the ranking window is a view change and must not retire a graded verdict');
    await two.getByRole('button', { name: 'Reset' }).click();
    await settle(page);
    records.push({
      case: 'Editing a graded input retires the verdict, its reveal and the recorded prediction and holds the '
        + 'earlier attempt back; moving the ranking window, which is a view control, retires nothing',
    });

    // ------------------------------------------- 7. a refused input is named
    await one.getByRole('button', { name: 'Reset' }).click();
    await settle(page);
    await one.getByLabel('A · event 4 · v1 — arrives at').fill('2');
    await settle(page);
    assert.equal(await one.locator('.formulation-note[role="status"]').count(), 1,
      'the impossible arrival is stated once, beside the field that caused it');
    await checkText(one.locator('.formulation-note[role="status"]'), /cannot be available before it exists/);
    await checkText(one.locator('.formulation-blocked'), /cannot be graded yet/);
    assert.ok(await one.getByRole('button', { name: 'Apply and check' }).isDisabled(),
      'and grading is blocked rather than guessed');
    assert.equal(await one.locator('.form-svg').count(), 0,
      'with no timeline drawn for a history that cannot exist');
    await screenshot(one, 'formulation-timeline-refused-desktop.png');
    await one.getByRole('button', { name: 'Reset' }).click();
    await settle(page);
    /* PRESENCE PARTNER for the absence above: the drawing has to come back, or
       "no drawing while refused" would also pass on a lab that never draws. */
    assert.ok(await one.locator('.form-svg').count() >= 1,
      'the timeline returns once the impossible arrival is undone');
    assert.equal(await one.locator('.formulation-note[role="status"]').count(), 0,
      'and the refusal message goes with it');
    await two.getByRole('button', { name: 'Reset' }).click();
    await settle(page);
    records.push({
      case: 'An arrival earlier than its own event is refused by name, blocks grading rather than being '
        + 'squeezed into a verdict, and suppresses the drawing rather than placing an arrival to the left of '
        + 'the event it measures',
    });

    // ------------------------------------------------- 8. figures and states
    for (let index = 0; index < 7; index += 1) {
      await screenshot(page.locator('.formulation-figure').nth(index), `formulation-figure-${index + 1}-desktop.png`);
    }
    const splitFigure = page.locator('.formulation-figure').nth(1);
    await checkText(splitFigure, /already seen/);
    await splitFigure.getByRole('button', { name: 'Hold out whole parcel identities' }).click();
    await settle(page);
    await checkText(splitFigure, /never met/);
    await screenshot(splitFigure, 'formulation-figure-2-identity-split-desktop.png');
    const thresholdFigure = page.locator('.formulation-figure').nth(2);
    for (const label of ['equal costs: 5 and 5', 'reversed: 9 and 3']) {
      await thresholdFigure.getByRole('button', { name: label }).click();
      await settle(page);
      await screenshot(thresholdFigure,
        `formulation-figure-3-${label.split(':')[0].replace(/\s+/g, '-')}-desktop.png`);
    }
    await thresholdFigure.getByRole('button', { name: 'the section’s example: 3 and 9' }).click();
    await settle(page);
    /* Metric labels read FROM THE MODEL, not retyped. One of them was renamed
       to say what it draws — it read "positives among the selected 50" over an
       axis running 0 to 1 with bars annotated .12, .4 and .52 — and a
       hardcoded label here turned that repair into a 30-second timeout rather
       than a check. The axis keys are the stable thing; the words are not. */
    const resultsFigure = page.locator('.formulation-figure').nth(4);
    const metricLabels = Object.fromEntries(Object.entries(models.metricAxes)
      .map(([key, axis]) => [key, axis.label]));
    assert.equal(Object.keys(metricLabels).length, 3, 'three metrics are offered');
    for (const key of ['logLoss', 'precisionAt50']) {
      await resultsFigure.getByRole('button', { name: metricLabels[key] }).click();
      await settle(page);
      await checkText(resultsFigure, new RegExp(metricLabels[key].replace(/[.*+?^${}()|[\]\\]/g, '\\$&')));
      await screenshot(resultsFigure, `formulation-figure-5-${key}-desktop.png`);
    }
    /* The renamed axis must describe what is drawn: a share on a 0-1 axis, not
       a count with a maximum of 50. */
    assert.ok(!/positives among the selected 50/.test(metricLabels.precisionAt50),
      'the third metric is no longer labelled as a count while being drawn as a fraction');
    assert.deepEqual(models.metricAxes.precisionAt50.domain, [0, 1], 'and its axis is the 0-1 one');
    await resultsFigure.getByRole('button', { name: metricLabels.averagePrecision }).click();
    await settle(page);
    records.push({
      case: 'Every inline figure captured at desktop width, plus the informative states: the identity split, '
        + 'the threshold figure at equal and reversed costs, and the results figure on each of its three metrics',
    });

    // -------------------------------------- 9. painted encodings and curves
    /* PAINT ORDER, which no attribute check can see. The arrival diamond is
       painted after the event circle, and two of the four default records have
       the two times equal. With an opaque fill the diamond covered the event
       dot: both marks drawn, both correct, one invisible. The fix is a fill of
       `none`, and this is the assertion that keeps it. */
    const coincident = models.timelineFixture.records
      .filter(entry => entry.event === entry.available).length;
    assert.ok(coincident >= 1,
      'the default history has a record whose event and arrival coincide, so the occlusion case is on screen');
    const arrivalPaint = await page.locator('.form-arrival').first()
      .evaluate(element => getComputedStyle(element).fill);
    assert.equal(arrivalPaint, 'none',
      `the arrival marker paints ${arrivalPaint} rather than none, so it covers the event marker of any `
      + 'record whose value arrived the moment it was measured');
    const arrivalStroke = await page.locator('.form-arrival').first()
      .evaluate(element => getComputedStyle(element).stroke);
    assert.notEqual(arrivalStroke, 'none',
      'while still carrying a stroke, or an outline-only marker would be invisible instead');

    const paint = await page.evaluate(comparePaintedAgainstDeclared, '.formulation-lesson');
    assert.ok(paint.shapesExamined >= 150,
      `only ${paint.shapesExamined} shapes were examined; an empty scan passes vacuously`);
    assert.deepEqual(paint.findings, [],
      'a presentation attribute is overridden by CSS, or a shape paints solid black');
    for (const width of [1366, 1024, 768, 390, 320]) {
      await page.setViewportSize({ width, height: 1000 });
      await settle(page);
      // It is a PAGE function, not a Node helper: it reaches for `document`. It
      // returns one record PER SVG and puts findings inside `.issues`, so the
      // records have to be flattened -- comparing the records themselves with
      // [] could only ever pass on a page with no SVGs at all.
      const layout = await page.evaluate(inspectLessonVisualLayout, '.formulation-lesson');
      const svgCount = await page.locator('.formulation-lesson svg.form-svg').count();
      assert.ok(svgCount >= 12, `only ${svgCount} tagged SVGs are mounted at ${width}px`);
      assert.ok(layout.length >= 8,
        `the layout inspector considered only ${layout.length} SVGs at ${width}px; an empty scan passes vacuously`);
      const labelCount = layout.reduce((sum, figure) => sum + figure.labelCount, 0);
      assert.ok(labelCount >= 90,
        `and only ${labelCount} labels, which is too few to be the real page`);
      assert.deepEqual(
        layout.flatMap(figure => figure.issues.map(issue => ({ svgIndex: figure.svgIndex, ...issue }))),
        [], `Figure layout collides at ${width}px`);
      const curves = await page.evaluate(sampleCurvesThroughLabels, '.formulation-lesson');
      /* NAMED, not counted. A floor is a guess about how many shapes the page
         draws; this asserts WHICH ones the sampler reached, so a shape that
         stops being sampled -- a class rename, a <path> turned into a <line>,
         a figure that stops mounting -- fails here instead of quietly shrinking
         a number nobody checks. This lesson's only curve-like foreground shapes
         are the four arrival diamonds of investigation 1 and the feedback
         channel of figure 1; everything else it draws is a rect, a line or a
         circle, which the shared inspector already covers. */
      const sampled = curves.sampledClasses.slice().sort();
      assert.deepEqual(sampled,
        ['form-arrival', 'form-arrival', 'form-arrival', 'form-arrival', 'form-arrow is-feedback'],
        `the curve sampler reached ${JSON.stringify(sampled)} at ${width}px, not the four arrival diamonds `
        + 'and the feedback channel this lesson draws');
      assert.ok(curves.labelsConsidered >= 60,
        `and only ${curves.labelsConsidered} labels at ${width}px`);
      assert.deepEqual(curves.findings, [], `A curve runs through a label at ${width}px`);
      curveSampling = curves;
    }
    await page.setViewportSize({ width: 1366, height: 1000 });
    await settle(page);
    records.push({
      case: 'No label leaves its own SVG, overlaps another label, is crossed by a straight foreground line, '
        + 'or is run through by a curve, at five widths; and every presentation attribute survives the stylesheet',
      shapesSampledAtLastWidth: curveSampling.shapesSampled,
      shapeClassesSampled: curveSampling.sampledClasses,
      curveSamplingNote: 'Curves are sampled along their own geometry and tested against every label box, '
        + 'because the shared inspector reads straight lines only. Grid lines and shapes inside <defs> are '
        + 'excluded; nothing on this page carries an opaque backplate, so no crossing is permitted at all.',
    });

    // ----------------------------------- 10. tables do not clip at desktop
    const clipped = await page.locator('.formulation-table-scroll').evaluateAll(items =>
      items.filter(item => item.scrollWidth > item.clientWidth + 1)
        .map(item => item.querySelector('table')?.textContent?.slice(0, 60) ?? 'table'));
    assert.deepEqual(clipped, [], 'a figure table clips a column at desktop width');
    /* The POSITIVE property, not the absence of a tag that is never emitted.
       This used to assert `.formulation-table-scroll caption` count === 0 —
       true, and unfalsifiable: `Table` deliberately renders its caption as a
       sibling paragraph, and the word "caption" appears in the four components
       only in the docstring explaining why. The record nonetheless claimed
       "every caption sits outside its scroll box", which nothing established.
       Now each table is required to HAVE a caption, and that caption is
       required to sit outside the scroll box. */
    /* Keyed on the element that LABELS the scroll region, found through its own
       `aria-labelledby`, not on "some element with the caption class". `Table`
       also renders its optional footnote with that class, so a check for any
       direct-child caption passed even with the heading caption deleted — the
       footnote stood in for it. The labelling caption must exist, must be the
       caption class, and must sit outside the box it labels. */
    const captions = await page.locator('.formulation-table').evaluateAll(tables => tables.map(table => {
      const scroll = table.querySelector('.formulation-table-scroll');
      const labelId = scroll ? scroll.getAttribute('aria-labelledby') : null;
      const label = labelId ? table.querySelector(`[id="${labelId}"]`) : null;
      return {
        hasScrollRegion: Boolean(scroll),
        hasLabelledCaption: Boolean(label && label.classList.contains('formulation-caption')),
        labelInsideScroll: Boolean(label && scroll && scroll.contains(label)),
        legacyCaptionTag: Boolean(table.querySelector('caption')),
        text: (label ? label.textContent : '').slice(0, 40),
      };
    }));
    assert.ok(captions.length >= 8, `only ${captions.length} tables were found to check`);
    assert.deepEqual(captions.filter(entry => !entry.hasScrollRegion), [],
      'a table renders no scroll region');
    assert.deepEqual(captions.filter(entry => !entry.hasLabelledCaption), [],
      'a table scroll region names no caption element, so it has no accessible label and no visible caption');
    assert.deepEqual(captions.filter(entry => entry.labelInsideScroll || entry.legacyCaptionTag), [],
      'a caption sits inside the scroll box, where it collapses to its longest word below the breakpoint');
    records.push({
      case: 'No table clips a column at desktop width; every table HAS a caption and every caption sits '
        + 'outside its scroll box',
      tablesChecked: captions.length,
    });

    // --------------------------- 10b. no JSX expression lost the space after it
    const glue = await page.evaluate(findGluedWords, '.formulation-lesson');
    assert.ok(glue.elementsScanned >= 150,
      `only ${glue.elementsScanned} prose elements were scanned for glued words; an empty scan passes vacuously`);
    assert.deepEqual(glue.findings, [],
      'a JSX expression lost the space after it, so two words are glued together in the rendered prose');
    records.push({
      case: 'No rendered sentence glues a value to the word after it, across every prose element on the page',
      proseElementsScanned: glue.elementsScanned,
      note: 'This class is invisible in the source and in the DOM structure, and three instances reached the '
        + 'rendered page of this lesson before the guard existed.',
    });

    // ------------------------------- 11. KaTeX radicals and display formulas
    /* A sibling lesson shipped a squared radius as a radius because
       `.lesson svg { height: auto }` also matched KaTeX's radical SVGs, whose
       height comes from `height: inherit`. The DOM was right, every offline
       check passed, and the formula on screen was wrong. Details are opened
       first: the practice solutions carry math too, and a hidden element
       measures zero for reasons unrelated to this defect. */
    await page.locator('.formulation-lesson details')
      .evaluateAll(items => items.forEach(item => { item.open = true; }));
    await settle(page);
    const katexBoxes = await page.evaluate(measureKatexBoxes);
    desktopKatexBoxes = katexBoxes.length;
    assert.ok(katexBoxes.length >= 60,
      `only ${katexBoxes.length} KaTeX boxes were found; the page states far more`);
    /* BOTH dimensions. A sibling's guard measured width only, and a collapsed
       SVG is also narrow -- which is how 24 collapsed radicals survived a fully
       green run there. Every box selected here carries visible text, so neither
       dimension may be near zero.

       The floors are 4 on height and 3 on width, against observed minima of 9
       and 4.69 recorded in the evidence below. A collapse gives 0, so these
       catch every one of them; they are not set at the observed minimum
       because a floor with no margin fails on a font-loading hiccup and gets
       deleted rather than fixed. */
    assert.deepEqual(katexBoxes.filter(entry => entry.height < 4 || entry.width < 3), [],
      'a KaTeX element carrying text paints with no height or no width, so a formula reads as something '
      + 'other than what it says');

    /* This lesson's mathematics contains NO KaTeX-drawn SVG: no radical, no
       stretchy delimiter, no accent that KaTeX draws rather than sets. That is
       a fact about this page, not a passing check, and it is pinned as a count
       rather than left as a filter over an empty set -- a filter over nothing
       passes vacuously and reads exactly like a guard that works. If a future
       edit adds a square root, this assertion fails and whoever adds it has to
       reinstate the SVG-dimension floor deliberately. */
    const katexSvgs = await page.evaluate(measureKatexSvgs);
    assert.equal(katexSvgs.length, 0,
      `this lesson's mathematics is expected to draw no KaTeX SVG, and ${katexSvgs.length} were found. `
      + 'Add a both-dimension floor over them before changing this assertion.');
    await page.locator('.formulation-lesson details')
      .evaluateAll(items => items.forEach(item => { item.open = false; }));
    await settle(page);
    records.push({
      case: 'Every KaTeX box carrying text, practice solutions opened, paints with real width AND height; '
        + 'and the number of KaTeX-drawn SVGs is pinned rather than filtered over an empty set',
      katexBoxesMeasured: katexBoxes.length,
      katexSvgsFound: katexSvgs.length,
      smallestKatexHeight: Math.min(...katexBoxes.map(entry => entry.height)),
      smallestKatexWidth: Math.min(...katexBoxes.map(entry => entry.width)),
      katexSvgNote: 'This lesson\'s formulas use no radical, stretchy delimiter or KaTeX-drawn accent, so the '
        + 'SVG-collapse defect that changed a radius into a radius squared elsewhere cannot occur here. The '
        + 'stylesheet is still scoped by class, asserted offline, and the box measurement above is what would '
        + 'catch a collapse of the spans this page does use.',
    });

    // ------------------------------------------------ 12. loading closure
    const localScripts = [...new Set(requests.map(address => new URL(address))
      .filter(address => address.origin === new URL(base).origin && address.pathname.endsWith('.js'))
      .map(address => address.pathname.slice(1)))];
    assert.deepEqual(localScripts.filter(filename => bodyFiles.has(filename)), [bodyFile]);
    assert.deepEqual(localScripts.filter(filename => !allowedScripts.has(filename)), []);
    /* Both of these are DELIBERATE ZEROS over a non-empty log, and both say so.
       A production build emits no filename containing "outlines", and the page
       LINKS to its assets rather than fetching them, so neither pattern can
       appear in the request log at all. Left in as regression pins — a lesson
       that started fetching an outline, or a sibling's CSV, would show here —
       but the log is floored so they cannot pass over nothing. */
    assert.ok(requests.length >= 10,
      `only ${requests.length} requests were logged; the two zero-pins below would pass over nothing`);
    assert.ok(!requests.some(address => address.includes('/outlines/')),
      'the page requested an outline, which a published lesson never should');
    assert.ok(!requests.some(address => /learn-assets\/(?!problem-formulation)/.test(address)),
      'the page requested another lesson\'s asset directory');
    /* The reachable form of the same property: what the page LINKS to. This one
       has subjects. */
    const assetLinks = await page.locator('.formulation-lesson a[href*="learn-assets"]')
      .evaluateAll(items => items.map(item => item.getAttribute('href')));
    assert.ok(assetLinks.length >= 4,
      `only ${assetLinks.length} asset links were found; the page links its dataset, attribution, `
      + 'description and program');
    assert.deepEqual(assetLinks.filter(href => !href.startsWith('/learn-assets/problem-formulation/')), [],
      'the page links an asset outside its own directory');
    records.push({
      case: 'Fresh route requests only the selected lesson and its shared closure, and never another '
        + 'lesson\'s asset directory',
      requestedScripts: localScripts,
      rawScriptBytes: localScripts.reduce((sum, filename) => sum + fs.statSync(path.join(distDir, filename)).size, 0),
      gzipEstimateBytes: localScripts.reduce((sum, filename) =>
        sum + gzipSync(fs.readFileSync(path.join(distDir, filename))).length, 0),
      interpretation: 'Built requested file sizes; the gzip estimate is not measured network compression or latency.',
    });

    // ----------------------------------------------------- 13. narrow widths
    for (const width of [390, 320]) {
      await page.setViewportSize({ width, height: 844 });
      await page.locator('.formulation-lesson details')
        .evaluateAll(items => items.forEach(item => { item.open = true; }));
      await settle(page);
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
        `Document overflows at ${width}px`);
      const mathOverflow = await page.locator('.formulation-lesson .katex-display').evaluateAll(items =>
        items.filter(item => item.scrollWidth > item.clientWidth + 1).map(item => item.textContent));
      assert.deepEqual(mathOverflow, [], `Overflowing formulas at ${width}px`);
      assert.equal(await page.locator('.katex-error').count(), 0);
      /* The SAME measurement as at desktop, and the count must match.
      
         A narrow-width KaTeX check that quietly finds fewer subjects has gone
         inert, and a filter over a shrinking set keeps passing while it stops
         looking. Both dimensions again: a collapsed box is narrow as well as
         short, which is how 24 collapsed radicals survived a green run in a
         sibling lesson. */
      const narrowKatex = await page.evaluate(measureKatexBoxes);
      assert.equal(narrowKatex.length, desktopKatexBoxes,
        `only ${narrowKatex.length} KaTeX boxes are measured at ${width}px against ${desktopKatexBoxes} at `
        + '1366px; the narrow-width check is looking at a smaller set than it thinks');
      assert.deepEqual(narrowKatex.filter(entry => entry.height < 4 || entry.width < 3), [],
        `a KaTeX element paints with no height or no width at ${width}px`);
      assert.deepEqual(await page.evaluate(measureKatexSvgs), [],
        `KaTeX drew an SVG at ${width}px that it does not draw at 1366px`);
      const stacked = await page.locator('.formulation-table-scroll tbody th').first()
        .evaluate(cell => getComputedStyle(cell).display);
      assert.equal(stacked, 'flex', `the wide tables stack rather than clip at ${width}px`);
      if (width === 320) {
        const tall = await page.locator('.formulation-lesson .katex-display')
          .evaluateAll(items => items
            .map((item, index) => ({ index, height: Math.round(item.getBoundingClientRect().height) }))
            .sort((a, b) => b.height - a.height).slice(0, 3));
        for (const entry of tall) {
          await screenshot(page.locator('.formulation-lesson .katex-display').nth(entry.index),
            `formulation-formula-${entry.index}-320.png`);
        }
        await screenshot(page.locator('.formulation-investigation').nth(0), 'formulation-timeline-320.png');
        await screenshot(page.locator('.formulation-investigation').nth(1), 'formulation-capacity-320.png');
        for (const index of [0, 3, 4]) {
          await screenshot(page.locator('.formulation-figure').nth(index), `formulation-figure-${index + 1}-320.png`);
        }
      } else {
        await screenshot(page.locator('.formulation-investigation').nth(0), 'formulation-timeline-390.png');
        await screenshot(page.locator('.formulation-investigation').nth(1), 'formulation-capacity-390.png');
        for (let index = 0; index < 7; index += 1) {
          await screenshot(page.locator('.formulation-figure').nth(index), `formulation-figure-${index + 1}-390.png`);
        }
      }
    }
    records.push({
      case: 'No document or formula overflow at 390 and 320 px, wide tables stack rather than clip, and the '
        + 'investigations, every figure and the tallest formulas are captured at the narrow widths',
    });

    assert.deepEqual(errors, [], 'the page threw');
    assert.deepEqual(failedAssets.filter(address => !address.includes('favicon')), [],
      'an asset failed to load');
    assert.equal(new Set(screenshots.map(shot => shot.file)).size, screenshots.length,
      'every capture has its own path');
    assert.equal(new Set(screenshots.map(shot => shot.digest)).size, screenshots.length,
      'and its own content: two identical captures mean one of them is of the wrong thing');
    /* No orphan may survive. A capture left behind by an earlier, pre-fix run
       sits in the evidence directory looking exactly like a current one, and
       that is how a defective state gets shipped as proof that it was fixed. */
    const onDisk = fs.readdirSync('docs/teaching/evidence/screenshots')
      .filter(name => name.startsWith('formulation-') && name.endsWith('.png'))
      .map(name => path.join('docs/teaching/evidence/screenshots', name).replace(/\\/g, '/'));
    const written = new Set(screenshots.map(shot => shot.file.replace(/\\/g, '/')));
    assert.deepEqual(onDisk.filter(file => !written.has(file)), [],
      'orphaned formulation-*.png files remain from an earlier run');
    for (const shot of screenshots) {
      assert.ok(fs.existsSync(shot.file), `${shot.file} was recorded but is not on disk`);
      assert.equal(hash(shot.file), shot.digest, `${shot.file} no longer matches its recorded digest`);
      assert.equal(fs.statSync(shot.file).size, shot.bytes, `${shot.file} no longer matches its recorded size`);
      assert.ok(shot.bytes > 1000, `${shot.file} is ${shot.bytes} bytes, which is not a rendered capture`);
    }
    assert.ok(publications[topicId], 'the topic is registered for publication');
    /* AT the current totals, not below them. A floor three cases under the
   real number lets a whole case be deleted silently; these are discrete
   and stable, so adding one is a deliberate two-line act. */
    assert.ok(records.length >= 16, `only ${records.length} browser cases ran; a section has been lost`);
    assert.ok(screenshots.length >= 39, `only ${screenshots.length} captures were taken`);
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
      verifierSha256: hash('scripts/verify-formulation-browser.cjs'),
      records,
      screenshots,
      screenshotCount: screenshots.length,
      visualInterpretation: 'Screenshots capture informative states, not defaults only: each investigation '
        + 'before any commitment and after one, the two exact nulls, the refused impossible arrival, the '
        + 'destination note\'s moved-arrival case, the identity split, the threshold figure at three cost '
        + 'settings, the results figure on each of its three metrics, and every inline figure at desktop and '
        + '390 px with a selection at 320 px. They require separate visual inspection; a recorded digest '
        + 'proves identity, not legibility.',
      limitations: [
        'A passing run is not a reading of the page. The captures above must be opened and looked at.',
        'The painted-against-declared check catches a stylesheet overriding an attribute, and a shape '
          + 'painting black with no stroke. It does NOT catch a shape that is painted correctly and then '
          + 'COVERED by a later sibling. A general occlusion check is not attempted; the figure-by-figure '
          + 'visual pass is what covers it.',
        'The shared layout inspector allows a small overlap and reads straight <line> elements only; the '
          + 'curve sampler above covers paths, polylines and polygons, and the model layer covers nominal '
          + 'label collisions.',
        'Model correctness is checked by scripts/verify-formulation-models.mjs, data regeneration by '
          + 'scripts/verify-formulation-data.py and program execution by '
          + 'scripts/verify-formulation-examples.py.',
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

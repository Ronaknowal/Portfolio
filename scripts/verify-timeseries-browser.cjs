// Production browser review of the time-series validation and forecasting
// lesson: visible content, the three investigations, prediction/commit/reveal/
// retirement, the leak property, figure geometry, narrow layouts and loading
// closure.
//
// Run against a preview of a production build:
//   npx vite build --outDir dist-ts
//   npx vite preview --outDir dist-ts --host 127.0.0.1 --port 4196 --strictPort
//   PLAYWRIGHT_PACKAGE=... DIST_DIR=dist-ts LEARNING_BASE_URL=http://127.0.0.1:4196 \
//     node scripts/verify-timeseries-browser.cjs
//
// --strictPort is not optional. Without it vite silently moves to the next free
// port and this verifier then measures whichever sibling lesson is serving
// there.
//
// The check this file exists for is section 3 below. The dominant defect class
// in these lessons is a page that grades a correct answer wrong, or shows the
// answer first -- and asserting that no verdict BANNER exists on first paint is
// what let that ship repeatedly, because a banner is easy to withhold while the
// graded number leaks out of a readout beside it. So the assertion here is a
// property, pinned numerically: for EVERY investigation, the exact text of the
// graded quantity -- computed independently in this process from
// timeseries-models.js, not read off the page -- must be ABSENT from that
// investigation's rendered text before a prediction is committed and PRESENT
// after. The absence assertion is always paired with the presence assertion, so
// a pin that could never match is caught rather than counted.
//
// Investigation 3 has TWO gates, because a forecasting lab that shows the
// outcome beside the forecast is not a forecasting lab. Both are pinned.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { gzipSync } = require('node:zlib');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const { inspectLessonVisualLayout } = require('./lib/lesson-visual-layout.cjs');

const distDir = process.env.DIST_DIR || 'dist-ts';
const base = (process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4196').replace(/\/+$/, '');
const topicId = 'time-series-validation-forecasting-baselines';
const sourcePath = `src/learn/data/topics/${topicId}.jsx`;
const route = `${base}/learn/path/full-curriculum/${topicId}?module=classical-ml`;
const evidencePath = 'docs/teaching/evidence/timeseries-browser.json';
const screenshotDirectory = 'docs/teaching/evidence/screenshots';
const read = filename => JSON.parse(fs.readFileSync(filename, 'utf8'));
const hash = filename => createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
const normalize = text => text.replace(/\s+/g, ' ').trim();
const ownedFiles = [
  sourcePath,
  'src/learn/data/timeseries-models.js',
  'src/learn/data/timeseries-data.js',
  'src/learn/data/timeseries-examples.js',
  'src/learn/components/lesson-labs/TimeSeriesShared.jsx',
  'src/learn/components/lesson-labs/TimeSeriesLabs.jsx',
  'src/learn/components/lesson-labs/TimeSeriesFigures.jsx',
  'src/learn/components/lesson-labs/timeseries-labs.css',
  `src/learn/data/curriculum/blueprints/${topicId}.js`,
  'public/learn-assets/time-series-validation/bike-sharing-daily.csv',
  'public/learn-assets/time-series-validation/ATTRIBUTION.txt',
  'public/learn-assets/time-series-validation/forecast-experiments.py',
];

/** Runs in the page. Samples every curve along its own geometry and reports the
 * labels it passes through.
 *
 * Copied from scripts/verify-bias-variance-browser.cjs, because
 * `scripts/lib/lesson-visual-layout.cjs` iterates `svg.querySelectorAll('line')`
 * and so cannot see a <path> or <polyline> at all -- which is how four data
 * curves crossed value labels behind a fully green run.
 *
 * THE SELECTOR WAS CHECKED AGAINST WHAT THIS LESSON ACTUALLY DRAWS. A sibling's
 * copy inspected 2 shapes of 228 because it queried for `path` on a lesson that
 * draws `line`. This lesson's curves are `polyline` (the horizon chart, the
 * development history chart) and its arrowheads are `polygon`, both of which
 * this query covers; the count of shapes it actually found is asserted below
 * rather than assumed.
 *
 * Grid lines are background and are skipped. Nothing on this page carries an
 * opaque backplate, so no crossing is allowed at all. */
function sampleCurvesThroughLabels(root) {
  const SAMPLES = 320;
  const HALO_ALLOWANCE = 0.02;
  const findings = [];
  let shapesInspected = 0;
  for (const svg of document.querySelectorAll(`${root} svg`)) {
    const labels = [...svg.querySelectorAll('text')]
      .filter(text => text.textContent.trim() && text.getClientRects().length)
      .map(text => ({
        text: text.textContent.trim().slice(0, 40),
        halo: text.classList.contains('ts-halo'),
        box: text.getBoundingClientRect(),
      }));
    if (!labels.length) continue;
    for (const shape of svg.querySelectorAll('path, polyline, polygon')) {
      if (shape.classList.contains('ts-grid')) continue;
      if (typeof shape.getTotalLength !== 'function') continue;
      const length = shape.getTotalLength();
      if (!(length > 0)) continue;
      const matrix = shape.getScreenCTM();
      if (!matrix) continue;
      shapesInspected += 1;
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
  return { findings, shapesInspected };
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

/** Runs in the page. THE CHECK B1 EXISTS FOR.
 *
 * A legend is a claim about the picture. Figure 1's legend said "not yet
 * observed (hatched)" and its caption repeated it, and the picture contained no
 * hatching at all: 31 hatch lines were in the DOM with every attribute correct,
 * emitted BEFORE the opaque cell rects that were then painted over them. SVG
 * has no z-index; document order is paint order. No attribute check can see
 * this, and it is the second lesson in this scope where the legend was right
 * and the picture was empty.
 *
 * So: for every figure, every encoding its legend NAMES must be realized by at
 * least one element carrying that `data-encoding`, and at least one such
 * element must have a point on itself that no LATER opaque shape covers. An
 * element hidden under something drawn after it counts as absent, because to a
 * reader it is.
 *
 * Occlusion is decided GEOMETRICALLY, in the SVG's own coordinate space, not
 * with `elementsFromPoint`. That call takes viewport coordinates, and this
 * reader scrolls an inner container rather than the window -- so every sample
 * point for a figure below the fold landed outside the window and the first
 * draft reported every encoding in every figure as covered. A check that fails
 * on everything is exactly as useless as one that fails on nothing, and the
 * tempting repair is to loosen it until it goes quiet.
 *
 * Only `rect` and `circle` count as occluders. They are what actually caused
 * this defect, and a bounding box is an exact cover test for an axis-aligned
 * rect but only an approximation for a path -- so a path is not allowed to
 * generate a finding it cannot justify. */
function legendEncodingsPainted(root) {
  const findings = [];
  const measured = [];
  let encodingsChecked = 0;

  const toRgb = value => {
    const match = /rgba?\((\d+),\s*(\d+),\s*(\d+)(?:,\s*([\d.]+))?\)/.exec(value || '');
    return match ? { r: +match[1], g: +match[2], b: +match[3] } : null;
  };
  const luminance = ({ r, g, b }) => {
    const channel = value => {
      const v = value / 255;
      return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
    };
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b);
  };
  const ratio = (a, b) => {
    if (!a || !b) return null;
    const [high, low] = [luminance(a), luminance(b)].sort((x, y) => y - x);
    return Number(((high + 0.05) / (low + 0.05)).toFixed(2));
  };
  const AREA = new Set(['rect', 'circle', 'ellipse', 'polygon', 'path']);
  const paintedColour = element => {
    const style = getComputedStyle(element);
    const hasStroke = style.stroke && style.stroke !== 'none' && Number(style.strokeOpacity) !== 0;
    const hasFill = style.fill && style.fill !== 'none';
    /* An area shape is read by its FILL -- that is what occupies the region a
       mark sits on. A line is read by its stroke. Preferring stroke everywhere
       measured a filled cell by its 1px border and reported the hatch at 2.79:1
       against a colour no reader sees. */
    if (AREA.has(element.tagName) && hasFill) return toRgb(style.fill);
    if (hasStroke) return toRgb(style.stroke);
    if (hasFill) return toRgb(style.fill);
    return null;
  };
  const opaque = element => {
    const style = getComputedStyle(element);
    if (style.fill === 'none') return false;
    return Number(style.fillOpacity) >= 0.9 && Number(style.opacity) >= 0.9;
  };
  const covers = (shape, x, y) => {
    const box = shape.getBoundingClientRect();
    if (x < box.left || x > box.right || y < box.top || y > box.bottom) return false;
    if (shape.tagName === 'circle') {
      const cx = box.left + box.width / 2;
      const cy = box.top + box.height / 2;
      return ((x - cx) ** 2 + (y - cy) ** 2) <= (box.width / 2) ** 2;
    }
    return true;
  };
  /* A HOLLOW closed shape paints on its outline, and its centre is a hole.
     Sampling the centre asked two wrong questions at once. For occlusion: a
     covered hole hides nothing, so an open marker with something drawn over its
     middle would have been reported as painted over when its ring is fully
     visible. For contrast: it compared the stroke against whatever sat inside
     the hole, which reported figure 5's cream coincidence ring at 1.75:1
     against the green marker it ENCIRCLES -- a comparison no reader makes, and
     a number that would have read as a legibility finding.

     Only closed area shapes are treated this way. A `line` or `path` has a
     bounding box whose edges its stroke need not touch at all -- a diagonal
     hatch line misses its own box corners -- so those keep the centre sample,
     which for them is on the stroke. */
  const HOLLOW_ELIGIBLE = new Set(['circle', 'ellipse', 'rect']);
  const samplePoints = element => {
    const box = element.getBoundingClientRect();
    const midX = box.left + box.width / 2;
    const midY = box.top + box.height / 2;
    const fill = getComputedStyle(element).fill;
    const hollow = !fill || fill === 'none' || Number(getComputedStyle(element).fillOpacity) === 0;
    if (hollow && HOLLOW_ELIGIBLE.has(element.tagName) && box.width > 3 && box.height > 3) {
      return [
        [box.left + 1, midY],
        [box.right - 1, midY],
        [midX, box.top + 1],
        [midX, box.bottom - 1],
      ];
    }
    return [
      [midX, midY],
      [midX, box.top + box.height * 0.25],
      [midX, box.top + box.height * 0.75],
    ];
  };

  document.querySelectorAll(`${root} .ts-figure`).forEach((figure, figureIndex) => {
    const keys = [...new Set([...figure.querySelectorAll('.ts-swatch')]
      .map(swatch => [...swatch.classList].find(name => name.startsWith('is-')))
      .filter(Boolean)
      .map(name => name.slice(3)))];
    for (const key of keys) {
      encodingsChecked += 1;
      const marks = [...figure.querySelectorAll(`[data-encoding="${key}"]`)]
        .filter(element => element.getClientRects().length);
      if (!marks.length) {
        findings.push({ figureIndex, key, problem: 'the legend names an encoding no drawn element claims' });
        continue;
      }
      let best = null;
      for (const element of marks) {
        const svg = element.ownerSVGElement;
        if (!svg) continue;
        const siblings = [...svg.querySelectorAll('rect, circle')].filter(shape => shape !== element && opaque(shape));
        const later = siblings.filter(shape =>
          (element.compareDocumentPosition(shape) & Node.DOCUMENT_POSITION_FOLLOWING) !== 0);
        const earlier = siblings.filter(shape =>
          (element.compareDocumentPosition(shape) & Node.DOCUMENT_POSITION_PRECEDING) !== 0);
        for (const [x, y] of samplePoints(element)) {
          if (later.some(shape => covers(shape, x, y))) continue;
          const beneath = earlier.filter(shape => covers(shape, x, y)).pop();
          best = {
            figureIndex, key, tag: element.tagName,
            laterOpaqueShapesInSvg: later.length,
            contrastAgainstWhatIsUnderIt: ratio(paintedColour(element), beneath ? paintedColour(beneath) : null),
          };
          break;
        }
        if (best) break;
      }
      if (best) measured.push(best);
      else {
        findings.push({
          figureIndex, key, marks: marks.length,
          problem: 'every element realizing this encoding is covered by an opaque shape drawn after it',
        });
      }
    }
  });
  return { findings, measured, encodingsChecked };
}

/** Runs in the page. Every arrow's drawn style must be the one its own input
 * maps to.
 *
 * Figure 3 styled arrows by LANE, so lane 1's first arrow -- whose input is the
 * origin's own observed count -- was drawn "input is this chain's own
 * prediction". The map comes from the model layer rather than being retyped
 * here, so the page is checked against the same table it draws from. */
function arrowStylesMatchTheirInput({ root, styleByInputKind }) {
  const findings = [];
  let checked = 0;
  for (const element of document.querySelectorAll(`${root} [data-input-kind]`)) {
    checked += 1;
    const kind = element.getAttribute('data-input-kind');
    const classes = element.getAttribute('class') || '';
    if (classes.includes('is-invalid')) continue;
    const expected = styleByInputKind[kind];
    if (!expected || !classes.includes(`is-${expected}`)) {
      findings.push({ inputKind: kind, classes, expectedStyle: expected ?? null });
    }
  }
  return { findings, checked };
}

/** Runs in the page. Reports any label whose painted box extends past the edge
 * of its own SVG.
 *
 * The shared inspector has a check of this name but computes boxes through
 * getBBox and a screen matrix; it did not see labels overflowing by one or two
 * pixels. These are the ones that get clipped on a real page, and a lane label
 * clipped at its first digit is a lane nobody can identify. Measured here with
 * getBoundingClientRect on both, at every width. */
function labelsOutsideTheirSvg(root) {
  const findings = [];
  let labelsMeasured = 0;
  document.querySelectorAll(`${root} svg.ts-diagram`).forEach((svg, index) => {
    const svgBox = svg.getBoundingClientRect();
    svg.querySelectorAll('text').forEach(text => {
      if (!text.textContent.trim() || !text.getClientRects().length) return;
      labelsMeasured += 1;
      const box = text.getBoundingClientRect();
      const worst = Math.max(svgBox.left - box.left, box.right - svgBox.right,
        svgBox.top - box.top, box.bottom - svgBox.bottom);
      if (worst > 0.5) {
        findings.push({
          svgIndex: index,
          label: text.textContent.trim().slice(0, 30),
          pixelsOutside: Number(worst.toFixed(2)),
        });
      }
    });
  });
  return { findings, labelsMeasured };
}

/** Runs in the page. Reports any label whose box overlaps a POINT MARK.
 *
 * The curve sampler reads path, polyline and polygon; the shared inspector
 * reads straight `line` elements. Neither looks at a `rect` or a `circle`, and
 * this lesson draws its target days, feature days, arrival days and issue days
 * as exactly those. An axis title landed squarely on a row of target marks and
 * every geometry check stayed green; only reading the screenshot found it.
 *
 * Container shapes are excluded by name rather than by guesswork: a calendar
 * cell, a strip cell, a comparison bar and a training band all carry text over
 * them BY DESIGN, and flagging those would be a check nobody could keep green.
 * What is reported is text over a mark that means one observation. */
function labelsOverPointMarks(root) {
  const POINT_MARKS = ['ts-target-mark', 'ts-day-dot', 'ts-feature-mark', 'ts-label-mark',
    'ts-arrival-mark', 'ts-issue-mark', 'ts-series-mark', 'ts-arrow-head', 'ts-coincident'];
  const findings = [];
  let marksInspected = 0;
  for (const svg of document.querySelectorAll(`${root} svg`)) {
    const labels = [...svg.querySelectorAll('text')]
      .filter(text => text.textContent.trim() && text.getClientRects().length)
      .map(text => ({ text: text.textContent.trim().slice(0, 40), box: text.getBoundingClientRect() }));
    if (!labels.length) continue;
    for (const mark of svg.querySelectorAll(POINT_MARKS.map(name => `.${name}`).join(', '))) {
      if (!mark.getClientRects().length) continue;
      marksInspected += 1;
      const box = mark.getBoundingClientRect();
      for (const label of labels) {
        const overlapX = Math.min(box.right, label.box.right) - Math.max(box.left, label.box.left);
        const overlapY = Math.min(box.bottom, label.box.bottom) - Math.max(box.top, label.box.top);
        if (overlapX > 1 && overlapY > 1) {
          findings.push({
            label: label.text,
            mark: mark.getAttribute('class') || mark.tagName,
            overlapPx: Number(Math.min(overlapX, overlapY).toFixed(2)),
          });
        }
      }
    }
  }
  return { findings, marksInspected };
}

/** Runs in the page. Reports any table cell clipped by its own scroll box at a
 *  width where the table is NOT meant to scroll. A figure table that clips a
 *  column at desktop is a figure that has lost data, not a figure that scrolls. */
function clippedTableColumns(root) {
  const findings = [];
  for (const box of document.querySelectorAll(`${root} .ts-table-scroll`)) {
    const table = box.querySelector('table');
    if (!table) continue;
    if (table.scrollWidth > box.clientWidth + 1) {
      findings.push({
        caption: normalizeText(box.getAttribute('aria-labelledby')),
        tableWidth: table.scrollWidth, boxWidth: box.clientWidth,
      });
    }
  }
  function normalizeText(id) {
    const label = id ? document.getElementById(id) : null;
    return label ? label.textContent.trim().slice(0, 80) : '(unlabelled)';
  }
  return findings;
}

/** The shared layout inspector returns TRIAGE: a grid line crossing a label, or
 * a label deliberately inside its own node, is not automatically a defect. So
 * its output is not failed on blindly -- it is reduced to a flat list of
 * findings and compared against a reviewed allow-list.
 *
 * The allow-list is empty. Every candidate it reported on this lesson was a
 * real defect and was fixed: an "origin" label one unit past the foot of its
 * own viewBox and sitting on the "h1" caption beneath it, an axis title printed
 * over the last tick label, and a final axis tick landing one day after a
 * regular one so that two numbers overlapped. Recording the candidates without
 * asserting on them, which is what this file did first, is how the next one
 * would have gone unnoticed.
 */
const ACCEPTED_LAYOUT_CANDIDATES = [];
const flattenLayout = report => (report ?? []).flatMap(entry =>
  (entry.issues ?? []).map(issue => ({
    svgIndex: entry.svgIndex,
    type: issue.type,
    labels: (issue.labels ?? [issue.label]).filter(Boolean).join(' + '),
  })));

(async () => {
  const startedAt = new Date().toISOString();
  const sourceHashes = Object.fromEntries(ownedFiles.map(filename => [filename, hash(filename)]));
  const build = read(`${distDir}/.vite/manifest.json`);
  const buildHash = hash(`${distDir}/.vite/manifest.json`);
  const publications = read('src/learn/data/lesson-manifest.json');
  const models = await import('../src/learn/data/timeseries-models.js');
  const { timeSeriesData } = await import('../src/learn/data/timeseries-data.js');
  const { timeSeriesExamples } = await import('../src/learn/data/timeseries-examples.js');

  assert.ok(publications[topicId], `${topicId} is not registered in lesson-manifest.json`);

  /* The independently computed answers this run pins the page against. Nothing
     here is read off the screen. */
  const fixedText = (value, digits = 6) => value.toFixed(digits);
  const donor = models.donorAnswer({
    history: models.fixtures.toy.history, period: 2, horizonCount: 4, selectedHorizon: 3,
    future: models.fixtures.toy.future,
  });
  const eligibility = models.eligibilityAnswer({
    origins: models.fixtures.eligibility.origins, cutoff: 12, horizon: 3, delay: 2, selected: [6, 7],
  });
  const request = models.requestAnswer({
    counts: timeSeriesData.series.counts, dates: timeSeriesData.series.dates,
    origin: 364, period: 7, horizon: 7,
  });
  const requestSeasonalMae = fixedText(request.seasonalMae);
  const requestNaiveMae = fixedText(request.naiveMae);
  const donorValueText = fixedText(donor.value, 4);
  const donorMaeText = fixedText(donor.summaries.seasonal.mae);

  const bodyFiles = new Set(Object.values(build)
    .filter(entry => entry.src && entry.src.includes('/data/topics/'))
    .map(entry => entry.file));
  const bodyEntry = Object.values(build).find(entry => entry.src && entry.src.endsWith(`${topicId}.jsx`));
  assert.ok(bodyEntry, `${topicId}.jsx is not in the production manifest; the lesson was not built`);
  const bodyFile = bodyEntry.file;
  const allowedScripts = new Set();
  /* Walk by manifest KEY. `imports` lists keys, not built filenames, so a
     lookup that only matched `src` or `file` silently found nothing for every
     shared chunk and left legitimate files looking unexpected. */
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
  const leakChecks = [];
  const layoutByWidth = {};
  const radicalsByWidth = {};

  /* EVERY ABSENCE ASSERTION GETS A PRESENCE PARTNER, in one call, so an
     unpaired one is structurally impossible to write.

     This exists because of a defect in this very file. The outcome pin read
     `<count with separators> rentals` -- a string the page never renders
     anywhere, because the chart description writes the count without a
     thousands separator and the outcome table writes it without the word. The
     absence assertion therefore passed on every run and proved nothing: it
     could not tell "correctly withheld" from "looking in the wrong place".
     Only a paired presence assertion catches that, so pairing is now the only
     way to express a pin here. */
  /* O5: the array is CLOSED OVER. It was an ordinary `const pins = []` in this
     scope, so `pins.push(...)` remained reachable from anywhere in the function
     and the record's claim that an unpaired absence "cannot be expressed" was
     stronger than the code enforced. Only the two helpers and a reader are
     exposed now, so the claim is literally true. */
  const { pinned, pinnedThroughTwoGates, pinRecords } = (() => {
  const pins = [];
  const pinned = (before, after, needle, label) => {
    assert.ok(!before.includes(needle), `${label}: "${needle}" is rendered BEFORE it should be`);
    assert.ok(after.includes(needle),
      `${label}: "${needle}" never appears even after it should, so its absence assertion proves nothing -- `
      + 'the pin is looking for text this page never renders');
    pins.push({ label, needle, absentBefore: true, presentAfter: true });
  };
  /* The two-gate form: absent at the start, STILL absent after the forecast is
     drawn, present only once the outcomes are asked for. */
  const pinnedThroughTwoGates = (before, middle, after, needle, label) => {
    assert.ok(!before.includes(needle), `${label}: "${needle}" is rendered before anything is committed`);
    assert.ok(!middle.includes(needle),
      `${label}: "${needle}" is rendered at the moment the forecast is issued`);
    assert.ok(after.includes(needle),
      `${label}: "${needle}" never appears even after the outcomes are revealed, so neither absence `
      + 'assertion proves anything');
    pins.push({ label, needle, absentBefore: true, absentBetweenGates: true, presentAfterReveal: true });
  };
  return { pinned, pinnedThroughTwoGates, pinRecords: () => pins.slice() };
  })();

  /** Runs in the page. Measures every KaTeX radical in BOTH dimensions.
   *
   * A sibling lesson's guard measured width only. A collapsed radical is also
   * narrow, so 24 of them survived a fully green run and that page published a
   * bound as its own radicand. */
  const measureRadicals = () => [...document.querySelectorAll('.ts-lesson .katex .sqrt svg')]
    .map(svg => {
      const box = svg.getBoundingClientRect();
      return { height: Math.round(box.height), width: Math.round(box.width) };
    });
  const checkText = async (locator, pattern) => assert.match(normalize(await locator.innerText()), pattern);
  const settle = page => page.evaluate(() =>
    new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  let status = 'failed';
  let failure = null;
  try {
    const context = await browser.newContext({ viewport: { width: 1366, height: 1000 } });
    const page = await context.newPage();
    const requests = [], errors = [], failedAssets = [];
    page.on('request', item => requests.push(item.url()));
    page.on('pageerror', error => errors.push(error.message));
    page.on('requestfailed', item => failedAssets.push(item.url()));
    const ready = async () => {
      await page.locator('.ts-lesson').waitFor();
      await page.evaluate(() => document.fonts.ready);
      await settle(page);
    };
    await page.goto(route, { waitUntil: 'domcontentloaded' });
    await ready();

    /** Unique path, unique content, and a recorded {file, digest, bytes}. That
     *  triple is how a stale orphan from a pre-fix run gets caught. */
    const screenshot = async (locator, filename) => {
      const destination = path.join(screenshotDirectory, filename);
      fs.mkdirSync(path.dirname(destination), { recursive: true });
      const viewport = page.viewportSize();
      const bounds = await locator.boundingBox();
      if (bounds && bounds.height > viewport.height - 160) {
        await page.setViewportSize({ width: viewport.width, height: Math.ceil(bounds.height) + 180 });
      }
      await locator.scrollIntoViewIfNeeded();
      await locator.evaluate(element =>
        window.scrollTo(0, element.getBoundingClientRect().top + window.scrollY - 100));
      await locator.screenshot({ path: destination });
      await page.setViewportSize(viewport);
      screenshots.push({ file: destination, digest: hash(destination), bytes: fs.statSync(destination).size });
    };
    const textOf = async index => normalize(await page.locator('.ts-investigation').nth(index).innerText());

    // ------------------------------------------------------------ 1. structure
    await checkText(page.locator('.reader-header h1'), /^Time-Series Validation & Forecasting Baselines$/);
    assert.equal(await page.locator('.ts-investigation').count(), 3);
    assert.equal(await page.locator('.ts-figure').count(), 5);
    assert.equal(await page.locator('.ts-practice').count(), 8);
    assert.equal(await page.locator('.python-example').count(), 3);
    assert.equal(await page.locator('.katex-error').count(), 0);
    const rendered = normalize(await page.locator('.ts-lesson').textContent());
    // Every measured headline number the manuscript states must be on the page.
    for (const row of timeSeriesData.development) {
      assert.ok(rendered.includes(row.mae.toFixed(2)),
        `the development MAE for ${row.key} (${row.mae.toFixed(2)}) is not rendered`);
    }
    for (const row of timeSeriesData.final) {
      assert.ok(rendered.includes(row.mae.toFixed(2)),
        `the final MAE for ${row.key} (${row.mae.toFixed(2)}) is not rendered`);
    }
    // The executed programs' real output, byte for byte.
    for (const key of ['baseline-rules', 'eligible-rows', 'baseline-loop']) {
      const example = timeSeriesExamples[key];
      assert.ok(normalize(rendered).includes(normalize(example.expected).slice(0, 80)),
        `the recorded output of ${key} is not rendered on the page`);
    }
    records.push({ case: 'Header, three investigations, five figures, eight practice tasks, three executed programs, every measured headline number, and no KaTeX error' });

    // ------------------------------------------- 2. the served dataset is ours
    const dataset = await page.request.get(`${base}${timeSeriesData.provenance.file}`);
    assert.equal(dataset.status(), 200, 'the lesson’s own dataset is not served');
    const bytes = Buffer.from(await dataset.body());
    assert.equal(bytes.length, timeSeriesData.provenance.bytes, 'the served file is not the packet file');
    assert.equal(createHash('sha256').update(bytes).digest('hex'), timeSeriesData.provenance.sha256);
    const attribution = await page.request.get(`${base}${timeSeriesData.provenance.attribution}`);
    assert.equal(attribution.status(), 200);
    assert.ok((await attribution.text()).includes('CC BY 4.0'));
    const program = await page.request.get(`${base}${timeSeriesData.provenance.program}`);
    assert.equal(program.status(), 200, 'the author program is actually served');
    assert.ok((await program.text()).includes('def baseline_forecasts'),
      'and is the program rather than a placeholder');
    records.push({ case: 'The lesson serves its own copy of the daily CSV at its recorded digest, with its attribution and the whole author program' });

    // ------------------------------------------- 3. nothing revealed on paint
    for (let index = 0; index < 3; index += 1) {
      const investigation = page.locator('.ts-investigation').nth(index);
      assert.equal(await investigation.locator('input[type="radio"]:checked').count(), 0,
        `investigation ${index + 1} has a prediction preselected`);
      assert.equal(await investigation.locator('input[type="checkbox"]:checked').count(), 0,
        `investigation ${index + 1} has a commitment preselected`);
      assert.equal(await investigation.locator('.ts-verdict').count(), 0,
        `investigation ${index + 1} shows a verdict before any prediction`);
      assert.equal(await investigation.locator('.ts-numeric-verdict').count(), 0,
        `investigation ${index + 1} shows numeric feedback before any prediction`);
      assert.equal(await investigation.locator('.ts-reveal').count(), 0,
        `investigation ${index + 1} shows its reveal before any prediction`);
      assert.ok(await investigation.getByRole('button', { name: /^(Apply|Record|Issue)/ }).first().isDisabled(),
        `investigation ${index + 1} can be committed without recording anything`);
    }
    records.push({ case: 'All three investigations open with nothing selected, no verdict, no numeric feedback, no reveal, and a disabled commit' });

    // ----------------------- 4. the leak property, pinned numerically, per lab
    // -- Investigation 1: the graded quantities are the donor position and the
    //    seasonal forecast at the inspected horizon.
    const one = page.locator('.ts-investigation').nth(0);
    const oneBefore = await textOf(0);
    // The learner's own inputs ARE on screen: they are what the question is
    // about, and a lab that hid them would be asking about nothing.
    assert.ok(oneBefore.includes('history 1'), 'investigation 1 must show the editable history it asks about');
    await screenshot(one, 'timeseries-donor-initial-desktop.png');
    await one.getByLabel(`position ${donor.donorPosition}`, { exact: false }).first().check();
    await one.locator('.ts-numeric-guess input').fill(String(donor.value));
    await one.getByRole('button', { name: 'Apply and reveal the donor' }).click();
    await settle(page);
    const oneAfter = await textOf(0);
    pinned(oneBefore, oneAfter, donorValueText, 'investigation 1, the graded horizon-3 forecast');
    pinned(oneBefore, oneAfter, donorMaeText, 'investigation 1, the seasonal MAE');
    /* `h3 ← pos 5` is a donor-strip CELL, not the strip's aria-label. The first
       draft pinned "The seasonal donor for each horizon", which is an attribute
       and never appears in rendered text -- so its absence assertion passed on
       every run and proved nothing. The paired presence assertion caught it
       immediately, which is the whole reason pairing is mandatory here. */
    for (const phrase of ['Depends on', `h3 ← pos ${donor.donorPosition}`, 'lands on position']) {
      pinned(oneBefore, oneAfter, phrase, `investigation 1, the phrase "${phrase}"`);
    }
    await checkText(one.locator('.ts-verdict'), /Your prediction matches/);
    leakChecks.push({ investigation: 1, pinnedValues: [donorValueText, donorMaeText] });
    await screenshot(one, 'timeseries-donor-committed-desktop.png');

    // -- Investigation 2: the graded quantity is the exact eligible set.
    const two = page.locator('.ts-investigation').nth(1);
    const twoBefore = await textOf(1);
    const eligibleText = eligibility.eligible.join(', ');
    await screenshot(two, 'timeseries-eligibility-initial-desktop.png');
    for (const origin of eligibility.eligible) {
      await two.getByRole('checkbox', { name: `origin ${origin}`, exact: true }).check();
    }
    await two.getByRole('button', { name: 'Record this set and check it' }).click();
    await settle(page);
    const twoAfter = await textOf(1);
    pinned(twoBefore, twoAfter, eligibleText, 'investigation 2, the eligible set');
    /* Concrete rows of the per-origin arithmetic, one admitted and one refused,
       rather than a fragment assembled from a template. An earlier draft pinned
       "= 12 ≤", which is not a string this table renders as a unit. */
    for (const phrase of ['Label arrives', 'Rows a fit could use', 'Wrongly included',
      '6 + 3 + 2 = 11 ≤ 12', '8 + 3 + 2 = 13 > 12']) {
      pinned(twoBefore, twoAfter, phrase, `investigation 2, the phrase "${phrase}"`);
    }
    await checkText(two.locator('.ts-verdict'), /Your prediction matches/);
    leakChecks.push({ investigation: 2, pinnedValues: [eligibleText, 'Rows a fit could use'] });
    await screenshot(two, 'timeseries-eligibility-committed-desktop.png');

    // The empty eligible set is a RESULT the lab must be able to reach and
    // grade, not a state it refuses. This is the degenerate case that matters
    // most for this topic.
    await two.getByRole('button', { name: 'Reset' }).click();
    await settle(page);
    await two.getByRole('button', { name: 'Cutoff 10, h 3, delay 2' }).click();
    await two.getByRole('checkbox', { name: 'none of them qualify' }).check();
    await two.getByRole('button', { name: 'Record this set and check it' }).click();
    await settle(page);
    await checkText(two.locator('.ts-verdict'), /Your prediction matches/);
    await checkText(two, /empty/i);
    await screenshot(two, 'timeseries-eligibility-empty-set-desktop.png');
    records.push({ case: 'The empty eligible set is reachable, is graded correct when predicted, and is reported as empty rather than as a broken state' });

    // -- Investigation 3: TWO gates. The forecast appears at the first, the
    //    outcomes and the comparison only at the second.
    const three = page.locator('.ts-investigation').nth(2);
    const threeBefore = await textOf(2);
    await screenshot(three, 'timeseries-request-initial-desktop.png');
    await three.getByLabel('higher than naive', { exact: true }).check();
    await three.locator('.ts-numeric-guess input').fill(String(request.value));
    await three.getByRole('button', { name: 'Issue the forecast' }).click();
    await settle(page);
    const threeIssued = await textOf(2);
    // GATE 1: the forecast is drawn, and was not drawn before.
    pinned(threeBefore, threeIssued, 'Copied from', 'investigation 3 gate 1, the issued forecast table');
    pinned(threeBefore, threeIssued, 'index 358', 'investigation 3 gate 1, the donor the rule copied');
    assert.equal(await three.locator('.ts-verdict').count(), 0,
      'investigation 3 grades the comparison before the outcomes are revealed');
    assert.ok(threeIssued.includes('the outcomes are not'),
      'and does not say that the outcomes are being withheld');
    await screenshot(three, 'timeseries-request-issued-desktop.png');
    await three.getByRole('button', { name: 'Reveal the outcomes' }).click();
    await settle(page);
    const threeRevealed = await textOf(2);
    // GATE 2: the outcomes and the comparison, absent at BOTH earlier states.
    pinnedThroughTwoGates(threeBefore, threeIssued, threeRevealed, requestSeasonalMae,
      'investigation 3 gate 2, the graded seasonal MAE');
    pinnedThroughTwoGates(threeBefore, threeIssued, threeRevealed, requestNaiveMae,
      'investigation 3 gate 2, the naive MAE');
    /* The outcomes themselves, in the exact form the outcome table renders.
       An earlier draft appended the word "rentals" to these, which the page
       never renders in that form anywhere, so BOTH absence assertions passed on
       a string that could never have appeared. */
    for (const actual of request.request.actual) {
      pinnedThroughTwoGates(threeBefore, threeIssued, threeRevealed,
        Math.round(actual).toLocaleString('en-US'),
        `investigation 3 gate 2, the outcome ${actual}`);
    }
    await checkText(three.locator('.ts-verdict'), /Your prediction matches/);
    leakChecks.push({
      investigation: 3, gates: 2,
      pinnedValues: [requestSeasonalMae, requestNaiveMae,
        ...request.request.actual.map(value => Math.round(value).toLocaleString('en-US'))],
    });
    await screenshot(three, 'timeseries-request-revealed-desktop.png');

    /* THE REVEALED STATE IS ITS OWN LAYOUT. The widest table on this page --
       seven target days scored against two rules -- exists ONLY here, and the
       geometry checks further down run with investigation 3 reset, so they
       never saw it. It clipped its last column and no check noticed; only the
       screenshot did. These three now run in the state that has the most on
       screen. */
    const revealedClipped = await page.evaluate(clippedTableColumns, '.ts-lesson');
    assert.deepEqual(revealedClipped, [],
      'a table clips a column once investigation 3 has revealed its outcomes');
    const revealedOutside = await page.evaluate(labelsOutsideTheirSvg, '.ts-lesson');
    assert.deepEqual(revealedOutside.findings, [],
      'a label is painted past the edge of its own SVG in the revealed state');
    const revealedMarks = await page.evaluate(labelsOverPointMarks, '.ts-lesson');
    assert.deepEqual(revealedMarks.findings, [],
      'a label sits on top of a point mark in the revealed state');
    records.push({
      case: 'The revealed state is checked as its own layout: no table clips a column, no label leaves its '
        + 'own SVG and no label sits on a point mark once investigation 3 has drawn its outcomes',
      labelsMeasuredInRevealedState: revealedOutside.labelsMeasured,
      pointMarksCheckedInRevealedState: revealedMarks.marksInspected,
    });
    records.push({
      case: 'Every investigation withholds its graded quantity until a prediction is committed, pinned by the '
        + 'exact text of that quantity computed in this process; investigation 3 additionally withholds the '
        + 'outcomes after the forecast is drawn',
      pins: leakChecks,
    });

    // ------------------------------------- 5. a verdict retires when an input changes
    await three.getByRole('button', { name: 'Reset' }).click();
    await settle(page);
    assert.equal(await three.locator('.ts-verdict').count(), 0, 'Reset did not retire the verdict');
    assert.equal(await three.locator('input[type="radio"]:checked').count(), 0,
      'Reset did not clear the recorded prediction');
    // And a preset fills inputs only, never the prediction.
    await three.getByRole('button', { name: 'A week later, 2012-01-07' }).click();
    await settle(page);
    assert.equal(await three.locator('input[type="radio"]:checked').count(), 0,
      'a suggested setup filled in the prediction as well as the inputs');
    assert.equal(await three.locator('.ts-verdict').count(), 0, 'a suggested setup revealed an answer');
    await screenshot(three, 'timeseries-request-preset-desktop.png');
    records.push({ case: 'Reset retires the verdict and clears the prediction; a suggested setup fills inputs only' });

    // ------------------------------- 6. the KaTeX radicals actually paint
    /* The single most damaging defect this effort has produced: a root-scoped
       `svg { height: auto }` rule also matches KaTeX's radical SVGs, whose
       height comes from `height: inherit`, and collapses them to nothing. The
       DOM stays right, there are zero .katex-error nodes, and the formula on
       screen becomes a DIFFERENT QUANTITY. Only a measurement on the rendered
       page can see it. Details are opened first: a practice solution carries a
       radical too, and a hidden element measures zero for reasons that have
       nothing to do with this defect.

       BOTH dimensions are measured. A check that measured only width would
       miss a collapsed radical, which is also narrow. */
    await page.locator('.ts-lesson details').evaluateAll(items => items.forEach(item => { item.open = true; }));
    await settle(page);
    const radicals = await page.evaluate(measureRadicals);
    radicalsByWidth[1366] = radicals;
    assert.ok(radicals.length >= 2,
      `only ${radicals.length} square-root radicals were found; the page renders two`);
    // BOTH dimensions, floored at 4px.
    assert.deepEqual(radicals.filter(entry => entry.height < 4 || entry.width < 4), [],
      'a square-root radical paints with no height or width, so a formula reads as a bare fraction');
    // And the lesson's own diagrams DO get the scoped rule, or the scoping was
    // achieved by giving up the layout.
    const diagramSizes = await page.evaluate(() => [...document.querySelectorAll('.ts-lesson svg.ts-diagram')]
      .map(svg => {
        const box = svg.getBoundingClientRect();
        return { height: Math.round(box.height), width: Math.round(box.width) };
      }));
    assert.ok(diagramSizes.length >= 6, `only ${diagramSizes.length} tagged diagrams were found`);
    assert.deepEqual(diagramSizes.filter(entry => entry.height < 20 || entry.width < 60), [],
      'a lesson diagram painted at almost no size');
    await page.locator('.ts-lesson details').evaluateAll(items => items.forEach(item => { item.open = false; }));
    await settle(page);
    records.push({
      case: 'Every square-root radical paints with real width AND height with the practice solutions opened, '
        + 'and every tagged diagram receives the scoped layout rule',
      radicalsMeasured: radicals.length,
      smallestRadical: radicals.reduce((least, entry) => Math.min(least, entry.height), Infinity),
      diagramsMeasured: diagramSizes.length,
    });

    // ------------------------------------------- 7. painted against declared
    const painted = await page.evaluate(comparePaintedAgainstDeclared, '.ts-lesson');
    assert.deepEqual(painted, [], 'a shape paints differently from what it declares, or paints black on black');
    const curves = await page.evaluate(sampleCurvesThroughLabels, '.ts-lesson');
    assert.ok(curves.shapesInspected >= 8,
      `the curve sampler inspected only ${curves.shapesInspected} shapes; its selector is not matching what `
      + 'this lesson draws');
    assert.deepEqual(curves.findings, [], 'a curve runs through a label at 1366px');
    const clipped = await page.evaluate(clippedTableColumns, '.ts-lesson');
    assert.deepEqual(clipped, [], 'a table clips a column at desktop width');
    const outside = await page.evaluate(labelsOutsideTheirSvg, '.ts-lesson');
    assert.ok(outside.labelsMeasured >= 80,
      `the edge check measured only ${outside.labelsMeasured} labels; its selector is not matching`);
    assert.deepEqual(outside.findings, [], 'a label is painted past the edge of its own SVG at 1366px');
    const legends = await page.evaluate(legendEncodingsPainted, '.ts-lesson');
    assert.ok(legends.encodingsChecked >= 18,
      `only ${legends.encodingsChecked} legend encodings were checked; the selector is not matching`);
    assert.deepEqual(legends.findings, [],
      'a figure legend names an encoding the painted picture does not contain');
    /* The HATCH specifically must clear 3:1 against whatever it marks. It is the
       redundant channel that distinguishes observed from unobserved, and with it
       gone the fill difference is 1.30:1 and the stroke difference 2.06:1 -- both
       under the floor for non-text contrast. The other encodings' contrast is
       measured and recorded but NOT asserted: a known cell painted over the known
       band is the same region twice by design and is ~1.2:1 legitimately, so a
       blanket floor would be a guard nobody could keep green. */
    const hatches = legends.measured.filter(entry => entry.key === 'unknown'
      && entry.contrastAgainstWhatIsUnderIt !== null);
    assert.ok(hatches.length >= 2,
      `only ${hatches.length} hatch encodings were measured against a backdrop; this floor exists so the `
      + 'contrast assertion below cannot pass by filtering an empty list');
    assert.deepEqual(hatches.filter(entry => entry.contrastAgainstWhatIsUnderIt < 3), [],
      'a hatch marks a region at under 3:1 against it, so the redundant channel is not resolvable');
    const arrows = await page.evaluate(arrowStylesMatchTheirInput,
      { root: '.ts-lesson', styleByInputKind: models.ARROW_STYLE_BY_INPUT_KIND });
    assert.ok(arrows.checked >= 20,
      `only ${arrows.checked} arrows carry an input kind; the selector is not matching`);
    assert.deepEqual(arrows.findings, [],
      'an arrow is drawn in a style that does not match the input it carries');
    const overlaid = await page.evaluate(labelsOverPointMarks, '.ts-lesson');
    assert.ok(overlaid.marksInspected >= 40,
      `the point-mark check inspected only ${overlaid.marksInspected} marks; its selector is not matching `
      + 'what this lesson draws');
    assert.deepEqual(overlaid.findings, [], 'a label sits on top of a point mark at 1366px');
    /* The shared inspector is a PAGE-SIDE function: it is passed to
       page.evaluate, not called in this process. It reads straight `line`
       elements only, which is why the curve sampler above runs too. */
    const layout = await page.evaluate(inspectLessonVisualLayout, '.ts-lesson');
    const layoutFindings = flattenLayout(layout);
    assert.deepEqual(
      layoutFindings.filter(finding => !ACCEPTED_LAYOUT_CANDIDATES.some(accepted =>
        accepted.type === finding.type && accepted.labels === finding.labels)),
      [], 'the shared layout inspector reports a candidate that is not on the reviewed allow-list');
    records.push({
      case: 'Computed paint matches every declared attribute, no curve crosses a label, no table clips a '
        + 'column at desktop, and the shared layout inspector reports its candidates',
      shapesSampledForCurveCrossings: curves.shapesInspected,
      pointMarksCheckedAgainstLabels: overlaid.marksInspected,
      labelsMeasuredAgainstSvgEdges: outside.labelsMeasured,
      legendEncodingsChecked: legends.encodingsChecked,
      legendEncodingContrast: legends.measured,
      arrowsCheckedAgainstTheirInputKind: arrows.checked,
      layoutCandidates: layout,
      layoutFindings,
      acceptedLayoutCandidates: ACCEPTED_LAYOUT_CANDIDATES,
    });

    // ----------------------------------------------- 8. every figure captured
    for (let index = 0; index < 5; index += 1) {
      await screenshot(page.locator('.ts-figure').nth(index), `timeseries-figure-${index + 1}-desktop.png`);
    }
    // The informative states, not the default one.
    const continuation = page.locator('.ts-figure').nth(2);
    await continuation.getByRole('button', { name: 'outcomes 22, 24, 26, 28' }).click();
    await settle(page);
    await screenshot(continuation, 'timeseries-figure-3-changed-continuation-desktop.png');
    const staircase = page.locator('.ts-figure').nth(3);
    await staircase.getByRole('button', { name: /^sliding/ }).click();
    await settle(page);
    await screenshot(staircase, 'timeseries-figure-4-sliding-desktop.png');
    await staircase.getByRole('button', { name: 'h = 7' }).click();
    await settle(page);
    await screenshot(staircase, 'timeseries-figure-4-horizon-7-desktop.png');
    await staircase.getByRole('button', { name: 'Advance the origin' }).click();
    await staircase.getByRole('button', { name: 'Advance the origin' }).click();
    await settle(page);
    await screenshot(staircase, 'timeseries-figure-4-third-origin-desktop.png');
    const comparison = page.locator('.ts-figure').nth(4);
    /* Figure 5 OPENS in the final view, so the default capture above is that
       view; capturing it again after clicking `final` produced a byte-identical
       file, which the uniqueness check refused. That identity is worth
       something, though, so it is asserted as a round trip instead of being
       photographed twice. */
    const finalViewText = normalize(await comparison.innerText());
    await comparison.getByRole('button', { name: 'development: choose' }).click();
    await settle(page);
    const developmentViewText = normalize(await comparison.innerText());
    assert.notEqual(developmentViewText, finalViewText, 'the comparison toggle changes nothing');
    assert.ok(developmentViewText.includes('Development results'), 'the development view is not shown');
    await screenshot(comparison, 'timeseries-figure-5-development-desktop.png');
    await comparison.getByRole('button', { name: 'final: assess' }).click();
    await settle(page);
    assert.equal(normalize(await comparison.innerText()), finalViewText,
      'toggling away from the final view and back does not restore it');
    records.push({ case: 'Every figure captured at desktop, plus the informative states: the changed continuation, the sliding window, horizon 7, the third rehearsal, and the development comparison view; the final view is the figure’s opening state and is asserted to round-trip rather than captured twice' });

    // ------------------------------------------------- 9. loading closure
    const localScripts = [...new Set(requests.map(address => new URL(address))
      .filter(address => address.origin === new URL(base).origin && address.pathname.endsWith('.js'))
      .map(address => address.pathname.slice(1)))];
    assert.deepEqual(localScripts.filter(filename => bodyFiles.has(filename)), [bodyFile],
      'the page loaded another lesson’s body');
    assert.deepEqual(localScripts.filter(filename => !allowedScripts.has(filename)), []);
    assert.ok(!requests.some(address => address.includes('/outlines/')));
    for (const sibling of ['learn-assets/pac-learning', 'learn-assets/evaluation-metrics',
      'learn-assets/regularization', 'learn-assets/semi-supervised-learning']) {
      assert.ok(!requests.some(address => address.includes(sibling)),
        `the page requested ${sibling}, which belongs to another lesson`);
    }
    assert.deepEqual(errors, [], 'the page raised a runtime error');
    assert.deepEqual(failedAssets.filter(address => !address.includes('favicon')), [],
      'an asset failed to load');
    records.push({
      case: 'Fresh route requests only the selected lesson and its shared closure, never another lesson’s dataset, with no runtime error',
      requestedScripts: localScripts,
      rawScriptBytes: localScripts.reduce((sum, filename) => sum + fs.statSync(path.join(distDir, filename)).size, 0),
      gzipEstimateBytes: localScripts.reduce((sum, filename) => sum + gzipSync(fs.readFileSync(path.join(distDir, filename))).length, 0),
      interpretation: 'Built requested file sizes; the gzip estimate is not measured network compression or latency.',
    });

    // ------------------------------------------------------- 10. narrow widths
    for (const width of [390, 320]) {
      await page.setViewportSize({ width, height: 1400 });
      await page.reload({ waitUntil: 'domcontentloaded' });
      await ready();
      const overflow = await page.evaluate(() => ({
        documentWidth: document.documentElement.scrollWidth,
        viewport: window.innerWidth,
      }));
      assert.ok(overflow.documentWidth <= overflow.viewport + 1,
        `the page scrolls sideways at ${width}px: ${overflow.documentWidth} against ${overflow.viewport}`);
      // KaTeX display math must not overflow, measured on the real page.
      const mathOverflow = await page.locator('.ts-lesson .katex-display').evaluateAll(items =>
        items.map(item => ({ text: item.textContent.trim().slice(0, 50), width: item.scrollWidth,
          box: item.clientWidth })).filter(entry => entry.width > entry.box + 1));
      assert.deepEqual(mathOverflow, [], `display math overflows its column at ${width}px`);
      assert.equal(await page.locator('.katex-error').count(), 0);
      /* The radicals again at this width, in BOTH dimensions, and the COUNT
         compared with the desktop count. Without the count check a narrow
         layout that rendered no radicals at all -- a changed selector, a
         details element that stopped opening -- would pass an emptiness test
         silently, which is how this class of guard goes inert. */
      await page.locator('.ts-lesson details').evaluateAll(items =>
        items.forEach(item => { item.open = true; }));
      await settle(page);
      const narrowRadicals = await page.evaluate(measureRadicals);
      radicalsByWidth[width] = narrowRadicals;
      assert.equal(narrowRadicals.length, radicalsByWidth[1366].length,
        `${width}px renders ${narrowRadicals.length} square-root radicals against `
        + `${radicalsByWidth[1366].length} at desktop; the check would be measuring a different set`);
      assert.deepEqual(narrowRadicals.filter(entry => entry.height < 4 || entry.width < 4), [],
        `a square-root radical paints with no height or width at ${width}px`);
      await page.locator('.ts-lesson details').evaluateAll(items =>
        items.forEach(item => { item.open = false; }));
      await settle(page);
      const narrowCurves = await page.evaluate(sampleCurvesThroughLabels, '.ts-lesson');
      assert.deepEqual(narrowCurves.findings, [], `a curve runs through a label at ${width}px`);
      const narrowPaint = await page.evaluate(comparePaintedAgainstDeclared, '.ts-lesson');
      assert.deepEqual(narrowPaint, [], `a shape paints wrongly at ${width}px`);
      const narrowMarks = await page.evaluate(labelsOverPointMarks, '.ts-lesson');
      assert.deepEqual(narrowMarks.findings, [], `a label sits on top of a point mark at ${width}px`);
      const narrowOutside = await page.evaluate(labelsOutsideTheirSvg, '.ts-lesson');
      assert.deepEqual(narrowOutside.findings, [],
        `a label is painted past the edge of its own SVG at ${width}px`);
      const narrowLegends = await page.evaluate(legendEncodingsPainted, '.ts-lesson');
      assert.equal(narrowLegends.encodingsChecked, legends.encodingsChecked,
        `${width}px checks ${narrowLegends.encodingsChecked} legend encodings against `
        + `${legends.encodingsChecked} at desktop; the check would be looking at a different set`);
      assert.deepEqual(narrowLegends.findings, [],
        `a figure legend names an encoding the painted picture does not contain at ${width}px`);
      const narrowArrows = await page.evaluate(arrowStylesMatchTheirInput,
        { root: '.ts-lesson', styleByInputKind: models.ARROW_STYLE_BY_INPUT_KIND });
      assert.deepEqual(narrowArrows.findings, [],
        `an arrow is drawn in a style that does not match its input at ${width}px`);
      const narrowLayout = await page.evaluate(inspectLessonVisualLayout, '.ts-lesson');
      layoutByWidth[width] = narrowLayout;
      const narrowFindings = flattenLayout(narrowLayout);
      assert.deepEqual(
        narrowFindings.filter(finding => !ACCEPTED_LAYOUT_CANDIDATES.some(accepted =>
          accepted.type === finding.type && accepted.labels === finding.labels)),
        [], `the shared layout inspector reports an unreviewed candidate at ${width}px`);
      for (let index = 0; index < 5; index += 1) {
        await screenshot(page.locator('.ts-figure').nth(index), `timeseries-figure-${index + 1}-${width}.png`);
      }
      for (let index = 0; index < 3; index += 1) {
        await screenshot(page.locator('.ts-investigation').nth(index),
          `timeseries-investigation-${index + 1}-${width}.png`);
      }
      records.push({
        case: `No sideways scroll, no display-math overflow, no curve through a label, and correct paint at ${width}px`,
        layoutCandidates: narrowLayout,
        layoutFindings: narrowFindings,
      });
    }
    await page.setViewportSize({ width: 1366, height: 1000 });

    // ------------------------------------------------ 11. the screenshots
    assert.equal(new Set(screenshots.map(shot => shot.file)).size, screenshots.length,
      'two captures were written to the same path');
    assert.equal(new Set(screenshots.map(shot => shot.digest)).size, screenshots.length,
      'two captures have identical content, so one of them is not the state it claims to be');
    /* No orphan may survive. A capture left behind by an earlier, pre-fix run
       is exactly the stale evidence the digests exist to catch, so every
       timeseries-*.png on disk must be one this run wrote, and every file this
       run recorded must still exist at the digest and size recorded. */
    const onDisk = fs.readdirSync(screenshotDirectory)
      .filter(name => name.startsWith('timeseries-') && name.endsWith('.png'))
      .map(name => path.join(screenshotDirectory, name).replace(/\\/g, '/'));
    const written = new Set(screenshots.map(shot => shot.file.replace(/\\/g, '/')));
    assert.deepEqual(onDisk.filter(file => !written.has(file)), [],
      'orphaned timeseries-*.png files remain from an earlier run');
    for (const shot of screenshots) {
      assert.ok(fs.existsSync(shot.file), `${shot.file} was recorded but is not on disk`);
      assert.equal(hash(shot.file), shot.digest, `${shot.file} no longer matches its recorded digest`);
      assert.ok(shot.bytes > 1000, `${shot.file} is only ${shot.bytes} bytes, which is not a rendered figure`);
    }
    assert.ok(screenshots.length >= 28, `only ${screenshots.length} captures were taken`);
    records.push({ case: `${screenshots.length} captures, each at a unique path with unique content and a recorded digest, with no orphan left from an earlier run` });

    status = 'passed';
  } catch (error) {
    failure = error?.stack ?? String(error);
    throw error;
  } finally {
    await browser.close();
    fs.mkdirSync('docs/teaching/evidence', { recursive: true });
    fs.writeFileSync(evidencePath, JSON.stringify({
      checkedAt: startedAt,
      completedAt: new Date().toISOString(),
      verifier: 'scripts/verify-timeseries-browser.cjs',
      verifierSha256: hash('scripts/verify-timeseries-browser.cjs'),
      stage: 'production browser review; independent and integration review are separate',
      route,
      distDir,
      buildManifestSha256: buildHash,
      sourceHashes,
      widths: [1366, 390, 320],
      layoutInspectorByWidth: layoutByWidth,
      cases: records,
      radicalsByWidth,
      radicalPolicy: 'Every .katex .sqrt svg is measured in BOTH dimensions and floored at 4px, at all three '
        + 'widths, and the count at each narrow width must equal the desktop count so the check cannot go '
        + 'inert by measuring an empty set.',
      pins: pinRecords(),
      pinPolicy: 'Every absence assertion is written through a helper that also asserts presence, so an '
        + 'unpaired absence cannot be expressed. An absence check alone cannot distinguish "correctly '
        + 'withheld" from "looking for text this page never renders" -- which is exactly what one pin in an '
        + 'earlier revision of this file was doing.',
      leakProperty: {
        statement: 'For every investigation, the exact text of the graded quantity -- computed in this '
          + 'process from timeseries-models.js, never read off the page -- is ABSENT from that '
          + 'investigation’s rendered text before a prediction is committed and PRESENT after. '
          + 'Investigation 3 has a second gate: after the forecast is drawn, the outcomes and the comparison '
          + 'are still absent until they are asked for.',
        pins: leakChecks,
        pairedWithPresence: 'Every absence assertion is paired with a presence assertion, so a pin that could '
          + 'never match is caught rather than silently counted as a pass.',
      },
      screenshots,
      screenshotCount: screenshots.length,
      limitations: [
        'A recorded digest proves identity, not legibility. The captures were opened and read; that judgement '
          + 'is recorded in the topic record, not here.',
        'The curve sampler reads polyline, path and polygon; the shared layout inspector reads straight lines '
          + 'only; and the point-mark check reads rect and circle marks. All three run because each is blind '
          + 'to what the others see -- an axis title on a row of target rects was invisible to the first two.',
        'This exercises the opening state and the states the cases above reach. It is not a proof that every '
          + 'reachable combination of controls renders correctly; the whole-grid sweeps in the model suite '
          + 'cover the arithmetic those states would show.',
      ],
      status,
      failure,
      passed: status === 'passed',
    }, null, 2) + '\n');
  }
  console.log(`PASS: ${records.length} browser cases at 1366, 390 and 320 px; ${screenshots.length} captures; `
    + `${leakChecks.length} investigations pinned numerically for the leak property.`);
})();

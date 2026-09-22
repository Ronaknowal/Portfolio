// Production browser review of the Calibration & Conformal Prediction lesson.
//
// Three things are measured here that no offline verifier can reach.
//
//   1. **The page actually renders.** A lesson has shipped in this repository
//      whose three offline verifiers were green while the body threw and the
//      error boundary replaced the whole page. So the first assertions are that
//      the lesson root exists, that nothing was logged to the console, and that
//      the recovery UI is absent.
//   2. **The graded number is not on screen before the prediction is recorded.**
//      Pinned two ways for every investigation: the `cal-graded` marker must be
//      absent, AND the graded value's own printed text must not appear anywhere
//      in that investigation's rendered text. Asserting the absence of a verdict
//      banner alone is what let this class of defect through on an earlier
//      lesson, so both halves are required and each expected string is computed
//      here in Node from the same model the page uses.
//   3. **The paint, not the DOM.** A stylesheet rule beats a presentation
//      attribute: twelve edges once painted at 1px while carrying correct width
//      attributes, and a `fill="none"` band once vanished entirely behind a CSS
//      `fill`. Every shape's computed style is compared with its attribute, and
//      a shape with NO attribute is examined rather than skipped.
//
// Curves are sampled along their own geometry with a routine copied from
// scripts/verify-bias-variance-browser.cjs, because the shared layout inspector
// iterates only <line> elements and is blind to <path> and <polyline>.
//
// Run against a preview of a production build:
//   npx vite build --outDir dist-cal
//   npx vite preview --outDir dist-cal --host 127.0.0.1 --port 4193
//   PLAYWRIGHT_PACKAGE=... DIST_DIR=dist-cal LEARNING_BASE_URL=http://127.0.0.1:4193 \
//     node scripts/verify-calibration-browser.cjs
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');

const distDir = process.env.DIST_DIR || 'dist-cal';
const base = (process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4193').replace(/\/+$/, '');
const topicId = 'calibration-conformal-prediction';
const sourcePath = `src/learn/data/topics/${topicId}.jsx`;
const route = `${base}/learn/path/full-curriculum/${topicId}?module=classical-ml`;
const evidencePath = 'docs/teaching/evidence/calibration-browser.json';
const read = filename => JSON.parse(fs.readFileSync(filename, 'utf8'));
const hash = filename => createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
const normalize = text => text.replace(/\s+/g, ' ').trim();
const ownedFiles = [
  sourcePath,
  'src/learn/data/calibration-models.js',
  'src/learn/data/calibration-data.js',
  'src/learn/data/calibration-examples.js',
  'src/learn/components/lesson-labs/CalibrationShared.jsx',
  'src/learn/components/lesson-labs/CalibrationLabs.jsx',
  'src/learn/components/lesson-labs/CalibrationFigures.jsx',
  'src/learn/components/lesson-labs/calibration-labs.css',
  `src/learn/data/curriculum/blueprints/${topicId}.js`,
  'public/learn-assets/calibration/banknote-subset.csv',
  'public/learn-assets/calibration/airfoil-subset.csv',
  'public/learn-assets/calibration/ATTRIBUTION.txt',
];

/** Runs in the page. Samples every curve along its own geometry and reports the
 * labels it passes through.
 *
 * Copied from scripts/verify-bias-variance-browser.cjs. The shared inspector
 * `scripts/lib/lesson-visual-layout.cjs` iterates `svg.querySelectorAll('line')`,
 * so <path> and <polyline> are invisible to it — which is how four data curves
 * crossed value labels behind a fully green run. This lesson draws its sigmoid
 * overlay as a <path> and its gap drops as <line>, so both classes matter here.
 *
 * <line> was nevertheless missing from this selector until the guard audit: the
 * subject census reported that the whole check had TWO shapes to sample in the
 * default state, because almost everything this lesson draws is a <line>. The
 * comment above named the right hazard and the code covered the other half of
 * it. Chromium implements getTotalLength on <line> like any other geometry
 * element, so including it needs nothing else.
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
        halo: text.classList.contains('cal-halo'),
        /* A label PRINTED ON a filled bar. The threshold rule has to cross the
           bar — that crossing is its whole meaning — and the label sits in the
           middle of the bar by design, which is the same allowance the
           shape-over-label check already makes for these two classes. It is
           granted only to the threshold rule, and only for these labels; every
           other shape over any other label is still a finding. */
        onBar: text.classList.contains('is-on-bar') || text.classList.contains('is-on-fill'),
        box: text.getBoundingClientRect(),
      }));
    if (!labels.length) continue;
    for (const shape of svg.querySelectorAll('path, polyline, polygon, line')) {
      if (shape.classList.contains('cal-gridline')) continue;
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
        /* The gold rule on dark ground and the dark rule drawn on a bar are
           the same mark, drawn in whichever ink reads on what it crosses. */
        const isThresholdRule = shape.classList.contains('cal-threshold')
          || shape.classList.contains('cal-threshold-on-bar');
        if (isThresholdRule && labels[index].onBar) return;
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

/** Runs in the page. Compares every shape's PAINTED stroke width and fill with
 * the attribute it carries, and reports a shape with no attribute separately
 * rather than skipping it. */
function comparePaintedWithAttributes(root) {
  const findings = [];
  for (const shape of document.querySelectorAll(`${root} svg :is(rect, circle, line, path, polyline, polygon)`)) {
    const style = getComputedStyle(shape);
    const painted = { strokeWidth: style.strokeWidth, fill: style.fill, stroke: style.stroke };
    const attributes = {
      strokeWidth: shape.getAttribute('stroke-width'),
      fill: shape.getAttribute('fill'),
      stroke: shape.getAttribute('stroke'),
    };
    const rect = shape.getBoundingClientRect();
    findings.push({
      tag: shape.tagName,
      className: shape.getAttribute('class') || '',
      painted,
      attributes,
      visible: rect.width > 0 || rect.height > 0,
      paintedWidthPx: parseFloat(painted.strokeWidth) || 0,
    });
  }
  return findings;
}

/** Runs in the page. Effective on-screen font size of every SVG label, after
 * the viewBox scaling the browser applies. */
function effectiveLabelSizes(root) {
  const out = [];
  for (const svg of document.querySelectorAll(`${root} svg`)) {
    const box = svg.getBoundingClientRect();
    const viewBox = svg.viewBox.baseVal;
    const scale = viewBox && viewBox.width ? box.width / viewBox.width : 1;
    for (const text of svg.querySelectorAll('text')) {
      if (!text.textContent.trim() || !text.getClientRects().length) continue;
      const style = getComputedStyle(text);
      const declared = parseFloat(style.fontSize) || 0;
      out.push({
        text: text.textContent.trim().slice(0, 30),
        declared,
        fontFamily: style.fontFamily,
        scale: Number(scale.toFixed(4)),
        effective: Number((declared * scale).toFixed(2)),
        shorthandRisk: getComputedStyle(text).font.includes('px') && declared === 0,
      });
    }
  }
  return out;
}

/** Runs in the page. Filled shapes that overlap an SVG label.
 *
 * The shared layout inspector reads `<line>` elements and the curve sampler
 * reads paths; neither looks at a `<rect>` against a `<text>`. A count rail
 * whose bars grew upward from twenty units under the axis covered the tick
 * labels that sit fourteen under it — every number correct, the labels gone,
 * and nothing offline able to see it. This closes that class.
 */
function shapesCoveringLabels(root) {
  const findings = [];
  for (const svg of document.querySelectorAll(`${root} svg`)) {
    const labels = [...svg.querySelectorAll('text')]
      .filter(text => text.textContent.trim() && text.getClientRects().length)
      .map(text => ({ text: text.textContent.trim().slice(0, 30), box: text.getBoundingClientRect() }));
    if (!labels.length) continue;
    for (const shape of svg.querySelectorAll('rect, circle, ellipse')) {
      const style = getComputedStyle(shape);
      if (style.fill === 'none' || style.fillOpacity === '0' || style.display === 'none') continue;
      const box = shape.getBoundingClientRect();
      if (!(box.width > 0 && box.height > 0)) continue;
      for (const label of labels) {
        const overlapWidth = Math.min(box.right, label.box.right) - Math.max(box.left, label.box.left);
        const overlapHeight = Math.min(box.bottom, label.box.bottom) - Math.max(box.top, label.box.top);
        if (overlapWidth <= 0.5 || overlapHeight <= 0.5) continue;
        findings.push({
          label: label.text,
          shape: shape.getAttribute('class') || shape.tagName,
          fill: style.fill,
          overlapArea: Number((overlapWidth * overlapHeight).toFixed(1)),
          fractionOfLabel: Number((
            (overlapWidth * overlapHeight) / (label.box.width * label.box.height)).toFixed(3)),
        });
      }
    }
  }
  return findings;
}

/** Runs in the page. SVG labels whose glyph box escapes their own SVG.
 *
 * A label can stay inside the DOM and still be trimmed by the viewport: a bin
 * number nine units above a dot that sits at the very top of the box has an
 * ascender above the viewBox, and the digit renders with its head cut off.
 * `getClientRects()` is non-empty either way, so a size check cannot see it.
 */
function labelsOutsideTheirSvg(root) {
  const out = [];
  for (const svg of document.querySelectorAll(`${root} svg`)) {
    const frame = svg.getBoundingClientRect();
    for (const text of svg.querySelectorAll('text')) {
      if (!text.textContent.trim() || !text.getClientRects().length) continue;
      const box = text.getBoundingClientRect();
      const over = {
        top: Number((frame.top - box.top).toFixed(2)),
        bottom: Number((box.bottom - frame.bottom).toFixed(2)),
        left: Number((frame.left - box.left).toFixed(2)),
        right: Number((box.right - frame.right).toFixed(2)),
      };
      const worst = Math.max(over.top, over.bottom, over.left, over.right);
      if (worst > 0.5) out.push({ text: text.textContent.trim().slice(0, 30), over, worst });
    }
  }
  return out;
}

/** Runs in the page. Contrast of every SVG label against what is painted behind it.
 *
 * The fourth "DOM right, picture wrong" mechanism, and the one this suite had
 * no way to see: a label can be the right size, inside its own box, with the
 * correct content, and still be unreadable. Phase C fixed the gold-on-gold
 * label in the leading temperature segment by eye; the non-leading segments
 * were left at 3.01:1 because nothing measured them.
 *
 * "Behind" means the topmost EARLIER-painted opaque shape whose box contains
 * the text's centre — SVG paints in document order, so a later sibling is in
 * front of the text, not behind it.
 */
function labelContrast(root) {
  const channel = value => {
    const v = value / 255;
    return v <= 0.04045 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4;
  };
  const luminance = rgb => {
    const parts = rgb.match(/\d+(\.\d+)?/g);
    if (!parts || parts.length < 3) return null;
    if (parts.length > 3 && Number(parts[3]) === 0) return null;
    return 0.2126 * channel(+parts[0]) + 0.7152 * channel(+parts[1]) + 0.0722 * channel(+parts[2]);
  };
  const out = [];
  for (const svg of document.querySelectorAll(`${root} svg`)) {
    const painted = [...svg.querySelectorAll('rect, circle, ellipse, polygon, path')];
    for (const text of svg.querySelectorAll('text')) {
      if (!text.textContent.trim() || !text.getClientRects().length) continue;
      const box = text.getBoundingClientRect();
      const centre = { x: box.left + box.width / 2, y: box.top + box.height / 2 };
      let behind = null;
      for (const shape of painted) {
        // Document order: only shapes painted BEFORE the text sit behind it.
        if (!(text.compareDocumentPosition(shape) & Node.DOCUMENT_POSITION_PRECEDING)) continue;
        const style = getComputedStyle(shape);
        if (style.fill === 'none' || Number(style.fillOpacity) === 0) continue;
        const shapeBox = shape.getBoundingClientRect();
        if (centre.x < shapeBox.left || centre.x > shapeBox.right) continue;
        if (centre.y < shapeBox.top || centre.y > shapeBox.bottom) continue;
        behind = { fill: style.fill, className: shape.getAttribute('class') || shape.tagName };
      }
      /* No shape behind it means the label sits on whatever the page is
         painted with, and that is still a contrast question. Leaving those
         out was a domain gap the guard audit found by counting subjects: SEVEN
         of this lesson's 150-odd labels sat over a filled shape, so the check
         was judging under five per cent of the text on the page and would have
         had nothing to say about a muted grey going one shade too dark. Walk
         up for the first element with an opaque background. */
      if (!behind) {
        let node = svg;
        while (node && node !== document.documentElement) {
          const background = getComputedStyle(node).backgroundColor;
          const parts = background.match(/\d+(\.\d+)?/g);
          const opaque = parts && (parts.length < 4 || Number(parts[3]) > 0.99);
          if (opaque) { behind = { fill: background, className: `page:${node.tagName.toLowerCase()}` }; break; }
          node = node.parentElement;
        }
      }
      if (!behind) continue;
      const style = getComputedStyle(text);
      const ink = luminance(style.fill);
      const ground = luminance(behind.fill);
      if (ink === null || ground === null) continue;
      const ratio = (Math.max(ink, ground) + 0.05) / (Math.min(ink, ground) + 0.05);
      out.push({
        text: text.textContent.trim().slice(0, 24),
        fontSize: parseFloat(style.fontSize),
        ink: style.fill, ground: behind.fill, behind: behind.className,
        ratio: Number(ratio.toFixed(2)),
      });
    }
  }
  return out;
}

/** Runs in the page. SVG labels that overlap one another.
 *
 * The standard names text-against-text explicitly, and neither the shape check
 * nor the curve sampler sees it: two labels can collide with no shape involved
 * at all, which is what happens when marks crowd together at one end of an axis.
 */
function labelsOverlappingLabels(root) {
  const out = [];
  for (const svg of document.querySelectorAll(`${root} svg`)) {
    const labels = [...svg.querySelectorAll('text')]
      .filter(text => text.textContent.trim() && text.getClientRects().length)
      .map(text => ({ text: text.textContent.trim().slice(0, 24), box: text.getBoundingClientRect() }));
    for (let i = 0; i < labels.length; i += 1) {
      for (let j = i + 1; j < labels.length; j += 1) {
        const a = labels[i].box;
        const b = labels[j].box;
        const width = Math.min(a.right, b.right) - Math.max(a.left, b.left);
        const height = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top);
        if (width > 0.5 && height > 0.5) {
          out.push({
            first: labels[i].text, second: labels[j].text,
            overlapArea: Number((width * height).toFixed(1)),
          });
        }
      }
    }
  }
  return out;
}

/** Runs in the page. Clipped table cells AND tables wider than their container.
 *
 * The per-cell half alone was inert: no individual cell was clipped while a
 * nine-column table ran 254 pixels past its own scroll box at desktop width,
 * so its last three columns could only be reached by scrolling sideways. A
 * scroll box is the right fallback on a phone; at 1366 px it is a table that
 * does not fit.
 */
function clippedCells({ root, desktopWidth }) {
  const out = [];
  for (const cell of document.querySelectorAll(`${root} .cal-table-scroll :is(th, td)`)) {
    /* Below the narrow breakpoint the header row is deliberately collapsed to a
       screen-reader-only sliver and each cell carries its label through
       `data-label` instead. Counting that as clipping reported 130 defects that
       were the accessibility affordance working. */
    const head = cell.closest('thead');
    if (head && getComputedStyle(head).clipPath !== 'none') continue;
    if (cell.scrollWidth > cell.clientWidth + 1) {
      out.push({ kind: 'cell', text: cell.textContent.trim().slice(0, 40), scrollWidth: cell.scrollWidth, clientWidth: cell.clientWidth });
    }
  }
  if (window.innerWidth >= desktopWidth) {
    for (const box of document.querySelectorAll(`${root} .cal-table-scroll`)) {
      if (box.scrollWidth > box.clientWidth + 1) {
        out.push({
          kind: 'table',
          caption: (document.getElementById(box.getAttribute('aria-labelledby'))?.textContent
            || box.previousElementSibling?.textContent || '').trim().slice(0, 60),
          overflow: box.scrollWidth - box.clientWidth,
        });
      }
    }
  }
  return out;
}

/** Runs in the page. Every rendered display formula with its measured width. */
function displayMathWidths(root) {
  return [...document.querySelectorAll(`${root} .katex-display`)].map(node => {
    const inner = node.querySelector('.katex') || node;
    return {
      text: node.textContent.trim().slice(0, 48),
      width: Math.ceil(inner.getBoundingClientRect().width),
      parentWidth: Math.floor(node.parentElement.getBoundingClientRect().width),
      scrollWidth: node.scrollWidth,
      clientWidth: node.clientWidth,
    };
  });
}

/** Runs in the page. How many subjects each layout guard actually has.
 *
 * Every layout assertion above is of the form `deepEqual(findings, [])`, and
 * each one is green in two different worlds: the page is correct, or the guard
 * found nothing to look at. The second is not hypothetical — this suite's own
 * clipped-cell check spent a phase scoped to a class name the markup no longer
 * used, and a renamed class, a figure removed, or a `text` node replaced by a
 * `foreignObject` would put every one of these back into that state with no
 * visible change to the result line.
 *
 * The queries below are the SAME queries the guards use, so the census cannot
 * drift away from them independently: if a guard's subject set empties, the
 * matching count here drops and the floor fails.
 */
function subjectCensus(root) {
  const svgs = [...document.querySelectorAll(`${root} svg`)];
  let labels = 0;
  let curves = 0;
  for (const svg of svgs) {
    labels += [...svg.querySelectorAll('text')]
      .filter(text => text.textContent.trim() && text.getClientRects().length).length;
    for (const shape of svg.querySelectorAll('path, polyline, polygon, line')) {
      if (shape.classList.contains('cal-gridline')) continue;
      if (typeof shape.getTotalLength !== 'function') continue;
      if (!(shape.getTotalLength() > 0)) continue;
      if (!shape.getScreenCTM()) continue;
      curves += 1;
    }
  }
  const cells = [...document.querySelectorAll(`${root} .cal-table-scroll :is(th, td)`)]
    .filter(cell => {
      const head = cell.closest('thead');
      return !(head && getComputedStyle(head).clipPath !== 'none');
    }).length;
  return {
    svgs: svgs.length,
    labels,
    curves,
    cells,
    scrollBoxes: document.querySelectorAll(`${root} .cal-table-scroll`).length,
    formulas: document.querySelectorAll(`${root} .katex-display`).length,
  };
}

/** Runs in the page. Every SVG KaTeX itself emitted, measured against the size
 * KaTeX asked for.
 *
 * KaTeX draws accents, radicals and stretchy delimiters as real <svg> elements
 * and sizes each one with `.katex svg { height: inherit }` — the SVG takes the
 * height of the span KaTeX sized for it. A lesson stylesheet that writes
 * `.cal-lesson svg { height: auto }` has the same specificity and loads later,
 * so it wins, and the result is silent: correct DOM, correct text content, no
 * `.katex-error`, every offline verifier green. A sibling lesson lost every
 * radical to zero height that way and printed a bound as its own radicand.
 * This lesson has no radical but one `\widehat{\mathrm{ECE}}`, and a hat that
 * is drawn wrong is the empirical quantity being presented as the population
 * one — the distinction the whole lesson turns on.
 *
 * The assertion is the invariant, not the number: an SVG under `.katex` must
 * be exactly as tall as the box KaTeX gave it. That catches a collapse to zero
 * and an inflation alike; the unscoped rule measured here produced 9.078px
 * against 4.641px, a factor of 1.956.
 */
function katexSvgGeometry(root) {
  return [...document.querySelectorAll(`${root} .katex svg`)].map(svg => {
    const box = svg.getBoundingClientRect();
    const host = svg.parentElement.getBoundingClientRect();
    const owner = svg.closest('.katex');
    return {
      formula: (owner ? owner.textContent : '').trim().slice(0, 40),
      height: Number(box.height.toFixed(3)),
      hostHeight: Number(host.height.toFixed(3)),
      declaredHeight: getComputedStyle(svg).height,
    };
  });
}

(async () => {
  const startedAt = new Date().toISOString();
  const sourceHashes = Object.fromEntries(ownedFiles.map(filename => [filename, hash(filename)]));
  const models = await import('../src/learn/data/calibration-models.js');
  const { calibrationData } = await import('../src/learn/data/calibration-data.js');
  const { calibrationExamples } = await import('../src/learn/data/calibration-examples.js');
  const { tracks } = await import('../src/learn/data/tracks.js');
  const module_ = tracks.find(track => track.id === 'classical-ml');
  assert.equal(module_.topicIds[34], topicId, 'the lesson sits at position 35 of its module');
  const position = module_.topicIds.indexOf(topicId) + 1;

  /* Every expected graded string is computed HERE, from the same model the page
     uses, for the exact draft each scripted interaction produces. A literal
     typed into this file would drift the moment a fixture changed. */
  const expected = {};
  {
    const cards = models.fixtures.cards.map(card => ({ ...card }));
    cards[0] = { ...cards[0], forecast: 0.123 };
    const report = models.reliability(cards.map(card => card.forecast), cards.map(card => card.outcome),
      models.fixtures.twoBinEdges);
    expected.reliabilityEce = models.gradedText(report.ece);
    expected.reliabilityEceRounded = String(Number(report.ece.toFixed(6)));
  }
  {
    const stages = models.pavStages(models.fixtures.tiedScores, models.fixtures.tiedLabels);
    expected.monotonePooled = models.gradedText(stages.merges[0].merged.mean);
    expected.monotonePooledShort = String(Number(stages.merges[0].merged.mean.toFixed(6)));
  }
  {
    /* The baseline the AUC verdict must quote: the state applied at that point
       in the script, read AS AN AUC. Reading it under whichever quantity
       happened to be applied alongside it is the defect this pins. */
    const cards = models.fixtures.cards.map(card => ({ ...card }));
    cards[0] = { ...cards[0], forecast: 0.123 };
    const auc = models.rocAuc(cards.map(card => card.forecast), cards.map(card => card.outcome));
    expected.aucBaseline = models.gradedText(auc.value);
    const ece = models.reliability(cards.map(card => card.forecast), cards.map(card => card.outcome),
      models.fixtures.twoBinEdges).ece;
    expected.eceOfSameState = models.gradedText(ece);
  }
  {
    const threshold = models.conformalThreshold(models.fixtures.labCalibrationScores, 0.05);
    assert.ok(!threshold.finite, 'the scripted rank case must be the unbounded one');
    expected.rankUnbounded = 'infinity';
    expected.rankK = String(threshold.k);
  }
  {
    const changed = models.fixtures.changedResiduals
      .map((value, index) => value / models.fixtures.localScales[index]);
    const threshold = models.conformalThreshold(changed, models.fixtures.intervalAlpha);
    const interval = models.normalizedInterval(models.fixtures.queries[0].centre,
      models.fixtures.queries[0].localScale, threshold.q);
    expected.intervalWidth = models.gradedText(interval.width);
  }

  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [];
  const screenshots = [];
  const layoutStatesChecked = [];
  const censusByState = [];
  let contrastsMeasured = 0;
  let katexSvgsMeasured = 0;
  /* Every size the stylesheet sets on SVG text under the lesson root. A
     label with any other declared size is being styled by something else. */
  const DECLARED_LABEL_SIZES = new Set([9.5, 10, 12]);
  const checkText = async (locator, pattern) => assert.match(normalize(await locator.innerText()), pattern);
  const settle = page => page.evaluate(() =>
    new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  const ready = async page => {
    await page.locator('.cal-lesson').waitFor({ timeout: 30000 });
    await page.evaluate(() => document.fonts.ready);
    await settle(page);
  };

  try {
    const context = await browser.newContext({ viewport: { width: 1366, height: 1000 } });
    const page = await context.newPage();
    const requests = []; const errors = []; const failedAssets = []; const consoleErrors = [];
    page.on('request', request => requests.push(request.url()));
    page.on('pageerror', error => errors.push(error.message));
    page.on('requestfailed', request => failedAssets.push(request.url()));
    page.on('console', message => { if (message.type() === 'error') consoleErrors.push(message.text()); });
    await page.goto(route, { waitUntil: 'domcontentloaded' });
    await ready(page);

    const screenshot = async (locator, filename) => {
      const destination = path.join('docs/teaching/evidence/screenshots', filename);
      fs.mkdirSync(path.dirname(destination), { recursive: true });
      const viewport = page.viewportSize();
      const bounds = await locator.boundingBox();
      if (bounds && bounds.height > viewport.height - 160) {
        await page.setViewportSize({ width: viewport.width, height: Math.ceil(bounds.height) + 180 });
      }
      await locator.scrollIntoViewIfNeeded();
      await settle(page);
      await locator.screenshot({ path: destination });
      await page.setViewportSize(viewport);
      const bytes = fs.readFileSync(destination);
      screenshots.push({
        file: destination, bytes: bytes.length,
        digest: createHash('sha256').update(bytes).digest('hex'),
        viewport: `${viewport.width}x${viewport.height}`,
      });
    };

    /* ------------------------------------------------------ 1 · it rendered */
    assert.equal(await page.locator('.lesson-error, .lesson-boundary-error').count(), 0,
      'the error boundary has replaced the lesson');
    assert.equal(errors.length, 0, `the page threw: ${errors.join(' | ')}`);
    assert.equal(await page.locator('.katex-error').count(), 0, 'a formula failed to render');
    await checkText(page.locator('.reader-header h1'), /^Calibration & Conformal Prediction$/);
    await checkText(page.locator('.reader-header__meta'),
      new RegExp(`${position} of ${module_.topicIds.length} topics on this route`));
    await checkText(page.locator('.reader-footer__previous'), /PAC Learning/);
    await checkText(page.locator('.reader-footer__next'), /Rademacher Complexity/);
    assert.equal(await page.locator('.cal-investigation').count(), 4, 'four investigations');
    assert.equal(await page.locator('.cal-practice').count(), 8, 'eight practice tasks');
    assert.equal(await page.locator('.python-example').count(), 4, 'four runnable programs with recorded output');
    assert.ok(await page.locator('.cal-figure').count() >= 12, 'at least twelve drawn figures');
    const rendered = normalize(await page.locator('.cal-lesson').textContent());
    for (const key of ['reliability', 'monotone', 'rank']) {
      assert.ok(rendered.includes(normalize(calibrationExamples[key].expected)),
        `the recorded output of ${key} is not on the page`);
    }
    assert.ok(rendered.includes(normalize(calibrationExamples.experiments.code.slice(0, 200))),
      'the complete experiment program is shown');
    for (const anchor of await page.locator('.lesson-intro a[href^="#"]')
      .evaluateAll(items => items.map(item => item.getAttribute('href')))) {
      assert.equal(await page.locator(`[id="${anchor.slice(1)}"]`).count(), 1, `broken lesson anchor ${anchor}`);
    }
    assert.ok(rendered.includes('First pass.'), 'the first-pass route is on the page');
    assert.ok(rendered.includes('CC BY 4.0'), 'the licence travels with the data');
    assert.ok(rendered.includes(calibrationData.provenance.files['banknote-subset.csv'].sha256),
      'and so does the banknote hash');
    assert.ok(rendered.includes(calibrationData.provenance.files['airfoil-subset.csv'].sha256),
      'and the airfoil hash');
    records.push({ case: 'The lesson renders with no thrown error, no error boundary and no KaTeX failure; four '
      + 'investigations, twelve figures, eight practice tasks, three displayed program outputs, every route '
      + 'anchor resolving, and both dataset hashes and licences visible' });

    /* Every served asset is reachable and byte-identical to what is recorded. */
    for (const [name, entry] of Object.entries(calibrationData.provenance.files)) {
      const response = await page.request.get(`${base}${entry.file}`);
      assert.equal(response.status(), 200, `${name} is not served`);
      const bytes = await response.body();
      assert.equal(bytes.length, entry.bytes, `${name} byte count`);
      assert.equal(createHash('sha256').update(bytes).digest('hex'), entry.sha256, `${name} digest`);
    }
    /* No sibling lesson's copy of the same data is fetched. */
    const siblingAssets = requests.filter(url => /\/learn-assets\/(pac-learning|evaluation-metrics|semi-supervised-learning|automl-nas|bias-variance|regularization)\//.test(url));
    assert.deepEqual(siblingAssets, [], `the page fetched another lesson's assets: ${siblingAssets.join(', ')}`);
    /* That assertion is also satisfied by a request log that is empty, which is
       what a renamed Playwright event would give, and it would then be green on
       a page that fetched every sibling's data. Floor it on the log itself.
       This lesson's own CSVs are NOT among the page's requests — its numbers
       are bundled into the module and the served files exist for provenance and
       for the learner to download — so the floor is that real page URLs under
       the preview origin are being recorded at all, which is the property the
       sibling filter depends on. */
    const pageRequests = requests.filter(url => url.startsWith(base));
    assert.ok(requests.length >= 10 && pageRequests.length >= 5,
      `the request log holds ${requests.length} entries, ${pageRequests.length} of them page URLs under `
      + `${base}; the listener is not recording, so the sibling-asset check has nothing to filter`);
    records.push({ case: `All ${Object.keys(calibrationData.provenance.files).length} served files reachable and `
      + 'byte-identical to their recorded digests, with no sibling lesson\'s copy fetched' });

    /* ------------------------------ 2 · nothing revealed on first paint (all four) */
    const investigations = page.locator('.cal-investigation');
    for (let index = 0; index < 4; index += 1) {
      const lab = investigations.nth(index);
      /* Scoped to the prediction group. The outcome and label radios elsewhere
         in a lab are the learner's INPUTS and are legitimately preselected —
         they are the entities being edited, not an answer. */
      assert.equal(await lab.locator('.cal-prediction input[type="radio"]:checked').count(), 0,
        `investigation ${index + 1} preselects a prediction`);
      assert.ok(await lab.getByRole('button', { name: 'Apply and check' }).isDisabled(),
        `investigation ${index + 1} can be checked without a prediction`);
      assert.equal(await lab.locator('.cal-verdict').count(), 0,
        `investigation ${index + 1} shows a verdict on first paint`);
      assert.equal(await lab.locator('.cal-graded').count(), 0,
        `investigation ${index + 1} shows a graded readout on first paint`);
      await screenshot(lab, `calibration-investigation-${index + 1}-initial-desktop.png`);
    }
    records.push({ case: 'All four investigations open with no prediction selected, the check disabled, no verdict '
      + 'and no gated readout' });

    /* --------------------------------- 3 · investigation 1, pinned numerically */
    const reliabilityLab = investigations.nth(0);
    await reliabilityLab.getByLabel('forecast for class 1').first().fill('0.123');
    await reliabilityLab.getByLabel('forecast for class 1').first().blur();
    await settle(page);
    let text = normalize(await reliabilityLab.innerText());
    assert.ok(expected.reliabilityEceRounded.length >= 6,
      'the pinned ECE is too short to be a distinctive string, so this check would not catch a leak');
    assert.ok(!text.includes(expected.reliabilityEce),
      `the graded ECE ${expected.reliabilityEce} is on screen before a prediction was recorded`);
    assert.ok(!text.includes(expected.reliabilityEceRounded),
      `the graded ECE ${expected.reliabilityEceRounded} is on screen before a prediction was recorded`);
    assert.equal(await reliabilityLab.locator('.cal-graded').count(), 0, 'and no gated readout either');
    await checkText(reliabilityLab.locator('.cal-pending'), /Draft inputs differ from the applied ones/);
    await reliabilityLab.getByLabel('higher than the applied state').check();
    await reliabilityLab.getByRole('button', { name: 'Apply and check' }).click();
    await settle(page);
    assert.equal(await reliabilityLab.locator('.cal-verdict').count(), 1, 'a verdict appears after committing');
    assert.ok(await reliabilityLab.locator('.cal-graded').count() >= 1, 'and the gated readout with it');
    text = normalize(await reliabilityLab.innerText());
    assert.ok(text.includes(expected.reliabilityEce),
      `the graded ECE ${expected.reliabilityEce} is still missing after committing`);
    await checkText(reliabilityLab.locator('.cal-verdict'), /the binned ECE went from/);
    await screenshot(reliabilityLab, 'calibration-investigation-1-checked-desktop.png');
    /* An edit retires the verdict. */
    await reliabilityLab.getByLabel('forecast for class 1').first().fill('0.2');
    await reliabilityLab.getByLabel('forecast for class 1').first().blur();
    await settle(page);
    assert.equal(await reliabilityLab.locator('.cal-verdict').count(), 0, 'an edit did not retire the verdict');
    assert.equal(await reliabilityLab.locator('.cal-graded').count(), 0, 'nor the gated readout');
    await checkText(reliabilityLab.locator('.cal-history.is-pending'), /stays hidden until you apply/);
    /* The AUC null: every outcome the same leaves no value at all, which is a
       fourth answer and not a number. */
    await reliabilityLab.getByRole('button', { name: 'Make every outcome 1 — AUC then has no value' }).click();
    await reliabilityLab.getByLabel('Which summary you are committing on').selectOption('auc');
    await settle(page);
    assert.equal(await reliabilityLab.locator('.cal-graded').count(), 0,
      'switching the graded quantity must retire the previous verdict');
    await reliabilityLab.getByLabel('it has no value at all for the new cards').check();
    await reliabilityLab.getByRole('button', { name: 'Apply and check' }).click();
    await settle(page);
    await checkText(reliabilityLab.locator('.cal-verdict'), /Your prediction matches/);
    await checkText(reliabilityLab, /no value/);
    /* The verdict names the ROC AUC, so the number it quotes as the baseline
       must BE an AUC. It once quoted the applied state's ECE — a figure that
       had never been an AUC — because the baseline was read under whichever
       quantity was applied rather than the one now being asked about. */
    const aucVerdict = normalize(await reliabilityLab.locator('.cal-verdict').innerText());
    assert.ok(aucVerdict.includes(expected.aucBaseline),
      `the AUC verdict does not quote the applied state's AUC (${expected.aucBaseline}): "${aucVerdict}"`);
    assert.ok(!aucVerdict.includes(expected.eceOfSameState),
      `the AUC verdict quotes ${expected.eceOfSameState}, which is that state's ECE and not its AUC: `
      + `"${aucVerdict}"`);
    await screenshot(reliabilityLab, 'calibration-investigation-1-auc-null-desktop.png');
    records.push({ case: 'Investigation 1: an edited forecast keeps the graded ECE off screen until a prediction '
      + 'is committed, pinned by the exact printed value as well as by the gate marker; the verdict names the '
      + 'move; an edit retires both; the previous attempt stays hidden; and switching the graded quantity to AUC '
      + 'on a one-class sample is graded as having no value rather than as a number' });

    /* --------------------------------- 4 · investigation 2, the pooled value */
    const monotoneLab = investigations.nth(1);
    await monotoneLab.getByRole('button', { name: /Tied scores/ }).click();
    await settle(page);
    text = normalize(await monotoneLab.innerText());
    assert.ok(!text.includes(expected.monotonePooled),
      `the pooled probability ${expected.monotonePooled} is on screen before a prediction was recorded`);
    assert.ok(!text.includes(expected.monotonePooledShort),
      `the pooled probability ${expected.monotonePooledShort} is on screen before a prediction was recorded`);
    assert.equal(await monotoneLab.locator('.cal-graded').count(), 0, 'and no gated readout');
    /* The first radio in this lab is a LABEL editor, not a prediction option;
       the prediction group is the one that has to be scoped to. */
    await monotoneLab.locator('.cal-prediction input[type="radio"]').first().check();
    await monotoneLab.getByRole('button', { name: 'Apply and check' }).click();
    await settle(page);
    assert.equal(await monotoneLab.locator('.cal-verdict').count(), 1, 'a verdict appears');
    text = normalize(await monotoneLab.innerText());
    assert.ok(text.includes(expected.monotonePooled),
      'the pooled probability is still missing after committing');
    assert.ok(text.includes('Averaging the two block probabilities instead would give'),
      'the count-weighted trap is explained, not merely computed');
    await screenshot(monotoneLab, 'calibration-investigation-2-checked-desktop.png');
    records.push({ case: 'Investigation 2: the pooled probability of the next required merge stays off screen '
      + 'until committed, pinned by its printed value at two precisions, and the explanation contrasts it with '
      + 'the average of the two block means' });

    /* ------------------------ 5 · investigation 3, the unbounded threshold */
    const rankLab = investigations.nth(2);
    await rankLab.getByRole('button', { name: 'Ask for 95% coverage instead' }).click();
    await settle(page);
    text = normalize(await rankLab.innerText());
    assert.ok(!/\binfinity\b/i.test(text.replace(/the whole label space/gi, '')),
      'the unbounded threshold is named before a prediction was recorded');
    assert.equal(await rankLab.locator('.cal-graded').count(), 0, 'and no gated readout');
    assert.ok(!text.includes('rank denominator n + 1'), 'nor the rank table');
    await rankLab.getByLabel('every class passes').check();
    await rankLab.getByLabel(/The rank k you expect/).fill(expected.rankK);
    await rankLab.getByRole('button', { name: 'Apply and check' }).click();
    await settle(page);
    assert.equal(await rankLab.locator('.cal-verdict').count(), 1, 'a verdict appears');
    text = normalize(await rankLab.innerText());
    assert.ok(text.includes('infinity'), 'the unbounded threshold is named after committing');
    assert.ok(text.includes('rank denominator n + 1'), 'and the rank table appears');
    await checkText(rankLab.locator('.cal-verdict'), /Your prediction matches/);
    /* The unbounded threshold must not be drawn as a point on the axis. */
    const thresholdLines = await rankLab.locator('svg .cal-threshold').count();
    assert.equal(thresholdLines, 0, 'an unbounded threshold was drawn as a line on the score axis');
    await screenshot(rankLab, 'calibration-investigation-3-unbounded-desktop.png');
    records.push({ case: 'Investigation 3: at a 95% target the rank exceeds the calibration set, and neither the '
      + 'word infinity nor the rank table appears until a prediction is committed; the correct rank is accepted '
      + 'exactly, and no threshold line is drawn on the axis for an unbounded threshold' });

    /* ------------------------------- 6 · investigation 4, the physical width */
    const intervalLab = investigations.nth(3);
    await intervalLab.getByRole('button', { name: 'Change the last two residuals to 12 and 16' }).click();
    await settle(page);
    text = normalize(await intervalLab.innerText());
    assert.ok(!text.includes(expected.intervalWidth),
      `the graded width ${expected.intervalWidth} is on screen before a prediction was recorded`);
    assert.equal(await intervalLab.locator('.cal-graded').count(), 0, 'and no gated readout');
    await intervalLab.getByLabel('the interval gets wider').check();
    await intervalLab.getByRole('button', { name: 'Apply and check' }).click();
    await settle(page);
    assert.equal(await intervalLab.locator('.cal-verdict').count(), 1, 'a verdict appears');
    text = normalize(await intervalLab.innerText());
    assert.ok(text.includes(expected.intervalWidth), 'the graded width is still missing after committing');
    await checkText(intervalLab.locator('.cal-verdict'), /Your prediction matches/);
    await screenshot(intervalLab, 'calibration-investigation-4-checked-desktop.png');
    /* The unit null, in the browser: doubling every scale must report unchanged. */
    await intervalLab.getByRole('button', { name: 'Back to the nine original pairs' }).click();
    await settle(page);
    await intervalLab.getByRole('button', { name: 'Double every calibration AND query scale — an exact null' }).click();
    await settle(page);
    await intervalLab.getByLabel(/unchanged/).check();
    await intervalLab.getByRole('button', { name: 'Apply and check' }).click();
    await settle(page);
    await checkText(intervalLab.locator('.cal-verdict'), /Your prediction matches/);
    await checkText(intervalLab.locator('.cal-verdict'), /the physical interval did not/);
    await screenshot(intervalLab, 'calibration-investigation-4-unit-null-desktop.png');
    records.push({ case: 'Investigation 4: the graded physical width stays off screen until committed; a doubled '
      + 'residual pair widens the interval and is graded so; and doubling every calibration and query scale is '
      + 'reported as unchanged with the dimensionless-threshold explanation' });

    /* ------------------------------------------ 7 · the paint, not the DOM */
    const painted = await page.evaluate(comparePaintedWithAttributes, '.cal-lesson');
    assert.ok(painted.length > 200, `only ${painted.length} shapes were inspected`);
    const widthAttributes = painted.filter(shape => shape.attributes.strokeWidth !== null);
    for (const shape of widthAttributes) {
      assert.equal(shape.paintedWidthPx, parseFloat(shape.attributes.strokeWidth),
        `${shape.tag}.${shape.className} carries stroke-width ${shape.attributes.strokeWidth} but paints at `
        + `${shape.painted.strokeWidth} — a stylesheet rule is overriding an attribute that encodes a quantity`);
    }
    /* A shape with NO attribute is examined rather than skipped: the SVG initial
       stroke-width is 1px, which can be wider than a quantity-carrying width. */
    const withoutAttribute = painted.filter(shape => shape.attributes.strokeWidth === null && shape.visible);
    assert.ok(withoutAttribute.length > 0, 'no attribute-free shape was found, so this case inspected nothing');
    for (const shape of withoutAttribute) {
      assert.ok(shape.paintedWidthPx > 0 || shape.painted.fill !== 'none',
        `${shape.tag}.${shape.className} has no stroke-width attribute, paints at `
        + `${shape.painted.strokeWidth} and has fill ${shape.painted.fill} — it is invisible`);
    }
    /* An outline-only shape must really be outline-only in the paint.
     *
     * Matched on whole class TOKENS, not by substring. A substring test treated
     * `cal-rail-bar` — a deliberately filled count bar — as if it were the
     * `cal-rail` axis and demanded it paint no fill. A guard that fires on
     * correct code gets weakened rather than fixed, which is how a real one
     * gets lost. */
    const OUTLINE_ONLY = new Set([
      'cal-area-missed', 'cal-diagonal', 'cal-sigmoid-line', 'cal-interval', 'cal-fit-line',
      'cal-rail', 'cal-threshold', 'cal-gap-line', 'cal-cap', 'cal-axis', 'cal-tick', 'cal-gridline',
    ]);
    const outlines = painted.filter(shape =>
      shape.className.split(/\s+/).some(token => OUTLINE_ONLY.has(token)));
    assert.ok(outlines.length > 0, 'no outline shape was found to check');
    assert.ok(new Set(outlines.flatMap(shape =>
      shape.className.split(/\s+/).filter(token => OUTLINE_ONLY.has(token)))).size >= 6,
    'fewer than six of the declared outline classes were actually found on the page');
    for (const shape of outlines) {
      assert.equal(shape.painted.fill, 'none',
        `${shape.className} paints a fill of ${shape.painted.fill}; a CSS fill beats a fill="none" attribute and `
        + 'turns an outline into a blob');
    }
    records.push({ case: `${painted.length} shapes compared against their attributes: `
      + `${widthAttributes.length} carrying a stroke-width attribute paint at exactly that width, `
      + `${withoutAttribute.length} without one are visible, and ${outlines.length} outline shapes paint no fill` });

    /* ---------------------------------- 8 · labels, curves and clipped cells */
    const sizes = await page.evaluate(effectiveLabelSizes, '.cal-lesson');
    assert.ok(sizes.length > 60, `only ${sizes.length} SVG labels were measured`);
    for (const label of sizes) {
      assert.ok(label.declared > 0,
        `"${label.text}" has a computed font-size of 0 — the font shorthand has reset it`);
      assert.ok(label.effective >= 8,
        `"${label.text}" renders at ${label.effective}px, below the readable floor`);
      assert.ok(label.effective <= 34,
        `"${label.text}" renders at ${label.effective}px, so a full-width SVG is scaling its text out of proportion`);
      /* The lesson-root rule has to REACH every label. Size alone cannot tell:
         a diagram that falls back to the browser default gets 16px, which is
         inside any sane range, so the family and the declared sizes are checked
         as well. A rule scoped to a class that misses some diagrams shows up
         here and nowhere else. */
      assert.ok(/mono|Consolas|Courier/i.test(label.fontFamily),
        `"${label.text}" has the declared font family ${label.fontFamily}; the lesson-root SVG text rule is not `
        + 'reaching it, so it has fallen back to the browser default');
      assert.ok(DECLARED_LABEL_SIZES.has(label.declared),
        `"${label.text}" has a declared font-size of ${label.declared}px, which is not one of the sizes this `
        + `lesson's stylesheet sets (${[...DECLARED_LABEL_SIZES].join(', ')}) — the text rule is not reaching it`);
    }
    const curveHits = await page.evaluate(sampleCurvesThroughLabels, '.cal-lesson');
    assert.deepEqual(curveHits, [], `a curve travels through a label: ${JSON.stringify(curveHits)}`);
    /* Filled shapes over labels. Only the stacked probability bars are allowed
       to contain text, because printing each class's share inside its own
       segment is what those bars are for. */
    const LABELS_MAY_SIT_INSIDE = new Set(['cal-prob-bar', 'cal-prob-bar is-leading', 'cal-lane']);
    const covered = await page.evaluate(shapesCoveringLabels, '.cal-lesson');
    const unexpected = covered.filter(entry => !LABELS_MAY_SIT_INSIDE.has(entry.shape));
    assert.deepEqual(unexpected, [],
      `a filled shape covers an SVG label: ${JSON.stringify(unexpected.slice(0, 6))}`);
    assert.ok(covered.length > 0,
      'no shape-over-label overlap was found at all, not even the stacked bars that are meant to have one — '
      + 'the check is looking at nothing');
    /* The same three layout measurements, reusable, because running them only
       on the opening state is how a defect hides: the dB axis of one interval
       procedure drew a tick at x = −5.6 that the default procedure never
       produced, and a nine-column table only overflowed once the widest values
       were in it. Every informative state below is measured too. */
    const layoutPass = async where => {
      /* What this pass has to look at, before any of it is judged. Floors, not
         equalities: the count legitimately moves with the viewport (a figure
         may drop a tick label at 320 px), but it cannot collapse. */
      const census = await page.evaluate(subjectCensus, '.cal-lesson');
      /* KaTeX's own SVGs, which this lesson's stylesheet must not resize. */
      const katexSvgs = await page.evaluate(katexSvgGeometry, '.cal-lesson');
      assert.ok(katexSvgs.length >= 1,
        `no KaTeX-emitted SVG was found (${where}); the accent, radical and stretchy-delimiter check has no `
        + 'subject and would pass on a page where every one of them had collapsed');
      katexSvgs.forEach(entry => {
        assert.ok(entry.height >= 1,
          `a KaTeX SVG in "${entry.formula}" is ${entry.height}px tall (${where}) — it has collapsed, and the `
          + 'accent or radical it draws is simply absent from the rendered formula');
        assert.ok(Math.abs(entry.height - entry.hostHeight) <= 0.5,
          `a KaTeX SVG in "${entry.formula}" renders ${entry.height}px tall inside a ${entry.hostHeight}px box `
          + `(${where}), a factor of ${(entry.height / entry.hostHeight).toFixed(3)}. KaTeX sizes these with `
          + '`height: inherit`, so a lesson rule matching a bare `svg` under the lesson root has overridden it');
      });
      katexSvgsMeasured += katexSvgs.length;
      /* Set just under the measured minimum across all ten states (svgs 17,
         labels 174, curves 225, cells 450 at narrow widths where the header
         row collapses, scroll boxes 25, formulas 10), not at a token value.
         A floor of 8 curves would have been met by the 2 the sampler could
         actually see before <line> was added to its selector. */
      const FLOORS = { svgs: 16, labels: 170, curves: 220, cells: 440, scrollBoxes: 24, formulas: 10 };
      Object.entries(FLOORS).forEach(([key, floor]) => {
        assert.ok(census[key] >= floor,
          `the layout guards have only ${census[key]} ${key} to inspect (${where}), below the floor of ${floor}: `
          + 'their subject set has drifted, so every empty-findings assertion below is green for the wrong reason');
      });
      censusByState.push({ where, ...census });
      const overlapping = await page.evaluate(labelsOverlappingLabels, '.cal-lesson');
      assert.deepEqual(overlapping, [],
        `two SVG labels overlap (${where}): ${JSON.stringify(overlapping.slice(0, 6))}`);
      const outside = await page.evaluate(labelsOutsideTheirSvg, '.cal-lesson');
      assert.deepEqual(outside, [],
        `an SVG label is trimmed by its own viewBox (${where}): ${JSON.stringify(outside.slice(0, 6))}`);
      const boxes = await page.evaluate(clippedCells, { root: '.cal-lesson', desktopWidth: 900 });
      assert.deepEqual(boxes, [],
        `a table does not fit its container (${where}): ${JSON.stringify(boxes)}`);
      const allCovered = await page.evaluate(shapesCoveringLabels, '.cal-lesson');
      assert.ok(allCovered.length > 0,
        `no shape-over-label overlap was found at all (${where}), not even the stacked bars that are meant to `
        + 'have one — the check is looking at nothing');
      const shapesOver = allCovered.filter(entry => !LABELS_MAY_SIT_INSIDE.has(entry.shape));
      assert.deepEqual(shapesOver, [],
        `a filled shape covers an SVG label (${where}): ${JSON.stringify(shapesOver.slice(0, 6))}`);
      /* Readability, not only position. WCAG asks 4.5:1 of small text and
         3:1 of text at 18.66 px or above; every label this lesson prints on a
         filled shape is small. */
      const contrasts = await page.evaluate(labelContrast, '.cal-lesson');
      const unreadable = contrasts.filter(entry => entry.ratio < (entry.fontSize >= 18.66 ? 3 : 4.5));
      assert.deepEqual(unreadable, [],
        `an SVG label is below the contrast its size requires (${where}): `
        + `${JSON.stringify(unreadable.slice(0, 6))}`);
      assert.ok(contrasts.length >= 170,
        `only ${contrasts.length} labels had a measurable background (${where}); the contrast guard is judging `
        + 'almost nothing, so an unreadable label would not be reported');
      contrastsMeasured += contrasts.length;
      layoutStatesChecked.push(where);
    };
    await layoutPass('default state at 1366 px');
    records.push({ case: `${sizes.length} SVG labels render between 8 and 34 px after viewBox scaling with no `
      + 'shorthand reset, no curve passes through a label, no two labels overlap, no label is trimmed by its own '
      + 'viewBox, and every table fits its container at 1366 px' });

    /* ----------------------------------------- 9 · display math at 320 px */
    const desktopMath = await page.evaluate(displayMathWidths, '.cal-lesson');
    assert.ok(desktopMath.length >= 8, `only ${desktopMath.length} display formulas were found`);
    await screenshot(page.locator('.cal-lesson'), 'calibration-lesson-desktop-1366.png');

    const widths = [390, 320];
    const narrowFindings = {};
    for (const width of widths) {
      await page.setViewportSize({ width, height: 900 });
      await settle(page);
      await page.evaluate(() => document.fonts.ready);
      const math = await page.evaluate(displayMathWidths, '.cal-lesson');
      assert.ok(math.length >= desktopMath.length,
        `${math.length} display formulas at ${width} px against ${desktopMath.length} at 1366 px; formulas have `
        + 'disappeared from the narrow layout, so the overflow check below is judging a smaller set than the '
        + 'one that has to fit');
      const overflowing = math.filter(entry => entry.width > entry.parentWidth + 1);
      assert.deepEqual(overflowing, [],
        `a display formula overflows its column at ${width} px: ${JSON.stringify(overflowing)}`);
      const horizontal = await page.evaluate(() =>
        document.documentElement.scrollWidth - document.documentElement.clientWidth);
      assert.ok(horizontal <= 1, `the page scrolls sideways by ${horizontal} px at ${width} px`);
      const narrowSizes = await page.evaluate(effectiveLabelSizes, '.cal-lesson');
      assert.ok(narrowSizes.length >= 120,
        `only ${narrowSizes.length} SVG labels were measured at ${width} px; the legibility check has lost its `
        + 'subject set and would report nothing however small the text got');
      const tiny = narrowSizes.filter(label => label.effective < 7);
      assert.deepEqual(tiny, [], `SVG labels are illegible at ${width} px: ${JSON.stringify(tiny.slice(0, 5))}`);
      await layoutPass(`${width} px`);
      narrowFindings[width] = { formulas: math.length, labels: narrowSizes.length };
      await screenshot(page.locator('.cal-lesson'), `calibration-lesson-narrow-${width}.png`);
      for (let index = 0; index < 4; index += 1) {
        await screenshot(page.locator('.cal-investigation').nth(index),
          `calibration-investigation-${index + 1}-narrow-${width}.png`);
      }
    }
    await page.setViewportSize({ width: 1366, height: 1000 });
    await settle(page);
    records.push({ case: `Display formulas measured at 1366, 390 and 320 px: none overflows its column, the page `
      + 'never scrolls sideways, and no SVG label falls below 7 px effective' });

    /* ----------------------------------- 10 · every figure, at desktop width */
    const figureCount = await page.locator('.cal-figure').count();
    for (let index = 0; index < figureCount; index += 1) {
      await screenshot(page.locator('.cal-figure').nth(index), `calibration-figure-${index + 1}-desktop.png`);
    }
    /* And every complete figure BLOCK — caption, drawing, tables and kind tags
       together, which is the unit a reader meets.

       Capturing only `.cal-figure` left the figures that carry no SVG at all
       with no capture of their own: the four label roles became reflowing HTML
       after its SVG text ran outside its viewBox, and from that moment nothing
       in the evidence showed it. A figure that cannot be looked at cannot be
       reviewed. */
    const blockCount = await page.locator('.cal-figure-block').count();
    assert.ok(blockCount >= 12, `only ${blockCount} figure blocks were found`);
    for (let index = 0; index < blockCount; index += 1) {
      await screenshot(page.locator('.cal-figure-block').nth(index),
        `calibration-block-${index + 1}-desktop.png`);
    }
    /* Informative figure states, not defaults only. */
    const measured = page.locator('.cal-figure-block').filter({ hasText: 'Figure 8.' });
    if (await measured.count()) {
      await measured.getByRole('button', { name: /Training-prior constant/ }).click();
      await settle(page);
      await layoutPass('figure 8 showing the training-prior constant');
      await screenshot(measured, 'calibration-figure-8-prior-constant-desktop.png');
    }
    const intervals = page.locator('.cal-figure-block').filter({ hasText: 'Figure 9.' });
    if (await intervals.count()) {
      await intervals.getByRole('button', { name: /Conformalized quantile regression/ }).click();
      await settle(page);
      await layoutPass('figure 9 showing conformalized quantile regression');
      await screenshot(intervals, 'calibration-figure-9-cqr-desktop.png');
      for (const name of ['Training-mean constant', 'Raw fitted']) {
        await intervals.getByRole('button', { name: new RegExp(name) }).click();
        await settle(page);
        await layoutPass(`figure 9 showing ${name}`);
      }
      for (const name of ['Naive sigmoid', 'Held-out isotonic', 'Held-out temperature']) {
        await measured.getByRole('button', { name: new RegExp(name) }).click();
        await settle(page);
        await layoutPass(`figure 8 showing ${name}`);
      }
    }

    assert.equal(new Set(screenshots.map(entry => entry.file)).size, screenshots.length,
      'every capture has its own path');
    assert.equal(new Set(screenshots.map(entry => entry.digest)).size, screenshots.length,
      'two captures are byte-identical, so one of them is a stale orphan or a duplicate state');
    /* Every calibration capture ON DISK must be one this run wrote.
     *
     * Distinct digests among the files this run recorded says nothing about a
     * file it no longer writes. Converting one figure from SVG to HTML reduced
     * the figure count by one, and `calibration-figure-17-desktop.png` survived
     * from the previous run showing a layout that no longer exists anywhere —
     * a picture of a defect that had already been fixed, sitting in the
     * evidence directory looking current. Orphans are named and removed. */
    const directory = 'docs/teaching/evidence/screenshots';
    const written = new Set(screenshots.map(entry => path.basename(entry.file)));
    const orphans = fs.readdirSync(directory)
      .filter(name => name.startsWith('calibration-') && name.endsWith('.png') && !written.has(name));
    orphans.forEach(name => fs.unlinkSync(path.join(directory, name)));
    const stillThere = fs.readdirSync(directory)
      .filter(name => name.startsWith('calibration-') && name.endsWith('.png'));
    assert.equal(stillThere.length, screenshots.length,
      `the evidence directory holds ${stillThere.length} calibration captures but this run wrote `
      + `${screenshots.length}`);
    records.push({ case: `${screenshots.length} captures, each with its own path and its own content digest; `
      + `${orphans.length} stale capture(s) from an earlier figure count removed` });

    assert.deepEqual(failedAssets, [], `a request failed: ${failedAssets.join(', ')}`);
    /* An empty console-error log proves nothing on its own: it is equally what a
       detached listener gives. Emit one known error and require it to arrive,
       then judge the rest. This is the only guard here whose subject set can be
       created on demand, so it is the only one that can be floored exactly. */
    const PROBE = 'cal-console-listener-probe';
    await page.evaluate(token => console.error(token), PROBE);
    await page.waitForTimeout(150);
    assert.ok(consoleErrors.some(text => text.includes(PROBE)),
      'the deliberate console.error probe never reached the listener, so the console-error assertion above was '
      + 'green because nothing was being recorded');
    const realConsoleErrors = consoleErrors.filter(text => !text.includes(PROBE));
    assert.deepEqual(realConsoleErrors, [],
      `the console logged an error: ${realConsoleErrors.join(' | ')}`);

    const evidence = {
      checkedAt: startedAt,
      finishedAt: new Date().toISOString(),
      route,
      distDir,
      buildManifestSha256: hash(`${distDir}/.vite/manifest.json`),
      sourceHashes,
      verifierSha256: hash('scripts/verify-calibration-browser.cjs'),
      widths: [1366, 390, 320],
      narrowFindings,
      layoutStatesChecked,
      censusByState,
      contrastMeasurements: contrastsMeasured,
      katexSvgsMeasured,
      shapesInspected: painted.length,
      svgLabelsMeasured: sizes.length,
      displayFormulas: desktopMath.length,
      screenshots,
      staleCapturesRemoved: [],
      cases: records,
      gradedValuePins: expected,
      scope: 'A production preview at 1366, 390 and 320 px. The lesson is checked to have rendered at all — no '
        + 'thrown error, no error boundary, no KaTeX failure — before anything else. Each of the four '
        + 'investigations is driven through a scripted interaction and checked twice over for the leak that '
        + 'matters: the gate marker must be absent AND the graded value\'s own printed text, computed here in '
        + 'Node from the same model the page uses, must not appear in that investigation\'s rendered text until '
        + 'a prediction is committed. Every shape\'s painted stroke width and fill is compared with its '
        + 'attribute, with attribute-free shapes examined rather than skipped. Curves are sampled along their '
        + 'own geometry for label crossings, SVG label sizes are measured after viewBox scaling, table cells are '
        + 'checked for clipping, and every display formula is measured against its column at all three widths.',
      limitations: [
        'Screenshots are captured and hashed here; a human still has to look at them, and this file cannot '
          + 'assert that anyone did.',
        'Keyboard traversal and screen-reader output are not exercised here.',
        'Numerical correctness belongs to scripts/verify-calibration-models.mjs; this run assumes it passed.',
      ],
      passed: true,
    };
    /* Floors on this run's own headline counts. A counter that is reported but
       never floored is decoration: a case that stops running still prints PASS
       with a smaller number nobody reads. The guard audit found four of the
       five verifiers reporting their headline counts unfloored; these are the
       browser layer's. */
    assert.ok(records.length >= 11, `only ${records.length} browser cases ran`);
    assert.ok(screenshots.length >= 51, `only ${screenshots.length} captures were written`);
    assert.ok(layoutStatesChecked.length >= 10,
      `only ${layoutStatesChecked.length} layout states were measured; a figure state stopped being visited`);
    assert.ok(katexSvgsMeasured >= 10,
      `only ${katexSvgsMeasured} KaTeX SVG measurements were taken across ${layoutStatesChecked.length} states`);
    fs.mkdirSync(path.dirname(evidencePath), { recursive: true });
    fs.writeFileSync(evidencePath, JSON.stringify(evidence, null, 2) + '\n');
    console.log(`PASS: ${records.length} browser cases at 1366, 390 and 320 px; ${painted.length} shapes compared `
      + `against their attributes, ${sizes.length} SVG labels measured, ${desktopMath.length} display formulas `
      + `checked for overflow, and ${screenshots.length} distinct captures recorded.`);
  } finally {
    await browser.close();
  }
})().catch(error => {
  /* A thrown assertion skips the evidence write above, which leaves the
     PREVIOUS run's `passed: true` on disk for a tree that fails. Overwrite it
     with a failing record so the evidence directory can never be greener than
     the code. */
  try {
    fs.mkdirSync(path.dirname(evidencePath), { recursive: true });
    fs.writeFileSync(evidencePath, JSON.stringify({
      checkedAt: new Date().toISOString(),
      verifier: 'scripts/verify-calibration-browser.cjs',
      route,
      passed: false,
      failure: { message: String(error && error.message), stack: String(error && error.stack).slice(0, 4000) },
      note: 'This run failed. Written from the failure path so a red tree cannot be read as green from an '
        + 'earlier run.',
    }, null, 2) + '\n');
  } catch { /* the failure itself must still surface below */ }
  console.error(error);
  process.exit(1);
});

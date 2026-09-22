// Production browser review of the end-to-end supervised learning and error
// analysis lesson: visible content, the three investigations, prediction,
// commitment and retirement, the held-out gate, the score-role property, figure
// geometry, narrow layouts and loading closure.
//
// Run against a preview of a production build:
//   npx vite build --outDir dist-e2e
//   npx vite preview --outDir dist-e2e --host 127.0.0.1 --port 4197 --strictPort
//   PLAYWRIGHT_PACKAGE=... DIST_DIR=dist-e2e LEARNING_BASE_URL=http://127.0.0.1:4197 \
//     node scripts/verify-endtoend-browser.cjs
//
// Two checks are why this file exists.
//
//   * SECTION 3, THE HELD-OUT GATE. This lesson's whole argument is that a
//     development selection score is not an independent estimate, and the
//     easiest way for the lesson to contradict itself is to have the held-out
//     number readable before it is earned. So the assertion is a property on the
//     rendered text of the WHOLE page, pinned numerically: the exact six-decimal
//     text of every held-out quantity -- computed independently in this process
//     from endtoend-models.js and endtoend-data.js, not read off the page --
//     must be ABSENT from the document before the decision is frozen and
//     PRESENT after. The absence assertion is paired with the presence assertion
//     so that a pin which could never match is caught rather than counted.
//
//   * SECTION 4, KATEX'S OWN SVGs. A bare `svg` rule under the lesson root also
//     matches KaTeX's radical and stretchy-delimiter SVGs, whose height comes
//     from `height: inherit`; a height rule there collapses them to nothing and
//     silently turns a formula into a different formula. This page renders three
//     of them. Every one is measured for BOTH width and height, because a
//     collapsed radical is also narrow and a width-only check reads as correct
//     while the defect ships.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { gzipSync } = require('node:zlib');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');

const distDir = process.env.DIST_DIR || 'dist-e2e';
const base = (process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4197').replace(/\/+$/, '');
const topicId = 'end-to-end-supervised-learning-error-analysis';
const sourcePath = `src/learn/data/topics/${topicId}.jsx`;
const route = `${base}/learn/path/full-curriculum/${topicId}?module=classical-ml`;
/* `--no-evidence` sends both the report and the captures somewhere disposable.
 *
 * The falsification harness runs this file against a deliberately broken page.
 * It snapshotted the evidence in its own heap and restored it in a `finally`,
 * which covers a throw but NOT a forced kill -- and on this machine a forced
 * kill delivers no catchable signal. A `--browser` case is the widest kill
 * window there is (a production build plus a browser run), so a kill inside one
 * would have left the broken page's report and captures on disk as the record,
 * with no recovery path: sources had an on-disk sidecar, evidence had nothing.
 * The cheapest and most complete fix is that a falsification run never writes
 * the record at all. */
const keepEvidence = !process.argv.includes('--no-evidence');
const evidencePath = keepEvidence
  ? 'docs/teaching/evidence/endtoend-browser.json'
  : 'scratch/endtoend/falsification-browser.json';
const shotDirectory = keepEvidence
  ? 'docs/teaching/evidence/screenshots'
  : 'scratch/endtoend/falsification-screenshots';
const read = filename => JSON.parse(fs.readFileSync(filename, 'utf8'));
const hash = filename => createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
const normalize = text => text.replace(/\s+/g, ' ').trim();
const LESSON = '.endtoend-lesson';
const ownedFiles = [
  sourcePath,
  'src/learn/data/endtoend-models.js',
  'src/learn/data/endtoend-data.js',
  'src/learn/data/endtoend-examples.js',
  'src/learn/components/lesson-labs/EndToEndShared.jsx',
  'src/learn/components/lesson-labs/EndToEndLabs.jsx',
  'src/learn/components/lesson-labs/EndToEndFigures.jsx',
  'src/learn/components/lesson-labs/endtoend-labs.css',
  `src/learn/data/curriculum/blueprints/${topicId}.js`,
  'public/learn-assets/end-to-end/wine.csv',
  'public/learn-assets/end-to-end/wine_study.py',
  'public/learn-assets/end-to-end/ATTRIBUTION.txt',
];

/** Runs in the page. Samples every curve along its own geometry and reports the
 * labels it passes through.
 *
 * Copied from scripts/verify-bias-variance-browser.cjs, because the shared
 * layout inspector iterates `svg.querySelectorAll('line')` and so cannot see a
 * <path>, <polyline> or <polygon> at all.
 *
 * The selector below is checked against what this lesson actually draws: its
 * figures are built from <line>, <rect>, <polygon> and <circle>, so `line` is
 * included and the caller asserts a floor on the number of shapes sampled. A
 * sampler that finds two shapes out of two hundred reports no findings and
 * reads as a pass; that has happened here before.
 *
 * Grid lines are background and are skipped. Nothing on this page carries an
 * opaque backplate, so no crossing is allowed at all. */
function sampleCurvesThroughLabels(root) {
  const SAMPLES = 320;
  const findings = [];
  let shapesSampled = 0;
  for (const svg of document.querySelectorAll(`${root} svg`)) {
    const labels = [...svg.querySelectorAll('text')]
      .filter(text => text.textContent.trim() && text.getClientRects().length)
      .map(text => ({ text: text.textContent.trim().slice(0, 40), box: text.getBoundingClientRect() }));
    if (!labels.length) continue;
    for (const shape of svg.querySelectorAll('path, polyline, polygon, line')) {
      if (shape.classList.contains('ete-grid')) continue;
      if (typeof shape.getTotalLength !== 'function') continue;
      const length = shape.getTotalLength();
      if (!(length > 0)) continue;
      const matrix = shape.getScreenCTM();
      if (!matrix) continue;
      shapesSampled += 1;
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
  return { findings, shapesSampled };
}

/** Runs in the page. Measures every SVG KaTeX renders, in BOTH dimensions. */
function measureKatexSvgs(root) {
  return [...document.querySelectorAll(`${root} .katex svg`)].map(svg => {
    const box = svg.getBoundingClientRect();
    const style = getComputedStyle(svg);
    return {
      width: Number(box.width.toFixed(3)),
      height: Number(box.height.toFixed(3)),
      declaredWidth: svg.getAttribute('width'),
      declaredHeight: svg.getAttribute('height'),
      computedHeight: style.height,
      inside: svg.closest('.katex-display') ? 'display' : 'inline',
    };
  });
}

/** Runs in the page. Every printed score, with the role it carries. */
function collectScoreRoles(root) {
  return [...document.querySelectorAll(`${root} .ete-score`)].map(element => ({
    role: element.getAttribute('data-role'),
    roleText: normalizeText(element.querySelector('.ete-score-role')?.textContent ?? ''),
    value: normalizeText(element.querySelector('.ete-score-value')?.textContent ?? ''),
  }));
  function normalizeText(text) { return text.replace(/\s+/g, ' ').trim(); }
}

/** Runs in the page. Labels that leave their own SVG, or overlap another. */
function inspectLabels(root) {
  const findings = [];
  for (const svg of document.querySelectorAll(`${root} svg`)) {
    const frame = svg.getBoundingClientRect();
    const labels = [...svg.querySelectorAll('text')]
      .filter(text => text.textContent.trim() && text.getClientRects().length);
    labels.forEach((text, index) => {
      const box = text.getBoundingClientRect();
      if (box.left < frame.left - 0.5 || box.right > frame.right + 0.5
        || box.top < frame.top - 0.5 || box.bottom > frame.bottom + 0.5) {
        findings.push({ kind: 'outside', label: text.textContent.trim().slice(0, 40) });
      }
      for (let other = index + 1; other < labels.length; other += 1) {
        const second = labels[other].getBoundingClientRect();
        const overlapX = Math.min(box.right, second.right) - Math.max(box.left, second.left);
        const overlapY = Math.min(box.bottom, second.bottom) - Math.max(box.top, second.top);
        if (overlapX > 0.5 && overlapY > 0.5) {
          findings.push({
            kind: 'overlap',
            label: text.textContent.trim().slice(0, 24),
            other: labels[other].textContent.trim().slice(0, 24),
          });
        }
      }
    });
  }
  return findings;
}

/** Runs in the page. What a shape is actually PAINTED as, not what its
 *  attribute says. A CSS rule beats a presentation attribute, and a shape
 *  painted the page's own ground colour is invisible while every attribute
 *  check passes. */
function inspectPaint(root) {
  const ground = getComputedStyle(document.body).backgroundColor;
  const findings = [];
  for (const svg of document.querySelectorAll(`${root} svg`)) {
    for (const shape of svg.querySelectorAll('rect, circle, polygon, line, polyline, path')) {
      const style = getComputedStyle(shape);
      const fill = style.fill;
      const stroke = style.stroke;
      const invisible = (fill === 'none' || fill === 'rgba(0, 0, 0, 0)')
        && (stroke === 'none' || stroke === 'rgba(0, 0, 0, 0)');
      const black = fill === 'rgb(0, 0, 0)' && (stroke === 'none' || stroke === 'rgba(0, 0, 0, 0)');
      const matchesGround = fill === ground && (stroke === 'none' || stroke === 'rgba(0, 0, 0, 0)');
      if (invisible || black || matchesGround) {
        findings.push({
          shape: shape.getAttribute('class') || shape.tagName,
          fill,
          stroke,
          reason: black ? 'black fill with no stroke' : invisible ? 'no fill and no stroke'
            : 'fill matches the page ground',
          attributeFill: shape.getAttribute('fill'),
          note: shape.getAttribute('fill') && shape.getAttribute('fill') !== fill
            ? 'attribute overridden by a stylesheet' : undefined,
        });
      }
    }
  }
  return findings;
}

/** Runs in the page. Every score-shaped number that is NOT inside a role badge.
 *
 * `collectScoreRoles` enumerates `.ete-score` and checks each match carries a
 * role — its domain is exactly the numbers that already have a badge, so it is
 * structurally incapable of finding one that lacks it. This is the complement,
 * and it is the assertion the lesson's headline invariant actually corresponds
 * to.
 *
 * Four exclusions, each DECLARED ON THE ELEMENT rather than assumed here, so a
 * region the sweep cannot classify is one someone wrote a reason for:
 * `[data-program-output]`, the study program's own stdout, where a badge cannot
 * go because the bytes are the program's; SVG text, which has no room for one
 * and whose figures carry their roles in the adjacent table; a subtree declared
 * a constructed fixture, whose numbers are not scores of this study at all; and
 * an individual `[data-role-exempt]` number with its reason attached. Anything
 * else must sit inside `.ete-score`.
 *
 * The program-output marker exists because the shared `CodeBlock` renders a
 * styled `<div>`, not a `<pre>`, so this could not recognise it by element name
 * — and matching on another component's styling would be a guard that breaks
 * silently when that component changes.
 */
function findUnbadgedScores({ root, heldOutForms }) {
  const findings = [];
  const scoreShaped = /(?<![\d.])\d+\.\d{4,}(?![\d])/;
  /* THE UNBADGED TEXT OF EACH BLOCK, not each text node.
     React renders `{a}/{b}` as three sibling text nodes — "35", "/", "36" — so
     the string `35/36` exists in the rendered block and in NO single node. A
     sweep that tested nodes one at a time could not see it, and the
     falsification case that removes a badge went inert for exactly that reason:
     the guard was looking at a unit smaller than the thing it hunts.
     So the excluded regions are removed and what remains of each block is
     joined. Removing a badged span cannot create a false positive — it can only
     break a string apart — while joining sibling nodes is what makes a real one
     visible. */
  const BLOCK = 'p, li, td, th, dd, dt, figcaption, h1, h2, h3, h4, h5, summary, div, section';
  const groups = new Map();
  const walker = document.createTreeWalker(document.querySelector(root), NodeFilter.SHOW_TEXT);
  while (walker.nextNode()) {
    const node = walker.currentNode;
    if (!node.textContent.trim()) continue;
    const element = node.parentElement;
    if (!element) continue;
    if (element.closest('pre, code, svg')) continue;
    if (element.closest('[data-program-output]')) continue;
    if (element.closest('[data-constructed-fixture]')) continue;
    if (element.closest('.ete-score')) continue;
    if (element.closest('[data-role-exempt]')) continue;
    const block = element.closest(BLOCK) ?? element;
    if (!groups.has(block)) groups.set(block, []);
    groups.get(block).push(node.textContent);
  }
  for (const [block, parts] of groups) {
    const text = parts.join('');
    const matched = text.match(scoreShaped);
    const heldOut = heldOutForms.find(form => text.includes(form));
    if (!matched && !heldOut) continue;
    findings.push({
      text: text.trim().replace(/\s+/g, ' ').slice(0, 90),
      matched: matched ? matched[0] : heldOut,
      reason: matched ? 'a score-shaped decimal outside any role badge'
        : 'a held-out quantity outside any role badge',
      within: block.closest('[data-investigation]')?.dataset?.investigation ?? 'lesson body',
    });
  }
  return findings;
}

/**
 * THE CLASS GUARD for "the page grades a correct answer wrong".
 *
 * Five variants of that defect have now been found across this effort, and each
 * per-lesson guard was written against the previous instance's surface rather
 * than against the invariant, so the class kept resurfacing. The invariant is:
 *
 *   the quantity a question NAMES is the quantity its grader COMPARES, and the
 *   reference is the value the page ITSELF DISPLAYS for that quantity.
 *
 * So this does not inspect code. It commits once to open the reveal, reads the
 * number the page prints in the element tagged with the question's own
 * `quantityKey`, resets, types exactly that number back into the field, commits
 * again, and requires the verdict to say it matches. A grader pointed at a
 * neighbouring quantity fails here even when every offline check agrees with
 * itself, because the page's own displayed value is the arbiter.
 *
 * It also asserts the negative: a number one away from the displayed one is
 * rejected. Without that, a grader that accepted everything would pass.
 */
async function assertGradesWhatItDisplays({
  page, assert, investigation, quantityKey, choice, label, captureAs, capture,
}) {
  const root = await page.$(investigation);
  assert.ok(root, `${label}: the investigation was not found`);
  const reset = async () => {
    await root.$eval('header button', node => node.click());
    await page.waitForTimeout(120);
  };
  const commit = async guess => {
    await root.$eval(`input[type="radio"][value="${choice}"]`, node => node.click());
    if (guess !== null) {
      await root.$eval('.ete-numeric-guess input', (node, value) => {
        const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
        setter.call(node, value);
        node.dispatchEvent(new Event('input', { bubbles: true }));
      }, String(guess));
    }
    await root.$eval('button.is-primary', node => node.click());
    await page.waitForTimeout(120);
    return (await root.evaluate(node => node.textContent)).replace(/\s+/g, ' ').trim();
  };

  await reset();
  await commit(null);
  const displayed = await root.$eval(`[data-graded-quantity="${quantityKey}"]`,
    node => node.textContent.trim());
  assert.ok(displayed.length > 0, `${label}: the page displays nothing for ${quantityKey}`);
  const shownValue = Number(displayed);
  assert.ok(Number.isFinite(shownValue),
    `${label}: the page displays "${displayed}" for ${quantityKey}, which is not a number`);

  await reset();
  const matched = await commit(shownValue);
  assert.ok(/which matches/.test(matched),
    `${label}: the page displays ${shownValue} for the quantity the question names, but typing that exact `
    + `number is not graded as matching. The verdict said: ${matched.slice(matched.indexOf('You wrote'), 260)}`);
  /* CAPTURED HERE, with the number typed in.
     The review found this lesson's acceptance capture had been taken with the
     numeric field EMPTY, so the verdict in the evidence carried no numeric
     clause and the one image that could have exhibited the blocking defect
     never exercised the path. The capture is taken at the moment the graded
     number is on screen. */
  if (captureAs && capture) await capture(await page.$(investigation), captureAs);

  await reset();
  const mismatched = await commit(shownValue + 1);
  assert.ok(/which does not match/.test(mismatched),
    `${label}: a number one away from the displayed value was accepted, so the grader accepts anything`);
  await reset();
  return { quantityKey, displayed: shownValue };
}

(async () => {
  const startedAt = new Date().toISOString();
  const sourceHashes = Object.fromEntries(ownedFiles.map(filename => [filename, hash(filename)]));
  const build = read(`${distDir}/.vite/manifest.json`);
  const buildHash = hash(`${distDir}/.vite/manifest.json`);
  const publications = read('src/learn/data/lesson-manifest.json');
  const { tracks } = await import('../src/learn/data/tracks.js');
  const { endToEndData } = await import('../src/learn/data/endtoend-data.js');
  const models = await import('../src/learn/data/endtoend-models.js');
  const { endToEndExamples } = await import('../src/learn/data/endtoend-examples.js');

  /* Computed here, in this process, from the model layer -- never read off the
     page. A pin taken from the page can only ever agree with it. */
  const heldOutPins = [
    endToEndData.heldOut.accuracy.value.toFixed(6),
    endToEndData.heldOut.balancedAccuracy.value.toFixed(6),
    endToEndData.heldOut.logLoss.value.toFixed(6),
  ];
  const wilson = models.wilsonInterval(endToEndData.heldOut.wilson.successes,
    endToEndData.heldOut.wilson.trials, endToEndData.heldOut.wilson.z);
  /* The Wilson bounds are a scale check and §8 prints them to three decimals,
     as the manuscript states them. Pinning them at six -- the precision every
     other computed value on this page uses -- was a pin that could never match:
     the absence assertion passed trivially and proved nothing. It was the
     PAIRED presence assertion that caught it, which is why the two are always
     written together. The pin is still computed here from the model layer, as
     the bracketed pair the page actually renders. */
  /* The two bounds SEPARATELY, not as a bracketed pair.
     They were pinned as `[lower, upper]` when they rendered as one run of text;
     badging each bound put a role label between them, so the pair stopped being
     a contiguous string and the absence assertion started passing for a reason
     that had nothing to do with the gate. The paired presence assertion caught
     it — twice now, on the same two numbers, which is the argument for the rule
     in one line. */
  const wilsonPins = [models.fixed(wilson.lower, 3), models.fixed(wilson.upper, 3)];
  /* Every textual form a held-out quantity takes on the page. The badge sweep
     uses it to insist each one sits inside a role badge once the gate opens. */
  const heldOutTextualForms = [
    ...heldOutPins,
    `${endToEndData.heldOut.correct} of ${endToEndData.heldOut.total}`,
    ...wilsonPins,
    /* THE BARE FORMS TOO, which is the whole point.
       This list originally held only the forms the page produces when the badge
       is present -- "35 of 36" is what `<Count>` renders. But an omission does
       not produce the badged form; it produces `35/36`, written by hand in
       prose, which is exactly what the review found in the compact report. A
       sweep that knows only the badged spelling cannot see the unbadged one,
       and the falsification case for this guard was inert until these were
       added. They must never appear anywhere outside a badge. */
    `${endToEndData.heldOut.correct}/${endToEndData.heldOut.total}`,
  ];
  const heldOutProgramLine = endToEndExamples.study.heldOutOutput.split('\n')[0];

  const module = tracks.find(track => track.id === 'classical-ml');
  const sourceKey = `src/learn/data/topics/${topicId}.jsx`;
  const bodyFile = build[sourceKey].file;
  const bodyFiles = new Set(Object.values(publications)
    .map(value => build[`src/learn/data/${value.replace(/^\.\//, '')}`]?.file).filter(Boolean));
  const allowedScripts = new Set();
  function addClosure(key) {
    if (!build[key] || allowedScripts.has(build[key].file)) return;
    allowedScripts.add(build[key].file);
    for (const child of build[key].imports || []) addClosure(child);
  }
  addClosure(Object.keys(build).find(key => build[key].isEntry && key.endsWith('index.html')));
  addClosure('src/learn/Reader.jsx');
  addClosure(sourceKey);

  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [];
  const screenshots = [];
  let status = 'failed';
  /* Declared here because the capture helper below reads the viewport, and the
     helper is defined before the page is opened. */
  let page;
  fs.mkdirSync(shotDirectory, { recursive: true });

  /* A tall element is captured with a viewport tall enough to hold it.
     Chromium stitches an element screenshot that exceeds the viewport and the
     seam paints as an opaque band across the image, which blanked part of a
     figure in the first Phase C run. Evidence that cannot be read is not
     evidence, so the viewport is grown for the shot and restored afterwards. */
  const capture = async (target, name) => {
    const file = path.join(shotDirectory, `endtoend-${name}.png`).replace(/\\/g, '/');
    assert.ok(!screenshots.some(shot => shot.file === file), `${file} would be captured twice`);
    const viewport = page.viewportSize();
    let grown = false;
    let clip = null;
    if (typeof target.boundingBox === 'function') {
      const first = await target.boundingBox();
      if (first && first.height > viewport.height - 40) {
        await page.setViewportSize({ width: viewport.width, height: Math.ceil(first.height) + 120 });
        grown = true;
        /* Scrolled to the document top before the clip is taken. The reader's
           header is sticky, so at any other scroll position it paints as an
           opaque band across whatever part of the element happens to be level
           with it -- which covered this investigation's own role note. At the
           top of the document it sits above the content it belongs to. */
        await page.evaluate(() => window.scrollTo(0, 0));
        await page.waitForTimeout(200);
        /* Captured as a page-coordinate CLIP rather than by asking the element
           to screenshot itself.
           An element screenshot scrolls the element to the top of the viewport,
           which is exactly where the reader's sticky header sits, so the
           element's own heading came back covered by it. A full-page capture
           clipped to the element's page box has no scroll position to get
           wrong. The re-measure matters: growing the viewport reflows, so the
           box taken before the resize is stale. */
        /* PAGE coordinates, not viewport-relative. `boundingBox()` returns the
           box relative to the viewport, and a full-page clip is measured from
           the top of the document; handing one to the other clips an empty
           region. The scroll offset is added here rather than assumed to be
           zero. */
        clip = await target.evaluate(element => {
          const box = element.getBoundingClientRect();
          return {
            x: Math.max(box.x + window.scrollX, 0),
            y: Math.max(box.y + window.scrollY, 0),
            width: box.width,
            height: box.height,
          };
        });
      }
    }
    if (clip) await page.screenshot({ path: file, fullPage: true, clip });
    else await target.screenshot({ path: file });
    if (grown) await page.setViewportSize(viewport);
    const digest = hash(file);
    const bytes = fs.statSync(file).size;
    assert.ok(!screenshots.some(shot => shot.digest === digest),
      `${file} is byte-identical to an earlier capture, so one of them shows the wrong state`);
    screenshots.push({ file, digest, bytes, name });
    return file;
  };

  try {
    const context = await browser.newContext({ viewport: { width: 1366, height: 1000 } });
    const requests = [];
    context.on('request', request => requests.push(request.url()));
    page = await context.newPage();
    const consoleErrors = [];
    page.on('console', message => { if (message.type() === 'error') consoleErrors.push(message.text()); });
    page.on('pageerror', error => consoleErrors.push(String(error)));

    // ------------------------------------------------------ 1. the page exists
    await page.goto(route, { waitUntil: 'networkidle' });
    await page.waitForSelector(LESSON, { timeout: 30000 });
    assert.equal(module.topicIds[38], topicId, 'the lesson sits at position 39 of its module');
    assert.ok(publications[topicId], 'the topic is registered for publication');
    const headings = await page.$$eval(`${LESSON} h2`, nodes => nodes.map(node => node.textContent.trim()));
    assert.ok(headings.length >= 9, `only ${headings.length} sections rendered`);
    const anchors = await page.$$eval('.lesson-intro nav a', nodes => nodes.map(node => node.getAttribute('href')));
    assert.ok(anchors.length >= 9, `the route list has only ${anchors.length} links`);
    /* Resolved with getElementById, not a CSS selector. Every section id on this
       page begins with its section number, and an identifier starting with a
       digit is not a valid CSS selector at all -- `page.$('#1-write-the-...')`
       throws rather than returning null, so a selector-based check here reports
       a harness error instead of a missing anchor. */
    const missingAnchors = await page.evaluate(hrefs => hrefs
      .filter(href => !document.getElementById(decodeURIComponent(href.slice(1)))), anchors);
    assert.deepEqual(missingAnchors, [], 'a route link points at a section that does not exist');
    assert.deepEqual(consoleErrors, [], `the page logged errors: ${consoleErrors.join(' | ')}`);
    records.push({ case: 'The lesson renders at position 39 with every section and a working route list',
      sections: headings.length, anchors: anchors.length });
    await capture(page, 'page-desktop-before-gate');

    // -------------------------------------------- 2. every score carries a role
    const roles = await page.evaluate(collectScoreRoles, LESSON);
    assert.ok(roles.length >= 25, `only ${roles.length} printed scores were found; the page shows more`);
    for (const score of roles) {
      assert.ok(models.SCORE_ROLES.includes(score.role),
        `a printed score carries the role "${score.role}", which is not one of the four`);
      assert.ok(score.roleText.length > 0,
        `a printed score's role is in an attribute but not visible as text: ${JSON.stringify(score)}`);
      assert.equal(score.roleText, models.ROLE_LABELS[score.role],
        'the visible role text beside a printed score differs from the role it carries');
      assert.ok(score.value.length > 0, 'a printed score has a role but no value');
    }
    const rolesSeen = new Set(roles.map(score => score.role));
    assert.ok(rolesSeen.has('training') && rolesSeen.has('validation'),
      `only the roles ${[...rolesSeen]} appear before the gate; training and validation both should`);
    assert.ok(!rolesSeen.has('held-out'),
      'a held-out score is printed before the decision that earns it has been frozen');
    /* The complement. `collectScoreRoles` can only inspect numbers that already
       carry a badge; this finds the ones that do not, which is the assertion
       the lesson's headline invariant actually corresponds to. */
    const unbadgedBefore = await page.evaluate(findUnbadgedScores, { root: LESSON, heldOutForms: heldOutTextualForms });
    assert.deepEqual(unbadgedBefore, [],
      'a score-shaped number is printed outside any role badge, against the invariant this lesson states '
      + 'in its own opening callout');
    const fixtures = await page.$$eval(`${LESSON} [data-constructed-fixture]`,
      nodes => nodes.map(node => node.dataset.constructedFixture));
    assert.equal(fixtures.length, 2,
      `${fixtures.length} subtrees are declared constructed fixtures; two are — the acceptance queue and `
      + 'practice 1, both of which invent their numbers rather than measuring this study');
    for (const reason of fixtures) {
      assert.ok(reason.length > 20, `a constructed-fixture declaration gives no reason: "${reason}"`);
    }
    const programOutputs = await page.$$eval(`${LESSON} [data-program-output]`,
      nodes => nodes.map(node => node.dataset.programOutput));
    assert.ok(programOutputs.length >= 1,
      'no program-output region is declared, so the unbadged-score sweep would flag the stdout of the study '
      + 'program itself');
    for (const reason of programOutputs) {
      assert.ok(reason.length > 20, `a program-output exclusion gives no reason: "${reason}"`);
    }
    const exempt = await page.$$eval(`${LESSON} [data-role-exempt]`,
      nodes => nodes.map(node => node.dataset.roleExempt));
    assert.ok(exempt.length >= 3, `only ${exempt.length} numbers declare a role exemption`);
    for (const reason of exempt) {
      assert.ok(reason.length > 20, `a role exemption gives no reason: "${reason}"`);
    }
    records.push({ case: 'Every printed score carries one of the four roles, visibly as well as in an attribute, '
      + 'and every number that does not is a declared exemption with a stated reason',
      scores: roles.length, rolesBeforeGate: [...rolesSeen],
      declaredFixtures: fixtures.length, declaredExemptions: exempt.length,
      declaredProgramOutputs: programOutputs.length });

    // ------------------------------------- 3. no held-out quantity before the gate
    const textBefore = normalize(await page.$eval(LESSON, node => node.textContent));
    for (const pin of [...heldOutPins, ...wilsonPins]) {
      assert.ok(!textBefore.includes(pin),
        `the held-out quantity ${pin} is readable before the decision it reports on has been frozen`);
    }
    /* ONE pin, not three, and paired below.
       This looped over the three numbers on the program's held-out line and
       asserted the absence of `test 0.966667`, `test 0.972222` and
       `test 0.128708`. The word `test` precedes exactly one of them, so two of
       the three iterations could never match anything and neither had a
       presence partner. The whole line is the string the page either shows or
       does not. */
    assert.ok(!textBefore.includes(heldOutProgramLine),
      `the study program's held-out line is readable before the gate: ${heldOutProgramLine}`);
    assert.ok(!textBefore.includes('[[12'), 'the held-out confusion matrix is readable before the gate');
    const sealed = await page.$$eval(`${LESSON} .ete-sealed`, nodes => nodes.length);
    assert.ok(sealed >= 3, `only ${sealed} sealed regions are shown; the report, the final paragraph and the `
      + 'Wilson check all quote held-out quantities');
    records.push({ case: 'No held-out quantity is in the document before the decision is frozen',
      pinnedQuantities: [...heldOutPins, ...wilsonPins], sealedRegions: sealed });

    // --------------------------------------------- 4. KaTeX's own SVGs survive
    const katexErrors = await page.$$eval(`${LESSON} .katex-error`, nodes => nodes.length);
    assert.equal(katexErrors, 0, `${katexErrors} KaTeX expressions failed to parse`);
    const katexSvgs = await page.evaluate(measureKatexSvgs, LESSON);
    assert.ok(katexSvgs.length >= 3,
      `only ${katexSvgs.length} KaTeX SVGs were measured; this page renders a radical and two stretchy `
      + 'delimiters, so a smaller number means the selector has stopped matching and the check is inert');
    /* A floor of "not zero" is weaker than it looks: a rule can leave a radical
       at a fraction of a pixel and still clear it. Four pixels is below every
       measurement this page produces and far above a collapse. */
    const KATEX_FLOOR = 4;
    for (const svg of katexSvgs) {
      assert.ok(svg.height >= KATEX_FLOOR,
        `a KaTeX SVG paints with height ${svg.height}: a square-root radical paints with no height or width `
        + 'when a stylesheet rule reaches it, and the formula then says something different');
      assert.ok(svg.width >= KATEX_FLOOR, `a KaTeX SVG paints with width ${svg.width}`);
    }
    records.push({ case: 'Every SVG KaTeX renders keeps a nonzero width AND height on the real page',
      katexSvgs: katexSvgs.length, measurements: katexSvgs });

    // ------------------------------------------- 5. figure geometry and paint
    const labelFindings = await page.evaluate(inspectLabels, LESSON);
    assert.deepEqual(labelFindings, [], 'a figure label leaves its SVG or overlaps another label');
    const curves = await page.evaluate(sampleCurvesThroughLabels, LESSON);
    assert.ok(curves.shapesSampled >= 40,
      `the curve sampler examined only ${curves.shapesSampled} shapes; this lesson draws far more, so the `
      + 'selector has stopped matching what it draws and the check is inert');
    assert.deepEqual(curves.findings, [], 'a drawn line or shape runs through a figure label');
    const paint = await page.evaluate(inspectPaint, LESSON);
    assert.deepEqual(paint, [], 'a shape is painted invisibly, or black with no stroke, or the page ground');
    records.push({ case: 'No label leaves its SVG, no shape runs through a label, nothing is painted invisibly',
      shapesSampled: curves.shapesSampled });
    for (const [index, figure] of (await page.$$(`${LESSON} .ete-figure`)).entries()) {
      await capture(figure, `figure-${index + 1}-desktop`);
    }

    // --------------------------------- 6. investigation B grades what it says
    const lab = await page.$(`${LESSON} .ete-investigation`);
    const labText = normalize(await lab.evaluate(node => node.textContent));
    const defaultComparison = models.sliceComparison({
      reference: 'linear_two', candidate: 'linear_three', cutoffHundredths: 400, side: 'lower',
    });
    assert.ok(!labText.includes(`${defaultComparison.candidateErrors} of ${defaultComparison.slice.n}`),
      'the slice explorer shows its graded error count before a prediction is committed');
    await capture(lab, 'investigation-b-before');
    await page.check(`${LESSON} .ete-investigation input[type="radio"][value="more"]`);
    await page.click(`${LESSON} .ete-investigation button.is-primary`);
    const afterText = normalize(await lab.evaluate(node => node.textContent));
    assert.ok(afterText.includes(`${defaultComparison.candidateErrors} of ${defaultComparison.slice.n}`),
      'the slice explorer does not show its graded error count after a prediction is committed, so the '
      + 'absence assertion above could never have failed');
    assert.ok(afterText.includes('matches'), 'a correct prediction is graded as matching');
    for (const id of defaultComparison.brokenIds) {
      assert.ok(afterText.includes(String(id)), `the newly wrong specimen ${id} is not named in the reveal`);
    }
    await capture(lab, 'investigation-b-after');
    records.push({ case: 'The slice explorer withholds its graded count until a prediction is committed, '
      + 'then names the specimens that moved',
      gradedCount: defaultComparison.candidateErrors, slice: defaultComparison.slice.n });

    // ------------------------------------------ 7. the gate opens, and only then
    const freeze = (await page.$$(`${LESSON} .ete-investigation`))[1];
    await freeze.scrollIntoViewIfNeeded();
    await capture(freeze, 'investigation-d-before');
    await freeze.$eval('input[type="radio"][value="linear_three"]', node => node.click());
    await freeze.$eval('button.is-primary', node => node.click());
    await page.waitForSelector(`${LESSON} .ete-heldout-step`, { timeout: 10000 });
    const midText = normalize(await page.$eval(LESSON, node => node.textContent));
    for (const pin of heldOutPins) {
      assert.ok(!midText.includes(pin),
        `the held-out quantity ${pin} became readable after the selection but before the second commitment`);
    }
    assert.ok(midText.includes(models.fixed(
      models.asSelectionCriterion(models.candidateByKey.linear_three.validationBalancedAccuracy).value, 6)),
    'the selection criterion is shown once the rule has been applied');
    await capture(freeze, 'investigation-d-selection-made');
    const step = await page.$(`${LESSON} .ete-heldout-step`);
    await step.$eval('input[type="radio"][value="higher"]', node => node.click());
    await step.$eval('button.is-primary', node => node.click());
    await page.waitForFunction(
      pin => document.querySelector('.endtoend-lesson').textContent.includes(pin),
      heldOutPins[1], { timeout: 10000 });
    const textAfter = normalize(await page.$eval(LESSON, node => node.textContent));
    for (const pin of [...heldOutPins, ...wilsonPins]) {
      assert.ok(textAfter.includes(pin),
        `the held-out quantity ${pin} is still missing after the decision was frozen, so the absence `
        + 'assertion in section 3 could never have failed');
    }
    /* The partners for the two absence assertions in section 3. Without these,
       either could pass because the string it hunts had stopped existing for a
       reason unrelated to the gate -- which is exactly what happened to the
       Wilson pin when its two bounds were badged. */
    assert.ok(textAfter.includes(heldOutProgramLine),
      'the held-out line the study program printed is still missing after the gate opened, so the absence '
      + 'assertion on it could never have failed');
    assert.ok(textAfter.includes('[[12'),
      'the held-out confusion matrix is still missing after the gate opened, so the absence assertion on it '
      + 'could never have failed');
    const rolesAfter = await page.evaluate(collectScoreRoles, LESSON);
    assert.ok(rolesAfter.some(score => score.role === 'held-out'),
      'no score is labelled held-out after the report opens');
    assert.ok(rolesAfter.some(score => score.role === 'selection'),
      'no score is labelled selection after a selection has been made');
    for (const score of rolesAfter) {
      assert.ok(models.SCORE_ROLES.includes(score.role),
        `a score printed after the gate carries the role "${score.role}"`);
    }
    const unbadgedAfter = await page.evaluate(findUnbadgedScores, { root: LESSON, heldOutForms: heldOutTextualForms });
    assert.deepEqual(unbadgedAfter, [],
      'a held-out quantity is printed outside any role badge once the report opens — the one place the '
      + 'lesson most loudly promises a badge');
    await capture(freeze, 'investigation-d-report-open');
    /* The report itself, not just the investigation that opened it. Its three
       scores must all be labelled held-out, and nothing else on the page may
       be. */
    const report = await page.$(`${LESSON} .ete-heldout-report`);
    assert.ok(report, 'the held-out report region is not in the document after the gate opened');
    await report.scrollIntoViewIfNeeded();
    await capture(report, 'held-out-report');
    const reportRoles = await page.evaluate(collectScoreRoles, `${LESSON} .ete-heldout-report`);
    assert.ok(reportRoles.length >= 3, `the report prints only ${reportRoles.length} scores`);
    assert.ok(reportRoles.some(score => score.role === 'held-out'),
      'the report prints no score labelled held-out');
    for (const score of reportRoles) {
      assert.ok(models.SCORE_ROLES.includes(score.role),
        `a score in the held-out report carries the role "${score.role}"`);
    }
    records.push({ case: 'The held-out report is absent before both commitments and complete after them',
      reportScores: reportRoles.length,
      reportRoles: [...new Set(reportRoles.map(score => score.role))],
      scoresAfter: rolesAfter.length,
      rolesAfter: [...new Set(rolesAfter.map(score => score.role))] });

    // ---------------------------- 8. a selection the study never tested is refused
    await freeze.$eval('button', node => node.click()); // Reset
    await page.waitForFunction(
      pin => !document.querySelector('.endtoend-lesson').textContent.includes(pin),
      heldOutPins[1], { timeout: 10000 });
    const afterReset = normalize(await page.$eval(LESSON, node => node.textContent));
    for (const pin of heldOutPins) {
      assert.ok(!afterReset.includes(pin), `resetting the investigation left ${pin} on the page`);
    }
    await freeze.$eval('input[type="checkbox"][value], .ete-checkbox input', node => node.click());
    await freeze.$$eval('.ete-checkbox input', nodes => {
      nodes.forEach(node => { if (node.checked) node.click(); });
      nodes[2].click();
    });
    await freeze.$eval('input[type="radio"][value="forest_two"]', node => node.click());
    await freeze.$eval('button.is-primary', node => node.click());
    const refusedText = normalize(await freeze.evaluate(node => node.textContent));
    assert.ok(refusedText.includes('No held-out evidence exists'),
      'selecting a candidate this study never evaluated on the test rows does not refuse a held-out number');
    for (const pin of heldOutPins) {
      assert.ok(!refusedText.includes(pin), `a refused selection still exposed ${pin}`);
    }
    await capture(freeze, 'investigation-d-refused');
    records.push({ case: 'A selection the study never evaluated on the test rows is refused a held-out number '
      + 'rather than given one' });

    // ---------------------------------- 9. the acceptance queue and its reversal
    const acceptance = (await page.$$(`${LESSON} .ete-investigation`))[2];
    await acceptance.scrollIntoViewIfNeeded();
    const baseline = models.acceptanceLedger({ thresholdHundredths: 60, wrongCost: 10, deferCost: 2 });
    const proposed = models.acceptanceLedger({ thresholdHundredths: 80, wrongCost: 10, deferCost: 2 });
    const acceptanceBefore = normalize(await acceptance.evaluate(node => node.textContent));
    assert.ok(acceptanceBefore.includes(String(baseline.cost)),
      'the active rule\'s own ledger is shown, because it is the baseline the prose states');
    assert.ok(!acceptanceBefore.includes(`Proposed · 0.8`),
      'the proposed rule\'s ledger is shown before a prediction is committed');
    await capture(acceptance, 'investigation-c-before');
    await acceptance.$eval('input[type="radio"][value="lower"]', node => node.click());
    await acceptance.$eval('button.is-primary', node => node.click());
    const acceptanceAfter = normalize(await acceptance.evaluate(node => node.textContent));
    assert.ok(acceptanceAfter.includes(String(proposed.cost)),
      'the proposed rule\'s total cost is not shown after the prediction is committed');
    assert.ok(acceptanceAfter.includes('matches'), 'the correct cost direction is graded as matching');
    await capture(acceptance, 'investigation-c-after');
    records.push({ case: 'The acceptance queue withholds the proposed ledger until a prediction is committed',
      baselineCost: baseline.cost, proposedCost: proposed.cost });

    // ---------------- 9b. every graded question grades the quantity it names
    const gradedQuestions = [];
    for (const question of [
      { investigation: `${LESSON} [data-investigation="slice"]`, quantityKey: 'slice-candidate-errors',
        choice: 'more', label: 'investigation B, the error count the candidate makes in the slice',
        captureAs: 'investigation-b-graded-with-number' },
      { investigation: `${LESSON} [data-investigation="acceptance"]`,
        quantityKey: 'acceptance-proposed-cost', choice: 'lower',
        label: 'investigation C, the total cost the proposed rule has',
        captureAs: 'investigation-c-graded-with-number' },
    ]) {
      gradedQuestions.push(await assertGradesWhatItDisplays({ page, assert, capture, ...question }));
    }
    assert.equal(gradedQuestions.length, 2,
      `${gradedQuestions.length} graded numeric questions were exercised; the page has two`);
    /* Every numeric question on the page declares the key of the value it is
       graded against, so none can be added without being covered. */
    const declaredKeys = await page.evaluate(() => [...document.querySelectorAll('.ete-numeric-guess')].length);
    assert.equal(declaredKeys, gradedQuestions.length,
      `${declaredKeys} numeric fields exist but ${gradedQuestions.length} were checked against the value the `
      + 'page displays; a graded question without that check is how this defect class keeps returning');
    records.push({
      case: 'Every graded numeric question grades the quantity it names, checked against the number the page '
        + 'itself displays for that quantity, with a neighbouring value rejected',
      questions: gradedQuestions,
    });

    // -------------------------------------------- 10. narrow layouts and overflow
    for (const width of [390, 320]) {
      await page.setViewportSize({ width, height: 1000 });
      await page.waitForTimeout(150);
      const overflow = await page.evaluate(() => ({
        scroll: document.documentElement.scrollWidth,
        client: document.documentElement.clientWidth,
      }));
      assert.ok(overflow.scroll <= overflow.client + 1,
        `the page scrolls horizontally at ${width}px: ${overflow.scroll} against ${overflow.client}`);
      const wideMath = await page.$$eval('.katex-display', (nodes, limit) => nodes
        .map(node => ({ text: node.textContent.trim().slice(0, 40), width: node.scrollWidth }))
        .filter(entry => entry.width > limit), width - 32);
      assert.deepEqual(wideMath, [], `display math overflows at ${width}px`);
      const narrowKatex = await page.evaluate(measureKatexSvgs, LESSON);
      assert.equal(narrowKatex.length, katexSvgs.length,
        `${narrowKatex.length} KaTeX SVGs were measured at ${width}px against ${katexSvgs.length} at `
        + 'desktop; the selector has stopped matching and the check is inert');
      for (const svg of narrowKatex) {
        assert.ok(svg.height >= KATEX_FLOOR && svg.width >= KATEX_FLOOR,
          `a KaTeX SVG paints ${svg.width} by ${svg.height} at ${width}px, at or below the ${KATEX_FLOOR}px `
          + 'floor: a collapsed radical is narrow as well as short, so both dimensions are checked');
      }
      const narrowLabels = await page.evaluate(inspectLabels, LESSON);
      assert.deepEqual(narrowLabels, [], `a figure label leaves its SVG or overlaps another at ${width}px`);
      await capture(page, `page-${width}`);
      const figures = await page.$$(`${LESSON} .ete-figure`);
      for (const [index, figure] of figures.slice(0, 4).entries()) {
        await capture(figure, `figure-${index + 1}-${width}`);
      }
      records.push({ case: `No horizontal overflow, no overflowing display math and no collapsed KaTeX SVG `
        + `at ${width}px`, width, displayBlocks: (await page.$$('.katex-display')).length,
      katexMeasurements: narrowKatex });
    }

    // ------------------------------- 11. tables keep their columns at desktop
    await page.setViewportSize({ width: 1366, height: 1000 });
    await page.waitForTimeout(150);
    const clipped = await page.$$eval(`${LESSON} .ete-table-scroll`, nodes => nodes
      .map(node => ({
        caption: node.getAttribute('aria-labelledby'),
        scroll: node.scrollWidth,
        client: node.clientWidth,
      }))
      .filter(entry => entry.scroll > entry.client + 1));
    assert.deepEqual(clipped, [], 'a figure table clips a column at desktop width');
    /* Both halves. `.ete-table caption` can never match, because `Table` renders
       its caption as a sibling paragraph by deliberate convention and has no
       `<caption>` path at all -- so the negative alone was structurally
       incapable of failing. What makes it meaningful is asserting that the
       sibling caption it was replaced by exists, on every table. */
    const captionShape = await page.$$eval(`${LESSON} .ete-table`, tables => ({
      tables: tables.length,
      insideCaptions: tables.filter(table => table.querySelector('caption')).length,
      siblingCaptions: tables.filter(table => table.querySelector(':scope > p.ete-caption')).length,
    }));
    assert.ok(captionShape.tables >= 10, `only ${captionShape.tables} tables were inspected`);
    assert.equal(captionShape.insideCaptions, 0,
      'a table caption is inside the table element, where it collapses to its longest word when the table '
      + 'blockifies at narrow width');
    assert.equal(captionShape.siblingCaptions, captionShape.tables,
      `${captionShape.siblingCaptions} of ${captionShape.tables} tables carry the sibling caption paragraph `
      + 'the convention requires; a table with no caption at all would have passed the negative check alone');
    records.push({ case: 'No table clips a column at desktop, and every caption sits outside its scroll box '
      + 'as a sibling paragraph rather than inside the table',
    tables: captionShape.tables, siblingCaptions: captionShape.siblingCaptions });

    // ------------------------------------------------- 12. loading closure
    const localScripts = [...new Set(requests.map(address => new URL(address))
      .filter(address => address.origin === new URL(base).origin && address.pathname.endsWith('.js'))
      .map(address => address.pathname.slice(1)))];
    assert.deepEqual(localScripts.filter(filename => bodyFiles.has(filename)), [bodyFile],
      'the route downloaded another lesson\'s body');
    assert.deepEqual(localScripts.filter(filename => !allowedScripts.has(filename)), [],
      'the route downloaded a script outside this lesson\'s closure');
    assert.ok(!requests.some(address => address.includes('/outlines/')),
      'a published lesson fetched an authoring outline');
    assert.ok(!requests.some(address => address.includes('wine.csv') && !address.includes('end-to-end')),
      'the page requested a Wine table from somewhere other than its own asset directory');
    for (const sibling of ['learn-assets/pca', 'learn-assets/bayesian-networks', 'learn-assets/feature-selection']) {
      assert.ok(!requests.some(address => address.includes(sibling)),
        `the page requested another lesson's asset directory: ${sibling}`);
    }
    records.push({
      case: 'Fresh route requests only this lesson and its shared closure, and never another lesson\'s Wine table',
      requestedScripts: localScripts.length,
      rawScriptBytes: localScripts.reduce((sum, filename) => sum + fs.statSync(path.join(distDir, filename)).size, 0),
      gzipEstimateBytes: localScripts.reduce(
        (sum, filename) => sum + gzipSync(fs.readFileSync(path.join(distDir, filename))).length, 0),
    });

    assert.deepEqual(consoleErrors, [], `the page logged errors: ${consoleErrors.join(' | ')}`);

    // ------------------------------------------------- 13. the captures themselves
    const onDisk = fs.readdirSync(shotDirectory)
      .filter(name => name.startsWith('endtoend-') && name.endsWith('.png'))
      .map(name => path.join(shotDirectory, name).replace(/\\/g, '/'));
    const written = new Set(screenshots.map(shot => shot.file));
    assert.deepEqual(onDisk.filter(file => !written.has(file)), [],
      'orphaned endtoend-*.png files remain from an earlier run');
    for (const shot of screenshots) {
      assert.ok(fs.existsSync(shot.file), `${shot.file} was recorded but is not on disk`);
      assert.equal(hash(shot.file), shot.digest, `${shot.file} no longer matches its recorded digest`);
      assert.equal(fs.statSync(shot.file).size, shot.bytes, `${shot.file} no longer matches its recorded size`);
      assert.ok(shot.bytes > 1000, `${shot.file} is ${shot.bytes} bytes, which is not a rendered capture`);
    }
    /* Printed tallies nobody floors are decoration: a whole section could be
       deleted and the run would still report passed. */
    assert.ok(records.length >= 12, `only ${records.length} browser cases ran; a section has been lost`);
    assert.ok(screenshots.length >= 24, `only ${screenshots.length} captures were taken`);
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
      verifierSha256: hash('scripts/verify-endtoend-browser.cjs'),
      records,
      screenshots,
      screenshotCount: screenshots.length,
      visualInterpretation: 'Screenshots capture informative states, not defaults only: the page before the '
        + 'held-out gate opens, each investigation before any commitment and after one, the selection made, '
        + 'the report open, a refused selection, and every inline figure at desktop plus a selection at 390 '
        + 'and 320 px. They require separate visual inspection; a recorded digest proves identity, not '
        + 'legibility.',
      limitations: [
        'A passing run is not a reading of the page. The captures above must be opened and looked at.',
        'The painted-against-declared check catches a stylesheet overriding an attribute, a shape painting '
          + 'black with no stroke, and a shape painting the page ground. It does NOT catch a shape that is '
          + 'painted correctly and then COVERED by a later sibling; the figure-by-figure visual pass covers '
          + 'that.',
        'The held-out property is asserted on rendered text. It establishes that the quantities are not in '
          + 'the document; it does not establish that the JavaScript bundle lacks them, which it does not '
          + 'and cannot, since the page computes the report once the gate opens.',
        'Model correctness is checked by scripts/verify-endtoend-models.mjs, data regeneration by '
          + 'scripts/verify-endtoend-data.py and program execution by scripts/verify-endtoend-examples.py.',
        'Screen-reader behaviour is not simulated here; roles, labels and descriptions are asserted '
          + 'structurally.',
      ],
    };
    fs.mkdirSync(path.dirname(evidencePath), { recursive: true });
    fs.writeFileSync(evidencePath, `${JSON.stringify(report, null, 2)}\n`);
    console.log(JSON.stringify({
      status: report.status, cases: records.length, evidencePath,
      sourceFiles: ownedFiles.length, screenshots: screenshots.length,
    }));
  }
})();

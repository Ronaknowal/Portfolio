// Production browser review of the Bayesian-networks lesson: visible content,
// the four investigations and their prediction/commit/retirement contract, the
// seven inline figures, graph geometry, display-math width, narrow layouts,
// sequence, completion and load-failure recovery.
//
// Run against a preview of a production build:
//   npx vite build --outDir dist-bn
//   npx vite preview --outDir dist-bn --host 127.0.0.1 --port 4191
//   PLAYWRIGHT_PACKAGE=... DIST_DIR=dist-bn LEARNING_BASE_URL=http://127.0.0.1:4191 \
//     node scripts/verify-bayesnet-browser.cjs
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const { inspectLessonVisualLayout } = require('./lib/lesson-visual-layout.cjs');

const distDir = process.env.DIST_DIR || 'dist-bn';
const base = (process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4191').replace(/\/+$/, '');
const topicId = 'bayesian-networks-causal-graphical-models';
const sourcePath = `src/learn/data/topics/${topicId}.jsx`;
const route = `${base}/learn/path/full-curriculum/${topicId}?module=classical-ml`;
const evidencePath = 'docs/teaching/evidence/bayesnet-browser.json';
const read = filename => JSON.parse(fs.readFileSync(filename, 'utf8'));
const hash = filename => createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
const normalize = text => text.replace(/\s+/g, ' ').trim();
const ownedFiles = [
  sourcePath,
  'src/learn/data/bayesnet-models.js',
  'src/learn/data/bayesnet-data.js',
  'src/learn/data/bayesnet-examples.js',
  'src/learn/components/lesson-labs/BayesNetShared.jsx',
  'src/learn/components/lesson-labs/BayesNetLabs.jsx',
  'src/learn/components/lesson-labs/BayesNetFigures.jsx',
  'src/learn/components/lesson-labs/bayesnet-labs.css',
  `src/learn/data/curriculum/blueprints/${topicId}.js`,
  'public/learn-assets/bayesian-networks/wine.csv',
  'public/learn-assets/bayesian-networks/ATTRIBUTION.txt',
  'public/learn-assets/bayesian-networks/network-experiments.py',
];

/** Runs in the page. Samples every curve along its own geometry and reports the
 * labels it passes through.
 *
 * `scripts/lib/lesson-visual-layout.cjs` iterates `svg.querySelectorAll('line')`,
 * so <path> and <polyline> are invisible to it — which is how four data curves
 * crossed value labels behind a fully green run. This lesson draws highlighted
 * path trails as polylines and fill edges as paths, so the same class is open
 * here; this closes it.
 *
 * Only one skip is granted, and it is the only one this lesson earns. Two
 * others were carried over from a sibling lesson and matched nothing: a
 * backplate allowance for `bn-halo`, a class that appears nowhere in this tree,
 * and a background allowance for `bn-grid`, which is styled but applied to no
 * element. A documented exemption for a class that does not exist describes a
 * leniency the guard does not actually have, so both are gone.
 *
 * Anything marked `bn-underlay` is background and is skipped.
 * The highlighted path trail is an underlay by construction: it joins node
 * centres, so it must pass under the labels of the nodes on the path, and it is
 * emitted before every node group so the nodes paint over it. That exemption is
 * only honest if the document order really is that way, so `checkUnderlayOrder`
 * below asserts it separately rather than trusting the class name.
 */
function sampleCurvesThroughLabels(root) {
  const SAMPLES = 320;
  const findings = [];
  for (const svg of document.querySelectorAll(`${root} svg`)) {
    const labels = [...svg.querySelectorAll('text')]
      .filter(text => text.textContent.trim() && text.getClientRects().length)
      .map(text => ({
        text: text.textContent.trim().slice(0, 40),
        box: text.getBoundingClientRect(),
      }));
    if (!labels.length) continue;
    for (const shape of svg.querySelectorAll('path, polyline, polygon')) {
      if (shape.classList.contains('bn-underlay')) continue;
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
        findings.push({
          label: labels[index].text,
          shape: shape.getAttribute('class') || shape.tagName,
          fractionOfCurveInsideLabel: Number((count / (SAMPLES + 1)).toFixed(4)),
        });
      });
    }
  }
  return findings;
}

/** Runs in the page. Reports the RENDERED pixel height of every visible SVG
 * label, not its user-unit font-size.
 *
 * A floor that reads the `font-size` attribute passes on a diagram scaled to a
 * third of its viewBox, and it also passes on a page with no SVG text at all.
 * This returns the subject count as well, so the caller can refuse an empty set.
 */
function measureSvgLabelPixels(root) {
  const sizes = [];
  for (const text of document.querySelectorAll(`${root} svg text`)) {
    if (!text.textContent.trim() || !text.getClientRects().length) continue;
    const box = text.getBoundingClientRect();
    sizes.push({
      text: text.textContent.trim().slice(0, 30),
      renderedPixels: Number(box.height.toFixed(2)),
      declared: Number(getComputedStyle(text).fontSize.replace('px', '')),
    });
  }
  return sizes;
}

/** Runs in the page. An underlay must precede every node it passes under.
 *
 * `sampleCurvesThroughLabels` skips anything marked `bn-underlay`. That skip is
 * only defensible while the element really is painted first: SVG has no
 * z-index, so paint order is document order. If an underlay ever moved after a
 * node group it would be drawn ON TOP of the labels and the skip would be
 * hiding a real defect.
 */
function checkUnderlayOrder(root) {
  const problems = [];
  for (const svg of document.querySelectorAll(`${root} svg`)) {
    const children = [...svg.querySelectorAll('*')];
    const underlays = children.filter(node => node.classList.contains('bn-underlay'));
    const nodeGroups = children.filter(node => node.classList.contains('bn-node'));
    if (!underlays.length || !nodeGroups.length) continue;
    const lastUnderlay = children.indexOf(underlays[underlays.length - 1]);
    const firstNode = children.indexOf(nodeGroups[0]);
    if (lastUnderlay > firstNode) {
      problems.push({ svg: svg.getAttribute('aria-label')?.slice(0, 60) ?? '(unlabelled)', lastUnderlay, firstNode });
    }
  }
  return problems;
}

/** Runs in the page. A figure's table that scrolls at desktop width is hiding
 * part of what the figure claims to show.
 *
 * Two of this lesson's figures put a five-column table into a half-width flex
 * column, and the column that fell outside the scroll box was in both cases the
 * one the surrounding prose pointed at: the second identification route, and
 * the second potential outcome. Nothing overflowed the page, no assertion
 * failed, and the claim was simply invisible. Investigation tables are exempt,
 * because a thirty-six-row working surface may legitimately scroll.
 */
function measureFigureTableClipping(root) {
  return [...document.querySelectorAll(`${root} .bn-figure .bn-table-scroll`)]
    .map(node => ({
      caption: (node.previousElementSibling?.textContent || '').trim().slice(0, 56),
      clientWidth: node.clientWidth,
      scrollWidth: node.scrollWidth,
      hidden: node.scrollWidth - node.clientWidth,
    }))
    .filter(entry => entry.hidden > 2);
}

/**
 * Runs in the page. What each investigation is showing, and every number in it.
 *
 * The contract printed inside every prediction box is "The answer appears once
 * a prediction is recorded". Asserting the absence of the verdict *banner* does
 * not test that: three investigations printed the graded quantity in ordinary
 * adjacent text while no banner existed. So this reports the marker class the
 * labs put on every graded readout, and also every numeric token in the
 * section, so the caller can check a specific value numerically rather than by
 * guessing at its formatting.
 */
function readInvestigations(root) {
  return [...document.querySelectorAll(`${root} .bn-investigation`)].map(section => {
    const text = section.textContent.replace(/\s+/g, ' ');
    return {
      title: (section.querySelector('h3')?.textContent ?? '').trim().slice(0, 60),
      graded: section.querySelectorAll('.bn-graded').length,
      verdicts: section.querySelectorAll('.bn-verdict').length,
      pathVerdicts: section.querySelectorAll('.bn-path-list li.is-active, .bn-path-list li.is-blocked').length,
      inputValues: [...section.querySelectorAll('input[type="number"]')].map(item => item.value),
      numbers: (text.match(/\d+\.\d+|\d+/g) ?? []).map(Number).filter(Number.isFinite),
      text,
    };
  });
}

/** Runs in the page. Display math that scrolls has overflowed its column. */
function measureDisplayMathOverflow(root) {
  return [...document.querySelectorAll(`${root} .katex-display`)]
    .map(node => ({
      text: node.textContent.trim().slice(0, 48),
      scrollWidth: node.scrollWidth,
      clientWidth: node.clientWidth,
      overflow: node.scrollWidth - node.clientWidth,
    }))
    .filter(entry => entry.overflow > 1);
}

(async () => {
  const startedAt = new Date().toISOString();
  const sourceHashes = Object.fromEntries(ownedFiles.map(filename => [filename, hash(filename)]));
  const build = read(`${distDir}/.vite/manifest.json`);
  const buildHash = hash(`${distDir}/.vite/manifest.json`);
  const { tracks } = await import('../src/learn/data/tracks.js');
  const { bayesnetExamples } = await import('../src/learn/data/bayesnet-examples.js');
  const data = await import('../src/learn/data/bayesnet-data.js');
  const models = await import('../src/learn/data/bayesnet-models.js');
  const module = tracks.find(track => track.id === 'classical-ml');
  assert.equal(module.topicIds[27], topicId, 'the lesson sits at position 28 of its module');
  const bodyFile = build[sourcePath].file;

  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [];
  const screenshots = [];
  /** Console errors from whichever page is being driven. A lesson that throws
   * during render is caught by React's boundary and never reaches `pageerror`,
   * so the console is the only place its message appears. */
  const pageConsoleErrors = [];
  const watchConsole = page => page.on('console', message => {
    if (message.type() === 'error') pageConsoleErrors.push(message.text().split('\n')[0].slice(0, 300));
  });
  const checkText = async (locator, pattern) => assert.match(normalize(await locator.innerText()), pattern);
  const settle = page => page.evaluate(() =>
    new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  /**
   * Wait for the lesson, but lose the race to the error boundary rather than to
   * a timeout.
   *
   * A throw in the lesson's module scope is caught by React's boundary, so it
   * never reaches `pageerror`; it surfaces only as a console error while the
   * body never mounts. Waiting for `.bn-lesson` alone then reports a bare
   * thirty-second timeout and says nothing about the cause. This races the two
   * and, if the boundary wins, fails with the actual message the page logged.
   */
  const ready = async page => {
    const outcome = await Promise.race([
      page.locator('.bn-lesson').waitFor({ timeout: 30000 }).then(() => 'rendered'),
      page.locator('.lesson-load-error').waitFor({ timeout: 30000 }).then(() => 'boundary'),
    ]).catch(() => 'timeout');
    if (outcome !== 'rendered') {
      const logged = pageConsoleErrors.length
        ? pageConsoleErrors.join('\n')
        : '(the page logged no console error)';
      throw new Error(`The lesson body did not render (${outcome}). The page reported:\n${logged}`);
    }
    await page.waitForFunction(() => [...document.fonts]
      .some(font => font.family.includes('Space Grotesk') && font.status === 'loaded'), null, { timeout: 20000 });
    await page.evaluate(() => document.fonts.ready);
    await settle(page);
  };

  try {
    const context = await browser.newContext({ viewport: { width: 1366, height: 1000 } });
    const page = await context.newPage();
    const errors = [];
    const failedAssets = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('requestfailed', request => failedAssets.push(request.url()));
    watchConsole(page);
    await page.goto(route, { waitUntil: 'domcontentloaded' });
    await ready(page);

    /** Every capture gets a unique path, and its digest and byte count are
     * recorded. A stale orphan from a pre-fix run is caught by the digest, not
     * by the filename. */
    const shoot = async (locator, filename) => {
      const destination = path.join('docs/teaching/evidence/screenshots', filename);
      fs.mkdirSync(path.dirname(destination), { recursive: true });
      assert.ok(!screenshots.some(entry => entry.file === destination),
        `Screenshot path reused: ${destination}`);
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
      const digest = hash(destination);
      assert.ok(bytes > 2000, `Screenshot ${destination} is ${bytes} bytes, which is too small to show anything`);
      assert.ok(!screenshots.some(entry => entry.digest === digest),
        `Screenshot ${destination} is byte-identical to ${screenshots.find(entry => entry.digest === digest)?.file}`);
      screenshots.push({ file: destination, digest, bytes });
    };

    // ------------------------------------------------------------ 1. structure
    await checkText(page.locator('.reader-header h1'), /^Bayesian Networks & Causal Graphical Models$/);
    await checkText(page.locator('.reader-header__meta'), /28 of \d+ topics on this route/);
    await checkText(page.locator('.reader-footer__previous'), /Hidden Markov/i);
    await checkText(page.locator('.reader-footer__next'), /Conditional Random Fields/i);
    assert.equal(await page.locator('.bn-investigation').count(), 4, 'four investigations render');
    assert.equal(await page.locator('.bn-practice').count(), 10, 'ten practice sections render');
    assert.equal(await page.locator('.python-example').count(), 2, 'two displayed programs render');
    assert.ok(await page.locator('.bn-figure').count() >= 7, 'at least the seven inline figures render');
    const rendered = normalize(await page.locator('.bn-lesson').textContent());
    for (const key of ['enumerate', 'pgmpy']) {
      const example = bayesnetExamples[key];
      assert.ok(rendered.includes(normalize(example.code)), `Missing actual code for ${key}`);
      assert.ok(rendered.includes(normalize(example.expected)), `Missing actual output for ${key}`);
    }
    assert.ok(rendered.includes(data.provenance.sha256), 'the served dataset digest is printed on the page');
    // Section ids begin with their number, so they are legal HTML ids but not
    // legal bare CSS selectors. Address them by attribute, as the other
    // lessons' reviews do; a fragment link resolves regardless.
    const anchors = await page.locator('.lesson-intro a[href^="#"]')
      .evaluateAll(items => items.map(item => item.getAttribute('href')));
    assert.equal(anchors.length, 10, 'the route lists all ten sections');
    for (const anchor of anchors) {
      assert.equal(await page.locator(`[id="${anchor.slice(1)}"]`).count(), 1,
        `Route anchor ${anchor} resolves to exactly one heading`);
    }
    assert.equal(await page.locator('.katex-error').count(), 0, 'no formula failed to render');
    // The downloads the prose offers must actually be served.
    for (const href of ['/learn-assets/bayesian-networks/wine.csv',
      '/learn-assets/bayesian-networks/network-experiments.py',
      '/learn-assets/bayesian-networks/ATTRIBUTION.txt']) {
      const response = await page.request.get(`${base}${href}`);
      assert.equal(response.status(), 200, `${href} is served`);
    }
    records.push({ case: 'Header, position, neighbours, four investigations, ten practice sections, both executed programs, every route anchor and all three served downloads' });

    // ------------------------------- 2. nothing is revealed before commitment
    //
    // Asserted as a property of every investigation, not as the absence of one
    // banner: the graded quantity is marked `bn-graded` wherever it is shown,
    // and none of those elements may exist before a prediction is recorded.
    const atFirstPaint = await page.evaluate(readInvestigations, '.bn-lesson');
    assert.equal(atFirstPaint.length, 4, 'all four investigations are on the page');
    for (const section of atFirstPaint) {
      assert.equal(section.graded, 0,
        `"${section.title}" displays the graded quantity before any prediction is recorded`);
      assert.equal(section.verdicts, 0, `"${section.title}" shows a verdict on first paint`);
      assert.equal(section.pathVerdicts, 0,
        `"${section.title}" labels a path blocked or active before the learner has answered that question`);
    }
    assert.equal(await page.locator('.bn-history').count(), 0, 'and no history either');
    for (const index of [0, 1, 2, 3]) {
      const apply = page.locator('.bn-investigation').nth(index).locator('button.is-primary');
      assert.ok(await apply.isDisabled(), `Investigation ${index + 1} cannot be applied without a prediction`);
    }
    // Investigation 2's verdict is reachable at first paint with no edit at all,
    // so its path list must not state one. Read the words, not just the classes.
    assert.ok(!/\bB–A–E\s*blocked\b/.test(atFirstPaint[1].text),
      'investigation 2 prints its verdict in the path list on first paint');
    // The state the blocking review finding was about: investigation 2 before
    // the learner has touched anything. Captured so it can be looked at.
    await shoot(page.locator('.bn-investigation').nth(1), 'bayesnet-investigation-2-first-paint-1366.png');
    await shoot(page.locator('.bn-investigation').nth(3), 'bayesnet-investigation-4-first-paint-1366.png');
    records.push({ case: 'Before any prediction, no investigation renders a graded readout, a verdict, a path verdict or a history entry, and every Apply is disabled' });

    // --------------- 2b. and the graded VALUE is absent, checked numerically
    //
    // The structural marker above is the general property. This pins two
    // specific numbers, because a readout added without the marker would pass
    // the class check while printing the answer in plain text.
    const near = (values, target) => values.some(value =>
      Math.abs(value - target) <= 1e-4 * Math.max(1, Math.abs(target)));

    const evidenceLabEarly = page.locator('.bn-investigation').nth(0);
    await evidenceLabEarly.getByRole('button', { name: /Reveal that there was an earthquake/i }).click();
    const draftPosterior = models.queryPosterior(models.alarmNetwork, { J: 1, M: 1, E: 1 }).posterior;
    const afterEdit = await page.evaluate(readInvestigations, '.bn-lesson');
    assert.equal(afterEdit[0].graded, 0, 'investigation 1 still shows no graded readout after an edit');
    assert.ok(!near(afterEdit[0].numbers, draftPosterior),
      `investigation 1 prints the posterior it is about to grade (${draftPosterior}) before commitment`);
    await evidenceLabEarly.getByRole('button', { name: /^Reset$/ }).click();

    /* The changed-response preset, not the randomised one. Under randomisation
       the draft's observed difference is .07, which is also the BASELINE's
       causal difference and therefore legitimately on screen; the probe would
       flag a coincidence rather than a leak. This preset's graded value,
       .1125, collides with nothing the panel is entitled to show. */
    const interventionEarly = page.locator('.bn-investigation').nth(3);
    await interventionEarly.getByRole('button', { name: /Make the low-load response harmless/i }).click();
    const draftService = models.serviceModel(models.fixtures.serviceChangedResponse);
    const afterService = await page.evaluate(readInvestigations, '.bn-lesson');
    assert.equal(afterService[3].graded, 0, 'investigation 4 still shows no graded readout after an edit');
    assert.ok(!near(afterService[3].numbers, draftService.associationDifference),
      `investigation 4 prints the observed difference it is about to grade `
      + `(${draftService.associationDifference}) before commitment`);
    await interventionEarly.getByRole('button', { name: /^Reset$/ }).click();
    records.push({ case: 'After an edit and before commitment, neither investigation 1 nor 4 prints the value it is about to grade, checked numerically rather than by formatting' });

    // ---------------------- 3. investigation 2 grades a correct answer right
    const pathLab = page.locator('.bn-investigation').nth(1);
    await pathLab.scrollIntoViewIfNeeded();
    // The alarm collider with nothing observed: the graph guarantees independence.
    await pathLab.getByRole('radio', { name: /guarantees independence/i }).check();
    await pathLab.locator('button.is-primary').click();
    await checkText(pathLab.locator('.bn-verdict'), /Your prediction matches/);
    // The gate opens: the marker that was absent before commitment is present
    // after it, so the readouts were withheld rather than deleted.
    const afterCommit = await page.evaluate(readInvestigations, '.bn-lesson');
    assert.ok(afterCommit[1].graded > 0,
      'investigation 2 shows its graded readout once a prediction is recorded');
    assert.ok(afterCommit[1].pathVerdicts > 0, 'and the path list now states its verdicts');
    await shoot(pathLab, 'bayesnet-investigation-2-correct-separation-1366.png');
    records.push({ case: 'Investigation 2 accepts the correct separation verdict for the unobserved collider, and reveals its path verdicts only then' });

    // Now observe the descendant and check the opposite verdict is accepted.
    await pathLab.getByRole('button', { name: /^Reset$/ }).click();
    await pathLab.getByRole('checkbox', { name: /^J$/ }).check();
    assert.equal(await pathLab.locator('.bn-verdict').count(), 0, 'editing the observation set retires the verdict');
    await pathLab.getByRole('radio', { name: /leaves dependence possible/i }).check();
    await pathLab.locator('button.is-primary').click();
    await checkText(pathLab.locator('.bn-verdict'), /Your prediction matches/);
    await checkText(pathLab.locator('.bn-path-list'), /descendant/);
    await shoot(pathLab, 'bayesnet-investigation-2-collider-descendant-1366.png');
    records.push({ case: 'Observing a collider descendant retires the verdict, opens the path, and the drawn reason names the descendant rule' });

    // An invalid edit is refused beside the control without destroying state.
    await pathLab.getByRole('button', { name: /^Reset$/ }).click();
    await pathLab.getByRole('combobox', { name: /Add or remove the arrow from/i }).selectOption('A');
    await pathLab.getByRole('combobox', { name: /…to/i }).selectOption('B');
    await pathLab.getByRole('button', { name: /Add this arrow/i }).click();
    await checkText(pathLab.locator('[role="alert"]'), /cycle|already present/i);
    assert.ok(await pathLab.locator('.bn-path-list li').count() > 0, 'the graph on screen survives a refused edit');
    records.push({ case: 'A cycle-creating edge is refused beside the control and the prior valid graph survives' });

    // ------------------------ 4. investigation 1 grades an exact null as null
    const evidenceLab = page.locator('.bn-investigation').nth(0);
    await evidenceLab.getByRole('button', { name: /Fix the alarm at 1 and keep only John/i }).click();
    await evidenceLab.getByRole('radio', { name: /^higher than the applied state$/ }).check();
    await evidenceLab.locator('button.is-primary').click();
    await shoot(evidenceLab, 'bayesnet-investigation-1-screened-off-1366.png');
    await evidenceLab.getByRole('button', { name: /^Reset$/ }).click();
    await evidenceLab.getByRole('button', { name: /Make John impossible/i }).click();
    await evidenceLab.getByRole('radio', { name: /no posterior at all/i }).check();
    await evidenceLab.locator('button.is-primary').click();
    await checkText(evidenceLab.locator('.bn-verdict'), /Your prediction matches/);
    await checkText(evidenceLab.locator('.bn-verdict'), /probability is exactly 0/);
    await shoot(evidenceLab, 'bayesnet-investigation-1-impossible-evidence-1366.png');
    records.push({ case: 'Investigation 1 reports impossible evidence as having no posterior, and accepts that as the correct prediction' });

    // ---------------------------- 5. investigation 3 and 4 informative states
    const measurementLab = page.locator('.bn-investigation').nth(2);
    await measurementLab.scrollIntoViewIfNeeded();
    // Nothing is revealed yet, so none of this specimen's four measured values
    // may be readable anywhere in the panel -- including as the prefilled value
    // of the control that edits them. A learner who can read all four off the
    // edit boxes has been handed the answer to the question being asked.
    const selectedLabel = await measurementLab
      .getByRole('combobox', { name: /Validation specimen/i }).inputValue();
    const selectedId = Number(selectedLabel.replace(/[^0-9]/g, ''));
    const firstSpecimen = data.validationSpecimens.find(row => row.id === selectedId)
      ?? data.validationSpecimens[0];
    const panel = (await page.evaluate(readInvestigations, '.bn-lesson'))[2];
    for (const measured of firstSpecimen.features) {
      const printed = String(measured);
      assert.ok(!panel.inputValues.includes(printed),
        `Specimen ${firstSpecimen.id}'s unrevealed measurement ${printed} is prefilled into an edit control`);
      /* Compared as a NUMBER against every numeric token in the panel. The
         previous version searched for the value surrounded by spaces, and the
         card renders as adjacent inline elements with no separating text
         ("alcohol13.17training median..."), so that half could never match:
         it would have passed just as happily with the value printed. */
      assert.ok(!panel.numbers.some(value => Math.abs(value - measured) < 1e-9),
        `Specimen ${firstSpecimen.id}'s unrevealed measurement ${printed} is printed in the panel`);
    }
    records.push({ case: 'Investigation 3 reveals none of the four measured values before they are bought, including through the controls that edit them' });
    await measurementLab.getByRole('radio').first().check();
    await measurementLab.locator('button.is-primary').click();
    await checkText(measurementLab.locator('.bn-verdict'), /nats/);
    await shoot(measurementLab, 'bayesnet-investigation-3-purchase-1366.png');
    const interventionLab = page.locator('.bn-investigation').nth(3);
    await interventionLab.scrollIntoViewIfNeeded();
    await interventionLab.getByRole('button', { name: /Assign the procedure at random/i }).click();
    await interventionLab.getByRole('radio', { name: /^it falls$/ }).check();
    await interventionLab.locator('button.is-primary').click();
    await checkText(interventionLab.locator('.bn-verdict'), /observed difference went from/);
    await shoot(interventionLab, 'bayesnet-investigation-4-randomised-1366.png');
    records.push({ case: 'Investigation 3 grades a measurement purchase and investigation 4 a randomised assignment, both with their arithmetic shown' });

    // Sequential topic-28 review: supported tiny masses, comparison semantics,
    // reset, accessible answer gating, and the complete worked frontdoor model.
    const resetLab = async lab => lab.getByRole('button', { name: /^Reset$/ }).click();
    const setNumber = async (lab, label, value) => {
      const input = lab.getByRole('spinbutton', { name: label });
      await input.fill(String(value));
      await input.press('Tab'); // commit/blur the real editor before other changes
    };
    const predict = async (lab, key) => {
      await lab.locator(`.bn-prediction input[type="radio"][value="${key}"]`).check();
      await lab.locator('button.is-primary').click();
      await checkText(lab.locator('.bn-verdict'), /Your prediction matches/);
    };
    const setEvidence = async (node, value) => evidenceLab.locator('.bn-state-choice')
      .filter({ has: page.locator('.bn-state-label', { hasText: new RegExp(`^${node} —`) }) })
      .locator(`input[value="${value}"]`).check();

    for (const [guess, accepted] of [[0.001005, true], [0.000995, true],
      [0.0010050001, false], [0.0009949999, false]]) {
      await resetLab(evidenceLab);
      await setEvidence('J', 'unknown');
      await setEvidence('M', 'unknown');
      await checkText(evidenceLab.locator('.bn-numeric-guess'), /Tolerance: 0.5%.*minimum absolute allowance/);
      await evidenceLab.locator('.bn-numeric-guess input').fill(String(guess));
      await predict(evidenceLab, 'lower');
      await checkText(evidenceLab.locator('.bn-verdict'), accepted ? /within .* of it/ : /outside .* of it/);
    }
    records.push({ case: 'I1 discloses the relative numeric allowance, accepts both inclusive endpoints around the .001 prior, and rejects both just-outside guesses' });

    await resetLab(evidenceLab);
    await setEvidence('E', '0');
    await setEvidence('A', '1');
    await setNumber(evidenceLab, /burglary prior/, 0);
    await setNumber(evidenceLab, /earthquake prior/, 0.9999);
    await setNumber(evidenceLab, /John when the alarm sounds/, 0.001);
    await setNumber(evidenceLab, /Mary when the alarm sounds/, 0.001);
    await predict(evidenceLab, 'lower');
    assert.ok(!/evidence probability is exactly 0|no posterior to report/.test(await evidenceLab.locator('.bn-verdict').textContent()),
      'positive ~1e-13 evidence must not become impossible');
    const rareMassRows = await evidenceLab.locator('.bn-graded tbody tr').evaluateAll(items => items.map(row => [...row.querySelectorAll('th,td')].map(cell => cell.textContent.trim())));
    assert.deepEqual(rareMassRows[1], ['B = 1 (burglary)', '0', '0'], 'supported evidence yields a zero burglary mass and zero share');
    assert.equal(rareMassRows[2][2], '1', 'the positive evidence normalises to one');
    await shoot(evidenceLab, 'bayesnet-sequential-rare-supported-evidence-1366.png');
    await resetLab(evidenceLab);
    await evidenceLab.getByRole('button', { name: /Make John impossible/i }).click();
    await predict(evidenceLab, 'undefined');
    await setNumber(evidenceLab, /earthquake prior/, 0.003);
    await predict(evidenceLab, 'undefined');
    await evidenceLab.getByRole('button', { name: /^Back to both calls$/ }).click();
    await predict(evidenceLab, 'defined');
    records.push({ case: 'I1 preserves positive sub-10^-12 support and a defined zero posterior; impossible-to-impossible remains undefined and recovery is explicitly newly defined' });

    await pathLab.getByRole('combobox', { name: /Add or remove the arrow from/i }).selectOption('A');
    await pathLab.getByRole('combobox', { name: /…to/i }).selectOption('B');
    await resetLab(pathLab);
    assert.equal(await pathLab.getByRole('combobox', { name: /Add or remove the arrow from/i }).inputValue(), 'B');
    assert.equal(await pathLab.getByRole('combobox', { name: /…to/i }).inputValue(), 'K');
    records.push({ case: 'I2 Reset restores the pending arrow selectors as well as the graph, predictions and errors' });

    await resetLab(measurementLab);
    await measurementLab.getByRole('combobox', { name: /Validation specimen/i }).selectOption('156');
    await predict(measurementLab, 'second');
    await measurementLab.getByRole('combobox', { name: /Question to commit on/i }).selectOption('edit');
    await measurementLab.locator('.bn-specimen-card').nth(0).getByRole('checkbox').check();
    await checkText(measurementLab.locator('.bn-prediction'), /probability of cultivar 2.*10⁻¹²/);
    await predict(measurementLab, 'higher');
    await checkText(measurementLab.locator('.bn-verdict'), /revealed evidence changed.*visibility change/);
    await checkText(measurementLab.locator('.bn-history'), /different question/);
    await measurementLab.locator('.bn-specimen-card').nth(2).getByRole('checkbox').check();
    await predict(measurementLab, 'higher');
    await checkText(measurementLab.locator('.bn-history'), /category is the same.*does not establish/);
    await measurementLab.getByRole('button', { name: /Hide every measurement/ }).click();
    await predict(measurementLab, 'lower');
    await measurementLab.getByRole('combobox', { name: /Validation specimen/i }).selectOption('80');
    await predict(measurementLab, 'unchanged');
    await checkText(measurementLab.locator('.bn-verdict'), /revealed binary evidence is identical/);
    records.push({ case: 'I3 names its actual cultivar-2 target, explains mask changes and identical encoded evidence truthfully, and separates purchase/edit history from same-category numerical changes' });

    await resetLab(interventionLab);
    await interventionLab.getByRole('button', { name: /Make the low-load response harmless/i }).click();
    const hiddenLaneNames = await interventionLab.locator('.bn-lane').evaluateAll(items => items.map(item => item.getAttribute('aria-label')));
    assert.equal(hiddenLaneNames.length, 4);
    for (const name of hiddenLaneNames) assert.ok(!/risk (?:undefined|\d)/.test(name), `Draft risk leaked through accessible name: ${name}`);
    await predict(interventionLab, 'lower');
    assert.ok((await interventionLab.locator('.bn-lane').first().getAttribute('aria-label')).includes('risk'),
      'committed risk remains available to assistive technology');
    await setNumber(interventionLab, /P\(procedure \| low load\)/, 0);
    await setNumber(interventionLab, /P\(procedure \| high load\)/, 0);
    await predict(interventionLab, 'undefined');
    await setNumber(interventionLab, /P\(failure \| procedure, low load\)/, 0.02);
    await predict(interventionLab, 'undefined');
    await interventionLab.getByRole('button', { name: /Assign the procedure at random/i }).click();
    await predict(interventionLab, 'defined');
    records.push({ case: 'I4 hides uncommitted risks in accessible names and distinguishes absent, still-absent and newly-defined observed differences' });

    await resetLab(interventionLab);
    await setNumber(interventionLab, /P\(procedure \| low load\)/, 0.001);
    await setNumber(interventionLab, /P\(procedure \| high load\)/, 0.999);
    const givenLane = interventionLab.locator('.bn-lane').filter({ has: page.locator('.bn-caption', { hasText: /^Observed population with the procedure given$/ }) });
    await checkText(givenLane.locator('.bn-lane-key'), /Low load: 0.001.*High load: 0.999/);
    assert.equal(await givenLane.locator('.bn-lane-part-label').count(), 0, 'small fractions have no clipped in-bar label');
    const shares = await givenLane.locator('.bn-lane-part').evaluateAll(items => items.map(item => Number.parseFloat(item.style.width)));
    assert.ok(Math.abs(shares[0] - 0.1) < 1e-10 && Math.abs(shares[1] - 99.9) < 1e-10,
      `Exact proportional shares without a minimum width: ${shares}`);
    await page.setViewportSize({ width: 390, height: 1000 });
    await settle(page);
    assert.equal(await givenLane.locator('.bn-lane-key').evaluate(item => item.scrollWidth > item.clientWidth + 1), false);
    await shoot(interventionLab, 'bayesnet-sequential-tiny-mixture-390.png');
    await page.setViewportSize({ width: 1366, height: 1000 });
    await resetLab(interventionLab);
    records.push({ case: 'I4 keeps .001/.999 weights outside the bars at 390px and preserves exact .1%/99.9% proportional widths' });

    const declaredTable = page.locator('.bn-table').filter({ has: page.locator('.bn-caption', { hasText: /^The declared generative model behind the frontdoor example$/ }) });
    const declaredRows = await declaredTable.locator('tbody tr').evaluateAll(items => items.map(row => [...row.querySelectorAll('th,td')].map(cell => cell.textContent.trim())));
    assert.deepEqual(declaredRows, [
      ['U', 'none', '0.5'], ['X', 'U = 0', '0.2'], ['X', 'U = 1', '0.8'],
      ['M', 'X = 0', '0.1'], ['M', 'X = 1', '0.9'],
      ['Y', 'M = 0, U = 0', '0.05'], ['Y', 'M = 0, U = 1', '0.4'],
      ['Y', 'M = 1, U = 0', '0.5'], ['Y', 'M = 1, U = 1', '0.9'],
    ], 'complete prepared frontdoor generative probabilities are visible, not just derived averages');
    await checkText(declaredTable.locator('..'), /Conditioning on the treatment instead/);
    records.push({ case: 'F5 restores all nine declared generative probabilities and identifies plain observational risks as conditioning on treatment' });

    // ------------------------------------------- 6. figures and their geometry
    const figureCount = await page.locator('.bn-figure').count();
    for (let index = 0; index < figureCount; index += 1) {
      const figure = page.locator('.bn-figure').nth(index);
      if (!(await figure.isVisible())) continue;
      await shoot(figure, `bayesnet-figure-${index + 1}-1366.png`);
    }
    records.push({ case: `All ${figureCount} rendered figures captured at desktop width for inspection` });

    for (const width of [1366, 768, 390, 320]) {
      await page.setViewportSize({ width, height: 1000 });
      await settle(page);
      const issues = await page.evaluate(inspectLessonVisualLayout, '.bn-lesson');
      assert.deepEqual(issues.flatMap(figure => figure.issues), [], `Figure layout collides at ${width}px`);
      // The shared inspector samples straight <line> elements only, so a path
      // trail or a fill arc can run through a label and still report clean.
      const curveHits = await page.evaluate(sampleCurvesThroughLabels, '.bn-lesson');
      assert.deepEqual(curveHits, [], `A curve runs through a label at ${width}px`);
      // The one exemption the sampler grants has to be earned.
      const underlayOrder = await page.evaluate(checkUnderlayOrder, '.bn-lesson');
      assert.deepEqual(underlayOrder, [],
        `An underlay is painted after a node group at ${width}px, so it covers the labels it is exempted for`);
      // Rendered pixels, not user units, and the subject set must be non-empty.
      const labels = await page.evaluate(measureSvgLabelPixels, '.bn-lesson');
      assert.ok(labels.length > 20,
        `Only ${labels.length} SVG labels were measurable at ${width}px, so the size floor checked almost nothing`);
      const smallest = labels.reduce((low, entry) => (entry.renderedPixels < low.renderedPixels ? entry : low));
      assert.ok(smallest.renderedPixels >= 8,
        `The smallest rendered SVG label at ${width}px is ${smallest.renderedPixels}px: "${smallest.text}"`);
      const overflowing = await page.evaluate(measureDisplayMathOverflow, '.bn-lesson');
      assert.deepEqual(overflowing, [], `Display math overflows its column at ${width}px`);
      const horizontal = await page.evaluate(() =>
        document.documentElement.scrollWidth - document.documentElement.clientWidth);
      assert.ok(horizontal <= 1, `The page scrolls horizontally by ${horizontal}px at ${width}px`);
      if (width === 1366) {
        const clipped = await page.evaluate(measureFigureTableClipping, '.bn-lesson');
        assert.deepEqual(clipped, [],
          'A figure table hides part of its content at desktop width, so a column the prose points at is unreadable');
      }
      records.push({ case: `${width}px: no label collision, no curve through a label, every SVG label at least 8 rendered pixels across ${labels.length} labels, no display-math overflow, no horizontal page scroll${width === 1366 ? ', and no figure table clipping a column' : ''}` });
      if (width === 320 || width === 390) {
        // The figures most at risk when the column narrows: the assembly strip,
        // the two chains and their fill arcs, the frontdoor trays, and the
        // paired-outcome matrices. Capturing the first three in DOM order would
        // photograph three small graphs and miss every table.
        for (const index of [0, 4, 9, 11]) {
          await shoot(page.locator('.bn-figure').nth(index), `bayesnet-figure-${index + 1}-${width}.png`);
        }
        await shoot(page.locator('.bn-investigation').nth(1), `bayesnet-investigation-2-${width}.png`);
        await shoot(page.locator('.bn-investigation').nth(3), `bayesnet-investigation-4-${width}.png`);
      }
    }
    await page.setViewportSize({ width: 1366, height: 1000 });

    // ------------------------------------------------------- 7. keyboard route
    await page.keyboard.press('Tab');
    const focusVisible = await page.evaluate(() => {
      const active = document.activeElement;
      if (!active || active === document.body) return null;
      const style = getComputedStyle(active);
      return { tag: active.tagName, outline: style.outlineStyle };
    });
    assert.ok(focusVisible, 'tabbing moves focus off the body');
    // The record said "the focused control has a visible outline style" while
    // the check only confirmed that focus had moved; an outlineStyle of 'none'
    // passed. Assert what the record claims.
    assert.notEqual(focusVisible.outline, 'none',
      `the focused ${focusVisible.tag} has outline-style ${focusVisible.outline}, so focus is not visible`);
    records.push({ case: `Keyboard focus enters the lesson and the focused ${focusVisible.tag} carries a visible outline style (${focusVisible.outline})` });

    // ------------------------------------------------------------ 8. sequence
    await page.locator('.reader-complete').click();
    assert.equal(new URL(page.url()).pathname.split('/').at(-1), topicId);
    await page.reload({ waitUntil: 'domcontentloaded' });
    await ready(page);
    assert.equal(await page.evaluate(id => JSON.parse(localStorage.getItem('kd-progress'))[id], topicId), true);
    records.push({ case: 'Completion persists under the stable ID without auto-advance' });
    assert.deepEqual(errors, [], 'no page error was raised');
    assert.deepEqual(failedAssets, [], 'no asset failed to load');
    await context.close();

    // ------------------------------------------------------------ 9. recovery
    for (const failure of ['import', 'render']) {
      const isolated = await browser.newContext();
      const trial = await isolated.newPage();
      watchConsole(trial);
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

    for (const [filename, expected] of Object.entries(sourceHashes)) {
      assert.equal(hash(filename), expected, `Source changed during the check: ${filename}`);
    }
    assert.equal(hash(`${distDir}/.vite/manifest.json`), buildHash, 'the build did not change during the check');

    const report = {
      startedAt, completedAt: new Date().toISOString(), topicId, status: 'passed',
      browser: await browser.version(), distDir, buildManifestHash: buildHash, sourceHashes,
      moduleTopicCount: module.topicIds.length,
      dataset: { file: data.provenance.file, sha256: data.provenance.sha256, bytes: data.provenance.bytes },
      modelExports: Object.keys(models).length,
      records, screenshots,
      visualInterpretation: 'The intended Space Grotesk font loaded before the checks ran. Screenshots capture '
        + 'informative states rather than defaults only: investigation 2 after a correct separation verdict and '
        + 'again with a collider descendant observed, investigation 1 on the screened-off null and on impossible '
        + 'evidence, investigation 3 after a purchase is graded, investigation 4 under randomised assignment, every '
        + 'inline figure at desktop width, and the first three figures plus the path inspector at 390 and 320 px. '
        + 'Each capture records its own byte count and SHA-256 so a stale file from an earlier run cannot be '
        + 'mistaken for a fresh one. They require separate visual inspection by a person.',
    };
    fs.writeFileSync(evidencePath, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify({
      status: report.status, cases: records.length, evidencePath,
      sourceFiles: ownedFiles.length, screenshots: screenshots.length,
    }));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });

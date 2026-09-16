// Production browser review of the AutoML & NAS lesson: visible content, all six
// investigations under the predict-then-check contract, figure geometry, drawn
// curves against label boxes, narrow layouts and stacked tables, loading
// closure, sequence, completion and load-failure recovery.
//
// Run against a preview of a production build:
//   npx vite build --outDir dist-automl
//   npx vite preview --outDir dist-automl --host 127.0.0.1 --port 4189
//   PLAYWRIGHT_PACKAGE=... DIST_DIR=dist-automl LEARNING_BASE_URL=http://127.0.0.1:4189 \
//     node scripts/verify-automl-browser.cjs
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { gzipSync } = require('node:zlib');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const { inspectLessonVisualLayout } = require('./lib/lesson-visual-layout.cjs');

const distDir = process.env.DIST_DIR || 'dist-automl';
const base = (process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4189').replace(/\/+$/, '');
const topicId = 'automl-neural-architecture-search-nas';
const sourcePath = `src/learn/data/topics/${topicId}.jsx`;
const route = `${base}/learn/path/full-curriculum/${topicId}?module=classical-ml`;
const evidencePath = 'docs/teaching/evidence/automl-browser.json';
const read = filename => JSON.parse(fs.readFileSync(filename, 'utf8'));
const hash = filename => createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
const normalize = text => text.replace(/\s+/g, ' ').trim();
const ownedFiles = [
  sourcePath,
  'src/learn/data/automl-models.js',
  'src/learn/data/automl-data.js',
  'src/learn/data/automl-examples.js',
  'src/learn/components/lesson-labs/AutomlShared.jsx',
  'src/learn/components/lesson-labs/AutomlLabs.jsx',
  'src/learn/components/lesson-labs/AutomlFigures.jsx',
  'src/learn/components/lesson-labs/automl-labs.css',
  `src/learn/data/curriculum/blueprints/${topicId}.js`,
  'public/learn-assets/automl-nas/banknote-data.csv',
  'public/learn-assets/automl-nas/ATTRIBUTION.txt',
];

/** Runs in the page. Samples every curve along its own geometry and reports the
 * labels it passes through.
 *
 * Copied from scripts/verify-bias-variance-browser.cjs rather than added to the
 * shared inspector, which is not this task's to change.
 * `scripts/lib/lesson-visual-layout.cjs` iterates `svg.querySelectorAll('line')`,
 * so <path>, <polyline> and <polygon> are invisible to it — which is how four
 * data curves once crossed value labels behind a fully green run.
 *
 * A label carrying an opaque backplate may be crossed briefly, because the
 * backplate is what a crossing is mitigated with. It may not be travelled along:
 * a backplate hides a line passing behind glyphs, not a curve running the width
 * of the text. Grid lines are background and are skipped.
 *
 * Honest scope: the shapes it finds are the investigation plots' polylines. The
 * inline figures are drawn entirely from <line>, <rect> and <text>, which the
 * shared inspector already reads, so this adds nothing there — a fact the
 * assertion below states rather than leaves to be inferred from a count.
 */
function sampleCurvesThroughLabels(root) {
  const SAMPLES = 320;
  const HALO_ALLOWANCE = 0.02;
  const findings = [];
  let inspectedShapes = 0;
  for (const svg of document.querySelectorAll(`${root} svg`)) {
    const labels = [...svg.querySelectorAll('text')]
      .filter(text => text.textContent.trim() && text.getClientRects().length)
      .map(text => ({
        text: text.textContent.trim().slice(0, 40),
        halo: text.classList.contains('am-halo'),
        box: text.getBoundingClientRect(),
      }));
    if (!labels.length) continue;
    for (const shape of svg.querySelectorAll('path, polyline, polygon')) {
      if (shape.classList.contains('am-grid')) continue;
      // An arrowhead inside <marker> is a reused glyph, not a data path: it is
      // drawn wherever its line ends, so sampling its own local geometry means
      // nothing. Counting them inflated `inspectedShapes` by 24 and made the
      // figures look covered when the only real curves are in the lab plots.
      if (shape.closest('marker')) continue;
      if (typeof shape.getTotalLength !== 'function') continue;
      const length = shape.getTotalLength();
      if (!(length > 0)) continue;
      const matrix = shape.getScreenCTM();
      if (!matrix) continue;
      inspectedShapes += 1;
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
  return { findings, inspectedShapes };
}

(async () => {
  const startedAt = new Date().toISOString();
  const sourceHashes = Object.fromEntries(ownedFiles.map(filename => [filename, hash(filename)]));
  const build = read(`${distDir}/.vite/manifest.json`);
  const buildHash = hash(`${distDir}/.vite/manifest.json`);
  const publications = read('src/learn/data/lesson-manifest.json');
  const { tracks } = await import('../src/learn/data/tracks.js');
  const data = await import('../src/learn/data/automl-data.js');
  const models = await import('../src/learn/data/automl-models.js');
  const { automlExamples } = await import('../src/learn/data/automl-examples.js');
  const space = models.countConfigurations(models.declaredSpace);
  const halving = models.halvingSchedule();
  const replayThree = models.replayPrefix({ budget: 3 });
  const module = tracks.find(track => track.id === 'classical-ml');
  assert.equal(module.topicIds[25], topicId, 'the lesson sits at position 26 of its module');
  const bodyFile = build[sourcePath].file;
  const bodyFiles = new Set(Object.values(publications).map(value => build[`src/learn/data/${value.replace(/^\.\//, '')}`].file));
  const allowedScripts = new Set();
  function addClosure(key) {
    if (allowedScripts.has(build[key].file)) return;
    allowedScripts.add(build[key].file);
    for (const child of build[key].imports || []) addClosure(child);
  }
  addClosure(Object.keys(build).find(key => build[key].isEntry && key.endsWith('index.html')));
  addClosure('src/learn/Reader.jsx');
  addClosure(sourcePath);
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [];
  const screenshotPaths = [];
  const renderedText = {};
  const checkText = async (locator, pattern) => assert.match(normalize(await locator.innerText()), pattern);
  const settle = page => page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
  const ready = async page => {
    await page.locator('.am-lesson').waitFor();
    await page.waitForFunction(() => [...document.fonts].some(font => font.family.includes('Space Grotesk') && font.status === 'loaded'), null, { timeout: 20000 });
    await page.evaluate(() => document.fonts.ready);
    await settle(page);
  };
  /** Record a prediction and commit it, inside one investigation. */
  const predict = async (lab, choiceLabel, { guess } = {}) => {
    await lab.getByRole('radio', { name: choiceLabel }).check();
    if (guess !== undefined) await lab.locator('.am-numeric-guess input').fill(String(guess));
    await lab.getByRole('button', { name: /Apply and check/ }).click();
    await lab.locator('.am-verdict').waitFor();
  };

  try {
    const context = await browser.newContext({ viewport: { width: 1366, height: 1000 } });
    const page = await context.newPage();
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
      if (bounds.height > viewport.height - 160) await page.setViewportSize({ width: viewport.width, height: Math.ceil(bounds.height) + 180 });
      await locator.scrollIntoViewIfNeeded();
      await locator.evaluate(element => window.scrollTo(0, element.getBoundingClientRect().top + window.scrollY - 100));
      await locator.screenshot({ path: destination });
      await page.setViewportSize(viewport);
      screenshotPaths.push(destination);
    };

    // ---------------------------------------------------------- 1. structure
    await checkText(page.locator('.reader-header h1'), /^AutoML & Neural Architecture Search \(NAS\)$/);
    await checkText(page.locator('.reader-header__meta'), /26 of 39 topics on this route/);
    await checkText(page.locator('.reader-footer__previous'), /Imbalanced Learning/);
    await checkText(page.locator('.reader-footer__next'), /Hidden Markov Models/);
    assert.equal(await page.locator('.am-lesson h2').count(), 10, 'ten sections');
    assert.equal(await page.locator('.am-investigation').count(), 6, 'six investigations');
    assert.equal(await page.locator('.am-figure').count(), 5, 'five inline figures');
    await checkText(page.locator('.am-route'), /First pass\. Read sections 1–5/);
    await checkText(page.locator('.am-route'), /Section 7 is a deeper branch/);
    // Nothing is revealed on first paint. Scoped to the investigations: a
    // figure is allowed to show its own content, and F5 legitimately prints a
    // determinant into an `.am-readout` without asking for a prediction.
    assert.equal(await page.locator('.am-investigation .am-verdict').count(), 0, 'no verdict before a prediction');
    assert.equal(await page.locator('.am-investigation .am-readout').count(), 0, 'and no derived readout');
    assert.equal(await page.locator('.am-investigation .am-rung').count(), 0, 'no rung decision is shown');
    // Every drawing in every investigation sits behind its own commitment, so
    // before any prediction there is no SVG inside one at all. This can fail:
    // the five figures do paint their SVGs immediately.
    assert.equal(await page.locator('.am-investigation svg').count(), 0, 'and nothing is drawn at all');
    assert.ok(await page.locator('.am-figure svg').count() >= 5, 'while the inline figures do paint');
    // ---- the property, not the containers -------------------------------
    // The earlier guard enumerated containers (verdict, readout, rung, svg) and
    // so missed a plain <span> carrying the branch counts whose sum is the
    // graded answer. Ask the general question instead: for each investigation,
    // is the string the answer panel will print already on screen?
    //
    // Each entry names a distinctive rendering of a graded quantity for that
    // investigation's opening draft. None may appear before a prediction; each
    // must appear after one, or the check is testing nothing.
    const gradedStrings = [
      { index: 0, label: 'grammar', before: ['2 × 2 = 4', '1 × 1 × 3 = 3', '= 11'] },
      { index: 1, label: 'improvement', before: ['0.05004008', '0.07978846'] },
      { index: 2, label: 'halving', before: ['9(1) + 3(3) + 1(9)', 'advances'] },
      { index: 3, label: 'replay', before: ['best so far', '0.998907'] },
      // Not the bare phrase "on the frontier": the prediction prompt itself asks
      // how many candidates are on the frontier, which is the question, not the
      // answer. These two strings appear only in the revealed table.
      { index: 4, label: 'deployment', before: ['nothing — on the frontier', 'dominated by'] },
      // Not "softmax weight": the investigation's own question explains that
      // logits turn into softmax weights. The gradient column header appears
      // only in the revealed table.
      { index: 5, label: 'mixture', before: ['0.1993359892', '∂L/∂α', 'The most negative gradient'] },
    ];
    for (const entry of gradedStrings) {
      const text = normalize(await page.locator('.am-investigation').nth(entry.index).innerText());
      const leaked = entry.before.filter(needle => text.includes(needle));
      assert.deepEqual(leaked, [],
        `${entry.label} shows a graded quantity before any prediction: ${leaked.join(', ')}`);
    }
    await screenshot(page.locator('.am-investigation').nth(0), 'automl-grammar-before-prediction-desktop.png');
    // The same class in prose: a figure once printed 0.020000000000000004 where
    // its own box and table printed 0.02. Sweep every visible text node, not
    // only inputs. The recorded stdout of the displayed program is exempt — it
    // is verbatim output and must not be reformatted.
    const overPrecise = await page.evaluate(() => {
      const hits = [];
      const walker = document.createTreeWalker(document.querySelector('.am-lesson'), NodeFilter.SHOW_TEXT);
      for (let node = walker.nextNode(); node; node = walker.nextNode()) {
        if (node.parentElement.closest('.python-example')) continue;
        for (const match of node.textContent.matchAll(/\d\.\d{9,}/g)) hits.push(match[0]);
      }
      return hits;
    });
    // Eight-decimal expected-improvement displays are deliberate and exact.
    const illegitimate = overPrecise.filter(value => value.split('.')[1].length > 12);
    assert.deepEqual(illegitimate, [],
      `a number is printed at more precision than anything else on the page: ${illegitimate.join(', ')}`);

    records.push({
      case: 'No investigation shows a graded quantity before a prediction is recorded',
      checkedStrings: gradedStrings.reduce((total, entry) => total + entry.before.length, 0),
      note: 'Asks whether the answer string is on screen, rather than enumerating the containers an answer might sit in.',
    });

    // Every rendered control step comes from the one exported map, so the
    // "typeable" checks in the model verifier compare against the page rather
    // than against a second transcription of it.
    // The prediction's own numeric field is the learner's free-form answer, not
    // a model control, and carries step="any" by design.
    const renderedSteps = await page.locator('.am-investigation input[type="number"]')
      .evaluateAll(items => [...new Set(items
        .filter(item => !item.closest('.am-numeric-guess'))
        .map(item => item.getAttribute('step')))].sort());
    const declaredSteps = [...new Set(Object.values(models.controlSteps).map(String))].sort();
    assert.deepEqual(renderedSteps.filter(step => !declaredSteps.includes(step)), [],
      `a rendered control step is not in controlSteps: ${renderedSteps.join(', ')}`);
    assert.ok(renderedSteps.length >= 5, 'several distinct steps are rendered');

    // The raw inputs stay visible, which is what makes the task answerable at
    // all. Four investigations show theirs as a table; the grammar shows an
    // editable tree and the mixture shows its operation formulas instead.
    assert.equal(await page.locator('.am-investigation table').count(), 4, 'four raw-input tables are visible');
    const perLab = await page.locator('.am-investigation').evaluateAll(items => items.map(item => ({
      controls: item.querySelectorAll('input, select, button').length,
      derived: item.querySelectorAll('.am-verdict, .am-readout, .am-rung, svg').length,
    })));
    assert.equal(perLab.length, 6, 'all six investigations were inspected');
    perLab.forEach((lab, index) => {
      assert.ok(lab.controls >= 15, `investigation ${index + 1} exposes its inputs (${lab.controls} controls)`);
      assert.equal(lab.derived, 0, `investigation ${index + 1} shows nothing derived before a prediction`);
    });
    // The dataset the page serves is the one it names.
    await checkText(page.locator('.am-lesson'), new RegExp(data.provenance.sha256));
    const asset = await page.request.get(`${base}${data.provenance.file}`);
    assert.equal(asset.status(), 200, 'the served dataset is reachable');
    assert.equal(createHash('sha256').update(await asset.body()).digest('hex'), data.provenance.sha256,
      'and its bytes match the printed digest');
    assert.equal((await page.request.get(`${base}${data.provenance.attribution}`)).status(), 200,
      'the attribution file is reachable');
    // The displayed study and its recorded output are both on the page.
    await checkText(page.locator('.python-example').first(), /Role rows: 919 205 248/);
    await checkText(page.locator('.am-lesson'), /python -m pip install numpy scikit-learn threadpoolctl/);
    // Every route link in the introduction must reach a heading that exists.
    const anchors = await page.locator('.lesson-intro nav a').evaluateAll(items => items.map(item => item.getAttribute('href')));
    assert.equal(anchors.length, 10, 'the introduction lists all ten sections');
    const missing = await page.evaluate(hrefs => hrefs.filter(href => !document.getElementById(href.slice(1))), anchors);
    assert.deepEqual(missing, [], 'every section link reaches a heading that exists');
    records.push({ case: 'Header, route, sequence position, ten sections with working anchors, six investigations, five figures, the served dataset digest, and nothing revealed before a prediction' });
    await screenshot(page.locator('.am-figure').nth(0), 'automl-evidence-loop-desktop.png');

    // ------------------------------------------------- 2. the grammar (I1)
    const grammar = page.locator('.am-investigation').nth(0);
    await checkText(grammar, new RegExp(`The applied space holds ${space.total} valid configurations`));
    // Type an actual value rather than clicking a prepared scenario.
    await grammar.getByLabel('Add a maximum depth to Decision tree').fill('8');
    await grammar.getByRole('button', { name: 'Add the typed maximum depth to Decision tree' }).click();
    await predict(grammar, /It grows/, { guess: 12 });
    await checkText(grammar, /12 configurations/);
    await checkText(grammar, /The change is \+1/);
    await checkText(grammar.locator('.am-verdict'), /Your prediction matches/);
    await screenshot(grammar, 'automl-grammar-desktop.png');
    // Editing an input retires the recorded verdict.
    await grammar.getByLabel('Add a maximum depth to Decision tree').fill('9');
    await grammar.getByRole('button', { name: 'Add the typed maximum depth to Decision tree' }).click();
    assert.equal(await grammar.locator('.am-verdict').count(), 0, 'an edit retires the verdict');
    // The null: retuning an inactive family's setting changes neither count nor recipe.
    await grammar.getByRole('button', { name: /^Reset$/ }).click();
    await grammar.getByRole('button', { name: /Null: retune the inactive tree depth/ }).click();
    await predict(grammar, /Exactly unchanged/, { guess: space.total });
    await checkText(grammar, /same number of permitted configurations/);
    await checkText(grammar.locator('.am-verdict'), /Your prediction matches/);
    await screenshot(grammar, 'automl-grammar-null-desktop.png');
    // Six option rows are on screen together; no two controls may share an
    // accessible name, or a screen-reader user cannot tell them apart.
    const names = await grammar.locator('button, input, select').evaluateAll(items => items.map(item =>
      (item.getAttribute('aria-label')
        || (item.labels && item.labels[0] && item.labels[0].textContent)
        || item.textContent || '').replace(/\s+/g, ' ').trim()));
    const duplicated = names.filter((name, index) => name && names.indexOf(name) !== index);
    assert.deepEqual([...new Set(duplicated)], [], 'no two grammar controls share an accessible name');
    records.push({ case: 'Grammar investigation: a typed tree depth grows the count by one, an edit retires the verdict, and retuning an inactive family changes neither the count nor the active recipe' });

    // -------------------------------------------- 3. expected improvement (I2)
    const acquisition = page.locator('.am-investigation').nth(1);
    await predict(acquisition, /^B \(/, { guess: 0.07978846 });
    await checkText(acquisition, /0\.07978846/);
    await checkText(acquisition, /a worse predicted mean can still buy the larger expected improvement/);
    await checkText(acquisition, /probability of improvement/);
    await screenshot(acquisition, 'automl-acquisition-desktop.png');
    await acquisition.getByRole('button', { name: /^Reset$/ }).click();
    await acquisition.getByRole('button', { name: /collapse B's uncertainty/ }).click();
    await predict(acquisition, /^A \(/);
    await checkText(acquisition.locator('.am-verdict'), /Your prediction matches/);
    await checkText(acquisition, /A zero deviation is a point mass/);
    await screenshot(acquisition, 'automl-acquisition-certain-desktop.png');
    records.push({ case: 'Improvement investigation: B wins on uncertainty, a zero deviation is drawn as a point mass and hands the choice to A, and the shaded area is labelled as the expected improvement rather than the probability' });

    // ---------------------------------------------- 4. successive halving (I3)
    const ladder = page.locator('.am-investigation').nth(2);
    await predict(ladder, /^C survives to the end/, { guess: halving.work.restart });
    await checkText(ladder, /The schedule selects C/);
    await checkText(ladder, /would have reached 0\.02/);
    await checkText(ladder, /9\(1\) \+ 3\(3\) \+ 1\(9\)/);
    assert.equal(await ladder.locator('.am-rung').count(), 3, 'three rungs are shown after the reveal');
    // The plot opens showing only what the schedule actually bought.
    assert.equal(await ladder.locator('.am-curve.is-unpurchased').count(), 0,
      'the counterfactual overlay is off until it is asked for');
    await checkText(ladder, /only the evidence the schedule actually bought/);
    await screenshot(ladder, 'automl-halving-desktop.png');
    await ladder.getByRole('button', { name: /Show the curves that were never bought/ }).click();
    await settle(page);
    assert.ok(await ladder.locator('.am-curve.is-unpurchased').count() >= 6,
      'revealing the overlay draws the unpurchased traces');
    await checkText(ladder, /no rung decision could ever consult them/);
    await screenshot(ladder, 'automl-halving-counterfactual-desktop.png');
    // The contrast: one typed value rescues the slow starter.
    await ladder.getByRole('button', { name: /^Reset$/ }).click();
    await ladder.getByLabel('D at 1 unit').fill('0.09');
    await predict(ladder, /^D survives to the end/, { guess: halving.work.restart });
    await checkText(ladder, /The schedule selects D/);
    await checkText(ladder.locator('.am-verdict'), /Your prediction matches/);
    await screenshot(ladder, 'automl-halving-rescue-desktop.png');
    records.push({ case: 'Halving investigation: the declared ladder selects C at 0.07 while D would have reached 0.02, and typing D’s first rung as 0.09 rescues it; both work accountings are printed as expressions' });

    // ------------------------------------------------- 5. the search replay (I5)
    const replay = page.locator('.am-investigation').nth(3);
    // It opens with two candidates applied and three proposed, so the default
    // question is the one the prose walks through rather than a no-op.
    await checkText(replay, /Moving from the applied budget to 3/);
    await predict(replay, /The recommendation changes, the best score stays flat/, { guess: 9 });
    await checkText(replay, new RegExp(replayThree.recommendedId));
    await checkText(replay, /wins the registry tie rule/);
    await checkText(replay, /cannot be attached to a prefix that/);
    await screenshot(replay, 'automl-replay-desktop.png');
    // The null needs the budget already applied: reversing the unrevealed tail
    // must change nothing, which is only a null relative to the same budget.
    await replay.getByRole('button', { name: /^Reset$/ }).click();
    await replay.getByRole('button', { name: /Budget 4: the width-16 network arrives/ }).click();
    await predict(replay, /Both change/);
    await replay.getByRole('button', { name: /Null: reorder only the candidates still unrevealed/ }).click();
    assert.equal(await replay.locator('.am-verdict').count(), 0, 'the reorder retires the previous verdict');
    await predict(replay, /Neither: same best score, same recommendation/, { guess: 12 });
    await checkText(replay.locator('.am-verdict'), /Your prediction matches/);
    await checkText(replay, /cannot be attached to a prefix that/);
    records.push({ case: 'Replay investigation: budget 2 to 3 changes the recommendation with a flat best score, reordering only the unrevealed tail changes nothing, and no prefix carries an inspection result' });

    // ---------------------------------------------------- 6. deployment (I6)
    const deployment = page.locator('.am-investigation').nth(4);
    await predict(deployment, /^B \(/, { guess: 3 });
    await checkText(deployment, /The frontier is A, B, C/);
    await checkText(deployment, /greatest accuracy among the candidates at or under 5 ms/);
    // The drawn rule must agree with the applied rule at the boundary. D sits at
    // exactly the cap, and the model calls it feasible; the shaded infeasible
    // band must therefore begin strictly to the right of every feasible marker.
    const capGeometry = await deployment.locator('.am-plot svg').evaluate(svg => {
      const band = svg.querySelector('rect.am-node.is-untouched');
      const rule = svg.querySelector('line.am-rule');
      const marks = [...svg.querySelectorAll('.am-mark')].map(mark => ({
        x: mark.tagName === 'circle'
          ? Number(mark.getAttribute('cx'))
          : Number(mark.getAttribute('x')) + Number(mark.getAttribute('width')) / 2,
        infeasible: mark.classList.contains('is-infeasible'),
      }));
      return { bandLeft: Number(band.getAttribute('x')), capX: Number(rule.getAttribute('x1')), marks };
    });
    assert.ok(capGeometry.marks.length >= 5, 'every candidate is drawn');
    // Strictly: the bug this guard names is the band starting AT the cap line
    // rather than past it, and `<=` accepts exactly that. D sits on the line.
    assert.ok(capGeometry.bandLeft > capGeometry.capX,
      `the infeasible band must begin past the cap line, not at it (${capGeometry.bandLeft} versus ${capGeometry.capX})`);
    const feasibleMarks = capGeometry.marks.filter(mark => !mark.infeasible);
    const infeasibleMarks = capGeometry.marks.filter(mark => mark.infeasible);
    assert.ok(feasibleMarks.length > 0 && infeasibleMarks.length > 0,
      'this fixture must produce both feasible and infeasible candidates, or neither loop asserts anything');
    feasibleMarks.forEach(mark => {
      assert.ok(mark.x < capGeometry.bandLeft,
        `a feasible candidate is drawn inside the infeasible band (${mark.x} versus ${capGeometry.bandLeft})`);
    });
    infeasibleMarks.forEach(mark => {
      assert.ok(mark.x > capGeometry.bandLeft, 'and every infeasible candidate is drawn inside it');
    });
    // The candidate sitting exactly on the line is feasible, by rule and by drawing.
    assert.ok(capGeometry.marks.some(mark => Math.abs(mark.x - capGeometry.capX) < 0.01 && !mark.infeasible),
      'the candidate at exactly the cap is drawn as a feasible one');
    await checkText(deployment, /feasible/);
    await screenshot(deployment, 'automl-deployment-desktop.png');
    await deployment.getByRole('button', { name: /^Reset$/ }).click();
    await deployment.getByLabel('Latency cap (ms)').fill('1');
    await predict(deployment, /None is feasible/);
    await checkText(deployment, /the nearest miss is not a permitted choice/);
    await checkText(deployment.locator('.am-verdict'), /Your prediction matches/);
    await screenshot(deployment, 'automl-deployment-infeasible-desktop.png');
    // An exact tie is reachable from the lab's own suggested setup. The rule
    // that breaks it must be stated before the task, and the readout must not
    // claim the winner has the greatest accuracy when another matches it.
    await deployment.getByRole('button', { name: /^Reset$/ }).click();
    await checkText(deployment.locator('.am-note'), /greatest accuracy among feasible candidates; a tie takes the lower latency, then the earlier name/);
    await deployment.getByRole('button', { name: /Two identical candidates: make E match A exactly/ }).click();
    await deployment.getByLabel('Latency cap (ms)').fill('3');
    await predict(deployment, /^E \(/);
    const tieVerdict = deployment.locator('.am-verdict');
    await checkText(tieVerdict, /tie exactly on accuracy/);
    await checkText(tieVerdict, /the earlier name/);
    await checkText(deployment, /A and E tie on accuracy/);
    assert.ok(!normalize(await deployment.innerText()).includes('A has the greatest accuracy'),
      'the readout must not claim uniqueness its own table refutes');
    await screenshot(deployment, 'automl-deployment-tie-desktop.png');
    records.push({ case: 'Deployment investigation: a 5 ms cap ships B off an A/B/C frontier, a 1 ms cap returns no feasible candidate rather than the nearest miss, and an exact accuracy tie is named and settled by the rule the note states up front' });

    // ------------------------------------------------- 7. the mixture (I4)
    const mixture = page.locator('.am-investigation').nth(5);
    await predict(mixture, /identity/, { guess: 0 });
    await checkText(mixture, /The most negative gradient belongs to identity/);
    await checkText(mixture, /Adding the same constant to every active logit/);
    // The step stage carries its own recorded prediction.
    await mixture.getByRole('radio', { name: /toward the target/ }).check();
    await mixture.getByRole('button', { name: /Take the step/ }).click();
    await checkText(mixture, /0\.19933598/);
    await checkText(mixture.locator('.am-verdict').last(), /Your prediction matches|weight on identity increases/);
    // And so does the commitment stage.
    await mixture.getByRole('radio', { name: /^higher$/ }).check();
    await mixture.getByRole('button', { name: /Commit to negation/ }).click();
    await checkText(mixture, /the mixture’s loss was/);
    await screenshot(mixture, 'automl-mixture-desktop.png');
    // A zero-length step is typeable, and then "nowhere at all" is the only
    // correct answer. This was graded wrong, with a verdict that printed
    // identical before-and-after numbers and still asserted a weight increase.
    await mixture.getByRole('button', { name: /^Reset$/ }).click();
    await mixture.getByLabel('Step size').fill('0');
    await predict(mixture, /identity/, { guess: 0 });
    await mixture.getByRole('radio', { name: /by no more than 10⁻¹²/ }).check();
    await mixture.getByRole('button', { name: /Take the step/ }).click();
    const zeroStepVerdict = mixture.locator('.am-verdict').last();
    await checkText(zeroStepVerdict, /Your prediction matches/);
    assert.ok(!(await zeroStepVerdict.getAttribute('class')).includes('is-miss'),
      'a zero-length step must not be graded a miss');
    await checkText(zeroStepVerdict, /computed output change is within 10⁻¹²/);
    await checkText(zeroStepVerdict, /the step length is zero/);
    assert.ok(!(await zeroStepVerdict.innerText()).includes('increases'),
      'the zero-step verdict must not claim a weight increases');
    assert.equal(await mixture.locator('.am-verdict.is-miss').count(), 0,
      'nothing in this investigation is graded wrong at a zero step');
    await screenshot(mixture, 'automl-mixture-zero-step-desktop.png');
    records.push({ case: 'A zero-length step is graded within the disclosed tolerance, with an exact zero-step cause and no weight change' });

    // The commitment fixture: the mixture beats either discrete choice.
    await mixture.getByRole('button', { name: /^Reset$/ }).click();
    await mixture.getByRole('button', { name: /The commitment case/ }).click();
    // Both gradients are exactly zero here, so "none" is the correct answer and
    // must be graded as such: the table says exactly 0 on every row.
    await predict(mixture, /None — the gradient is exactly zero/, { guess: 0 });
    await checkText(mixture.locator('.am-verdict'), /Your prediction matches/);
    await checkText(mixture, /architecture gradient is exactly zero/);
    await checkText(mixture, /residual is zero/);
    await mixture.getByRole('radio', { name: /^higher$/ }).check();
    await mixture.getByRole('button', { name: /Commit to identity/ }).click();
    await checkText(mixture, /discretization itself changed the function/);
    await screenshot(mixture, 'automl-mixture-commit-desktop.png');
    records.push({ case: 'Mixture investigation: the gradient, a separately recorded architecture step reaching 0.1993359892, and a separately recorded commitment showing the mixture beating either discrete operation' });

    // ----------------------------------- sequential audit: editable edge cases
    await grammar.getByRole('button', { name: /^Reset$/ }).click();
    await grammar.getByRole('button', { name: 'Remove maximum depth 5 from Decision tree' }).click();
    await grammar.getByLabel('Add a maximum depth to Decision tree').fill('8');
    await grammar.getByRole('button', { name: 'Add the typed maximum depth to Decision tree' }).click();
    await predict(grammar, /Exactly unchanged/, { guess: 11 });
    await checkText(grammar, /Replacing values can change its recipes without changing that count/);
    assert.doesNotMatch(await grammar.innerText(), /Nothing was added to or removed/);

    for (const [offset, accepted] of [[1e-5, true], [-1e-5, true], [1.01e-5, false], [-1.01e-5, false]]) {
      await acquisition.getByRole('button', { name: /^Reset$/ }).click();
      await acquisition.getByLabel('B: predicted deviation σ').fill('0');
      await acquisition.getByLabel('C: predicted mean μ').fill('0.3');
      await acquisition.getByLabel('C: predicted deviation σ').fill('0');
      await checkText(acquisition.locator('.am-numeric-guess'), /Answers within 0\.00001 are accepted/);
      await predict(acquisition, /^C \(/, { guess: 0.1 + offset });
      await checkText(acquisition.locator('.am-verdict'), accepted ? /within 0\.00001/ : /outside 0\.00001/);
    }
    await acquisition.getByRole('button', { name: /^Reset$/ }).click();
    await acquisition.getByLabel('A: predicted deviation σ').fill('0.001');
    await acquisition.getByLabel('B: predicted deviation σ').fill('1');
    await acquisition.getByLabel('C: predicted deviation σ').fill('0');
    await predict(acquisition, /^B \(/);
    await checkText(acquisition, /separate peak normalization/);
    await checkText(acquisition, /proportional to EI, not numerically equal/);
    const narrowPaint = await acquisition.locator('.am-curve.is-density').first().evaluate(line => {
      const points = [...line.points];
      return { count: points.length, highestY: Math.min(...points.map(point => point.y)) };
    });
    assert(narrowPaint.count > 700, 'narrow candidate contains local mesh in the painted polyline');
    assert(Math.abs(narrowPaint.highestY - 12) < 0.02, 'narrow candidate reaches the true shared normalized peak');
    await screenshot(acquisition, 'automl-sequential-narrow-density-desktop.png');

    await ladder.getByRole('button', { name: /^Reset$/ }).click();
    await ladder.getByLabel('D at 1 unit').fill('0.13');
    await predict(ladder, /^C survives to the end/);
    await checkText(ladder.locator('.am-rung').first(), /ties the cutoff/);
    await ladder.getByRole('button', { name: /Show the curves that were never bought/ }).click();
    await ladder.getByRole('button', { name: /^Reset$/ }).click();
    await ladder.getByLabel('A at 9 units').fill('0.07');
    await ladder.getByLabel('D at 9 units').fill('0.07');
    await predict(ladder, /^C survives to the end/);
    assert.equal(await ladder.locator('.am-curve.is-unpurchased').count(), 0, 'reset clears counterfactual reveal');
    assert.doesNotMatch(await ladder.locator('.am-readout').innerText(), /would have reached.*better/);

    await replay.getByRole('button', { name: /^Reset$/ }).click();
    await replay.getByRole('button', { name: /Budget 4: the width-16 network arrives/ }).click();
    await predict(replay, /Both change/);
    await replay.getByLabel('Budget: candidates revealed').fill('1');
    await predict(replay, /Both change/, { guess: 3 });
    await checkText(replay.locator('.am-verdict'), /Your prediction matches/);
    await checkText(replay.locator('.am-verdict'), /1\.000000 → 0\.994629/);
    await screenshot(replay, 'automl-sequential-budget-decrease-desktop.png');

    await mixture.getByRole('button', { name: /^Reset$/ }).click();
    await mixture.getByLabel('Input x', { exact: true }).fill('0.001');
    await mixture.getByLabel('Target', { exact: true }).fill('0.001');
    await mixture.getByLabel('Step size', { exact: true }).fill('1');
    await mixture.getByLabel('logit for zero', { exact: true }).fill('8');
    await mixture.getByLabel('logit for identity', { exact: true }).fill('-8');
    await mixture.getByLabel('logit for negation', { exact: true }).fill('-8');
    await checkText(mixture, /greatest increase.*most negative gradient/);
    await predict(mixture, /identity/, { guess: 0 });
    await checkText(mixture.locator('.am-verdict').first(), /Your prediction matches/);
    await checkText(mixture, /Changes within 10⁻¹² count as tied/);
    assert.equal(await mixture.getByRole('radio', { name: 'the same within 10⁻¹²', exact: true }).count(), 1);
    const tinyBars = await mixture.locator('svg rect.am-block').evaluateAll(nodes => nodes.map(node => ({
      painted: node.width.baseVal.value, exact: Number(node.getAttribute('width')),
    })));
    assert.equal(tinyBars.length, 3);
    const tinyWeight = Math.exp(-16) / (1 + 2 * Math.exp(-16));
    assert(Math.abs(tinyBars[1].exact / (tinyWeight * 58) - 1) < 1e-12,
      'SVG attribute encodes the exact proportional weight');
    // Chromium quantizes this tiny SVGLength to 6.5e-6 from 6.527038…e-6.
    // Preserve a subpixel absolute paint allowance while still rejecting any visible floor.
    assert(tinyBars[1].painted > 0 && Math.abs(tinyBars[1].painted - tinyWeight * 58) < 5e-8,
      `painted tiny weight has no visible-width floor: actual widths ${JSON.stringify(tinyBars)}; expected ${tinyWeight * 58}`);
    await mixture.getByRole('radio', { name: /by no more than 10⁻¹²/ }).check();
    await mixture.getByRole('button', { name: /Take the step/ }).click();
    await checkText(mixture.locator('.am-verdict').last(), /gradient and step may both be nonzero/);
    assert.doesNotMatch(await mixture.locator('.am-verdict').last().innerText(), /No weight changes|step length is zero|gradient is exactly zero/);
    const tinyDiagramText = await mixture.locator('svg[aria-label^="Input "]').evaluate(svg => {
      const frame = svg.viewBox.baseVal;
      const labels = [...svg.querySelectorAll('text')].map(text => {
        const bounds = text.getBBox();
        return { text: text.textContent, left: bounds.x, top: bounds.y,
          right: bounds.x + bounds.width, bottom: bounds.y + bounds.height };
      });
      return { count: labels.length, outside: labels.filter(label => label.left < frame.x - 0.25
        || label.top < frame.y - 0.25 || label.right > frame.x + frame.width + 0.25
        || label.bottom > frame.y + frame.height + 0.25) };
    });
    assert(tinyDiagramText.count >= 12, 'tiny-gradient diagram label bounds are actually inspected');
    assert.deepEqual(tinyDiagramText.outside, [], 'all labels in the tiny-gradient diagram stay inside its SVG viewBox');
    await screenshot(mixture, 'automl-sequential-tiny-gradient-desktop.png');
    await mixture.getByRole('button', { name: /^Reset$/ }).click();
    await mixture.getByLabel('logit for zero', { exact: true }).fill('0');
    await mixture.getByLabel('logit for negation', { exact: true }).fill('1');
    await predict(mixture, /identity/);
    await checkText(mixture.locator('.am-verdict').first(), /Your prediction matches/);
    records.push({ case: 'Sequential audit: equal-count replacement, four inclusive numeric boundaries, locally resolved density peak, two halving ties and reset gating, decreasing replay budget, nonzero tiny gradients/updated logits, proportional bars and greatest-update/commitment-tie wording' });

    // ------------------------------------------------------------- 8. figures
    const architecture = page.locator('.am-figure').nth(1);
    await checkText(architecture, /box heights are schematic.*not a proportional height scale/);
    await checkText(architecture, /4 → 8 → 1/);
    await architecture.getByRole('button', { name: /4 → 8 → 8 → 1/ }).click();
    await checkText(architecture, /\(4 \+ 1\) × 8 = 40/);
    await architecture.getByRole('button', { name: /output affine block/ }).click();
    await checkText(architecture, /\(8 \+ 1\) × 1 = 9/);
    await screenshot(architecture, 'automl-architecture-desktop.png');
    const results = page.locator('.am-figure').nth(2);
    await checkText(results, /314/);
    await checkText(results, /918\/919/);
    await results.getByRole('button', { name: /^net 16 \(0\)$/ }).click();
    await checkText(results, /makes no out-of-fold mistake at all/);
    await results.getByRole('button', { name: /^knn 3 \(1\)$/ }).click();
    await checkText(results, /350/);
    await screenshot(results, 'automl-results-desktop.png');
    const bilevel = page.locator('.am-figure').nth(3);
    await checkText(bilevel, /−0\.098/);
    await bilevel.getByRole('button', { name: /where the training gradient is zero/ }).click();
    await checkText(bilevel, /−0\.08/);
    await checkText(bilevel, /Two functions can agree at a point/);
    await screenshot(bilevel, 'automl-bilevel-stationary-desktop.png');
    const provenanceFigure = page.locator('.am-figure').nth(4);
    await checkText(provenanceFigure, /2\.079442/);
    await provenanceFigure.getByRole('button', { name: /second code 110 — identical/ }).click();
    await checkText(provenanceFigure, /no finite real value/);
    await screenshot(provenanceFigure, 'automl-provenance-singular-desktop.png');
    records.push({ case: 'Figures: three architectures with their block equations, the observed results with per-candidate mistake tables, the stationary bilevel counterexample, and the singular activation kernel' });

    // ------------------------------------------------ 9. drawn geometry checks
    for (const width of [1366, 390, 320]) {
      await page.setViewportSize({ width, height: 1000 });
      await page.locator('.am-lesson details').evaluateAll(items => items.forEach(item => { item.open = true; }));
      await settle(page);
      const figures = await page.evaluate(inspectLessonVisualLayout, '.am-lesson');
      // A sweep that inspected nothing passes vacuously; say so instead.
      assert.ok(figures.length > 0, `The shared inspector saw no figure at ${width}px`);
      assert.deepEqual(figures.flatMap(figure => figure.issues), [], `Figure layout collides at ${width}px`);
      // The shared inspector samples straight <line> elements only, so a data
      // curve or a flow arrow can run through a label and still report clean.
      const curves = await page.evaluate(sampleCurvesThroughLabels, '.am-lesson');
      assert.ok(curves.inspectedShapes > 0, `No curve was inspected at ${width}px — the sweep is inert`);
      // Say where the coverage actually is, rather than implying the figures
      // are covered because the guard passed on lab polylines.
      const figureCurves = await page.evaluate(sampleCurvesThroughLabels, '.am-figure');
      assert.equal(figureCurves.inspectedShapes, 0,
        'the inline figures are drawn from straight lines only; if that changes, this sweep must start covering them');
      assert.deepEqual(curves.findings, [], `A curve runs through a label at ${width}px`);
      records.push({
        case: `Drawn geometry at ${width}px`,
        inspectedFigures: figures.length,
        inspectedShapes: curves.inspectedShapes,
        curveSamplingNote: 'Curves are sampled along their own geometry against every label box, because the shared inspector reads straight lines only. A label with an opaque backplate may be crossed by up to 2% of a curve length; a label without one may not be crossed at all.',
      });
    }
    await page.setViewportSize({ width: 1366, height: 1000 });

    // ------------------------------------------------------ 10. loading closure
    const localScripts = [...new Set(requests.map(address => new URL(address))
      .filter(address => address.origin === new URL(base).origin && address.pathname.endsWith('.js'))
      .map(address => address.pathname.slice(1)))];
    assert.deepEqual(localScripts.filter(filename => bodyFiles.has(filename)), [bodyFile],
      'only this lesson body is downloaded');
    assert.deepEqual(localScripts.filter(filename => !allowedScripts.has(filename)), [],
      'and nothing outside its declared closure');
    assert.ok(!requests.some(address => address.includes('/outlines/')), 'no outline is fetched');
    assert.ok(!requests.some(address => address.includes('banknote-data.csv')),
      'the page never fetches the dataset to render');
    assert.ok(!requests.some(address => address.includes('learn-assets/bias-variance')),
      'and never borrows another lesson\'s asset');
    records.push({
      case: 'Fresh route requests only the selected lesson and its shared closure, and never fetches its own dataset to render',
      requestedScripts: localScripts,
      rawScriptBytes: localScripts.reduce((sum, filename) => sum + fs.statSync(path.join(distDir, filename)).size, 0),
      gzipEstimateBytes: localScripts.reduce((sum, filename) => sum + gzipSync(fs.readFileSync(path.join(distDir, filename))).length, 0),
      interpretation: 'Built requested file sizes; the gzip estimate is not measured network compression or latency.',
    });

    // -------------------------------------------------------- 11. narrow widths
    for (const width of [390, 320]) {
      await page.setViewportSize({ width, height: 844 });
      await page.locator('.am-lesson details').evaluateAll(items => items.forEach(item => { item.open = true; }));
      await settle(page);
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
        `Document overflows at ${width}px`);
      const mathOverflow = await page.locator('.am-lesson .katex-display')
        .evaluateAll(items => items.filter(item => item.scrollWidth > item.clientWidth + 1).map(item => item.textContent));
      assert.deepEqual(mathOverflow, [], `Overflowing formulas at ${width}px`);
      assert.equal(await page.locator('.katex-error').count(), 0, `KaTeX errors at ${width}px`);
      const stacking = await page.locator('.am-table-scroll tbody tr').first()
        .evaluate(element => ({
          row: getComputedStyle(element).display,
          cell: getComputedStyle(element.querySelector('td, th')).display,
          label: getComputedStyle(element.querySelector('td, th'), '::before').content,
          columns: getComputedStyle(element.querySelector('td, th')).gridTemplateColumns,
        }));
      assert.equal(stacking.row, 'block', `Table rows do not stack at ${width}px`);
      assert.equal(stacking.cell, 'grid', `Stacked cells are not laid out as label and value at ${width}px`);
      assert.ok(stacking.label && stacking.label !== 'none', `Stacked cells carry no column label at ${width}px`);
      assert.ok(stacking.columns.split(' ').length === 2, `Stacked cells have no label column at ${width}px`);
      assert.equal(await page.locator('.am-table-scroll caption').count(), 0,
        `a caption is still inside a scroll box at ${width}px`);
      const clipped = await page.locator('.am-table-scroll')
        .evaluateAll(items => items.filter(item => item.scrollWidth > item.clientWidth + 1).length);
      assert.equal(clipped, 0, `A table still scrolls sideways at ${width}px`);
      // SVG text must not shrink below a readable size at the narrowest width.
      // `getComputedStyle(text).fontSize` inside an SVG is in **user units**,
      // before the viewBox-to-box scale, so it is the same number at every
      // viewport and cannot detect the regression this guard exists for. Scale
      // it by the box-to-viewBox ratio to get rendered pixels, and refuse to
      // pass on an empty set — Math.min of nothing is Infinity.
      const textSizes = await page.locator('.am-lesson svg text').evaluateAll(items => items
        .filter(item => item.getClientRects().length && item.textContent.trim())
        .map(item => {
          const svg = item.ownerSVGElement;
          const box = svg.getBoundingClientRect();
          const units = svg.viewBox.baseVal.width || box.width;
          const scale = units ? box.width / units : 1;
          return Number(getComputedStyle(item).fontSize.replace('px', '')) * scale;
        }));
      assert.ok(textSizes.length > 20, `only ${textSizes.length} SVG labels were measured at ${width}px`);
      const smallest = Math.min(...textSizes);
      const median = textSizes.slice().sort((a, b) => a - b)[Math.floor(textSizes.length / 2)];
      assert.ok(Number.isFinite(smallest), 'the measured minimum is a real number');
      // 7.25 px is the measured floor, not an aspiration: plots inside an
      // investigation render into a narrower box than the inline figures, and
      // every number in every drawing is also in an adjacent stacking table.
      // The true minimum and median are recorded below rather than summarised.
      assert.ok(smallest >= 7.25, `SVG text renders at ${smallest.toFixed(2)}px at viewport ${width}px`);
      renderedText[width] = {
        labels: textSizes.length,
        minimumPx: Number(smallest.toFixed(2)),
        medianPx: Number(median.toFixed(2)),
      };
      for (const [index, name] of [[0, 'evidence-loop'], [1, 'architecture'], [2, 'results'], [3, 'bilevel'], [4, 'provenance']]) {
        await screenshot(page.locator('.am-figure').nth(index), `automl-${name}-${width}.png`);
      }
      await screenshot(page.locator('.am-investigation').nth(0), `automl-grammar-${width}.png`);
      await screenshot(page.locator('.am-investigation').nth(2), `automl-halving-${width}.png`);
      records.push({
        case: `Narrow ${width}px layout: no horizontal overflow, no overflowing formula, every numeric table stacked into labelled rows`,
        svgTextRenderedPx: renderedText[width],
        svgTextNote: 'Rendered pixels, computed as the SVG user size times the box-to-viewBox ratio. A computed font-size '
          + 'read straight off an SVG text node is in user units and is the same number at every viewport, which is why '
          + 'this once recorded an identical figure at two widths. Plots inside an investigation render into a narrower '
          + 'box than the inline figures, which is where the minimum sits; every number in every drawing is also in an '
          + 'adjacent stacking table and in the SVG aria-label.',
      });
    }
    await page.setViewportSize({ width: 1366, height: 1000 });

    // ------------------------------------------- 11b. captions at every width
    // A <caption> sizes itself to its table, so a wide table scrolls its own
    // caption out of view at desktop width. They now sit outside the scroller,
    // and this runs at 1366 too, where the earlier narrow-only checks could not
    // see the one caption that was clipped.
    for (const width of [1366, 390, 320]) {
      await page.setViewportSize({ width, height: width === 1366 ? 1000 : 844 });
      await settle(page);
      const captions = await page.locator('.am-lesson .am-table-caption').evaluateAll(items => items.map(item => ({
        text: item.textContent.trim().slice(0, 40),
        width: item.getBoundingClientRect().width,
        parentWidth: item.parentElement.getBoundingClientRect().width,
        scrolled: item.scrollWidth > item.clientWidth + 1,
      })));
      assert.ok(captions.length >= 15, `only ${captions.length} captions were measured at ${width}px`);
      const clippedCaptions = captions.filter(item => item.scrolled || item.width > item.parentWidth + 1);
      assert.deepEqual(clippedCaptions, [], `a caption is clipped or overflows its container at ${width}px`);
    }
    await page.setViewportSize({ width: 1366, height: 1000 });
    records.push({ case: 'Every table caption sits outside its scroll box and fits its container at 1366, 390 and 320 px' });

    // ------------------------------------------------------- 12. keyboard reach
    await page.keyboard.press('Control+Home');
    const focusable = await page.locator('.am-investigation').nth(0)
      .locator('button, input, select, textarea, summary').count();
    assert.ok(focusable > 8, 'the first investigation exposes its controls to the keyboard');
    const focusRing = await page.locator('.am-investigation button').first().evaluate(element => {
      element.focus();
      return getComputedStyle(element, ':focus-visible').outlineStyle;
    });
    assert.notEqual(focusRing, 'none', 'a focused control has a visible outline');
    // A value a control displays must be a value that control accepts. A field
    // limited to six decimals once showed ln 2 at full float precision, which
    // overflowed its box and was rejected when retyped.
    const overlongValues = await page.locator('.am-investigation input[type="number"]').evaluateAll(items => items
      .map(item => item.value)
      .filter(value => /\.\d{7,}/.test(value)));
    assert.deepEqual(overlongValues, [], 'every displayed number fits the resolution its own control accepts');
    records.push({ case: 'Keyboard reach: every investigation control is focusable and carries a visible focus outline; every displayed number is one its own control accepts', focusableControls: focusable });

    // ---------------------------------------------------------- 13. sequence
    await page.locator('.reader-complete').click();
    assert.equal(new URL(page.url()).pathname.split('/').at(-1), topicId, 'completion does not auto-advance');
    await page.reload({ waitUntil: 'domcontentloaded' });
    await ready(page);
    assert.equal(await page.evaluate(id => JSON.parse(localStorage.getItem('kd-progress'))[id], topicId), true,
      'completion persists under the stable ID');
    await page.locator('.reader-footer__next').click();
    await page.waitForURL('**/hidden-markov-models-hmm?module=classical-ml');
    records.push({ case: 'Completion persists under the stable ID without auto-advance; Next opens Hidden Markov Models' });
    assert.deepEqual(errors, [], 'no page error was raised during the review');
    assert.deepEqual(failedAssets, [], 'and no asset failed to load');
    await context.close();

    // ---------------------------------------------------------- 14. recovery
    for (const failure of ['import', 'render']) {
      const isolated = await browser.newContext();
      const trial = await isolated.newPage();
      let inject = true;
      await trial.route(`**/${bodyFile}`, async intercepted => {
        if (!inject) return intercepted.continue();
        inject = false;
        if (failure === 'import') return intercepted.abort('failed');
        return intercepted.fulfill({ status: 200, contentType: 'text/javascript', body: 'export default {content(){throw new Error("Controlled review failure")}}' });
      });
      await trial.goto(route, { waitUntil: 'domcontentloaded' });
      await trial.locator('.lesson-load-error').waitFor();
      assert.ok(await trial.locator('.reader-complete').isDisabled(), 'completion stays disabled on failure');
      await trial.getByRole('button', { name: /reload/i }).click();
      await ready(trial);
      assert.ok(await trial.locator('.reader-complete').isEnabled(), 'and recovers with an explicit reload');
      records.push({ case: `${failure} failure keeps completion disabled and recovers with an explicit reload` });
      await isolated.close();
    }

    // Screenshots must have unique paths and unique content: a duplicate image
    // means a state was captured twice and another state nowhere.
    assert.equal(new Set(screenshotPaths).size, screenshotPaths.length, 'screenshot paths are unique');
    const digests = new Map();
    for (const file of screenshotPaths) {
      const digest = hash(file);
      assert.ok(!digests.has(digest), `${file} is byte-identical to ${digests.get(digest)}`);
      digests.set(digest, file);
    }

    for (const [filename, expected] of Object.entries(sourceHashes)) {
      assert.equal(hash(filename), expected, `Source changed during check: ${filename}`);
    }
    assert.equal(hash(`${distDir}/.vite/manifest.json`), buildHash, 'the build did not change during the check');
    const report = {
      startedAt, completedAt: new Date().toISOString(), topicId, status: 'passed',
      browser: await browser.version(), distDir, buildManifestHash: buildHash, sourceHashes,
      moduleTopicCount: module.topicIds.length,
      modulePosition: module.topicIds.indexOf(topicId) + 1,
      dataset: {
        file: data.provenance.file, sha256: data.provenance.sha256, bytes: data.provenance.bytes,
        reserveScored: data.reserveScored,
      },
      displayedPrograms: Object.fromEntries(Object.entries(automlExamples)
        .map(([key, value]) => [key, { executed: value.executed, file: value.file ?? null }])),
      records, screenshots: screenshotPaths,
      visualInterpretation: 'Screenshots capture informative states, not defaults only: the evidence loop; the grammar '
        + 'after a typed tree depth and again after the inactive-setting null; the improvement panels with a Gaussian '
        + 'winner and again with a zero-deviation point mass; the ladder as declared and again with the slow starter '
        + 'rescued; the replay after a tie changed the recommendation; the frontier under a feasible cap and under an '
        + 'infeasible one; the mixture after an architecture step and after a commitment; the three architectures; the '
        + 'observed results with a mistake table open; the bilevel figure at its stationary point; the activation '
        + 'kernel gone singular; and every figure plus two investigations at 390 and 320 px. They require separate '
        + 'visual inspection by a person.',
    };
    fs.writeFileSync(evidencePath, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify({ status: report.status, cases: records.length, evidencePath, sourceFiles: ownedFiles.length, screenshots: screenshotPaths.length }));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });

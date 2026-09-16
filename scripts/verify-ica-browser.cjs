// Production checks for the implemented ICA lesson, its six figures and its
// two investigations. DIST_DIR and LEARNING_BASE_URL allow a per-topic build
// and preview, so this can run beside other topics' verifiers.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { gzipSync } = require('node:zlib');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const { inspectLessonVisualLayout } = require('./lib/lesson-visual-layout.cjs');

const distDir = process.env.DIST_DIR || 'dist';
const base = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4181';
const id = 'independent-component-analysis-ica';
const route = `${base}/learn/path/full-curriculum/${id}?module=classical-ml`;
const source = `src/learn/data/topics/${id}.jsx`;
const hash = file => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const normalize = value => value.replace(/\s+/g, ' ').trim();
const records = [];
const screenshots = [];
const layouts = [];
const owned = [source, 'src/learn/data/ica-models.js', 'src/learn/data/ica-data.js', 'src/learn/data/ica-examples.js',
  'src/learn/components/lesson-labs/IcaShared.jsx', 'src/learn/components/lesson-labs/IcaFigures.jsx',
  'src/learn/components/lesson-labs/IcaLabs.jsx', 'src/learn/components/lesson-labs/ica-labs.css',
  `src/learn/data/curriculum/blueprints/${id}.js`,
  // The registry is the integration owner's file, recorded here so the build
  // this pass exercises is known to contain the blueprint registration.
  'src/learn/data/curriculum/blueprints/index.js'];

(async () => {
  const build = JSON.parse(fs.readFileSync(`${distDir}/.vite/manifest.json`, 'utf8'));
  const sources = Object.fromEntries(owned.map(file => [file, hash(file)]));
  const { icaExamples } = await import('../src/learn/data/ica-examples.js');
  const { ICA_RECORDING } = await import('../src/learn/data/ica-data.js');
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const context = await browser.newContext({ viewport: { width: 1366, height: 1000 } });
  const page = await context.newPage();
  const errors = [];
  const requests = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => requests.push(request.url()));

  const lab = kind => page.locator(`[data-ica-lab="${kind}"]`);
  const button = (target, name) => target.getByRole('button', { name, exact: true });
  const text = async (target, pattern) => assert.match(normalize(await target.innerText()), pattern);
  const commitRadio = async (target, choice, action) => {
    await target.getByRole('radio', { name: choice, exact: true }).check();
    await button(target, 'Commit prediction').click();
    await button(target, action).click();
  };
  const capture = async (target, name) => {
    const destination = `docs/teaching/evidence/screenshots/ica-${name}.png`;
    fs.mkdirSync(path.dirname(destination), { recursive: true });
    await target.screenshot({ path: destination });
    screenshots.push(destination);
  };

  try {
    await page.goto(route, { waitUntil: 'networkidle' });
    await page.locator('.ica-lesson').waitFor();
    await page.evaluate(() => document.fonts.ready);
    await page.addStyleTag({ content: '*{scroll-behavior:auto!important;animation:none!important;transition:none!important}' });
    // Hide only the page chrome, so an element capture is never overlaid by a
    // sticky header. No lesson content is touched.
    const hideChrome = () => page.evaluate(() => [...document.body.querySelectorAll('*')].forEach(element => {
      if (!element.closest('main') && ['fixed', 'sticky'].includes(getComputedStyle(element).position)) element.style.visibility = 'hidden';
    }));
    await hideChrome();

    // ---- structure and manuscript conservation -----------------------------
    await text(page.locator('.reader-header__meta'), /18 of 39 topics/);
    await text(page.locator('.reader-footer__previous'), /t-SNE, UMAP/);
    await text(page.locator('.reader-footer__next'), /Non-Negative Matrix Factorization/);
    assert.equal(await page.locator('.ica-lesson > h2').count(), 11);
    assert.equal(await page.locator('[data-ica-figure]').count(), 6);
    assert.equal(await page.locator('[data-ica-lab]').count(), 2, 'rotation and contribution open; the scale question is gated');
    assert.equal(await page.locator('.ic-figure-p1 .ic-lock').count(), 1, 'the frozen index carries a lock marker beside its text');
    assert.equal(await page.locator('.python-example').count(), 2);
    assert.equal(await page.locator('.ica-lesson > details').count(), 12);
    assert.equal(await page.locator('.ica-lesson > details[open]').count(), 0);
    for (const anchor of await page.locator('.lesson-intro a[href^="#"]').evaluateAll(items => items.map(item => item.hash))) {
      assert.equal(await page.locator(`[id="${anchor.slice(1)}"]`).count(), 1, `anchor ${anchor}`);
    }
    const body = normalize(await page.locator('.ica-lesson').textContent());
    for (const example of Object.values(icaExamples)) {
      assert.ok(body.includes(normalize(example.code)), `complete displayed ${example.file}`);
      assert.ok(body.includes(normalize(example.expected)), `actual executed output of ${example.file}`);
    }
    assert.equal(await page.locator('.katex-error').count(), 0);
    assert.doesNotMatch(body, /\[Inline figure|Inline figure [MDWFPE]1 —|visual packet|not been executed here/);
    // Every claim class the manuscript insists on, still on the page.
    for (const claim of [
      'noiseless, instantaneous, constant linear mixture',
      '20,000 time samples need not provide the information of 20,000 independent draws',
      'Equal source kurtoses do not destroy separation',
      'its excess kurtosis is also zero',
      'PCA coordinate 4 has the largest test value in this fixed comparison',
      'It is not a fetal-beat detector',
      'reconstructs sensor values to numerical precision even when the components are unhelpful',
      'It is not an anatomical source location',
      '800,000,000 bytes',
      'There is no universal “five times components squared” sample threshold',
    ]) assert.ok(body.includes(claim), `manuscript claim preserved: ${claim}`);
    for (const [name, expected] of [['iterations', String(ICA_RECORDING.iterations)], ['held-out PCA', '0.450178'], ['held-out ICA', '0.343966'], ['held-out channel', '0.119806']]) {
      assert.ok(body.includes(expected), `${name} value ${expected} shown`);
    }
    for (const asset of ['r01-first20s.csv', 'data-provenance.md', 'ica_by_hand.py', 'ica_recording.py']) {
      const response = await page.request.get(`${base}/learn-assets/ica/${asset}`);
      assert.equal(response.status(), 200, asset);
      const content = await response.text();
      if (asset.endsWith('.csv')) assert.equal(content.trim().split('\n').length, 20001);
      if (asset === 'data-provenance.md') assert.ok(content.includes('ODC-By') || content.includes('Open Data Commons'));
      if (asset.endsWith('.py')) assert.ok(body.includes(normalize(content)), `${asset} is the displayed program`);
    }
    records.push('11 sections, 6 inline figures, 2 executed programs, 2 open investigations, 6 closed hint/solution pairs, 10 manuscript claims, 4 offline assets, module sequence 18 of 39');

    // ---- investigations start unset ---------------------------------------
    for (const kind of ['rotation', 'contribution']) {
      assert.equal(await lab(kind).locator('input[type=radio]:checked').count(), 0);
      assert.ok(await button(lab(kind), 'Commit prediction').isDisabled(), `${kind} needs a recorded prediction`);
    }
    assert.equal(await page.locator('[data-ica-scale-gate]').count(), 1);

    // ---- R1 rotation -------------------------------------------------------
    const rotation = lab('rotation');
    await text(rotation, /Applied population: Binary ±1 at 45°\. Staged: Binary ±1 at 45°/);
    await text(rotation, /κ\(y₁\) −1/);
    assert.equal(await rotation.locator('.ic-marker-proposed').count(), 0, 'the proposed angle is not pre-revealed');
    await button(rotation, 'Draft 30°').click();
    await rotation.getByRole('radio', { name: 'larger', exact: true }).check();
    await button(rotation, 'Commit prediction').click();
    assert.equal(await rotation.locator('.ic-marker-proposed').count(), 1, 'the proposed angle is marked once committed');
    await button(rotation, 'Apply angle').click();
    await text(rotation.locator('[data-ica-result]'), /You recorded larger; the actual change was larger. \|κ\| went from 1 at 45° to 1.25 at 30°/);
    await capture(rotation, 'rotation-matched-1366');
    // An arbitrary decimal angle, and the exact equal-angle case.
    await rotation.getByLabel('Proposed angle θ in degrees (staged)', { exact: true }).fill('12.3456');
    await commitRadio(rotation, 'larger', 'Apply angle');
    await text(rotation.locator('[data-ica-result]'), /the actual change was larger/);
    await text(rotation, /Applied angle 12.35°/);
    await rotation.getByLabel('Proposed angle θ in degrees (staged)', { exact: true }).fill('12.3456');
    await commitRadio(rotation, 'the same', 'Apply angle');
    await text(rotation.locator('[data-ica-result]'), /the actual change was the same/);
    // Practice 2 transfer: Laplace at 30 degrees is exactly 1.875.
    await rotation.getByLabel('Source family (staged)', { exact: true }).selectOption('laplace');
    await rotation.getByLabel('Proposed angle θ in degrees (staged)', { exact: true }).fill('30');
    await commitRadio(rotation, 'larger', 'Apply angle');
    await text(rotation.locator('[data-ica-result]'), /not the same-family comparison the prompt describes/);
    await text(rotation, /κ\(y₁\) 1.875/);
    await rotation.getByLabel('Proposed angle θ in degrees (staged)', { exact: true }).fill('45');
    await commitRadio(rotation, 'smaller', 'Apply angle');
    await text(rotation.locator('[data-ica-result]'), /\|κ\| went from 1.875 at 30° to 1.5 at 45°/);
    // The Gaussian null: every angle gives the same value and no direction.
    await rotation.getByLabel('Source family (staged)', { exact: true }).selectOption('gaussian');
    await commitRadio(rotation, 'the same', 'Apply angle');
    await rotation.getByLabel('Proposed angle θ in degrees (staged)', { exact: true }).fill('137.5');
    await commitRadio(rotation, 'the same', 'Apply angle');
    await text(rotation.locator('[data-ica-result]'), /the actual change was the same.*supplies no source direction/);
    // The clause must describe what happened: at a Gaussian pair nothing moved.
    await text(rotation.locator('[data-ica-result]'), /Your direction changed while the fourth moment did not/);
    assert.doesNotMatch(normalize(await rotation.locator('[data-ica-result]').innerText()), /changed the fourth moment/);
    await text(rotation, /flat at zero/);
    await capture(rotation, 'rotation-gaussian-null-1366');
    // Binary 0 to 90 degrees: the direction changes, the fourth moment does not.
    await rotation.getByLabel('Source family (staged)', { exact: true }).selectOption('binary');
    await rotation.getByLabel('Proposed angle θ in degrees (staged)', { exact: true }).fill('0');
    await commitRadio(rotation, 'the same', 'Apply angle');
    await capture(rotation, 'rotation-family-restart-1366');
    await rotation.getByLabel('Proposed angle θ in degrees (staged)', { exact: true }).fill('90');
    await commitRadio(rotation, 'the same', 'Apply angle');
    await text(rotation.locator('[data-ica-result]'), /\|κ\| went from 2 at 0° to 2 at 90°.*Your direction changed while the fourth moment did not/);
    await capture(rotation, 'rotation-equal-kurtosis-1366');
    await rotation.getByLabel('Proposed angle θ in degrees (staged)', { exact: true }).fill('0');
    await commitRadio(rotation, 'the same', 'Apply angle');
    // A wrong prediction stays neutral and keeps the next attempt available.
    await commitRadio(rotation, 'smaller', 'Apply angle');
    await text(rotation.locator('[data-ica-result]'), /You recorded smaller; the actual change was the same/);
    await text(rotation.locator('[data-ica-result]'), /You applied the same direction, so the fourth moment did not move either/);
    await capture(rotation, 'rotation-missed-1366');
    // Invalid input, and retirement of a committed prediction.
    await rotation.getByLabel('Proposed angle θ in degrees (staged)', { exact: true }).fill('181');
    await text(rotation, /The angle must lie between 0 and 180 degrees/);
    assert.ok(await button(rotation, 'Commit prediction').isDisabled());
    await button(rotation, 'Reset').click();
    await text(rotation, /Applied population: Binary ±1 at 45°/);
    await rotation.getByRole('radio', { name: 'larger', exact: true }).check();
    await button(rotation, 'Commit prediction').click();
    await rotation.getByLabel('Proposed angle θ in degrees (staged)', { exact: true }).fill('60');
    assert.equal(await rotation.locator('input[type=radio]:checked').count(), 0, 'editing the angle retires the prediction');
    assert.ok(await button(rotation, 'Apply angle').isDisabled());
    assert.equal(await rotation.locator('.ic-marker-proposed').count(), 0);
    await commitRadio(rotation, 'larger', 'Apply angle');
    await text(rotation, /Applied angle 60°/);
    await rotation.getByLabel('Proposed angle θ in degrees (staged)', { exact: true }).fill('90');
    await commitRadio(rotation, 'larger', 'Apply angle');
    await text(rotation, /Applied angle 90°/);
    await button(rotation, 'Undo apply').click();
    await text(rotation.locator('[data-ica-result]'), /Previous result \(history/);
    await text(rotation, /Applied angle 60°/);
    assert.equal(await rotation.locator('input[type=radio]:checked').count(), 0, 'undo clears the prediction too');
    records.push('R1: matched, missed, decimal, exact equal-angle, Laplace 1.875 transfer, Gaussian null, family restart, invalid angle, prediction retirement, undo');

    // ---- C1 contributions --------------------------------------------------
    const contribution = lab('contribution');
    await text(contribution.locator('[data-ica-notice]'), /Change at least one amplitude to start/);
    assert.equal(await contribution.locator('.ic-error').count(), 0, 'a required first action is not presented as a failure');
    assert.ok(await button(contribution, 'Commit prediction').isDisabled(), 'the solved example cannot be the first task');
    await contribution.getByLabel('Source 1 amplitude (staged)', { exact: true }).fill('2');
    await contribution.getByLabel('Source 2 amplitude (staged)', { exact: true }).fill('-3');
    await contribution.getByLabel('Keep-set (staged)', { exact: true }).selectOption('first');
    await contribution.getByLabel('Predicted sensor 1 amplitude', { exact: true }).fill('4');
    await contribution.getByLabel('Predicted sensor 2 amplitude (optional transfer)', { exact: true }).fill('2');
    await button(contribution, 'Commit prediction').click();
    await button(contribution, 'Reveal reconstruction').click();
    await text(contribution.locator('[data-ica-result]'), /Sensor 1: you recorded 4, the reconstruction is 4 — a match.*Sensor 2: you recorded 2, the reconstruction is 2 — a match/);
    await text(contribution.locator('[data-ica-result]'), /full observation is \(1, −4\).*retains \(4, 2\) and removes \(−3, −6\)/);
    await capture(contribution, 'contribution-matched-1366');
    // Keep the other component, then neither: the specification's fixtures.
    await contribution.getByLabel('Keep-set (staged)', { exact: true }).selectOption('second');
    await contribution.getByLabel('Predicted sensor 1 amplitude', { exact: true }).fill('-3');
    await button(contribution, 'Commit prediction').click();
    await button(contribution, 'Reveal reconstruction').click();
    await text(contribution.locator('[data-ica-result]'), /the reconstruction is −3 — a match/);
    await contribution.getByLabel('Keep-set (staged)', { exact: true }).selectOption('none');
    await contribution.getByLabel('Predicted sensor 1 amplitude', { exact: true }).fill('1');
    await button(contribution, 'Commit prediction').click();
    await button(contribution, 'Reveal reconstruction').click();
    await text(contribution.locator('[data-ica-result]'), /you recorded 1, the reconstruction is 0 — a difference of \+1/);
    await capture(contribution, 'contribution-missed-1366');
    // A zero source makes removing that component a null operation.
    await contribution.getByLabel('Source 1 amplitude (staged)', { exact: true }).fill('2');
    await contribution.getByLabel('Source 2 amplitude (staged)', { exact: true }).fill('0');
    await contribution.getByLabel('Keep-set (staged)', { exact: true }).selectOption('first');
    await contribution.getByLabel('Predicted sensor 1 amplitude', { exact: true }).fill('4');
    await button(contribution, 'Commit prediction').click();
    await button(contribution, 'Reveal reconstruction').click();
    await text(contribution.locator('[data-ica-result]'), /retains \(4, 2\) and removes \(0, 0\)/);
    // Out-of-range amplitude is refused rather than silently clamped.
    await contribution.getByLabel('Source 1 amplitude (staged)', { exact: true }).fill('5');
    await text(contribution, /Amplitudes must lie between −4 and 4/);
    assert.ok(await button(contribution, 'Commit prediction').isDisabled());
    await capture(contribution, 'contribution-out-of-range-1366');
    await contribution.getByLabel('Source 1 amplitude (staged)', { exact: true }).fill('2');
    records.push('C1: changed amplitudes required first, matched and missed numeric predictions, every keep-set, zero-source null, out-of-range refusal');

    // ---- C1b scale ambiguity ----------------------------------------------
    const scale = lab('scale');
    assert.equal(await page.locator('[data-ica-scale-gate]').count(), 0, 'the scale question opens after an exclusion trial');
    assert.equal(await scale.locator('input[type=radio]:checked').count(), 0);
    // No answer may sit beside an unset question: only the current state shows.
    assert.equal(await scale.locator('[data-ica-scale-unrevealed]').count(), 1);
    assert.equal(await scale.locator('[data-ica-result]').count(), 0);
    assert.doesNotMatch(normalize(await scale.innerText()), /after/, 'no "after" column before the first apply');
    await capture(scale, 'scale-initial-1366');
    await commitRadio(scale, 'stay the same', 'Apply scaling');
    await text(scale.locator('[data-ica-result]'), /the observed sensors stayed the same/);
    assert.equal(await scale.locator('[data-ica-scale-unrevealed]').count(), 0);
    await capture(scale, 'scale-compensated-1366');
    await button(scale, 'Draft c = −2').click();
    await commitRadio(scale, 'stay the same', 'Apply scaling');
    await text(scale.locator('[data-ica-result]'), /stayed the same.*sign ambiguity/);
    await capture(scale, 'scale-negative-1366');
    await scale.getByLabel('Operation (staged)', { exact: true }).selectOption('source');
    await scale.getByLabel('Scale c (staged)', { exact: true }).fill('2');
    await commitRadio(scale, 'stay the same', 'Apply scaling');
    await text(scale.locator('[data-ica-result]'), /the observed sensors changed.*physical contribution really changed/);
    // S2: a source-only rescale of a component this keep-set drops changes
    // nothing, and the explanation has to say that rather than contradict it.
    await scale.getByLabel('Component to rescale (staged)', { exact: true }).selectOption('2');
    await commitRadio(scale, 'stay the same', 'Apply scaling');
    await text(scale.locator('[data-ica-result]'), /stayed the same.*this keep-set already drops component 2/);
    assert.doesNotMatch(normalize(await scale.locator('[data-ica-result]').innerText()), /it changes the physical contribution/);
    await capture(scale, 'scale-source-only-null-1366');
    await scale.getByLabel('Component to rescale (staged)', { exact: true }).selectOption('1');
    await button(scale, 'Draft c = 0').click();
    await text(scale, /c = 0 is invalid/);
    assert.ok(await button(scale, 'Commit prediction').isDisabled());
    await capture(scale, 'scale-zero-refusal-1366');
    await button(scale, 'Reset').click();
    assert.equal(await scale.locator('[data-ica-scale-unrevealed]').count(), 1, 'Reset hides the answer again');
    records.push('C1b: compensated scale null, negative-c sign flip, source-only change, c = 0 refusal without dividing');

    // Sequential audit: every graded decimal change must remain visible in its
    // feedback, even when the ordinary six-place readout would round it away.
    const readNumber = value => Number(value.replace(/\.$/, '').replaceAll('−', '-'));
    await button(rotation, 'Reset').click();
    await rotation.getByLabel('Proposed angle θ in degrees (staged)', { exact: true }).fill('45.001');
    await commitRadio(rotation, 'larger', 'Apply angle');
    const tinyRotation = normalize(await rotation.locator('[data-ica-result]').innerText());
    assert.match(tinyRotation, /the actual change was larger/);
    const rotationValues = tinyRotation.match(/went from ([\d.e+−-]+) at ([\d.e+−-]+)° to ([\d.e+−-]+) at ([\d.e+−-]+)°, a signed difference of ([\d.e+−-]+)/);
    assert.ok(rotationValues, 'the actual comparison and signed difference are present');
    assert.ok(readNumber(rotationValues[3]) > readNumber(rotationValues[1]), 'a graded kurtosis change is visible');
    assert.equal(readNumber(rotationValues[2]), 45);
    assert.equal(readNumber(rotationValues[4]), 45.001, 'feedback preserves the entered angle that caused the change');
    assert.ok(readNumber(rotationValues[5]) > 1e-10, 'a nonzero graded difference is not printed as zero');
    await capture(rotation.locator('.ic-prediction'), 'rotation-tiny-change');

    // A source-only identity scaling needs its own explanation, without
    // pretending that the learner divided the mixing column.
    await scale.getByLabel('Operation (staged)', { exact: true }).selectOption('source');
    await scale.getByLabel('Scale c (staged)', { exact: true }).fill('1');
    await commitRadio(scale, 'stay the same', 'Apply scaling');
    const identityScaling = normalize(await scale.locator('[data-ica-result]').innerText());
    assert.match(identityScaling, /the observed sensors stayed the same/);
    assert.match(identityScaling, /(?:c\s*=\s*1|factor of 1|multiplying by 1|scale (?:is|of) 1)/i);
    assert.doesNotMatch(identityScaling, /compensated product is fixed by construction/);

    await contribution.getByLabel('Source 1 amplitude (staged)', { exact: true }).fill('2.00000001');
    await contribution.getByLabel('Source 2 amplitude (staged)', { exact: true }).fill('0');
    await contribution.getByLabel('Keep-set (staged)', { exact: true }).selectOption('first');
    await contribution.getByLabel('Predicted sensor 1 amplitude', { exact: true }).fill('4');
    await button(contribution, 'Commit prediction').click();
    await button(contribution, 'Reveal reconstruction').click();
    const tinyContribution = normalize(await contribution.locator('[data-ica-result]').innerText());
    const contributionValues = tinyContribution.match(/Sensor 1: you recorded ([\d.e+−-]+), the reconstruction is ([\d.e+−-]+) — a difference of ([\d.e+−-]+)/);
    assert.ok(contributionValues, 'a missed prediction keeps its numerical evidence');
    assert.equal(readNumber(contributionValues[1]), 4);
    assert.ok(readNumber(contributionValues[2]) > 4, 'actual amplitude differs visibly from the rejected prediction');
    assert.ok(readNumber(contributionValues[3]) < -1e-9, 'rejected error is not printed as zero');

    await contribution.getByLabel('Source 1 amplitude (staged)', { exact: true }).fill('0.00000001');
    await contribution.getByLabel('Predicted sensor 1 amplitude', { exact: true }).fill('0.00000002');
    await button(contribution, 'Commit prediction').click();
    await button(contribution, 'Reveal reconstruction').click();
    await scale.getByLabel('Operation (staged)', { exact: true }).selectOption('source');
    await commitRadio(scale, 'change', 'Apply scaling');
    const tinyScale = normalize(await scale.locator('[data-ica-result]').innerText());
    assert.match(tinyScale, /the observed sensors changed/);
    const scaleValues = tinyScale.match(/reconstruction went from \(([\d.e+−-]+), ([\d.e+−-]+)\) to \(([\d.e+−-]+), ([\d.e+−-]+)\)/);
    assert.ok(scaleValues, 'scale feedback preserves before and after sensor values');
    assert.ok(readNumber(scaleValues[3]) > readNumber(scaleValues[1]), 'source-only change is visible in sensor 1');
    assert.ok(readNumber(scaleValues[4]) > readNumber(scaleValues[2]), 'source-only change is visible in sensor 2');
    await capture(scale.locator('.ic-prediction'), 'scale-tiny-change');

    await button(contribution, 'Reset').click();
    await text(contribution.locator('[data-ica-notice]'), /Change at least one amplitude to start/);
    assert.equal(await lab('scale').count(), 0, 'parent Reset closes the follow-up investigation');
    assert.equal(await page.locator('[data-ica-scale-gate]').count(), 1);
    await contribution.getByLabel('Predicted sensor 1 amplitude', { exact: true }).fill('1');
    assert.ok(await button(contribution, 'Commit prediction').isDisabled(), 'Reset restores the first independent-task gate');
    // Restore a useful changed state for the existing responsive captures.
    await contribution.getByLabel('Source 1 amplitude (staged)', { exact: true }).fill('2');
    await contribution.getByLabel('Source 2 amplitude (staged)', { exact: true }).fill('0');
    await contribution.getByLabel('Keep-set (staged)', { exact: true }).selectOption('first');
    await contribution.getByLabel('Predicted sensor 1 amplitude', { exact: true }).fill('4');
    await button(contribution, 'Commit prediction').click();
    await button(contribution, 'Reveal reconstruction').click();
    assert.equal(await scale.locator('[data-ica-scale-unrevealed]').count(), 1);
    for (const [prediction, accepted] of [
      ['4.000000001', true], ['3.999999999', true],
      ['4.00000000101', false], ['3.99999999899', false],
    ]) {
      await contribution.getByLabel('Predicted sensor 1 amplitude', { exact: true }).fill(prediction);
      await button(contribution, 'Commit prediction').click();
      await button(contribution, 'Reveal reconstruction').click();
      const boundaryResult = normalize(await contribution.locator('[data-ica-result]').innerText());
      assert.match(boundaryResult, accepted ? /a match within tolerance/ : /a difference of/, `inclusive 1e-9 boundary for ${prediction}`);
    }
    assert.doesNotMatch(await page.locator('.ic-figure-e1').innerText(), /correlations above use every one of the 20,000 instants/);
    records.push('Sequential audit: tiny graded R1/C1/C1b changes remain visible, source-only c=1 has identity feedback, parent Reset restores the independent-task gate, inclusive C1 tolerance handles floating-point dust, and E1 respects the correlation intervals');

    // ---- layout, responsive and display math ------------------------------
    for (const width of [1366, 768, 390, 320]) {
      await page.setViewportSize({ width, height: 1000 });
      await page.evaluate(() => document.fonts.ready);
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), `page overflow at ${width}`);
      const overflowingMath = await page.$$eval('.ica-lesson .katex-display', nodes => nodes
        .filter(node => node.scrollWidth > node.clientWidth + 1)
        .map(node => node.textContent.trim().slice(0, 70)));
      assert.deepEqual(overflowingMath, [], `display math overflows at ${width}`);
      // A clipped numeral reads as a complete, wrong number. Below the stacking
      // breakpoint these tables become one labelled block per row, so nothing may
      // overflow; above it they are ordinary scrollable regions, where content
      // wider than the column is reachable rather than hidden.
      if (width <= 560) {
        const stacked = await page.$$eval('.ica-lesson .ic-table-wrap.is-stack',
          nodes => nodes.map(node => getComputedStyle(node.querySelector('thead')).display));
        assert.deepEqual(stacked.filter(display => display !== 'none'), [],
          `an exact-value table did not stack at ${width}`);
        const clipped = await page.$$eval('.ica-lesson .ic-table-wrap.is-stack', nodes => nodes
          .filter(node => node.scrollWidth > node.clientWidth + 1)
          .map(node => node.textContent.trim().slice(0, 60)));
        assert.deepEqual(clipped, [], `an exact-value table is clipped at ${width}`);
      }
      layouts.push({ width, report: await page.evaluate(inspectLessonVisualLayout, '.ica-lesson') });
      if ([1366, 390].includes(width)) {
        for (const key of ['m1', 'd1', 'w1', 'f1', 'p1', 'e1']) await capture(page.locator(`.ic-figure-${key}`), `${key}-${width}`);
      }
      if (width === 390) {
        for (const kind of ['rotation', 'contribution', 'scale']) await capture(lab(kind), `${kind}-390`);
        // P1's stage summary stacks rather than breaking its labels mid-word.
        assert.equal(await page.locator('.ic-figure-p1 .ic-lane-row.is-output .ic-lane-stages').evaluate(node =>
          getComputedStyle(node).gridTemplateColumns.split(' ').length), 1, 'P1 stacks its stages on a phone');
      }
      if (width === 320) {
        for (const key of ['m1', 'd1', 'w1', 'f1', 'p1', 'e1']) await capture(page.locator(`.ic-figure-${key}`), `${key}-320`);
        await capture(lab('rotation'), 'rotation-320');
        await capture(lab('contribution'), 'contribution-320');
      }
    }
    // 640 CSS px at 200% browser zoom is the same layout as 320 CSS px.
    const zoomContext = await browser.newContext({ viewport: { width: 320, height: 700 }, deviceScaleFactor: 2 });
    const zoomPage = await zoomContext.newPage();
    await zoomPage.goto(route, { waitUntil: 'networkidle' });
    await zoomPage.locator('.ica-lesson').waitFor();
    await zoomPage.evaluate(() => document.fonts.ready);
    assert.ok(await zoomPage.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'no page overflow at the 200% zoom equivalent');
    await zoomPage.evaluate(() => [...document.body.querySelectorAll('*')].forEach(element => {
      if (!element.closest('main') && ['fixed', 'sticky'].includes(getComputedStyle(element).position)) element.style.visibility = 'hidden';
    }));
    await zoomPage.locator('[data-ica-lab=contribution]').screenshot({ path: 'docs/teaching/evidence/screenshots/ica-contribution-zoom.png' });
    screenshots.push('docs/teaching/evidence/screenshots/ica-contribution-zoom.png');
    await zoomContext.close();
    await page.setViewportSize({ width: 780, height: 1000 });
    await page.evaluate(() => { document.documentElement.style.fontSize = '200%'; });
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), '200% root text preserves wrapping at 780 px');
    await page.evaluate(() => { document.documentElement.style.fontSize = ''; });
    await page.setViewportSize({ width: 1366, height: 1000 });
    records.push('1366/768/390/320 layout inspection with no page, display-math or exact-table overflow, every figure captured at three widths, P1 stacking on a phone, 200% zoom equivalent');

    // ---- loading, completion and recovery ---------------------------------
    const otherBodies = Object.entries(build).filter(([key]) => key.startsWith('src/learn/data/topics/') && key !== source).map(([, entry]) => entry.file);
    assert.equal(requests.filter(url => otherBodies.some(file => url.endsWith(file))).length, 0, 'no other lesson body loaded');
    assert.equal(requests.filter(url => /r01-first20s\.csv/.test(url) && !url.includes('learn-assets')).length, 0);
    await page.locator('.reader-complete').click();
    await text(page.locator('.reader-complete'), /Completed/);
    await page.reload();
    await page.locator('.ica-lesson').waitFor();
    await text(page.locator('.reader-complete'), /Completed/);
    await page.locator('.reader-complete').click();
    assert.deepEqual(errors, []);
    const failureContext = await browser.newContext();
    const failurePage = await failureContext.newPage();
    await failurePage.route(`**/${build[source].file}`, intercepted => intercepted.abort());
    await failurePage.goto(route);
    await failurePage.locator('.lesson-load-error').waitFor();
    assert.ok(await failurePage.locator('.reader-complete').isDisabled());
    await failurePage.unroute(`**/${build[source].file}`);
    await failurePage.getByRole('button', { name: 'Try again', exact: true }).click();
    try { await failurePage.locator('.ica-lesson').waitFor({ timeout: 3000 }); }
    catch { await failurePage.getByRole('button', { name: 'Reload page', exact: true }).click(); }
    await failurePage.locator('.ica-lesson').waitFor();
    assert.equal(await failurePage.locator('.reader-complete').isDisabled(), false);
    await failureContext.close();
    records.push('only the active lesson body loads, the CSV is fetched by nobody on mount, completion persists across reload, failed import recovers');

    const payload = {
      lessonChunk: build[source].file,
      decodedBytes: fs.statSync(`${distDir}/${build[source].file}`).size,
      gzipBytes: gzipSync(fs.readFileSync(`${distDir}/${build[source].file}`)).length,
    };
    const evidence = {
      checkedAt: new Date().toISOString(),
      status: 'passed',
      distDir,
      baseUrl: base,
      sources,
      buildHash: hash(`${distDir}/.vite/manifest.json`),
      records,
      screenshots,
      layouts,
      payload,
      errors,
      recordingResults: ICA_RECORDING.results.map(item => ({ method: item.method, chosen: item.chosen, testAbs: item.testAbs })),
    };
    fs.writeFileSync('docs/teaching/evidence/ica-browser.json', `${JSON.stringify(evidence, null, 2)}\n`);
    console.log(`PASS: ${records.length} browser behaviour groups; ${screenshots.length} captures; payload ${payload.gzipBytes} gzip bytes.`);
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });

// Production checks for the prepared manifold lesson and its four investigations.
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { gzipSync } = require('node:zlib');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const { inspectLessonVisualLayout } = require('./lib/lesson-visual-layout.cjs');

const base = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4183';
const id = 't-sne-umap-manifold-learning';
const route = `${base}/learn/path/full-curriculum/${id}?module=classical-ml`;
const source = `src/learn/data/topics/${id}.jsx`;
const hash = file => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const normalize = value => value.replace(/\s+/g, ' ').trim();
const records = [], screenshots = [], layouts = [];
const owned = [source, 'src/learn/data/manifold-models.js', 'src/learn/data/manifold-data.js',
  'src/learn/data/manifold-examples.js', 'src/learn/components/lesson-labs/ManifoldLabs.jsx',
  'src/learn/components/lesson-labs/ManifoldShared.jsx', 'src/learn/components/lesson-labs/ManifoldFigures.jsx',
  'src/index.css', `src/learn/data/curriculum/blueprints/${id}.js`,
  ...['manifold-labs', 'manifold-figures', 'manifold-lesson'].map(name => `src/learn/components/lesson-labs/${name}.css`)];

(async () => {
  const build = JSON.parse(fs.readFileSync('dist/.vite/manifest.json', 'utf8'));
  const sources = Object.fromEntries(owned.map(file => [file, hash(file)]));
  const { manifoldExamples } = await import('../src/learn/data/manifold-examples.js');
  const { MANIFOLD_DIGITS } = await import('../src/learn/data/manifold-data.js');
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const context = await browser.newContext({ viewport: { width: 1366, height: 1000 } });
  const page = await context.newPage();
  const errors = [], requests = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('request', request => requests.push(request.url()));
  const text = async (target, pattern) => assert.match(normalize(await target.innerText()), pattern);
  const button = (target, name) => target.getByRole('button', { name, exact: true });
  const lab = kind => page.locator(`[data-manifold-lab="${kind}"]`);
  const choose = async (target, prediction, action) => {
    await target.getByRole('radio', { name: prediction, exact: true }).check();
    await button(target, 'Commit prediction').click();
    await button(target, action).click();
  };
  const predictNumber = async (target, value, action) => {
    await target.getByLabel('Your predicted value', { exact: true }).fill(String(value));
    await button(target, 'Commit prediction').click();
    await button(target, action).click();
  };
  const capture = async (target, name) => {
    const destination = `docs/teaching/evidence/screenshots/manifold-${name}.png`;
    fs.mkdirSync(path.dirname(destination), { recursive: true });
    await target.screenshot({ path: destination });
    screenshots.push(destination);
  };
  try {
    await page.goto(route, { waitUntil: 'networkidle' });
    await page.locator('.manifold-lesson').waitFor();
    await page.waitForFunction(() => [...document.fonts].some(font => font.family.includes('Space Grotesk') && font.status === 'loaded'));
    await page.evaluate(() => document.fonts.ready);
    await page.addStyleTag({ content: '*{scroll-behavior:auto!important;animation:none!important;transition:none!important}' });
    await text(page.locator('.reader-header__meta'), /17 of 39 topics/);
    await text(page.locator('.reader-footer__previous'), /Gaussian Mixture/);
    await text(page.locator('.reader-footer__next'), /Independent Component Analysis/);
    assert.equal(await page.locator('.manifold-lesson > h2').count(), 13);
    assert.equal(await page.locator('[data-manifold-figure]').count(), 9);
    assert.equal(await page.locator('[data-manifold-lab]').count(), 4);
    assert.equal(await page.locator('.python-example').count(), 4);
    assert.equal(await page.locator('.manifold-lesson > details').count(), 14);
    assert.equal(await page.locator('.manifold-lesson > details[open]').count(), 0);
    for (const anchor of await page.locator('.lesson-intro a[href^="#"]').evaluateAll(items => items.map(item => item.hash))) {
      assert.equal(await page.locator(`[id="${anchor.slice(1)}"]`).count(), 1);
    }
    const body = normalize(await page.locator('.manifold-lesson').textContent());
    for (const example of Object.values(manifoldExamples)) {
      assert.ok(body.includes(normalize(example.code)), `Complete displayed ${example.file}`);
      assert.ok(body.includes(normalize(example.expected)), `Actual output ${example.file}`);
    }
    assert.equal(await page.locator('.katex-error').count(), 0);
    assert.doesNotMatch(body, /not been executed here|execution.*deferred|\[Inline figure F/);
    for (const asset of ['digits-300.csv', 'calculated-inputs.json', 'data-provenance.md']) {
      const response = await page.request.get(`${base}/learn-assets/manifold-learning/${asset}`);
      assert.equal(response.status(), 200);
      if (asset.endsWith('.csv')) assert.equal((await response.text()).trim().split('\n').length, 301);
      if (asset.endsWith('.json')) assert.ok((await response.json()).layouts);
    }
    records.push('13 complete sections,9 inline figures,4 complete programs,4 labs,7 closed hint/solution pairs,assets and module sequence');
    for (const kind of ['graph', 'probability', 'fuzzy', 'digits']) {
      assert.equal(await lab(kind).locator('input[type=radio]:checked').count(), 0);
      assert.ok(await button(lab(kind), 'Commit prediction').isDisabled());
    }

    const graph = lab('graph');
    await button(graph, 'Draft U · ε 2').click();
    await choose(graph, 'Shorter', 'Apply graph');
    await text(graph.locator('[data-manifold-distance]'), /shortest path: 2 Direct/);
    await text(graph.locator('[data-manifold-result-announcement]'), /matches.*6 → 2/);
    assert.equal(await graph.locator('[data-manifold-status]').getAttribute('aria-live'), 'off');
    await button(graph, 'Draft ε 2.5').click();
    await choose(graph, 'Same length', 'Apply graph');
    await text(graph.locator('[data-manifold-result]'), /matches/);
    await button(graph, 'Draft U · ε 0.75').click();
    await choose(graph, 'No path', 'Apply graph');
    await text(graph.locator('[data-manifold-distance]'), /shortest path: No path/);
    await button(graph, 'Draft U · ε 1.5').click();
    await choose(graph, 'A path will exist', 'Apply graph');
    await text(graph.locator('[data-manifold-distance]'), /4\.82843/);
    await graph.getByLabel('D y coordinate', { exact: true }).fill('');
    assert.ok(await button(graph, 'Commit prediction').isDisabled());
    await text(graph.locator('[data-manifold-distance]'), /4\.82843/);
    await button(graph, 'Reset').click();
    await graph.getByLabel('D x coordinate', { exact: true }).fill('1.25');
    await choose(graph, 'No path', 'Apply graph');
    await text(graph.locator('[data-manifold-distance]'), /No path/);
    await button(graph, 'Undo apply').click();
    await text(graph.locator('[data-manifold-distance]'), /shortest path: 6 Direct/);
    records.push('Graph shorter/null/disconnected/reconnected,arbitrary coordinate edit,invalid draft preserves applied route,and undo');

    const probability = lab('probability');
    await probability.getByLabel('Proposed bandwidth σ', { exact: true }).fill('2');
    await choose(probability, 'Decrease', 'Apply bandwidth');
    await text(probability.locator('[data-manifold-result]'), /matches/);
    await button(probability, 'Draft distances 2, 2, 2').click();
    await probability.getByLabel('Proposed bandwidth σ', { exact: true }).fill('0.5');
    await probability.getByLabel('Predict a quantity', { exact: true }).selectOption('perplexity');
    await choose(probability, 'Stay the same', 'Apply bandwidth');
    await text(probability.locator('[data-manifold-result]'), /Perplexity: 3 → 3/);
    await text(probability.locator('[data-manifold-probability=B]'), /0\.333333/);
    await button(probability, 'Reset').click();
    await probability.getByRole('radio', { name: 'Increase', exact: true }).check();
    await button(probability, 'Commit prediction').click();
    await probability.getByLabel('Distance to D', { exact: true }).fill('4');
    await probability.getByLabel('Distance to D', { exact: true }).fill('3');
    assert.ok(await button(probability, 'Apply bandwidth').isDisabled());
    assert.equal(await probability.locator('input[type=radio]:checked').count(), 0);
    records.push('Affinity changed-bandwidth effect,equal-distance entropy null,and round-trip edit invalidates committed prediction');

    // Near saturation must not grade a change while printing identical values.
    await button(probability, 'Reset').click();
    for (const [candidate, distance] of [['B', '0.25'], ['C', '3'], ['D', '6']]) {
      await probability.getByLabel(`Distance to ${candidate}`, { exact: true }).fill(distance);
    }
    await probability.getByLabel('Proposed bandwidth σ', { exact: true }).fill('0.25');
    await choose(probability, 'Increase', 'Apply bandwidth');
    await probability.getByLabel('Proposed bandwidth σ', { exact: true }).fill('0.5');
    await choose(probability, 'Decrease', 'Apply bandwidth');
    await text(probability.locator('[data-manifold-result]'), /matches.*probability: 1 → 0\.999999982742/);
    await probability.getByLabel('Predict a quantity', { exact: true }).selectOption('perplexity');
    await probability.getByLabel('Proposed bandwidth σ', { exact: true }).fill('0.25');
    await choose(probability, 'Decrease', 'Apply bandwidth');
    await probability.getByLabel('Proposed bandwidth σ', { exact: true }).fill('0.5');
    await choose(probability, 'Increase', 'Apply bandwidth');
    await text(probability.locator('[data-manifold-result]'), /matches.*Perplexity: 1 → 1\.00000032574/);
    await capture(probability.locator('.mf-prediction'), 'near-saturation-comparison');
    await button(probability, 'Reset').click();
    records.push('Near-saturation probability and entropy comparisons expose their graded difference rather than printing1→1');

    const fuzzy = lab('fuzzy');
    for (const [estimate, accepted] of [[0.62, true], [0.63, true], [0.619, false], [0.631, false]]) {
      await predictNumber(fuzzy, estimate, 'Apply connection');
      const result = await fuzzy.locator('[data-manifold-result]').first().innerText();
      assert.equal(result.includes('Your prediction matches.'), accepted, `Inclusive tolerance for estimate${estimate}`);
    }
    records.push('Fuzzy estimates atboth inclusive0.005 boundaries accepted;neighboring0.006 errors rejected');
    await predictNumber(fuzzy, 0.625, 'Apply connection');
    await text(fuzzy.locator('[data-manifold-union]'), /^0\.625$/);
    await fuzzy.getByText('Explore the ideal pair cost with this applied weight', { exact: true }).click();
    const pair = fuzzy.locator('[data-manifold-pair]');
    await pair.getByLabel('Proposed separation r', { exact: true }).fill('2');
    await choose(pair, 'Higher cost', 'Apply separation');
    await text(pair.locator('[data-manifold-result]'), /1\.08958/);
    await pair.getByLabel("Reflect the pair's orientation", { exact: true }).check();
    await choose(pair, 'Same cost', 'Apply separation');
    await text(pair.locator('[data-manifold-result]'), /matches.*Reflection/);
    await fuzzy.getByLabel('Shared distance d', { exact: true }).fill('2.25');
    assert.equal(await pair.getByLabel('Proposed separation r', { exact: true }).inputValue(), '2');
    await text(pair.locator('.mf-readout'), /1\.08958/);
    assert.ok(await button(pair, 'Apply separation').isDisabled());
    await fuzzy.getByText('Explore the ideal pair cost with this applied weight', { exact: true }).click();
    await button(fuzzy, 'Draft neither direction retained').click();
    await predictNumber(fuzzy, 0, 'Apply connection');
    await text(fuzzy.locator('[data-manifold-union]'), /^0$/);
    await fuzzy.getByLabel('Local scale σᵢ', { exact: true }).fill('2.75');
    await predictNumber(fuzzy, 0, 'Apply connection');
    await text(fuzzy.locator('[data-manifold-result]').first(), /matches.*cannot create this edge/);
    await button(fuzzy, 'Reset').click();
    await button(fuzzy, 'Draft both offsets at d').click();
    await predictNumber(fuzzy, 1, 'Apply connection');
    await text(fuzzy.locator('[data-manifold-union]'), /^1$/);
    records.push('Fuzzy union0.625,support-null,changed scale,union1 boundary,and ideal-pair cost/reflection null');

    const digits = lab('digits');
    await digits.getByLabel('Query source-row ID', { exact: true }).selectOption('30');
    assert.equal(await digits.locator('[data-manifold-neighbors]').count(), 0);
    await predictNumber(digits, 5, 'Reveal neighbors');
    await text(digits.locator('[data-manifold-retained]'), /^5 \/ 10/);
    const inputs = await digits.locator('[data-manifold-neighbors=input] li').evaluateAll(items => items.map(item => Number(item.dataset.sourceRow)));
    assert.deepEqual(inputs, [0, 166, 229, 160, 36, 276, 140, 266, 178, 79]);
    await digits.getByLabel('Candidate saved map', { exact: true }).selectOption('pca');
    assert.equal(await digits.locator('[data-manifold-neighbors]').count(), 0);
    await predictNumber(digits, 6, 'Reveal neighbors');
    await text(digits.locator('[data-manifold-retained]'), /^6 \/ 10/);
    await digits.getByLabel('Candidate saved map', { exact: true }).selectOption('tsne-p30-s19');
    await predictNumber(digits, 5, 'Reveal neighbors');
    await text(digits.locator('[data-manifold-retained]'), /^5 \/ 10/);
    await digits.getByLabel('Neighbor count k', { exact: true }).selectOption('20');
    assert.equal(await digits.locator('[data-manifold-neighbors]').count(), 0);
    await predictNumber(digits, 10, 'Reveal neighbors');
    assert.equal(await digits.locator('[data-manifold-neighbors] li').count(), 40);
    await digits.getByLabel('Query source-row ID', { exact: true }).focus();
    await page.keyboard.press('ArrowDown');
    assert.equal(await digits.locator('[data-manifold-neighbors]').count(), 0);
    assert.equal(await digits.getByLabel('Query source-row ID', { exact: true }).locator('option').count(), 300);
    records.push('Real source30 local ranking reversal,PCA-init seed null,query/k/map invalidate reveal,40-tile bound,300-source keyboard alternative');

    // Hide only page chrome for diagnostic captures, preserving all lesson content.
    await page.evaluate(() => [...document.body.querySelectorAll('*')].forEach(element => {
      if (!element.closest('main') && ['fixed', 'sticky'].includes(getComputedStyle(element).position)) element.style.visibility = 'hidden';
    }));
    for (const width of [1366, 768, 390, 320]) {
      await page.setViewportSize({ width, height: 1000 });
      await page.evaluate(() => document.fonts.ready);
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), `Page overflow at ${width}`);
      layouts.push({ width, report: await page.evaluate(inspectLessonVisualLayout, '.manifold-lesson') });
      if ([1366, 390].includes(width)) {
        for (const kind of ['graph', 'probability', 'fuzzy', 'digits']) {
          if (kind === 'digits') {
            await digits.getByLabel('Query source-row ID', { exact: true }).selectOption('30');
            await digits.getByLabel('Neighbor count k', { exact: true }).selectOption('10');
            await predictNumber(digits, 5, 'Reveal neighbors');
          }
          await capture(lab(kind), `${kind}-${width}`);
          if (kind === 'digits' && width === 390) {
            await capture(digits.locator('.mf-candidate-map'), 'digits-map-390');
            await capture(digits.locator('[data-manifold-neighbors=input]'), 'digits-input-neighbors-390');
            await capture(digits.locator('[data-manifold-neighbors=map]'), 'digits-map-neighbors-390');
          }
        }
      }
      if (width === 390) {
        const before = await digits.locator('[data-manifold-retained]').textContent();
        await button(digits, 'PCA reference').click();
        assert.equal(await digits.locator('.mf-candidate-map').isVisible(), false);
        await button(digits, 'Candidate map').click();
        assert.equal(await digits.locator('[data-manifold-retained]').textContent(), before);
      }
    }
    // Browser zoom changes the CSS viewport and pixel density together. CSS
    // zoom alone would leave media queries at640px, unlike actual200% zoom.
    const zoomContext = await browser.newContext({ viewport: { width: 320, height: 600 }, deviceScaleFactor: 2 });
    const zoomPage = await zoomContext.newPage();
    await zoomPage.goto(route, { waitUntil: 'networkidle' });
    await zoomPage.locator('.manifold-lesson').waitFor();
    await zoomPage.evaluate(() => document.fonts.ready);
    assert.ok(await zoomPage.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), '640px at200% zoom equivalent:320CSSpx,no page overflow');
    await zoomPage.locator('[data-manifold-lab=graph]').screenshot({ path: 'docs/teaching/evidence/screenshots/manifold-graph-zoom.png' });
    screenshots.push('docs/teaching/evidence/screenshots/manifold-graph-zoom.png');
    await zoomContext.close();
    await page.setViewportSize({ width: 780, height: 1000 });
    await page.evaluate(() => { document.documentElement.style.fontSize = '200%'; });
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), '200% root text preserves wrapping at780px');
    await page.evaluate(() => { document.documentElement.style.fontSize = ''; });
    records.push('1366/768/390/320 layout inspection,desktop/narrow informative captures,200% zoom equivalent,and display-only tabs preserve answers');

    const otherBodies = Object.entries(build).filter(([key]) => key.startsWith('src/learn/data/topics/') && key !== source).map(([, entry]) => entry.file);
    assert.equal(requests.filter(url => otherBodies.some(file => url.endsWith(file))).length, 0, 'No other lesson body loaded');
    assert.equal(requests.filter(url => /umap-digits|calculated-inputs/.test(url)).length, 0, 'No external fit/download on mount');
    await page.locator('.reader-complete').click();
    await text(page.locator('.reader-complete'), /Completed/);
    await page.reload();
    await page.locator('.manifold-lesson').waitFor();
    await text(page.locator('.reader-complete'), /Completed/);
    await page.locator('.reader-complete').click();
    const revisitCount = requests.filter(url => url.endsWith(build[source].file)).length;
    await page.locator('.reader-footer__next').click();
    await page.waitForURL(/independent-component-analysis-ica/);
    await page.goBack();
    await page.locator('.manifold-lesson').waitFor();
    assert.equal(requests.filter(url => url.endsWith(build[source].file)).length, revisitCount, 'Revisit reuses loaded lesson');
    assert.deepEqual(errors, []);
    records.push('Only active lesson body loaded;completion persists;revisit module cache reused;no render errors');
    const failureContext = await browser.newContext();
    const failurePage = await failureContext.newPage();
    await failurePage.route(`**/${build[source].file}`, intercepted => intercepted.abort());
    await failurePage.goto(route);
    await failurePage.locator('.lesson-load-error').waitFor();
    assert.ok(await failurePage.locator('.reader-complete').isDisabled());
    await failurePage.unroute(`**/${build[source].file}`);
    await failurePage.getByRole('button', { name: 'Try again', exact: true }).click();
    // A rejected browser module can stay cached; the UI explicitly offers a
    // full reload. Test that recovery path without misreporting retry success.
    try { await failurePage.locator('.manifold-lesson').waitFor({ timeout: 3000 }); }
    catch { await failurePage.getByRole('button', { name: 'Reload page', exact: true }).click(); }
    await failurePage.locator('.manifold-lesson').waitFor();
    assert.equal(await failurePage.locator('.reader-complete').isDisabled(), false);
    await failureContext.close();
    records.push('Failed body import shows recovery UI and disables completion;retry/reload restores the lesson');
    const sharedLayoutSpotChecks = [];
    for (const neighbor of ['pca-dimensionality-reduction', 'gaussian-mixture-models-gmm-em-algorithm']) {
      await page.goto(`${base}/learn/path/full-curriculum/${neighbor}?module=classical-ml`);
      await page.locator('.reader-article').waitFor();
      await page.waitForFunction(() => document.querySelector('.reader-article')?.textContent.length > 5000);
      for (const width of [1366, 780, 390]) {
        await page.setViewportSize({ width, height: 1000 });
        assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), `${neighbor} shared flex layout at${width}`);
      }
      sharedLayoutSpotChecks.push({ topic: neighbor, widths: [1366, 780, 390], sourceHash: hash(`src/learn/data/topics/${neighbor}.jsx`) });
    }
    records.push('Shared reader min-width repair checked on unchanged PCA and GMM bodies at1366/780/390');
    const payload = { lessonChunk: build[source].file, decodedBytes: fs.statSync(`dist/${build[source].file}`).size,
      gzipBytes: gzipSync(fs.readFileSync(`dist/${build[source].file}`)).length };
    fs.writeFileSync('docs/teaching/evidence/manifold-browser.json', JSON.stringify({ checkedAt: new Date().toISOString(), status: 'passed', sources,
      buildHash: hash('dist/.vite/manifest.json'), records, screenshots, layouts, payload, errors, sharedLayoutSpotChecks, mapCount: MANIFOLD_DIGITS.layouts.length }, null, 2) + '\n');
    console.log(`PASS: ${records.length} browser behavior groups; ${screenshots.length} captures; ${MANIFOLD_DIGITS.layouts.length} actual maps; payload ${payload.gzipBytes} gzip bytes.`);
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

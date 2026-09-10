const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const directory = 'scratch/bayesian-inference-review/browser';
fs.mkdirSync(directory, { recursive: true });

async function range(page, lab, name, key) {
  const input = lab.getByRole('slider', { name, exact: true });
  await input.focus();
  await page.keyboard.press(key);
  assert.equal(await input.evaluate(node => document.activeElement === node), true);
  return Number(await input.inputValue());
}
async function readout(lab, name) {
  return lab.locator('.bayesian-readout > div').filter({ has: lab.page().locator('dt', { hasText: name }) }).locator('dd').innerText();
}

(async () => {
  const { bayesianInferenceExamples } = await import('../src/learn/data/bayesian-inference-examples.js');
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const widths = process.env.REVIEW_WIDTHS ? process.env.REVIEW_WIDTHS.split(',').map(Number) : [1440, 390, 320];
  const results = [];
  try {
    for (const width of widths) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      const errors = [], consoleErrors = [], environmental = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => {
        if (message.type() !== 'error') return;
        const text = message.text();
        if (text.includes('ERR_NETWORK_ACCESS_DENIED') || text.includes('WebSocket') || text.includes('websocket')) environmental.push(text);
        else consoleErrors.push(text);
      });
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/bayesian-inference-conjugate-priors?module=math-foundations', { waitUntil: 'networkidle' });
      const lesson = page.locator('.bayesian-inference-lesson');
      await lesson.waitFor();
      assert.equal(await lesson.locator('[data-investigation]').count(), 5);
      const anchors = [];
      for (const link of await lesson.locator('.lesson-intro nav a').all()) {
        const href = await link.getAttribute('href');
        await link.scrollIntoViewIfNeeded();
        await link.focus();
        await page.keyboard.press('Enter');
        const heading = await page.evaluate(hash => {
          const node = document.getElementById(hash.slice(1));
          return node ? { text: node.textContent, top: node.getBoundingClientRect().top } : null;
        }, href);
        assert.ok(heading && heading.top >= 40 && heading.top < 300, JSON.stringify({ href, heading }));
        anchors.push(heading.text);
        await page.screenshot({ path: `${directory}/reading-${width}-${anchors.length}.png` });
      }
      assert.equal(anchors.length, 10);
      const scope = id => lesson.locator(`[data-investigation="${id}"]`);
      const reset = async lab => lab.getByRole('button', { name: 'Reset investigation', exact: true }).click();
      const capture = async (lab, name) => {
        await lab.screenshot({ path: `${directory}/${name}-${width}.png` });
        await lab.locator('h3').scrollIntoViewIfNeeded();
        await page.screenshot({ path: `${directory}/${name}-reading-${width}.png` });
      };
      const update = scope('beta-update');
      assert.equal(await readout(update, 'Posterior mean'), '0.714286');
      assert.equal(await readout(update, 'Equal-tailed'), '[0.461868, 0.90908]');
      await update.getByRole('combobox', { name: 'Prior', exact: true }).selectOption('strong');
      assert.equal(await readout(update, 'Posterior mean'), '0.56');
      await range(page, update, 'Successes', 'Home');
      await range(page, update, 'Failures', 'Home');
      assert.match(await update.innerText(), /posterior equals prior/);
      await update.getByRole('combobox', { name: 'Prior', exact: true }).selectOption('uniform');
      assert.equal(await readout(update, 'Mode'), 'not unique');
      await capture(update, 'beta-uniform');
      await range(page, update, 'Successes', 'End');
      await range(page, update, 'Rate threshold', 'End');
      assert.equal(await readout(update, 'P(θ >'), '0');
      await capture(update, 'beta-boundary');
      await reset(update);
      await capture(update, 'beta');

      const batch = scope('batch-prediction');
      assert.equal(await readout(batch, 'Integrated selected tail'), '0.250736');
      assert.equal(await readout(batch, 'Plug-in selected tail'), '0.172858');
      await range(page, batch, 'Future batch size', 'Home');
      assert.equal(await readout(batch, 'Integrated selected tail'), await readout(batch, 'Plug-in selected tail'));
      await batch.getByRole('combobox', { name: 'Rate uncertainty' }).selectOption('100,40');
      await range(page, batch, 'Future batch size', 'End');
      await range(page, batch, 'At least this many successes', 'End');
      await capture(batch, 'batch-concentrated');
      await reset(batch);
      await capture(batch, 'batch');

      const exposure = scope('gamma-exposure');
      assert.equal(await readout(exposure, 'Posterior mean'), '3.142857');
      await range(page, exposure, 'First interval events', 'Home');
      await range(page, exposure, 'Second interval events', 'Home');
      assert.equal(await readout(exposure, 'Count / exposure'), '0');
      assert.ok(Number(await readout(exposure, 'Posterior mean')) > 0);
      await capture(exposure, 'exposure-zero');
      await reset(exposure);
      await capture(exposure, 'exposure');

      const normal = scope('normal-precision');
      assert.equal(await readout(normal, 'Posterior variance'), '0.5');
      assert.equal(await readout(normal, 'Variance of next'), '4.5');
      await range(page, normal, 'Independent measurement count', 'End');
      assert.equal(await readout(normal, 'Posterior variance'), '0.090909');
      assert.equal(await readout(normal, 'Variance of next'), '4.090909');
      await capture(normal, 'normal-more-data');
      await reset(normal);
      await capture(normal, 'normal');

      const patterns = scope('predictive-patterns');
      assert.match(await patterns.locator('.bayesian-result').innerText(), /0.131628/);
      await patterns.getByRole('combobox', { name: 'Predictive reference' }).selectOption('same-count');
      assert.match(await patterns.locator('.bayesian-result').innerText(), /0.044444/);
      await patterns.getByRole('button', { name: 'Failures separated', exact: true }).focus();
      await page.keyboard.press('Enter');
      assert.equal(await patterns.getByRole('button', { name: 'Failures separated' }).getAttribute('aria-pressed'), 'true');
      assert.match(await patterns.locator('.bayesian-result').innerText(), /probability: 1/);
      await capture(patterns, 'patterns-conditional');
      await reset(patterns);
      await capture(patterns, 'patterns');

      const initialDisclosures = await lesson.locator('details[open]').count();
      assert.equal(initialDisclosures, 0);
      const practices = lesson.locator('.lesson-check').filter({ has: page.locator('h3') });
      assert.equal(await practices.count(), 8);
      for (const practice of await practices.all()) assert.ok(await practice.locator(':scope > p').isVisible());
      for (const summary of await lesson.locator('summary').all()) {
        await summary.scrollIntoViewIfNeeded();
        await summary.focus();
        await page.keyboard.press('Enter');
        assert.equal(await summary.evaluate(node => node.parentElement.open), true);
      }
      const codeBlocks = lesson.locator('.python-example');
      assert.equal(await codeBlocks.count(), 11);
      for (const example of Object.values(bayesianInferenceExamples)) {
        const rendered = codeBlocks.filter({ has: page.getByRole('heading', { name: example.title, exact: true }) });
        const text = await rendered.innerText();
        assert.ok(text.includes(example.code.trim()), example.title);
        assert.ok(text.includes(example.expected.trim()), example.title);
      }
      for (const [index, figure] of (await lesson.locator('.bayesian-inline').all()).entries()) {
        await figure.scrollIntoViewIfNeeded();
        await page.screenshot({ path: `${directory}/inline-reading-${width}-${index}.png` });
      }
      const mathOverflow = await lesson.locator('.katex-display').evaluateAll(nodes => nodes.map((node, index) => ({ index, text: node.textContent.slice(0, 150), width: node.clientWidth, scroll: node.scrollWidth })).filter(item => item.scroll > item.width + 1));
      const svgOverflow = await lesson.locator('svg').evaluateAll(nodes => nodes.flatMap(svg => [...svg.querySelectorAll('text')].map(node => {
        const box = node.getBBox();
        return { text: node.textContent, left: box.x, right: box.x + box.width, limit: svg.viewBox.baseVal.width };
      }).filter(item => item.left < -1 || item.right > item.limit + 1)));
      const dimensions = await page.evaluate(() => ({ width: innerWidth, scroll: document.documentElement.scrollWidth }));
      await lesson.locator('.lesson-sources').scrollIntoViewIfNeeded();
      await page.screenshot({ path: `${directory}/sources-reading-${width}.png` });
      const result = { width, at: new Date().toISOString(), anchors, codeExamples: 11, visiblePracticeQuestions: 8, initialDisclosuresClosed: true, mathOverflow, svgOverflow, dimensions, errors, consoleErrors, environmental };
      results.push(result);
      fs.writeFileSync(`${directory}/results.json`, JSON.stringify({ at: new Date().toISOString(), results }, null, 2));
      assert.deepEqual(errors, []);
      assert.deepEqual(consoleErrors, []);
      assert.deepEqual(svgOverflow, []);
      assert.ok(dimensions.scroll <= dimensions.width + 1);
      // Preserve every width's evidence before reporting any long formula.
      await page.close();
    }
    console.log(JSON.stringify(results));
    assert.ok(results.every(result => result.mathOverflow.length === 0), 'Reflow the reported mathematical lines.');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

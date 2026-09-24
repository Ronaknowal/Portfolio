const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const directory = 'scratch/probability-distributions-review/browser';
fs.mkdirSync(directory, { recursive: true });

async function range(page, scope, name, value) {
  const locator = scope.getByRole('slider', { name, exact: true });
  const settings = await locator.evaluate(node => ({ minimum: Number(node.min), step: Number(node.step) }));
  await locator.focus();
  await page.keyboard.press('Home');
  for (let index = 0; index < Math.round((value - settings.minimum) / settings.step); index += 1) await page.keyboard.press('ArrowRight');
  assert.ok(Math.abs(Number(await locator.inputValue()) - value) < 1e-9, name);
}

async function readout(scope, label) {
  return scope.locator('dl > div').filter({ has: scope.page().locator('dt', { hasText: label }) }).locator('dd').innerText();
}

(async () => {
  const { probabilityDistributionExamples } = await import('../src/learn/data/probability-distributions-examples.js');
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const widths = process.env.REVIEW_WIDTHS ? process.env.REVIEW_WIDTHS.split(',').map(Number) : [1440, 390, 320];
  const results = [];
  try {
    for (const width of widths) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/probability-distributions-bayes-theorem?module=math-foundations', { waitUntil: 'networkidle' });
      const lesson = page.locator('.probability-distributions-lesson');
      await lesson.waitFor();
      assert.equal(await lesson.locator('[data-investigation]').count(), 6);
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
      assert.equal(anchors.length, 9);
      const scope = id => lesson.locator(`[data-investigation="${id}"]`);
      const reset = async lab => lab.getByRole('button', { name: 'Reset investigation', exact: true }).click();
      const capture = async (lab, label) => {
        await lab.screenshot({ path: `${directory}/${label}-${width}.png` });
        await lab.locator('h3').scrollIntoViewIfNeeded();
        await page.screenshot({ path: `${directory}/${label}-reading-${width}.png` });
      };
      const events = scope('probability-events');
      assert.equal(await readout(events, 'P(A|B)'), '0.666667');
      for (const face of [4, 5, 6]) {
        await events.getByRole('button', { name: `Given event B, face ${face}`, exact: true }).focus();
        await page.keyboard.press('Enter');
      }
      assert.equal(await readout(events, 'P(A|B)'), 'undefined: zero evidence');
      await capture(events, 'events-empty');
      await reset(events);
      await events.getByRole('button', { name: 'Given event B, face 4', exact: true }).click();
      assert.equal(await readout(events, 'P(A|B)'), '0.5');
      await reset(events);
      await capture(events, 'events');

      const bayes = scope('probability-bayes');
      assert.match(await bayes.innerText(), /8\.75576%/);
      await range(page, bayes, 'Prior P(H)', .2);
      assert.match(await bayes.innerText(), /70\.37037%/);
      await bayes.getByRole('button', { name: 'Not flagged −', exact: true }).click();
      assert.match(await bayes.innerText(), /1\.369863%/);
      await capture(bayes, 'bayes-negative');
      await range(page, bayes, 'Sensitivity P(+|H)', 0);
      await range(page, bayes, 'False-positive rate P(+|not H)', 0);
      await bayes.getByRole('button', { name: 'Flagged +', exact: true }).click();
      assert.match(await bayes.getByRole('status').innerText(), /zero total probability/);
      await reset(bayes);
      await capture(bayes, 'bayes');

      const evidence = scope('probability-evidence');
      await range(page, evidence, 'Copy share c', .5);
      assert.equal(await readout(evidence, 'Actual posterior P(H|pair)'), '14.537964%');
      await capture(evidence, 'evidence-partial');
      await range(page, evidence, 'Copy share c', 1);
      assert.equal(await readout(evidence, 'Actual posterior P(H|pair)'), '8.75576%');
      await evidence.getByRole('button', { name: '+−', exact: true }).click();
      assert.equal(await readout(evidence, 'Actual posterior P(H|pair)'), 'undefined: zero evidence');
      await reset(evidence);

      const counts = scope('probability-counts');
      assert.equal(await readout(counts, 'P(X ≤ 1)'), '0.8');
      await range(page, counts, 'Marked objects', 3);
      await range(page, counts, 'Number of draws', 2);
      assert.equal(await readout(counts, 'Variance of count'), '0.4');
      await counts.getByRole('button', { name: 'Replace and remix', exact: true }).click();
      assert.equal(await readout(counts, 'Variance of count'), '0.5');
      await counts.getByRole('button', { name: 'Keep it out', exact: true }).click();
      await range(page, counts, 'Number of draws', 6);
      assert.equal(await readout(counts, 'Variance of count'), '0');
      await range(page, counts, 'Threshold k in X ≤ k', 3);
      assert.equal(await readout(counts, 'P(X ≤ 3)'), '1');
      await capture(counts, 'counts-all');
      await reset(counts);
      await capture(counts, 'counts');

      const density = scope('probability-density');
      assert.equal(await readout(density, 'Total interval probability'), '0.5');
      await range(page, density, 'Immediate completion probability q', .3);
      assert.equal(await readout(density, 'Total interval probability'), '0.35');
      await density.getByRole('button', { name: 'milliseconds', exact: true }).click();
      assert.equal(await readout(density, 'Total interval probability'), '0.35');
      assert.match(await density.innerText(), /0\.0035 per millisecond/);
      await capture(density, 'density-units');
      await range(page, density, 'Left endpoint as fraction of width', 0);
      await range(page, density, 'Right endpoint as fraction of width', 0);
      assert.equal(await readout(density, 'Total interval probability'), '0.3');
      assert.equal(await readout(density, 'Continuous interval area'), '0');
      await capture(density, 'density-atom');
      await range(page, density, 'Immediate completion probability q', 1);
      assert.equal(await readout(density, 'Total interval probability'), '1');
      await reset(density);
      await capture(density, 'density');

      const arrivals = scope('probability-arrivals');
      assert.equal(await readout(arrivals, 'P(no arrivals in window)'), '0.049787');
      assert.equal(await readout(arrivals, 'P(first wait > window)'), '0.049787');
      await range(page, arrivals, 'Rate, events per minute', 6);
      await range(page, arrivals, 'Observation window, minutes', 2);
      assert.match(await arrivals.innerText(), /At least 12 arrivals/);
      await range(page, arrivals, 'First-wait quantile probability', .99);
      await capture(arrivals, 'arrivals-tail');
      await range(page, arrivals, 'Observation window, minutes', 0);
      assert.equal(await readout(arrivals, 'P(no arrivals in window)'), '1');
      assert.equal(await readout(arrivals, 'P(at least one arrival)'), '0');
      await reset(arrivals);
      await capture(arrivals, 'arrivals');
      for (const figure of ['.probability-tail-figure', '.probability-normal-units']) {
        await lesson.locator(figure).screenshot({ path: `${directory}/${figure.slice(1)}-${width}.png` });
      }

      const codeBlocks = lesson.locator('.python-example');
      assert.equal(await codeBlocks.count(), 9);
      for (const example of Object.values(probabilityDistributionExamples)) {
        const rendered = codeBlocks.filter({ has: page.getByRole('heading', { name: example.title, exact: true }) });
        const text = await rendered.innerText();
        assert.ok(text.includes(example.code.trim()), example.title);
        assert.ok(text.includes(example.expected.trim()), example.title);
      }
      const initiallyOpen = await lesson.locator('details[open]').count();
      assert.equal(initiallyOpen, 0);
      for (const summary of await lesson.locator('summary').all()) {
        await summary.scrollIntoViewIfNeeded();
        await summary.focus();
        await page.keyboard.press('Enter');
        assert.ok(await summary.evaluate(node => node.parentElement.open));
      }
      const mathOverflow = await lesson.locator('.katex-display').evaluateAll(nodes => nodes.map(node => ({ text: node.textContent.slice(0, 110), width: node.clientWidth, scroll: node.scrollWidth })).filter(item => item.scroll > item.width + 1));
      const svgOverflow = await lesson.locator('svg').evaluateAll(nodes => nodes.flatMap(svg => [...svg.querySelectorAll('text')].map(node => {
        const box = node.getBBox();
        return { text: node.textContent, left: box.x, right: box.x + box.width, limit: svg.viewBox.baseVal.width };
      }).filter(item => item.left < -1 || item.right > item.limit + 1)));
      const dimensions = await page.evaluate(() => ({ width: innerWidth, scroll: document.documentElement.scrollWidth }));
      await lesson.locator('.lesson-sources').screenshot({ path: `${directory}/sources-${width}.png` });
      const result = { width, checkedAt: new Date().toISOString(), anchors, codeExamples: 9, initialDisclosuresClosed: initiallyOpen === 0, mathOverflow, svgOverflow, dimensions, errors };
      results.push(result);
      fs.writeFileSync(`${directory}/results.json`, JSON.stringify({ checkedAt: new Date().toISOString(), results }, null, 2));
      assert.deepEqual(errors, []);
      assert.deepEqual(mathOverflow, []);
      assert.deepEqual(svgOverflow, []);
      assert.ok(dimensions.scroll <= dimensions.width + 1);
      await page.close();
    }
    console.log(JSON.stringify(results));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

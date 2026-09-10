const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const assert = require('node:assert/strict');
const directory = 'scratch/randomized-linear-algebra-review';
fs.mkdirSync(directory, { recursive: true });
const number = value => value === null ? 'undefined' : String(Number(value.toFixed(4)));
const pair = vector => `[${vector.map(number).join(', ')}]`;
async function capture(page, locator, name) {
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = 'hidden'));
  await locator.screenshot({ path: `${directory}/${name}.png` });
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = ''));
}
async function select(lab, label, value) {
  await lab.getByLabel(label).selectOption({ value: String(value) });
}
(async () => {
  const models = await import(pathToFileURL(path.resolve('src/learn/data/randomized-linear-algebra-models.js')));
  const { randomizedLinearAlgebraExamples: examples } = await import(pathToFileURL(path.resolve('src/learn/data/randomized-linear-algebra-examples.js')));
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  try {
    for (const width of [1440, 390]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/randomized-linear-algebra?module=math-foundations');
      const lesson = page.locator('.randomized-linear-algebra-lesson');
      await lesson.waitFor();
      assert.equal(await lesson.locator('.rla-lab').count(), 4);
      assert.equal(await lesson.locator('.python-example').count(), Object.keys(examples).length);
      for (const example of Object.values(examples)) {
        const block = lesson.locator('.python-example').filter({ has: page.locator('h3', { hasText: example.title }) });
        assert.equal(await block.count(), 1);
        assert.ok((await block.textContent()).includes(example.code), example.title);
        assert.ok((await block.textContent()).includes(example.expected), example.title + ' stdout');
      }
      const anchors = await lesson.locator('.lesson-intro a[href^="#"]').evaluateAll(nodes => nodes.map(node => node.hash.slice(1)));
      assert.equal(anchors.length, 8);
      for (const anchor of anchors) assert.equal(await page.locator(`[id="${anchor}"]`).count(), 1, anchor);
      await capture(page, lesson.locator('.lesson-intro'), `intro-${width}`);

      const probe = lesson.getByRole('region', { name: 'Weighted column probe investigation' });
      const weightsToCheck = [[1, 1, 0], [1, 1, -1], [0, 0, 0], [2, 2, 2], [-2, -2, -2], [0, 1, 0], [1, -2, 1], [-1, 2, 0]];
      for (const weights of weightsToCheck) {
        for (let index = 0; index < 3; index += 1) await select(probe, `Column ${index + 1} weight`, weights[index]);
        const state = models.weightedProbeState(weights);
        const captions = await probe.locator('figcaption').allTextContents();
        assert.ok(captions[0].includes(pair(state.output)));
        assert.ok(captions[1].includes(number(state.residualSquared)));
        const points = probe.locator('.rla-plane').nth(1).locator('circle');
        for (let index = 0; index < 3; index += 1) {
          assert.ok(Math.abs(Number(await points.nth(index).getAttribute('cx')) - (160 + 118 * state.projected[index][0] / 4)) < 1e-9);
          assert.ok(Math.abs(Number(await points.nth(index).getAttribute('cy')) - (160 - 118 * state.projected[index][1] / 4)) < 1e-9);
        }
        assert.match(await probe.locator('.rla-feedback').innerText(), state.direction ? /observed unit direction/ : /cancelled to zero/);
        if (weights[0] === 1 && weights[2] === -1) await capture(page, probe, `probe-cancellation-${width}`);
      }
      await probe.getByRole('button', { name: 'Reset probe' }).focus();
      await page.keyboard.press('Enter');
      assert.deepEqual(await probe.locator('select').evaluateAll(nodes => nodes.map(node => Number(node.value))), [1, 1, 0]);
      await capture(page, probe, `probe-default-${width}`);

      const spectral = lesson.getByRole('region', { name: 'Spectral sketch budget investigation' });
      let spectralCases = 0;
      for (const preset of Object.keys(models.sketchSpectra)) {
        await select(spectral, 'Spectrum', preset);
        for (const rank of [1, 4]) {
          await select(spectral, 'Target rank k', rank);
          for (const extra of [0, 6 - rank]) {
            await select(spectral, 'Extra probes p', extra);
            for (const iterations of [0, 3]) {
              await select(spectral, 'Subspace iterations q', iterations);
              await select(spectral, 'Probe seed', 42);
              const state = models.spectralSketchState(preset, rank, extra, iterations, 42);
              assert.deepEqual(await spectral.locator('.rla-readings dd').allTextContents(), [
                String(state.width), String(state.observedRank), number(state.rangeErrorSquared), number(state.truncationErrorSquared),
                number(state.errorSquared), number(state.floorSquared), state.relativeError === null ? 'Undefined: the input norm is zero' : number(state.relativeError), String(state.passes),
              ]);
              const bars = await spectral.locator('.rla-spectrum-row .rla-bar-track > div').evaluateAll(nodes => nodes.map(node => parseFloat(node.style.width)));
              state.spectrum.forEach((value, index) => assert.ok(Math.abs(bars[index] - 100 * value / 12) < 1e-4));
              const budget = await spectral.locator('.rla-error-track > div').evaluateAll(nodes => nodes.map(node => parseFloat(node.style.width)));
              const ratio = state.totalSquared ? 100 / state.totalSquared : 0;
              assert.ok(Math.abs(budget[0] - ratio * state.rangeErrorSquared) < 1e-4);
              assert.ok(Math.abs(budget[1] - ratio * state.truncationErrorSquared) < 1e-4);
              spectralCases += 1;
            }
          }
        }
        await capture(page, spectral, `spectrum-${preset}-${width}`);
      }
      await spectral.getByRole('button', { name: 'Reset spectrum' }).click();
      assert.deepEqual(await spectral.locator('select').evaluateAll(nodes => nodes.map(node => node.value)), ['fast', '2', '1', '0', '7']);
      await spectral.getByText('Inspect actual matrices and numerical boundaries', { exact: true }).click();
      assert.equal(await spectral.locator('details .rla-matrix').count(), 3);
      await spectral.getByText('Inspect actual matrices and numerical boundaries', { exact: true }).click();
      await capture(page, spectral, `spectrum-default-${width}`);

      const rows = lesson.getByRole('region', { name: 'Observation sketch investigation' });
      let rowCases = 0;
      for (const indices of [[0, 1, 2, 3], [0, 7], [], [7], [4, 6], [0, 1, 2, 3, 4, 5, 6, 7], [1, 3, 5, 7]]) {
        const checkboxes = rows.getByRole('checkbox');
        for (let index = 0; index < 8; index += 1) await checkboxes.nth(index).setChecked(indices.includes(index));
        const state = models.rowSketchState(indices);
        assert.deepEqual(await rows.locator('.rla-readings dd').allTextContents(), [
          state.sketchFit ? `y = ${number(state.sketchFit.intercept)} + ${number(state.sketchFit.slope)}x` : 'No unique intercept and slope',
          state.sketchResidualSquared === null ? 'Not reported without a unique fit' : number(state.sketchResidualSquared),
          state.originalResidualSquared === null ? 'Not reported without a unique fit' : number(state.originalResidualSquared), number(state.fullResidualSquared),
        ]);
        const circles = await rows.locator('.rla-regression circle').evaluateAll(nodes => nodes.map(node => ({ x: Number(node.getAttribute('cx')), fill: node.getAttribute('fill') })));
        circles.forEach((circle, index) => {
          assert.ok(Math.abs(circle.x - (46 + state.observations[index].x / 20 * 290)) < 1e-9);
          assert.equal(circle.fill, indices.includes(index) ? '#e4bd6d' : '#11151b');
        });
        if (!indices.length || indices.length === 2) await capture(page, rows, `rows-${indices.join('-') || 'empty'}-${width}`);
        rowCases += 1;
      }
      await rows.getByRole('button', { name: 'First four / reset', exact: true }).click();
      await rows.getByRole('checkbox').nth(7).focus();
      await page.keyboard.press('Space');
      assert.equal(await rows.getByRole('checkbox').nth(7).isChecked(), true);
      await rows.getByRole('button', { name: 'First four / reset', exact: true }).click();
      await capture(page, rows, `rows-default-${width}`);

      const trace = lesson.getByRole('region', { name: 'Random sign trace investigation' });
      let traceCases = 0;
      for (const preset of Object.keys(models.traceMatrices)) {
        await select(trace, 'Trace matrix', preset);
        await select(trace, 'Sign seed', 7);
        for (let count = 0; count <= 32; count += 1) {
          if (count) await trace.getByRole('button', { name: 'Draw one probe', exact: true }).click();
          const state = models.traceProbeState(preset, 7, count);
          assert.ok((await trace.locator('.rla-feedback').innerText()).includes(count ? `mean ${number(state.mean)}, exact reference trace ${state.exactTrace}, absolute error ${number(state.absoluteError)}` : 'No probes yet'));
          const points = await trace.locator('.rla-history circle').evaluateAll(nodes => nodes.map(node => Number(node.getAttribute('cx'))));
          assert.equal(points.length, count);
          points.forEach((value, index) => assert.ok(Math.abs(value - (46 + index * 285 / 31)) < 1e-9));
          traceCases += 1;
        }
        assert.equal(await trace.getByRole('button', { name: 'Draw one probe', exact: true }).isDisabled(), true);
        await capture(page, trace, `trace-${preset}-${width}`);
      }
      await trace.getByRole('button', { name: 'Previous probe', exact: true }).click();
      assert.ok((await trace.locator('.rla-feedback').innerText()).startsWith('31 probes'));
      await trace.getByRole('button', { name: 'Reset estimator', exact: true }).click();
      assert.equal(await trace.getByRole('button', { name: 'Previous probe', exact: true }).isDisabled(), true);
      await capture(page, trace, `trace-default-${width}`);

      const sources = lesson.locator('.lesson-sources');
      const sourceLinks = await sources.locator('a').evaluateAll(nodes => nodes.map(node => ({ text: node.textContent, href: node.href })));
      assert.equal(sourceLinks.length, 6);
      assert.ok(sourceLinks.every(link => /^https:\/\//.test(link.href) && link.text.trim()));
      assert.equal(await lesson.locator('.katex-error').count(), 0);
      const mathOverflow = await lesson.locator('.katex-display').evaluateAll(nodes => nodes.filter(node => node.scrollWidth > node.clientWidth + 2).map(node => ({ text: node.textContent, width: node.clientWidth, scroll: node.scrollWidth })));
      assert.deepEqual(mathOverflow, [], 'A displayed formula requires horizontal scrolling');
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1));
      const figures = lesson.locator('.rla-inline-figure');
      for (let index = 0; index < await figures.count(); index += 1) await capture(page, figures.nth(index), `inline-${index + 1}-${width}`);
      for (let index = 0; index < anchors.length; index += 1) {
        await page.locator(`[id="${anchors[index]}"]`).scrollIntoViewIfNeeded();
        await page.evaluate(() => window.scrollBy(0, -95));
        await page.screenshot({ path: `${directory}/reading-${index + 1}-${width}.png` });
      }
      await capture(page, sources, `sources-${width}`);
      assert.deepEqual(errors, []);
      results.push({ width, probeCases: weightsToCheck.length, spectralCases, rowCases, traceCases, programs: Object.keys(examples).length, anchors: anchors.length, sourceLinks: sourceLinks.length, mathOverflow, errors });
      await page.close();
    }
    fs.writeFileSync(`${directory}/browser-results.json`, JSON.stringify({ at: new Date().toISOString(), results }, null, 2));
    console.log(JSON.stringify(results));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

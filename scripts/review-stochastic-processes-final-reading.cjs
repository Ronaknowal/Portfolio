const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const directory = 'scratch/stochastic-processes-browser';

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  try {
    for (const width of process.env.REVIEW_WIDTH ? [Number(process.env.REVIEW_WIDTH)] : [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 } });
      await page.routeWebSocket('**', socket => socket.close());
      const errors = [], failedRequests = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('requestfailed', request => failedRequests.push(request.url()));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/stochastic-processes-markov-chains-brownian-motion-poisson?module=math-foundations', { waitUntil: 'networkidle' });
      const lesson = page.locator('.stochastic-processes-lesson');
      await lesson.waitFor();
      await page.evaluate(() => document.fonts.ready);
      await lesson.locator('details').evaluateAll(elements => elements.forEach(element => { element.open = true; }));
      const geometry = await lesson.evaluate(element => {
        function clipped(node) {
          for (let parent = node.parentElement; parent; parent = parent.parentElement) {
            if (['auto', 'hidden', 'scroll', 'clip'].includes(getComputedStyle(parent).overflowX)) return true;
          }
          return false;
        }
        return {
          documentWidth: document.documentElement.scrollWidth,
          fonts: [...document.fonts].some(font => font.family === 'Space Grotesk' && font.status === 'loaded'),
          equations: [...element.querySelectorAll('.katex-display')].map(node => ({
            width: node.getBoundingClientRect().width, content: node.scrollWidth,
            tex: node.querySelector('annotation')?.textContent,
          })),
          unclippedOutside: [...element.querySelectorAll('*')].filter(node =>
            node.getBoundingClientRect().right > innerWidth+1 && !clipped(node) &&
            getComputedStyle(node).position !== 'absolute').map(node => ({
              tag: node.tagName, className: String(node.className), text: node.textContent.slice(0, 100),
              right: node.getBoundingClientRect().right,
            })),
          svgOverflow: [...element.querySelectorAll('.process-figure')].flatMap(svg =>
            [...svg.querySelectorAll('text')].filter(node => {
              const box = node.getBBox();
              return box.x < -0.5 || box.x+box.width > svg.viewBox.baseVal.width+0.5;
            }).map(node => node.textContent)),
          mathErrors: [...element.querySelectorAll('.katex-error')].map(node => node.textContent),
          wordCount: element.innerText.split(/\s+/).length,
        };
      });
      fs.writeFileSync(directory + '/final-geometry-' + width + '.json', JSON.stringify(geometry, null, 2));
      const targets = [
        ['intro', lesson.locator('.lesson-intro')],
        ['markov-definition', lesson.locator('h2').nth(1)],
        ['poisson-splitting', lesson.getByRole('heading', { name: 'Independent marking creates separate Poisson streams' })],
        ['brownian-definition', lesson.locator('h2').nth(7)],
        ['bridges', lesson.locator('h2').nth(8)],
        ['variation', lesson.getByRole('heading', { name: 'Rough paths accumulate squared movement' })],
        ['fair-process', lesson.getByRole('heading', { name: 'A fair process depends on the information available' })],
        ['practice-end', lesson.locator('.lesson-check').last()],
        ['references', lesson.locator('.lesson-sources')],
      ];
      for (const [name, target] of targets) {
        await target.evaluate(node => window.scrollTo(0, window.scrollY + node.getBoundingClientRect().top - 175));
        await page.screenshot({ path: directory + '/final-reading-' + name + '-' + width + '.png' });
      }
      for (let index = 0; index < await lesson.locator('.process-inline').count(); index += 1) {
        const target = lesson.locator('.process-inline').nth(index);
        await target.scrollIntoViewIfNeeded();
        await page.addStyleTag({ content: '.learn-nav { visibility:hidden !important; }' });
        await target.screenshot({ path: directory + '/final-inline-' + index + '-' + width + '.png' });
        await page.addStyleTag({ content: '.learn-nav { visibility:visible !important; }' });
      }
      const boundary = await page.evaluate(async () => {
        const m = await import('/src/learn/data/stochastic-processes-models.js');
        const mean = m.arrivalState({ rates: [6, 0.01], interval: [2, 2 + 2 ** -51] }).intervalMean;
        const rejected = [];
        for (const run of [
          () => m.parseProcessNumber('1e-9999', 0, 1, 'p'),
          () => m.integratedRate(Number.MIN_VALUE, [Number.MIN_VALUE, 1], 1),
          () => m.bridgeState({ fraction: Number.MIN_VALUE }),
        ]) {
          try { run(); rejected.push(false); } catch (error) { rejected.push(error instanceof RangeError); }
        }
        return {
          tinyIntervalMean: mean, expected: 0.01 * 2 ** -51, rejected,
          impossible: m.splitCountLaw(0, 1, 2, 0.4),
          possibleZero: m.splitCountLaw(0, 0, 0, 0.4),
          meanThree: m.splitCountLaw(3, 1, 2, 0.4),
        };
      });
      const arrival = page.getByRole('region', { name: 'Arrival clocks and event counts', exact: true });
      for (const value of ['0', '1', '0.5']) {
        await arrival.getByLabel(/^Independent probability of A/).selectOption(value);
        assert.ok((await arrival.innerText()).includes('conditional counts are deterministic'));
      }
      const brownian = page.getByRole('region', { name: 'Brownian paths and coupled refinement', exact: true });
      await brownian.getByLabel(/^Number of grid intervals/).selectOption({ value: '8' });
      await brownian.getByLabel(/^Diffusion σ/).selectOption('2');
      await brownian.getByLabel(/^Drift μ/).selectOption('0.5');
      await brownian.getByLabel(/^Brownian horizon/).selectOption('2');
      await brownian.getByLabel(/^Highlighted Brownian path/).selectOption('3');
      assert.ok(await brownian.locator('.process-compact-table').evaluate(node => node.scrollWidth <= node.clientWidth+1));
      await brownian.locator('details').evaluateAll(nodes => nodes.forEach(node => { node.open = false; }));
      await arrival.locator('details').evaluateAll(nodes => nodes.forEach(node => { node.open = false; }));
      await page.addStyleTag({ content: '.learn-nav { visibility:hidden !important; }' });
      await brownian.screenshot({ path: directory + '/final-brownian-summary-' + width + '.png' });
      await brownian.locator('.process-compact-table').screenshot({ path: directory + '/final-brownian-comparison-' + width + '.png' });
      await arrival.screenshot({ path: directory + '/final-arrival-conditioning-' + width + '.png' });
      await page.addStyleTag({ content: '.learn-nav { visibility:visible !important; }' });
      assert.ok(await lesson.locator('.process-controls button').evaluateAll(nodes =>
        nodes.every(node => node.scrollWidth <= node.clientWidth+1)));
      assert.ok((await lesson.innerText()).includes('For μ=3,r=.4,m=1,n=2'));
      assert.ok((await lesson.innerText()).includes('conditioning on a positive total is undefined'));
      results.push({ width, geometry, boundary, errors, failedRequests });
      fs.writeFileSync(directory + '/final-reading-results.json', JSON.stringify({ checkedAt: new Date().toISOString(), results }, null, 2));
      assert.equal(geometry.documentWidth, width);
      assert.ok(geometry.fonts);
      assert.deepEqual(geometry.svgOverflow, []);
      assert.deepEqual(geometry.mathErrors, []);
      assert.ok(geometry.equations.every(value => value.content <= value.width+1));
      assert.deepEqual(errors, []);
      assert.deepEqual(failedRequests, []);
      assert.equal(boundary.tinyIntervalMean, boundary.expected);
      assert.ok(boundary.rejected.every(Boolean));
      assert.equal(boundary.impossible.conditional, null);
      assert.equal(boundary.impossible.conditioningEventPossible, false);
      assert.equal(boundary.possibleZero.conditional, 1);
      assert.ok(Math.abs(boundary.meanThree.joint - 0.0967860609071275) < 1e-14);
      await page.close();
    }
  } finally { await browser.close(); }
  console.log(JSON.stringify({ widths: results.map(row => row.width), passed: true }, null, 2));
})().catch(error => { console.error(error); process.exitCode = 1; });

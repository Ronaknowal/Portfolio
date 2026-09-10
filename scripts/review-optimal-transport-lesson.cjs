const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const output = path.resolve('scratch/optimal-transport-review/browser');
fs.mkdirSync(output, { recursive: true });
const format = value => value !== 0 && Math.abs(value) < .0001 ? value.toExponential(2) : Number(value.toFixed(4)).toString();
(async () => {
  const { optimalTransportExamples } = await import('../src/learn/data/optimal-transport-examples.js');
  const model = await import('../src/learn/data/optimal-transport-models.js');
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      const errors = [], warnings = [], failedRequests = [];
      page.on ('pageerror', error => errors.push(error.message));
      page.on ('console', event => { if (event.type() === 'error') errors.push(event.text()); if (event.type() === 'warning') warnings.push(event.text()); });
      page.on ('requestfailed', request => failedRequests.push({ url: request.url(), failure: request.failure() }));
      await page.routeWebSocket('**', socket => socket.close());
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/optimal-transport-wasserstein-distance-sinkhorn?module=mathematical-statistical-foundations');
      const lesson = page.locator('.optimal-transport-lesson');
      await lesson.locator('h2').last().waitFor();
      await page.evaluate(() => document.fonts.ready);
      const lab = name => lesson.locator('[data-transport-lab="' + name + '"]');
      const metric = (region, label) => region.locator('.transport-metrics > div').filter({ has: page.locator('dt', { hasText: label }) }).locator('dd');
      const range = (region, label, value) => region.getByRole('slider', { name: label, exact: true }).fill(String(value));
      async function capture(region, name) {
        await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = 'hidden'));
        await region.screenshot({ path: path.join(output, name + '-' + width + '.png') });
        await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = ''));
      }
      let states = 0;
      const plan = lab('mass-ledger');
      await range(plan, 'Source mass at 0', .2); await range(plan, 'Target mass at 0', .7);
      assert.equal(await metric(plan, 'Minimum cost').innerText(), '1.3');
      await plan.getByRole('button', { name: 'Inspect route 2 to 0' }).click();
      assert((await plan.locator('.transport-observation').innerText()).includes('0.5 mass × 2 cost = 1'));
      await capture(plan, 'mass-split'); states++;
      await plan.getByRole('combobox').selectOption('indifferent');
      assert((await plan.innerText()).includes('every feasible plan has the same cost')); states++;
      await plan.getByRole('combobox').selectOption('changed');
      await plan.getByRole('button', { name: 'Use a minimum-cost plan' }).click();
      assert.equal(await plan.getByRole('slider', { name: 'Position in feasible interval' }).inputValue(), '0'); states++;
      await range(plan, 'Source mass at 0', 0);
      assert((await plan.innerText()).includes('only one plan')); states++;
      await plan.getByRole('button', { name: 'Reset investigation' }).click();
      const dual = lab('dual-certificate');
      await range(dual, 'Source price f1', 3);
      assert.equal(await metric(dual, 'Primal − dual gap').innerText(), '.5'.replace(/^\./, '0.'));
      await capture(dual, 'dual-loose');
      await dual.getByRole('button', { name: 'Use a tight certificate' }).click();
      assert.equal(await metric(dual, 'Primal − dual gap').innerText(), '0'); states += 2;
      const cdf = lab('cumulative-crossings');
      await cdf.getByRole('combobox').selectOption('far');
      await range(cdf, 'Distance between bins', 2);
      assert.equal(await metric(cdf, 'Total area / W₁').innerText(), '2.4');
      await cdf.locator('.transport-buttons button').last().click();
      await capture(cdf, 'cdf-distance'); states++;
      await cdf.getByRole('combobox').selectOption('identical');
      assert.equal(await metric(cdf, 'Total area / W₁').innerText(), '0'); states++;
      await cdf.getByRole('button', { name: 'Reset investigation' }).click();
      const scaling = lab('alternating-scaling');
      const scalingReference = model.sinkhornScaling({ source: [.6, .4], target: [.3, .7], epsilon: .5, traceSteps: 40 });
      for (let step = 0; step <= 3; step++) {
        assert.equal(await metric(scaling, 'Current marginal residual').innerText(), format(scalingReference.trace[step].residual));
        await capture(scaling, 'scaling-' + step); states++;
        if (step < 3) await scaling.getByRole('button', { name: 'Next correction' }).click();
      }
      await scaling.getByRole('button', { name: /Show\s?40 corrections/ }).click();
      assert(await scaling.getByRole('button', { name: 'Next correction' }).isDisabled()); states++;
      await scaling.getByRole('button', { name: 'Reset investigation' }).click();
      assert(await scaling.getByRole('button', { name: 'Previous correction' }).isDisabled());
      const stable = lab('log-domain');
      assert.equal(await metric(stable, 'Kernel entries rounded to zero').innerText(), '4 of 4');
      await capture(stable, 'stable-underflow'); states++;
      await range(stable, 'Cost added to every route', 0);
      assert.equal(await metric(stable, 'Kernel entries rounded to zero').innerText(), '0 of 4'); states++;
      await stable.getByRole('button', { name: 'Reset investigation' }).click();
      const bias = lab('objective-bias');
      assert.equal(await metric(bias, 'Debiased Sε (closed form)').innerText(), '0.5388');
      await capture(bias, 'bias-contract'); states++;
      await bias.getByRole('combobox').selectOption('same');
      assert.equal(await metric(bias, 'Debiased Sε (closed form)').innerText(), '0'); states++;
      await bias.getByRole('combobox').selectOption('shifted');
      const start = performance.now();
      await range(bias, 'Comparison entropy epsilon', .1);
      assert((await bias.locator('.transport-observation').innerText()).includes('not certified converged'));
      const changedControlMilliseconds = performance.now() - start;
      assert.equal(await metric(bias, 'Debiased Sε (closed form)').innerText(), '1');
      await capture(bias, 'bias-capped'); states++;
      await bias.getByRole('button', { name: 'Reset investigation' }).click();
      let keyboardControls = 0;
      for (const region of await lesson.locator('[data-transport-lab]').all()) {
        for (const control of await region.locator('input:enabled,select:enabled,button:enabled').all()) {
          await control.focus(); await page.keyboard.press('Tab'); await page.keyboard.press('Shift+Tab');
          assert(await control.evaluate(node => node === document.activeElement));
          const style = await control.evaluate(node => ({ outline: getComputedStyle(node).outlineStyle, width: getComputedStyle(node).outlineWidth, height: node.getBoundingClientRect().height }));
          assert.notEqual(style.outline, 'none'); assert(parseFloat(style.width) >= 2); assert(style.height >= 43);
          keyboardControls++;
        }
      }
      await plan.getByRole('slider', { name: 'Source mass at 0' }).focus();
      await page.keyboard.press('Home'); await page.keyboard.press('ArrowRight');
      assert.equal(await plan.getByRole('slider', { name: 'Source mass at 0' }).inputValue(), '.05'.replace(/^\./, '0.'));
      await plan.getByRole('button', { name: 'Reset investigation' }).click();
      assert.equal(await lesson.locator('h2').count(), 10);
      for (let index = 0; index < 10; index++) {
        const heading = lesson.locator('h2').nth(index);
        assert.equal((await lesson.locator('.lesson-intro nav a').nth(index).getAttribute('href')).slice(1), await heading.getAttribute('id'));
        await heading.evaluate(node => window.scrollTo(0, node.getBoundingClientRect().top + scrollY - 100));
        await page.screenshot({ path: path.join(output, 'reading-' + (index + 1) + '-' + width + '.png') });
      }
      for (let index = 0; index < 3; index++) await capture(lesson.locator('.transport-inline').nth(index), 'inline-' + index);
      for (const example of Object.values(optimalTransportExamples)) {
        const block = lesson.locator('.python-example').filter({ has: page.locator('h3', { hasText: example.title }) });
        assert.equal(await block.count(), 1); assert((await block.innerText()).includes(example.code)); assert((await block.innerText()).includes(example.expected));
        assert((await block.evaluate(node => node.previousElementSibling.textContent)).includes(example.question));
      }
      const checks = lesson.locator('.lesson-check');
      assert.equal(await checks.count(), 11);
      for (const check of await checks.all()) {
        assert((await check.locator('p').first().innerText()).length > 80);
        for (const summary of await check.locator('summary').all()) { await summary.focus(); await page.keyboard.press('Enter'); }
        assert((await check.locator('details').last().innerText()).length > 150);
      }
      await capture(checks.last(), 'practice-transfer');
      await lesson.locator('details').evaluateAll(nodes => nodes.forEach(node => node.open = true));
      const equations = await lesson.locator('.katex-display').evaluateAll(nodes => nodes.map((node, index) => ({ index, width: node.clientWidth, scroll: node.scrollWidth, text: node.textContent.slice(0, 180) })));
      const overflowMath = equations.filter(row => row.scroll > row.width + 2);
      for (let index = 0; index < equations.length; index++) await capture(lesson.locator('.katex-display').nth(index), 'equation-' + index);
      const figures = await lesson.locator('svg').evaluateAll(nodes => nodes.map((svg, index) => {
        const b = svg.getBoundingClientRect();
        return { index, minimumTextPx: Math.min(...[...svg.querySelectorAll('text')].map(node => parseFloat(getComputedStyle(node).fontSize) * b.width / svg.viewBox.baseVal.width)), clipped: [...svg.querySelectorAll('text')].filter(node => { const t = node.getBoundingClientRect(); return t.left < b.left - 1 || t.right > b.right + 1 || t.top < b.top - 1 || t.bottom > b.bottom + 1; }).map(node => node.textContent) };
      }));
      let keyboardTables = 0;
      for (const table of await lesson.locator('.lesson-table-wrap,.transport-table').all()) if (await table.evaluate(node => node.scrollWidth > node.clientWidth + 1)) {
        await table.focus(); for (let index = 0; index < 8; index++) await page.keyboard.press('ArrowRight');
        await page.waitForFunction(node => node.scrollLeft > 0, await table.elementHandle()); keyboardTables++;
      }
      await capture(lesson.locator('.lesson-sources'), 'sources');
      const pageOverflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1);
      const applicationErrors = errors.filter(text => !text.includes('WebSocket connection') && !text.startsWith('[vite] failed to connect to websocket.'));
      const result = { width, states, keyboardControls, keyboardTables, changedControlMilliseconds, equations, overflowMath, figures, pageOverflow, errors, applicationErrors, warnings, failedRequests };
      results.push(result); fs.writeFileSync(path.join(output, 'in-progress.json'), JSON.stringify(results, null, 2));
      assert.equal(await lesson.locator('.katex-error').count(), 0); assert.equal(await lesson.locator('p p,p div,p section').count(), 0);
      assert.deepEqual(applicationErrors, []); assert.equal(pageOverflow, false);
      assert.deepEqual(failedRequests, []);
      if (!process.env.TRANSPORT_AUDIT) { assert.deepEqual(overflowMath, []); assert.deepEqual(figures.flatMap(figure => figure.clipped), []); }
      await page.close();
    }
  } finally { await browser.close(); }
  fs.writeFileSync(path.join(output, 'results.json'), JSON.stringify({ at: new Date().toISOString(), results }, null, 2));
  console.log(results.map(({ width, states, keyboardControls, overflowMath, pageOverflow }) => ({ width, states, keyboardControls, overflowMath, pageOverflow })));
})().catch(error => { console.error(error); process.exitCode = 1; });

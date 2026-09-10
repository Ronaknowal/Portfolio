const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const output = path.resolve('scratch/rate-distortion-review/browser');
fs.mkdirSync(output, { recursive: true });
const format = (value, digits = 6) => value !== 0 && Math.abs(value) < .0001 ? value.toExponential(2) : Number(value.toFixed(digits)).toString();
(async () => {
  const { rateDistortionExamples } = await import('../src/learn/data/rate-distortion-examples.js');
  const model = await import('../src/learn/data/rate-distortion-models.js');
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      const errors = [], warnings = [], failedRequests = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', event => { if (event.type() === 'error') errors.push(event.text()); if (event.type() === 'warning') warnings.push(event.text()); });
      page.on('requestfailed', request => failedRequests.push({ url: request.url(), failure: request.failure() }));
      await page.routeWebSocket('**', socket => socket.close());
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/rate-distortion-theory?module=math-foundations');
      const lesson = page.locator('.rate-distortion-lesson');
      await lesson.locator('h2').last().waitFor();
      await page.evaluate(() => document.fonts.ready);
      const lab = name => lesson.locator('[data-rate-distortion-lab="' + name + '"]');
      const metric = (region, label) => region.locator('.rate-distortion-metrics > div').filter({ has: page.locator('dt', { hasText: label }) }).locator('dd');
      const range = (region, label, value) => region.getByRole('slider', { name: label, exact: true }).fill(String(value));
      async function capture(region, name) {
        await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = 'hidden'));
        await region.screenshot({ path: path.join(output, name + '-' + width + '.png') });
        await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = ''));
      }
      let states = 0;
      const code = lab('codebook');
      assert.equal(await metric(code, 'Average wrong-bit fraction').innerText(), '0.25');
      await code.getByRole('button', { name: 'Encode block 101' }).click();
      assert((await code.locator('.rate-distortion-observation').innerText()).includes('1 of 3'));
      await capture(code, 'codebook-majority'); states++;
      await code.getByRole('combobox', { name: 'Shared codebook' }).selectOption('constant');
      await range(code, 'Probability of source bit 1', .2);
      assert.equal(await metric(code, 'Average wrong-bit fraction').innerText(), '0.2');
      assert.equal(await metric(code, 'Worst supported block fraction').innerText(), '1');
      assert.equal(await metric(code, 'Fixed bits / original bit').innerText(), '0');
      await capture(code, 'codebook-constant'); states++;
      await range(code, 'Probability of source bit 1', 0);
      assert((await code.locator('.rate-distortion-observation').innerText()).includes('zero probability'));
      assert.equal(await metric(code, 'Worst supported block fraction').innerText(), '0'); states++;
      await code.getByRole('combobox').selectOption('parity'); await range(code, 'Probability of source bit 1', .5);
      assert.equal(await metric(code, 'Average wrong-bit fraction').innerText(), '0.166667'); states++;
      await code.getByRole('combobox').selectOption('lossless');
      assert.equal(await metric(code, 'Fixed bits / original bit').innerText(), '1');
      assert.equal(await metric(code, 'Average wrong-bit fraction').innerText(), '0'); states++;
      await code.getByRole('button', { name: 'Reset investigation' }).click();
      const binary = lab('binary-frontier');
      await binary.getByRole('button', { name: 'Inspect biased p=.2, D=.1' }).click();
      assert.equal(await metric(binary, 'Required information').innerText(), '0.252933 bits');
      assert((await binary.locator('.rate-distortion-observation').innerText()).includes('0.4375'));
      await capture(binary, 'frontier-biased'); states++;
      await range(binary, 'Allowed error fraction D', .8);
      assert.equal(await metric(binary, 'Required information').innerText(), '0 bits');
      assert.equal(await metric(binary, 'Actual error in shown law').innerText(), '0.2');
      await capture(binary, 'frontier-zero-rate'); states++;
      await range(binary, 'Binary source probability p', 1);
      assert((await binary.locator('.rate-distortion-observation').innerText()).includes('undefined: input0')); states++;
      await range(binary, 'Binary source probability p', .5);
      await range(binary, 'Allowed error fraction D', 0);
      await range(binary, 'Distortion penalty lambda', 0);
      assert.equal(await metric(binary, 'Required information').innerText(), '1 bits'); states++;
      await binary.getByRole('button', { name: 'Reset investigation' }).click();
      const optimizer = lab('finite-optimizer');
      const reference = model.finiteRateDistortion({ ...model.RATE_DISTORTION_SCENARIOS.levels, lambda: .5 });
      for (let index = 0; index <= 2; index++) {
        assert.equal(await metric(optimizer, 'Expected distortion D').innerText(), format(reference.trace[index].distortion));
        assert.equal(await metric(optimizer, 'Objective gap bound').innerText(), format(reference.trace[index].gap));
        await capture(optimizer, 'optimizer-' + index); states++;
        if (index < 2) await optimizer.getByRole('button', { name: 'Next update' }).click();
      }
      await optimizer.getByRole('button', { name: 'Previous update' }).click();
      assert.equal(await metric(optimizer, 'Expected distortion D').innerText(), format(reference.trace[1].distortion));
      await optimizer.getByRole('button', { name: 'Inspect bounded run' }).click();
      assert((await optimizer.innerText()).includes('stops at 120 updates'));
      assert(await optimizer.getByRole('button', { name: 'Next update' }).isDisabled()); states++;
      await optimizer.getByRole('combobox', { name: 'Initial output support' }).selectOption('missing');
      await optimizer.getByRole('button', { name: 'Inspect bounded run' }).click();
      assert.equal(await metric(optimizer, 'Expected distortion D').innerText(), '1.1');
      assert((await optimizer.innerText()).includes('zero starting probability remains zero'));
      await capture(optimizer, 'optimizer-missing-support'); states++;
      await optimizer.getByRole('combobox', { name: 'Initial output support' }).selectOption('positive');
      await optimizer.getByRole('combobox', { name: 'Source and distortion rule' }).selectOption('alarm');
      const interactionStart = performance.now();
      await range(optimizer, 'Optimizer distortion weight', .1);
      await optimizer.getByRole('button', { name: 'Inspect bounded run' }).click();
      const changedControlMilliseconds = performance.now() - interactionStart;
      await optimizer.getByRole('button', { name: 'Inspect original 3 output 0', exact: true }).click();
      const alarmReference = model.finiteRateDistortion({ ...model.RATE_DISTORTION_SCENARIOS.alarm, lambda: .1 });
      assert.equal(await metric(optimizer, 'Expected distortion D').innerText(), format(alarmReference.distortion));
      await capture(optimizer, 'optimizer-weighted'); states++;
      await optimizer.getByRole('combobox', { name: 'Source and distortion rule' }).selectOption('binary');
      await range(optimizer, 'Optimizer distortion weight', 2);
      assert.equal(await metric(optimizer, 'Expected distortion D').innerText(), '0.2');
      assert(await optimizer.getByRole('button', { name: 'Next update' }).isDisabled()); states++;
      await optimizer.getByRole('button', { name: 'Reset investigation' }).click();
      const allocation = lab('gaussian-allocation');
      assert.equal(await metric(allocation, 'Bits / vector').innerText(), '1.084963');
      await capture(allocation, 'gaussian-unequal'); states++;
      await range(allocation, 'Total squared-error budget', 0);
      assert.equal(await metric(allocation, 'Bits / vector').innerText(), '∞'); states++;
      await range(allocation, 'Total squared-error budget', 11);
      assert.equal(await metric(allocation, 'Bits / vector').innerText(), '0');
      assert.equal(await metric(allocation, 'Total squared error used').innerText(), '10'); states++;
      await allocation.getByRole('combobox').selectOption('equal');
      await range(allocation, 'Total squared-error budget', 2.4);
      await capture(allocation, 'gaussian-equal'); states++;
      await allocation.getByRole('combobox').selectOption('small');
      await capture(allocation, 'gaussian-small'); states++;
      await allocation.getByRole('button', { name: 'Reset investigation' }).click();
      let keyboardControls = 0;
      for (const region of await lesson.locator('[data-rate-distortion-lab]').all()) {
        for (const control of await region.locator('input:enabled,select:enabled,button:enabled').all()) {
          await control.focus(); await page.keyboard.press('Tab'); await page.keyboard.press('Shift+Tab');
          assert(await control.evaluate(node => node === document.activeElement));
          const style = await control.evaluate(node => ({ outline: getComputedStyle(node).outlineStyle, width: getComputedStyle(node).outlineWidth, height: node.getBoundingClientRect().height }));
          assert.notEqual(style.outline, 'none'); assert(parseFloat(style.width) >= 2); assert(style.height >= 43);
          keyboardControls++;
        }
      }
      await code.getByRole('slider').focus(); await page.keyboard.press('Home'); await page.keyboard.press('ArrowRight');
      assert.equal(await code.getByRole('slider').inputValue(), '0.05');
      await code.getByRole('button', { name: 'Encode block 110' }).focus(); await page.keyboard.press('Enter');
      assert.equal(await code.getByRole('button', { name: 'Encode block 110' }).getAttribute('aria-pressed'), 'true');
      await code.getByRole('button', { name: 'Reset investigation' }).click();
      assert.equal(await lesson.locator('h2').count(), 9);
      for (let index = 0; index < 9; index++) {
        const heading = lesson.locator('h2').nth(index);
        assert.equal((await lesson.locator('.lesson-intro nav a').nth(index).getAttribute('href')).slice(1), await heading.getAttribute('id'));
        await heading.evaluate(node => window.scrollTo(0, node.getBoundingClientRect().top + scrollY - 100));
        await page.screenshot({ path: path.join(output, 'reading-' + (index + 1) + '-' + width + '.png') });
      }
      for (let index = 0; index < 3; index++) await capture(lesson.locator('.rate-distortion-inline').nth(index), 'inline-' + index);
      for (const example of Object.values(rateDistortionExamples)) {
        const block = lesson.locator('.python-example').filter({ has: page.locator('h3', { hasText: example.title }) });
        assert.equal(await block.count(), 1); assert((await block.innerText()).includes(example.code)); assert((await block.innerText()).includes(example.expected));
      }
      const checks = lesson.locator('.lesson-check');
      assert.equal(await checks.count(), 12);
      for (const check of await checks.all()) {
        for (const summary of await check.locator('summary').all()) { await summary.focus(); await page.keyboard.press('Enter'); }
        assert((await check.locator('details').last().innerText()).length > 150);
      }
      await capture(checks.last(), 'practice-audio');
      await lesson.locator('details').evaluateAll(nodes => nodes.forEach(node => node.open = true));
      const equations = await lesson.locator('.katex-display').evaluateAll(nodes => nodes.map((node, index) => ({ index, width: node.clientWidth, scroll: node.scrollWidth, text: node.textContent.slice(0, 150) })));
      const overflowMath = equations.filter(row => row.scroll > row.width + 2);
      for (let index = 0; index < equations.length; index++) await capture(lesson.locator('.katex-display').nth(index), 'equation-' + index);
      const figures = await lesson.locator('svg').evaluateAll(nodes => nodes.map((svg, index) => {
        const b = svg.getBoundingClientRect();
        return { index, minimumTextPx: Math.min(...[...svg.querySelectorAll('text')].map(node => parseFloat(getComputedStyle(node).fontSize) * b.width / svg.viewBox.baseVal.width)), clipped: [...svg.querySelectorAll('text')].filter(node => { const t = node.getBoundingClientRect(); return t.left < b.left - 1 || t.right > b.right + 1 || t.top < b.top - 1 || t.bottom > b.bottom + 1; }).map(node => node.textContent) };
      }));
      let keyboardTables = 0;
      for (const table of await lesson.locator('.lesson-table-wrap,.rate-distortion-matrix').all()) if (await table.evaluate(node => node.scrollWidth > node.clientWidth + 1)) {
        await table.focus(); for (let index = 0; index < 8; index++) await page.keyboard.press('ArrowRight');
        await page.waitForFunction(node => node.scrollLeft > 0, await table.elementHandle()); keyboardTables++;
      }
      await capture(lesson.locator('.lesson-sources'), 'sources');
      const pageOverflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1);
      const applicationErrors = errors.filter(text => !text.includes('WebSocket connection') && !text.startsWith('[vite] failed to connect to websocket.'));
      const result = { width, states, keyboardControls, keyboardTables, changedControlMilliseconds, equations, overflowMath, figures, pageOverflow, errors, applicationErrors, warnings, failedRequests };
      results.push(result); fs.writeFileSync(path.join(output, 'in-progress.json'), JSON.stringify(results, null, 2));
      assert.equal(await lesson.locator('.katex-error').count(), 0); assert.equal(await lesson.locator('p p,p div,p section').count(), 0);
      assert.deepEqual(applicationErrors, []); assert.equal(pageOverflow, false); assert.deepEqual(failedRequests, []);
      if (!process.env.RATE_DISTORTION_AUDIT) { assert.deepEqual(overflowMath, []); assert.deepEqual(figures.flatMap(figure => figure.clipped), []); assert(figures.every(figure => figure.minimumTextPx >= 14)); }
      await page.close();
    }
  } finally { await browser.close(); }
  fs.writeFileSync(path.join(output, 'results.json'), JSON.stringify({ at: new Date().toISOString(), browser: browser.version(), results }, null, 2));
  console.log(results.map(({ width, states, keyboardControls, overflowMath, figures, pageOverflow }) => ({ width, states, keyboardControls, overflowMath, figures, pageOverflow })));
})().catch(error => { console.error(error); process.exitCode = 1; });

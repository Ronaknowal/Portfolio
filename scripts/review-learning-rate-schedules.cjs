const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const directory = 'scratch/learning-rate-schedule-browser';
fs.mkdirSync(directory, { recursive: true });

async function setRange(page, locator, value) {
  const { minimum, step } = await locator.evaluate(node => ({ minimum: Number(node.min || 0), step: Number(node.step || 1) }));
  await locator.focus();
  await page.keyboard.press('Home');
  for (let index = 0; index < Math.round((value - minimum) / step); index += 1) await page.keyboard.press('ArrowRight');
  assert.equal(Number(await locator.inputValue()), value);
}

(async () => {
  const { learningRateScheduleExamples } = await import('../src/learn/data/learning-rate-schedule-examples.js');
  const { buildScheduleClockTrace, buildPlateauTrace, parseValidationMetrics } = await import('../src/learn/data/learning-rate-schedule-models.js');
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/learning-rate-schedules-cosine-warmup-onecyclelr?module=math-foundations', { waitUntil: 'networkidle' });
      const lesson = page.locator('.learning-rate-schedules-lesson');
      await lesson.waitFor();
      assert.equal(await lesson.locator('.schedule-lab').count(), 4);
      const links = lesson.locator('.lesson-intro nav a');
      const anchors = [];
      for (let index = 0; index < await links.count(); index += 1) {
        const link = links.nth(index);
        const href = await link.getAttribute('href');
        await link.scrollIntoViewIfNeeded(); await link.focus(); await page.keyboard.press('Enter');
        const box = await page.evaluate(hash => {
          const target = document.getElementById(decodeURIComponent(hash.slice(1)));
          if (!target) return null;
          const bounds = target.getBoundingClientRect();
          return { top: bounds.top, bottom: bounds.bottom };
        }, href);
        assert.ok(box && box.top >= 60 && box.bottom < 1000, JSON.stringify({ href, box }));
        anchors.push(href);
      }
      const examples = lesson.locator('.python-example');
      assert.equal(await examples.count(), Object.keys(learningRateScheduleExamples).length);
      let exampleIndex = 0;
      for (const example of Object.values(learningRateScheduleExamples)) {
        const block = examples.filter({ has: page.getByRole('heading', { name: example.title, exact: true }) });
        assert.equal(await block.count(), 1, example.title);
        exampleIndex += 1;
        await block.locator('details').evaluateAll(nodes => nodes.forEach(node => node.open = true));
        const displayed = await block.innerText();
        assert.ok(displayed.includes(example.code), example.title);
        assert.ok(displayed.includes(example.expected), example.title);
      }
      await lesson.locator('details').evaluateAll(nodes => nodes.forEach(node => node.open = false));
      const shape = lesson.locator('[data-lab="schedule-shape"]');
      for (const policy of ['constant', 'cosine', 'linear', 'exponential', 'step', 'one-cycle', 'restart']) {
        await shape.getByLabel('Rate policy', { exact: true }).selectOption(policy);
        await setRange(page, shape.getByLabel('Inspect update index', { exact: true }), 23);
        assert.ok((await shape.innerText()).includes('23 / 23'));
        await shape.screenshot({ style: ".learn-nav { visibility: hidden; }", path: `${directory}/shape-${policy}-${width}.png` });
      }
      await shape.getByRole('button', { name: 'Reset schedule', exact: true }).click();
      assert.equal(await shape.getByLabel('Rate policy', { exact: true }).inputValue(), 'cosine');
      await shape.getByLabel('Update budget', { exact: true }).selectOption('12');
      await setRange(page, shape.getByLabel('Warmup updates', { exact: true }), 0);
      const noise = lesson.locator('[data-lab="schedule-noise"]');
      await noise.screenshot({ style: ".learn-nav { visibility: hidden; }", path: `${directory}/noise-default-${width}.png` });
      await setRange(page, noise.getByLabel('Noise standard deviation σ', { exact: true }), 0);
      await setRange(page, noise.getByLabel('Inspect completed updates', { exact: true }), 24);
      await noise.getByLabel('Compared policy', { exact: true }).selectOption('one-cycle');
      await noise.screenshot({ style: ".learn-nav { visibility: hidden; }", path: `${directory}/noise-zero-${width}.png` });
      await noise.getByRole('button', { name: 'Reset noise comparison', exact: true }).click();
      const clock = lesson.locator('[data-lab="schedule-clock"]');
      let clockStates = 0;
      for (const accumulation of [1, 2, 3]) for (const policy of ['committed', 'microbatch', 'advance-first']) {
        await clock.getByLabel('Microbatches per attempt', { exact: true }).selectOption(String(accumulation));
        await clock.getByLabel('Schedule clock policy', { exact: true }).selectOption(policy);
        const expected = buildScheduleClockTrace({ accumulation, policy, skipSecond: true }).states;
        for (let index = 1; index < expected.length; index += 1) {
          await clock.getByRole('button', { name: 'Next microbatch', exact: true }).click();
          const readout = await clock.locator('.schedule-readout').innerText();
          assert.ok(readout.includes(expected[index].action));
          assert.ok(readout.includes(`${expected[index].attempt} / ${expected[index].committed}`));
          clockStates += 1;
        }
        assert.equal(await clock.getByRole('button', { name: 'Next microbatch', exact: true }).isDisabled(), true);
      }
      await clock.screenshot({ style: ".learn-nav { visibility: hidden; }", path: `${directory}/clock-fault-${width}.png` });
      await clock.getByRole('button', { name: 'Reset', exact: true }).click();
      await clock.getByRole('button', { name: 'Finish', exact: true }).focus(); await page.keyboard.press('Enter');
      await clock.screenshot({ style: ".learn-nav { visibility: hidden; }", path: `${directory}/clock-correct-${width}.png` });
      const plateau = lesson.locator('[data-lab="schedule-plateau"]');
      const expectedPlateau = buildPlateauTrace(parseValidationMetrics('1 .9 .9 .89 .88 .88 .9 .87 .87 .87 .87 .87'));
      for (let index = 0; index < expectedPlateau.length; index += 1) {
        if (index) await plateau.getByRole('button', { name: 'Next observation', exact: true }).click();
        const state = expectedPlateau[index];
        assert.ok((await plateau.locator('.schedule-readout').innerText()).includes(state.reduced ? 'rate halved' : state.triggered ? 'triggered, but floor prevented change' : 'no reduction'));
      }
      await plateau.screenshot({ style: ".learn-nav { visibility: hidden; }", path: `${directory}/plateau-final-${width}.png` });
      await plateau.getByLabel('Validation losses (apply to use edits)', { exact: true }).fill('NaN 1');
      await plateau.getByRole('button', { name: 'Apply validation losses', exact: true }).click();
      assert.equal(await plateau.getByRole('alert').count(), 1);
      await plateau.getByLabel('Validation losses (apply to use edits)', { exact: true }).fill('1 1 .9 .9');
      await plateau.getByRole('button', { name: 'Apply validation losses', exact: true }).click();
      assert.equal(await plateau.getByRole('alert').count(), 0);
      await setRange(page, plateau.getByLabel('Patience', { exact: true }), 0);
      await setRange(page, plateau.getByLabel('Absolute threshold δ', { exact: true }), 0);
      await setRange(page, plateau.getByLabel('Cooldown observations', { exact: true }), 0);
      await plateau.getByRole('button', { name: 'Finish', exact: true }).focus(); await page.keyboard.press('Enter');
      await plateau.screenshot({ style: ".learn-nav { visibility: hidden; }", path: `${directory}/plateau-ties-${width}.png` });
      await lesson.locator('details').evaluateAll(nodes => nodes.forEach(node => node.open = true));
      const maths = await lesson.locator('.katex-display').evaluateAll(nodes => nodes.map(node => ({ width: node.clientWidth, scroll: node.scrollWidth, text: node.textContent })));
      fs.writeFileSync(`${directory}/math-${width}.json`, JSON.stringify(maths, null, 2));
      assert.ok(maths.every(item => item.scroll <= item.width + 2), JSON.stringify(maths.filter(item => item.scroll > item.width + 2)));
      const svg = await lesson.locator('svg.schedule-plot').evaluateAll(nodes => nodes.map(node => ({ title: node.querySelector('title')?.textContent, description: node.querySelector('desc')?.textContent, invalid: /NaN|Infinity/.test(node.innerHTML) })));
      assert.ok(svg.every(item => item.title && item.description && !item.invalid));
      const figures = lesson.locator('.schedule-inline');
      for (let index = 0; index < await figures.count(); index += 1) await figures.nth(index).screenshot({ style: ".learn-nav { visibility: hidden; }", path: `${directory}/inline-${index + 1}-${width}.png` });
      for (let index = 0; index < maths.length; index += 1) await lesson.locator('.katex-display').nth(index).screenshot({ style: ".learn-nav { visibility: hidden; }", path: `${directory}/equation-${index + 1}-${width}.png` });
      const resourceLinks = await lesson.locator('.lesson-sources a').evaluateAll(nodes => nodes.map(node => ({ href: node.href, target: node.target, rel: node.rel })));
      assert.ok(resourceLinks.length >= 12);
      assert.ok(resourceLinks.every(link => link.href.startsWith('https://') && link.target === '_blank' && link.rel.includes('noreferrer')));
      await lesson.locator('details').evaluateAll(nodes => nodes.forEach(node => node.open = false));
      const headings = lesson.locator('h2');
      for (let index = 0; index < await headings.count(); index += 1) {
        await headings.nth(index).evaluate(node => window.scrollTo({ top: window.scrollY + node.getBoundingClientRect().top - 90, behavior: 'instant' }));
        await page.screenshot({ path: `${directory}/reading-${index + 1}-${width}.png` });
      }
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 2), false);
      assert.deepEqual(errors, []);
      results.push({ width, anchors: anchors.length, examples: exampleIndex, clockStates, plateauStates: expectedPlateau.length, equations: maths.length, sources: resourceLinks.length, errors });
      await page.close();
    }
    fs.writeFileSync(`${directory}/results.json`, JSON.stringify({ checkedAt: new Date().toISOString(), results }, null, 2));
    console.log(JSON.stringify(results));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

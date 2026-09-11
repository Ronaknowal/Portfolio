const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const directory = path.resolve('scratch/numerical-methods-browser');
const normalize = text => text.replace(/\s+/g, ' ').trim();
fs.mkdirSync(directory, { recursive: true });
(async () => {
  const { numericalMethodsExamples: examples } = await import(pathToFileURL(path.resolve('src/learn/data/numerical-methods-examples.js')));
  const models = await import(pathToFileURL(path.resolve('src/learn/data/numerical-methods-models.js')));
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [], errors = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => { if (message.type() === 'error' && !message.text().includes('[vite]')) errors.push(message.text()); });
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/numerical-methods-finite-differences-quadrature-root-finding?module=math-foundations', { waitUntil: 'domcontentloaded', timeout: 60000 });
      const lesson = page.locator('.numerical-methods-lesson');
      await lesson.waitFor();
      await page.evaluate(() => document.fonts.ready);
      const record = { width, anchors: [], states: [], captures: [], programs: [], geometry: [], fonts: await page.evaluate(() => [...document.fonts].filter(font => font.status === 'loaded').map(font => font.family)) };
      assert(record.fonts.length > 0, 'actual webfonts loaded');
      async function shot(target, name) {
        await target.evaluate(node => window.scrollTo({ top: node.getBoundingClientRect().top + scrollY - 85, behavior: 'instant' }));
        await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
        await page.waitForTimeout(150);
        const file = `${name}-${width}.png`;
        await page.screenshot({ path: path.join(directory, file) });
        record.captures.push(file);
      }
      async function press(region, name) {
        const button = region.getByRole('button', { name, exact: true });
        await button.focus();
        assert(await button.evaluate(node => document.activeElement === node && getComputedStyle(node).outlineStyle !== 'none'));
        await page.keyboard.press('Enter');
      }
      await shot(lesson.locator('.lesson-intro'), 'ordinary-intro');
      for (const link of await lesson.locator('nav[aria-label="In this lesson"] a').all()) {
        const href = await link.getAttribute('href');
        const target = lesson.locator(`[id="${href.slice(1)}"]`);
        assert.equal(await target.count(), 1);
        await link.focus(); await page.keyboard.press('Enter');
        await page.waitForFunction(id => { const box = document.getElementById(id).getBoundingClientRect(); return box.top >= 40 && box.top <= 150; }, href.slice(1));
        record.anchors.push(href);
        await shot(target, `ordinary-section-${record.anchors.length}`);
      }
      await shot(lesson.locator('.nm-inline'), 'inline-pump');
      for (const lab of await lesson.locator('[data-investigation]').all()) await shot(lab, `${await lab.getAttribute('data-investigation')}-initial`);
      assert.equal(await lesson.locator('[data-investigation]').count(), 6);

      const bracket = lesson.locator('[data-investigation="numerical-bracket"]');
      for (const key of ['square', 'cycle', 'repeated']) {
        await bracket.getByLabel('Bracket problem', { exact: true }).selectOption(key);
        const problem = models.rootProblems[key];
        const expected = models.bracketTrace(problem.f, problem.lower, problem.upper, { maximumSteps: 8, tolerance: .001 });
        for (let index = 0; index < expected.steps.length; index++) {
          const text = await bracket.locator('[aria-live]').innerText();
          assert(text.includes(`mathematical midpoint error bound ${models.formatNumerical(expected.steps[index].radius)}`));
          record.states.push({ lab: 'bracket', key, index });
          if (index < expected.steps.length - 1) await press(bracket, 'Next step');
        }
        if (!expected.steps.length) assert((await bracket.innerText()).includes('does not prove there is no root'));
      }
      await press(bracket, 'Reset');
      assert.equal(await bracket.getByLabel('Bracket problem', { exact: true }).inputValue(), 'square');
      await press(bracket, 'Next step'); await press(bracket, 'Back');

      const newton = lesson.locator('[data-investigation="numerical-newton"]');
      for (const key of ['cycle', 'square', 'repeated']) for (const method of ['plain', 'safe']) {
        await newton.getByLabel('Newton problem', { exact: true }).selectOption(key);
        await newton.getByLabel('Newton method', { exact: true }).selectOption(method);
        const problem = models.rootProblems[key];
        const expected = models.newtonTrace(problem, problem.start, { safeguarded: method === 'safe', maximumSteps: 8 });
        for (let index = 0; index < expected.steps.length; index++) {
          assert((await newton.locator('[aria-live]').innerText()).includes(`Use ${models.formatNumerical(expected.steps[index].next)}`));
          record.states.push({ lab: 'newton', key, method, index });
          if (index < expected.steps.length - 1) await press(newton, 'Next step');
        }
      }
      await press(newton, 'Reset'); await shot(newton, 'newton-cycle');

      const difference = lesson.locator('[data-investigation="numerical-difference"]');
      for (const stencil of ['central', 'forward', 'boundary', 'second']) for (const offset of ['0', '100000000']) {
        await difference.getByLabel('Derivative stencil', { exact: true }).selectOption(stencil);
        await difference.getByLabel('Function offset', { exact: true }).selectOption(offset);
        const slider = difference.getByRole('slider', { name: 'Negative power of ten', exact: true });
        await slider.focus(); await page.keyboard.press('Home');
        for (let exponent = 1; exponent <= 16; exponent++) {
          const expected = models.derivativeEstimate(1, 10 ** -exponent, stencil, Number(offset));
          const text = await difference.locator('[aria-live]').innerText();
          assert(text.includes(expected.status));
          if (expected.error !== null) assert(text.includes(`absolute error ${models.formatNumerical(expected.error)}`));
          record.states.push({ lab: 'difference', stencil, offset, exponent });
          if (exponent < 16) await page.keyboard.press('ArrowRight');
        }
      }
      await shot(difference, 'difference-small-spacing'); await press(difference, 'Reset');
      const quadrature = lesson.locator('[data-investigation="numerical-quadrature"]');
      for (const key of ['polynomial', 'sine']) for (const method of ['trapezoid', 'simpson']) for (const panels of ['2', '4', '8', '16']) {
        await quadrature.getByLabel('Area function', { exact: true }).selectOption(key);
        await quadrature.getByLabel('Area method', { exact: true }).selectOption(method);
        await quadrature.getByLabel('Number of panels', { exact: true }).selectOption(panels);
        const expected = models.compositeQuadrature(models.integralProblems[key].f, 0, 1, Number(panels), method);
        assert((await quadrature.locator('[aria-live]').innerText()).includes(`Area ${models.formatNumerical(expected.estimate)}`));
        record.states.push({ lab: 'quadrature', key, method, panels });
      }
      await press(quadrature, 'Reset');
      const adaptive = lesson.locator('[data-investigation="numerical-adaptive"]');
      for (const key of ['peak', 'blind']) for (const tolerance of ['0.0001', '0.000001', '0.00000001']) {
        await adaptive.getByLabel('Adaptive function', { exact: true }).selectOption(key);
        await adaptive.getByLabel('Requested area tolerance', { exact: true }).selectOption(tolerance);
        for (let depth = 1; depth <= 8; depth++) {
          const expected = models.adaptiveSimpson(models.integralProblems[key].f, 0, 1, { tolerance: Number(tolerance), maximumDepth: depth });
          const text = await adaptive.locator('[aria-live]').innerText();
          assert(text.includes(expected.status));
          assert(text.includes(`${expected.samples.length} evaluations`));
          record.states.push({ lab: 'adaptive', key, tolerance, depth });
          if (depth < 8) await press(adaptive, 'Allow another level');
        }
      }
      await shot(adaptive, 'adaptive-blind'); await press(adaptive, 'Reset');
      const calibration = lesson.locator('[data-investigation="numerical-calibration"]');
      for (const target of ['.7', '.8', '.85']) for (const panels of ['8', '32', '128']) for (const [key, candidate] of [['Home', 2], ['End', 4]]) {
        await calibration.getByLabel('Target litres', { exact: true }).selectOption(target);
        await calibration.getByLabel('Integral panels', { exact: true }).selectOption(panels);
        const slider = calibration.getByRole('slider', { name: 'Candidate minutes', exact: true });
        await slider.focus(); await page.keyboard.press(key);
        const expected = models.pumpCalibration(Number(target), Number(panels), candidate);
        assert((await calibration.locator('[aria-live]').innerText()).includes(`bound: ${models.formatNumerical(expected.timeBound)} minutes`));
        record.states.push({ lab: 'calibration', target, panels, candidate });
      }
      await press(calibration, 'Reset'); await shot(calibration, 'calibration-unresolved');

      assert.equal(await lesson.locator('.nm-practice').count(), 10);
      for (const item of await lesson.locator('.nm-practice').all()) {
        const summaries = item.locator(':scope > details > summary');
        assert.equal(await summaries.count(), 2);
        await summaries.nth(0).focus(); await page.keyboard.press('Enter');
        await summaries.nth(1).focus(); await page.keyboard.press('Enter');
      }
      for (const example of Object.values(examples)) {
        const block = lesson.locator('.python-example').filter({ has: page.getByRole('heading', { name: example.title, exact: true }) });
        assert.equal(await block.count(), 1);
        assert(normalize(await block.evaluate(node => node.previousElementSibling.textContent)).includes(normalize(example.question)));
        const text = normalize(await block.innerText());
        assert(text.includes(normalize(example.code)), `actual code ${example.title}`);
        assert(text.includes(normalize(example.expected)), `actual output ${example.title}`);
        record.programs.push(example.title);
      }
      await shot(lesson.locator('.python-example').first(), 'first-program');
      await shot(lesson.locator('.python-example').first().locator('.lesson-note'), 'first-output');
      await shot(lesson.locator('.nm-practice').nth(8), 'changed-practice');
      await shot(lesson.locator('.lesson-sources'), 'sources');
      assert.equal(await lesson.locator('.katex-error').count(), 0);
      for (const equation of await lesson.locator('.katex-display').all()) {
        const size = await equation.evaluate(node => ({ scroll: node.scrollWidth, width: node.clientWidth }));
        record.geometry.push({ type: 'equation', ...size });
      }
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1);
      assert(!overflow, `page overflow ${width}`);
      for (const svg of await lesson.locator('svg.nm-plot').all()) {
        const clipped = await svg.evaluate(node => {
          const box = node.getBoundingClientRect();
          return [...node.querySelectorAll('text')].map(text => ({ text: text.textContent, rect: text.getBoundingClientRect() })).filter(({ rect }) => rect.left < box.left - 2 || rect.right > box.right + 2 || rect.top < box.top - 2 || rect.bottom > box.bottom + 2).map(item => item.text);
        });
        record.geometry.push({ type: 'plot-labels', clipped });
      }
      assert(!normalize(await lesson.innerText()).includes('\\u2212'), 'no literal unicode escapes');
      records.push(record);
      fs.writeFileSync(path.join(directory, 'progress.json'), JSON.stringify({ records, errors }, null, 2));
      await page.close();
    }
    assert.deepEqual(errors, []);
    fs.writeFileSync(path.join(directory, 'results.json'), JSON.stringify({ checkedAt: new Date().toISOString(), passed: true, records, errors }, null, 2));
    console.log('All browser controls, question/code/output, anchors, keyboard and page checks passed at 1440/390/320. Inspect recorded equation and plot-label geometry before freeze.');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

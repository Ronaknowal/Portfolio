const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const directory = 'scratch/ordinary-differential-equations-independent';
const fingerprint = path => ({ path, sha256: createHash('sha256').update(fs.readFileSync(path)).digest('hex') });

(async () => {
  const { ordinaryDifferentialEquationsExamples: examples } = await import('../src/learn/data/ordinary-differential-equations-examples.js');
  const native = JSON.parse(fs.readFileSync(`${directory}/results.json`, 'utf8'));
  const result = { at: new Date().toISOString(), sources: native.sources, records: [], images: [] };
  for (const source of result.sources) assert.equal(fingerprint(source.path).sha256, source.sha256);
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  result.browser = browser.version();
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      const errors = [], failedRequests = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('requestfailed', request => failedRequests.push(request.url()));
      page.on('console', message => { if (message.type() === 'error' && !message.text().includes('[vite]')) errors.push(message.text()); });
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/ordinary-differential-equations-linear-systems?module=math-foundations', { waitUntil: 'networkidle' });
      const lesson = page.locator('.ode-lesson');
      await lesson.waitFor();
      await page.evaluate(() => document.fonts.ready);
      const fonts = await page.evaluate(() => [...document.fonts].filter(font => font.status === 'loaded').map(font => font.family));
      assert(fonts.includes('Space Grotesk') && fonts.includes('JetBrains Mono'));
      assert.equal(await page.locator('vite-error-overlay').count(), 0);
      assert.equal(await page.locator('.lesson-guide').count(), 0);
      assert.equal(await lesson.locator('.ode-lab').count(), 8);
      const lab = name => lesson.getByRole('region', { name, exact: true });
      const range = (region, name, value) => region.getByRole('slider', { name, exact: true }).fill(String(value));
      async function capture(target, name) {
        const height = Math.max(1000, Math.ceil((await target.boundingBox()).height) + 160);
        await page.setViewportSize({ width, height });
        await target.evaluate(node => scrollTo({ top: scrollY + node.getBoundingClientRect().top - 80, behavior: 'instant' }));
        const path = `${directory}/${name}-${width}.png`;
        await target.screenshot({ path });
        result.images.push({ ...fingerprint(path), viewport: { width, height } });
        const clipped = await target.locator('svg text').evaluateAll(nodes => nodes.filter(node => {
          const box = node.getBBox(), view = node.ownerSVGElement.viewBox.baseVal;
          return box.x < -1 || box.y < -1 || box.x + box.width > view.width + 1 || box.y + box.height > view.height + 1;
        }).map(node => node.textContent));
        assert.deepEqual(clipped, [], name);
        await page.setViewportSize({ width, height: 1000 });
      }
      const field = lab('Rate and direction field investigation');
      await field.getByLabel('Rate law', { exact: true }).selectOption('logistic');
      await range(field, 'Initial state', 3);
      await range(field, 'Inspection time', 2.5);
      const expected = 30 / (3 + 7 * Math.exp(-2.5));
      const point = await field.locator('svg circle').evaluate(node => [Number(node.getAttribute('cx')), Number(node.getAttribute('cy'))]);
      assert(Math.abs(point[0] - (49 + 2.5 * 35)) < 1e-11);
      assert(Math.abs(point[1] - (180 - expected * 142 / 20)) < 1e-10);
      assert((await field.locator('.ode-readout').innerText()).includes('rising'));
      await capture(field, 'changed-logistic-trajectory');

      const waiting = lab('Nonunique waiting solutions investigation');
      await range(waiting, 'Departure time', 2.2);
      assert((await waiting.innerText()).includes('At t = 3 it equals 0.64'));
      await capture(waiting, 'changed-departure');

      const oscillator = lab('Position velocity and energy investigation');
      await oscillator.getByLabel('Damping c', { exact: true }).selectOption('4');
      await oscillator.getByLabel('Initial velocity', { exact: true }).selectOption('-2');
      await range(oscillator, 'Motion inspection time', 0.8);
      const q = Math.exp(-1.6), v = -2 * q;
      const strip = await oscillator.locator('.ode-state-strip').innerText();
      assert(strip.includes(`q = ${Number(q.toFixed(4))} m`) && strip.includes(`v = ${Number(v.toFixed(4))} m/s`));
      await capture(oscillator, 'critical-eigenvector-motion');

      const columns = lab('Fundamental matrix columns investigation');
      await columns.getByLabel('Linear system', { exact: true }).selectOption('nilpotent');
      await columns.getByLabel('Initial vector', { exact: true }).selectOption('sum');
      await range(columns, 'Matrix inspection time', 1.7);
      assert((await columns.innerText()).includes('selected state is [2.7, 1]'));
      const stationary = columns.locator('circle[data-stationary-trajectory="initial e₁"]');
      assert.equal(await stationary.count(), 1);
      const stationaryPoint = await stationary.evaluate(node => [Number(node.getAttribute('cx')), Number(node.getAttribute('cy'))]);
      assert(Math.abs(stationaryPoint[0] - (48 + 218 / 4)) < 1e-12);
      assert(Math.abs(stationaryPoint[1] - (182 - 0.06 / 1.12 * 144)) < 1e-12);
      await capture(columns, 'changed-summed-columns');

      const forcing = lab('Initial and forced response investigation');
      await range(forcing, 'First interval power', 35);
      await range(forcing, 'Second interval power', 5);
      await range(forcing, 'Input switch time', 2.4);
      await range(forcing, 'Response inspection time', 1.2);
      const initialResponse = 40 * Math.exp(-0.24), inputResponse = 17.5 * -Math.expm1(-0.24);
      assert((await forcing.locator('.ode-readout').innerText()).includes(`${Number(initialResponse.toFixed(4))} + ${Number(inputResponse.toFixed(4))} + 0`));
      await capture(forcing, 'second-input-not-yet-active');

      const order = lab('Order of two continuous stages investigation');
      await order.getByLabel('Stage initial state', { exact: true }).selectOption('second');
      await order.getByLabel('Stage order', { exact: true }).selectOption('lower');
      assert((await order.innerText()).includes('Final state: [1, 1]'));
      await capture(order, 'changed-chronological-stages');

      const numerical = lab('Numerical stages and error investigation');
      await numerical.getByLabel('Step method', { exact: true }).selectOption('midpoint');
      await numerical.getByLabel('Requested step h', { exact: true }).selectOption('0.35');
      await numerical.getByLabel('Decay rate k', { exact: true }).selectOption('4');
      const cells = await numerical.locator('tbody tr').evaluateAll(rows => rows.map(row => [...row.querySelectorAll('td')].map(cell => Number(cell.textContent))));
      assert.deepEqual(cells, [[0, 40, -160], [0.175, 12, -48]]);
      assert((await numerical.locator('.ode-readout').innerText()).includes('15 steps; final time 5'));
      await capture(numerical, 'changed-midpoint-probes');

      const boundary = lab('Boundary conditions and uniqueness investigation');
      await boundary.getByLabel('Right endpoint', { exact: true }).selectOption('half-pi');
      await boundary.getByLabel('Requested endpoint value', { exact: true }).selectOption('1');
      await range(boundary, 'Candidate initial slope', -1.5);
      assert((await boundary.innerText()).includes('one solution'));
      assert((await boundary.innerText()).includes('required slope is 1'));
      await capture(boundary, 'candidate-does-not-meet-boundary');

      let keyboardActions = 0;
      for (const button of await lesson.locator('.ode-lab button').all()) {
        await button.focus();
        assert(await button.evaluate(node => document.activeElement === node && getComputedStyle(node).outlineStyle !== 'none'));
        await page.keyboard.press('Enter'); keyboardActions += 1;
      }
      const slider = field.getByRole('slider', { name: 'Initial state', exact: true });
      await slider.focus(); await page.keyboard.press('ArrowRight'); assert.equal(await slider.inputValue(), '11'); keyboardActions += 1;
      for (const summary of await lesson.locator('.ode-depth > summary, .ode-practice details > summary').all()) {
        await summary.focus(); await page.keyboard.press('Enter'); keyboardActions += 1;
      }
      const programs = await lesson.locator('.python-example').evaluateAll(nodes => nodes.map(node => ({ title: node.querySelector('h3').textContent,
        question: node.previousElementSibling.textContent.replace(/^Before running:\s*/, ''),
        blocks: [...node.children].filter(child => getComputedStyle(child).whiteSpace === 'pre').map(child => [...child.childNodes].filter(item => item.nodeType === Node.TEXT_NODE).map(item => item.textContent).join('')) })));
      assert.equal(programs.length, 13);
      for (const program of programs) {
        const example = Object.values(examples).find(row => row.title === program.title);
        assert(example); assert.equal(program.question, example.question);
        assert.equal(program.blocks[0].trim(), example.code.trim()); assert.equal(program.blocks[1].trim(), example.expected.trim());
      }
      const equations = await lesson.locator('.katex-display').evaluateAll(nodes => nodes.map(node => ({ width: node.clientWidth, content: node.scrollWidth })));
      assert.equal(equations.length, 23); assert(equations.every(row => row.content <= row.width + 1));
      const anchors = await lesson.locator('.lesson-intro a[href^="#"]').evaluateAll(nodes => nodes.map(node => !!document.getElementById(node.hash.slice(1))));
      assert.equal(anchors.length, 14); assert(anchors.every(Boolean));
      for (let index = 0; index < 3; index += 1) await capture(lesson.locator('.ode-inline-figure').nth(index), `inline-${index}`);
      await capture(lesson.locator('.ode-balance'), 'energy-balance');
      await capture(lesson.locator('.ode-practice').last(), 'changed-complete-heating-task');
      const amended = await page.evaluate(async () => {
        const model = await import('/src/learn/data/ordinary-differential-equations-models.js');
        const inherited = [1, ,]; Object.setPrototypeOf(inherited, { 1: 2 });
        const invalid = [() => model.linearSystemState([[0, 0], [0, 0]], inherited, 1), () => model.coolingState(1e-320, 0, 1e-5, 1e6)].map(action => { try { action(); return false; } catch { return true; } });
        return { initial: model.coolingState(0, 1e-200, 1, 1).state, tinyTime: model.coolingState(1e-200, 1e-200, 1, 1).state, invalid };
      });
      assert.equal(amended.initial, 1e-200); assert.equal(amended.tinyTime, 2e-200); assert(amended.invalid.every(Boolean));
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
      assert.deepEqual(errors, []); assert.deepEqual(failedRequests, []);
      result.records.push({ width, fonts, investigations: 8, inlineFigures: 4, programs: 13, equations: 23, anchors: 14, keyboardActions, amended, errors, failedRequests });
      await page.close();
    }
    result.passed = true;
    fs.writeFileSync(`${directory}/browser-results.json`, JSON.stringify(result, null, 2) + '\n');
    console.log(JSON.stringify({ passed: true, widths: result.records.map(row => row.width), images: result.images.length }));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const directory = 'scratch/single-variable-calculus-browser';
fs.mkdirSync(directory, { recursive: true });
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const result = { at: new Date().toISOString(), browser: browser.version(), widths: [], images: [] };
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1050 } });
      const errors = [], states = [], overflow = [], svgTextOverflow = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/single-variable-calculus-limits-derivatives-integrals?module=math-foundations', { waitUntil: 'domcontentloaded' });
      const lesson = page.locator('.single-calculus-lesson');
      await lesson.waitFor(); await page.evaluate(() => document.fonts.ready);
      assert(await page.evaluate(() => document.fonts.check('16px "Space Grotesk"')));
      assert.equal(await lesson.locator('.calculus-lab').count(), 8);
      assert.equal(await lesson.locator('.python-example').count(), 15);
      assert.equal(await lesson.locator('.calculus-practice').count(), 13);
      assert.equal(await lesson.locator('.calculus-figure').count(), 4);
      const lab = index => lesson.locator('.calculus-lab').nth(index);
      async function capture(locator, name) {
        const height = Math.max(1050, Math.ceil((await locator.boundingBox()).height) + 160);
        await page.setViewportSize({ width, height });
        await locator.evaluate(node => scrollTo({ top: node.getBoundingClientRect().top + scrollY - 84, behavior: 'instant' }));
        const screenshot = `${directory}/${name}-${width}.png`;
        await locator.screenshot({ path: screenshot });
        result.images.push({ path: screenshot, captureViewport: {width, height}, sha256: crypto.createHash('sha256').update(fs.readFileSync(screenshot)).digest('hex') });
        const overflowing = await locator.locator('svg text').evaluateAll(nodes => nodes.filter(node => {
          const box = node.getBBox(), vb = node.ownerSVGElement.viewBox.baseVal;
          return box.x < -1 || box.y < -1 || box.x + box.width > vb.width + 1 || box.y + box.height > vb.height + 1;
        }).map(node => ({ text: node.textContent, box: { x: node.getBBox().x, width: node.getBBox().width } })));
        if (overflowing.length) svgTextOverflow.push({ name, overflowing });
        states.push({ name, text: await locator.innerText() });
        await page.setViewportSize({ width, height: 1050 });
      }
      const initial = [];
      for (let i = 0; i < 8; i++) initial.push(await lab(i).innerText());
      await lab(0).getByRole('slider', { name: 'Base time', exact: true }).fill('1');
      await lab(0).getByRole('combobox').selectOption('-0.1');
      await capture(lab(0), 'turning-point');
      assert((await lab(0).innerText()).includes('0 m/s'));
      await lab(0).getByRole('button', { name: 'Reset motion rate' }).click();
      await lab(1).getByRole('slider', { name: 'Output tolerance epsilon' }).fill('0.84');
      await lab(1).getByRole('slider', { name: 'Input radius delta' }).fill('0.2');
      await capture(lab(1), 'exact-limit-boundary');
      await lab(1).getByRole('combobox').selectOption('hole');
      await capture(lab(1), 'changed-center');
      await lab(1).getByRole('combobox').selectOption('jump');
      await capture(lab(1), 'jump-witness');
      await lab(1).getByRole('button', { name: 'Reset limit guarantee' }).click();
      await lab(2).getByRole('button', { name: 'Make the outer slope zero' }).click();
      assert(await lab(2).getByRole('slider', { name: 'Input x', exact: true }).evaluate(node => node.validity.valid));
      await capture(lab(2), 'zero-outer-slope');
      await lab(2).getByRole('slider', { name: 'Input change h' }).fill('-0.5');
      await capture(lab(2), 'negative-composition-change');
      await lab(2).getByRole('button', { name: 'Reset composition' }).click();
      await lab(3).getByRole('combobox').selectOption('cusp');
      await capture(lab(3), 'cusp-tied-endpoints');
      const candidateState = await lab(3).locator('.calculus-candidates').innerText();
      await lab(3).getByRole('textbox', { name: 'Left endpoint' }).fill('');
      await lab(3).getByRole('button', { name: 'Apply interval' }).click();
      assert(await lab(3).getByRole('alert').isVisible());
      assert.equal(await lab(3).locator('.calculus-candidates').innerText(), candidateState);
      await capture(lab(3), 'invalid-interval-retains-state');
      await lab(3).getByRole('textbox', { name: 'Left endpoint' }).fill('3.25');
      await lab(3).getByRole('textbox', { name: 'Right endpoint' }).fill('3.75');
      await lab(3).getByRole('combobox').selectOption('motion');
      await lab(3).getByRole('button', { name: 'Apply interval' }).click();
      assert.equal(await lab(3).locator('.calculus-candidates > div').count(), 2);
      await capture(lab(3), 'interval-excludes-critical-points');
      await lab(3).getByRole('button', { name: 'Reset extrema' }).click();
      await lab(4).getByRole('combobox', { name: 'Rectangle count' }).selectOption('3');
      await lab(4).getByRole('combobox', { name: 'Sample in each strip' }).selectOption('midpoint');
      assert.equal(await lab(4).locator('rect.positive-panel, rect.negative-panel').count(), 3);
      await capture(lab(4), 'signed-three-rectangles');
      await lab(4).getByRole('slider', { name: 'Upper time T' }).fill('0.25');
      await lab(4).getByRole('combobox', { name: 'Rectangle count' }).selectOption('64');
      assert.equal(await lab(4).locator('rect.positive-panel, rect.negative-panel').count(), 64);
      await capture(lab(4), 'early-accumulation');
      await lab(4).getByRole('button', { name: 'Reset accumulation' }).click();
      await lab(5).getByRole('slider', { name: 'Continuous rate k' }).fill('-0.8');
      await lab(5).getByRole('slider', { name: 'Period length' }).fill('2');
      await capture(lab(5), 'decay-tangent');
      await lab(5).getByRole('slider', { name: 'Continuous rate k' }).fill('0');
      await capture(lab(5), 'zero-growth');
      await lab(5).getByRole('button', { name: 'Reset exponential rate' }).click();
      await lab(6).getByRole('combobox').selectOption('log');
      await lab(6).getByRole('slider', { name: 'Evaluation input' }).fill('1.5');
      await lab(6).getByRole('slider', { name: 'Polynomial degree' }).fill('12');
      await capture(lab(6), 'log-outside-series-radius');
      await lab(6).getByRole('slider', { name: 'Evaluation input' }).fill('-0.9');
      await capture(lab(6), 'log-near-domain-edge');
      await lab(6).getByRole('button', { name: 'Reset Taylor approximation' }).click();
      for (const endpoint of ['endpoint', 'tail']) for (const power of ['0.5', '1', '2']) {
        await lab(7).getByRole('combobox', { name: 'Difficult endpoint' }).selectOption(endpoint);
        await lab(7).getByRole('combobox', { name: 'Power p in x to minus p' }).selectOption(power);
        await lab(7).getByRole('slider', { name: 'Cutoff exponent q' }).fill('4');
        await capture(lab(7), `improper-${endpoint}-${power}`);
      }
      await lab(7).getByRole('button', { name: 'Reset improper integral' }).click();
      for (let i = 0; i < 8; i++) assert.equal(await lab(i).innerText(), initial[i], `reset ${i}`);
      let keyboardChecks = 0;
      for (let i = 0; i < 8; i++) {
        const slider = lab(i).getByRole('slider').first();
        if (await slider.count()) {
          const previous = await slider.inputValue();
          const key = Number(previous) >= Number(await slider.getAttribute('max')) ? 'ArrowLeft' : 'ArrowRight';
          await slider.focus(); await slider.press(key);
          assert.notEqual(await slider.inputValue(), previous); keyboardChecks++;
          const reset = lab(i).getByRole('button', { name: /^Reset/ });
          await reset.focus(); await reset.press('Enter'); keyboardChecks++;
          assert.equal(await lab(i).innerText(), initial[i]);
        }
      }
      for (let i = 0; i < 4; i++) await capture(lesson.locator('.calculus-figure').nth(i), `inline-${i}`);
      for (const detail of await lesson.locator('details').all()) {
        if (!await detail.evaluate(node => node.open)) await detail.locator(':scope > summary').click();
      }
      const equations = await lesson.locator('.katex-display').evaluateAll(nodes => nodes.map((node,index) => ({ index, client: node.clientWidth, scroll: node.scrollWidth, text: node.textContent })));
      overflow.push(...equations.filter(node => node.scroll > node.client + 2));
      assert.equal(await lesson.locator('.katex-error').count(), 0);
      const anchors = await lesson.locator('a[href^="#"]').evaluateAll(nodes => nodes.map(node => ({ href: node.getAttribute('href'), exists: !!document.getElementById(decodeURIComponent(node.hash.slice(1))) })));
      assert(anchors.length >= 14 && anchors.every(link => link.exists));
      for (const link of await lesson.locator('a[href^="#"]').all()) { await link.click(); assert(await page.evaluate(() => !!document.getElementById(decodeURIComponent(location.hash.slice(1))))); }
      await capture(lesson.locator('.calculus-practice').nth(8), 'changed-decay-practice');
      const docOverflow = await page.evaluate(() => ({ viewport: innerWidth, actual: document.documentElement.scrollWidth }));
      result.widths.push({ width, keyboardChecks, states, equations: equations.length, equationOverflow: overflow, svgTextOverflow, anchors, docOverflow, errors });
      fs.writeFileSync(`${directory}/results.json`, JSON.stringify(result, null, 2));
      assert.deepEqual(errors, []);
      assert.deepEqual(overflow, [], 'equation overflow');
      assert.deepEqual(svgTextOverflow, [], 'SVG text overflow');
      assert(docOverflow.actual <= docOverflow.viewport);
      await page.close();
    }
    result.passed = true;
    fs.writeFileSync(`${directory}/results.json`, JSON.stringify(result, null, 2));
    console.log(JSON.stringify({ passed: true, widths: result.widths.map(r => ({ width: r.width, states: r.states.length, equations: r.equations, keyboardChecks: r.keyboardChecks })), images: result.images.length }));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

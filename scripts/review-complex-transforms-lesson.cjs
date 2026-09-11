const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const directory = 'scratch/complex-transforms-browser';
fs.mkdirSync(directory, { recursive: true });
const widths = process.argv.includes('--narrow') ? [320] : [1440, 390, 320];
const sourcePaths = ['src/learn/data/topics/complex-numbers-fourier-laplace-transforms.jsx', 'src/learn/data/complex-transforms-models.js', 'src/learn/data/complex-transforms-examples.js', 'src/learn/components/lesson-labs/ComplexTransformsLabs.jsx', 'src/learn/components/lesson-labs/complex-transforms-labs.css', 'src/learn/data/curriculum/blueprints/complex-numbers-fourier-laplace-transforms.js'];
const fingerprints = () => sourcePaths.map(path => ({ path, sha256: crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex') }));
async function slider(input, next) {
  await input.evaluate((node, value) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(node, String(value));
    node.dispatchEvent(new Event('input', { bubbles: true }));
    node.dispatchEvent(new Event('change', { bubbles: true }));
  }, next);
}
async function capture(page, target, name, width, whole = false) {
  await target.evaluate(node => scrollTo({ top: scrollY + node.getBoundingClientRect().top - 145, behavior: 'instant' }));
  if (whole) {
    const height = Math.max(1000, Math.ceil((await target.boundingBox()).height) + 160);
    await page.setViewportSize({ width, height });
    await target.screenshot({ path: `${directory}/${name}-${width}.png` });
    await page.setViewportSize({ width, height: 1000 });
  } else await page.screenshot({ path: `${directory}/${name}-${width}.png` });
}
async function value(region, label) {
  return region.locator('.transform-values>div').filter({ has: region.page().getByText(label, { exact: true }) }).locator('dd').innerText();
}
(async () => {
  const { complexTransformExamples: examples } = await import('../src/learn/data/complex-transforms-examples.js');
  const models = await import('../src/learn/data/complex-transforms-models.js');
  const format = (number, digits = 3) => Math.abs(number) < 1e-11 ? '0' : Math.abs(number) >= 1e6 ? number.toExponential(3) : Number(number.toFixed(digits)).toString();
  const complexText = z => `${format(z[0])} ${z[1] < -1e-11 ? '−' : '+'} ${format(Math.abs(z[1]))}i`;
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const sourceHashes = fingerprints();
  const results = [];
  try {
    for (const width of widths) {
      const page = await browser.newPage({ viewport: { width, height: 1000 } });
      await page.routeWebSocket('**', socket => socket.close());
      const errors = [], warnings = [], failedRequests = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => { if (['warning', 'error'].includes(message.type()) && !message.text().startsWith('[vite]')) warnings.push(message.text()); });
      page.on('requestfailed', request => failedRequests.push(request.url()));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/complex-numbers-fourier-laplace-transforms?module=math-foundations', { waitUntil: 'networkidle' });
      const lesson = page.locator('.complex-transforms-lesson');
      await lesson.waitFor();
      await page.evaluate(() => document.fonts.ready);
      assert(await page.evaluate(() => document.fonts.check('16px "Space Grotesk"')));
      assert.equal(await page.locator('vite-error-overlay').count(), 0);
      assert.equal(await lesson.locator('.katex-error').count(), 0);
      const anchors = await lesson.locator('.lesson-intro a[href^="#"]').evaluateAll(nodes => nodes.map(node => ({ href: node.getAttribute('href'), exists: !!document.getElementById(node.getAttribute('href').slice(1)) })));
      assert.equal(anchors.length, 12); assert(anchors.every(row => row.exists), JSON.stringify(anchors));
      const programs = await lesson.locator('.python-example').evaluateAll(nodes => nodes.map(node => ({ title: node.querySelector('h3').textContent, question: node.previousElementSibling.textContent.replace(/^Before running\.\s*/, ''), blocks: [...node.children].filter(child => getComputedStyle(child).whiteSpace === 'pre').map(child => [...child.childNodes].filter(n => n.nodeType === Node.TEXT_NODE).map(n => n.textContent).join('')) })));
      assert.equal(programs.length, 17);
      for (const program of programs) {
        const example = Object.values(examples).find(row => row.title === program.title);
        assert(example); assert.equal(program.question, example.question);
        assert.equal(program.blocks[0].trim(), example.code.trim()); assert.equal(program.blocks[1].trim(), example.expected.trim());
      }
      const equations = await lesson.locator('.katex-display').evaluateAll(nodes => nodes.map((node, index) => ({ index, client: node.clientWidth, scroll: node.scrollWidth, text: node.textContent })));
      const overflowEquations = equations.filter(row => row.scroll > row.client + 2);
      fs.writeFileSync(`${directory}/equations-${width}.json`, JSON.stringify(equations, null, 2));
      for (let i = 0; i < 12; i += 1) await capture(page, lesson.locator('h2').nth(i), `reading-${i + 1}`, width);
      for (let i = 0; i < 3; i += 1) await capture(page, lesson.locator('.transform-inline').nth(i), `inline-${i + 1}`, width, true);
      for (let i = 0; i < 10; i += 1) await capture(page, lesson.locator('.transform-lab').nth(i), `default-${i + 1}`, width, true);
      let states = 0;
      const arithmetic = lesson.locator('[data-transform-lab="arithmetic"]');
      for (const operation of ['multiply', 'add', 'divide']) {
        await arithmetic.getByLabel('Complex operation', { exact: true }).selectOption(operation);
        for (const [real, imaginary] of [[0, 1], [-3, 0], [1.25, -.5], [0, 0]]) {
          await slider(arithmetic.getByLabel('Real part of w', { exact: true }), real);
          await slider(arithmetic.getByLabel('Imaginary part of w', { exact: true }), imaginary);
          if (operation === 'divide' && real === 0 && imaginary === 0) assert((await arithmetic.locator('p[role="status"]').innerText()).includes('Division by zero'));
          else assert.equal(await value(arithmetic, 'Result'), complexText(models.complexOperation([1, 2], [real, imaginary], operation).result));
          if (operation === 'multiply' && real === 0 && imaginary === 0) assert.equal(await arithmetic.locator('.transform-plane svg path').count(), 1);
          states += 1;
        }
      }
      await capture(page, arithmetic, 'division-zero', width, true);
      await arithmetic.getByRole('button', { name: 'Reset complex numbers' }).focus(); await page.keyboard.press('Enter');
      await arithmetic.getByLabel('Real part of w', { exact: true }).focus(); await page.keyboard.press('ArrowRight');
      assert.equal(await arithmetic.getByLabel('Real part of w', { exact: true }).inputValue(), '2.25'); states += 2;
      const synthesis = lesson.locator('[data-transform-lab="synthesis"]');
      for (const index of [0, 64, 128, 256]) {
        await slider(synthesis.getByLabel('Synthesis time', { exact: true }), index);
        assert.equal(await value(synthesis, 'Measured sum'), format(models.phasorSynthesis(index/256).signal)); states += 1;
      }
      await slider(synthesis.getByLabel('Synthesis third-tone phase', { exact: true }), 0); states += 1;
      const projection = lesson.locator('[data-transform-lab="projection"]');
      for (const k of [-3, 0, 1, 2, 3]) for (const offset of [-2, 0, 2]) {
        await slider(projection.getByLabel('Selected harmonic k', { exact: true }), k); await slider(projection.getByLabel('Signal offset', { exact: true }), offset);
        assert.equal(await value(projection, 'Full-period coefficient cₖ'), complexText(models.harmonicProjection(2, 1, Math.PI / 2, offset, k).coefficient)); states += 1;
      }
      await slider(projection.getByLabel('Third-tone phase', { exact: true }), 0);
      await slider(projection.getByLabel('Integration cursor', { exact: true }), 64);
      assert.equal(await value(projection, 'Integral so far'), complexText(models.harmonicProjection(2, 1, 0, 2, 3).points[64].accumulated)); states += 1;
      await capture(page, projection, 'changed-projection', width, true);
      const convergence = lesson.locator('[data-transform-lab="convergence"]');
      for (const terms of [1, 8, 32, 64]) {
        await slider(convergence.getByLabel('Number of odd harmonics', { exact: true }), terms);
        assert.equal(await value(convergence, 'First peak height'), format(models.squareConvergence(terms).firstPeak, 6)); states += 1;
      }
      await capture(page, convergence, 'moving-peak', width, true);
      const dft = lesson.locator('[data-transform-lab="dft"]');
      for (const text of ['2, 0, -2, 0', '1, -1, 2, -2, 3, -3, 4, -4']) {
        await dft.getByLabel('DFT sample draft', { exact: true }).fill(text); await dft.getByRole('button', { name: 'Apply sample list' }).click();
        const samples = models.parseRealSamples(text);
        for (const k of [0, 1, samples.length - 1]) {
          await slider(dft.getByLabel('DFT bin k', { exact: true }), k);
          assert.equal(await value(dft, 'Selected coefficient'), complexText(models.finiteFourier(samples, k).spectrum[k])); states += 1;
        }
      }
      await dft.getByRole('checkbox').check();
      assert.equal(await dft.getByRole('checkbox').isChecked(), true); states += 1;
      await dft.getByRole('checkbox').focus(); await page.keyboard.press('Space');
      assert.equal(await dft.getByRole('checkbox').isChecked(), false); states += 1;
      await dft.getByLabel('DFT sample draft', { exact: true }).fill('4,-4,4,4,4,4,4,-4');
      await dft.getByRole('button', { name: 'Apply sample list' }).click();
      await slider(dft.getByLabel('DFT bin k', { exact: true }), 1);
      await dft.getByRole('checkbox').check();
      const inverseChart = dft.locator('.transform-chart svg');
      const inverseDots = await inverseChart.locator('circle').evaluateAll(nodes => nodes.map(node => Number(node.getAttribute('cy'))));
      assert.equal(inverseDots.length, 16);
      assert(inverseDots.every(y => y >= 24 && y <= 164), JSON.stringify(inverseDots));
      assert(models.finiteFourier([4,-4,4,4,4,4,4,-4], 1, true).reconstructed[0][0] > 6);
      await capture(page, dft, 'expanded-inverse-range', width, true); states += 1;
      await dft.getByRole('checkbox').uncheck();
      for (const text of ['1,,2,3', '1,NaN,2,3', '1,1e-999,2,3', '1,5,2,3']) {
        const previous = await value(dft, 'Selected coefficient');
        await dft.getByLabel('DFT sample draft', { exact: true }).fill(text); await dft.getByRole('button', { name: 'Apply sample list' }).click();
        assert((await dft.getByRole('alert').innerText()).includes('unchanged')); assert.equal(await value(dft, 'Selected coefficient'), previous); states += 1;
      }
      await capture(page, dft, 'invalid-dft-preserves-state', width, true);
      await dft.getByLabel('DFT sample draft', { exact: true }).fill('1e-12,0,0,0');
      await dft.getByRole('button', { name: 'Apply sample list' }).click();
      assert.equal(await value(dft, 'Its principal phase'), 'Not displayed (≤10⁻¹⁰)'); states += 1;
      await capture(page, dft, 'tiny-valid-dft', width, true);
      const alias = lesson.locator('[data-transform-lab="alias"]');
      for (const rate of [8, 16, 32]) for (const frequency of [0, 8, 13, 24]) {
        await alias.getByLabel('Sampling rate', { exact: true }).selectOption(String(rate)); await slider(alias.getByLabel('Continuous tone frequency', { exact: true }), frequency);
        assert.equal(await value(alias, 'Maximum sample discrepancy'), '0'); states += 1;
      }
      await slider(alias.getByLabel('Tone phase', { exact: true }), -Math.PI / 2);
      await capture(page, alias, 'changed-alias', width, true); states += 1;
      const window = lesson.locator('[data-transform-lab="window"]');
      for (const count of [32, 64, 128]) for (const padding of [count, 256]) for (const taper of ['rectangular', 'hann']) {
        await window.getByLabel('Observed samples N', { exact: true }).selectOption(String(count)); await window.getByLabel('DFT length M', { exact: true }).selectOption(String(padding)); await window.getByLabel('Time window', { exact: true }).selectOption(taper);
        const target = models.windowSpectrum(count, padding, taper, 5.5);
        assert.equal(await value(window, 'Σ w²x² / Σ w²'), `${format(target.weightedMeanSquare, 6)} V²`);
        assert.equal(await value(window, 'Sum of one-sided density × Δf'), `${format(target.densityIntegral, 6)} V²`); states += 1;
      }
      await slider(window.getByLabel('Windowed tone frequency', { exact: true }), 6.125);
      const curveBefore = await window.locator('svg path').allAttributeValues?.('d') ?? await window.locator('svg path').evaluateAll(nodes => nodes.map(node => node.getAttribute('d')));
      await slider(window.getByLabel('Inspect frequency', { exact: true }), 7);
      assert.deepEqual(await window.locator('svg path').evaluateAll(nodes => nodes.map(node => node.getAttribute('d'))), curveBefore); states += 2;
      await capture(page, window, 'padded-hann', width, true);
      const convolution = lesson.locator('[data-transform-lab="convolution"]');
      for (const boundary of ['linear', 'circular']) {
        await convolution.getByLabel('Convolution boundary', { exact: true }).selectOption(boundary);
        for (let index = 0; index < (boundary === 'linear' ? 5 : 4); index += 1) {
          await slider(convolution.getByLabel('Convolution output index', { exact: true }), index);
          const expected = models.convolutionState([1, 2, 0, -1], [1, 1], boundary === 'circular', index).output[index];
          assert((await convolution.locator('.transform-equation-line').innerText()).includes(`y[${index}] = ${expected}`)); states += 1;
        }
      }
      await slider(convolution.getByLabel('Convolution output index', { exact: true }), 0);
      assert.equal(await convolution.getByText('wrapped index', { exact: true }).count(), 1); await capture(page, convolution, 'wrapped-tail', width, true);
      const filter = lesson.locator('[data-transform-lab="filter"]');
      for (const initial of [-3, 0, 2]) for (const rate of [.5, 2, 20]) {
        await slider(filter.getByLabel('Initial output y(0)', { exact: true }), initial); await slider(filter.getByLabel('Decay rate a', { exact: true }), rate);
        assert.equal(await value(filter, 'At the cursor: total'), format(initial)); states += 1;
      }
      await slider(filter.getByLabel('Filter third-tone phase', { exact: true }), 0); await slider(filter.getByLabel('Response cursor', { exact: true }), 64);
      assert.equal(await value(filter, 'At the cursor: total'), format(models.filterResponse(20, 0, 2).points[64].total)); states += 1;
      await capture(page, filter, 'changed-filter', width, true);
      await slider(filter.getByLabel('Initial output y(0)', { exact: true }), -3);
      await slider(filter.getByLabel('Filter third-tone phase', { exact: true }), Math.PI / 4);
      await slider(filter.getByLabel('Response cursor', { exact: true }), 0);
      const activePhase = Number(await filter.getByLabel('Filter third-tone phase', { exact: true }).inputValue());
      assert.equal(await value(filter, 'Transient part'), format(models.filterResponse(20, activePhase, -3).correction));
      assert(models.filterResponse(20, activePhase, -3).correction < -5);
      const responseBoxes = await filter.locator('.transform-chart svg path').evaluateAll(nodes => nodes.map(node => { const box = node.getBBox(); return { y: box.y, bottom: box.y + box.height }; }));
      assert.equal(responseBoxes.length, 3);
      assert(responseBoxes.every(box => box.y >= 24 - .01 && box.bottom <= 164 + .01), JSON.stringify(responseBoxes));
      await capture(page, filter, 'expanded-transient-range', width, true); states += 1;
      const laplace = lesson.locator('[data-transform-lab="laplace"]');
      for (const side of ['left', 'right']) for (const sigma of [-3, -1, 0, 3]) for (const omega of [0, 1]) {
        await laplace.getByLabel('Exponential support', { exact: true }).selectOption(side); await slider(laplace.getByLabel('Real part sigma', { exact: true }), sigma); await slider(laplace.getByLabel('Angular frequency omega', { exact: true }), omega);
        const target = models.laplaceRegion(1, sigma, omega, 2, side);
        assert.equal(await value(laplace, 'Finite integral'), complexText(target.value)); states += 1;
      }
      await slider(laplace.getByLabel('Finite integration horizon', { exact: true }), 8);
      await laplace.getByLabel('Exponential support', { exact: true }).selectOption('left');
      await capture(page, laplace, 'outside-roc-large', width, true); states += 1;
      for (const lab of await lesson.locator('.transform-lab').all()) {
        await lab.getByRole('button', { name: /^Reset / }).focus(); await page.keyboard.press('Enter'); states += 1;
      }
      assert(Math.abs(Number(await filter.getByLabel('Decay rate a', { exact: true }).inputValue()) - 2 * Math.PI) < 1e-12);
      await alias.getByLabel('Sampling rate', { exact: true }).focus(); await page.keyboard.press('ArrowDown');
      assert.equal(await alias.getByLabel('Sampling rate', { exact: true }).inputValue(), '32'); states += 1;
      await alias.getByRole('button', { name: 'Reset sampling' }).click();
      let disclosures = 0;
      for (const details of await lesson.locator('details').all()) {
        await details.locator('summary').focus(); await page.keyboard.press('Enter'); assert.equal(await details.getAttribute('open'), ''); disclosures += 1;
      }
      await capture(page, lesson.locator('.lesson-check').last(), 'final-practice', width, true);
      const controls = await lesson.locator('button,input,select,summary').count();
      const geometry = await lesson.locator('.transform-chart svg,.transform-plane svg').evaluateAll(nodes => nodes.map((svg, index) => ({ index, width: svg.getBoundingClientRect().width, minimumText: Math.min(...[...svg.querySelectorAll('text')].map(node => parseFloat(getComputedStyle(node).fontSize) * svg.getBoundingClientRect().width / svg.viewBox.baseVal.width)), outOfBounds: [...svg.querySelectorAll('text')].map(node => ({ text: node.textContent, box: node.getBBox() })).filter(row => row.box.x < -2 || row.box.x + row.box.width > svg.viewBox.baseVal.width + 2).map(row => row.text) })));
      const documentOverflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
      const result = { width, states, disclosures, controls, programs: programs.length, anchors, equations: equations.length, overflowEquations, geometry, errors, warnings, failedRequests, documentOverflow };
      results.push(result); fs.writeFileSync(`${directory}/results.json`, JSON.stringify({ checkedAt: new Date().toISOString(), results }, null, 2));
      assert.equal(documentOverflow, false); assert.deepEqual(errors, []); assert.deepEqual(warnings, []); assert.deepEqual(failedRequests, []);
      assert.equal(overflowEquations.length, 0, JSON.stringify(overflowEquations));
      assert(geometry.every(row => row.outOfBounds.length === 0), JSON.stringify(geometry.filter(row => row.outOfBounds.length)));
      await page.close();
    }
    assert.deepEqual(fingerprints(), sourceHashes, 'Production sources changed during browser review');
    fs.writeFileSync(`${directory}/results.json`, JSON.stringify({ checkedAt: new Date().toISOString(), passed: true, sourceHashes, results }, null, 2));
    console.log(JSON.stringify(results.map(({ width, states, disclosures, controls, programs, equations }) => ({ width, states, disclosures, controls, programs, equations }))));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

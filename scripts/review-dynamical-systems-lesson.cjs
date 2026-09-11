const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const output = path.resolve('scratch/dynamical-systems-browser');
fs.mkdirSync(output, { recursive: true });
const hash = file => createHash('sha256').update(fs.readFileSync(file)).digest('hex');

(async () => {
  const { dynamicalSystemsExamples: examples } = await import('../src/learn/data/dynamical-systems-examples.js');
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1050 }, reducedMotion: 'reduce' });
      const errors = [];
      const hmrDiagnostics = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => {
        if (['error', 'warning'].includes(message.type())) {
          if (message.text().includes('[vite]')) hmrDiagnostics.push(message.text());
          else errors.push(message.text());
        }
      });
      page.on('requestfailed', request => errors.push(request.url() + ': ' + request.failure().errorText));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/dynamical-systems-theory-chaos?module=math-foundations');
      const lesson = page.locator('.dynamical-systems-lesson');
      await lesson.locator('h2').last().waitFor();
      await page.evaluate(() => document.fonts.ready);
      const anchors = await lesson.locator('.lesson-intro nav a').evaluateAll(nodes => nodes.map(node => ({ href: node.hash, present: !!document.getElementById(node.hash.slice(1)) })));
      assert.equal(anchors.length, 11);
      assert(anchors.every(anchor => anchor.present), JSON.stringify(anchors));
      assert.equal(await lesson.locator('.python-example').count(), 11);
      for (const [key, example] of Object.entries(examples)) {
        const block = lesson.locator('.python-example').filter({ has: page.getByRole('heading', { name: example.title, exact: true }) });
        const rendered = (await block.innerText()).replace(/\r\n/g, '\n');
        assert(rendered.includes(example.code.trim()), key + ' full code');
        assert(rendered.includes(example.expected.trim()), key + ' full output');
        assert(await block.evaluate(node => node.previousElementSibling.textContent.startsWith('Before running.')));
      }
      const checks = lesson.locator('.lesson-check');
      assert.equal(await checks.count(), 16);
      for (const details of await checks.locator('details').all()) {
        await details.locator('summary').click();
        assert((await details.innerText()).length > 30);
        await details.locator('summary').click();
      }
      const select = async (label, value) => lesson.getByRole('combobox', { name: label, exact: true }).selectOption(String(value));
      const slider = async (label, value) => lesson.getByRole('slider', { name: label, exact: true }).fill(String(value));
      const region = name => lesson.getByRole('region', { name, exact: true });
      const metric = async (parent, label) => {
        const values = await parent.evaluate((node, key) => [...node.querySelectorAll('.dynamics-metrics > div')].filter(item => item.querySelector('dt').textContent.startsWith(key)).map(item => item.querySelector('dd').textContent), label);
        assert.equal(values.length, 1, label);
        return values[0];
      };
      const shot = async (locator, name) => {
        await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = 'hidden'));
        await locator.screenshot({ path: path.join(output, name + '-' + width + '.png') });
        await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = ''));
      };
      let states = 0;
      const scalar = region('Scalar phase line investigation');
      for (const kind of ['pitchfork', 'tilted']) {
        await select('Scalar rule', kind);
        const values = await scalar.getByRole('combobox', { name: 'Scalar parameter' }).locator('option').evaluateAll(nodes => nodes.map(node => node.value));
        for (const parameter of values) for (const initial of [-1.8, 0, 0.2, 1.8]) {
          await select('Scalar parameter', parameter);
          await slider('Scalar initial state', initial);
          assert(!(await scalar.innerText()).includes('NaN'));
          assert((await metric(scalar, 'Equilibria')).includes('attracting'));
          states++;
        }
      }
      await select('Scalar parameter', 0);
      await shot(scalar, 'phase-line');
      await scalar.getByRole('button', { name: 'Reset phase line' }).click();
      const planar = region('Linked phase and time investigation');
      for (const mode of ['center', 'spiral', 'saddle', 'transient', 'hopf']) {
        await select('Planar system', mode);
        for (const initial of ['right', 'upper', 'near', 'origin']) {
          await select('Planar initial state', initial);
          await slider('Phase trace step', 320);
          if (initial === 'origin') assert.equal(await metric(planar, 'Radius'), '0');
          states++;
        }
      }
      await select('Planar initial state', 'near');
      for (const parameter of [-0.25, 0, 0.25, 1]) { await select('Hopf parameter', parameter); states++; }
      await select('Hopf parameter', 0.25);
      assert.equal(await metric(planar, 'Cycle radius'), '0.5');
      await shot(planar, 'hopf');
      await planar.getByRole('button', { name: 'Reset planar view' }).click();
      const logistic = region('Logistic cobweb investigation');
      for (const growth of [1, 2.5, 3, 3.2, 3.5, 3.83, 3.9, 4]) {
        await select('Logistic growth', growth);
        for (const initial of [0, 0.2, 0.5, 0.75]) {
          await select('Logistic initial state', initial);
          await logistic.getByRole('button', { name: 'Advance one update' }).click();
          const expected = growth * (growth * initial * (1 - initial)) * (1 - growth * initial * (1 - initial));
          assert(Math.abs(Number(await metric(logistic, 'Current xₙ')) - expected) < 0.00006);
          states++;
        }
      }
      await logistic.getByRole('button', { name: 'Compute finite bifurcation atlas' }).click();
      assert.equal(await logistic.locator('.dynamics-atlas').count(), 1);
      assert.equal(await logistic.locator('.dynamics-atlas').evaluate(node => (node.getAttribute('d').match(/M/g) || []).length), 151 * 48);
      await shot(logistic.locator('.dynamics-plot').last(), 'bifurcation');
      await logistic.getByRole('button', { name: 'Reset logistic' }).click();
      const sensitivity = region('Finite sensitivity investigation');
      for (const preset of ['irregular', 'stable', 'critical', 'fixed', 'equal']) {
        await select('Sensitivity case', preset);
        for (const iteration of [0, 1, 20, 70]) { await slider('Sensitivity iteration', iteration); states++; }
        if (preset === 'fixed') { assert((await metric(sensitivity, 'Selected reference / nearby')).startsWith('0.75 /')); assert.equal(await metric(sensitivity, 'Finite mean log derivative'), '0.6931'); }
        if (preset === 'critical') assert.equal(await metric(sensitivity, 'Tangent log gain'), '−∞');
        if (preset === 'equal') assert.equal(await metric(sensitivity, 'Actual separation'), '0');
      }
      await select('Sensitivity case', 'critical');
      await shot(sensitivity, 'critical-sensitivity');
      await sensitivity.getByRole('button', { name: 'Reset sensitivity' }).click();
      const folding = region('Stretch and fold coordinate investigation');
      for (let index = 2; index < 10; index++) await folding.getByRole('button', { name: index % 2 ? 'Append R' : 'Append L' }).click();
      assert(await folding.getByRole('button', { name: 'Append L' }).isDisabled());
      assert(await folding.getByRole('button', { name: 'Append R' }).isDisabled());
      for (const coordinate of [0, 0.2, 0.5, 0.75, 1]) { await slider('Tent coordinate y', coordinate); states++; }
      await folding.locator('summary').click();
      assert((await folding.locator('.dynamics-orbit').innerText()).includes('0/256'));
      await shot(folding, 'folding');
      await folding.getByRole('button', { name: 'Reset folding' }).click();
      const lorenz = region('Lorenz state projection investigation');
      for (const rho of [0.5, 10, 28]) for (const projection of ['xz', 'xy']) for (const step of [0, 2400, 4800]) {
        await select('Lorenz rho', rho); await select('Lorenz projection', projection); await slider('Lorenz trace step', step);
        assert.equal(await metric(lorenz, 'Time'), String(step * 0.005));
        if (step === 0) assert.equal(await metric(lorenz, 'Full state'), '1, 1, 1');
        states++;
      }
      await shot(lorenz, 'lorenz');
      await lorenz.getByRole('button', { name: 'Reset Lorenz' }).click();
      const integration = region('Numerical integration energy investigation');
      for (const step of [0.25, 0.5, 1, 1.5]) {
        await select('Cooling integration step', step);
        assert.equal(await metric(integration, 'Euler multiplier'), String(1 - 2 * step));
        states++;
      }
      for (const step of [0.1, 0.2, 0.25, 0.5]) {
        await select('Oscillator integration step', step);
        assert.equal(await metric(integration, 'Steps to t=20'), String(20 / step));
        assert.equal(await metric(integration, 'Final modified E'), '0.5');
        states++;
      }
      await shot(integration, 'numerical-energy');
      await integration.getByRole('button', { name: 'Reset integration' }).click();
      let keyboardControls = 0;
      await page.keyboard.press('Tab');
      for (const control of await lesson.locator('.dynamics-control input,.dynamics-control select,.dynamics-lab button:not(:disabled)').all()) {
        await control.focus();
        assert(await control.evaluate(node => node === document.activeElement));
        assert(await control.evaluate(node => getComputedStyle(node).outlineStyle !== 'none'));
        keyboardControls++;
      }
      await lesson.getByRole('slider', { name: 'Sensitivity iteration' }).focus();
      await page.keyboard.press('ArrowRight');
      assert.equal(await lesson.getByRole('slider', { name: 'Sensitivity iteration' }).inputValue(), '21');
      let keyboardScrolls = 0;
      for (const plot of await lesson.locator('.dynamics-plot').all()) {
        if (await plot.evaluate(node => node.scrollWidth > node.clientWidth + 2)) {
          await plot.focus(); await page.keyboard.press('ArrowRight'); await page.waitForTimeout(140);
          assert(await plot.evaluate(node => node.scrollLeft > 0));
          await plot.evaluate(node => node.scrollLeft = 0); keyboardScrolls++;
        }
      }
      await lesson.locator(':scope > details').evaluateAll(nodes => nodes.forEach(node => node.open = true));
      const geometry = await page.evaluate(() => ({
        documentWidth: document.documentElement.scrollWidth, width: innerWidth,
        mathErrors: document.querySelectorAll('.dynamical-systems-lesson .katex-error').length,
        equations: [...document.querySelectorAll('.dynamical-systems-lesson .katex-display')].map((node, index) => ({ index, scroll: node.scrollWidth, client: node.clientWidth, text: node.textContent.slice(0, 100) })),
        font: getComputedStyle(document.querySelector('.dynamical-systems-lesson p')).fontFamily,
        minimumPlotFont: Math.min(...[...document.querySelectorAll('.dynamical-systems-lesson svg text')].map(node => parseFloat(getComputedStyle(node).fontSize) * node.getScreenCTM().a)),
        nonfinitePaths: [...document.querySelectorAll('.dynamical-systems-lesson svg path')].filter(node => /NaN|Infinity/.test(node.getAttribute('d'))).length,
      }));
      fs.writeFileSync(path.join(output, 'geometry-' + width + '.json'), JSON.stringify(geometry, null, 2));
      assert(geometry.documentWidth <= width + 1, JSON.stringify(geometry));
      assert.equal(geometry.mathErrors, 0);
      assert(geometry.equations.every(item => item.scroll <= item.client + 3), JSON.stringify(geometry.equations));
      assert(geometry.minimumPlotFont >= 14, geometry.minimumPlotFont);
      assert.equal(geometry.nonfinitePaths, 0);
      await lesson.locator(':scope > details').evaluateAll(nodes => nodes.forEach(node => node.open = false));
      for (const index of [0, 2, 4, 7, 9]) {
        await lesson.locator('h2').nth(index).evaluate(node => window.scrollTo({ top: node.getBoundingClientRect().top + scrollY - 90, behavior: 'instant' }));
        await page.screenshot({ path: path.join(output, 'reading-' + (index + 1) + '-' + width + '.png') });
      }
      await shot(lesson.locator('.dynamics-figure').first(), 'state-figure');
      await shot(lesson.locator('.dynamics-figure').nth(1), 'stability-figure');
      await shot(lesson.locator('.dynamics-figure').nth(2), 'equilibrium-branches');
      assert.deepEqual(errors, []);
      records.push({ width, states, anchors: anchors.length, programs: 11, checks: 16, keyboardControls, keyboardScrolls, geometry, errors, hmrDiagnostics });
      console.log('Passed width', width, 'states', states);
      await page.close();
    }
    const sources = ['src/learn/data/topics/dynamical-systems-theory-chaos.jsx', 'src/learn/components/lesson-labs/DynamicalSystemsLabs.jsx', 'src/learn/components/lesson-labs/dynamical-systems-labs.css', 'src/learn/data/dynamical-systems-models.js', 'src/learn/data/dynamical-systems-examples.js'].map(file => ({ file, sha256: hash(file) }));
    fs.writeFileSync(path.join(output, 'results.json'), JSON.stringify({ verifiedAt: new Date().toISOString(), passed: true, browser: await browser.version(), sources, records }, null, 2));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

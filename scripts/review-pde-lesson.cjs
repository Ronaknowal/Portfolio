const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const directory = 'scratch/pde-browser';
fs.mkdirSync(directory, { recursive: true });
const widths = process.argv.includes('--narrow') ? [320] : [1440, 390, 320];
const paths = ['src/learn/data/topics/partial-differential-equations-conservation-boundary-conditions.jsx', 'src/learn/data/pde-models.js', 'src/learn/data/pde-examples.js', 'src/learn/components/lesson-labs/PdeLabs.jsx', 'src/learn/components/lesson-labs/pde-labs.css', 'src/learn/data/curriculum/blueprints/partial-differential-equations-conservation-boundary-conditions.js'];
const hashes = () => paths.map(path => ({ path, sha256: crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex') }));
const format = (value, digits = 4) => value === null ? 'not applicable' : Math.abs(value) > 1e5 || (value !== 0 && Math.abs(value) < 1e-5) ? value.toExponential(3) : Number(value.toFixed(digits)).toString();
async function slider(region, label, value) {
  const input = region.getByLabel(label, { exact: true });
  await input.evaluate((node, next) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(node, String(next));
    node.dispatchEvent(new Event('input', { bubbles: true }));
    node.dispatchEvent(new Event('change', { bubbles: true }));
  }, value);
  return Number(await input.inputValue());
}
async function value(region, label) {
  return region.locator('.pde-values>div').filter({ has: region.page().getByText(label, { exact: true }) }).locator('dd').innerText();
}
async function capture(page, target, name, width, whole = false) {
  await target.evaluate(node => scrollTo({ top: scrollY + node.getBoundingClientRect().top - 135, behavior: 'instant' }));
  if (whole) {
    const height = Math.max(1000, Math.ceil((await target.boundingBox()).height) + 160);
    await page.setViewportSize({ width, height });
    await target.screenshot({ path: `${directory}/${name}-${width}.png` });
    await page.setViewportSize({ width, height: 1000 });
  } else await page.screenshot({ path: `${directory}/${name}-${width}.png` });
}
async function geometry(lesson) {
  return lesson.locator('.pde-figure svg').evaluateAll(nodes => nodes.map((svg, index) => ({
    index,
    width: svg.getBoundingClientRect().width,
    minimumText: Math.min(...[...svg.querySelectorAll('text')].map(node => parseFloat(getComputedStyle(node).fontSize) * svg.getBoundingClientRect().width / svg.viewBox.baseVal.width)),
    outOfBounds: [...svg.querySelectorAll('text')].map(node => ({ text: node.textContent, box: node.getBBox() })).filter(row => row.box.x < -2 || row.box.x + row.box.width > svg.viewBox.baseVal.width + 2 || row.box.y < -2 || row.box.y + row.box.height > svg.viewBox.baseVal.height + 2).map(row => row.text),
    badCurves: [...svg.querySelectorAll('[data-pde-curve]')].map(node => node.getBBox()).filter(box => box.x < 47 - .02 || box.x + box.width > 298 + .02 || box.y < 28 - .02 || box.y + box.height > 166 + .02).map(box => ({ x: box.x, y: box.y, width: box.width, height: box.height }))
  })));
}
(async () => {
  const model = await import('../src/learn/data/pde-models.js');
  const { pdeExamples } = await import('../src/learn/data/pde-examples.js');
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const sourceHashes = hashes(), results = [];
  try {
    for (const width of widths) {
      const page = await browser.newPage({ viewport: { width, height: 1000 } });
      await page.routeWebSocket('**', socket => socket.close());
      const errors = [], warnings = [], failedRequests = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => { if (['warning', 'error'].includes(message.type()) && !message.text().startsWith('[vite]')) warnings.push(message.text()); });
      page.on('requestfailed', request => failedRequests.push(request.url()));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/partial-differential-equations-conservation-boundary-conditions?module=math-foundations', { waitUntil: 'networkidle' });
      const lesson = page.locator('.pde-lesson');
      await lesson.waitFor();
      await page.evaluate(() => document.fonts.ready);
      assert(await page.evaluate(() => document.fonts.check('16px "Space Grotesk"')));
      assert.equal(await page.locator('vite-error-overlay').count(), 0);
      assert.equal(await lesson.locator('.katex-error').count(), 0);
      const anchors = await lesson.locator('.lesson-intro a[href^="#"]').evaluateAll(nodes => nodes.map(node => ({ href: node.getAttribute('href'), exists: !!document.getElementById(node.getAttribute('href').slice(1)) })));
      assert.equal(anchors.length, 12); assert(anchors.every(row => row.exists), JSON.stringify(anchors));
      const programs = await lesson.locator('.python-example').evaluateAll(nodes => nodes.map(node => ({ title: node.querySelector('h3').textContent, question: node.previousElementSibling.textContent.replace(/^Before running\.\s*/, ''), blocks: [...node.children].filter(child => getComputedStyle(child).whiteSpace === 'pre').map(child => [...child.childNodes].filter(n => n.nodeType === Node.TEXT_NODE).map(n => n.textContent).join('')) })));
      assert.equal(programs.length, 15);
      for (const program of programs) {
        const example = Object.values(pdeExamples).find(row => row.title === program.title);
        assert(example); assert.equal(program.question, example.question);
        assert.equal(program.blocks[0].trim(), example.code.trim()); assert.equal(program.blocks[1].trim(), example.expected.trim());
      }
      const equations = await lesson.locator('.katex-display').evaluateAll(nodes => nodes.map((node, index) => ({ index, client: node.clientWidth, scroll: node.scrollWidth, text: node.textContent })));
      fs.writeFileSync(`${directory}/equations-${width}.json`, JSON.stringify(equations, null, 2));
      const overflowEquations = equations.filter(row => row.scroll > row.client + 2);
      for (let i = 0; i < 12; i++) await capture(page, lesson.locator('h2').nth(i), `reading-${i + 1}`, width);
      for (let i = 0; i < await lesson.locator('.pde-inline').count(); i++) await capture(page, lesson.locator('.pde-inline').nth(i), `inline-${i + 1}`, width, true);
      for (let i = 0; i < await lesson.locator('.pde-lab').count(); i++) await capture(page, lesson.locator('.pde-lab').nth(i), `default-${i + 1}`, width, true);
      const defaultGeometry = await geometry(lesson);
      let states = 0;
      const transport = lesson.locator('[data-pde-lab="transport"]');
      for (const curvature of [0, 2]) for (const t of [0, .2, .6, 1]) for (const x of [0, .3, 1]) {
        await transport.getByLabel('Inflow history', { exact: true }).selectOption(String(curvature));
        await slider(transport, 'Transport time', t); await slider(transport, 'Transport observation x', x);
        const state = model.transportState(t, x, curvature);
        assert.equal(await value(transport, 'Observed value'), format(state.value));
        assert.equal(await value(transport, 'Total amount'), format(state.mass)); states++;
      }
      await capture(page, transport, 'transport-inflow', width, true);
      const heat = lesson.locator('[data-pde-lab="heat"]');
      for (const t of [.002, .05, .2, .5]) for (const x of [0, .25, .5, 1]) {
        await slider(heat, 'Positive heat time theta', t); await slider(heat, 'Heat inspection position', x);
        assert.equal(await value(heat, 'Selected: fixed values'), format(model.heatValue(x, t)));
        assert.equal(await value(heat, 'Selected: insulated'), format(model.heatValue(x, t, 'neumann')));
        assert.equal(await value(heat, 'Mean: insulated'), '0.5'); states++;
      }
      await heat.getByRole('button', { name: 'Show exact initial profile' }).click();
      assert.equal(await heat.getByLabel('Positive heat time theta', { exact: true }).count(), 0);
      await slider(heat, 'Heat inspection position', .5); assert.equal(await value(heat, 'Selected: fixed values'), '1');
      await capture(page, heat, 'heat-exact-initial', width, true); states++;
      await heat.getByRole('button', { name: 'Compare at θ=0.05' }).click();
      assert.equal(await heat.getByLabel('Positive heat time theta', { exact: true }).inputValue(), '0.05'); states++;
      const wave = lesson.locator('[data-pde-lab="wave"]');
      for (const velocity of [0, .5]) for (const t of [0, .4, 1]) for (const x of [-2, .4, 2]) {
        await wave.getByLabel('Initial velocity', { exact: true }).selectOption(String(velocity));
        await slider(wave, 'Wave time', t); await slider(wave, 'Wave observation x', x);
        const state = model.waveState(t, x, velocity);
        assert.equal(await value(wave, 'Total at observation'), format(state.total));
        assert.equal(await value(wave, 'Velocity contribution'), format(state.velocityContribution)); states++;
      }
      await capture(page, wave, 'wave-finite-support', width, true);
      const poisson = lesson.locator('[data-pde-lab="poisson"]');
      for (const boundary of ['neumann', 'dirichlet']) for (const source of ['uniform', 'balanced', 'sloped']) for (const [left, right] of [[1, 1], [0, 0], [-1, 1], [2, -2]]) {
        await poisson.getByLabel('Poisson boundary type', { exact: true }).selectOption(boundary);
        await poisson.getByLabel('Poisson source', { exact: true }).selectOption(source);
        await slider(poisson, `Left ${boundary === 'neumann' ? 'outward flux' : 'value'}`, left);
        await slider(poisson, `Right ${boundary === 'neumann' ? 'outward flux' : 'value'}`, right);
        if (boundary === 'neumann') await slider(poisson, 'Selected solution mean', 1);
        const state = model.poissonState(source, boundary, left, right, 1);
        assert.equal(await poisson.getByText('No steady solution for these data.', { exact: true }).count(), state.compatible ? 0 : 1);
        if (state.compatible) assert.equal(await value(poisson, 'Solution mean'), format(state.selectedMean));
        states++;
      }
      await poisson.getByRole('button', { name: 'Reset Poisson problem' }).click();
      await slider(poisson, 'Right outward flux', 0);
      await capture(page, poisson, 'poisson-incompatible', width, true); states++;
      const harmonic = lesson.locator('[data-pde-lab="harmonic"]');
      for (const n of [1, 3, 5]) for (const x of [0, .3, .5, 1]) for (const y of [0, .5, 1]) {
        await harmonic.getByLabel('Boundary harmonic n', { exact: true }).selectOption(String(n));
        await slider(harmonic, 'Harmonic horizontal position', x); await slider(harmonic, 'Harmonic vertical position y', y);
        assert.equal(await value(harmonic, 'Selected field value'), format(model.harmonicValue(x, y, n), 7)); states++;
      }
      await slider(harmonic, 'Harmonic horizontal position', .3); await slider(harmonic, 'Harmonic vertical position y', .5);
      await capture(page, harmonic, 'harmonic-five-modes', width, true);
      const burgers = lesson.locator('[data-pde-lab="burgers"]');
      for (const direction of ['descending', 'ascending']) {
        await burgers.getByLabel('Burgers initial states', { exact: true }).selectOption(direction);
        for (const time of [0, .2, .6, 1]) {
          await slider(burgers, 'Burgers time', time);
          const state = model.burgersState(direction === 'descending' ? 2 : 0, direction === 'descending' ? 0 : 2, time);
          assert.equal(await value(burgers, 'Candidate jump entropy production'), format(state.entropyProduction)); states++;
        }
      }
      await capture(page, burgers, 'burgers-fan', width, true);
      await burgers.getByRole('checkbox').focus(); await page.keyboard.press('Space');
      assert(await burgers.getByRole('checkbox').isChecked());
      assert.equal(await value(burgers, 'Displayed state'), 'Inadmissible jump');
      await capture(page, burgers, 'burgers-expansion', width, true); states++;
      const inverse = lesson.locator('[data-pde-lab="inverse"]');
      for (const n of [1, 6, 12]) for (const t of [.002, .03, .05]) {
        await slider(inverse, 'Inverse heat mode n', n); await slider(inverse, 'Inverse observation time', t);
        assert.equal(await value(inverse, 'Required inverse gain'), format(model.inverseHeatState(n, t).gain)); states++;
      }
      await capture(page, inverse, 'inverse-small-data', width, true);
      const rod = lesson.locator('[data-pde-lab="rod"]');
      for (const length of [.25, 1, 2]) for (const amplitude of [0, .2, 1]) for (const t of [0, .6, 20]) {
        await slider(rod, 'Rod length L', length); await slider(rod, 'Initial excess amplitude', amplitude); await slider(rod, 'Physical rod time', t);
        const state = model.forcedRod(t, amplitude, .05, length);
        assert.equal(await value(rod, 'Exact uniform excess'), `${format(state.excess)} K`);
        assert.equal(await value(rod, 'Earliest required time'), `${format(state.settlingTime)} s`);
        assert.equal(await value(rod, 'Requirement now'), state.excess <= .05 ? 'Within tolerance' : 'Not yet within tolerance'); states++;
      }
      await slider(rod, 'Uniform temperature tolerance', .01);
      assert.equal(await value(rod, 'Earliest required time'), `${format(model.forcedRod(20, 1, .01, 2).settlingTime)} s`);
      await capture(page, rod, 'rod-changed-geometry', width, true); states++;
      const changedGeometry = await geometry(lesson);
      for (const lab of await lesson.locator('.pde-lab').all()) {
        await lab.getByRole('button', { name: /^Reset / }).focus(); await page.keyboard.press('Enter'); states++;
      }
      assert.equal(await value(poisson, 'Solution mean'), '0');
      await transport.getByLabel('Transport time', { exact: true }).focus(); await page.keyboard.press('ArrowRight');
      assert.equal(await transport.getByLabel('Transport time', { exact: true }).inputValue(), '0.36'); states++;
      await transport.getByRole('button', { name: /^Reset / }).click();
      let disclosures = 0;
      for (const details of await lesson.locator('details').all()) {
        await details.locator('summary').focus(); await page.keyboard.press('Enter');
        assert.equal(await details.getAttribute('open'), ''); disclosures++;
      }
      await capture(page, lesson.locator('.lesson-check').last(), 'practice-open', width, true);
      const documentOverflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth);
      const controls = await lesson.locator('button,input,select,summary').count();
      const result = { width, states, disclosures, controls, programs: programs.length, anchors, equations: equations.length, overflowEquations, defaultGeometry, changedGeometry, documentOverflow, errors, warnings, failedRequests };
      results.push(result);
      fs.writeFileSync(`${directory}/results.json`, JSON.stringify({ checkedAt: new Date().toISOString(), results }, null, 2));
      assert.equal(documentOverflow, false); assert.deepEqual(errors, []); assert.deepEqual(warnings, []); assert.deepEqual(failedRequests, []);
      assert.deepEqual(overflowEquations, []);
      assert(defaultGeometry.concat(changedGeometry).every(row => !row.outOfBounds.length && !row.badCurves.length), JSON.stringify(defaultGeometry.concat(changedGeometry).filter(row => row.outOfBounds.length || row.badCurves.length)));
      await page.close();
    }
    assert.deepEqual(hashes(), sourceHashes);
    fs.writeFileSync(`${directory}/results.json`, JSON.stringify({ checkedAt: new Date().toISOString(), passed: true, sourceHashes, results }, null, 2));
    console.log(JSON.stringify(results.map(({ width, states, disclosures, controls, programs, equations }) => ({ width, states, disclosures, controls, programs, equations }))));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

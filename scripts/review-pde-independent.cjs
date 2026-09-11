const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const directory = 'scratch/pde-independent-review/browser';
const hash = path => crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex');
fs.mkdirSync(directory, { recursive: true });

async function setRange(region, label, target) {
  const input = region.getByLabel(label, { exact: true });
  await input.evaluate((node, value) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(node, String(value));
    node.dispatchEvent(new Event('input', { bubbles: true }));
    node.dispatchEvent(new Event('change', { bubbles: true }));
  }, target);
  assert.equal(Number(await input.inputValue()), target);
}
async function metric(region, text) {
  const row = region.locator('.pde-values > div').filter({ has: region.page().getByText(text, { exact: true }) });
  return row.locator('dd').innerText();
}
const rounded = value => Math.abs(value) > 1e5 || value !== 0 && Math.abs(value) < 1e-5 ? value.toExponential(3) : Number(value.toFixed(4)).toString();

(async () => {
  const author = JSON.parse(fs.readFileSync('docs/teaching/evidence/pde-author-review.json', 'utf8'));
  const sources = author.sourceHashes.map(({ path }) => ({ path, sha256: hash(path) }));
  const { pdeExamples } = await import('../src/learn/data/pde-examples.js');
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [], images = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      const errors = [];
      page.on('pageerror', e => errors.push(e.message));
      page.on('console', m => { if (['error', 'warning'].includes(m.type()) && !m.text().startsWith('[vite]')) errors.push(m.text()); });
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/partial-differential-equations-conservation-boundary-conditions?module=math-foundations', { waitUntil: 'domcontentloaded', timeout: 60000 });
      const lesson = page.locator('.pde-lesson');
      await lesson.waitFor({ timeout: 60000 });
      await page.evaluate(() => document.fonts.ready);
      assert(await page.evaluate(() => document.fonts.check('16px "Space Grotesk"')));
      const capture = async (target, name) => {
        await target.evaluate(node => scrollTo({ top: scrollY + node.getBoundingClientRect().top - 114, behavior: 'instant' }));
        const path = `${directory}/${name}-${width}.png`;
        await page.screenshot({ path });
        images.push({ path, sha256: hash(path) });
      };
      for (const i of [0, 4, 7, 8, 10, 11]) await capture(lesson.locator('h2').nth(i), `reading-${i + 1}`);
      const anchors = await lesson.locator('.lesson-intro a[href^="#"]').evaluateAll(nodes => nodes.map(n => !!document.getElementById(n.hash.slice(1))));
      assert.equal(anchors.length, 12); assert(anchors.every(Boolean));
      assert.equal(await lesson.locator('.katex-error').count(), 0);
      const equations = await lesson.locator('.katex-display').evaluateAll(nodes => nodes.map(n => ({ width: n.clientWidth, scroll: n.scrollWidth })));
      assert.equal(equations.length, 22); assert(equations.every(r => r.scroll <= r.width + 2));
      let states = 0;

      const transport = lesson.locator('[data-pde-lab="transport"]');
      await setRange(transport, 'Transport time', .73); await setRange(transport, 'Transport observation x', .21);
      assert.equal(await metric(transport, 'Observed value'), '1.0208');
      assert.equal(await metric(transport, 'Data coordinate'), 't=0.52');
      await capture(transport.locator('.pde-two'), 'transport-inflow');
      await setRange(transport, 'Transport observation x', .91);
      assert.equal(await metric(transport, 'Observed value'), '1.18');
      await transport.getByLabel('Inflow history', { exact: true }).selectOption('0');
      assert.equal(await metric(transport, 'Observed value'), '1.18'); states += 3;
      const input = transport.getByLabel('Transport observation x', { exact: true });
      await input.focus(); await page.keyboard.press('ArrowLeft'); assert.equal(await input.inputValue(), '0.9'); states++;
      await transport.getByRole('button', { name: 'Reset transport' }).click();

      const heat = lesson.locator('[data-pde-lab="heat"]');
      await heat.getByRole('button', { name: 'Show exact initial profile' }).focus(); await page.keyboard.press('Enter');
      assert.equal(await heat.getByLabel('Positive heat time theta', { exact: true }).count(), 0);
      assert.equal(await metric(heat, 'Mean: fixed values'), '0.5'); assert.equal(await metric(heat, 'Squared norm: fixed values'), '0.375');
      await heat.getByRole('button', { name: 'Compare at θ=0.05' }).click();
      await setRange(heat, 'Positive heat time theta', .014); await setRange(heat, 'Heat inspection position', 0);
      assert.equal(await metric(heat, 'Selected: fixed values'), '0');
      assert.equal(await metric(heat, 'Selected: insulated'), rounded((1 - Math.exp(-4 * Math.PI ** 2 * .014)) / 2));
      await capture(heat.locator('.pde-values'), 'heat-boundary-meaning'); states += 3;
      await heat.getByRole('button', { name: 'Reset heat comparison' }).click();

      const wave = lesson.locator('[data-pde-lab="wave"]');
      await setRange(wave, 'Wave time', .84); await setRange(wave, 'Wave observation x', 1.65);
      assert.equal(await metric(wave, 'Initial interval'), '[0.81, 2.49]');
      await capture(wave.locator('.pde-two'), 'wave-dependence');
      await wave.getByLabel('Initial velocity', { exact: true }).selectOption('0');
      assert.equal(await metric(wave, 'Velocity contribution'), '0');
      assert.equal(await metric(wave, 'Total at observation'), rounded((1 - .81 ** 2) ** 3 / 2));
      await setRange(wave, 'Wave observation x', 2); await setRange(wave, 'Wave time', .1);
      assert.equal(await metric(wave, 'Total at observation'), '0'); states += 3;
      await wave.getByRole('button', { name: 'Reset wave' }).click();

      const poisson = lesson.locator('[data-pde-lab="poisson"]');
      await poisson.getByLabel('Poisson source', { exact: true }).selectOption('balanced');
      assert((await poisson.locator('.pde-message[role="status"]').innerText()).includes('No steady solution'));
      await capture(poisson.locator('.pde-message[role="status"]'), 'poisson-impossible');
      await setRange(poisson, 'Left outward flux', 2); await setRange(poisson, 'Right outward flux', -2); await setRange(poisson, 'Selected solution mean', 1);
      assert.equal(await metric(poisson, 'Solution mean'), '1');
      assert.equal(await metric(poisson, 'Integrated source'), '0');
      await capture(poisson.locator('.pde-figure'), 'poisson-offset-family');
      await poisson.getByLabel('Poisson boundary type', { exact: true }).selectOption('dirichlet');
      assert.equal(await poisson.getByLabel('Selected solution mean', { exact: true }).count(), 0); states += 3;
      await poisson.getByRole('button', { name: 'Reset Poisson problem' }).click();

      const harmonic = lesson.locator('[data-pde-lab="harmonic"]');
      await harmonic.getByLabel('Boundary harmonic n', { exact: true }).selectOption('5');
      await setRange(harmonic, 'Harmonic horizontal position', .3); await setRange(harmonic, 'Harmonic vertical position y', 1);
      assert.equal(await metric(harmonic, 'Selected field value'), '-1');
      await capture(harmonic.locator('.pde-two'), 'harmonic-top-negative');
      await setRange(harmonic, 'Harmonic vertical position y', 0); assert.equal(await metric(harmonic, 'Selected field value'), '0'); states += 2;
      await harmonic.getByRole('button', { name: 'Reset harmonic field' }).click();

      const burgers = lesson.locator('[data-pde-lab="burgers"]');
      await burgers.getByLabel('Burgers initial states', { exact: true }).selectOption('ascending'); await setRange(burgers, 'Burgers time', .38);
      assert.equal(await metric(burgers, 'Displayed state'), 'Rarefaction fan');
      await capture(burgers.locator('.pde-two'), 'rarefaction');
      await burgers.getByRole('checkbox').focus(); await page.keyboard.press('Space');
      assert.equal(await metric(burgers, 'Displayed state'), 'Inadmissible jump');
      assert.equal(await metric(burgers, 'Candidate jump entropy production'), '0.6667');
      await capture(burgers.locator('.pde-two'), 'expansion-gap');
      await setRange(burgers, 'Burgers time', 0); assert.equal(await metric(burgers, 'Displayed state'), 'Initial data'); states += 3;
      await burgers.getByRole('button', { name: 'Reset conservation law' }).click();

      const inverse = lesson.locator('[data-pde-lab="inverse"]');
      await setRange(inverse, 'Inverse heat mode n', 11); await setRange(inverse, 'Inverse observation time', .042);
      assert.equal(await metric(inverse, 'Natural log amplitude'), rounded(-121 * Math.PI ** 2 * .042));
      await capture(inverse.locator('.pde-values'), 'inverse-small-observation'); states++;
      await inverse.getByRole('button', { name: 'Reset inverse heat' }).click();

      const rod = lesson.locator('[data-pde-lab="rod"]');
      await setRange(rod, 'Rod length L', 1.75); await setRange(rod, 'Initial excess amplitude', .35); await setRange(rod, 'Uniform temperature tolerance', .07);
      const settling = Math.log(5) * 1.75 ** 2 / (.25 * Math.PI ** 2);
      assert.equal(await metric(rod, 'Earliest required time'), `${rounded(settling)} s`);
      await setRange(rod, 'Physical rod time', .4);
      assert.equal(await metric(rod, 'Requirement now'), 'Not yet within tolerance');
      await setRange(rod, 'Physical rod time', 2);
      assert.equal(await metric(rod, 'Requirement now'), 'Within tolerance');
      await capture(rod.locator('.pde-figure'), 'rod-changed-length'); states += 3;
      await rod.getByRole('button', { name: 'Reset forced rod' }).click();

      const programs = await lesson.locator('.python-example').evaluateAll(nodes => nodes.map(n => ({ text: n.textContent, question: n.previousElementSibling.textContent })));
      assert.equal(programs.length, 15);
      for (const e of Object.values(pdeExamples)) { const p = programs.find(n => n.text.includes(e.code.trim())); assert(p); assert(p.text.includes(e.expected.trim())); assert(p.question.includes(e.question)); }
      const selectedPractice = lesson.locator('.lesson-check').filter({ has: page.getByRole('heading', { name: '10. Reverse a different nonlinear jump', exact: true }) });
      for (const summary of await selectedPractice.locator('summary').all()) { await summary.focus(); await page.keyboard.press('Enter'); assert(await summary.evaluate(n => n.parentElement.open)); }
      assert((await selectedPractice.innerText()).includes('−16/3')); await capture(selectedPractice, 'changed-entropy-practice');
      const data = heat.locator('summary'); await data.focus(); await page.keyboard.press('Enter'); assert(await data.evaluate(n => n.parentElement.open)); states += 2;
      const bounds = await lesson.locator('svg [data-pde-curve]').evaluateAll(nodes => nodes.map(n => { const b = n.getBBox(); return { x: b.x, y: b.y, w: b.width, h: b.height }; }));
      assert(bounds.every(b => b.x >= 46.98 && b.x + b.w <= 298.02 && b.y >= 27.98 && b.y + b.h <= 166.02));
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 2), false);
      assert.deepEqual(errors, []);
      results.push({ width, actualFont: true, changedStates: states, labs: 8, actualPrograms: 15, equations: 22, anchors: 12, errors, documentOverflow: false });
      await page.close();
    }
    assert.deepEqual(sources, sources.map(({ path }) => ({ path, sha256: hash(path) })));
    const record = { checkedAt: new Date().toISOString(), status: 'passed', sourceHashes: sources, results, images, limits: 'Reviewer-owned changed-state and ordinary-reading checks, separate from the author full behavioral suite; no full screen-reader session or general beginner study.' };
    fs.writeFileSync(`${directory}/results.json`, JSON.stringify(record, null, 2));
    console.log(JSON.stringify(results, null, 2));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

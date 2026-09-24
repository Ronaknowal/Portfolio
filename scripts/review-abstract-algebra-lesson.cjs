const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const directory = 'scratch/abstract-algebra-browser';
fs.mkdirSync(directory, { recursive: true });
const sourcePaths = ['src/learn/data/topics/abstract-algebra-groups-symmetry-actions.jsx', 'src/learn/data/abstract-algebra-models.js', 'src/learn/data/abstract-algebra-examples.js', 'src/learn/components/lesson-labs/AbstractAlgebraLabs.jsx', 'src/learn/components/lesson-labs/abstract-algebra-labs.css', 'src/learn/data/curriculum/blueprints/abstract-algebra-groups-symmetry-actions.js'];
const hash = path => crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex');
const fingerprints = () => sourcePaths.map(path => ({ path, sha256: hash(path) }));

(async () => {
  const models = await import('../src/learn/data/abstract-algebra-models.js');
  const { abstractAlgebraExamples: examples } = await import('../src/learn/data/abstract-algebra-examples.js');
  const sourceBefore = fingerprints();
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [], images = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1080 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      const errors = [], failedContent = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => { if (['error', 'warning'].includes(message.type()) && !message.text().startsWith('[vite]')) errors.push(message.text()); });
      page.on('requestfailed', request => { if (request.url().includes('/src/learn/')) failedContent.push(request.url()); });
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/abstract-algebra-groups-symmetry-actions?module=math-foundations', { waitUntil: 'domcontentloaded', timeout: 60000 });
      const lesson = page.locator('.abstract-algebra-lesson');
      await lesson.waitFor({ timeout: 60000 });
      await page.evaluate(() => document.fonts.ready);
      assert(await page.evaluate(() => document.fonts.check('16px "Space Grotesk"')));
      assert.equal(await page.locator('vite-error-overlay').count(), 0);
      const capture = async (locator, name) => {
        await locator.first().evaluate(node => scrollTo({ top: scrollY + node.getBoundingClientRect().top - 115, behavior: 'instant' }));
        await page.waitForTimeout(100);
        const path = `${directory}/${name}-${width}.png`;
        await page.screenshot({ path });
        images.push({ path, sha256: hash(path), opened: false });
      };
      let states = 0;
      assert.equal(await lesson.locator('h2').count(), 11);
      assert.equal(await lesson.locator('[data-algebra-lab]').count(), 4);
      assert.deepEqual(await lesson.locator('nav a').evaluateAll(nodes => nodes.map(node => node.hash).filter(hash => !document.getElementById(hash.slice(1)))), []);
      for (let index = 0; index < 11; index += 1) await capture(lesson.locator('h2').nth(index), `reading-${index + 1}`);
      await capture(lesson.locator('.algebra-figure').nth(0), 'cayley-and-action');
      await capture(lesson.locator('.algebra-figure').nth(1), 'modular-division');
      const composition = lesson.locator('[data-algebra-lab="composition"]');
      assert((await composition.getByRole('status').innerText()).includes('vertex is 0 in the first lane and 2 in the second'));
      await capture(composition.locator('.algebra-route').first(), 'composition-default');
      await composition.getByRole('button', { name: 'Previous stage', exact: true }).focus(); await page.keyboard.press('Enter');
      assert((await composition.getByRole('status').innerText()).includes('stage 1'));
      await composition.getByRole('button', { name: 'Previous stage', exact: true }).click();
      assert(await composition.getByRole('button', { name: 'Previous stage', exact: true }).isDisabled());
      await composition.getByRole('button', { name: 'Next stage', exact: true }).click();
      await composition.getByRole('button', { name: 'Next stage', exact: true }).click();
      states += 4;
      for (const [g, h, vertex] of [[6, 3, 0], [0, 7, 3], [1, 1, 1], [4, 4, 2], [3, 5, 3]]) {
        await composition.getByLabel('Outer move g', { exact: true }).selectOption(String(g));
        await composition.getByLabel('Inner move h', { exact: true }).selectOption(String(h));
        await composition.getByLabel('Track starting vertex', { exact: true }).selectOption(String(vertex));
        const expected = models.compositionState(g, h, vertex);
        const text = await composition.getByRole('status').innerText();
        assert(text.includes(`vertex is ${expected.firstRoute[2]} in the first lane and ${expected.secondRoute[2]} in the second`));
        assert(text.includes(`gh=${models.squareElementNames[expected.product]}`));
        states += 1;
      }
      await composition.locator('summary').focus(); await page.keyboard.press('Enter');
      assert.equal(await composition.locator('tbody tr').count(), 8);
      await capture(composition.locator('details'), 'composition-table');
      await composition.getByRole('button', { name: 'Reset composition', exact: true }).click();
      states += 1;

      const orbit = lesson.locator('[data-algebra-lab="orbits"]');
      assert((await orbit.getByRole('status').innerText()).includes('2 distinct states'));
      await orbit.getByRole('button', { name: 'Adjacent marks', exact: true }).click();
      assert((await orbit.getByRole('status').innerText()).includes('4 distinct states'));
      await orbit.getByRole('button', { name: 'Constant coloring', exact: true }).click();
      assert((await orbit.getByRole('status').innerText()).includes('1 distinct states'));
      await orbit.getByRole('button', { name: 'Opposite marks', exact: true }).click();
      await orbit.getByLabel('Available colors', { exact: true }).selectOption('3');
      for (const [index, value] of [0, 0, 1, 2].entries()) await orbit.getByLabel(`Color at vertex ${index}`, { exact: true }).selectOption(String(value));
      assert((await orbit.locator('.algebra-result').innerText()).includes('21 classes'));
      assert((await orbit.getByRole('status').innerText()).includes('8 distinct states'));
      await capture(orbit.locator('.algebra-orbit-collection'), 'ternary-orbit');
      for (const g of [0, 1, 2, 4, 5]) { await orbit.getByLabel('Inspect one transformation', { exact: true }).selectOption(String(g)); states += 1; }
      await capture(orbit.locator('h4'), 'fixed-cycle-tally');
      await orbit.getByLabel('Allowed symmetries', { exact: true }).selectOption('rotations');
      assert((await orbit.locator('.algebra-result').innerText()).includes('24 classes'));
      assert((await orbit.getByRole('status').innerText()).includes('4 distinct states'));
      assert.equal(await orbit.getByLabel('Inspect one transformation', { exact: true }).locator('option').count(), 4);
      await orbit.getByLabel('Available colors', { exact: true }).selectOption('2');
      assert.equal(await orbit.getByLabel('Color at vertex 3', { exact: true }).inputValue(), '0');
      await orbit.getByRole('button', { name: 'Reset colorings', exact: true }).click();
      assert((await orbit.getByRole('status').innerText()).includes('2 distinct states'));
      states += 8;

      const cosets = lesson.locator('[data-algebra-lab="cosets"]');
      await cosets.getByLabel('Representative from H', { exact: true }).selectOption('1');
      assert((await cosets.getByRole('status').innerText()).includes('output block changed'));
      assert((await cosets.locator('.algebra-output-block').innerText()).includes('r³s'));
      await capture(cosets, 'bad-representative');
      await cosets.getByLabel('Subgroup H', { exact: true }).selectOption('rotations');
      for (const index of [0, 1, 2, 3]) { await cosets.getByLabel('Representative from H', { exact: true }).selectOption(String(index)); assert((await cosets.getByRole('status').innerText()).includes('same output block')); states += 1; }
      await capture(cosets, 'normal-representative');
      await cosets.getByRole('button', { name: 'Reset representatives', exact: true }).click();
      states += 2;

      const map = lesson.locator('[data-algebra-lab="equivariance"]');
      await map.getByLabel('Processing map', { exact: true }).selectOption('shift');
      assert((await map.getByRole('status').innerText()).includes('Current-input defect: 6 V'));
      await map.getByLabel('Data transformation g', { exact: true }).selectOption('1');
      assert((await map.getByRole('status').innerText()).includes('Current-input defect: 0 V'));
      assert((await map.getByRole('status').innerText()).includes('transformations: 1.'));
      await map.getByLabel('Sensor readings', { exact: true }).fill('1,1,1,1');
      await map.getByRole('button', { name: 'Apply readings', exact: true }).focus(); await page.keyboard.press('Enter');
      await map.getByLabel('Data transformation g', { exact: true }).selectOption('4');
      assert((await map.getByRole('status').innerText()).includes('Current-input defect: 0 V'));
      const preserved = await map.getByRole('status').innerText();
      for (const bad of ['1,2,,4', '1,2,3,0.1', '1,2,3,100']) { await map.getByLabel('Sensor readings', { exact: true }).fill(bad); await map.getByRole('button', { name: 'Apply readings', exact: true }).click(); assert(await map.getByRole('alert').isVisible()); assert.equal(await map.getByRole('status').innerText(), preserved); states += 1; }
      await capture(map.locator('form'), 'invalid-preserved-input');
      await map.getByLabel('Sensor readings', { exact: true }).fill('-1, 0, 2, 3');
      await map.getByRole('button', { name: 'Apply readings', exact: true }).click();
      assert.equal(await map.getByRole('alert').count(), 0);
      for (const mode of ['raw', 'rotations', 'averaged', 'tied']) {
        await map.getByLabel('Processing map', { exact: true }).selectOption(mode);
        const expected = models.equivarianceState([-1, 0, 2, 3], mode, 4);
        assert((await map.getByRole('status').innerText()).includes(`Current-input defect: ${expected.inputSpecificDefect} V`));
        assert((await map.getByRole('status').innerText()).includes(`transformations: ${expected.allInputDefect}.`));
        states += 1;
      }
      for (const [name, value] of [['Self weight', -1], ['Neighbor weight', 1.25], ['Opposite weight', 0.75]]) await map.getByRole('slider', { name, exact: true }).fill(String(value));
      await map.getByRole('slider', { name: 'Self weight', exact: true }).focus(); await page.keyboard.press('ArrowRight');
      assert.equal(await map.getByRole('slider', { name: 'Self weight', exact: true }).inputValue(), '-0.75');
      assert((await map.getByRole('status').innerText()).includes('transformations: 0.'));
      await capture(map.locator('.algebra-route').first(), 'changed-sensor-route');
      await map.locator('summary').focus(); await page.keyboard.press('Enter');
      assert.equal(await map.locator('details tbody tr').count(), 8);
      await capture(map.locator('details'), 'all-input-certificate');
      await map.getByRole('button', { name: 'Reset sensor map', exact: true }).click();
      assert.equal(await map.getByLabel('Sensor readings', { exact: true }).inputValue(), '1, 2, 4, 8');
      await map.getByLabel('Processing map', { exact: true }).selectOption('averaged');
      await capture(map.locator('.algebra-route').first(), 'averaged-sensor-route');
      const numericGeometry = await map.locator('.algebra-square svg g').evaluateAll(nodes => nodes.map(node => {
        const rect = node.querySelector('rect').getBBox();
        const value = node.querySelector('text').getBBox();
        const position = node.querySelector('.algebra-position').getBBox();
        return { valueFits: value.x >= rect.x + 1 && value.x + value.width <= rect.x + rect.width - 1, labelsSeparated: position.y + position.height <= value.y || position.y >= value.y + value.height };
      }));
      assert(numericGeometry.every(row => row.valueFits && row.labelsSeparated), JSON.stringify(numericGeometry));
      await map.getByLabel('Sensor readings', { exact: true }).fill('20,20,20,20');
      await map.getByRole('button', { name: 'Apply readings', exact: true }).click();
      await map.getByLabel('Processing map', { exact: true }).selectOption('raw');
      assert((await map.getByRole('status').innerText()).includes('transformations: 14.'));
      await map.getByRole('button', { name: 'Reset sensor map', exact: true }).click();
      states += 8;

      for (const summary of await lesson.locator('.lesson-check summary').all()) { await summary.focus(); await page.keyboard.press('Enter'); assert(await summary.evaluate(node => node.parentElement.open)); assert((await summary.evaluate(node => node.nextElementSibling.textContent)).trim().length > 50); }
      const checks = lesson.locator('.lesson-check');
      assert.equal(await checks.count(), 10);
      const changedPractice = checks.filter({ has: page.getByRole('heading', { name: 'H. Produce a changed sensor-map audit', exact: true }) });
      assert((await changedPractice.innerText()).includes('141/4'));
      await capture(changedPractice, 'changed-practice');
      const deep = lesson.locator('details').filter({ has: page.locator('summary').filter({ hasText: 'Deeper: why this is the nearest tied matrix' }) });
      await deep.locator('summary').click();
      await capture(deep, 'projection-proof');
      const actualPrograms = await lesson.locator('.python-example').evaluateAll(nodes => nodes.map(node => ({ title: node.querySelector('h3').textContent, question: node.previousElementSibling.textContent.replace(/^Before running:\s*/, ''), blocks: [...node.children].filter(child => getComputedStyle(child).whiteSpace === 'pre').map(child => [...child.childNodes].filter(n => n.nodeType === Node.TEXT_NODE).map(n => n.textContent).join('')) })));
      assert.equal(actualPrograms.length, examples.length);
      for (const actual of actualPrograms) { const example = examples.find(e => e.title === actual.title); assert.equal(actual.question, example.question); assert.equal(actual.blocks[0].trim(), example.code.trim()); assert.equal(actual.blocks[1].trim(), example.expected.trim()); }
      await capture(lesson.locator('.python-example').last(), 'actual-program');
      await capture(lesson.locator('.lesson-sources'), 'references');
      if (width < 600) {
        const scroll = lesson.getByRole('region', { name: 'Eight-element Cayley graph; scroll horizontally on a narrow screen', exact: true });
        await scroll.focus(); await page.keyboard.press('ArrowRight'); await page.waitForTimeout(140);
        assert(await scroll.evaluate(node => node.scrollLeft > 0));
        await scroll.evaluate(node => { node.scrollLeft = node.scrollWidth - node.clientWidth; });
        await capture(scroll, 'cayley-scrolled');
      }
      const equations = await lesson.locator('.katex-display').evaluateAll(nodes => nodes.map(node => ({ client: node.clientWidth, scroll: node.scrollWidth })));
      assert(equations.every(row => row.scroll <= row.client + 2), JSON.stringify(equations));
      assert.equal(await lesson.locator('.katex-error').count(), 0);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 2), false);
      assert.deepEqual(errors, []); assert.deepEqual(failedContent, []);
      results.push({ width, publicFont: true, states, actualPrograms: actualPrograms.length, sectionAnchors: 11, checkpointAndPracticeBlocks: 10, equations: equations.length, errors, failedContent, documentOverflow: false });
      await page.close();
    }
    assert.deepEqual(fingerprints(), sourceBefore);
    fs.writeFileSync(`${directory}/results.json`, JSON.stringify({ checkedAt: new Date().toISOString(), status: 'passed', productionSources: sourceBefore, results, images }, null, 2));
    console.log(JSON.stringify(results, null, 2));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const output = path.resolve('scratch/category-theory-browser');
fs.mkdirSync(output, { recursive: true });
const hash = file => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const format = value => Number(value.toPrecision(7)).toString();

(async () => {
  const model = await import('../src/learn/data/category-theory-models.js');
  const { categoryTheoryExamples: examples } = await import('../src/learn/data/category-theory-examples.js');
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1050 }, reducedMotion: 'reduce' });
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => {
        if (['error', 'warning'].includes(message.type()) && !message.text().startsWith('[vite]')) errors.push(message.text());
      });
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/category-theory-emerging-use-in-ml?module=math-foundations', { waitUntil: 'domcontentloaded', timeout: 60000 });
      const lesson = page.locator('.category-theory-lesson');
      await lesson.locator('h2').last().waitFor();
      await page.evaluate(() => document.fonts.ready);
      const anchors = await lesson.locator('.lesson-intro nav a').evaluateAll(nodes => nodes.map(node => ({ hash: node.hash, exists: !!document.getElementById(node.hash.slice(1)) })));
      assert.equal(anchors.length, 12);
      assert(anchors.every(anchor => anchor.exists));
      assert.equal(await lesson.locator('.category-lab').count(), 7);
      assert.equal(await lesson.locator('.category-figure').count(), 3);
      for (const [key, example] of Object.entries(examples)) {
        const block = lesson.locator('.python-example').filter({ has: page.getByRole('heading', { name: example.title, exact: true }) });
        const text = (await block.innerText()).replace(/\r\n/g, '\n');
        assert(text.includes(example.code.trim()), key + ' complete code');
        assert(text.includes(example.expected), key + ' actual output');
        assert(await block.evaluate(node => node.previousElementSibling.textContent.startsWith('Before running.')));
      }
      assert.equal(await lesson.locator('.lesson-check').count(), 14);
      for (const disclosure of await lesson.locator('.lesson-check details').all()) {
        await disclosure.locator('summary').click();
        assert((await disclosure.innerText()).length > 35);
        await disclosure.locator('summary').click();
      }
      const select = (label, value) => lesson.getByRole('combobox', { name: label, exact: true }).selectOption(String(value));
      const slider = (label, value) => lesson.getByRole('slider', { name: label, exact: true }).fill(String(value));
      const region = name => lesson.getByRole('region', { name, exact: true });
      async function shot(locator, name) {
        await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = 'hidden'));
        await locator.screenshot({ path: path.join(output, name + '-' + width + '.png') });
        await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = ''));
      }
      let states = 0;
      // Record defaults before manipulating them: ordinary reading matters.
      for (let index = 0; index < 7; index++) await shot(lesson.locator('.category-lab').nth(index), 'default-lab-' + index);
      for (let index = 0; index < 3; index++) await shot(lesson.locator('.category-figure').nth(index), 'inline-' + index);
      const composition = region('Typed composition investigation');
      for (const name of Object.keys(model.finiteFunctionChoices)) {
        await select('f: A → B', name);
        for (let input = 0; input < 3; input++) for (const group of ['first', 'last']) {
          await select('Starting element in A', input);
          await select('Group the calculation', group);
          const expected = model.compositionSnapshot(input, name);
          assert.equal(await composition.locator('.category-active-node').count(), 4);
          assert.equal(await composition.locator('.category-active-edge').count(), 3);
          assert((await composition.locator('.category-result').innerText()).includes('sends ' + input + ' to ' + expected.path[3]));
          states++;
        }
      }
      await composition.getByRole('button', { name: 'Reset investigation' }).click();
      const schema = region('Schema functor investigation');
      for (let mask = 0; mask < 8; mask++) {
        for (let sensor = 0; sensor < 3; sensor++) {
          await select('Inspect sensor', sensor);
          await select('Stored direct site for s' + sensor, (mask >> sensor) & 1);
        }
        const direct = [0, 1, 2].map(i => (mask >> i) & 1);
        const expected = model.schemaSnapshot(2, direct);
        const text = await schema.locator('.category-result').innerText();
        assert(expected.violations.length ? expected.violations.every(i => text.includes('s' + i)) : text.includes('Every sensor'));
        states++;
      }
      await schema.getByRole('button', { name: 'Repair direct sites from devices' }).click();
      assert((await schema.locator('.category-result').innerText()).includes('Every sensor'));
      await shot(schema, 'repaired-schema');
      await schema.getByRole('button', { name: 'Reset investigation' }).click();
      const natural = region('Naturality square investigation');
      const lists = { distinct: [2, 0, 1], repeated: [2, 0, 2, 1], empty: [], singleton: [2] };
      for (const [list, values] of Object.entries(lists)) for (const operation of ['reverse', 'sort']) for (const mapping of Object.keys(model.finiteFunctionChoices)) {
        await select('Input list', list);
        await select('Proposed list transformation', operation);
        await select('Element function', mapping);
        const expected = model.naturalitySnapshot(values, operation, mapping);
        const arrays = await natural.locator('.category-tokens').evaluateAll(nodes => nodes.map(node => node.getAttribute('aria-label')));
        assert.equal(arrays[3], expected.topThenRight.length ? expected.topThenRight.join(', ') : 'empty list');
        assert.equal(arrays[4], expected.leftThenBottom.length ? expected.leftThenBottom.join(', ') : 'empty list');
        assert((await natural.locator('.category-result').innerText()).includes(expected.commutesHere ? 'agree' : 'disagree'));
        states++;
      }
      await select('Input list', 'distinct'); await select('Proposed list transformation', 'sort'); await select('Element function', 'reverse');
      await shot(natural, 'naturality-counterexample');
      await natural.getByRole('button', { name: 'Reset investigation' }).click();
      const product = region('Universal product investigation');
      for (const mode of ['complete', 'missing', 'duplicate']) {
        await select('Candidate object', mode);
        for (const a of [0, 1]) for (const b of [0, 1]) {
          await product.getByRole('button', { name: 'Choose pair (' + a + ', ' + b + ')' }).click();
          const expected = model.productSnapshot(mode, a, b);
          const text = await product.locator('.category-result').innerText();
          assert(text.includes(expected.mediators.length === 1 ? 'one mediator' : expected.mediators.length + ' mediators'));
          states++;
        }
      }
      await product.getByRole('button', { name: 'Choose pair (0, 0)' }).click();
      await shot(product, 'nonunique-mediator');
      await product.getByRole('button', { name: 'Reset investigation' }).click();
      const probability = region('Stochastic copying investigation');
      for (const percent of [0, 25, 50, 75, 100]) {
        await slider('Chance of bit 1, percent', percent);
        const expected = model.stochasticCopySnapshot(percent);
        const actual = await probability.locator('.category-joint tbody td span').allTextContents();
        assert.deepEqual(actual, [...expected.copied.flat(), ...expected.independent.flat()].map(format));
        states++;
      }
      await probability.getByRole('button', { name: 'Reset investigation' }).click();
      const tangent = region('Tangent composition investigation');
      for (const choice of Object.keys(model.tangentFunctionChoices)) for (const input of [-3, -1, 0, 0.1, 1, 2, 3]) for (const v of [-2, 0, 2]) {
        await select('Polynomial composition', choice);
        await slider('Input x', input); await slider('Input tangent v', v);
        const expected = model.tangentSnapshot(input, v, choice, 1);
        const actual = await tangent.locator('.category-tangent strong').allTextContents();
        assert.deepEqual(actual, [v, expected.intermediateTangent, expected.outputTangent].map(format));
        states++;
      }
      await tangent.getByRole('button', { name: 'Reset investigation' }).click();
      const adjunction = region('Image preimage adjunction investigation');
      for (let s = 0; s < 16; s++) for (let t = 0; t < 8; t++) {
        for (let i = 0; i < 4; i++) await adjunction.getByRole('checkbox', { name: 'Include source ' + 'abcd'[i] }).setChecked(!!(s >> i & 1));
        for (let i = 0; i < 3; i++) await adjunction.getByRole('checkbox', { name: 'Include target ' + i }).setChecked(!!(t >> i & 1));
        const expected = model.adjunctionSnapshot(s, t);
        const actual = await adjunction.locator('.category-metrics dd').allTextContents();
        assert.equal(actual[0], '{' + expected.image.join(', ') + '}');
        assert.equal(actual[1], '{' + expected.preimage.map(i => 'abcd'[i]).join(', ') + '}');
        assert((await adjunction.locator('.category-result').innerText()).includes('is ' + String(expected.imageContained)));
        states++;
      }
      await adjunction.getByRole('button', { name: 'Reset investigation' }).click();
      let keyboardControls = 0;
      await page.keyboard.press('Tab');
      for (const control of await lesson.locator('.category-lab :is(input,select,button)').all()) {
        await control.focus();
        assert(await control.evaluate(node => document.activeElement === node && getComputedStyle(node).outlineStyle !== 'none'));
        keyboardControls++;
      }
      const chance = probability.getByRole('slider');
      await chance.focus(); await page.keyboard.press('ArrowRight'); assert.equal(await chance.inputValue(), '51');
      await probability.getByRole('button', { name: 'Reset investigation' }).click();
      const sourceA = adjunction.getByRole('checkbox', { name: 'Include source a' });
      await sourceA.focus(); await page.keyboard.press('Space'); assert.equal(await sourceA.isChecked(), false);
      await adjunction.getByRole('button', { name: 'Reset investigation' }).click();
      await lesson.locator(':scope > details').evaluateAll(nodes => nodes.forEach(node => node.open = true));
      const geometry = await lesson.evaluate(node => ({
        width: innerWidth, documentWidth: document.documentElement.scrollWidth,
        equations: [...node.querySelectorAll('.katex-display')].map((equation, index) => ({ index, width: equation.clientWidth, scroll: equation.scrollWidth })),
        mathErrors: node.querySelectorAll('.katex-error').length,
        minimumSvgFont: Math.min(...[...node.querySelectorAll('svg text')].map(text => parseFloat(getComputedStyle(text).fontSize) * text.getScreenCTM().a)),
        nonfinitePaths: [...node.querySelectorAll('svg path')].filter(path => /NaN|Infinity/.test(path.getAttribute('d'))).length,
        loadedFonts: [...document.fonts].filter(font => font.status === 'loaded').map(font => font.family),
      }));
      fs.writeFileSync(path.join(output, 'geometry-' + width + '.json'), JSON.stringify(geometry, null, 2));
      assert(geometry.documentWidth <= width + 1, JSON.stringify(geometry));
      assert(geometry.equations.every(eq => eq.scroll <= eq.width + 2), JSON.stringify(geometry.equations));
      assert.equal(geometry.mathErrors, 0); assert.equal(geometry.nonfinitePaths, 0);
      assert(geometry.minimumSvgFont >= 14, geometry.minimumSvgFont);
      assert(geometry.loadedFonts.some(font => font.includes('Space Grotesk')));
      await lesson.locator(':scope > details').evaluateAll(nodes => nodes.forEach(node => node.open = false));
      for (const index of [0, 2, 4, 6, 7, 9, 11]) {
        await lesson.locator('h2').nth(index).evaluate(node => window.scrollTo({ top: node.getBoundingClientRect().top + scrollY - 90, behavior: 'instant' }));
        await page.screenshot({ path: path.join(output, 'reading-' + (index + 1) + '-' + width + '.png') });
      }
      assert((await lesson.innerText()).includes('Next in this module: Differential Geometry'));
      assert.deepEqual(errors, []);
      records.push({ width, states, programs: Object.keys(examples).length, practiceAndCheckpoints: 14, anchors, keyboardControls, geometry, errors });
      console.log('Passed', width, 'with', states, 'checked states');
      await page.close();
    }
    const files = ['src/learn/data/topics/category-theory-emerging-use-in-ml.jsx', 'src/learn/data/category-theory-models.js', 'src/learn/data/category-theory-examples.js', 'src/learn/components/lesson-labs/CategoryTheoryLabs.jsx', 'src/learn/components/lesson-labs/category-theory-labs.css'];
    fs.writeFileSync(path.join(output, 'results.json'), JSON.stringify({ passed: true, verifiedAt: new Date().toISOString(), browser: await browser.version(), sources: files.map(file => ({ file, sha256: hash(file) })), records }, null, 2) + '\n');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const output = path.resolve('scratch/topology-tda-browser');
fs.mkdirSync(output, { recursive: true });
const hash = file => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const format = value => value === Infinity ? '∞' : Number(value.toPrecision(6)).toString();

(async () => {
  const model = await import('../src/learn/data/topology-tda-models.js');
  const { topologyTdaExamples: examples } = await import('../src/learn/data/topology-tda-examples.js');
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
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/topology-topological-data-analysis-tda?module=math-foundations');
      const lesson = page.locator('.topology-tda-lesson');
      await lesson.locator('h2').last().waitFor();
      await page.evaluate(() => document.fonts.ready);
      const anchors = await lesson.locator('.lesson-intro nav a').evaluateAll(nodes => nodes.map(node => ({ hash: node.hash, exists: !!document.getElementById(node.hash.slice(1)) })));
      assert.equal(anchors.length, 13);
      assert(anchors.every(anchor => anchor.exists));
      assert.equal(await lesson.locator('.tda-lab').count(), 7);
      assert.equal(await lesson.locator('.tda-figure').count(), 3);
      for (const [key, example] of Object.entries(examples)) {
        const block = lesson.locator('.python-example').filter({ has: page.getByRole('heading', { name: example.title, exact: true }) });
        const text = (await block.innerText()).replace(/\r\n/g, '\n');
        assert(text.includes(example.code.trim()), key + ' complete code');
        assert(text.includes(example.expected), key + ' actual output');
        assert(await block.evaluate(node => node.previousElementSibling.textContent.startsWith('Before running.')));
      }
      assert.equal(await lesson.locator('.lesson-check').count(), 15);
      for (const disclosure of await lesson.locator('.lesson-check details').all()) {
        await disclosure.locator('summary').click();
        assert((await disclosure.innerText()).length > 40);
        await disclosure.locator('summary').click();
      }
      const region = name => lesson.getByRole('region', { name, exact: true });
      const select = (label, value) => lesson.getByRole('combobox', { name: label, exact: true }).selectOption(String(value));
      const slider = (label, value) => lesson.getByRole('slider', { name: label, exact: true }).fill(String(value));
      async function metric(parent, label) {
        const values = await parent.evaluate((node, name) => [...node.querySelectorAll('.tda-metrics > div')].filter(item => item.querySelector('dt').textContent === name).map(item => item.querySelector('dd').textContent), label);
        assert.equal(values.length, 1, label);
        return values[0];
      }
      async function shot(locator, name) {
        await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = 'hidden'));
        await locator.screenshot({ path: path.join(output, name + '-' + width + '.png') });
        await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = ''));
      }
      let states = 0;
      const chains = region('Chains and fillings investigation');
      for (const kind of Object.keys(model.TOPOLOGY_COMPLEX_FIXTURES)) {
        await select('Choose the complex', kind);
        const fixture = model.TOPOLOGY_COMPLEX_FIXTURES[kind];
        const complex = model.complexFromFacets(fixture.facets);
        const selectable = complex.filter(simplex => simplex.dimension === fixture.chainDimension);
        const keys = selectable.map(simplex => simplex.key);
        const expected = model.chainBoundary(complex, keys);
        assert.equal(await metric(chains, 'Cycle?'), expected.isCycle ? 'Yes: boundary is zero' : 'No');
        await chains.getByRole('button', { name: 'Clear chain', exact: true }).click();
        assert.equal(await metric(chains, 'Selected chain'), '0 (empty chain)');
        assert.equal(await metric(chains, 'Boundary of a higher chain?'), 'Yes: class is zero');
        for (const simplex of selectable) {
          await chains.getByRole('button', { name: '{' + simplex.vertices.join(',') + '}', exact: true }).click();
          states += 1;
        }
        assert.equal(await metric(chains, 'Cycle?'), expected.isCycle ? 'Yes: boundary is zero' : 'No');
        if (kind === 'shell') await shot(chains, 'tetrahedral-shell');
        states += 2;
      }
      await chains.getByRole('button', { name: 'Reset', exact: true }).click();
      await chains.getByRole('button', { name: '{0,2}', exact: true }).click();
      assert.equal(await metric(chains, 'Its boundary'), '{0} + {2}');
      await shot(chains, 'chain-boundary');
      await chains.getByRole('button', { name: 'Reset', exact: true }).click();

      const rips = region('Rips filtration and persistence investigation');
      for (const [kind, fixture] of Object.entries(model.TOPOLOGY_POINT_FIXTURES)) {
        for (const scale of [0.5, 1, 2]) {
          await select('Point sample', kind);
          await select('Uniform coordinate scale', scale);
          const filtration = model.ripsFiltration(fixture.points.map(point => point.map(value => value * scale)));
          const bars = model.persistentHomology(filtration).intervals.filter(bar => bar.dimension === 1 && bar.death > bar.birth);
          for (const [button, threshold] of [['Vertices', 0], ['Exact H₁ birth', bars[0].birth], ['Exact H₁ death', bars[0].death], ['Full 2-skeleton', Math.max(...filtration.map(simplex => simplex.birth))]]) {
            await rips.getByRole('button', { name: button, exact: true }).click();
            const expected = model.bettiAt(filtration, threshold);
            assert.equal(await metric(rips, 'Components β₀'), String(expected.betti[0]));
            assert.equal(await metric(rips, 'Unfilled loop classes β₁'), String(expected.betti[1]));
            const sliderValue = Number(await rips.getByRole('slider').inputValue());
            // Edge serializes range values to fewer digits than binary64 can
            // distinguish. The lab's computed event remains unrounded.
            assert(Math.abs(sliderValue - threshold) <= 1e-12 * Math.max(1, threshold), 'Event agrees with range serialization precision');
            states += 1;
          }
        }
      }
      await rips.getByRole('button', { name: 'Reset', exact: true }).click();
      await shot(rips, 'square-persistence');
      await rips.getByRole('button', { name: 'Exact H₁ death', exact: true }).click();
      await shot(rips, 'square-filled');
      await rips.getByRole('button', { name: 'Reset', exact: true }).click();

      const reduction = region('Persistent boundary reduction investigation');
      for (const kind of ['triangle', 'square']) {
        await select('Reduction fixture', kind);
        const filtration = kind === 'triangle' ? model.complexFromFacets([[0, 1, 2]]) : model.ripsFiltration(model.TOPOLOGY_POINT_FIXTURES.square.points);
        const result = model.persistentHomology(filtration, { captureTrace: true });
        const frames = result.trace.flatMap(step => step.stages);
        for (let index = 0; index < frames.length; index += 1) {
          const expectedBoundary = frames[index].boundary.map(row => '{' + result.ordered[row].vertices.join(',') + '}').join(' + ') || '0';
          assert.equal(await metric(reduction, 'Its boundary'), expectedBoundary);
          if (index === 6 && kind === 'triangle') await shot(reduction, 'boundary-xor');
          if (index < frames.length - 1) await reduction.getByRole('button', { name: 'Next operation', exact: true }).click();
          states += 1;
        }
        assert(await reduction.getByRole('button', { name: 'Next operation', exact: true }).isDisabled());
        await reduction.getByRole('button', { name: 'Back', exact: true }).click();
        await reduction.getByRole('button', { name: 'Reset', exact: true }).click();
        assert(await reduction.getByRole('button', { name: 'Back', exact: true }).isDisabled());
      }
      await select('Reduction fixture', 'triangle');
      await reduction.getByRole('button', { name: 'Jump to first XOR', exact: true }).click();
      await shot(reduction, 'reduction-matrix');

      const matching = region('Diagram matching investigation');
      const fixtures = { close: [[[1, 4]], [[1.2, 3.8]]], greedy: [[[2, 10], [4, 10]], [[3, 10], [0, 10]]], copies: [[[1, 4], [1, 4]], [[1, 4]]], empty: [[[1, 4]], []] };
      for (const [kind, [first, second]] of Object.entries(fixtures)) {
        for (const power of [Infinity, 1, 2]) {
          await select('Diagram pair', kind);
          await select('Cost aggregation', power);
          const assignments = first.map((_, index) => index < second.length ? index : null);
          assert.equal(await metric(matching, 'Your matching cost'), format(model.evaluateDiagramMatching(first, second, assignments, power).value));
          await matching.getByRole('button', { name: 'Use an optimal matching', exact: true }).click();
          assert.equal(await metric(matching, 'Your matching cost'), format(model.optimalDiagramMatching(first, second, power).value));
          states += 2;
        }
      }
      await select('Diagram pair', 'greedy');
      await select('Cost aggregation', Infinity);
      await matching.getByRole('button', { name: 'Use an optimal matching', exact: true }).click();
      await shot(matching, 'matching-optimum');
      await select('Match A1 (4, 10)', 1);
      assert.equal(await matching.getByRole('combobox', { name: 'Match A0 (2, 10)', exact: true }).inputValue(), 'diagonal');
      await matching.getByRole('button', { name: 'Reset', exact: true }).click();

      const pixels = region('Cubical image filtration investigation');
      const patterns = { ring: [1, 1, 1, 1, 3, 1, 1, 1, 1], gap: [1, 3, 1, 1, 3, 1, 1, 1, 1], corner: [1, 3, 3, 3, 1, 3, 3, 3, 3] };
      for (const [kind, values] of Object.entries(patterns)) {
        await select('Image starting pattern', kind);
        for (const threshold of [0, 1, 2, 3]) {
          await slider('Include pixels with value at most', threshold);
          const expected = model.pixelComplex(values, threshold);
          assert.equal(await metric(pixels, 'Vertices V'), String(expected.counts[0]));
          assert.equal(await metric(pixels, 'Holes β₁ = β₀ − Euler'), String(expected.betti[1]));
          states += 1;
        }
      }
      await pixels.getByRole('button', { name: 'Reset', exact: true }).click();
      await shot(pixels, 'pixel-ring');
      await pixels.getByRole('button', { name: /Pixel row 2, column 2,/ }).click();
      assert.equal(await metric(pixels, 'Holes β₁ = β₀ − Euler'), '0');
      await pixels.getByRole('button', { name: 'Reset', exact: true }).click();

      const features = region('Persistence landscapes and images investigation');
      const featureCases = { overlap: [[0, 3], [1, 4]], single: [[0, 3]], short: [[1, 1.2]], empty: [] };
      for (const [kind, diagram] of Object.entries(featureCases)) {
        await select('Finite bars', kind);
        for (const bandwidth of [0.1, 0.5, 1.5]) {
          await slider('Gaussian bandwidth σ', bandwidth);
          const expected = model.persistenceImage(diagram, { bandwidth });
          for (const pixel of expected.pixels) {
            await features.getByRole('button', { name: new RegExp('^Select pixel birth ' + pixel.column + ' to ' + (pixel.column + 1) + ', persistence ' + pixel.row + ' to') }).click();
            assert.equal(await metric(features, 'Selected pixel sum'), format(pixel.value));
          }
          for (const time of [0, 1.5, 2, 4]) {
            await slider('Inspect landscape coordinate t', time);
            assert((await features.innerText()).includes('λ₁=' + format(model.landscapeAt(diagram, time, 1).value)));
            states += 1;
          }
          states += 16;
        }
      }
      await features.getByRole('button', { name: 'Reset', exact: true }).click();
      await shot(features, 'landscape-image');

      const mapper = region('Mapper lens and overlap investigation');
      for (const intervalCount of [2, 3, 4, 5]) for (const overlap of [0.1, 0.4, 0.65]) for (const clusterDistance of [0.1, 0.6, 1.8]) {
        await select('Number of cover intervals', intervalCount);
        await select('Overlap fraction of interval width', overlap);
        await select('Within-band cluster edge threshold', clusterDistance);
        const expected = model.mapperGraph({ intervalCount, overlap, clusterDistance });
        assert.equal(await metric(mapper, 'Mapper nodes / overlap edges'), expected.nodes.length + ' / ' + expected.edges.length);
        assert.equal(await metric(mapper, 'Focused node IDs'), expected.membership[3].join(', '));
        await mapper.locator('.tda-member-list button').first().click();
        assert.equal(await metric(mapper, 'Focused observations'), expected.nodes[0].members.join(', '));
        states += 2;
      }
      await mapper.getByRole('button', { name: 'Reset', exact: true }).click();
      await shot(mapper, 'mapper-loop');
      await select('Within-band cluster edge threshold', 1.8);
      await shot(mapper, 'mapper-chain');
      await mapper.getByRole('button', { name: 'Reset', exact: true }).click();
      await select('Number of cover intervals', 3);
      await select('Overlap fraction of interval width', 0.65);
      assert((await mapper.innerText()).includes('higher-dimensional nerve simplices'));
      await shot(mapper, 'mapper-higher-nerve');
      await mapper.getByRole('button', { name: 'Reset', exact: true }).click();

      let keyboardControls = 0;
      await page.keyboard.press('Tab');
      for (const control of await lesson.locator('.tda-control input,.tda-control select,.tda-lab button:not(:disabled)').all()) {
        await control.focus();
        assert(await control.evaluate(node => node === document.activeElement));
        assert(await control.evaluate(node => getComputedStyle(node).outlineStyle !== 'none'));
        keyboardControls += 1;
      }
      const imageThreshold = lesson.getByRole('slider', { name: 'Include pixels with value at most', exact: true });
      await imageThreshold.focus();
      await page.keyboard.press('ArrowRight');
      assert.equal(await imageThreshold.inputValue(), '2');
      let keyboardScrolls = 0;
      for (const container of await lesson.locator('.tda-matrix-scroll').all()) {
        if (await container.evaluate(node => node.scrollWidth > node.clientWidth + 2)) {
          await container.focus();
          await page.keyboard.press('ArrowRight');
          await page.waitForTimeout(150);
          assert(await container.evaluate(node => node.scrollLeft > 0));
          keyboardScrolls += 1;
          await container.evaluate(node => node.scrollLeft = 0);
        }
      }
      await lesson.locator(':scope > details').evaluateAll(nodes => nodes.forEach(node => node.open = true));
      const geometry = await lesson.evaluate(node => ({
        width: innerWidth, documentWidth: document.documentElement.scrollWidth,
        equations: [...node.querySelectorAll('.katex-display')].map((equation, index) => ({ index, width: equation.clientWidth, scroll: equation.scrollWidth })),
        mathErrors: node.querySelectorAll('.katex-error').length,
        minimumSvgFont: Math.min(...[...node.querySelectorAll('svg text')].map(text => parseFloat(getComputedStyle(text).fontSize) * text.getScreenCTM().a)),
        nonfinitePaths: [...node.querySelectorAll('svg path')].filter(path => /NaN|Infinity/.test(path.getAttribute('d'))).length,
        font: getComputedStyle(node.querySelector('p')).fontFamily,
        loadedFonts: [...document.fonts].filter(font => font.status === 'loaded').map(font => ({ family: font.family, weight: font.weight })),
      }));
      fs.writeFileSync(path.join(output, 'geometry-' + width + '.json'), JSON.stringify(geometry, null, 2));
      assert(geometry.documentWidth <= width + 1, JSON.stringify(geometry));
      assert(geometry.equations.every(equation => equation.scroll <= equation.width + 2), JSON.stringify(geometry.equations));
      assert.equal(geometry.mathErrors, 0);
      assert.equal(geometry.nonfinitePaths, 0);
      assert(geometry.minimumSvgFont >= 14, geometry.minimumSvgFont);
      assert(geometry.loadedFonts.some(font => font.family.includes('Space Grotesk')));
      await lesson.locator(':scope > details').evaluateAll(nodes => nodes.forEach(node => node.open = false));
      for (let index = 0; index < 3; index += 1) await shot(lesson.locator('.tda-figure').nth(index), 'inline-' + index);
      for (const index of [0, 2, 4, 5, 8, 9, 12]) {
        await lesson.locator('h2').nth(index).evaluate(node => window.scrollTo({ top: node.getBoundingClientRect().top + scrollY - 90, behavior: 'instant' }));
        await page.screenshot({ path: path.join(output, 'reading-' + (index + 1) + '-' + width + '.png') });
      }
      assert((await lesson.innerText()).includes('Next in this module: Category Theory'));
      assert.deepEqual(errors, []);
      records.push({ width, states, programs: Object.keys(examples).length, checkpointsAndPractice: 15, anchors, keyboardControls, keyboardScrolls, geometry, errors });
      console.log('Passed', width, 'with', states, 'checked states');
      await page.close();
    }
    const sources = ['src/learn/data/topics/topology-topological-data-analysis-tda.jsx', 'src/learn/components/lesson-labs/TopologyTdaLabs.jsx', 'src/learn/components/lesson-labs/topology-tda-labs.css', 'src/learn/data/topology-tda-models.js', 'src/learn/data/topology-tda-examples.js'].map(file => ({ file, sha256: hash(file) }));
    fs.writeFileSync(path.join(output, 'results.json'), JSON.stringify({ passed: true, verifiedAt: new Date().toISOString(), browser: await browser.version(), sources, records }, null, 2) + '\n');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

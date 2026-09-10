const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('fs');
const path = require('path');
const assert = require('assert/strict');
const output = path.resolve('scratch/spectral-browser');
fs.mkdirSync(output, { recursive: true });

(async () => {
  const { spectralGraphExamples } = await import('../src/learn/data/spectral-graph-examples.js');
  const model = await import('../src/learn/data/spectral-graph-models.js');
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1050 }, reducedMotion: 'reduce' });
      const errors = [];
      page.on('pageerror', e => errors.push(e.message));
      page.on('console', e => { if (['error', 'warning'].includes(e.type())) errors.push(e.text()); });
      page.on('requestfailed', r => errors.push(r.url() + ': ' + r.failure().errorText));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/spectral-graph-theory');
      const lesson = page.locator('.spectral-lesson');
      await lesson.locator('h2').last().waitFor();
      await page.evaluate(() => document.fonts.ready);
      const region = name => lesson.getByRole('region', { name, exact: true });
      async function slider(parent, name, value) { await parent.getByRole('slider', { name, exact: true }).fill(String(value)); }
      async function shot(locator, name) {
        await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = 'hidden'));
        await locator.screenshot({ path: path.join(output, `${name}-${width}.png`) });
        await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = ''));
      }
      const anchors = await lesson.locator('nav a').evaluateAll(links => links.map(link => ({ href: link.getAttribute('href'), present: !!document.getElementById(link.hash.slice(1)) })));
      assert.equal(anchors.length, 10);
      assert(anchors.every(item => item.present), JSON.stringify(anchors));
      assert.equal(await lesson.locator('.python-example').count(), 9);
      for (const example of spectralGraphExamples) {
        const container = lesson.locator('.python-example').filter({ has: page.getByRole('heading', { name: example.title, exact: true }) });
        const text = (await container.innerText()).replace(/\r\n/g, '\n');
        assert(text.includes(example.code.replace(/\r\n/g, '\n').trim()));
        assert(text.includes(example.expected.trim()));
        assert(await container.evaluate(node => node.previousElementSibling.textContent.includes('Before running:')));
      }
      const checks = lesson.locator('.lesson-check');
      assert.equal(await checks.count(), 12);
      for (let i = 0; i < await checks.count(); i++) {
        const check = checks.nth(i);
        assert((await check.locator('p').first().innerText()).length > 60);
        const details = check.locator('details');
        for (let j = 0; j < await details.count(); j++) {
          const detail = details.nth(j);
          await detail.locator('summary').click();
          assert((await detail.innerText()).length > 40);
          await detail.locator('summary').click();
        }
      }
      let states = 0;
      const spectrum = region('Graph spectrum lab');
      for (const weight of [0, .2, 1, 2]) {
        await slider(spectrum, 'Bridge weight', weight);
        for (let mode = 0; mode < 6; mode++) {
          await spectrum.getByRole('combobox', { name: 'Eigenvector', exact: true }).selectOption(String(mode));
          const expected = model.bridgeSpectrum(weight);
          const rows = await spectrum.getByRole('region', { name: `Node values in u${mode + 1}`, exact: true }).locator('tbody td').allTextContents();
          const vector = rows.map(Number);
          assert(Math.abs(model.dot(vector, vector) - 1) < 0.0004);
          const residual = model.multiply(expected.laplacian, vector).map((value, i) => value - expected.values[mode] * vector[i]);
          assert(Math.sqrt(model.dot(residual,residual)) < 0.001);
          states++;
        }
      }
      await spectrum.getByRole('button', { name: 'Reset spectrum', exact: true }).click();
      assert.equal(await spectrum.getByRole('slider').inputValue(), '0.2');
      await shot(spectrum, 'spectrum-default');
      const rowFigure = lesson.getByRole('button', { name: 'Two group values', exact: true });
      await rowFigure.click();
      assert((await lesson.locator('.laplacian-row').innerText()).includes('0.4'));
      await lesson.getByRole('button', { name: 'Constant signal', exact: true }).click();
      states++;
      const cut = region('Spectral cut sweep');
      for (const kind of ['bridge', 'unequal']) {
        await cut.getByRole('combobox', { name: 'Cut graph', exact: true }).selectOption(kind);
        for (const weight of [0, .2, 2]) {
          await slider(cut, 'Cut bridge weight', weight);
          const expected = model.cutSweep(weight, kind);
          for (let i = 0; i < expected.candidates.length; i++) {
            await cut.getByRole('combobox', { name: 'Sweep threshold', exact: true }).selectOption(String(i));
            assert((await cut.locator('.lesson-results').innerText()).includes(`Conductance ${model.formatSpectral(expected.candidates[i].conductance)}`));
            states++;
          }
          await cut.getByRole('button', { name: 'Choose lowest sweep conductance', exact: true }).click();
          assert.equal(Number(await cut.getByRole('combobox', { name: 'Sweep threshold', exact: true }).inputValue()), expected.candidates.indexOf(expected.best));
        }
      }
      await shot(cut, 'cut-unequal');
      await cut.getByRole('button', { name: 'Reset cut', exact: true }).click();
      const embedding = region('Spectral row embedding and clustering');
      for (const weight of [0, .08, 1.5]) {
        await slider(embedding, 'Embedding bridge weight', weight);
        for (const seed of ['spread', 'nearby']) {
          await embedding.getByRole('combobox', { name: 'Embedding seed centers', exact: true }).selectOption(seed);
          const expected = await page.evaluate(async args => (await import('/src/learn/data/spectral-graph-models.js')).clusteringState(...args), [weight, seed]);
          for (let frame = 0; frame < expected.frames.length; frame++) {
            assert((await embedding.locator('.lesson-results').innerText()).includes(expected.frames[frame].phase));
            if (expected.frames[frame].loss !== null) assert((await embedding.locator('.lesson-results').innerText()).includes(model.formatSpectral(expected.frames[frame].loss, 6)));
            if (weight === 0 && seed === 'nearby' && expected.frames[frame].empty.length) {
              assert((await embedding.locator('.lesson-results').innerText()).includes('Empty centers'));
              if (frame === 1) await shot(embedding, 'embedding-empty');
            }
            if (frame < expected.frames.length - 1) await embedding.getByRole('button', { name: 'Next clustering step', exact: true }).click();
            states++;
          }
          assert(await embedding.getByRole('button', { name: 'Next clustering step', exact: true }).isDisabled());
        }
      }
      await embedding.getByRole('button', { name: 'Reset embedding', exact: true }).click();
      await embedding.getByRole('combobox', { name: 'Embedding inspected node', exact: true }).selectOption('8');
      assert((await embedding.locator('.spectral-row-equation').innerText()).includes('Node I'));
      await shot(embedding, 'embedding-default');
      const filter = region('Graph spectral filtering');
      for (const kind of ['heat', 'ridge', 'cutoff']) {
        await filter.getByRole('combobox', { name: 'Spectral filter', exact: true }).selectOption(kind);
        for (const signal of Object.keys(model.spectralSignals)) {
          await filter.getByRole('combobox', { name: 'Filter signal', exact: true }).selectOption(signal);
          for (const weight of [0, .2, 2]) {
            await slider(filter, 'Filter bridge weight', weight);
            for (const amount of kind === 'cutoff' ? [1, 2, 3, 6] : [0, 1, 10]) {
              await slider(filter, 'Filter amount', amount);
              const state = kind === 'cutoff' ? await page.evaluate(async args => (await import('/src/learn/data/spectral-graph-models.js')).filterState(...args), [weight,signal,kind,amount]) : model.filterState(weight, signal, kind, amount);
              const actual = await filter.locator('.spectral-signal-comparison > div > strong:last-of-type').allTextContents();
              actual.forEach((value, i) => assert(Math.abs(Number(value)-state.output[i]) <= 0.00051));
              assert.equal((await filter.locator('.lesson-results').innerText()).includes('cutoff splits'), state.repeatedBoundary);
              states++;
            }
          }
        }
      }
      await filter.getByRole('button', { name: 'Reset filter', exact: true }).click();
      await shot(filter, 'filter-default');
      const keyboard = await lesson.locator('button, select, input, summary, [tabindex="0"]').count();
      await spectrum.getByRole('slider').focus();
      await page.keyboard.press('ArrowRight');
      assert.equal(await spectrum.getByRole('slider').inputValue(), '0.25');
      await spectrum.getByRole('button', { name: 'Reset spectrum', exact: true }).focus();
      await page.keyboard.press('Enter');
      assert.equal(await spectrum.getByRole('slider').inputValue(), '0.2');
      let scrolls = 0;
      for (const element of await lesson.locator('[tabindex="0"]').all()) {
        const overflow = await element.evaluate(node => node.scrollWidth > node.clientWidth + 2);
        if (!overflow) continue;
        await element.focus();
        await page.keyboard.press('ArrowRight');
        await page.waitForTimeout(150);
        assert(await element.evaluate(node => node.scrollLeft > 0));
        await element.evaluate(node => node.scrollLeft = 0);
        scrolls++;
      }
      await lesson.locator('details').evaluateAll(nodes => nodes.forEach(node => node.open = true));
      const mathErrors = await lesson.locator('.katex-error').allTextContents();
      assert.deepEqual(mathErrors, []);
      const geometry = await lesson.evaluate(node => ({
        pageWidth: document.documentElement.clientWidth, scrollWidth: document.documentElement.scrollWidth,
        math: [...node.querySelectorAll('.katex-display')].map(e => ({ width: e.clientWidth, scrollWidth: e.scrollWidth, text: e.textContent.slice(0,80) })),
        svgText: [...node.querySelectorAll('svg text')].map(e => { const b=e.getBBox(), v=e.ownerSVGElement.viewBox.baseVal; return { text:e.textContent, fits:b.x>=-1&&b.y>=-1&&b.x+b.width<=v.width+1&&b.y+b.height<=v.height+1 }; }).filter(x=>!x.fits),
      }));
      fs.writeFileSync(path.join(output, `geometry-${width}.json`), JSON.stringify(geometry,null,2));
      assert(geometry.scrollWidth <= geometry.pageWidth + 1, JSON.stringify(geometry));
      assert(geometry.math.every(e=>e.scrollWidth<=e.width+2), JSON.stringify(geometry.math.filter(e=>e.scrollWidth>e.width+2)));
      assert.deepEqual(geometry.svgText, []);
      await lesson.locator('details').evaluateAll(nodes => nodes.forEach(node => node.open = false));
      for (const index of width === 390 ? [0,1,2,3,4,5,6,7,8,9] : [0,3,5,7]) {
        await lesson.locator('h2').nth(index).evaluate(node => window.scrollTo(0,window.scrollY+node.getBoundingClientRect().top-100));
        await page.screenshot({ path: path.join(output, `reading-${index+1}-${width}.png`) });
      }
      await shot(lesson.locator('.lesson-sources'), 'sources');
      assert.deepEqual(errors, []);
      records.push({ width, states, anchors:anchors.length, examples:9, checks:12, keyboardControls:keyboard, localKeyboardScrolls:scrolls, math:geometry.math.length, errors });
      console.log('passed', width, states);
      await page.close();
    }
    fs.writeFileSync(path.join(output,'results.json'),JSON.stringify({verifiedAt:new Date().toISOString(),records},null,2));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exit(1); });

const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const directory = path.resolve('scratch/naive-bayes-browser');
fs.mkdirSync(directory, { recursive: true });

(async () => {
  const m = await import('../src/learn/data/naive-bayes-models.js');
  const { naiveBayesExamples: examples } = await import('../src/learn/data/naive-bayes-examples.js');
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [];
  const percent = value => m.formatNaiveBayesNumber(value * 100, 2) + '%';
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1080 }, reducedMotion: 'reduce' });
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/naive-bayes-probabilistic-classifiers', { waitUntil: 'domcontentloaded', timeout: 60000 });
      const lesson = page.locator('.naive-bayes-lesson');
      await lesson.waitFor({ timeout: 60000 });
      await page.evaluate(() => document.fonts.ready);
      assert(await page.evaluate(() => document.fonts.check('16px "Space Grotesk"')));
      let states = 0;
      const capture = async (locator, name) => {
        await locator.first().evaluate(node => scrollTo({ top: node.getBoundingClientRect().top + scrollY - 85, behavior: 'instant' }));
        await page.waitForTimeout(100);
        await page.screenshot({ path: path.join(directory, name + '-' + width + '.png') });
      };
      const lab = name => lesson.locator('[data-investigation="' + name + '"]');
      const slide = async (area, name, value) => {
        await area.getByRole('slider', { name, exact: true }).fill(String(value));
        states += 1;
      };
      assert.equal(await lesson.locator('h2').count(), 11);
      assert.equal(await lesson.locator('[data-investigation]').count(), 6);
      assert.deepEqual(await lesson.locator('nav a').evaluateAll(nodes => nodes.map(node => node.hash).filter(hash => !document.getElementById(hash.slice(1)))), []);
      for (let i = 0; i < 11; i += 1) await capture(lesson.locator('h2').nth(i), 'reading-' + (i + 1));
      await capture(lesson.locator('.nb-model-fork'), 'model-fork');
      await capture(lesson.locator('.nb-corpus'), 'corpus');
      await capture(lesson.locator('.nb-pooling'), 'complement-pooling');
      await capture(lesson.locator('.nb-fit-lanes'), 'calibration-ownership');

      const token = lab('token-evidence');
      const apply = async (text, alpha) => {
        await token.getByRole('textbox', { name: 'Message', exact: true }).fill(text);
        await token.getByRole('textbox', { name: 'Alpha', exact: true }).fill(String(alpha));
        await token.getByRole('button', { name: 'Apply message and alpha' }).click();
      };
      for (const [message, alpha] of [['free meeting', 1], ['free free money', .25], ['free meeting', 0], ['zephyronic', 1], ['', 1]]) {
        await apply(message, alpha);
        const expected = m.tokenEvidenceState(message, alpha);
        for (let i = 0; i < expected.frames.length; i += 1) {
          if (i) await token.getByRole('button', { name: 'Next step', exact: true }).click();
          const text = await token.locator('.nb-evidence-readout').innerText();
          assert(text.includes('Log odds: ' + m.formatNaiveBayesNumber(expected.frames[i].logOdds)));
          if (expected.frames[i].probabilities) assert(text.includes(percent(expected.frames[i].probabilities[1])));
          else assert(text.includes('posterior is undefined'));
          states += 1;
        }
        if (alpha === 0) await capture(token.locator('.nb-evidence-readout'), 'zero-support');
        if (message === 'zephyronic') assert((await token.innerText()).includes('Ignored by this fixed vocabulary'));
      }
      await apply('free meeting', 1);
      await token.getByRole('button', { name: 'Next step', exact: true }).click();
      const beforeError = await token.locator('.nb-evidence-readout').innerText();
      for (const alpha of ['', '-1', 'NaN', '1e-300']) {
        await apply('changed message', alpha);
        assert.equal(await token.locator('.nb-evidence-readout').innerText(), beforeError);
        assert(await token.getByRole('alert').isVisible());
      }
      await capture(token, 'invalid-alpha');
      await token.getByRole('button', { name: 'Reset', exact: true }).focus();
      await page.keyboard.press('Enter');
      assert.equal(await token.getByRole('textbox', { name: 'Message', exact: true }).inputValue(), 'free');
      await token.getByRole('button', { name: 'Next step', exact: true }).click();
      await capture(token.locator('.nb-evidence-readout'), 'token-evidence');
      await token.getByRole('button', { name: 'Back', exact: true }).click();
      assert((await token.locator('.nb-evidence-readout').innerText()).includes('0 of 1'));

      const presence = lab('presence-evidence');
      for (const count of [2, 3, 4, 5]) {
        await presence.getByRole('button', { name: 'Add one free', exact: true }).click();
        const expected = m.presenceEvidenceState([count, 0, 0, 0, 0]);
        const text = await presence.locator('.nb-result-pair').innerText();
        assert(text.includes(percent(expected.bernoulli.probabilities[1])));
        assert(text.includes(percent(expected.multinomial.probabilities[1])));
        states += 1;
      }
      assert(await presence.getByRole('button', { name: 'Add one free', exact: true }).isDisabled());
      await capture(presence, 'repeat-presence');
      for (let i = 0; i < 5; i += 1) await presence.getByRole('button', { name: 'Remove one free', exact: true }).click();
      assert((await presence.locator('.nb-result-pair').innerText()).includes(percent(m.presenceEvidenceState([0,0,0,0,0]).bernoulli.probabilities[1])));
      await capture(presence.locator('.nb-result-pair'), 'all-absent');
      await presence.getByRole('button', { name: 'Reset presence comparison' }).click();

      const observation = lab('gaussian-observation');
      for (const reading of [-5, -1.4, 0, 1.4, 5]) for (const interval of [.1, .4, 1]) {
        await slide(observation, 'Reading in mV', reading);
        await slide(observation, 'Interval width in mV', interval);
        const expected = m.gaussianObservationState(reading, interval);
        const text = await observation.innerText();
        assert(text.includes(percent(expected.probabilities[1])));
        for (const mass of expected.masses) assert(text.includes(percent(mass)));
      }
      await observation.getByRole('button').click();
      await slide(observation, 'Reading in mV', 1.4);
      await capture(observation.locator('figure'), 'density-crossing');
      await observation.getByRole('slider', { name: 'Reading in mV', exact: true }).focus();
      await page.keyboard.press('ArrowRight');
      assert.equal(await observation.getByRole('slider', { name: 'Reading in mV', exact: true }).inputValue(), '1.5');
      const geometry = lab('gaussian-geometry');
      for (const mode of ['equal', 'unequal']) {
        await geometry.getByRole('combobox').selectOption(mode);
        for (const point of [[0,0], [-2,-2], [-8,-8], [6,6], [.25,-.25]]) {
          await slide(geometry, 'Probe x', point[0]); await slide(geometry, 'Probe y', point[1]);
          assert((await geometry.innerText()).includes(percent(m.gaussianGeometryState(mode, point).probabilities[1])));
        }
        await capture(geometry.locator('figure'), 'geometry-' + mode);
      }
      await geometry.getByRole('button').click();

      const copied = lab('copied-alarm');
      for (const positive of [true, false]) {
        await copied.getByRole('combobox').selectOption(String(positive));
        for (let count = 1; count <= 5; count += 1) {
          await slide(copied, 'Number of recorded copies', count);
          const expected = m.copiedAlarmState(count, positive);
          const text = await copied.innerText();
          assert(text.includes(percent(expected.naive.probabilities[1])));
          assert(text.includes(percent(expected.truth.probabilities[1])));
          assert(text.includes(percent(expected.accuracy)));
          if (count === 2 && positive) {
            assert((await copied.locator('.nb-result-pair').innerText()).includes('Decision: normal'));
            await capture(copied.locator('.nb-result-pair'), 'copied-tie');
          }
        }
      }
      await copied.getByRole('button').click();
      await capture(copied.locator('figure'), 'copied-causal');
      await capture(copied.locator('.nb-result-pair'), 'copied-risk');
      const reliability = lab('reliability-bins');
      for (const mapping of [false, true]) {
        await reliability.getByRole('checkbox').setChecked(mapping);
        for (let bins = 2; bins <= 6; bins += 1) {
          await slide(reliability, 'Number of reliability bins', bins);
          const expected = m.reliabilityState(bins, mapping);
          assert.equal(await reliability.locator('circle').count(), expected.bins.filter(bin => bin.count).length);
          assert((await reliability.innerText()).includes(m.formatNaiveBayesNumber(expected.brier, 6)));
          assert((await reliability.innerText()).includes(m.formatNaiveBayesNumber(expected.logLoss, 6)));
        }
        await capture(reliability.locator('figure'), 'reliability-' + mapping);
      }
      await reliability.getByRole('button').click();
      const table = reliability.locator('.lesson-table-wrap').first();
      await table.focus(); await page.keyboard.press('ArrowRight'); await page.waitForTimeout(120);
      const tableGeometry = await table.evaluate(node => ({ width: node.clientWidth, content: node.scrollWidth, offset: node.scrollLeft }));
      if (tableGeometry.content > tableGeometry.width) assert(tableGeometry.offset > 0, 'Keyboard table scrolling');

      // Read all actual expanded teaching and verify displayed programs regardless of ordering.
      await lesson.locator('details').evaluateAll(nodes => nodes.forEach(node => { node.open = true; }));
      assert.equal(await lesson.locator('.python-example').count(), examples.length);
      for (const example of examples) {
        const block = lesson.locator('.python-example').filter({ hasText: example.title });
        assert.equal(await block.count(), 1, example.id);
        assert((await block.innerText()).includes(example.code.trim()), 'Displayed source ' + example.id);
        assert((await block.innerText()).includes(example.expected.trim()), 'Displayed stdout ' + example.id);
        assert(await lesson.getByText(example.question, { exact: false }).count(), 'Question ' + example.id);
      }
      assert.equal(await lesson.locator('section.lesson-check').count(), 12);
      for (const check of await lesson.locator('div.lesson-check').all()) {
        assert((await check.locator('p').first().innerText()).length > 40);
        assert((await check.locator('details').innerText()).length > 150);
      }
      assert.equal(await lesson.locator('.katex-error').count(), 0);
      const equations = await lesson.locator('.katex-display').evaluateAll(nodes => nodes.map((node, index) => ({
        index, width: node.getBoundingClientRect().width, content: node.scrollWidth, text: node.textContent
      })));
      const wideEquations = equations.filter(eq => eq.content > eq.width + 1);
      fs.writeFileSync(path.join(directory, 'equations-' + width + '.json'), JSON.stringify(equations, null, 2));
      assert.deepEqual(wideEquations, [], 'Wide equations');
      const labels = await lesson.locator('svg text').evaluateAll(nodes => nodes.map(node => ({
        text: node.textContent, left: node.getBBox().x, right: node.getBBox().x + node.getBBox().width,
        bound: node.ownerSVGElement.viewBox.baseVal.width
      })));
      assert(labels.every(label => label.left >= -.5 && label.right <= label.bound + .5), 'Clipped SVG labels ' + JSON.stringify(labels.filter(label => label.left < -.5 || label.right > label.bound + .5)));
      const overflow = await page.evaluate(() => ({ page: document.documentElement.scrollWidth, viewport: innerWidth }));
      assert(overflow.page <= width + 1, 'Page overflow ' + JSON.stringify(overflow));
      await capture(lesson.locator('.python-example').filter({ hasText: examples.find(item => item.id === 'calibrated-report').title }), 'calibrated-program');
      await capture(lesson.locator('section.lesson-check').last(), 'changed-report');
      await capture(lesson.locator('section.lesson-check').last().locator('details').last(), 'changed-solution');
      await capture(lesson.locator('.lesson-sources'), 'sources');
      assert.deepEqual(errors, []);
      records.push({ width, states, errors, equations, overflow, labels, tableGeometry, programs: examples.length,
        practice: 12, checkpoints: await lesson.locator('div.lesson-check').count(), fonts: await page.evaluate(() => ({ sans: document.fonts.check('16px "Space Grotesk"'), mono: document.fonts.check('16px "JetBrains Mono"') })) });
      fs.writeFileSync(path.join(directory, 'partial-results.json'), JSON.stringify(records, null, 2));
      console.log('Passed ' + width + ': ' + states + ' operated states');
      await page.close();
    }
    fs.writeFileSync(path.join(directory, 'results.json'), JSON.stringify({ completedAt: new Date().toISOString(), records }, null, 2));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

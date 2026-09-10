const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { pathToFileURL } = require('node:url');
const folder = path.resolve('scratch/gradient-variants-browser');
fs.mkdirSync(folder, { recursive: true });

async function capture(page, locator, name) {
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = 'hidden'));
  await locator.screenshot({ path: path.join(folder, name) });
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = ''));
}

async function range(lab, label, value) {
  const input = lab.getByRole('slider', { name: label, exact: true });
  await input.evaluate((node, next) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(node, String(next));
    node.dispatchEvent(new Event('input', { bubbles: true }));
    node.dispatchEvent(new Event('change', { bubbles: true }));
  }, value);
  assert.equal(Number(await input.inputValue()), value);
}

(async () => {
  const models = await import(pathToFileURL(path.resolve('src/learn/data/gradient-variants-models.js')).href);
  const { gradientVariantsExamples } = await import(pathToFileURL(path.resolve('src/learn/data/gradient-variants-examples.js')).href);
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  try {
    for (const width of [1440, 390]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/gradient-descent-variants-sgd-adam-adagrad-rmsprop-lamb-lars?module=math-foundations');
      await page.locator('.optimizer-lesson').waitFor({ timeout: 60000 });
      assert.equal(await page.locator('.optimizer-lab').count(), 5);
      const counts = { batch: 0, momentum: 0, adaptive: 0, decay: 0, layer: 0 };
      const number = models.formatOptimizerNumber;
      const batch = page.locator('section[aria-label="Sampled gradient investigation"]');
      await capture(page, batch, `batch-default-${width}.png`);
      for (const theta of [-1, 0, 1]) {
        await range(batch, 'Parameter theta', theta);
        for (let mask = 1; mask < 16; mask++) {
          const selected = [0, 1, 2, 3].filter(index => mask & (1 << index));
          for (const index of selected) await batch.getByRole('checkbox').nth(index).check();
          for (const index of [0, 1, 2, 3].filter(index => !selected.includes(index))) await batch.getByRole('checkbox').nth(index).uncheck();
          const expected = models.batchGradientState(theta, selected, .2, 'mean');
          assert.equal(await batch.locator('dd').nth(0).innerText(), number(expected.meanGradient));
          assert.equal(await batch.locator('dd').nth(4).innerText(), `${number(expected.fullLoss)} → ${number(expected.nextFullLoss)}`);
          counts.batch++;
        }
      }
      await batch.getByRole('button', { name: 'Reset batch' }).click();
      await batch.getByRole('checkbox').nth(3).click();
      assert.match(await batch.getByRole('alert').innerText(), /Select one through four/);
      assert(await batch.getByRole('checkbox').nth(3).isChecked());
      await batch.getByRole('checkbox').nth(2).check();
      await batch.getByLabel('Batch reduction', { exact: true }).selectOption('sum');
      await range(batch, 'Batch learning rate', .4);
      assert.equal(await batch.locator('dd').nth(2).innerText(), '-2');
      const slider = batch.getByRole('slider', { name: 'Parameter theta', exact: true });
      await slider.focus();
      await page.keyboard.press('Home');
      await page.keyboard.press('ArrowRight');
      assert.equal(await slider.inputValue(), '-3.75');
      counts.batch += 2;

      const momentum = page.locator('section[aria-label="Momentum and lookahead investigation"]');
      for (const method of ['sgd', 'momentum', 'nesterov']) {
        await momentum.getByLabel('Trajectory method', { exact: true }).selectOption(method);
        for (let step = 0; step <= 8; step++) {
          const expected = models.momentumTrajectoryState(method, .08, .8, 24, 20).frames[step];
          assert.match(await momentum.locator('p[role=status]').innerText(), new RegExp(`Loss ${number(expected.loss).replace('.', '\\.')} at update ${step}`));
          if (step < 8) await momentum.getByRole('button', { name: 'Next', exact: true }).click();
          counts.momentum++;
        }
        await capture(page, momentum, `momentum-${method}-${width}.png`);
      }
      await range(momentum, 'Momentum beta', 0);
      await range(momentum, 'Vertical curvature c', 30);
      await range(momentum, 'Trajectory learning rate', .25);
      while (await momentum.getByRole('button', { name: 'Next', exact: true }).isEnabled()) await momentum.getByRole('button', { name: 'Next', exact: true }).click();
      assert.match(await momentum.locator('p[role=status]').innerText(), /stops before a coordinate would exceed/);
      await capture(page, momentum, `momentum-divergent-${width}.png`);
      await momentum.getByRole('button', { name: 'Restart', exact: true }).click();
      const next = momentum.getByRole('button', { name: 'Next', exact: true });
      await next.focus();
      await page.keyboard.press('Enter');
      assert.match(await momentum.locator('p[role=status]').innerText(), /at update 1/);
      await momentum.getByRole('button', { name: 'Previous', exact: true }).focus();
      await page.keyboard.press('Space');
      assert.match(await momentum.locator('p[role=status]').innerText(), /at update 0/);
      counts.momentum += 3;

      const adaptive = page.locator('section[aria-label="Adaptive gradient history investigation"]');
      for (const method of ['adagrad', 'rmsprop', 'adam']) {
        await adaptive.getByLabel('Adaptive method', { exact: true }).selectOption(method);
        for (const profile of ['constant', 'alternating', 'sparse', 'spike']) {
          await adaptive.getByLabel('Gradient history', { exact: true }).selectOption(profile);
          for (let step = 0; step < 8; step++) {
            const expected = models.adaptiveHistoryState(method, profile).frames[step];
            assert.equal(await adaptive.locator('tbody tr').nth(1).locator('td').last().innerText(), number(expected.displacement[1]));
            if (step < 7) await adaptive.getByRole('button', { name: 'Next', exact: true }).click();
            counts.adaptive++;
          }
        }
      }
      await adaptive.getByLabel('Gradient history', { exact: true }).selectOption('sparse');
      for (let index = 0; index < 3; index++) await adaptive.getByRole('button', { name: 'Next', exact: true }).click();
      await capture(page, adaptive, `adaptive-sparse-${width}.png`);
      await adaptive.getByRole('checkbox').uncheck();
      assert.match(await adaptive.locator('p[role=status]').innerText(), /Correction is off/);
      await range(adaptive, 'Recent-square decay beta2', .99);
      assert.equal(await adaptive.locator('tbody tr').nth(0).locator('td').last().innerText(), number(models.adaptiveHistoryState('adam', 'sparse', .1, .9, .99, 1e-6, false).frames[0].displacement[0]));
      counts.adaptive += 2;

      const decay = page.locator('section[aria-label="Coupled L2 and AdamW investigation"]');
      for (const preset of ['zero', 'unequal', 'opposed']) {
        await decay.getByLabel('Decay input case', { exact: true }).selectOption(preset);
        for (const steps of [1, 2, 8]) {
          await range(decay, 'Decay update count', steps);
          const expected = models.decayComparisonState(preset, .1, .1, steps);
          assert.equal(await decay.locator('tbody tr').last().locator('td').nth(1).innerText(), `[${expected.methods[1].theta.map(number).join(', ')}]`);
          counts.decay++;
        }
      }
      await decay.getByLabel('Decay input case', { exact: true }).selectOption('zero');
      await range(decay, 'Decay update count', 1);
      await capture(page, decay, `decay-zero-${width}.png`);
      await range(decay, 'Decay learning rate', .3);
      await range(decay, 'Decay coefficient lambda', 0);
      assert.equal(await decay.locator('tbody td').nth(0).innerText(), await decay.locator('tbody td').nth(1).innerText());
      counts.decay++;

      const layer = page.locator('section[aria-label="Layer relative update investigation"]');
      for (const method of ['sgd', 'lars', 'lamb']) {
        await layer.getByLabel('Layer method', { exact: true }).selectOption(method);
        for (const preset of ['ordinary', 'zeroWeight', 'zeroGradient']) {
          await layer.getByLabel('Layer boundary case', { exact: true }).selectOption(preset);
          const expected = models.layerScaleState(method, .1, .1, 0, .1, preset);
          assert.equal(await layer.locator('tbody tr').nth(1).locator('td').nth(2).innerText(), number(expected.blocks[1].ratio));
          counts.layer++;
        }
      }
      await layer.getByLabel('Layer boundary case', { exact: true }).selectOption('ordinary');
      await capture(page, layer, `layer-lamb-${width}.png`);
      for (const [label, value] of [['Second-block weight scale', 2], ['Layer learning rate', .5], ['LARS trust coefficient', 1], ['Layer decay coefficient', 1]]) await range(layer, label, value);
      await layer.getByLabel('Layer boundary case', { exact: true }).selectOption('zeroWeight');
      assert((await layer.innerText()).includes('undefined at θ=0'));
      counts.layer++;

      const anchors = await page.locator('.lesson-intro a[href^="#"]').evaluateAll(nodes => nodes.map(node => node.hash.slice(1)));
      assert.equal(anchors.length, 9);
      for (const anchor of anchors) {
        assert.equal(await page.locator(`[id="${anchor}"]`).count(), 1, anchor);
        await page.locator(`.lesson-intro a[href="#${anchor}"]`).click();
        await page.waitForFunction(id => { const top = document.getElementById(id).getBoundingClientRect().top; return top > -2 && top < innerHeight; }, anchor);
      }
      assert.equal(await page.locator('.optimizer-practice').count(), 6);
      assert.equal(await page.locator('.optimizer-practice details[open]').count(), 0);
      const summary = page.locator('.optimizer-practice summary').first();
      await summary.focus();
      await page.keyboard.press('Enter');
      assert.equal(await page.locator('.optimizer-practice details[open]').count(), 1);
      await page.locator('.optimizer-lesson details').evaluateAll(nodes => nodes.forEach(node => node.open = true));
      const text = await page.locator('.optimizer-lesson').innerText();
      for (const [name, example] of Object.entries(gradientVariantsExamples)) {
        assert(text.includes(example.code), `${name}: complete code rendered`);
        assert(text.includes(example.expected), `${name}: expected output rendered`);
      }
      assert.equal(await page.locator('.katex-error').count(), 0);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
      const math = await page.locator('.optimizer-lesson .katex-display').evaluateAll(nodes => nodes.map(node => ({ width: node.getBoundingClientRect().width, scroll: node.scrollWidth, client: node.clientWidth })));
      const focus = await page.locator('.optimizer-lab button,.optimizer-lab input,.optimizer-lab select,.optimizer-lab [tabindex="0"]').count();
      const sources = await page.locator('.lesson-sources a').count();
      assert.deepEqual(errors, []);
      results.push({ width, counts, anchors: anchors.length, completePrograms: Object.keys(gradientVariantsExamples).length, practiceGroups: 6, focusTargets: focus, sources, math, pageErrors: errors, pageOverflow: false });
      await page.close();
    }
  } finally { await browser.close(); }
  fs.writeFileSync(path.join(folder, 'results.json'), JSON.stringify({ checkedAt: new Date().toISOString(), status: 'passed', results }, null, 2));
  console.log(JSON.stringify(results, null, 2));
})().catch(error => { console.error(error); process.exitCode = 1; });

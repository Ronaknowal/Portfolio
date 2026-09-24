const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { pathToFileURL } = require('node:url');
const folder = path.resolve('scratch/duality-kkt-browser');
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
  const models = await import(pathToFileURL(path.resolve('src/learn/data/duality-kkt-models.js')).href);
  const { dualityKktExamples } = await import(pathToFileURL(path.resolve('src/learn/data/duality-kkt-examples.js')).href);
  const number = models.formatDualityNumber;
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  try {
    for (const width of [1440, 390]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/convex-duality-lagrangian-methods-kkt-conditions?module=math-foundations');
      await page.locator('.duality-lesson').waitFor({ timeout: 60000 });
      assert.equal(await page.locator('.duality-lab').count(), 4);
      const counts = { projection: 0, scalar: 0, sensitivity: 0, resource: 0 };
      const projection = page.getByRole('region', { name: 'Projection and bound investigation', exact: true });
      await capture(page, projection, `projection-default-${width}.png`);
      for (const budget of [-2, 5, 7, 10]) {
        await range(projection, 'Projection budget b', budget);
        for (const multiplier of [-2, 0, 2, 12]) {
          await range(projection, 'Bound multiplier lambda', multiplier);
          for (const candidate of [[2, 2], [3, 4], [8, 8]]) {
            await range(projection, 'Candidate x', candidate[0]);
            await range(projection, 'Candidate y', candidate[1]);
            const expected = models.projectionCertificateState(budget, candidate, multiplier);
            assert.equal(await projection.locator('dd').nth(0).innerText(), number(expected.objective));
            assert.equal(await projection.locator('dd').nth(2).innerText(), number(expected.lagrangian));
            assert.equal(await projection.locator('dd').nth(6).innerText(), number(expected.certifiedGap));
            const circle = projection.locator('circle.duality-contour');
            assert(Math.abs(Number(await circle.getAttribute('r')) - Math.sqrt(expected.objective) * 246 / 13) < 1e-8);
            counts.projection++;
          }
        }
        await projection.getByRole('button', { name: 'Use exact optimal pair' }).click();
        assert.equal(await projection.locator('dd').nth(6).innerText(), '0');
        counts.projection++;
      }
      await projection.getByRole('button', { name: 'Reset projection' }).click();
      await projection.getByRole('button', { name: 'Try unconstrained target' }).click();
      assert.match(await projection.getByRole('status').innerText(), /No certificate/);
      await capture(page, projection, `projection-infeasible-${width}.png`);
      await projection.getByRole('slider', { name: 'Candidate x', exact: true }).focus();
      await page.keyboard.press('Home');
      await page.keyboard.press('ArrowRight');
      assert.equal(await projection.getByRole('slider', { name: 'Candidate x', exact: true }).inputValue(), '-2.5');
      await projection.getByRole('button', { name: 'Reset projection' }).click();

      const scalar = page.getByRole('region', { name: 'Four KKT conditions investigation', exact: true });
      await capture(page, scalar, `scalar-positive-${width}.png`);
      for (const center of [-2, -1, 0, 1, 2]) {
        await range(scalar, 'Quadratic center c', center);
        for (const candidate of [-2, 0, 1, 4]) {
          await range(scalar, 'KKT candidate x', candidate);
          for (const multiplier of [-2, 0, 2, 6]) {
            await range(scalar, 'KKT multiplier lambda', multiplier);
            const expected = models.scalarKktState(center, candidate, multiplier);
            assert.equal(await scalar.locator('.duality-checks strong').allTextContents().then(values => values.filter(value => value.startsWith('✓')).length), Object.values(expected.conditions).filter(Boolean).length);
            assert((await scalar.getByRole('status').innerText()).includes(expected.allConditions ? 'All four checks hold' : 'At least one required check fails'));
            counts.scalar++;
          }
        }
      }
      for (const [label, center] of [['Active · positive price', -1], ['Active · zero price', 0], ['Inactive · zero price', 1]]) {
        await scalar.getByRole('button', { name: label, exact: true }).click();
        assert.equal(await scalar.getByRole('slider', { name: 'Quadratic center c', exact: true }).inputValue(), String(center));
        assert.match(await scalar.getByRole('status').innerText(), /All four checks hold/);
        counts.scalar++;
      }
      await scalar.getByRole('button', { name: 'Active · zero price', exact: true }).click();
      await capture(page, scalar, `scalar-zero-${width}.png`);
      await range(scalar, 'Quadratic center c', -1);
      await range(scalar, 'KKT candidate x', 1);
      await range(scalar, 'KKT multiplier lambda', 4);
      await capture(page, scalar, `scalar-complementarity-fails-${width}.png`);
      await scalar.getByRole('button', { name: 'Solve current center', exact: true }).focus();
      await page.keyboard.press('Enter');
      assert.match(await scalar.getByRole('status').innerText(), /All four checks hold/);

      const sensitivity = page.getByRole('region', { name: 'Multiplier sensitivity investigation', exact: true });
      await capture(page, sensitivity, `sensitivity-smooth-${width}.png`);
      for (const mode of ['quadratic', 'kink']) {
        await sensitivity.getByLabel('Value function', { exact: true }).selectOption(mode);
        for (const base of mode === 'quadratic' ? [0, 5, 7, 10] : [-2, 0, 2]) {
          await range(sensitivity, 'Original right-hand side', base);
          for (const change of [-2, -0.5, 0, 0.5, 2]) {
            await range(sensitivity, 'Right-hand-side change delta', change);
            const expected = models.sensitivityState(mode, base, change, 0.5);
            assert.equal(await sensitivity.locator('dd').nth(0).innerText(), number(expected.price));
            assert.equal(await sensitivity.locator('dd').nth(2).innerText(), number(expected.actualChange));
            assert.equal(await sensitivity.locator('dd').nth(4).innerText(), number(expected.supportingGap));
            counts.sensitivity++;
          }
        }
      }
      await sensitivity.getByRole('button', { name: 'Inspect the corner' }).click();
      for (const price of [0, 0.25, 0.5, 0.75, 1]) {
        await range(sensitivity, 'Optimal kink multiplier', price);
        assert.equal(await sensitivity.locator('dd').nth(0).innerText(), number(price));
        assert.match(await sensitivity.locator('dd').nth(5).innerText(), /does not exist/);
        counts.sensitivity++;
      }
      await capture(page, sensitivity, `sensitivity-kink-${width}.png`);
      await sensitivity.getByRole('button', { name: 'Active constraint · zero price' }).click();
      await range(sensitivity, 'Right-hand-side change delta', -0.5);
      assert.equal(await sensitivity.locator('dd').nth(2).innerText(), '0.125');
      await capture(page, sensitivity, `sensitivity-active-zero-${width}.png`);
      await sensitivity.getByRole('button', { name: 'Reset sensitivity' }).click();
      const select = sensitivity.getByLabel('Value function', { exact: true });
      await select.focus();
      await page.keyboard.press('End');
      await page.keyboard.press('Enter');
      assert.equal(await select.inputValue(), 'kink');

      const resource = page.getByRole('region', { name: 'Resource price iteration investigation', exact: true });
      await capture(page, resource, `resource-default-${width}.png`);
      for (const budget of [0, 2.5, 5, 8]) {
        await range(resource, 'Shared resource budget', budget);
        for (const rate of [0, 1, 3, 4]) {
          await range(resource, 'Price step size alpha', rate);
          const expected = models.resourceDualAscentState(budget, rate, 0, 20);
          for (let step = 0; step <= 5; step++) {
            const frame = expected.frames[step];
            assert.equal(await resource.locator('dd').nth(2).innerText(), number(frame.dualValue));
            assert.equal(await resource.locator('dd').nth(3).innerText(), number(frame.repairedObjective));
            assert.equal(await resource.locator('dd').nth(4).innerText(), number(frame.certificateGap));
            if (step < 5) await resource.getByRole('button', { name: 'Next price', exact: true }).click();
            counts.resource++;
          }
        }
      }
      await resource.getByRole('button', { name: 'Reset resource' }).click();
      for (const [budget, step] of [[2.75, 18], [3.25, 17], [4, 16]]) {
        await range(resource, 'Shared resource budget', budget);
        await range(resource, 'Price step size alpha', 1.75);
        for (let index = 0; index < step; index++) await resource.getByRole('button', { name: 'Next price', exact: true }).click();
        const expected = models.resourceDualAscentState(budget, 1.75, 0, 20).frames[step];
        assert(expected.subtractedGap < 0);
        assert(expected.certificateGap >= 0);
        assert.equal(await resource.locator('dd').nth(4).innerText(), number(expected.certificateGap));
        await capture(page, resource, `resource-cancellation-${budget}-${width}.png`);
        counts.resource++;
      }
      await resource.getByRole('button', { name: 'Reset resource' }).click();
      await range(resource, 'Price step size alpha', 3);
      await resource.getByRole('button', { name: 'Show state 20' }).click();
      assert(await resource.getByRole('button', { name: 'Next price', exact: true }).isDisabled());
      await capture(page, resource, `resource-cycle-${width}.png`);
      await range(resource, 'Initial resource price', 20);
      assert.match(await resource.locator('output').innerText(), /State 0/);
      assert(await resource.getByRole('button', { name: 'Previous price' }).isDisabled());
      await resource.getByRole('button', { name: 'Next price', exact: true }).focus();
      await page.keyboard.press('Space');
      assert.match(await resource.locator('output').innerText(), /State 1/);
      await resource.getByRole('button', { name: 'Previous price' }).click();
      await resource.getByRole('button', { name: 'Reset resource' }).click();
      await resource.getByRole('button', { name: 'Show state 20' }).click();
      await capture(page, resource, `resource-converging-${width}.png`);

      const anchors = await page.locator('.lesson-intro a[href^="#"]').evaluateAll(nodes => nodes.map(node => node.hash.slice(1)));
      assert.equal(anchors.length, 9);
      for (const anchor of anchors) {
        assert.equal(await page.locator(`[id="${anchor}"]`).count(), 1, anchor);
        await page.locator(`.lesson-intro a[href="#${anchor}"]`).click();
        await page.waitForFunction(id => { const top = document.getElementById(id).getBoundingClientRect().top; return top > -2 && top < innerHeight; }, anchor);
      }
      assert.equal(await page.locator('.duality-practice').count(), 6);
      assert.equal(await page.locator('.duality-practice details[open]').count(), 0);
      await page.locator('.duality-practice summary').first().focus();
      await page.keyboard.press('Enter');
      assert.equal(await page.locator('.duality-practice details[open]').count(), 1);
      await page.locator('.duality-lesson details').evaluateAll(nodes => nodes.forEach(node => node.open = true));
      const text = (await page.locator('.duality-lesson').innerText()).replaceAll('\r\n', '\n');
      for (const [name, example] of Object.entries(dualityKktExamples)) {
        assert(text.includes(example.code.replaceAll('\r\n', '\n')), `${name}: complete code rendered`);
        assert(text.includes(example.expected.replaceAll('\r\n', '\n')), `${name}: expected output rendered`);
      }
      assert.equal(await page.locator('.katex-error').count(), 0);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false);
      const focusTargets = await page.locator('.duality-lab button,.duality-lab input,.duality-lab select').count();
      const sources = await page.locator('.lesson-sources a').count();
      assert.deepEqual(errors, []);
      results.push({ width, counts, anchors: anchors.length, completePrograms: 10, practiceGroups: 6, focusTargets, sources, pageErrors: errors, pageOverflow: false });
      await page.close();
    }
  } finally { await browser.close(); }
  fs.writeFileSync(path.join(folder, 'results.json'), JSON.stringify({ checkedAt: new Date().toISOString(), status: 'passed', results }, null, 2));
  console.log(JSON.stringify(results, null, 2));
})().catch(error => { console.error(error); process.exitCode = 1; });

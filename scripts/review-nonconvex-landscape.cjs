const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const folder = path.resolve('scratch/nonconvex-landscape-browser');
fs.mkdirSync(folder, { recursive: true });

async function range(lab, label, value) {
  await lab.getByRole('slider', { name: label, exact: true }).evaluate((node, next) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(node, String(next));
    node.dispatchEvent(new Event('input', { bubbles: true }));
    node.dispatchEvent(new Event('change', { bubbles: true }));
  }, value);
}

async function capture(page, element, name) {
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = 'hidden'));
  await element.screenshot({ path: path.join(folder, name) });
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = ''));
}

async function state(lab) {
  return JSON.parse(await lab.getAttribute('data-state'));
}

async function geometry(lesson) {
  const clipped = await lesson.locator('.landscape-plot').evaluateAll(nodes => nodes.flatMap((svg, index) => {
    const border = svg.getBoundingClientRect();
    return [...svg.querySelectorAll('text')].filter(node => {
      const box = node.getBoundingClientRect();
      return box.left < border.left - 1 || box.right > border.right + 1 || box.top < border.top - 1 || box.bottom > border.bottom + 1;
    }).map(node => ({ index, text: node.textContent }));
  }));
  assert.deepEqual(clipped, [], 'SVG text stays inside its viewport');
}

(async () => {
  const models = await import(pathToFileURL(path.resolve('src/learn/data/nonconvex-landscape-models.js')));
  const { nonconvexLandscapeExamples: examples } = await import(pathToFileURL(path.resolve('src/learn/data/nonconvex-landscape-examples.js')));
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  try {
    for (const width of [1440, 390]) {
      const page = await browser.newPage({ viewport: { width, height: 1050 }, reducedMotion: 'reduce' });
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.routeWebSocket('**', socket => socket.close());
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/non-convex-optimization-landscape');
      const lesson = page.locator('.landscape-lesson');
      await lesson.waitFor();
      await lesson.locator('.landscape-lab').first().waitFor();
      const counts = { wells: 0, stationary: 0, noise: 0, symmetry: 0 };
      const wells = lesson.locator('[data-lab="wells"]');
      for (const tilt of [-0.25, 0, 0.15, 0.25]) {
        await range(wells, 'Tilt δ', tilt);
        for (const initial of [-1.5, -0.8, 0, 0.8, 1.5]) {
          await range(wells, 'Initial x', initial);
          for (const rate of [0, 0.12, 0.3]) {
            await range(wells, 'Learning rate η', rate);
            assert.equal((await state(wells)).step, 0);
            await wells.getByRole('button', { name: 'Next step', exact: true }).click();
            const expected = models.wellState(tilt, initial, rate);
            assert.equal((await state(wells)).x, expected.frames[1].x);
            await wells.getByRole('button', { name: 'Show final step' }).click();
            assert.equal((await state(wells)).x, expected.frames[60].x);
            assert.equal(await wells.locator('dd').nth(1).innerText(), models.landscapeNumber(expected.frames[60].value));
            assert(await wells.getByRole('button', { name: 'Next step', exact: true }).isDisabled());
            counts.wells += 2;
          }
        }
      }
      await wells.getByRole('button', { name: 'Previous step' }).click();
      assert.equal((await state(wells)).step, 59);
      await wells.getByRole('button', { name: 'Restart trace' }).click();
      assert.equal((await state(wells)).step, 0);
      await range(wells, 'Tilt δ', 0.15);
      await range(wells, 'Initial x', 0.8);
      await range(wells, 'Learning rate η', 0.12);
      await wells.getByRole('button', { name: 'Show final step' }).click();
      await capture(page, wells, `wells-local-${width}.png`);

      const stationary = lesson.locator('[data-lab="stationary"]');
      for (const kind of Object.keys(models.stationaryPresets)) {
        await stationary.getByRole('combobox').selectOption(kind);
        for (const degrees of [0, 45, 90, 225, 360]) {
          await range(stationary, 'Direction angle in degrees', degrees);
          for (const radius of [0, 0.2, 1]) {
            await range(stationary, 'Distance r', radius);
            const expected = models.stationaryState(kind, degrees, radius);
            assert.deepEqual(await state(stationary), JSON.parse(JSON.stringify(expected)));
            assert.equal(await stationary.locator('dd').nth(2).innerText(), `${models.landscapeNumber(expected.actual)}; ${models.landscapeNumber(expected.quadratic)}`);
            counts.stationary++;
          }
        }
      }
      await stationary.getByRole('combobox').selectOption('flatSaddle');
      await range(stationary, 'Direction angle in degrees', 90);
      await range(stationary, 'Distance r', 0.5);
      await capture(page, stationary, `stationary-quartic-${width}.png`);

      const noise = lesson.locator('[data-lab="noise"]');
      for (const direction of ['none', 'stable', 'unstable', 'both']) {
        await noise.getByRole('combobox').selectOption(direction);
        for (const rate of [0.02, 0.12, 0.25]) {
          await range(noise, 'Saddle learning rate η', rate);
          for (const amplitude of [0, 0.15, 0.5]) {
            await range(noise, 'Noise amplitude a', amplitude);
            for (const initialY of [-0.05, 0, 0.001]) {
              await range(noise, 'Initial y perturbation', initialY);
              const expected = models.saddleNoiseState(direction, rate, amplitude, initialY);
              assert.equal((await state(noise)).step, 0);
              await noise.getByRole('button', { name: 'Show final step' }).click();
              assert.deepEqual((await state(noise)).point, expected.frames.at(-1).point);
              assert.equal(await noise.locator('dd').first().innerText(), expected.frames.at(-1).point.map(models.landscapeNumber).join(', '));
              if (expected.stopped) assert.match(await noise.locator('p[role=status]').innerText(), /Stopped before update/);
              counts.noise++;
            }
          }
        }
      }
      await geometry(lesson);
      await noise.getByRole('combobox').selectOption('unstable');
      await range(noise, 'Saddle learning rate η', 0.25);
      await range(noise, 'Noise amplitude a', 0.5);
      await range(noise, 'Initial y perturbation', 0);
      await noise.getByRole('button', { name: 'Show final step' }).click();
      assert.match(await noise.locator('p[role=status]').innerText(), /Stopped before update/);
      await capture(page, noise, `noise-window-exit-${width}.png`);
      await noise.getByRole('button', { name: 'Previous step' }).click();
      await noise.getByRole('button', { name: 'Restart trace' }).click();
      await noise.getByRole('combobox').selectOption('stable');
      await range(noise, 'Saddle learning rate η', 0.12);
      await range(noise, 'Noise amplitude a', 0.15);
      await noise.getByRole('button', { name: 'Next step', exact: true }).focus();
      await page.keyboard.press('Space');
      assert.equal((await state(noise)).step, 1);
      await noise.getByRole('button', { name: 'Show final step' }).click();
      await capture(page, noise, `noise-stable-${width}.png`);

      const symmetry = lesson.locator('[data-lab="symmetry"]');
      for (const exponent of [-3, -1, 0, 2, 3]) {
        await range(symmetry, 'Scale exponent log₂(s)', exponent);
        for (const direction of ['normal', 'tangent']) {
          await symmetry.getByRole('combobox').selectOption(direction);
          for (const displacement of [-0.3, 0, 0.1, 0.3]) {
            await range(symmetry, 'Signed displacement ε', displacement);
            const expected = models.symmetryState(exponent, displacement, direction);
            assert.deepEqual(await state(symmetry), JSON.parse(JSON.stringify(expected)));
            assert.equal(await symmetry.locator('dd').nth(3).innerText(), expected.eigenvalues.map(models.landscapeNumber).join(', '));
            counts.symmetry++;
          }
        }
      }
      await capture(page, symmetry, `symmetry-tangent-${width}.png`);
      await symmetry.getByRole('combobox').selectOption('normal');
      await range(symmetry, 'Scale exponent log₂(s)', 2);
      await range(symmetry, 'Signed displacement ε', 0.1);
      await capture(page, symmetry, `symmetry-normal-${width}.png`);
      await geometry(lesson);

      const anchors = await lesson.locator('.lesson-intro a[href^="#"]').evaluateAll(nodes => nodes.map(node => node.hash.slice(1)));
      assert.equal(anchors.length, 9);
      for (const anchor of anchors) {
        assert.equal(await page.locator(`[id="${anchor}"]`).count(), 1, anchor);
        await lesson.locator(`.lesson-intro a[href="#${anchor}"]`).click();
        await page.waitForFunction(id => { const top = document.getElementById(id).getBoundingClientRect().top; return top >= -2 && top < innerHeight; }, anchor);
      }
      assert.equal(await lesson.locator('.landscape-practice').count(), 7);
      await lesson.locator('.landscape-practice summary').first().focus();
      await page.keyboard.press('Enter');
      assert.equal(await lesson.locator('.landscape-practice details[open]').count(), 1);
      await lesson.locator('details').evaluateAll(nodes => nodes.forEach(node => node.open = true));
      const text = await lesson.innerText();
      for (const [key, example] of Object.entries(examples)) {
        assert(text.includes(example.code), `${key} complete code rendered`);
        assert(text.includes(example.expected), `${key} complete output rendered`);
      }
      assert.equal(await lesson.locator('.katex-error').count(), 0);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
      assert.deepEqual(errors, []);
      results.push({ width, counts, anchors: anchors.length, completePrograms: 10, practiceGroups: 7, sources: await lesson.locator('.lesson-sources a').count(), pageErrors: errors, pageOverflow: false });
      await page.close();
    }
  } finally { await browser.close(); }
  fs.writeFileSync(path.join(folder, 'results.json'), JSON.stringify({ at: new Date().toISOString(), status: 'passed', results }, null, 2));
  console.log(JSON.stringify(results, null, 2));
})().catch(error => { console.error(error); process.exitCode = 1; });

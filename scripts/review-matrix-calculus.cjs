const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const directory = path.resolve(__dirname, '../scratch/matrix-calculus-browser');
fs.mkdirSync(directory, { recursive: true });
const number = value => {
  if (value === null) return 'undefined';
  if (value === 0 || Object.is(value, -0)) return '0';
  if (Math.abs(value) < 0.00001 || Math.abs(value) >= 100000) return value.toExponential(3);
  return String(Number(value.toFixed(6)));
};
const pair = values => '(' + values.map(number).join(', ') + ')';
async function capture(page, target, name) {
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = 'hidden'));
  await target.screenshot({ path: path.join(directory, name) });
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = ''));
}
async function range(input, value) {
  await input.focus();
  await input.press('Home');
  for (let index = -3; index < value; index += 0.5) await input.press('ArrowRight');
  assert.equal(Number(await input.inputValue()), value);
}
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  const errors = [];
  for (const width of [1440, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(process.env.LESSON_URL || 'http://127.0.0.1:5173/learn/path/full-curriculum/matrix-calculus-jacobians');
    await page.locator('[data-lab="local-jacobian"]').waitFor();
    assert.equal(await page.locator('[data-lab]').count(), 4);
    assert.equal(await page.locator('.python-example').count(), 10);
    assert.equal(await page.locator('.katex-error').count(), 0);
    const anchors = await page.locator('.lesson-intro nav a').evaluateAll(nodes => nodes.map(node => node.hash.slice(1)));
    assert.equal(anchors.length, 10);
    assert.equal(new Set(anchors).size, 10);
    for (const id of anchors) assert.equal(await page.locator('[id="' + id + '"]').count(), 1, 'anchor ' + id);
    await page.locator('.lesson-intro nav a').nth(1).click();
    assert.ok(page.url().endsWith('#2-one-slope-for-each-input-output-pair'));
    assert.equal(await page.locator('a[href="/learn/path/full-curriculum/tensor-algebra-einsum-notation"]').count(), 1);
    assert.equal(await page.locator('section.calculus-practice').count(), 6);
    const firstHint = page.locator('section.calculus-practice').first().locator('summary').first();
    await firstHint.focus();
    await page.keyboard.press('Enter');
    assert.equal(await firstHint.locator('..').getAttribute('open'), '');
    await page.keyboard.press('Space');
    assert.equal(await firstHint.locator('..').getAttribute('open'), null);
    for (const [name, heading] of [['slopes', /^1\./], ['jacobian-reading', /^2\./], ['matrix-gradients', /^6\./], ['practice-reading', /^10\./]]) {
      await page.getByRole('heading', { name: heading }).evaluate(node => window.scrollTo(0, window.scrollY + node.getBoundingClientRect().top - 100));
      await page.screenshot({ path: path.join(directory, name + '-' + width + '.png') });
    }
    const local = page.locator('[data-lab="local-jacobian"]');
    let localStates = 0;
    for (const input of [[2, 3], [0, 0], [-3, 3], [1, -1]]) {
      await range(local.locator('input').nth(0), input[0]);
      await range(local.locator('input').nth(1), input[1]);
      for (const [key, direction] of Object.entries({ mixed: [1, -2], first: [1, 0], second: [0, 1], zero: [0, 0] })) {
        await local.locator('select').nth(0).selectOption(key);
        for (const step of [-0.5, -0.25, 0, 0.01, 0.125, 0.25, 0.5]) {
          await local.locator('select').nth(1).selectOption(String(step));
          const rate = [2 * input[0] * direction[0] + direction[1], input[1] * direction[0] + input[0] * direction[1]];
          const predicted = rate.map(value => value * step);
          const expectedError = [step ** 2 * direction[0] ** 2, step ** 2 * direction[0] * direction[1]];
          const text = await local.locator('[data-result="local"]').innerText();
          assert.ok(text.includes('Rate Jv=' + pair(rate)), text);
          assert.ok(text.includes('Predicted change=' + pair(predicted)), text);
          const displayedError = text.match(/Actual − predicted=\(([^)]+)\)/)[1].split(',').map(Number);
          displayedError.forEach((value, index) => assert.ok(Math.abs(value - expectedError[index]) < 1e-9, text));
          assert.equal(await local.locator('.calculus-curve').evaluateAll(nodes => nodes.some(node => /NaN|Infinity/.test(node.getAttribute('d')))), false);
          localStates++;
        }
      }
    }
    await local.getByRole('button', { name: 'Reset local experiment', exact: true }).click();
    assert.equal(await local.locator('select').nth(1).inputValue(), '0.25');
    await capture(page, local, 'local-' + width + '.png');
    const chain = page.locator('[data-lab="chain-rule"]');
    let chainStates = 0;
    for (const mode of ['forward', 'reverse']) {
      await chain.locator('select').nth(0).selectOption(mode);
      for (const input of [[1, 2], [0, 0], [-1, 1], [2, -1]]) {
        await chain.locator('select').nth(1).selectOption(input.join(','));
        for (const [weightKey, weights] of Object.entries({ difference: [1, -1], first: [1, 0], sum: [1, 1], zero: [0, 0] })) {
          await chain.locator('select').nth(2).selectOption(weightKey);
          for (const [seed, direction] of Object.entries({ first: [1, 0], second: [0, 1], mixed: [1, -1], zero: [0, 0] })) {
            await chain.locator('select').nth(3).selectOption(seed);
            assert.match(await chain.locator('[data-result="chain"]').innerText(), /Stage 1 of 4/);
            assert.equal(await chain.locator('.calculus-derivative').filter({ hasText: 'not propagated' }).count(), 3);
            await chain.getByRole('button', { name: 'Show complete trace', exact: true }).click();
            const [a, b] = input, [q, r] = weights;
            // Differentiate the expanded scalar quadratic, not the displayed chain model.
            const gradient = [2 * (q + r) * a + (4 * q - 2 * r) * b, (4 * q - 2 * r) * a + 2 * (4 * q + r) * b];
            const rate = gradient[0] * direction[0] + gradient[1] * direction[1];
            assert.ok((await chain.locator('[data-result="chain"]').innerText()).includes('g_x·v=' + number(rate)));
            if (mode === 'reverse') assert.ok((await chain.locator('.calculus-derivative').last().innerText()).endsWith(pair(gradient)));
            else assert.ok((await chain.locator('.calculus-derivative').last().innerText()).endsWith(number(rate)));
            chainStates++;
          }
        }
      }
    }
    await chain.getByRole('button', { name: 'Reset chain', exact: true }).click();
    for (let stage = 1; stage <= 3; stage++) {
      await chain.getByRole('button', { name: 'Propagate one stage', exact: true }).click();
      assert.match(await chain.locator('[data-result="chain"]').innerText(), new RegExp('Stage ' + (stage + 1) + ' of 4'));
    }
    await chain.getByRole('button', { name: 'Previous stage', exact: true }).click();
    assert.match(await chain.locator('[data-result="chain"]').innerText(), /Stage 3 of 4/);
    await chain.locator('select').nth(0).selectOption('reverse');
    await chain.getByRole('button', { name: 'Show complete trace', exact: true }).click();
    await capture(page, chain, 'reverse-' + width + '.png');

    const affine = page.locator('[data-lab="affine-gradient"]');
    let affineStates = 0;
    const fixtures = {
      ordinary: [[[1, 2], [3, -1]], [[1, 0], [0, 2]]],
      repeated: [[[1, 2], [1, 2]], [[1, 0], [1, 0]]],
      zero: [[[0, 0], [0, 0]], [[1, 0], [0, 2]]],
    };
    for (const [fixture, [X, T]] of Object.entries(fixtures)) {
      await affine.locator('select').nth(0).selectOption(fixture);
      for (const reduction of ['sum', 'mean']) {
        await affine.locator('select').nth(1).selectOption(reduction);
        const divisor = reduction === 'mean' ? 2 : 1;
        const G = X.map((row, i) => [(row[0] + 2 * row[1] - T[i][0]) / divisor, (-row[0] + row[1] + 1 - T[i][1]) / divisor]);
        for (const parameter of ['weight', 'bias']) {
          await affine.locator('select').nth(2).selectOption(parameter);
          for (let row = 0; row < (parameter === 'weight' ? 2 : 1); row++) {
            for (let column = 0; column < 2; column++) {
              await affine.getByRole('button', { name: 'Select ∂L/∂' + (parameter === 'weight' ? 'W' : 'b') + ' row ' + row + ', column ' + column, exact: true }).click();
              const expected = G.reduce((sum, gradient, i) => sum + gradient[column] * (parameter === 'weight' ? X[i][row] : 1), 0);
              assert.equal(Number(await affine.locator('[data-gradient]').getAttribute('data-gradient')), expected);
              assert.equal(await affine.locator('.calculus-matrix button[aria-pressed="true"]').count(), 1);
              affineStates++;
            }
          }
        }
      }
    }
    await affine.getByRole('button', { name: 'Reset batch', exact: true }).click();
    await capture(page, affine, 'affine-' + width + '.png');
    await affine.locator('select').nth(2).selectOption('bias');
    await affine.getByRole('button', { name: 'Select ∂L/∂b row 0, column 1', exact: true }).click();
    assert.equal(Number(await affine.locator('[data-gradient]').getAttribute('data-gradient')), -1.5);
    await capture(page, affine, 'bias-' + width + '.png');

    const difference = page.locator('[data-lab="finite-difference"]');
    let differenceStates = 0;
    for (const kind of ['cubic', 'absolute']) {
      await difference.locator('select').nth(0).selectOption(kind);
      for (const point of [0, 0.3, -0.3, 2]) {
        await difference.locator('select').nth(1).selectOption(String(point));
        for (let index = 0; index < 10; index++) {
          await difference.locator('select').nth(2).selectOption({ value: String(index) });
          assert.equal(await difference.locator('select').nth(2).inputValue(), String(index));
          const text = await difference.locator('[data-result="difference"]').innerText();
          const exact = kind === 'cubic' ? 3 * point * point : point === 0 ? null : Math.sign(point);
          assert.ok(text.includes('Known derivative: ' + number(exact)), text);
          assert.equal(await difference.locator('p[role="status"]').count(), exact === null ? 1 : 0);
          if (exact === null) {
            assert.ok(text.includes('left/backward: -1'));
            assert.ok(text.includes('Right/forward: 1'));
            assert.ok(text.includes('central: 0'));
          }
          assert.equal(await difference.locator('.calculus-curve').evaluateAll(nodes => nodes.some(node => /NaN|Infinity/.test(node.getAttribute('d')))), false);
          differenceStates++;
        }
      }
    }
    await difference.getByRole('button', { name: 'Reset difference check', exact: true }).click();
    await capture(page, difference, 'difference-' + width + '.png');
    await difference.locator('select').nth(0).selectOption('absolute');
    await difference.locator('select').nth(1).selectOption('0');
    await capture(page, difference, 'kink-' + width + '.png');
    await difference.locator('summary').first().focus();
    await page.keyboard.press('Enter');
    assert.equal(await difference.locator('details').first().getAttribute('open'), '');

    await page.locator('.matrix-calculus-lesson').evaluate(node => node.querySelectorAll('details').forEach(item => item.open = true));
    for (const [index, name] of ['jacobian-figure', 'branch-figure', 'bias-figure', 'square-figure'].entries()) {
      await capture(page, page.locator('.calculus-inline').nth(index), name + '-' + width + '.png');
    }
    for (const lab of [local, chain, affine, difference]) {
      await lab.locator('input,select,button').first().focus();
      await page.keyboard.press('Tab');
      assert.equal(await lab.evaluate(node => node.contains(document.activeElement)), true);
      assert.equal(await lab.evaluate(node => node.scrollWidth > node.clientWidth + 2), false, await lab.getAttribute('data-lab') + ': overflow');
      assert.ok((await lab.locator('input,select,button').evaluateAll(nodes => nodes.map(node => node.getBoundingClientRect().height))).every(height => height >= 43));
    }
    assert.equal(await page.locator('.katex-error').count(), 0);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 2), false);
    const sources = await page.locator('.lesson-sources a').evaluateAll(nodes => nodes.map(node => ({ href: node.href, rel: node.rel })));
    assert.equal(sources.length, 9);
    assert.ok(sources.every(source => source.rel.includes('noopener') && source.rel.includes('noreferrer')));
    const mathOverflow = await page.locator('.katex-display').evaluateAll(nodes => nodes.filter(node => node.scrollWidth > node.clientWidth + 2).map(node => node.textContent));
    assert.deepEqual(mathOverflow, [], 'Display equations must fit the reviewed viewport.');
    results.push({ width, localStates, chainStates, affineStates, differenceStates, programs: 10, anchors: 10, sources: sources.length, mathOverflow });
    console.log('Matrix Calculus ' + width + 'px passed.');
    await page.close();
  }
  assert.deepEqual(errors, []);
  fs.writeFileSync(path.join(directory, 'results.json'), JSON.stringify({ date: new Date().toISOString(), results, errors }, null, 2));
  await browser.close();
})().catch(error => { console.error(error); process.exit(1); });

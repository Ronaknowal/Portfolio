const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const directory = path.resolve(__dirname, '../scratch/vector-tensor-browser');
fs.mkdirSync(directory, { recursive: true });
const number = value => String(Number(value.toFixed(3)));
const pair = values => '(' + values.map(number).join(', ') + ')';

async function setRange(input, value, minimum) {
  await input.focus();
  await input.press('Home');
  for (let step = minimum; step < value; step++) await input.press('ArrowRight');
  assert.equal(Number(await input.inputValue()), value);
}

async function capture(page, element, filename) {
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = 'hidden'));
  await element.screenshot({ path: path.join(directory, filename) });
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = ''));
}

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const errors = [], results = [];
  for (const width of [1440, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(process.env.LESSON_URL || 'http://127.0.0.1:5173/learn/path/full-curriculum/vectors-matrices-tensor-operations');
    await page.locator('[data-lab="vector-projection"]').waitFor();
    assert.equal(await page.locator('[data-lab]').count(), 4);
    assert.equal(await page.locator('.python-example').count(), 10);
    assert.equal(await page.locator('.katex-error').count(), 0);
    assert.ok(await page.locator('.katex-display').count() >= 12);
    const anchors = await page.locator('.lesson-intro nav a').evaluateAll(nodes => nodes.map(node => node.hash.slice(1)));
    assert.equal(anchors.length, 10);
    assert.equal(new Set(anchors).size, 10);
    for (const id of anchors) assert.equal(await page.locator('[id="' + id + '"]').count(), 1);
    await page.locator('.lesson-intro nav a').last().click();
    assert.ok(page.url().endsWith('#10-practise-explain-and-check-readiness'));
    assert.ok(await page.locator('a[href="/learn/path/full-curriculum/matrix-decompositions-svd-qr-cholesky-lu"]').count());
    assert.ok((await page.locator('.python-example pre').allTextContents()).every(text => text.trim()));
    const practice = page.locator('section.vector-practice');
    assert.equal(await practice.count(), 6);
    assert.equal(await practice.locator('details[open]').count(), 0);
    await practice.first().locator('summary').first().focus();
    await page.keyboard.press('Enter');
    assert.equal(await practice.first().locator('details').first().getAttribute('open'), '');
    await page.keyboard.press('Space');
    assert.equal(await practice.first().locator('details').first().getAttribute('open'), null);

    const projection = page.locator('[data-lab="vector-projection"]');
    let projectionStates = 0;
    for (const direction of ['diagonal', 'horizontal', 'zero']) {
      await projection.locator('select').selectOption(direction);
      for (const vector of [[3, 2], [-1, 2], [-2, -1], [0, 0], [4, -4], [-4, 4]]) {
        await setRange(projection.locator('input').nth(0), vector[0], -4);
        await setRange(projection.locator('input').nth(1), vector[1], -4);
        const u = direction === 'diagonal' ? [2, 1] : direction === 'horizontal' ? [1, 0] : [0, 0];
        const dot = vector[0] * u[0] + vector[1] * u[1];
        assert.ok((await projection.locator('[data-result="dot"]').innerText()).endsWith(number(dot)));
        if (direction === 'zero') {
          assert.equal(await projection.locator('[data-result="projection"]').count(), 0);
          assert.match(await projection.locator('p[role="status"]').innerText(), /no direction/);
        } else {
          const coefficient = dot / (u[0] ** 2 + u[1] ** 2);
          const projected = u.map(value => coefficient * value);
          assert.ok((await projection.locator('[data-result="projection"]').innerText()).endsWith(pair(projected)));
          assert.ok((await projection.locator('[data-result="residual"]').innerText()).endsWith(pair(vector.map((value, index) => value - projected[index]))));
          const circles = await projection.locator('.vector-endpoint.original').evaluateAll(nodes => nodes.map(node => [Number(node.getAttribute('cx')), Number(node.getAttribute('cy'))]));
          assert.deepEqual(circles, [[170 + 22 * vector[0], 170 - 22 * vector[1]]]);
        }
        projectionStates++;
      }
    }
    await projection.getByRole('button', { name: 'Reset projection', exact: true }).click();
    assert.equal(await projection.locator('input').first().inputValue(), '1');
    assert.equal(await projection.locator('select').inputValue(), 'diagonal');
    await capture(page, projection, 'projection-' + width + '.png');

    const maps = { shear: [[1, 1], [0, 1]], identity: [[1, 0], [0, 1]], rotate: [[0, -1], [1, 0]], reflect: [[-1, 0], [0, 1]], stretch: [[2, 0], [0, 1]], collapse: [[1, 1], [0, 0]], zero: [[0, 0], [0, 0]] };
    const map = page.locator('[data-lab="linear-map"]');
    let mapStates = 0;
    for (const [key, matrix] of Object.entries(maps)) {
      await map.locator('select').selectOption(key);
      for (const vector of [[2, 1], [-3, 3], [0, 0], [3, 3]]) {
        await setRange(map.locator('input').nth(0), vector[0], -3);
        await setRange(map.locator('input').nth(1), vector[1], -3);
        const output = matrix.map(row => row[0] * vector[0] + row[1] * vector[1]);
        assert.ok((await map.locator('[data-result="map-output"]').innerText()).endsWith(pair(output)));
        const endpoint = await map.locator('svg').nth(1).locator('.vector-endpoint.original').evaluate(node => [Number(node.getAttribute('cx')), Number(node.getAttribute('cy'))]);
        const domains = await map.locator('svg').evaluateAll(nodes => nodes.map(node => node.dataset.domain));
        assert.equal(domains[0], domains[1], 'input/output use one common range');
        const [lower, upper] = domains[0].split(',').map(Number);
        const scale = 264 / (upper - lower);
        assert.ok(Math.abs(endpoint[0] - (38 + (output[0] - lower) * scale)) < 1e-9);
        assert.ok(Math.abs(endpoint[1] - (302 - (output[1] - lower) * scale)) < 1e-9);
        mapStates++;
      }
      if (key === 'collapse') await capture(page, map, 'collapse-' + width + '.png');
    }
    await map.getByRole('button', { name: 'Reset map', exact: true }).click();
    assert.equal(await map.locator('select').inputValue(), 'shear');
    await capture(page, map, 'shear-' + width + '.png');

    const product = page.locator('[data-lab="matrix-product"]');
    const fixtures = [
      { left: [[2, 1], [0, 3], [1, 2]], right: [[3, 2], [5, 4]] },
      { left: [[1, 2, 3]], right: [[1], [0], [-1]] },
      { left: [[1], [2], [3]], right: [[1, -1, 0]] },
      { left: [[0, 0]], right: [[3], [4]] },
      { left: [[.5, -2]], right: [[2, .5], [-1, 3]] },
      { left: [[20, -20, 0]], right: [[20], [-20], [20]] },
    ];
    let productStates = 0;
    for (const fixture of fixtures) {
      await product.locator('input').nth(0).fill(fixture.left.map(row => row.join(',')).join(';'));
      await product.locator('input').nth(1).fill(fixture.right.map(row => row.join(',')).join(';'));
      await product.getByRole('button', { name: 'Apply matrices', exact: true }).click();
      const expected = fixture.left.map(row => fixture.right[0].map((_, column) => row.reduce((sum, value, index) => sum + value * fixture.right[index][column], 0)));
      const outputTable = product.locator('.vector-value-matrix').nth(2);
      assert.deepEqual((await outputTable.locator('tbody tr').evaluateAll(rows => rows.map(row => [...row.querySelectorAll('button')].map(button => Number(button.textContent))))), expected);
      for (let row = 0; row < expected.length; row++) for (let column = 0; column < expected[0].length; column++) {
        await outputTable.getByRole('button', { name: new RegExp('^Output row ' + row + ', column ' + column + ':') }).click();
        let partial = 0;
        for (let term = 0; term <= fixture.right.length; term++) {
          assert.match(await product.locator('[data-result="partial"]').innerText(), new RegExp('Partial sum: ' + String(partial).replace('.', '\\.') + ' ·'));
          assert.equal(await product.locator('.vector-product-terms .is-included').count(), term);
          productStates++;
          if (term < fixture.right.length) {
            partial += fixture.left[row][term] * fixture.right[term][column];
            await product.getByRole('button', { name: 'Add next term', exact: true }).click();
          }
        }
        assert.equal(await product.getByRole('button', { name: 'Add next term', exact: true }).isDisabled(), true);
        await product.getByRole('button', { name: 'Remove last term', exact: true }).click();
        assert.equal(await product.locator('.vector-product-terms .is-included').count(), fixture.right.length - 1);
      }
    }
    const preserved = await product.locator('.vector-product-grid').innerText();
    for (const invalid of ['', '1,', '1;2,3', '21', 'NaN', '1,2,3,4', '1,2']) {
      await product.locator('input').first().fill(invalid);
      await product.getByRole('button', { name: 'Apply matrices', exact: true }).click();
      assert.equal(await product.getByRole('alert').count(), 1);
      assert.equal(await product.locator('.vector-product-grid').innerText(), preserved);
    }
    await product.getByRole('button', { name: 'Reset product', exact: true }).click();
    assert.equal(await product.getByRole('alert').count(), 0);
    await product.getByRole('button', { name: 'Add next term', exact: true }).click();
    await capture(page, product, 'product-' + width + '.png');

    const tensor = page.locator('[data-lab="tensor-reduction"]');
    let tensorStates = 0;
    for (const dataset of ['ramp', 'repeat', 'impulse']) {
      await tensor.locator('select').nth(0).selectOption(dataset);
      for (let axis = 0; axis < 3; axis++) {
        await tensor.locator('select').nth(1).selectOption(String(axis));
        const dimensions = [2, 2, 3], remaining = [0, 1, 2].filter(value => value !== axis);
        for (let row = 0; row < dimensions[remaining[0]]; row++) for (let column = 0; column < dimensions[remaining[1]]; column++) {
          await tensor.getByRole('button', { name: new RegExp('^Output row ' + row + ', column ' + column + ':') }).click();
          const selected = await tensor.locator('.vector-tensor-source .is-selected').evaluateAll(cells => cells.map(cell => ({ coordinate: cell.dataset.coordinate.split(',').map(Number), value: Number(cell.querySelector('strong').textContent) })));
          assert.equal(selected.length, dimensions[axis]);
          for (const cell of selected) {
            assert.equal(cell.coordinate[remaining[0]], row);
            assert.equal(cell.coordinate[remaining[1]], column);
            const [session, time, channel] = cell.coordinate;
            const expected = dataset === 'ramp' ? 6 * session + 3 * time + channel : dataset === 'repeat' ? 10 * time + channel : session === 1 && time === 0 && channel === 2 ? 12 : 0;
            assert.equal(cell.value, expected);
          }
          const mean = selected.reduce((sum, cell) => sum + cell.value, 0) / selected.length;
          assert.match(await tensor.locator('[data-result="reduction"]').innerText(), new RegExp('= ' + number(mean).replace('.', '\\.') + '\\.'));
          tensorStates++;
        }
      }
    }
    await tensor.getByRole('button', { name: 'Reset tensor', exact: true }).click();
    assert.equal(await tensor.locator('select').nth(1).inputValue(), '1');
    await capture(page, tensor, 'tensor-' + width + '.png');
    for (const [index, name] of ['addition', 'composition', 'reindex'].entries()) {
      await capture(page, page.locator('.vector-inline-figure').nth(index), name + '-' + width + '.png');
    }
    for (const lab of [projection, map, product, tensor]) {
      await lab.locator('input,select,button').first().focus();
      await page.keyboard.press('Tab');
      assert.equal(await lab.evaluate(node => node.contains(document.activeElement)), true);
      assert.equal(await lab.evaluate(node => node.scrollWidth > node.clientWidth + 2), false);
      assert.ok((await lab.locator('input,select,button').evaluateAll(nodes => nodes.map(node => node.getBoundingClientRect().height))).every(height => height >= 43));
    }
    await page.locator('.vectors-lesson').evaluate(node => node.querySelectorAll('details').forEach(details => details.open = true));
    assert.equal(await page.locator('.katex-error').count(), 0);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 2), false);
    const sources = await page.locator('.lesson-sources a').evaluateAll(nodes => nodes.map(node => ({ href: node.href, rel: node.rel })));
    assert.ok(sources.length >= 10);
    assert.ok(sources.every(source => source.rel.includes('noopener') && source.rel.includes('noreferrer')));
    results.push({ width, projectionStates, mapStates, productStates, tensorStates, programs: 10, anchors: 10, sourceLinks: sources.length });
    console.log('Vectors ' + width + 'px checks passed.');
    await page.close();
  }
  assert.deepEqual(errors, []);
  fs.writeFileSync(path.join(directory, 'results.json'), JSON.stringify({ date: new Date().toISOString(), results, errors }, null, 2));
  await browser.close();
})().catch(error => { console.error(error); process.exit(1); });

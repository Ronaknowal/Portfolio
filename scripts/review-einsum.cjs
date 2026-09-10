const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const directory = path.resolve(__dirname, '../scratch/einsum-browser');
fs.mkdirSync(directory, { recursive: true });
const number = value => Math.abs(value) < 1e-12 ? '0' : String(Number(value.toFixed(6)));
const pair = values => '(' + values.map(number).join(', ') + ')';
async function capture(page, locator, filename) {
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = 'hidden'));
  await locator.screenshot({ path: path.join(directory, filename) });
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = ''));
}
(async () => {
  const models = await import(pathToFileURL(path.resolve(__dirname, '../src/learn/data/einsum-models.js')));
  const examples = await import(pathToFileURL(path.resolve(__dirname, '../src/learn/data/einsum-examples.js')));
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  const errors = [];
  for (const width of [1440, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
    page.on('pageerror', error => errors.push(error.message));
    await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/tensor-algebra-einsum-notation');
    await page.locator('[data-lab="index-contraction"]').waitFor();
    assert.equal(await page.locator('[data-lab]').count(), 4);
    assert.equal(await page.locator('.python-example').count(), 10);
    assert.equal(await page.locator('.katex-error').count(), 0);
    const text = await page.locator('.einsum-lesson').innerText();
    assert.ok(text.includes('same index choice'));
    const anchors = await page.locator('.lesson-intro nav a').evaluateAll(nodes => nodes.map(node => node.hash.slice(1)));
    assert.equal(anchors.length, 10);
    for (const id of anchors) assert.equal(await page.locator('[id="' + id + '"]').count(), 1, id);
    await page.locator('.lesson-intro nav a').nth(2).click();
    assert.ok(page.url().endsWith('#3-separate-diagonals-from-reductions'));
    assert.equal(await page.locator('.einsum-practice').count(), 6);
    const hint = page.locator('.einsum-practice').first().locator('summary').first();
    await hint.focus(); await page.keyboard.press('Enter');
    assert.equal(await hint.locator('..').getAttribute('open'), '');
    await page.keyboard.press('Space');
    assert.equal(await hint.locator('..').getAttribute('open'), null);
    const workbench = page.locator('[data-lab="index-contraction"]');
    let contractionStates = 0;
    for (const [key, preset] of Object.entries(models.contractionPresets)) {
      await workbench.getByLabel('Starting operation').selectOption({ value: key });
      const expected = models.inspectContraction(preset.expression, preset.operands);
      assert.equal(await workbench.getByLabel('Explicit expression', { exact: true }).inputValue(), preset.expression);
      const cells = workbench.locator('.einsum-output button');
      assert.equal(await cells.count(), expected.cells.length);
      for (let index = 0; index < expected.cells.length; index++) {
        await cells.nth(index).click();
        assert.equal(await cells.nth(index).getAttribute('aria-pressed'), 'true');
        assert.ok((await workbench.locator('[data-result="sum"]').innerText()).endsWith('= ' + number(expected.cells[index].value)));
        assert.equal(await workbench.locator('.einsum-terms li').count(), expected.cells[index].terms.length);
        contractionStates++;
      }
    }
    await workbench.getByRole('button', { name: 'Reset inspector', exact: true }).click();
    await workbench.getByLabel('Explicit expression', { exact: true }).fill('ik,kj->ikj');
    await workbench.getByRole('button', { name: 'Inspect expression', exact: true }).click();
    assert.equal(await workbench.locator('.einsum-output button').count(), 8);
    const saved = await workbench.locator('[data-result="contract"]').innerText();
    for (const expression of ['ik,kj', 'ik,kj->ii', 'ik,kj->z', 'i,j->ij', 'ii->i']) {
      await workbench.getByLabel('Explicit expression', { exact: true }).fill(expression);
      await workbench.getByRole('button', { name: 'Inspect expression', exact: true }).click();
      assert.equal(await workbench.getByRole('alert').count(), 1);
      assert.equal(await workbench.locator('[data-result="contract"]').innerText(), saved);
    }
    await workbench.getByRole('button', { name: 'Reset inspector', exact: true }).click();
    await workbench.locator('.einsum-output button').nth(1).click();
    await capture(page, workbench, 'contraction-' + width + '.png');

    const attention = page.locator('[data-lab="attention-contraction"]');
    let attentionStates = 0;
    for (let batch = 0; batch < 2; batch++) for (let query = 0; query < 2; query++) for (let mask = 1; mask < 8; mask++) for (const scaled of [false, true]) {
      await attention.getByLabel(/^Batch/).selectOption({ value: String(batch) });
      await attention.getByLabel(/^Query position/).selectOption({ value: String(query) });
      await attention.getByLabel('Divide scores by √2', { exact: true }).setChecked(scaled);
      const allowed = Array.from({ length: 3 }, (_, key) => Boolean(mask & (1 << key)));
      for (let key = 0; key < 3; key++) await attention.getByLabel('Key ' + key, { exact: true }).setChecked(allowed[key]);
      const expected = models.attentionRow(batch, query, allowed, scaled);
      assert.equal(await attention.getByRole('alert').count(), 0);
      assert.ok((await attention.locator('[data-result="context"]').innerText()).includes(pair(expected.context)));
      for (const row of expected.rows) assert.ok((await attention.locator('[data-key="' + row.key + '"]').innerText()).includes('Weight ' + number(row.weight)));
      attentionStates++;
    }
    for (let key = 0; key < 3; key++) await attention.getByLabel('Key ' + key, { exact: true }).uncheck();
    assert.ok((await attention.getByRole('alert').innerText()).includes('No allowed key'));
    assert.equal(await attention.locator('[data-result="context"]').count(), 0);
    await capture(page, attention, 'all-masked-' + width + '.png');
    await attention.getByRole('button', { name: 'Reset attention', exact: true }).click();
    await attention.getByLabel('Key 1', { exact: true }).uncheck();
    assert.ok((await attention.locator('[data-result="context"]').innerText()).includes('(2, 1)'));
    await capture(page, attention, 'attention-' + width + '.png');
    await attention.getByRole('button', { name: 'Reset attention', exact: true }).click();

    const orders = page.locator('[data-lab="contraction-order"]');
    let orderStates = 0;
    for (const dimensions of [[5, 40, 2, 30], [40, 5, 30, 2], [2, 20, 3, 4], [1, 1, 1, 1], [100, 100, 100, 100]]) {
      for (let index = 0; index < 4; index++) await orders.getByLabel('Axis ' + ['a', 'b', 'c', 'd'][index], { exact: true }).fill(String(dimensions[index]));
      await orders.getByRole('button', { name: 'Compare orders', exact: true }).click();
      const expected = models.contractionOrders(dimensions);
      assert.equal(await orders.locator('[data-result="left-cost"]').innerText(), expected.left.total.toLocaleString() + ' total multiplications');
      assert.equal(await orders.locator('[data-result="right-cost"]').innerText(), expected.right.total.toLocaleString() + ' total multiplications');
      orderStates++;
    }
    const savedCost = await orders.locator('[data-result="left-cost"]').innerText();
    for (const value of ['', '0', '1.5', '101', '-2']) {
      await orders.getByLabel('Axis a', { exact: true }).fill(value);
      await orders.getByRole('button', { name: 'Compare orders', exact: true }).click();
      assert.equal(await orders.getByRole('alert').count(), 1);
      assert.equal(await orders.locator('[data-result="left-cost"]').innerText(), savedCost);
    }
    await orders.getByRole('button', { name: 'Reset dimensions', exact: true }).click();
    await capture(page, orders, 'orders-' + width + '.png');
    const basis = page.locator('[data-lab="tensor-basis"]');
    let basisStates = 0;
    for (const name of Object.keys(models.basisPresets)) for (let index = 0; index < models.basisVectors.length; index++) {
      await basis.getByLabel(/^New basis/).selectOption({ value: name });
      await basis.getByLabel(/^Old vector coordinates/).selectOption({ value: String(index) });
      const expected = models.basisChange(name, models.basisVectors[index]);
      const stateText = await basis.locator('[data-result="basis"]').innerText();
      assert.ok(stateText.includes('Measurement: ' + number(expected.measurement) + ' = ' + number(expected.newMeasurement)));
      assert.ok(stateText.includes('metric expression ' + number(expected.metricNormSquared)));
      const coordinates = await basis.locator('svg [x1], svg [cx]').evaluateAll(nodes => nodes.flatMap(node => ['x1', 'y1', 'x2', 'y2', 'cx', 'cy'].filter(name => node.hasAttribute(name)).map(name => Number(node.getAttribute(name)))));
      assert.ok(coordinates.every(Number.isFinite));
      basisStates++;
    }
    await basis.getByRole('button', { name: 'Reset basis', exact: true }).click();
    await capture(page, basis, 'basis-' + width + '.png');
    const basisSelect = basis.getByLabel(/^New basis/);
    await basisSelect.focus(); await basisSelect.press('Home'); await basisSelect.press('ArrowDown');
    assert.equal(await basisSelect.inputValue(), 'shear');
    await page.keyboard.press('Tab');
    assert.ok(await page.evaluate(() => document.activeElement.tagName === 'SELECT'));
    for (const [index, name] of [[0, 'diagonal'], [1, 'batch-pairs']]) await capture(page, page.locator('.einsum-inline').nth(index), name + '-' + width + '.png');
    for (const [prefix, name] of [['1.', 'axes-reading'], ['3.', 'rules-reading'], ['5.', 'covariance-reading'], ['6.', 'attention-reading'], ['8.', 'basis-reading'], ['10.', 'practice-reading']]) {
      const heading = page.getByRole('heading', { name: new RegExp('^' + prefix.replace('.', '\\.')) });
      await heading.evaluate(node => window.scrollTo(0, window.scrollY + node.getBoundingClientRect().top - 95));
      await page.screenshot({ path: path.join(directory, name + '-' + width + '.png') });
    }
    const sources = page.locator('.lesson-sources a');
    assert.equal(await sources.count(), 9);
    assert.equal(await page.locator('a[href="/learn/path/full-curriculum/randomized-linear-algebra"]').count(), 1);
    await capture(page, page.locator('.lesson-sources'), 'sources-' + width + '.png');
    await page.locator('.einsum-lesson details').evaluateAll(nodes => nodes.forEach(node => node.open = true));
    for (const example of Object.values(examples.einsumExamples)) {
      assert.ok((await page.locator('.einsum-lesson').innerText()).includes(example.expected));
    }
    const mathOverflow = await page.locator('.katex-display').evaluateAll(nodes => nodes.filter(node => node.scrollWidth > node.clientWidth + 2).map(node => ({ text: node.textContent.slice(0, 100), scroll: node.scrollWidth, client: node.clientWidth })));
    assert.deepEqual(mathOverflow, [], JSON.stringify(mathOverflow));
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'page overflow');
    const shortControls = await page.locator('.einsum-lab button, .einsum-lab select, .einsum-lab input:not([type=checkbox])').evaluateAll(nodes => nodes.filter(node => node.getBoundingClientRect().height < 43).map(node => node.outerHTML));
    assert.deepEqual(shortControls, []);
    results.push({ width, contractionStates, attentionStates, orderStates, basisStates, invalidUiGroups: 11, sources: 9, programs: 10, anchors: anchors.length, mathOverflow });
    await page.close();
  }
  await browser.close();
  assert.deepEqual(errors, []);
  fs.writeFileSync(path.join(directory, 'results.json'), JSON.stringify({ date: new Date().toISOString(), results, errors }, null, 2));
  console.log(JSON.stringify(results, null, 2));
})().catch(error => { console.error(error); process.exit(1); });



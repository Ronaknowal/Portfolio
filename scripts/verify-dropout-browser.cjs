const assert = require('node:assert/strict');
const fs = require('node:fs');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const base = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4194';
const directory = 'docs/teaching/evidence/dropout-implementation';
fs.mkdirSync(directory, { recursive: true });
const report = { checkedAt: new Date().toISOString(), base, status: 'running', checks: [] };
const passed = name => report.checks.push(name);

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const context = await browser.newContext({ viewport: { width: 1366, height: 1000 }, reducedMotion: 'reduce' });
    const page = await context.newPage();
    const errors = [], requests = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('request', request => requests.push(request.url()));
    await require('./lib/lesson-browser-fonts.cjs')(page);
    await page.goto(base + '/learn/path/full-curriculum/dropout-droppath-stochastic-depth?module=deep-learning-fundamentals');
    await page.locator('[data-lab="dropout-update"]').waitFor();
    assert.equal(await page.locator('.dropout-lesson [data-lab]').count(), 8);
    assert.equal(await page.locator('.katex-error').count(), 0);
    assert.ok(!requests.some(url => url.endsWith('dropout-experiments.py')));
    passed('Eight current-output investigations, valid math, no eager program request');
    const anchors = await page.locator('.dropout-lesson .lesson-intro a').evaluateAll(nodes => nodes.map(node => document.getElementById(node.hash.slice(1)) !== null));
    assert.ok(anchors.every(Boolean));
    passed('All integrated route anchors resolve');

    const lab = id => page.locator('[data-lab="dropout-' + id + '"]');
    await assert.match(await lab('update').locator('[data-result]').innerText(), /Output 2 · loss 0.5/);
    await lab('update').getByLabel('Update drop probability', { exact: true }).fill('1');
    assert.match(await lab('update').locator('[data-result]').innerText(), /Output 0 · loss 0.5 · weight gradient \[0, 0\]/);
    await lab('update').getByRole('button', { name: 'Reset', exact: true }).click();
    await lab('update').getByRole('button', { name: 'Apply one SGD update' }).click();
    assert.match(await lab('update').locator('[data-result]').innerText(), /Output 1.6 · loss 0.18/);
    await lab('update').getByRole('button', { name: 'Reset', exact: true }).click();
    await lab('update').getByLabel('Target', { exact: true }).fill('');
    assert.equal(await lab('update').locator('[aria-invalid=true]').count(), 1);
    assert.match(await lab('update').locator('[data-result]').innerText(), /Output 2 · loss 0.5/);
    await lab('update').getByLabel('Target', { exact: true }).fill('1');
    passed('Gradient update, p1 null, deterministic reset and invalid edit preserve valid result');

    await lab('expectation').getByLabel('Outcome drop probability', { exact: true }).fill('0.25');
    assert.match(await lab('expectation').locator('[data-result]').innerText(), /mean \[1, 2\] · variances \[0.33333, 1.33333\]/);
    await lab('expectation').getByRole('button', { name: 'Inspect a scaling mistake' }).click();
    assert.match(await lab('expectation').locator('[data-result]').innerText(), /mean \[0.5625, 1.125\]/);
    await lab('expectation').getByRole('button', { name: 'Reset', exact: true }).click();
    await lab('expectation').getByLabel('Outcome drop probability slider', { exact: true }).focus();
    await page.keyboard.press('ArrowRight');
    assert.notEqual(await lab('expectation').getByLabel('Outcome drop probability', { exact: true }).inputValue(), '0.5');
    passed('Probability-weighted outcomes, scaling contrast and keyboard slider');

    await lab('geometry').getByLabel('Mask geometry').selectOption('row');
    assert.match(await lab('geometry').locator('[data-result]').innerText(), /2 independent bits/);
    await lab('geometry').getByRole('button', { name: 'Hide only example 1 / channel 2' }).click();
    assert.equal(await lab('geometry').locator('.dropout-grid > [data-mask="0"]').count(), 4);
    await lab('geometry').getByLabel('Add to every input', { exact: true }).fill('1');
    assert.match(await lab('geometry').locator('.dropout-grid b').first().innerText(), /2 → 4/);
    await lab('geometry').getByLabel('Geometry drop probability', { exact: true }).fill('0');
    assert.equal(await lab('geometry').locator('.dropout-grid > [data-mask="0"]').count(), 0);
    await lab('geometry').getByLabel('Geometry drop probability', { exact: true }).fill('1');
    assert.equal(await lab('geometry').locator('.dropout-grid > [data-mask="0"]').count(), 16);
    await lab('geometry').getByLabel('Geometry mode').selectOption('eval');
    assert.equal(await lab('geometry').locator('.dropout-grid > [data-mask="0"]').count(), 0);
    await lab('geometry').getByRole('button', { name: 'Reset', exact: true }).click();
    passed('Broadcast grouping, editable bits and changed-input geometry');

    await lab('branch').getByLabel('Mask placement').selectOption('whole');
    assert.match(await lab('branch').locator('.dropout-join b').innerText(), /\[0, 0\]/);
    await lab('branch').getByLabel('Mask placement').selectOption('branch');
    await lab('branch').getByRole('button', { name: 'Change the residual input' }).click();
    assert.match(await lab('branch').locator('.dropout-join b').innerText(), /\[1, 3\]/);
    await lab('branch').getByRole('button', { name: 'Branch bit: 0 · drop', exact: true }).click();
    await lab('branch').getByLabel('Branch drop probability', { exact: true }).fill('1');
    assert.match(await lab('branch').locator('[data-result]').innerText(), /Dropping the correction preserves the direct path/);
    passed('Residual lane preserved across changed input and lost by whole-sum masking');

    await lab('depth').getByLabel('Schedule convention').selectOption('original');
    assert.match(await lab('depth').locator('[data-result]').innerText(), /Expected active branches: 2.75/);
    await lab('depth').getByLabel('Execution strategy').selectOption('lazy');
    assert.match(await lab('depth').locator('[data-result]').innerText(), /Branch calls: 2 \/ 4/);
    assert.equal(await lab('depth').getByText('F skipped', { exact: false }).count(), 2);
    assert.equal(await lab('depth').getByText('Raw F: not computed', { exact: false }).count(), 2);
    await lab('depth').getByLabel('Number of blocks', { exact: true }).fill('1');
    await lab('depth').getByLabel('Schedule convention').selectOption('zero-first');
    assert.match(await lab('depth').locator('[data-result]').innerText(), /Expected active branches: 1 \/ 1/);
    passed('Schedule conventions, conditional call counter and L1 boundary');

    await lab('mode').getByRole('button', { name: 'Apply this forward pass' }).click();
    assert.match(await lab('mode').locator('table').innerText(), /Batches tracked\s+1\s+2/);
    await lab('mode').getByRole('button', { name: 'Inspect clean ordinary evaluation' }).click();
    assert.match(await lab('mode').locator('[data-result]').innerText(), /\[-0.35355, 0.35355\]/);
    await lab('mode').getByRole('button', { name: 'Select MC mode only' }).click();
    assert.match(await lab('mode').locator('table').innerText(), /Batches tracked\s+1\s+1/);
    passed('no_grad leaves train buffers active; ordinary/MC evaluation preserve buffers');

    await lab('measured').getByLabel('Recorded seed').selectOption('2');
    assert.match(await lab('measured').locator('[data-result]').innerText(), /-0.004140226/);
    await lab('measured').getByText('Exact values at every plotted checkpoint', { exact: true }).click();
    const checkpoints = lab('measured').getByRole('table', { name: 'All six checkpoints for the displayed baseline and variant' });
    assert.equal(await checkpoints.locator('tbody tr').count(), 6);
    assert.match(await checkpoints.innerText(), /400/);
    await lab('measured').getByLabel('Recorded masking configuration').selectOption('0-row');
    assert.match(await lab('measured').locator('[data-result]').innerText(), /Identical comparison/);
    await lab('monte-carlo').getByLabel('Saved draws in prefix', { exact: true }).fill('1');
    assert.match(await lab('monte-carlo').locator('table').innerText(), /Undefined for one draw/);
    assert.match(await lab('monte-carlo').locator('[data-result]').innerText(), /disagreement 0\./);
    await lab('monte-carlo').getByLabel('Saved draws in prefix', { exact: true }).fill('100');
    assert.match(await lab('monte-carlo').locator('[data-result]').innerText(), /Entropy difference versus all 100 = 0/);
    passed('Recorded fit reversal, matched null comparison and real MC prefix boundaries');

    await page.getByText('Read the scratch mask and its axis contract', { exact: true }).click();
    await page.locator('.neural-program-source').first().waitFor();
    assert.match(await page.locator('.neural-program-source').first().innerText(), /def mask_values/);
    assert.ok(requests.some(url => url.endsWith('dropout-experiments.py')));
    passed('Canonical source is loaded only when its disclosure opens');
    for (const width of [1366, 390, 320]) {
      await page.setViewportSize({ width, height: 1000 });
      const overflow = await page.evaluate(() => ({
        page: document.documentElement.scrollWidth > innerWidth + 1,
        controls: [...document.querySelectorAll('.dropout-lesson input,.dropout-lesson select,.dropout-lesson button')].filter(node => {
          const box = node.getBoundingClientRect(); return box.width > 0 && (box.left < -1 || box.right > innerWidth + 1);
        }).map(node => node.getAttribute('aria-label') || node.textContent),
      }));
      assert.equal(overflow.page, false, JSON.stringify(overflow));
      assert.deepEqual(overflow.controls, []);
    }
    const greenSurfaces = await page.locator('.dropout-lesson .neural-lab, .dropout-lesson button').evaluateAll(nodes => nodes.filter(node => {
      const values = getComputedStyle(node).backgroundColor.match(/[\d.]+/g)?.map(Number) || [];
      return values[1] > values[0] + 3 && values[1] > values[2] + 3;
    }).length);
    assert.equal(greenSurfaces, 0);
    passed('Desktop/390/320 bounds and neutral/amber control surfaces');
    await page.setViewportSize({ width: 1366, height: 1000 });
    await lab('geometry').screenshot({ path: directory + '/geometry-desktop.png' });
    await page.setViewportSize({ width: 390, height: 900 });
    await lab('branch').screenshot({ path: directory + '/branch-mobile.png' });
    assert.deepEqual(errors, []);
    passed('No page errors; screenshots retained for independent visual inspection');
    report.status = 'passed';
    await context.close();
  } catch (error) {
    report.status = 'failed';
    report.failure = error.stack;
    process.exitCode = 1;
  } finally {
    await browser.close();
    fs.writeFileSync(directory + '/browser.json', JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify(report, null, 2));
  }
})();

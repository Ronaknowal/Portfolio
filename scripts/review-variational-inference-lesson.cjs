const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const output = path.resolve(__dirname, '../scratch/variational-inference-lesson-review');
fs.mkdirSync(output, { recursive: true });

async function capture(page, target, filename) {
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => { node.style.visibility = 'hidden'; }));
  await target.screenshot({ path: path.join(output, filename) });
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => { node.style.visibility = ''; }));
}

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  for (const width of [1440, 390, 320]) {
    const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
    await page.routeWebSocket('**', socket => socket.close());
    const errors = [], failedRequests = [], consoleErrors = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('requestfailed', request => failedRequests.push({ url: request.url(), failure: request.failure() }));
    page.on('console', message => {
      if (message.type() === 'error' && !message.text().startsWith('[vite] failed to connect to websocket.')) consoleErrors.push({ text: message.text(), location: message.location() });
    });
    await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/variational-inference?module=math-foundations');
    const balance = page.getByRole('region', { name: 'ELBO probability balance investigation', exact: true });
    await balance.waitFor({ timeout: 60000 });
    assert.equal(await page.locator('.python-example').count(), 7);
    assert.equal(await page.locator('.vi-practice').count(), 8);
    assert.equal(await page.locator('.katex-error').count(), 0);
    const anchors = await page.locator('.lesson-intro a[href^="#"]').evaluateAll(nodes => nodes.map(node => node.hash.slice(1)));
    assert.equal(anchors.length, 9);
    for (const id of anchors) {
      assert.equal(await page.locator(`[id="${id}"]`).count(), 1, id);
      await page.locator(`.lesson-intro a[href="#${id}"]`).click();
      await page.waitForFunction(anchor => { const top = document.getElementById(anchor).getBoundingClientRect().top; return top >= -1 && top < innerHeight; }, id);
    }
    await capture(page, balance, `balance-default-${width}.png`);
    await balance.getByRole('button', { name: 'Fit best allowed distribution', exact: true }).click();
    assert.match(await balance.getByRole('status').innerText(), /excess above that family gap is 0\./);
    assert.ok(Math.abs(Number(await balance.getByLabel('Probability of A', { exact: true }).inputValue()) - 0.20521309615767264) < 1e-10);
    await balance.getByLabel('Allowed family', { exact: true }).selectOption('free');
    await balance.getByRole('button', { name: 'Fit best allowed distribution', exact: true }).click();
    assert.match(await balance.locator('.vi-balance').innerText(), /KL gap\s+0/);
    await capture(page, balance, `balance-exact-${width}.png`);
    await balance.getByLabel('Probability of A', { exact: true }).fill('1');
    assert.match(await balance.locator('.vi-balance').innerText(), /1.60944/);
    await balance.getByText('Inspect each probability and ELBO term', { exact: true }).click();
    assert.equal(await balance.locator('tbody tr').count(), 3);
    await balance.getByRole('button', { name: 'Reset probability balance', exact: true }).focus();
    await page.keyboard.press('Enter');
    assert.equal(await balance.getByLabel('Allowed family', { exact: true }).inputValue(), 'restricted');

    const gaussian = page.getByRole('region', { name: 'Gaussian dependence investigation', exact: true });
    await capture(page, gaussian, `gaussian-positive-${width}.png`);
    await gaussian.getByLabel('Target correlation', { exact: true }).fill('-0.6');
    const decisions = await gaussian.locator('.vi-decision-table').innerText();
    assert.match(decisions, /0\.8/);
    assert.match(decisions, /3\.2/);
    assert.match(decisions, /1\.28/);
    await capture(page, gaussian, `gaussian-negative-${width}.png`);
    await gaussian.getByLabel('Gaussian approximation', { exact: true }).selectOption('full');
    assert.match(await gaussian.locator('dl').innerText(), /KL\(q \|\| p\)\s+0/);
    await gaussian.getByLabel('Gaussian approximation', { exact: true }).selectOption('marginals');
    await gaussian.getByLabel('Target correlation', { exact: true }).fill('0');
    assert.match(await gaussian.getByRole('status').innerText(), /exact/);
    await gaussian.getByRole('button', { name: 'Reset Gaussian comparison', exact: true }).click();

    const coordinates = page.getByRole('region', { name: 'Coordinate ascent investigation', exact: true });
    assert.equal(await coordinates.getByRole('button', { name: 'Previous update', exact: true }).isDisabled(), true);
    await coordinates.getByRole('button', { name: 'Update factor 1', exact: true }).click();
    assert.match(await coordinates.locator('dl').innerText(), /\(3\.4, 2\)/);
    await coordinates.getByRole('button', { name: 'Update factor 2', exact: true }).click();
    assert.match(await coordinates.locator('dl').innerText(), /\(3\.4, 0\.92\)/);
    await capture(page, coordinates, `coordinates-two-updates-${width}.png`);
    await coordinates.getByRole('button', { name: 'Inspect 24 updates', exact: true }).click();
    assert.match(await coordinates.locator('dl').innerText(), /1\.01771/);
    assert.equal(await coordinates.getByRole('button', { name: 'Update factor 1', exact: true }).isDisabled(), true);
    await coordinates.getByText('Coordinate means and objective history', { exact: true }).click();
    assert.equal(await coordinates.locator('tbody tr').count(), 25);
    await capture(page, coordinates, `coordinates-converging-${width}.png`);
    await coordinates.getByLabel('Coordinate target correlation', { exact: true }).fill('0');
    await coordinates.getByRole('button', { name: 'Inspect 24 updates', exact: true }).click();
    assert.match(await coordinates.locator('dl').innerText(), /Total KL\s+0/);
    await coordinates.getByRole('button', { name: 'Reset coordinate ascent', exact: true }).click();

    const mixture = page.getByRole('region', { name: 'Two-mode approximation investigation', exact: true });
    await capture(page, mixture, `mixture-broad-${width}.png`);
    await mixture.getByRole('button', { name: 'Candidate N(3,1)', exact: true }).click();
    assert.match(await mixture.locator('dl').innerText(), /0\.99865/);
    await capture(page, mixture, `mixture-one-mode-${width}.png`);
    await mixture.getByRole('button', { name: 'Candidate N(−3,1)', exact: true }).click();
    assert.match(await mixture.locator('dl').innerText(), /0\.00135/);
    await mixture.getByLabel('Target peak distance d', { exact: true }).fill('0');
    await mixture.getByLabel('Approximation mean', { exact: true }).fill('0');
    assert.match(await mixture.locator('dl').innerText(), /Approximate KL\(q \|\| p\)\s+0/);
    await mixture.getByLabel('Approximation standard deviation', { exact: true }).fill('0.3');
    await capture(page, mixture, `mixture-narrow-${width}.png`);
    await mixture.getByText('Density values at selected coordinates', { exact: true }).click();
    assert.equal(await mixture.locator('tbody tr').count(), 9);
    await mixture.getByRole('button', { name: 'Reset broad candidate', exact: true }).click();

    const gradient = page.getByRole('region', { name: 'Variational gradient investigation', exact: true });
    await capture(page, gradient, `gradient-default-${width}.png`);
    const before = await gradient.locator('.vi-gradient-flow').innerText();
    await gradient.getByLabel('Gradient seed', { exact: true }).fill('0');
    await gradient.getByRole('button', { name: 'Apply gradient seed', exact: true }).click();
    assert.match(await gradient.getByRole('alert').innerText(), /have not changed/);
    assert.equal(await gradient.locator('.vi-gradient-flow').innerText(), before);
    await gradient.getByLabel('Gradient seed', { exact: true }).fill('29');
    await gradient.getByRole('button', { name: 'Apply gradient seed', exact: true }).click();
    assert.equal(await gradient.getByRole('alert').count(), 0);
    assert.notEqual(await gradient.locator('.vi-gradient-flow').innerText(), before);
    await gradient.getByLabel('Independent noise draws', { exact: true }).selectOption('10');
    await gradient.getByLabel('Inspect noise draw', { exact: true }).fill('10');
    await gradient.getByRole('button', { name: 'Match q to target', exact: true }).click();
    assert.match(await gradient.getByRole('status').innerText(), /zero for every draw/);
    const matched = await gradient.locator('.vi-decision-table tbody tr').last().innerText();
    assert.match(matched, /Score\s+0\s+0/);
    await capture(page, gradient, `gradient-matched-${width}.png`);
    await gradient.getByText('Inspect gradient MCSE and per-draw ingredients', { exact: true }).click();
    assert.equal(await gradient.locator('.vi-table tbody tr').count(), 7);
    await gradient.getByLabel('Gradient approximation mean', { exact: true }).focus();
    await page.keyboard.press('ArrowRight');
    assert.equal(await gradient.getByLabel('Gradient approximation mean', { exact: true }).inputValue(), '1.6');
    const focus = await gradient.getByLabel('Gradient approximation mean', { exact: true }).evaluate(node => ({ active: document.activeElement === node, width: getComputedStyle(node).outlineWidth }));
    assert.ok(focus.active && parseFloat(focus.width) >= 2);
    await gradient.getByRole('button', { name: 'Reset gradient experiment', exact: true }).click();
    assert.equal(await gradient.getByLabel('Gradient seed', { exact: true }).inputValue(), '7');
    await capture(page, page.locator('.vi-amortization'), `amortization-${width}.png`);

    const exercise = page.locator('.vi-practice').last();
    await exercise.getByText('Hint', { exact: true }).click();
    assert.equal(await exercise.locator('details').last().getAttribute('open'), null);
    await exercise.getByText('Show explained solution', { exact: true }).click();
    assert.match(await exercise.innerText(), /4\.8/);
    await capture(page, exercise, `practice-changed-model-${width}.png`);

    // Open optional mathematical derivations for the width check as well.
    await page.locator('.variational-inference-lesson > details').evaluateAll(nodes => nodes.forEach(node => { node.open = true; }));
    const geometry = await page.evaluate(() => {
      const math = [...document.querySelectorAll('.katex-display')].map(node => ({ text: node.textContent.slice(0, 100), scroll: node.scrollWidth, parent: node.parentElement.clientWidth }));
      const labels = [...document.querySelectorAll('.vi-lab svg text')].map(node => {
        const rect = node.getBoundingClientRect(), svg = node.ownerSVGElement.getBoundingClientRect();
        return { text: node.textContent, effectiveFont: parseFloat(getComputedStyle(node).fontSize) * node.ownerSVGElement.getScreenCTM().a, inside: rect.left >= svg.left - 1 && rect.right <= svg.right + 1 && rect.top >= svg.top - 1 && rect.bottom <= svg.bottom + 1 };
      });
      return { pageWidth: document.documentElement.scrollWidth, viewport: innerWidth, math, labels };
    });
    assert.ok(geometry.pageWidth <= width, JSON.stringify(geometry));
    assert.ok(geometry.math.every(item => item.scroll <= item.parent + 1), JSON.stringify(geometry.math));
    assert.ok(geometry.labels.every(item => item.inside && item.effectiveFont >= 13.5), JSON.stringify(geometry.labels));

    for (const [name, selector] of [['opening', '.lesson-intro'], ['dependence', '[id="3-see-what-a-restricted-family-loses"]'], ['gradient', '[id="6-differentiate-through-an-expectation"]'], ['amortization-context', '.vi-amortization'], ['references', '.lesson-sources']]) {
      await page.locator(selector).evaluate(node => window.scrollTo(0, node.getBoundingClientRect().top + window.scrollY - 180));
      await page.screenshot({ path: path.join(output, `reading-${name}-${width}.png`) });
    }
    assert.equal(errors.length, 0, JSON.stringify(errors));
    assert.ok(consoleErrors.every(item => item.text === 'Failed to load resource: net::ERR_NETWORK_ACCESS_DENIED' && item.location.url.startsWith('https://fonts.googleapis.com/css2?family=JetBrains+Mono')), JSON.stringify(consoleErrors));
    assert.ok(failedRequests.every(item => item.url.startsWith('https://fonts.googleapis.com/css2?family=JetBrains+Mono') && item.failure.errorText === 'net::ERR_NETWORK_ACCESS_DENIED'), JSON.stringify(failedRequests));
    results.push({ width, anchors, programs: 7, practices: 8, geometry, focus, errors, failedRequests, consoleErrors });
    await page.close();
  }
  fs.writeFileSync(path.join(output, 'results.json'), JSON.stringify(results, null, 2));
  await browser.close();
  console.log('VI desktop390/320 interaction, reading, math/geometry, keyboard, error and practice review passed.');
})();

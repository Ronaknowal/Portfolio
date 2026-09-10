const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const output = path.resolve(__dirname, '../scratch/maximum-likelihood-lesson-review');
fs.mkdirSync(output, { recursive: true });
async function capture(page, target, name) {
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => { node.style.visibility = 'hidden'; }));
  await target.screenshot({ path: path.join(output, name) });
  await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => { node.style.visibility = ''; }));
}
async function range(locator, value) {
  await locator.fill(String(value));
}
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  for (const width of [1440, 390, 320]) {
    const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
    await page.routeWebSocket('**', socket => socket.close());
    const errors = [];
    const failedRequests = [];
    const consoleErrors = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('requestfailed', request => failedRequests.push({ url: request.url(), failure: request.failure() }));
    page.on('console', message => { if (message.type() === 'error' && !message.text().startsWith('[vite] failed to connect to websocket.')) consoleErrors.push({ text: message.text(), location: message.location() }); });
    await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/maximum-likelihood-map-estimation?module=math-foundations');
    const binary = page.getByRole('region', { name: 'Bernoulli likelihood investigation', exact: true });
    const location = page.getByRole('region', { name: 'Measurement residual investigation', exact: true });
    const beta = page.getByRole('region', { name: 'Prior and posterior investigation', exact: true });
    const sampling = page.getByRole('region', { name: 'Repeated sample estimate investigation', exact: true });
    await binary.waitFor({ timeout: 60000 });
    assert.equal(await page.locator('.python-example').count(), 7);
    assert.equal(await page.locator('.katex-error').count(), 0);
    const anchors = await page.locator('.lesson-intro a[href^="#"]').evaluateAll(nodes => nodes.map(node => node.hash.slice(1)));
    for (const id of anchors) {
      assert.equal(await page.locator(`[id="${id}"]`).count(), 1, id);
      await page.locator(`.lesson-intro a[href="#${id}"]`).click();
      await page.waitForFunction(anchor => { const top = document.getElementById(anchor).getBoundingClientRect().top; return top >= -1 && top < innerHeight; }, id);
    }
    await capture(page, binary, `binary-default-${width}.png`);
    await range(binary.getByLabel('Candidate p', { exact: true }), .75);
    assert.match(await binary.locator('dl').innerText(), /0.10547/);
    await binary.getByLabel('Observed event', { exact: true }).selectOption('count');
    assert.match(await binary.locator('dl').innerText(), /0.42188/);
    await range(binary.getByLabel('Candidate p', { exact: true }), 1);
    assert.match(await binary.locator('dl').innerText(), /−∞/);
    await capture(page, binary, `binary-impossible-${width}.png`);
    await binary.getByLabel('Binary observations', { exact: true }).fill('1 2');
    await binary.getByRole('button', { name: 'Apply observations', exact: true }).click();
    assert.match(await binary.getByRole('alert').innerText(), /zeros and ones/);
    assert.equal(await binary.locator('.mle-data-strip span').count(), 4);
    await binary.getByLabel('Binary observations', { exact: true }).fill('');
    await binary.getByRole('button', { name: 'Apply observations', exact: true }).click();
    assert.match(await binary.locator('dl').innerText(), /Every p/);
    await capture(page, binary, `binary-empty-${width}.png`);
    await binary.getByLabel('Binary observations', { exact: true }).fill(Array(20).fill('1').concat(Array(20).fill('0')).join(' '));
    await binary.getByRole('button', { name: 'Apply observations', exact: true }).click();
    await binary.getByLabel('Observed event', { exact: true }).selectOption('sequence');
    await range(binary.getByLabel('Candidate p', { exact: true }), .5);
    assert.match(await binary.locator('dl').innerText(), /e-13/);
    await capture(page, binary, `binary-small-probability-${width}.png`);
    assert(await binary.locator('svg text').evaluateAll(nodes => nodes.every(element => {
      const bounds = element.getBoundingClientRect();
      const frame = element.ownerSVGElement.getBoundingClientRect();
      return bounds.left >= frame.left - 1 && bounds.right <= frame.right + 1;
    })));
    await binary.getByRole('button', { name: 'Reset binary experiment', exact: true }).focus();
    await page.keyboard.press('Enter');
    await binary.getByLabel('Candidate p', { exact: true }).focus();
    await page.keyboard.press('ArrowRight');
    assert.equal(await binary.getByLabel('Candidate p', { exact: true }).inputValue(), '0.51');
    const focusStyle = await binary.getByLabel('Candidate p', { exact: true }).evaluate(element => ({ outline: getComputedStyle(element).outlineWidth, active: document.activeElement === element }));
    assert(focusStyle.active && parseFloat(focusStyle.outline) >= 2);

    await capture(page, location, `residual-default-${width}.png`);
    await location.getByRole('button', { name: 'Use median midpoint', exact: true }).click();
    assert.equal(await location.getByLabel('Candidate center', { exact: true }).inputValue(), '3.5');
    await location.getByRole('button', { name: 'Add the outlier 27', exact: true }).click();
    await location.getByRole('button', { name: 'Use sample mean', exact: true }).click();
    assert.match(await location.locator('dl').innerText(), /8.6/);
    await location.getByText('Observation and residual table', { exact: true }).click();
    assert.equal(await location.locator('tbody tr').count(), 5);
    await capture(page, location, `residual-outlier-${width}.png`);
    await location.getByLabel('Measurements', { exact: true }).fill('Infinity');
    await location.getByRole('button', { name: 'Apply measurements', exact: true }).click();
    assert.match(await location.getByRole('alert').innerText(), /finite/);
    await location.getByLabel('Measurements', { exact: true }).fill('5 5 5');
    await location.getByRole('button', { name: 'Apply measurements', exact: true }).click();
    assert.match(await location.locator('dl').innerText(), /Squared cost Σ\(x−c\)²\s+0/);
    await capture(page, location, `residual-identical-${width}.png`);

    await capture(page, beta, `posterior-default-${width}.png`);
    assert.match(await beta.locator('dl').innerText(), /0.66667/);
    assert.match(await beta.locator('dl').innerText(), /0.625/);
    await beta.getByRole('button', { name: 'Four successes, uniform prior', exact: true }).click();
    assert.match(await beta.locator('dl').innerText(), /0.83333/);
    await capture(page, beta, `posterior-boundary-${width}.png`);
    await beta.getByRole('button', { name: 'No data, uniform prior', exact: true }).click();
    assert.match(await beta.locator('dl').innerText(), /Every p/);
    await capture(page, beta, `posterior-flat-${width}.png`);
    await range(beta.getByLabel('Prior beta', { exact: true }), 12);
    await range(beta.getByLabel('Failures', { exact: true }), 20);
    assert.match(await beta.locator('dl').innerText(), /0.0303/);

    await capture(page, sampling, `sampling-default-${width}.png`);
    await range(sampling.getByLabel('Assumed true p', { exact: true }), .2);
    await range(sampling.getByLabel('Observations per study', { exact: true }), 10);
    assert.match(await sampling.locator('dl').innerText(), /0.016/);
    await sampling.getByText('Exact mass table (rounded for display)', { exact: true }).click();
    assert.equal(await sampling.locator('tbody tr').count(), 11);
    await capture(page, sampling, `sampling-changed-${width}.png`);
    await range(sampling.getByLabel('Assumed true p', { exact: true }), 0);
    assert.equal(await sampling.locator('tbody tr').first().locator('td').last().innerText(), '1');
    await capture(page, page.locator('.mle-coordinate'), `coordinate-${width}.png`);
    const practice = page.locator('.mle-practice').first();
    await practice.getByText('Hint', { exact: true }).focus();
    await page.keyboard.press('Enter');
    assert.equal(await practice.locator('details').first().evaluate(element => element.open), true);
    assert.equal(await practice.locator('details').last().evaluate(element => element.open), false);
    await capture(page, practice, `independent-hint-${width}.png`);
    await practice.getByText('Show explained solution', { exact: true }).click();
    await capture(page, practice, `independent-solution-${width}.png`);
    const geometry = await page.evaluate(() => ({ page: document.documentElement.scrollWidth, viewport: innerWidth, svgText: [...document.querySelectorAll('.mle-lab svg text,.mle-coordinate svg text')].map(element => { const r = element.getBoundingClientRect(); const s = element.ownerSVGElement.getBoundingClientRect(); return { text: element.textContent, renderedSize: parseFloat(getComputedStyle(element).fontSize) * s.width / element.ownerSVGElement.viewBox.baseVal.width, left: r.left, right: r.right, svgLeft: s.left, svgRight: s.right }; }) }));
    assert(geometry.page <= width + 1, JSON.stringify(geometry));
    assert(geometry.svgText.every(row => row.renderedSize >= 13.5 && row.left >= row.svgLeft - 1 && row.right <= row.svgRight + 1));
    const equations = await page.locator('.maximum-likelihood-lesson .katex-display').evaluateAll(nodes => nodes.map(element => ({ width: element.clientWidth, scroll: element.scrollWidth })));
    assert(equations.every(row => row.scroll <= row.width + 1), JSON.stringify(equations));
    assert.equal(errors.length, 0, errors.join('\n'));
    assert(consoleErrors.every(row => row.location.url.startsWith('https://fonts.googleapis.com/')), JSON.stringify(consoleErrors));
    for (const heading of ['3-see-why-a-normal-fit-chooses-the-mean', '7-look-deeper-a-mode-needs-a-coordinate']) {
      await page.locator(`[id="${heading}"]`).evaluate(element => window.scrollTo({ top: window.scrollY + element.getBoundingClientRect().top - 180, behavior: 'instant' }));
      await page.screenshot({ path: path.join(output, `reading-${heading}-${width}.png`) });
    }
    results.push({ width, anchors, focusStyle, geometry, equations, errors, failedRequests, consoleErrors });
    await page.close();
  }
  fs.writeFileSync(path.join(output, 'results.json'), JSON.stringify(results, null, 2));
  await browser.close();
  console.log('MLE/MAP desktop/390/320 state changes, boundary/error handling, keyboard, anchors, math rendering and screenshots passed.');
})().catch(error => { console.error(error); process.exitCode = 1; });

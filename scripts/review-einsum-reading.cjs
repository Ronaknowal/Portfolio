const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const directory = path.resolve(__dirname, '../scratch/einsum-browser');
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  for (const width of [1440, 390, 320]) {
    const page = await browser.newPage({ viewport: { width, height: 1000 } });
    const errors = [];
    page.on('pageerror', error => errors.push(error.message));
    await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/tensor-algebra-einsum-notation');
    await page.locator('[data-lab="tensor-basis"]').waitFor();
    assert.ok((await page.locator('.einsum-lesson').innerText()).includes('NumPy 2.3.5'));
    const maskedEquation = page.locator('.katex-display').filter({ hasText: 'blocked' });
    assert.equal(await maskedEquation.count(), 1);
    assert.ok((await maskedEquation.innerText()).includes('allowed'));
    const attention = page.locator('[data-lab="attention-contraction"]');
    await attention.getByLabel('Key 1', { exact: true }).uncheck();
    assert.ok((await attention.locator('[data-key="1"]').innerText()).includes('Weight 0'));
    assert.ok((await attention.locator('[data-result="context"]').innerText()).includes('(2, 1)'));
    assert.ok((await attention.innerText()).includes('For an allowed key'));
    assert.ok((await attention.innerText()).includes('A blocked key has weight 0'));
    await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = 'hidden'));
    await maskedEquation.screenshot({ path: path.join(directory, 'final-masked-equation-' + width + '.png') });
    await attention.screenshot({ path: path.join(directory, 'final-masked-attention-' + width + '.png') });
    await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = ''));
    for (const key of [0, 2]) await attention.getByLabel('Key ' + key, { exact: true }).uncheck();
    assert.ok((await attention.getByRole('alert').innerText()).includes('No allowed key'));
    assert.equal(await attention.locator('[data-result="context"]').count(), 0);
    await attention.getByRole('button', { name: 'Reset attention', exact: true }).click();
    for (const prefix of ['1.', '3.', '5.', '6.', '8.', '10.']) {
      const heading = page.getByRole('heading', { name: new RegExp('^' + prefix.replace('.', '\\.')) });
      await heading.evaluate(node => window.scrollTo(0, window.scrollY + node.getBoundingClientRect().top - 95));
      await page.screenshot({ path: path.join(directory, 'final-reading-' + prefix.replace('.', '') + '-' + width + '.png') });
    }
    const basis = page.locator('[data-lab="tensor-basis"]');
    for (const name of ['identity', 'shear', 'stretch', 'rotation']) {
      await basis.getByLabel(/^New basis/).selectOption({ value: name });
      const labels = basis.locator('svg text').filter({ hasText: /^b[12]$/ });
      assert.equal(await labels.count(), 2);
      const outside = await labels.evaluateAll(nodes => nodes.some(node => {
        const box = node.getBoundingClientRect(), svg = node.ownerSVGElement.getBoundingClientRect();
        return box.left < svg.left || box.right > svg.right || box.top < svg.top || box.bottom > svg.bottom;
      }));
      assert.equal(outside, false);
      await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = 'hidden'));
      await basis.screenshot({ path: path.join(directory, 'final-basis-' + name + '-' + width + '.png') });
      await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = ''));
    }
    await page.locator('.einsum-lesson details').evaluateAll(nodes => nodes.forEach(node => node.open = true));
    const mathOverflow = await page.locator('.katex-display').evaluateAll(nodes => nodes.filter(node => node.scrollWidth > node.clientWidth + 2).map(node => ({ text: node.textContent.slice(0, 120), width: node.clientWidth, scroll: node.scrollWidth })));
    assert.deepEqual(mathOverflow, []);
    assert.equal(await page.locator('.katex-error').count(), 0);
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
    await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = 'hidden'));
    await page.locator('.lesson-sources').screenshot({ path: path.join(directory, 'final-sources-' + width + '.png') });
    assert.equal(await page.locator('.lesson-sources a').count(), 9);
    assert.deepEqual(errors, []);
    results.push({ width, ordinarySections: 6, labelledBasisFigures: 4, maskedEquationAndReadout: true, maskedInteractionCases: 2, mathOverflow, references: 9, pageOverflow: false, errors });
    await page.close();
  }
  await browser.close();
  fs.writeFileSync(path.join(directory, 'reading-results.json'), JSON.stringify({ date: new Date().toISOString(), results }, null, 2));
  console.log(JSON.stringify(results, null, 2));
})().catch(error => { console.error(error); process.exit(1); });



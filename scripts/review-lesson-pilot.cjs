const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const fs = require('fs');
const assert = require('assert/strict');

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  fs.mkdirSync('scratch/lesson-pilot-review', { recursive: true });
  const slugs = ['hypothesis-testing-confidence-intervals', 'bayesian-inference-conjugate-priors', 'spectral-graph-theory'];
  for (const width of [1440, 390]) {
    await page.setViewportSize({ width, height: 900 });
    for (const slug of slugs) {
      await page.goto(`${process.env.LEARNING_BASE_URL || 'http://127.0.0.1:5173'}/learn/topic/${slug}`);
      await page.locator('.lesson-lab').waitFor();
      assert.equal(await page.locator('.lesson-guide').count(), 0);
      assert.equal(await page.locator('.katex-error').count(), 0);
      const links = await page.locator('.lesson-intro a').evaluateAll(nodes => nodes.map(n => n.hash.slice(1)));
      for (const id of links) assert.equal(await page.locator(`[id="${id}"]`).count(), 1, id);
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), `${slug} overflows at ${width}px`);
      const answer = page.locator('.lesson-check details').first();
      await answer.locator('summary').click();
      assert.equal(await answer.getAttribute('open'), '');
      // Hide only the fixed site nav in element captures, so it cannot obscure the lab.
      await page.locator('.lesson-lab').screenshot({ path: `scratch/lesson-pilot-review/${slug}-${width}.png`, style: '.learn-nav { visibility: hidden !important; }' });
      if (slug.startsWith('hypothesis')) {
        const before = await page.locator('.lesson-results').innerText();
        await page.getByRole('slider', { name: 'Observations per experiment' }).fill('100');
        assert.notEqual(await page.locator('.lesson-results').innerText(), before);
        await page.getByLabel('Confidence level').selectOption('99');
        await page.getByRole('button', { name: 'Draw 40 new experiments' }).click();
        await page.getByRole('slider', { name: 'Observations per experiment' }).fill('5');
        assert.ok(await page.locator('.lesson-lab svg').evaluate(svg => [...svg.querySelectorAll('line')].every(line => +line.getAttribute('x1') >= 0 && +line.getAttribute('x2') <= 550)));
      } else if (slug.startsWith('bayesian')) {
        assert.match(await page.locator('.lesson-results').innerText(), /0\.579/);
        await page.getByRole('slider', { name: 'Successes', exact: true }).fill('0');
        await page.getByRole('slider', { name: 'Failures', exact: true }).fill('0');
        assert.match(await page.locator('.lesson-results').innerText(), /Beta\(2, 2\)/);
        await page.getByLabel('Prior', { exact: true }).selectOption('20');
        assert.match(await page.locator('.lesson-results').innerText(), /Beta\(20, 20\)/);
        await page.getByRole('button', { name: 'Reset example' }).click();
      } else {
        await page.getByRole('slider', { name: 'Bridge weight' }).fill('0');
        assert.match(await page.locator('.lesson-results').innerText(), /Two disconnected/);
        await page.getByRole('slider', { name: 'Bridge weight' }).fill('2');
        await page.getByLabel('Eigenvector', { exact: true }).selectOption('5');
        assert.match(await page.locator('.lesson-lab caption').last().innerText(), /u6/);
      }
      console.log(`${width}px: ${slug}: navigation, controls, solutions and layout checked`);
    }
  }
  assert.deepEqual(errors, [], `Browser errors: ${errors.join('; ')}`);
  await browser.close();
})().catch(error => { console.error(error); process.exit(1); });

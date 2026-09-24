const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('fs');
const path = require('path');
const assert = require('assert/strict');

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1050 } });
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => {
        if (['error', 'warning'].includes(message.type())) errors.push(message.text());
      });
      page.on('requestfailed', request => errors.push(request.url() + ': ' + request.failure().errorText));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/entropy-cross-entropy-kl-divergence');
      const lab = page.locator('[data-entropy-lab="logits"]');
      await lab.waitFor();
      await page.evaluate(() => document.fonts.ready);
      await lab.getByRole('slider', { name: 'Score gap g', exact: true }).fill('1000');
      await lab.getByRole('combobox', { name: 'Observed class', exact: true }).selectOption('2');
      const geometry = await lab.locator('.entropy-pipeline').evaluate(node => {
        const bounds = node.getBoundingClientRect();
        return {
          direction: getComputedStyle(node).flexDirection,
          width: bounds.width,
          scrollWidth: node.scrollWidth,
          arrows: [...node.querySelectorAll('[aria-hidden]')].map(arrow => ({ transform: getComputedStyle(arrow).transform, bounds: arrow.getBoundingClientRect().toJSON() })),
          steps: [...node.querySelectorAll('span:not([aria-hidden])')].map(step => step.getBoundingClientRect().toJSON())
        };
      });
      assert(geometry.scrollWidth <= geometry.width + 1);
      assert.equal(geometry.direction, width <= 620 ? 'column' : 'row');
      if (width <= 620) {
        assert(geometry.steps.every((step, i) => i === 0 || step.y > geometry.steps[i - 1].bottom));
        assert(geometry.arrows.every(arrow => arrow.transform === 'matrix(0, 1, -1, 0, 0, 0)'));
      }
      await lab.scrollIntoViewIfNeeded();
      await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = 'hidden'));
      await lab.screenshot({ path: path.resolve('scratch/entropy-browser/logits-final-' + width + '.png') });
      const classifier = page.locator('.entropy-case-row').first().locator('..');
      assert((await classifier.innerText()).includes('cases 1–8'));
      assert((await classifier.innerText()).includes('same 0–5 nats scale'));
      assert.deepEqual(await classifier.locator('p').evaluateAll(nodes => nodes.filter(node => node.scrollWidth > node.clientWidth + 1).map(node => node.textContent)), []);
      await classifier.screenshot({ path: path.resolve('scratch/entropy-browser/classifier-final-' + width + '.png') });
      await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = ''));
      assert.deepEqual(errors, []);
      results.push({ width, geometry, errors });
      await page.close();
    }
  } finally {
    await browser.close();
  }
  fs.writeFileSync('scratch/entropy-browser/pipeline-polish-results.json', JSON.stringify({ passed: true, timestamp: new Date().toISOString(), results }, null, 2));
  console.log(results.map(result => ({ width: result.width, direction: result.geometry.direction, errors: result.errors })));
})().catch(error => { console.error(error); process.exit(1); });

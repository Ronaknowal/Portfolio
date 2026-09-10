const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const directory = 'scratch/second-order-review/browser';
fs.mkdirSync(directory, { recursive: true });

async function setRange(page, locator, value) {
  const settings = await locator.evaluate(node => ({ minimum: Number(node.min), step: Number(node.step) }));
  await locator.focus();
  await page.keyboard.press('Home');
  for (let index = 0; index < Math.round((value - settings.minimum) / settings.step); index += 1) await page.keyboard.press('ArrowRight');
  assert.ok(Math.abs(Number(await locator.inputValue()) - value) < 1e-10);
}

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const widths = process.env.REVIEW_WIDTHS ? process.env.REVIEW_WIDTHS.split(',').map(Number) : [1440, 390, 320];
  const previous = fs.existsSync(`${directory}/results.json`) ? JSON.parse(fs.readFileSync(`${directory}/results.json`)).results : [];
  const results = previous.filter(result => !widths.includes(result.width));
  try {
    for (const width of widths) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/second-order-methods-l-bfgs-k-fac-shampoo-natural-gradient?module=math-foundations', { waitUntil: 'networkidle' });
      const lesson = page.locator('.second-order-methods-lesson');
      await lesson.waitFor();
      assert.equal(await lesson.locator('.second-order-lab').count(), 6);
      const anchors = [];
      for (const link of await lesson.locator('.lesson-intro nav a').all()) {
        const href = await link.getAttribute('href');
        await link.scrollIntoViewIfNeeded();
        await link.focus();
        await page.keyboard.press('Enter');
        const arrival = await page.evaluate(hash => {
          const node = document.getElementById(hash.slice(1));
          return node ? { text: node.textContent, top: node.getBoundingClientRect().top } : null;
        }, href);
        assert.ok(arrival, `Missing section ${href}`);
        assert.ok(arrival.top >= 40 && arrival.top < 300, JSON.stringify(arrival));
        anchors.push(arrival.text);
        await page.screenshot({ path: `${directory}/reading-${width}-${anchors.length}.png` });
      }
      const geometry = page.getByRole('region', { name: 'Curvature and update geometry', exact: true });
      await geometry.getByRole('button', { name: 'Next update', exact: true }).focus();
      for (let index = 0; index < 12; index += 1) await page.keyboard.press('Enter');
      assert.match(await geometry.innerText(), /GD after 12 updates/);
      await setRange(page, geometry.getByRole('slider', { name: 'Valley rotation in degrees', exact: true }), 30);
      await geometry.getByRole('button', { name: 'Next update', exact: true }).click();
      await geometry.screenshot({ path: `${directory}/geometry-${width}.png` });
      await geometry.getByRole('button', { name: 'Reset geometry', exact: true }).click();
      const safeguard = page.getByRole('region', { name: 'Newton direction and step acceptance', exact: true });
      await safeguard.getByRole('button', { name: 'Try a near-flat model', exact: true }).click();
      assert.match(await safeguard.innerText(), /0\.125/);
      await safeguard.screenshot({ path: `${directory}/safeguard-${width}.png` });
      await safeguard.getByRole('button', { name: 'Inspect an indefinite model', exact: true }).click();
      assert.match(await safeguard.innerText(), /indefinite/);
      await safeguard.getByRole('button', { name: 'Reset safeguard', exact: true }).click();
      const history = page.getByRole('region', { name: 'L-BFGS history and two-loop recursion', exact: true });
      for (let memory = 0; memory <= 3; memory += 1) {
        await setRange(page, history.getByRole('slider', { name: 'Retained history budget', exact: true }), memory);
        const next = history.getByRole('button', { name: 'Next transformation', exact: true });
        let transformations = 0;
        while (await next.isEnabled()) {
          await next.focus();
          await page.keyboard.press('Enter');
          transformations += 1;
          assert.ok(transformations < 10);
        }
        assert.equal(transformations, 2 * memory + 1);
      }
      await history.getByRole('checkbox', { name: 'Reverse the last gradient change' }).check();
      assert.match(await history.innerText(), /Rejected: nonpositive curvature/);
      await history.screenshot({ path: `${directory}/history-${width}.png` });
      await history.getByRole('button', { name: 'Reset history', exact: true }).click();
      const natural = page.getByRole('region', { name: 'Natural gradient probability geometry', exact: true });
      await setRange(page, natural.getByRole('slider', { name: 'Natural step fraction', exact: true }), 1);
      assert.match(await natural.innerText(), /Direct probability update: p=0\.8/);
      await natural.screenshot({ path: `${directory}/natural-${width}.png` });
      await setRange(page, natural.getByRole('slider', { name: 'Starting success probability', exact: true }), 0.05);
      assert.match(await natural.innerText(), /Logit update mapped back: p=1 − 2\.639e-6/);
      await natural.screenshot({ path: `${directory}/natural-boundary-${width}.png` });
      await natural.getByRole('button', { name: 'Reset probability geometry', exact: true }).click();
      const fisher = page.getByRole('region', { name: 'K-FAC expectation factorization', exact: true });
      await fisher.getByRole('combobox').selectOption('difference');
      await fisher.screenshot({ path: `${directory}/fisher-${width}.png` });
      await setRange(page, fisher.getByRole('slider', { name: 'Layer strength', exact: true }), 0);
      const errorText = await fisher.locator('dl > div').first().innerText();
      assert.match(errorText, /error\s+0$/);
      await fisher.getByRole('button', { name: 'Reset Fisher factors', exact: true }).click();
      const shampoo = page.getByRole('region', { name: 'Shampoo matrix accumulation and inverse roots', exact: true });
      await shampoo.getByRole('button', { name: 'Accumulate next gradient', exact: true }).click();
      await shampoo.getByRole('button', { name: 'Accumulate next gradient', exact: true }).click();
      assert.ok(await shampoo.getByRole('button', { name: 'Accumulate next gradient', exact: true }).isDisabled());
      await setRange(page, shampoo.getByRole('slider', { name: 'Row rotation in degrees', exact: true }), 45);
      await shampoo.screenshot({ path: `${directory}/shampoo-${width}.png` });
      await shampoo.getByRole('button', { name: 'Reset Shampoo', exact: true }).click();
      for (let index = 0; index < 2; index += 1) await lesson.locator('.second-order-inline').nth(index).screenshot({ path: `${directory}/inline-${index + 1}-${width}.png` });
      const summaries = lesson.locator('.lesson-check summary');
      for (const summary of await summaries.all()) {
        await summary.scrollIntoViewIfNeeded();
        await summary.focus();
        await page.keyboard.press('Enter');
        assert.ok(await summary.evaluate(node => node.parentElement.open));
      }
      assert.equal(await lesson.locator('.python-example').count(), 9);
      const { secondOrderMethodsExamples } = await import('../src/learn/data/second-order-methods-examples.js');
      const blocks = await lesson.locator('pre').allTextContents();
      fs.writeFileSync(`${directory}/code-inspection.json`, JSON.stringify({ first: blocks[0], expected: secondOrderMethodsExamples.quadraticSolve.code, sampleCount: blocks.length }, null, 2));
      for (const example of Object.values(secondOrderMethodsExamples)) {
        const renderedExample = lesson.locator('.python-example').filter({ has: page.getByRole('heading', { name: example.title, exact: true }) });
        const text = await renderedExample.innerText();
        assert.ok(text.includes(example.code.trim()), `Missing code: ${example.title}`);
        assert.ok(text.includes(example.expected.trim()), `Missing output: ${example.title}`);
      }
      const dimensions = await page.evaluate(() => ({ width: innerWidth, scroll: document.documentElement.scrollWidth }));
      assert.ok(dimensions.scroll <= dimensions.width + 1, JSON.stringify(dimensions));
      const mathOverflow = await lesson.locator('.katex-display').evaluateAll(nodes => nodes.map(node => ({ text: node.textContent.slice(0, 100), width: node.clientWidth, scroll: node.scrollWidth })).filter(value => value.scroll > value.width + 1));
      const svgOverflow = await lesson.locator('svg').evaluateAll(nodes => nodes.flatMap(svg => [...svg.querySelectorAll('text')].map(node => {
        const bounds = node.getBBox();
        return { text: node.textContent, x: bounds.x, right: bounds.x + bounds.width, limit: svg.viewBox.baseVal.width };
      }).filter(value => value.x < -1 || value.right > value.limit + 1)));
      await lesson.locator('.lesson-sources').screenshot({ path: `${directory}/sources-${width}.png` });
      results.push({ checkedAt: new Date().toISOString(), width, anchors, codeExamples: Object.keys(secondOrderMethodsExamples).length, mathOverflow, svgOverflow, dimensions, errors });
      fs.writeFileSync(`${directory}/results.json`, JSON.stringify({ checkedAt: new Date().toISOString(), results }, null, 2));
      assert.deepEqual(errors, []);
      assert.deepEqual(mathOverflow, []);
      assert.deepEqual(svgOverflow, []);
      await page.close();
    }
    console.log(JSON.stringify(results, null, 2));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });

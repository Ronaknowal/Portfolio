const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const directory = 'scratch/differential-geometry-independent-review';
fs.mkdirSync(directory, { recursive: true });

async function setSlider(locator, value) {
  await locator.evaluate((node, next) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(node, String(next));
    node.dispatchEvent(new Event('input', { bubbles: true }));
    node.dispatchEvent(new Event('change', { bubbles: true }));
  }, value);
}

async function readout(region, label) {
  return region.locator('.dg-readouts > div').filter({ has: region.page().getByText(label, { exact: true }) }).locator('dd').innerText();
}

async function capture(page, target, name, width) {
  const style = await page.addStyleTag({ content: '.learn-nav { visibility:hidden !important; }' });
  await target.screenshot({ path: `${directory}/${name}-${width}.png` });
  await style.evaluate(node => node.remove());
}

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 } });
      await page.routeWebSocket('**', socket => socket.close());
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => {
        if (message.type() === 'error' && !message.text().startsWith('[vite]')) errors.push(message.text());
      });
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/differential-geometry-riemannian-manifolds?module=math-foundations', { waitUntil: 'domcontentloaded', timeout: 60000 });
      const lesson = page.locator('.differential-geometry-lesson');
      await lesson.waitFor();
      await page.evaluate(() => document.fonts.ready);
      const atlas = page.getByRole('region', { name: 'Circle atlas investigation', exact: true });
      await atlas.getByRole('button', { name: 'Inspect α seam', exact: true }).focus();
      await page.keyboard.press('Enter');
      assert.equal(await atlas.getByText('Not in this chart’s domain.', { exact: true }).count(), 1);
      assert.equal(await atlas.getByLabel('Circle position', { exact: true }).inputValue(), '180');
      await atlas.getByLabel('Circle position', { exact: true }).focus();
      await page.keyboard.press('ArrowRight');
      assert.equal(await atlas.getByLabel('Circle position', { exact: true }).inputValue(), '181');
      assert.equal(await atlas.getByText('Not in this chart’s domain.', { exact: true }).count(), 0);

      const metric = page.getByRole('region', { name: 'Metric and differential investigation', exact: true });
      await setSlider(metric.getByLabel('Coordinate shear', { exact: true }), -.4);
      await setSlider(metric.getByLabel('Physical y-motion cost', { exact: true }), 2);
      assert.equal(await readout(metric, 'Gradient in x,y'), '(2, -0.25)');
      assert.equal(await readout(metric, 'Gradient in u,v'), '(1.9, -0.25)');
      await capture(page, metric, 'metric-changed', width);

      const transport = page.getByRole('region', { name: 'Parallel transport investigation', exact: true });
      await setSlider(transport.getByLabel('Longitude wedge', { exact: true }), 55);
      await setSlider(transport.getByLabel('Initial tangent angle', { exact: true }), 30);
      await transport.getByRole('button', { name: 'Route: N → A → B → N', exact: true }).focus();
      await page.keyboard.press('Enter');
      assert.equal(await readout(transport, 'Final oriented turn'), '-55°');
      assert.equal(await readout(transport, 'Current ambient arrow'), '(0.9063, -0.4226, 0)');
      await capture(page, transport, 'transport-reversed', width);
      await setSlider(transport.getByLabel('Transport progress', { exact: true }), 1.5);
      assert.equal(await readout(transport, 'Final oriented turn'), '-55°');
      assert.ok((await transport.innerText()).includes('complete loop’s result even while the left progress marker is mid-route'));

      const status = await page.evaluate(() => ({
        actualFont: [...document.fonts].some(font => font.family === 'Space Grotesk' && font.status === 'loaded'),
        documentWidth: document.documentElement.scrollWidth,
      }));
      assert.equal(status.actualFont, true);
      assert.equal(status.documentWidth, width);
      assert.equal(await lesson.locator('.katex-error').count(), 0);
      assert.deepEqual(errors, []);
      results.push({ width, status, keyboardStates: 3, changedMetric: true, reversedLoop: true, partialProgressReadoutContract: true, errors });
      await page.close();
    }
  } finally {
    await browser.close();
  }
  fs.writeFileSync(`${directory}/browser-results.json`, JSON.stringify({ checkedAt: new Date().toISOString(), results, captureNote: 'Only the fixed navigation header is hidden while capturing individual investigations; behavior and lesson source remain unchanged.' }, null, 2));
})().catch(error => { console.error(error); process.exitCode = 1; });

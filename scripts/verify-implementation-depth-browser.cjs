const fs = require('fs');
const assert = require('node:assert/strict');
const crypto = require('crypto');
const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const fonts = JSON.parse(fs.readFileSync('scratch/kmeans-revision-review/fonts/manifest.json', 'utf8').replace(/^\uFEFF/, ''));
const report = { status: 'running', checkedAt: new Date().toISOString(), checks: [], captures: [], errors: [], sourceHashes: {} };
const cases = [
  { id: 'loss-functions-ce-mse-focal-contrastive-triplet', module: 'deep-learning-fundamentals', heading: 'Build the objectives, then control the library', summary: 'Read the scratch objectives and matched library checks', asset: 'loss-mechanisms.py', chunk: 'loss-mechanisms-program', code: 'def cross_entropy' },
  { id: 'batch-layer-group-rms-normalization', module: 'deep-learning-fundamentals', heading: 'Implement the derivative and connect it to the module', summary: 'Read the manual normalization derivatives and library checks', asset: 'normalization-backward.py', chunk: 'normalization-backward-program', code: 'def backward' },
  { id: 'numpy-arrays-broadcasting-vectorization', module: 'programming-scientific-computing', code: 'def calibrate_sensor_means' },
  { id: 'feature-scaling-encoding-imputation', module: 'classical-ml', heading: 'Build the saved state, then recognize it in the library', code: 'def fit_preparation' },
  { id: 'perceptrons-neurons-activation-functions', module: 'deep-learning-fundamentals', heading: 'Implement the neuron, then match the library', summary: 'Read and run the NumPy / library neuron comparison', assetPath: 'perceptrons/activation-mechanisms.py', chunk: 'activation-mechanisms.py', code: 'def activation' },
  { id: 'backpropagation-automatic-differentiation', file: 'backprop', module: 'deep-learning-fundamentals', heading: 'Move one complete training step between implementations', summary: 'Read and run the teaching-engine / PyTorch bridge', assetPath: 'backpropagation/engine-library-bridge.py', chunk: 'engine-library-bridge.py', code: 'def run' },
  { id: 'transfer-learning-fine-tuning-strategies', module: 'deep-learning-fundamentals', summary: 'Read and run the manual / autograd LoRA comparison', assetPath: 'transfer-learning/lora-mechanism-bridge.py', chunk: 'lora-mechanism-bridge.py', code: 'def forward_and_gradients' },
];
const base = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4194';
const receipt = 'docs/teaching/evidence/implementation-depth-browser.json';
fs.mkdirSync('docs/teaching/evidence/screenshots/implementation-depth', { recursive: true });
const save = () => fs.writeFileSync(receipt, JSON.stringify(report, null, 2) + '\n');
save();
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    report.manifestHash = hash('dist/.vite/manifest.json');
    for (const item of cases) {
      const context = await browser.newContext({ viewport: { width: 1366, height: 950 }, reducedMotion: 'reduce' });
      await context.route('https://fonts.googleapis.com/**', route => route.fulfill({ path: fonts.stylesheet, contentType: 'text/css', headers: { 'access-control-allow-origin': '*' } }));
      await context.route('https://fonts.gstatic.com/**', route => fonts.files[route.request().url()] ? route.fulfill({ path: fonts.files[route.request().url()], contentType: 'font/ttf', headers: { 'access-control-allow-origin': '*' } }) : route.abort());
      const page = await context.newPage();
      const requests = [];
      page.on('request', request => requests.push(request.url()));
      page.on('pageerror', error => report.errors.push({ id: item.id, error: error.message }));
      await page.goto(`${base}/learn/path/full-curriculum/${item.id}?module=${item.module}`, { timeout: 120000 });
      await page.locator('main.reader-content').waitFor();
      await page.evaluate(() => document.fonts.ready);
      if (item.summary) {
        if (item.heading) await page.getByRole('heading', { name: item.heading, exact: true }).waitFor();
        assert.equal(requests.some(url => url.includes(item.chunk)), false, 'code loaded before disclosure');
        const summary = page.locator('summary').filter({ hasText: item.summary });
        await summary.click();
        const details = summary.locator('..');
        await details.getByText(item.code, { exact: false }).first().waitFor();
        assert.ok((await details.innerText()).includes(item.code));
        assert.ok(requests.some(url => url.includes(item.chunk)), 'no dynamic program request');
        const assetPath = item.assetPath || `${item.id}/${item.asset}`;
        const response = await context.request.get(`${base}/learn-assets/${assetPath}`);
        assert.equal(response.status(), 200);
        const source = fs.readFileSync(`public/learn-assets/${assetPath}`, 'utf8').replace(/\r\n/g, '\n');
        assert.equal((await response.text()).replace(/\r\n/g, '\n'), source);
        assert.ok((await details.innerText()).replace(/\r\n/g, '\n').includes(source.trim()), 'displayed/download code mismatch');
        await summary.click();
        await page.waitForFunction(text => ![...document.querySelectorAll('details')].some(node => !node.open && node.textContent.includes(text)), item.code);
        report.checks.push(`${item.id}: lazy code, exact download/display, close unmount`);
      } else {
        await page.getByText(item.code, { exact: false }).first().waitFor();
        report.checks.push(`${item.id}: loop and vectorized code visible`);
      }
      for (const width of [1366, 390, 320]) {
        await page.setViewportSize({ width, height: 950 });
        const target = item.heading ? page.getByRole('heading', { name: item.heading, exact: true }) : item.summary ? page.locator('summary').filter({ hasText: item.summary }) : page.getByText(item.code, { exact: false }).first();
        await target.scrollIntoViewIfNeeded();
        const state = await page.evaluate(() => ({ width: innerWidth, scroll: document.documentElement.scrollWidth, errors: document.querySelectorAll('.katex-error,.lesson-load-error').length }));
        assert.ok(state.scroll <= width + 1, JSON.stringify(state));
        assert.equal(state.errors, 0);
        const file = `docs/teaching/evidence/screenshots/implementation-depth/${item.id}-${width}.png`;
        await page.screenshot({ path: file });
        report.captures.push({ path: file, sha256: hash(file) });
        report.checks.push(`${item.id}: ${width}px contained and rendered`);
        if (width === 320 && item.summary) {
          const summary = page.locator('summary').filter({ hasText: item.summary });
          await summary.click();
          const code = summary.locator('..').locator('pre').first();
          await code.waitFor();
          await code.focus();
          assert.equal(await code.getAttribute('tabindex'), '0');
          await code.press('ArrowRight');
          assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'open code overflow');
          report.checks.push(`${item.id}: open phone code is keyboard-focusable and locally contained`);
          await summary.click();
        }
      }
      const topicFile = `src/learn/data/topics/${item.file || item.id}.jsx`;
      report.sourceHashes[topicFile] = hash(topicFile);
      await context.close();
    }
    assert.deepEqual(report.errors, []);
    report.status = 'passed';
  } catch (error) { report.status = 'failed'; report.failure = error.stack; process.exitCode = 1; }
  finally { await browser.close(); save(); }
  console.log(JSON.stringify({ status: report.status, checks: report.checks.length, captures: report.captures.length, failure: report.failure }));
})();

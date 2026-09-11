const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const directory = 'scratch/pde-wave-boundary';
fs.mkdirSync(directory, { recursive: true });
const sourcePaths = JSON.parse(fs.readFileSync('docs/teaching/evidence/pde-author-review-before-independent-amendment.json')).sourceHashes.map(source => source.path);
const hashes = () => sourcePaths.map(path => ({ path, sha256: crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex') }));
const format = value => Math.abs(value) > 1e5 || (value !== 0 && Math.abs(value) < 1e-5) ? value.toExponential(3) : Number(value.toFixed(4)).toString();
async function slider(region, label, value) {
  await region.getByLabel(label, { exact: true }).evaluate((node, value) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(node, String(value));
    node.dispatchEvent(new Event('input', { bubbles: true }));
    node.dispatchEvent(new Event('change', { bubbles: true }));
  }, value);
}
async function capture(page, target, name, width) {
  await target.evaluate(node => scrollTo({ top: scrollY + node.getBoundingClientRect().top - 115, behavior: 'instant' }));
  await page.screenshot({ path: `${directory}/${name}-${width}.png` });
}
(async () => {
  const { waveState } = await import('../src/learn/data/pde-models.js');
  const { pdeExamples } = await import('../src/learn/data/pde-examples.js');
  const sourceHashes = hashes();
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1100 } });
      await page.routeWebSocket('**', socket => socket.close());
      const errors = [], failedRequests = [], diagnostics = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('requestfailed', request => failedRequests.push(request.url()));
      page.on('console', message => {
        if (!['warning', 'error'].includes(message.type())) return;
        if (message.text().startsWith('[vite]') && /websocket|connection|server|reconnect/i.test(message.text())) diagnostics.push(message.text());
        else errors.push(message.text());
      });
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/partial-differential-equations-conservation-boundary-conditions?module=math-foundations', { waitUntil: 'networkidle' });
      const lesson = page.locator('.pde-lesson');
      await lesson.waitFor();
      await page.evaluate(() => document.fonts.ready);
      assert(await page.evaluate(() => document.fonts.check('16px "Space Grotesk"')));
      assert.equal(await page.locator('vite-error-overlay').count(), 0);
      const programs = await lesson.locator('.python-example').evaluateAll(nodes => nodes.map(node => ({ title: node.querySelector('h3').textContent, blocks: [...node.children].filter(child => getComputedStyle(child).whiteSpace === 'pre').map(child => [...child.childNodes].filter(node => node.nodeType === Node.TEXT_NODE).map(node => node.textContent).join('')) })));
      assert.equal(programs.length, 15);
      for (const program of programs) {
        const expected = Object.values(pdeExamples).find(example => example.title === program.title);
        assert.equal(program.blocks[0].trim(), expected.code.trim());
        assert.equal(program.blocks[1].trim(), expected.expected.trim());
      }
      const wave = lesson.locator('[data-pde-lab="wave"]');
      let states = 0;
      for (const velocity of [0, .5]) for (const time of [0, .02, .4, 1]) for (const x of [-1, 0, 1]) {
        await wave.getByLabel('Initial velocity', { exact: true }).selectOption(String(velocity));
        await slider(wave, 'Wave time', time);
        await slider(wave, 'Wave observation x', x);
        const actual = await wave.locator('.pde-values>div').filter({ has: page.getByText('Total at observation', { exact: true }) }).locator('dd').innerText();
        assert.equal(actual, format(waveState(time, x, velocity).total));
        states += 1;
      }
      const imported = await page.evaluate(async () => (await import('/src/learn/data/pde-models.js')).waveValue(1, 1e-6, .5));
      assert(imported.total > 0 && imported.velocityContribution > 0);
      await wave.getByRole('button', { name: 'Reset wave', exact: true }).focus();
      await page.keyboard.press('Enter');
      assert.equal(await wave.getByLabel('Wave time', { exact: true }).inputValue(), '0.6');
      await capture(page, wave, 'wave-final', width);
      const maximum = lesson.locator('p').filter({ hasText: 'First work up to any time T′' });
      assert.equal(await maximum.count(), 1);
      await capture(page, maximum, 'maximum-final', width);
      const units = lesson.locator('p').filter({ hasText: 'suppressed coefficient of x(L−x)' });
      assert.equal(await units.count(), 1);
      await capture(page, units, 'units-final', width);
      const arithmetic = lesson.locator('p').filter({ hasText: 'sixteen nonnegative polynomial terms' });
      assert.equal(await arithmetic.count(), 1);
      await capture(page, arithmetic, 'arithmetic-final', width);
      const equations = await lesson.locator('.katex-display').evaluateAll(nodes => nodes.map(node => ({ client: node.clientWidth, scroll: node.scrollWidth })));
      assert(equations.every(row => row.scroll <= row.client + 2));
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth - innerWidth);
      assert(overflow <= 2, `Document overflow ${overflow}`);
      assert.deepEqual(errors, []);
      assert.deepEqual(failedRequests, []);
      results.push({ width, states, programs: programs.length, equations: equations.length, errors, failedRequests, intentionalHmrDiagnostics: diagnostics, overflow, loadedModuleBoundary: imported });
      await page.close();
    }
    assert.deepEqual(hashes(), sourceHashes);
    fs.writeFileSync('docs/teaching/evidence/pde-independent-amendment-browser.json', JSON.stringify({ checkedAt: new Date().toISOString(), passed: true, sourceHashes, results }, null, 2) + '\n');
    console.log(JSON.stringify(results.map(({ width, states, programs }) => ({ width, states, programs }))));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

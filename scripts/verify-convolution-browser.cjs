const fs = require('node:fs');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const base = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4194';
const directory = 'docs/teaching/evidence/convolution-browser';
fs.mkdirSync(directory, { recursive: true });
const report = { passed: false, checkedAt: new Date().toISOString(), base, groups: [], screenshots: [], errors: [] };
const save = () => fs.writeFileSync(`${directory}/report.json`, JSON.stringify(report, null, 2) + '\n');
save();
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1366, height: 1000 }, reducedMotion: 'reduce' });
    page.on('pageerror', error => report.errors.push(error.message));
    const requests = [];
    page.on('request', request => requests.push(request.url()));
    await require('./lib/lesson-browser-fonts.cjs')(page);
    await page.goto(`${base}/learn/path/full-curriculum/convolution-pooling-receptive-fields?module=deep-learning-fundamentals`);
    await page.locator('[data-lab="convolution-patch"]').waitFor();
    await page.evaluate(() => document.fonts.ready);
    const lab = id => page.locator(`[data-lab="convolution-${id}"]`);
    const expectText = async (locator, pattern) => assert.match(await locator.innerText(), pattern);
    const patch = lab('patch');
    await expectText(patch, /Output \(0, 1\).*5/);
    const center = patch.getByRole('spinbutton', { name: 'Image, row 2, column 2', exact: true });
    await center.fill('2');
    await patch.getByRole('button', { name: 'Output, row 2, column 2: -1', exact: true }).click();
    await expectText(patch, /Output \(1, 1\).*−?\(?-1/);
    await center.fill('100');
    assert.equal(await center.getAttribute('aria-invalid'), 'true');
    await expectText(patch, /current result retains 2/);
    await patch.getByRole('button', { name: 'Reset patch', exact: true }).click();
    await patch.getByRole('button', { name: 'Zero filter', exact: true }).click();
    assert.equal(await patch.getByRole('button', { name: /^Output, row .*: 0$/ }).count(), 4);
    await patch.getByRole('button', { name: 'Reset patch', exact: true }).click();
    report.groups.push('Editable patch/product/output correspondence, invalid input, zero null and reset');
    const update = lab('update');
    await update.getByRole('spinbutton', { name: 'Target at output 1', exact: true }).fill('2');
    await expectText(update, /2\.61/);
    await update.getByRole('spinbutton', { name: 'Step size', exact: true }).fill('0.01');
    await expectText(update, /1\.7001/);
    await update.getByRole('slider', { name: 'Step size slider', exact: true }).press('Home');
    await expectText(update, /Zero rate preserves/);
    const slider = update.getByRole('slider', { name: 'Step size slider', exact: true });
    await slider.scrollIntoViewIfNeeded();
    const bounds = await slider.boundingBox();
    await page.mouse.move(bounds.x + 4, bounds.y + bounds.height / 2);
    await page.mouse.down();
    await page.mouse.move(bounds.x + bounds.width * .4, bounds.y + bounds.height / 2, { steps: 10 });
    await page.mouse.up();
    assert.ok(Number(await slider.inputValue()) > .02);
    await update.getByRole('button', { name: 'Reset update', exact: true }).click();
    report.groups.push('Target-dependent derivatives, overshoot/descent, real keyboard and pointer slider gestures');
    const geometry = lab('geometry');
    await geometry.getByRole('spinbutton', { name: 'Right padding', exact: true }).fill('2');
    await expectText(geometry, /8 outputs/);
    await expectText(geometry, /first center 1/);
    await geometry.getByRole('spinbutton', { name: 'Left padding', exact: true }).fill('2');
    await geometry.getByRole('spinbutton', { name: 'Right padding', exact: true }).fill('1');
    await expectText(geometry, /first center 0/);
    await geometry.getByRole('spinbutton', { name: 'Input width', exact: true }).fill('3');
    await geometry.getByRole('spinbutton', { name: 'Dilation', exact: true }).fill('3');
    await expectText(geometry, /does not fit/);
    await geometry.getByRole('button', { name: 'Reset geometry', exact: true }).click();
    report.groups.push('Geometry shape/alignment distinction and impossible-operation feedback');
    const receptive = lab('receptive');
    await receptive.getByRole('spinbutton', { name: 'First dilation', exact: true }).fill('1');
    await expectText(receptive, /7 connected offsets in a bound of width 7/);
    await receptive.getByRole('button', { name: 'Reset receptive fields', exact: true }).click();
    await expectText(receptive, /5 connected offsets in a bound of width 9/);
    await lab('shift').getByRole('combobox', { name: 'Boundary rule', exact: true }).selectOption('zero');
    await expectText(lab('shift'), /Maximum difference: 1/);
    await lab('shift').getByRole('button', { name: 'Reset shifts', exact: true }).click();
    await lab('transpose').getByRole('spinbutton', { name: 'Output-side value 1', exact: true }).fill('0');
    await expectText(lab('transpose'), /〈Cx, g〉 = -4/);
    await lab('transpose').getByRole('button', { name: 'Reset transpose', exact: true }).click();
    report.groups.push('Sparse ancestry, boundary contrast and updated transpose identity');
    assert.ok(!requests.some(url => url.endsWith('calculated-inputs.json')), 'Recorded model data must remain deferred');
    await lab('measured').getByRole('button', { name: 'Open recorded viewer', exact: true }).click();
    await page.getByRole('combobox', { name: 'Measured model', exact: true }).waitFor();
    await expectText(lab('measured'), /0\.08332/);
    await page.getByRole('combobox', { name: 'Recorded seed', exact: true }).selectOption('3');
    await expectText(lab('measured'), /0\.05194/);
    await expectText(lab('measured'), /cnn_max, seed 1, update 400/);
    await page.getByRole('combobox', { name: 'Recorded update', exact: true }).selectOption('0');
    await expectText(lab('measured'), /Recorded update 0, seed 3/);
    await lab('measured').getByRole('button', { name: 'Reset recorded view', exact: true }).click();
    assert.ok(!(await lab('measured').innerText()).includes('undefined'));
    report.groups.push('Deferred real measurements, exact metric rendering and separate saved-map provenance');
    for (const file of ['convolution-experiments.py', 'convolution_pullbacks.py']) {
      assert.ok(!requests.some(url => url.endsWith(file)));
      const disclosure = page.locator('.convolution-program').filter({ has: page.locator(`a[href$="${file}"]`) });
      await disclosure.locator('summary').click();
      await disclosure.locator('pre').waitFor();
      assert.equal((await disclosure.locator('pre').innerText()).replace(/\r\n/g, '\n'), fs.readFileSync(`public/learn-code/convolution-pooling-receptive-fields/${file}`, 'utf8').replace(/\r\n/g, '\n'));
      await disclosure.locator('summary').click();
      await disclosure.locator('pre').waitFor({ state: 'detached' });
    }
    report.groups.push('Both canonical source programs fetch only on demand, show exact bytes and unmount');
    for (const width of [1366, 390, 320]) {
      await page.setViewportSize({ width, height: 1000 });
      for (const name of ['patch', 'channels', 'update', 'geometry', 'pooling', 'receptive', 'measured', 'influence', 'shift', 'transpose']) {
        const current = lab(name);
        await current.scrollIntoViewIfNeeded();
        const size = await page.evaluate(() => ({ width: innerWidth, scroll: document.documentElement.scrollWidth }));
        assert.ok(size.scroll <= size.width + 1, `${name}: overflow at ${width}`);
        const controls = await current.locator('input,select,button').evaluateAll(nodes => nodes.filter(node => node.getBoundingClientRect().width > 0).map(node => { const b = node.getBoundingClientRect(); return { left: b.left, right: b.right, width: b.width }; }));
        assert.ok(controls.every(box => box.left >= -1 && box.right <= width + 1 && box.width >= 25), `${name}: clipped controls at ${width}`);
      }
      for (const name of width === 1366 ? ['patch', 'measured'] : ['patch', 'receptive']) {
        await lab(name).scrollIntoViewIfNeeded();
        const file = `${name}-${width}.png`;
        await page.screenshot({ path: `${directory}/${file}` });
        report.screenshots.push(`${directory}/${file}`);
      }
      report.groups.push(`All ten labs readable without page overflow or clipped controls at ${width}px`);
    }
    assert.equal(await page.locator('.katex-error').count(), 0);
    assert.deepEqual(report.errors, []);
    report.passed = true;
    report.verifierHash = createHash('sha256').update(fs.readFileSync(__filename)).digest('hex');
    console.log(`PASS: ${report.groups.length} convolution browser groups.`);
  } catch (error) { report.failure = error.stack; process.exitCode = 1; console.error(error.stack); }
  finally { await browser.close(); save(); }
})();

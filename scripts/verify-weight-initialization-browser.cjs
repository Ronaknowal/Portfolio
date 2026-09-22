// Scoped production integration. Parent runs after the single shared build.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const output = 'docs/teaching/evidence/weight-initialization-browser';
fs.mkdirSync(output, { recursive: true });
const reportPath = path.join(output, 'report.json');
fs.writeFileSync(reportPath, JSON.stringify({ passed: false, status: 'running' }));
const base = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4194';
const groups = [], captures = [];
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1366, height: 1000 }, reducedMotion: 'reduce' });
    const errors = []; page.on('pageerror', error => errors.push(error.message));
    await require('./lib/lesson-browser-fonts.cjs')(page);
    await page.goto(`${base}/learn/path/full-curriculum/weight-initialization-xavier-kaiming-p?module=deep-learning-fundamentals`);
    const root = page.locator('.initialization-lesson'); await root.waitFor();
    assert.equal(await root.locator('[data-lab]').count(), 6);
    const result = name => root.locator(`[data-result="${name}"]`);
    const number = name => root.getByRole('spinbutton', { name, exact: true });
    const click = name => root.getByRole('button', { name, exact: true }).click();
    const set = async (name, value) => { await number(name).fill(String(value)); };
    assert.match(await result('moments').innerText(), /variance = q − mean² = 0.6875/);
    await click('Outer pair ±3'); assert.match(await result('moments').innerText(), /variance = q − mean² = 1.5/);
    await root.getByRole('combobox', { name: 'Activation', exact: true }).selectOption('identity'); assert.match(await result('moments').innerText(), /unchanged/);
    await set('Value A', -4); await set('Value A', -3); await click('Reset moment investigation');
    await set('Value A', 6); assert.equal(await number('Value A').getAttribute('aria-invalid'), 'true'); assert.match(await result('moments').innerText(), /0.6875/); await number('Value A').blur();
    groups.push('Moments default, actual entity changes, rapid updates, identity, invalid buffer and reset');
    const gain = root.getByRole('slider', { name: 'Smaller one-layer gain squared slider', exact: true });
    assert.equal(await gain.inputValue(), '0.1');
    assert.equal(await number('Smaller one-layer gain squared').inputValue(), '0.1');
    await gain.press('ArrowRight');
    assert.equal(await gain.inputValue(), '0.1025');
    assert.equal(await number('Smaller one-layer gain squared').inputValue(), '0.1025');
    await click('Reset geometry investigation'); assert.equal(await gain.inputValue(), '0.1');
    await click('Five layers · smaller gain 0.2'); assert.match(await result('geometry').innerText(), /0.00032/);
    await click('Identity comparison'); assert.match(await result('geometry').innerText(), /Norm gain: 1/);
    await set('Perturbation vertical', 0); assert.match(await result('geometry').innerText(), /undefined: the zero input/);
    await click('Reset geometry investigation'); await root.getByRole('checkbox', { name: /local Jacobian/ }).check();
    assert.match(await result('geometry').innerText(), /Displayed map gains: 0, 1.41421/);
    await set('Base point horizontal', 0); assert.match(await root.locator('[data-lab="initialization-geometry"]').innerText(), /derivative is not unique/);
    await click('Reset geometry investigation'); groups.push('Geometry contrast, identity, zero vector, gated Jacobian and kink convention');
    await root.getByRole('combobox', { name: 'Recorded initialization', exact: true }).selectOption('zero'); assert.match(await result('signal').innerText(), /Layer 20 \/ input mean square: 0/);
    await set('Weight variance', .02); assert.match(await result('recurrence').innerText(), /q \/ q₀ = 1\./);
    await click('Reset signal investigation'); groups.push('Actual zero-rail fixture and exact recurrence identity');
    await click('Distinct features, zero head'); assert.match(await result('symmetry').innerText(), /output 0, loss 0.5/);
    await click('Apply one gradient step'); assert.doesNotMatch(await result('symmetry').innerText(), /output 0, loss 0.5/);
    await set('Symmetry step size', 0); const before = (await result('symmetry').innerText()).split(': ')[1]; await click('Apply one gradient step'); assert.equal((await result('symmetry').innerText()).split(': ')[1], before);
    await click('Reset symmetry investigation'); groups.push('Zero head opens later gradient route; zero step is a real null');
    await set('Target hidden width', 256); await set('Base Adam learning rate', .004); assert.match(await result('width').innerText(), /Hidden rate = 0.0005; readout input multiplier = 0.125/);
    await set('Target hidden width', 32); assert.match(await result('width').innerText(), /both full procedures coincide/); await click('Reset width investigation');
    await root.getByRole('combobox', { name: 'Training initialization', exact: true }).selectOption('zero'); assert.match(await result('training').innerText(), /12\/120/); await click('Reset training comparison'); groups.push('Width arithmetic and measured-training selection preserve distinct evidence');
    const slider = root.getByRole('slider', { name: 'Value A slider', exact: true }); await slider.scrollIntoViewIfNeeded(); await slider.focus(); await slider.press('Home'); assert.equal(await number('Value A').inputValue(), '-5'); await slider.press('End'); assert.equal(await number('Value A').inputValue(), '5');
    await slider.scrollIntoViewIfNeeded(); const box = await slider.boundingBox(); await page.mouse.move(box.x + box.width * .35, box.y + box.height / 2); await page.mouse.down(); await page.mouse.move(box.x + box.width * .7, box.y + box.height / 2, { steps: 6 }); await page.mouse.up(); assert.ok(Number(await number('Value A').inputValue()) > 0 && Number(await number('Value A').inputValue()) < 5); await click('Reset moment investigation'); groups.push('Real pointer gesture and keyboard endpoints change a meaningful value');
    for (const title of ['Read the complete CPU experiment', 'Read the complete orthogonal and μP library bridge']) { await root.locator('summary').filter({ hasText: title }).click(); }
    await root.locator('.neural-program-source').first().waitFor(); assert.equal(await root.locator('.neural-program-source').count(), 2); groups.push('Both canonical long programs load on explicit disclosure');
    for (const width of [1366, 390, 320]) {
      await page.setViewportSize({ width, height: 1000 });
      await page.evaluate(() => document.fonts.ready);
      for (const seed of ['1', '2', '3']) {
        await root.getByRole('combobox', { name: 'Propagation seed', exact: true }).selectOption(seed);
        for (const scheme of ['small', 'xavier', 'kaiming', 'large', 'zero', 'orthogonal']) {
          await root.getByRole('combobox', { name: 'Recorded initialization', exact: true }).selectOption(scheme);
          const escapedLabels = await root.locator('[data-lab="initialization-signal"] .init-chart').evaluateAll(charts => charts.flatMap(chart => {
            const frame = chart.getBoundingClientRect();
            return [...chart.querySelectorAll('text')].filter(label => { const box = label.getBoundingClientRect(); return box.left < frame.left - .5 || box.right > frame.right + .5 || box.top < frame.top - .5 || box.bottom > frame.bottom + .5; }).map(label => label.textContent);
          }));
          assert.deepEqual(escapedLabels, [], `${width}/${seed}/${scheme}: plot labels must fit the SVG with loaded site fonts`);
        }
      }
      await click('Reset signal investigation');
      for (const lab of ['initialization-moments', 'initialization-geometry', 'initialization-width']) {
        const item = root.locator(`[data-lab="${lab}"]`); await item.scrollIntoViewIfNeeded();
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false, `Page overflow ${width}/${lab}`);
        const file = path.join(output, `${lab}-${width}.png`); await item.screenshot({ path: file }); const bytes = fs.readFileSync(file); captures.push({ file: file.replaceAll('\\', '/'), bytes: bytes.length, sha256: crypto.createHash('sha256').update(bytes).digest('hex') });
      }
      const malformed = await root.locator('.katex-error').count(); assert.equal(malformed, 0);
      const collapsed = await root.locator('.katex svg').evaluateAll(items => items.filter(node => node.getBoundingClientRect().height < 1 && getComputedStyle(node).display !== 'none').length); assert.equal(collapsed, 0);
      const backgrounds = await root.locator('.neural-lab, .neural-lab button, .neural-lab select').evaluateAll(items => items.map(item => getComputedStyle(item).backgroundColor));
      assert.ok(backgrounds.every(color => { const rgb = color.match(/\d+/g); return !rgb || Number(rgb[1]) <= Math.max(Number(rgb[0]), Number(rgb[2])); }), 'Unexpected green surface');
    }
    groups.push('Desktop390/320 layouts, actual math glyph geometry, neutral surfaces and retained captures');
    assert.deepEqual(errors, []);
    fs.writeFileSync(reportPath, JSON.stringify({ passed: true, checkedAt: new Date().toISOString(), groups, captures, limits: 'Source/model/native proof and human visual review are separate; screenshots must be inspected.' }, null, 2) + '\n');
    console.log(JSON.stringify({ passed: true, groups: groups.length, captures: captures.length }));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

// Parent-owned production integration. No fitting or dependency installation.
const fs = require('node:fs'), path = require('node:path'), crypto = require('node:crypto'), assert = require('node:assert/strict');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const directory = 'docs/teaching/evidence/capsule-browser'; fs.mkdirSync(directory, { recursive: true });
const report = path.join(directory, 'report.json'), base = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4194';
fs.writeFileSync(report, JSON.stringify({ passed: false, status: 'running' }));
const groups = [], captures = [], latencies = [];
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1366, height: 1000 }, reducedMotion: 'reduce' });
    await require('./lib/lesson-browser-fonts.cjs')(page);
    const errors = [], requests = []; page.on('pageerror', error => errors.push(error.message)); page.on('request', request => requests.push(request.url()));
    await page.goto(`${base}/learn/path/full-curriculum/capsule-networks?module=deep-learning-fundamentals`);
    const root = page.locator('.capsule-lesson'); await root.waitFor();
    const result = name => root.locator(`[data-result="${name}"]`), click = name => root.getByRole('button', { name, exact: true }).click(), input = name => root.getByRole('spinbutton', { name, exact: true }), set = (name, value) => input(name).fill(String(value));
    assert.equal(await root.locator('[data-lab]').count(), 4);
    assert.equal(requests.some(url => /frozen-model|capsule-learning\.py|capsule-mechanics\.py/.test(url)), false);
    assert.match(await result('grouping').innerText(), /= 27/); await click('Primary cell row 2, column 1'); await root.getByRole('combobox', { name: 'Capsule type', exact: true }).selectOption('0'); assert.match(await result('grouping').innerText(), /= 36/); await click('Reset grouping');
    groups.push('Actual tensor-cell/type changes preserve the documented child index');
    assert.match(await result('routing').innerText(), /A 0.934615; B 0.778585/); await click('Oppose child 2’s A vote'); assert.match(await result('routing').innerText(), /parent B is longer/); await click('Zero all votes'); assert.match(await result('routing').innerText(), /A 0; B 0/); await click('Identical parents'); assert.match(await result('routing').innerText(), /equal lengths/); await click('Reset vote workshop');
    const beforeRotation = await result('routing').innerText(); await click('Rotate all votes 90°'); assert.equal(await result('routing').innerText(), beforeRotation); await click('Reset vote workshop'); await set('Vote x', -1.2); assert.doesNotMatch(await result('routing').innerText(), /A 0.934615; B 0.778585/); await set('Vote x', 4); assert.equal(await input('Vote x').getAttribute('aria-invalid'), 'true'); await input('Vote x').blur(); await click('Reset vote workshop');
    groups.push('Vote default/opposition/zero/symmetry/rotation, direct edit, invalid buffer and reset');
    assert.match(await result('squash').innerText(), /Radial sensitivity 0.64; tangent sensitivity 0.4/); await click('Long vector [3,4]'); assert.match(await result('squash').innerText(), /Radial sensitivity 0.014793; tangent sensitivity 0.192308/); await click('Zero perturbation'); assert.match(await result('squash').innerText(), /output change 0;/); await click('Zero vector'); assert.match(await root.locator('[data-lab="capsule-squash"]').innerText(), /no radial direction/); await click('Reset squash investigation');
    await click('Same correct count, different prediction'); assert.match(await result('evidence').innerText(), /Correct 117\/120; 1 predictions differ/); await click('Reset recorded comparison');
    groups.push('Finite versus derivative squash, zero cases and actual same-count/different-label evidence');
    await click('Make child 3 inactive'); const means = (await result('em').innerText()).split(';')[0]; await set('EM vote x', -4); assert.equal((await result('em').innerText()).split(';')[0], means); await click('Remove all active evidence'); assert.match(await result('em').innerText(), /not identified/);
    await set('Selected child activation', '1e-20'); assert.match(await result('em').innerText(), /A not reliably identified.*B not reliably identified/);
    assert.equal(await input('Selected child activation').inputValue(), '1e-20');
    const em = root.locator('[data-lab="capsule-em"]'); assert.equal(await em.locator('[data-em-identified=false]').count(), 2); assert.equal(await em.locator('ellipse, [data-em-mean]').count(), 0);
    await page.setViewportSize({ width: 390, height: 1000 });
    const guardedCapture = path.join(directory, 'capsule-em-guarded-390.png');
    await em.screenshot({ path: guardedCapture, style: '.workspace-nav, .learning-skip { visibility: hidden !important; }' });
    const guardedBytes = fs.readFileSync(guardedCapture); captures.push({ file: guardedCapture.replaceAll('\\', '/'), bytes: guardedBytes.length, sha256: crypto.createHash('sha256').update(guardedBytes).digest('hex') });
    await page.setViewportSize({ width: 1366, height: 1000 });
    await click('Reset EM investigation'); assert.match(await result('em').innerText(), /0.10881/); assert.equal(await em.locator('[data-em-identified=true]').count(), 2); assert.equal(await em.locator('ellipse').count(), 2);
    await set('Vote map horizontal scale', 1); assert.match(await result('geometry').innerText(), /norm 0/); await click('Reset coordinate frames'); groups.push('Inactive-child EM, absent-evidence state and actual commuting-map null');
    await root.locator('summary').filter({ hasText: 'Open the real pixel and latent-coordinate investigations' }).click(); await result('frozen').waitFor(); assert.match(await result('frozen').innerText(), /Original winner 4; current winner 4/);
    const initialScores = await result('frozen').innerText(); await click('Flip pixel 3,3'); assert.notEqual(await result('frozen').innerText(), initialScores); await click('Restore input'); assert.equal(await result('frozen').innerText(), initialScores);
    await click('Right'); assert.notEqual(await result('frozen').innerText(), initialScores); await click('Restore input'); await click('Zero shift'); assert.equal(await result('frozen').innerText(), initialScores);
    await click('Add 0.1 to the latent coordinate'); assert.doesNotMatch(await result('latent').innerText(), /Maximum pixel change 0\. Encoder/); assert.equal(await result('frozen').innerText(), initialScores); await click('Edit a masked-out class'); assert.match(await result('latent').innerText(), /Maximum pixel change 0\. Encoder/); await click('Reset frozen investigation');
    groups.push('Demand-loaded real model, changed pixel and shift, zero shift, decoder-only change, masked-out null and reset');
    const slider = root.getByRole('slider', { name: 'Selected pixel intensity slider', exact: true }); await set('Selected pixel intensity', .3); assert.equal(await slider.inputValue(), '0.3'); await slider.scrollIntoViewIfNeeded(); await slider.focus(); await slider.press('Home'); assert.equal(await input('Selected pixel intensity').inputValue(), '0'); await slider.press('End'); assert.equal(await input('Selected pixel intensity').inputValue(), '1');
    await slider.scrollIntoViewIfNeeded(); const box = await slider.boundingBox(); await page.mouse.move(box.x + box.width * .15, box.y + box.height / 2); await page.mouse.down(); await page.mouse.move(box.x + box.width * .75, box.y + box.height / 2, { steps: 5 }); await page.mouse.up(); assert.ok(Number(await input('Selected pixel intensity').inputValue()) > 0 && Number(await input('Selected pixel intensity').inputValue()) < 1);
    for (const value of [0, .25, .5, .75, 1]) { const start = Date.now(); await set('Selected pixel intensity', value); await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))); latencies.push(Date.now() - start); } await click('Reset frozen investigation');
    groups.push('Actual pixel pointer drag, keyboard range endpoints and five complete input-to-painted-result observations');
    for (const title of ['Read the complete trainable capsule program', 'Read the complete NumPy routing, derivative, geometry and EM program', 'Read the independent NumPy encoder and saved-state comparison']) await root.locator('summary').filter({ hasText: title }).click(); await root.locator('.neural-program-source').first().waitFor(); assert.equal(await root.locator('.neural-program-source').count(), 3);
    for (const width of [1366, 390, 320]) {
      await page.setViewportSize({ width, height: 1000 }); await page.evaluate(() => document.fonts.ready);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false, `${width}: page overflow`);
      const outside = await root.locator('.capsule-plane').evaluateAll(charts => charts.flatMap(chart => { const frame = chart.getBoundingClientRect(); return [...chart.querySelectorAll('text')].filter(label => { const r = label.getBoundingClientRect(); return r.left < frame.left - .5 || r.right > frame.right + .5 || r.top < frame.top - .5 || r.bottom > frame.bottom + .5; }).map(label => label.textContent); })); assert.deepEqual(outside, [], `${width}: SVG labels`);
      const smallPixels = await root.locator('.capsule-pixels-editable button').evaluateAll(items => items.filter(item => item.getBoundingClientRect().width < 43.5 || item.getBoundingClientRect().height < 43.5).length); assert.equal(smallPixels, 0);
      for (const lab of ['capsule-votes', 'capsule-frozen', 'capsule-em']) { const item = root.locator(`[data-lab="${lab}"]`); await item.scrollIntoViewIfNeeded(); const file = path.join(directory, `${lab}-${width}.png`); await item.screenshot({ path: file, style: '.workspace-nav, .learning-skip { visibility: hidden !important; }' }); const bytes = fs.readFileSync(file); captures.push({ file: file.replaceAll('\\', '/'), bytes: bytes.length, sha256: crypto.createHash('sha256').update(bytes).digest('hex') }); }
      assert.equal(await root.locator('.katex-error').count(), 0); assert.equal(await root.locator('.katex svg').evaluateAll(items => items.filter(item => item.getBoundingClientRect().height < 1 && getComputedStyle(item).display !== 'none').length), 0);
      const backgrounds = await root.locator('.neural-lab, button, select').evaluateAll(items => items.map(item => getComputedStyle(item).backgroundColor)); assert.ok(backgrounds.every(color => { const rgb = color.match(/\d+/g); return !rgb || Number(rgb[1]) <= Math.max(Number(rgb[0]), Number(rgb[2])); }));
    }
    assert.deepEqual(errors, []); groups.push('Opened programs, desktop390/320 geometry, full-size pixel targets, font/math bounds, neutral theme and retained captures');
    const retryPage = await browser.newPage({ viewport: { width: 390, height: 1000 } });
    await require('./lib/lesson-browser-fonts.cjs')(retryPage);
    let damaged = false;
    await retryPage.route('**/frozen-model.f32', route => { if (!damaged) { damaged = true; return route.fulfill({ status: 200, body: Buffer.alloc(16), contentType: 'application/octet-stream' }); } return route.continue(); });
    await retryPage.goto(`${base}/learn/path/full-curriculum/capsule-networks?module=deep-learning-fundamentals`);
    await retryPage.locator('.capsule-frozen > summary').click(); await retryPage.getByRole('button', { name: 'Retry frozen model', exact: true }).waitFor();
    assert.equal(await retryPage.locator('[data-result="frozen"]').count(), 0); await retryPage.getByRole('button', { name: 'Retry frozen model', exact: true }).click(); await retryPage.locator('[data-result="frozen"]').waitFor(); await retryPage.close();
    groups.push('Malformed binary yields honest unavailable state; explicit retry loads the actual model without a substitute');
    fs.writeFileSync(report, JSON.stringify({ passed: true, checkedAt: new Date().toISOString(), base, groups, captures, inputToPaintMs: latencies, limits: 'Input-to-paint includes automation and two frames; not a pure inference benchmark. Only screenshot capture suppresses fixed navigation and skip-link overlays; interaction and geometry checks use the unchanged page. Captures require actual visual inspection. Native and independent review are separate.' }, null, 2) + '\n');
    console.log(JSON.stringify({ passed: true, groups: groups.length, captures: captures.length }));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

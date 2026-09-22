const assert = require('node:assert/strict');
const fs = require('node:fs');
const { createHash } = require('node:crypto');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const base = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4194';
const read = p => JSON.parse(fs.readFileSync(p, 'utf8').replace(/^\uFEFF/, ''));
const hash = p => createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const sources = ['src/learn/components/lesson-labs/GaussianProcessLabs.jsx', 'src/learn/components/lesson-labs/gaussian-process.css', 'src/learn/data/k-means-hierarchical-models.js', 'src/learn/components/lesson-labs/KMeansHierarchicalLabs.jsx', 'scripts/verify-classical-control-followup-browser.cjs'];
const receipt = 'docs/teaching/evidence/classical-control-followup-browser.json';
const record = { checkedAt: new Date().toISOString(), base, passed: false, sourceHashes: Object.fromEntries(sources.map(p => [p, hash(p)])), manifestHash: hash('dist/.vite/manifest.json'), groups: [], screenshots: [], errors: [] };
const save = () => fs.writeFileSync(receipt, `${JSON.stringify(record, null, 2)}\n`);
save();
const close = (a, b) => assert.ok(Math.abs(a - b) < 1e-6, `${a} versus ${b}`);
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const context = await browser.newContext({ viewport: { width: 1366, height: 900 }, hasTouch: true });
    const fonts = read(process.env.LEARNING_FONT_FIXTURES || 'scratch/kmeans-revision-review/fonts/manifest.json');
    await context.route('https://fonts.googleapis.com/**', r => r.fulfill({ path: fonts.stylesheet, contentType: 'text/css' }));
    await context.route('https://fonts.gstatic.com/**', r => fonts.files[r.request().url()] ? r.fulfill({ path: fonts.files[r.request().url()], contentType: 'font/ttf' }) : r.abort());
    const page = await context.newPage();
    page.on('pageerror', error => record.errors.push(error.message));
    page.on('console', message => { if (message.type() === 'error' && !/Failed to load resource/.test(message.text())) record.errors.push(message.text()); });
    const open = async id => {
      await page.goto(`${base}/learn/path/full-curriculum/${id}?module=classical-ml`, { waitUntil: 'domcontentloaded' });
      await page.locator('.reader-article h2,.reader-article h3').first().waitFor();
      await page.evaluate(() => document.fonts.ready);
    };
    const capture = async (lab, name) => {
      const path = `docs/teaching/evidence/screenshots/classical-controls-${name}.png`;
      const style = await page.addStyleTag({ content: '.learn-nav {visibility:hidden!important}' });
      try { await lab.screenshot({ path }); } finally { await style.evaluate(node => node.remove()); }
      record.screenshots.push({ path, sha256: hash(path) });
    };
    await open('gaussian-processes-gp');
    const fields = await page.locator('.gp-numeric-field').evaluateAll(nodes => nodes.map(n => {
      const number = n.querySelector('input[type=number]'), range = n.querySelector('input[type=range]');
      return { label: [...number.labels].map(l => l.textContent).join(''), number: number.value, range: range.value };
    }));
    assert.equal(fields.length, 21);
    for (const field of fields) { assert.ok(field.label.trim()); assert.equal(Number(field.number), Number(field.range), field.label); }
    record.groups.push({ name: 'All 21 typed fields have individual accessible labels and exact initial slider values', fields });
    const gp = page.locator('[data-gp-lab=direct]');
    const mean = async () => Number(await gp.locator('tbody tr').filter({ hasText: /^Mean/ }).locator('td').last().innerText());
    await gp.getByRole('spinbutton', { name: 'Observed value', exact: true }).fill('3');
    close(await mean(), 1.2);
    assert.equal(await gp.getByRole('slider', { name: 'Observed value slider' }).inputValue(), '3');
    await gp.getByRole('slider', { name: 'Covariance ρ slider' }).press('ArrowRight');
    let covariance = Number(await gp.getByRole('slider', { name: 'Covariance ρ slider' }).inputValue());
    assert.notEqual(covariance, 0.5);
    close(Number(await gp.getByRole('spinbutton', { name: 'Covariance ρ', exact: true }).inputValue()), covariance);
    close(await mean(), covariance * 3 / 1.25);
    const gpSlider = gp.getByRole('slider', { name: 'Covariance ρ slider' });
    await gpSlider.scrollIntoViewIfNeeded();
    const gpBox = await gpSlider.boundingBox();
    await gpSlider.click({ position: { x: gpBox.width * 0.7, y: gpBox.height / 2 } });
    covariance = Number(await gpSlider.inputValue());
    close(Number(await gp.getByRole('spinbutton', { name: 'Covariance ρ', exact: true }).inputValue()), covariance);
    close(await mean(), covariance * 3 / 1.25);
    record.groups.push({ name: 'Typed, keyboard and pointer edits share one model state and update the independent conditional-mean formula' });
    await gp.getByRole('spinbutton', { name: 'Observed value', exact: true }).fill('');
    assert.equal(await gp.getByRole('alert').count(), 1);
    assert.equal(await gp.locator('[data-gp-result]').count(), 0);
    await gp.getByRole('slider', { name: 'Observed value slider' }).press('ArrowRight');
    assert.equal(await gp.getByRole('alert').count(), 0);
    await gp.getByRole('button', { name: 'Reset investigation', exact: true }).click();
    close(await mean(), 0.8);
    record.groups.push({ name: 'Invalid typed input is explained, slider edits recover, reset restores the original conditional mean' });
    for (const width of [390, 320]) {
      await page.setViewportSize({ width, height: 900 });
      const sizes = await gp.locator('input').evaluateAll(nodes => nodes.map(n => n.getBoundingClientRect().toJSON()));
      assert.ok(sizes.every(r => r.width > 30 && r.left >= 0 && r.right <= width + 1));
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
      await capture(gp, `gp-${width}`);
      record.groups.push({ name: `GP controls fit ${width}px viewport without page overflow` });
    }
    await page.setViewportSize({ width: 1366, height: 900 });
    await open('k-means-hierarchical-clustering');
    const km = page.locator('.kh-investigation').filter({ has: page.getByRole('heading', { name: 'A change of units can change the clustering question' }) });
    const weight = km.getByRole('slider', { name: /Weight on squared vertical differences/ });
    const score = async () => Number((await km.locator('.kh-readout strong').innerText()).replace('SSE ', ''));
    await weight.press('ArrowRight');
    assert.equal(await weight.inputValue(), '1.01'); close(await score(), 1.01);
    await weight.fill('2.37'); close(await score(), 2.37);
    await km.getByRole('combobox', { name: 'Vertical measurement unit' }).selectOption('10');
    close(await score(), 9);
    await weight.press('Home'); assert.equal(await weight.inputValue(), '0.01'); close(await score(), 1);
    await weight.press('End'); assert.equal(await weight.inputValue(), '4'); close(await score(), 9);
    await km.getByRole('button', { name: 'Reset', exact: true }).click();
    assert.equal(await weight.inputValue(), '1'); close(await score(), 1);
    record.groups.push({ name: 'K-Means nonpreset weights, both units, slider endpoints and reset match independent rectangle optima without a render error' });
    for (const width of [390, 320]) {
      await page.setViewportSize({ width, height: 900 });
      await weight.scrollIntoViewIfNeeded();
      const box = await weight.boundingBox();
      await page.touchscreen.tap(box.x + box.width * 0.6, box.y + box.height / 2);
      const current = Number(await weight.inputValue());
      assert.notEqual(current, 1);
      close(await score(), Math.min(9, current));
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
      await capture(km, `kmeans-${width}`);
      record.groups.push({ name: `K-Means slider touch and output at ${width}px` });
      await km.getByRole('button', { name: 'Reset', exact: true }).click();
    }
    assert.deepEqual(record.errors, []);
    assert.equal(await page.locator('.lesson-load-error').count(), 0);
    record.passed = true; save();
    console.log(`PASS ${record.groups.length} focused GP / K-Means production groups.`);
  } catch (error) { record.errors.push(error.stack); save(); throw error; }
  finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

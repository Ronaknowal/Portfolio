// Run against the parent's production build; no training or pretrained downloads.
const fs = require('node:fs'), path = require('node:path'), crypto = require('node:crypto'), assert = require('node:assert/strict');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const directory = 'docs/teaching/evidence/landmark-architecture-browser'; fs.mkdirSync(directory, { recursive: true });
const report = path.join(directory, 'report.json'); fs.writeFileSync(report, JSON.stringify({ passed: false, status: 'running' }));
const base = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4194', groups = [], captures = [];
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 1366, height: 1000 }, reducedMotion: 'reduce' });
    await require('./lib/lesson-browser-fonts.cjs')(page);
    const errors = [], requested = []; page.on('pageerror', error => errors.push(error.message)); page.on('request', request => requested.push(request.url()));
    await page.goto(`${base}/learn/path/full-curriculum/landmark-architectures-lenet-alexnet-vgg-resnet-efficientnet?module=deep-learning-fundamentals`);
    const root = page.locator('.landmark-lesson'); await root.waitFor(); assert.equal(await root.locator('[data-lab]').count(), 5);
    assert.equal(requested.some(url => /recorded-feature-maps\.json|landmark_builders\.py|architecture-experiments\.py/.test(url)), false);
    const result = name => root.locator(`[data-result="${name}"]`), click = name => root.getByRole('button', { name, exact: true }).click();
    const input = name => root.getByRole('spinbutton', { name, exact: true });
    const set = async (name, value) => input(name).fill(String(value));
    assert.match(await result('head').innerText(), /123,642,856/); await click('Changed seven-class direct head'); assert.match(await result('head').innerText(), /43,911/);
    await click('One-position equality'); assert.match(await result('head').innerText(), /903/); await click('Reset head budget');
    await set('Final channels', 0); assert.equal(await input('Final channels').getAttribute('aria-invalid'), 'true'); assert.match(await result('head').innerText(), /123,642,856/); await input('Final channels').blur();
    groups.push('Exact dense/direct/GAP counts, one-position null, invalid dimension and reset');
    assert.match(await result('context').innerText(), /0.73106, 0.26894/); await click('Raise B’s mean to 3'); assert.match(await result('context').innerText(), /Current gates: 0.5, 0.5/);
    await click('Same means, varied cells'); const gateBefore = await result('context').innerText(); await click('Rotate both maps 180°'); assert.equal(await result('context').innerText(), gateBefore); await click('Reset context gate');
    await set('Context channel B selected cell 1,1', 8); assert.doesNotMatch(await result('context').innerText(), /Current gates: 0.73106, 0.26894/); await click('Reset context gate'); groups.push('Real context cell edit, cross-channel gate change, mean-preserving rotation and reset');
    await click('Resolution √2'); assert.match(await result('scaling').innerText(), /Parameter factor 1; MAC factor 2/);
    await set('Depth multiplier', 1.5); await set('Resolution multiplier', 1.25); await set('Target MAC budget multiplier', 2); await click('Apply width that meets this budget'); assert.match(await root.locator('[data-lab="landmark-scaling"]').innerText(), /Infeasible here/);
    await set('Target MAC budget multiplier', 3); await click('Apply width that meets this budget'); assert.match(await result('scaling').innerText(), /MAC factor 3/); await click('Reset scaling investigation'); groups.push('Equal-compute contrast and actual feasible/infeasible solve without clamping');
    assert.match(await result('eligibility').innerText(), /Parallel branches, Inverted with gate/); await click('Exact parallel boundary'); assert.match(await result('eligibility').innerText(), /Eligible: Parallel branches\./);
    await set('Conv and linear MAC budget', 41919); assert.match(await result('eligibility').innerText(), /Eligible: none/); await click('Reset recorded comparison');
    await root.getByRole('combobox', { name: 'Architecture seed', exact: true }).selectOption('2'); assert.match(await result('architecture-pair').innerText(), /plain 115\/120; Residual 115\/120/); await root.getByRole('combobox', { name: 'Architecture seed', exact: true }).selectOption('3'); assert.match(await result('architecture-pair').innerText(), /plain 112\/120; Residual 114\/120/); await click('Reset recorded comparison'); groups.push('Exact cost boundary, below-bound exclusion, seed tie and reversed comparison');
    assert.match(await result('score').innerText(), /= 2.5/); await click('Increase negative-weight cell'); assert.match(await result('score').innerText(), /= 1.5/); const scoreBefore = await result('score').innerText(); await click('Rotate feature positions 180°'); assert.match(await result('score').innerText(), /= 1.5/); assert.notEqual(await result('score').innerText(), scoreBefore);
    await click('Zero class weights'); assert.match(await result('score').innerText(), /= 0.5/); await click('Reset score construction'); groups.push('Signed contribution edit, permutation changes map but preserves score, zero-weight null');
    const slider = root.getByRole('slider', { name: 'Class weight A slider', exact: true }); await slider.scrollIntoViewIfNeeded(); await slider.focus(); await slider.press('Home'); assert.equal(await input('Class weight A').inputValue(), '-4'); await slider.press('End'); assert.equal(await input('Class weight A').inputValue(), '4');
    await slider.scrollIntoViewIfNeeded(); const box = await slider.boundingBox(); await page.mouse.move(box.x + box.width * .3, box.y + box.height / 2); await page.mouse.down(); await page.mouse.move(box.x + box.width * .65, box.y + box.height / 2, { steps: 6 }); await page.mouse.up(); assert.ok(Number(await input('Class weight A').inputValue()) > 0 && Number(await input('Class weight A').inputValue()) < 4); await click('Reset score construction'); groups.push('Actual pointer drag and keyboard endpoints update the signed model');
    await root.getByRole('button', { name: 'Intermediate row 1, column 1', exact: true }).click(); assert.equal(await root.getByRole('button', { name: 'Intermediate row 1, column 1', exact: true }).getAttribute('aria-pressed'), 'true'); assert.equal(await root.locator('.landmark-small-grid[aria-label]').filter({ has: page.locator('[data-active=true]') }).count() >= 1, true); groups.push('Selected intermediate position reveals its real input patch');
    await root.locator('summary').filter({ hasText: 'Inspect actual trained feature maps, including a mistake' }).click(); await root.locator('[data-result="recorded-map"]').waitFor(); assert.ok(requested.some(url => /recorded-feature-maps\.json/.test(url)));
    assert.match(await root.locator('[data-lab="landmark-recorded-maps"]').innerText(), /Source 379; actual digit 8; model chooses 5/); await root.getByRole('combobox', { name: 'Class map', exact: true }).selectOption('8'); assert.match(await root.locator('[data-lab="landmark-recorded-maps"]').innerText(), /Current class map: 8/); await click('Reset saved-map view');
    for (const title of ['Read the complete small architecture experiment', 'Read all five explicit architecture builders and the library comparison']) await root.locator('summary').filter({ hasText: title }).click(); await root.locator('.neural-program-source').nth(1).waitFor(); assert.equal(await root.locator('.neural-program-source').count(), 2); groups.push('Deferred actual observations and complete source load only when opened');
    for (const width of [1366, 390, 320]) {
      await page.setViewportSize({ width, height: 1000 });
      await page.evaluate(() => document.fonts.ready);
      const escapedGrid = await root.locator('.landmark-intermediate').evaluate(grid => {
        const box = grid.getBoundingClientRect(), container = grid.parentElement.getBoundingClientRect();
        return box.left < container.left - .5 || box.right > container.right + .5;
      });
      assert.equal(escapedGrid, false, `${width}: intermediate grid must stay inside its convolution-stage container`);
      for (const lab of ['landmark-context', 'landmark-score', 'landmark-recorded-maps']) {
        const item = root.locator(`[data-lab="${lab}"]`); await item.scrollIntoViewIfNeeded(); assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false, `${width}/${lab} overflow`);
        // Capture the real viewport. Expanding a viewport to screenshot a very
        // tall lesson section can repeatedly reflow its flex stages and fixed nav.
        const file = path.join(directory, `${lab}-${width}.png`); await page.screenshot({ path: file, animations: 'disabled' }); const bytes = fs.readFileSync(file); captures.push({ file: file.replaceAll('\\', '/'), bytes: bytes.length, sha256: crypto.createHash('sha256').update(bytes).digest('hex') });
      }
      assert.equal(await root.locator('.katex-error').count(), 0); assert.equal(await root.locator('.katex svg').evaluateAll(items => items.filter(item => item.getBoundingClientRect().height < 1 && getComputedStyle(item).display !== 'none').length), 0);
      const colors = await root.locator('.neural-lab, button, select').evaluateAll(items => items.map(item => getComputedStyle(item).backgroundColor)); assert.ok(colors.every(color => { const rgb = color.match(/\d+/g); return !rgb || Number(rgb[1]) <= Math.max(Number(rgb[0]), Number(rgb[2])); }));
    }
    await root.locator('.landmark-intermediate').scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(directory, 'composition-320.png'), animations: 'disabled' });
    assert.deepEqual(errors, []); groups.push('Desktop390/320 page geometry, rendered math, neutral control surfaces and retained viewport captures');
    fs.writeFileSync(report, JSON.stringify({ passed: true, checkedAt: new Date().toISOString(), groups, captures, limits: 'Screenshots still need human/model visual inspection; independent native/model review is separate.' }, null, 2) + '\n');
    console.log(JSON.stringify({ passed: true, groups: groups.length, captures: captures.length }));
  } finally { await browser.close(); }
})().catch(error => { fs.writeFileSync(report, JSON.stringify({ passed: false, checkedAt: new Date().toISOString(), base, completedGroups: groups, failure: error.stack }, null, 2)); console.error(error); process.exitCode = 1; });

const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const { createHash } = require('node:crypto');

(async () => {
  const base = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4184';
  const directory = 'scratch/crf-implementation';
  fs.mkdirSync(directory, { recursive: true });
  const checks = [], screenshots = [], errors = [];
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    for (const width of [1440, 390, 320]) {
      const context = await browser.newContext({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      const page = await context.newPage();
      page.on('pageerror', error => errors.push(error.message));
      await page.goto(`${base}/learn/path/full-curriculum/conditional-random-fields-crf?module=classical-ml`, { waitUntil: 'domcontentloaded', timeout: 60000 });
      const lesson = page.locator('.crf-lesson');
      await lesson.waitFor();
      await page.evaluate(() => document.fonts.ready);
      assert.equal(await lesson.locator('.katex-error').count(), 0);
      assert.equal(await lesson.locator('h2').count(), 11);
      for (const anchor of await lesson.locator('.lesson-intro nav a').evaluateAll(nodes => nodes.map(node => node.getAttribute('href')))) assert.equal(await lesson.locator(`[id="${anchor.slice(1)}"]`).count(), 1, anchor);
      checks.push({ width, check: 'complete lesson and math; all navigation anchors resolve' });
      const first = lesson.locator('[data-crf-lab="trellis"]');
      await first.getByRole('button', { name: 'Pair reverses the choice', exact: true }).click();
      await first.getByRole('combobox').selectOption('different');
      await first.getByRole('button', { name: 'Commit prediction', exact: true }).click();
      await first.getByRole('button', { name: 'Calculate', exact: true }).click();
      assert.match(await first.getByRole('status').innerText(), /Prediction matched/);
      assert.match(await first.locator('.crf-result').innerText(), /Independent winners: AA \(Z = 32\). Chain winners: AB \(Z = 50\)/);
      const painted = await first.locator('.crf-trellis line').evaluateAll(lines => lines.map(line => ({ stroke: getComputedStyle(line).stroke, width: getComputedStyle(line).strokeWidth })));
      assert.equal(painted.filter(line => line.width === '3px').length, 1);
      assert.equal(painted.filter(line => line.width === '1px').length, 3);
      await first.getByRole('button', { name: 'Trace AA', exact: true }).click();
      assert.match(await first.locator('.crf-result').innerText(), /Selected path AA: first input 3 × pair 1 × second input 6 = 18/);
      checks.push({ width, check: 'committed contrasting prediction, selected factor trace and painted edge encoding' });
      if (width !== 320) {
        const path = `${directory}/trellis-contrast-${width}.png`;
        await first.locator('.crf-result').screenshot({ path, style: '.learn-nav { visibility: hidden !important; }' }); screenshots.push(path);
      }
      await first.getByRole('spinbutton', { name: 'Position 2 · A', exact: true }).fill('12');
      assert.equal(await first.getByRole('button', { name: 'Calculate', exact: true }).isDisabled(), true);
      assert.equal(await first.locator('.crf-result').count(), 0);
      await first.getByRole('button', { name: 'Explore without a prediction', exact: true }).click();
      assert.match(await first.getByRole('status').innerText(), /No prediction was graded/);
      assert.match(await first.locator('.crf-result').innerText(), /Independent winners: AA \(Z = 56\). Chain winners: AA \(Z = 74\)/);
      await first.getByRole('spinbutton', { name: 'Position 2 · A', exact: true }).fill('0');
      assert.equal(await first.getByRole('button', { name: 'Explore without a prediction', exact: true }).isDisabled(), true);
      assert.equal(await first.locator('input[aria-invalid="true"]').count(), 1);
      await first.getByRole('button', { name: 'Four-way tie', exact: true }).click();
      await first.getByRole('combobox').selectOption('same');
      await first.getByRole('button', { name: 'Commit prediction', exact: true }).click();
      await first.getByRole('button', { name: 'Calculate', exact: true }).click();
      assert.match(await first.locator('.crf-result').innerText(), /Chain winners: AA, AB, BA, BB/);
      await first.getByRole('button', { name: 'Reset', exact: true }).click();
      assert.equal(await first.getByRole('combobox').inputValue(), '');
      assert.equal(await first.getByRole('button', { name: 'Calculate', exact: true }).isDisabled(), true);
      checks.push({ width, check: 'stale prediction and output retired, ungraded null, invalid factors, all ties and reset' });
      const bias = lesson.locator('[data-crf-lab="bias"]');
      await bias.getByRole('spinbutton', { name: 'Later A factor', exact: true }).fill('.02');
      await bias.getByRole('combobox').selectOption('decrease');
      await bias.getByRole('button', { name: 'Commit prediction', exact: true }).click();
      await bias.getByRole('button', { name: 'Apply', exact: true }).click();
      assert.match(await bias.getByRole('status').innerText(), /Prediction matched/);
      assert.match(await bias.locator('.crf-result').innerText(), /0.019608/);
      if (width !== 320) { const path = `${directory}/normalizers-${width}.png`; await bias.locator('.crf-result').screenshot({ path, style: '.learn-nav { visibility: hidden !important; }' }); screenshots.push(path); }
      await bias.getByRole('button', { name: 'Equal evidence, changed prior', exact: true }).click();
      assert.equal(await bias.getByRole('button', { name: 'Apply', exact: true }).isDisabled(), true);
      await bias.getByRole('combobox').selectOption('unchanged');
      await bias.getByRole('button', { name: 'Commit prediction', exact: true }).click();
      await bias.getByRole('button', { name: 'Apply', exact: true }).click();
      assert.match(await bias.getByRole('status').innerText(), /matched: route A is unchanged/);
      await bias.getByRole('spinbutton', { name: 'First-branch probability p', exact: true }).fill('.25');
      await bias.getByRole('spinbutton', { name: 'Later A factor', exact: true }).fill('3');
      await bias.getByRole('button', { name: 'Explore without a prediction', exact: true }).click();
      const bars = await bias.locator('.crf-bar-row').allTextContents();
      assert.ok(bars.includes('Local route A0.25') && bars.includes('Global route A0.5'));
      await bias.getByRole('button', { name: 'Reset', exact: true }).click();
      assert.equal(await bias.getByRole('combobox').inputValue(), '');
      checks.push({ width, check: 'label-bias contrast and changed-prior null, independent transfer, stale prediction and reset' });
      const bio = lesson.getByRole('figure', { name: 'BIO predecessor rules and hard start mask' });
      await bio.getByRole('combobox').selectOption('I-ORG');
      assert.equal(await bio.locator('.crf-bio-routes > div').filter({ hasText: '→ allowed' }).count(), 2);
      const real = lesson.getByRole('figure', { name: 'Inspect real EWT development tagging errors' });
      assert.ok(await real.locator('li[data-error="true"]').count());
      await real.locator('li[data-error="true"] button').first().click();
      const marginalRows = await real.locator('.crf-real-detail .crf-bar-row strong').allTextContents();
      assert.ok(Math.abs(marginalRows.map(Number).reduce((a,b)=>a+b,0) - 1) < 2e-6);
      if (width !== 320) { const path = `${directory}/real-errors-${width}.png`; await real.screenshot({ path, style: '.learn-nav { visibility: hidden !important; }' }); screenshots.push(path); }
      const count = lesson.getByRole('figure', { name: 'Observed and expected AB feature counts' });
      await count.getByRole('combobox').selectOption('AA');
      assert.match(await count.innerText(), /− 0.8 = -0.8/);
      checks.push({ width, check: 'BIO same-type rules, real default errors and marginal agreement, negative training direction' });
      // Keyboard edit and action, not just programmatic clicking.
      const input = bias.getByRole('spinbutton', { name: 'Later A factor', exact: true });
      await input.focus(); await page.keyboard.press('Control+A'); await page.keyboard.type('1'); await page.keyboard.press('Tab');
      assert.equal(await input.inputValue(), '1');
      const explore = bias.getByRole('button', { name: 'Explore without a prediction', exact: true });
      await explore.focus(); await page.keyboard.press('Enter');
      assert.match(await bias.getByRole('status').innerText(), /No prediction graded/);
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth > innerWidth + 1), false);
      const svgIssues = await lesson.locator('svg text').evaluateAll(nodes => nodes.filter(node => {const text=node.getBoundingClientRect(); const svg=node.ownerSVGElement.getBoundingClientRect(); return text.left < svg.left-1 || text.right > svg.right+1 || text.top < svg.top-1 || text.bottom > svg.bottom+1;}).map(node=>node.textContent));
      assert.deepEqual(svgIssues, []);
      assert.equal(await lesson.locator('.python-example').count(), 2);
      assert.match(await lesson.locator('.python-example').nth(1).innerText(), /selected chain \[0.791892 0.175/);
      checks.push({ width, check: 'keyboard editing and activation, no page overflow, SVG label bounds, exact native output shown' });
      if (width !== 320) for (const [name, accessible] of [
        ['factor-chain', 'Observed words and the output factor chain'],
        ['messages', 'Prefix and suffix messages meet'],
        ['counts', 'Observed and expected AB feature counts'],
        ['bio-rules', 'BIO predecessor rules and hard start mask'],
        ['neural-flow', 'Neural encoder and CRF training versus decoding'],
      ]) {
        const path = `${directory}/${name}-${width}.png`;
        await lesson.getByRole('figure', {name:accessible, exact:true}).screenshot({path, style: '.learn-nav { visibility: hidden !important; }'}); screenshots.push(path);
      }
      await context.close();
    }
    assert.deepEqual(errors, []);
    const evidence = { status: 'passed', base, checks, screenshots: screenshots.map(path => ({ path, sha256: createHash('sha256').update(fs.readFileSync(path)).digest('hex') })), errors, visualInspection: 'Screenshots retained for a separate human/model visual pass; these assertions are not image inspection.' };
    fs.writeFileSync('docs/teaching/evidence/crf-browser.json', JSON.stringify(evidence,null,2)+'\n');
    console.log(JSON.stringify({ status: 'passed', checks: checks.length, screenshots: screenshots.length }));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });


const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const { createHash } = require('node:crypto');

const base = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4184';
const output = 'scratch/active-learning-review';
fs.mkdirSync(output, { recursive: true });
const results = [];
const screenshots = [];
const errors = [];
async function capture(locator, name) {
  await locator.scrollIntoViewIfNeeded();
  const path = `${output}/${name}.png`;
  await locator.screenshot({ path, animations: 'disabled', style: '.learn-nav { visibility: hidden !important; }' });
  screenshots.push({ path, sha256: createHash('sha256').update(fs.readFileSync(path)).digest('hex') });
}
async function assertNoResult(lab) { assert.equal(await lab.locator('.active-result').count(), 0, 'No computed result before commitment'); }
async function checkResult(lab, phrase) {
  const text = await lab.locator('.active-result').innerText();
  assert.ok(text.includes(phrase), text);
}

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    for (const width of [1440, 780, 390, 320]) {
      const context = await browser.newContext({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      const page = await context.newPage();
      page.on('pageerror', error => errors.push(error.message));
      await page.goto(`${base}/learn/path/full-curriculum/active-learning?module=classical-ml`, { waitUntil: 'domcontentloaded' });
      const lesson = page.locator('.active-learning-lesson');
      await lesson.waitFor();
      assert.equal(await lesson.locator('[data-active-figure]').count(), 7);
      assert.equal(await lesson.locator('[data-active-lab]').count(), 3);
      assert.equal(await lesson.locator('.active-section-anchor').count(), 10);
      assert.equal(await lesson.locator('.active-answer').count(), 16);
      const route = await lesson.locator('nav[aria-label="In this lesson"] a').evaluateAll(links => links.map(link => ({ href: link.getAttribute('href'), exists: Boolean(document.querySelector(link.getAttribute('href'))) })));
      assert.equal(route.length, 10); assert.ok(route.every(link => link.exists));
      assert.equal(await lesson.locator('.katex-error').count(), 0);
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), `Page overflow ${width}`);
      results.push({ width, check: 'full reader structure, math, route and page bounds' });
      if ([1440, 390].includes(width)) {
        for (const figure of await lesson.locator('[data-active-figure]').all()) await capture(figure, `${width}-${await figure.getAttribute('data-active-figure')}`);
      }

      const threshold = lesson.locator('[data-active-lab="threshold"]');
      await threshold.getByRole('button', { name: 'Start run', exact: true }).click();
      await assertNoResult(threshold);
      assert.equal(await threshold.getByLabel('Simulation threshold', { exact: false }).count(), 0);
      assert.ok(await threshold.getByRole('button', { name: 'Acquire this answer', exact: true }).isDisabled());
      await threshold.getByLabel('Predict survivors if answer 0', { exact: true }).fill('4');
      await threshold.getByLabel('Predict survivors if answer 1', { exact: true }).fill('4');
      await threshold.getByRole('button', { name: 'Record prediction', exact: true }).focus();
      await page.keyboard.press('Enter');
      await threshold.getByRole('button', { name: 'Acquire this answer', exact: true }).click();
      await checkResult(threshold, 'Prediction matches.');
      assert.ok((await threshold.innerText()).includes('2 seed + 1 newly acquired = 3 total'));
      if ([1440, 390].includes(width)) await capture(threshold, `${width}-threshold-informed`);
      await threshold.getByRole('button', { name: 'Choose another question', exact: true }).click();
      await assertNoResult(threshold);
      await threshold.getByLabel('Unused query', { exact: true }).selectOption('6');
      await threshold.getByLabel('Predict survivors if answer 0', { exact: true }).fill('2');
      await threshold.getByLabel('Predict survivors if answer 1', { exact: true }).fill('2');
      await threshold.getByRole('button', { name: 'Record prediction', exact: true }).click();
      await threshold.getByRole('button', { name: 'Acquire this answer', exact: true }).click();
      await checkResult(threshold, 'Prediction matches.');
      await threshold.getByRole('button', { name: 'Reset run', exact: true }).click();
      await threshold.getByRole('button', { name: 'Start run', exact: true }).click();
      await threshold.getByLabel('Unused query', { exact: true }).selectOption('0');
      await threshold.getByLabel('Predict survivors if answer 0', { exact: true }).fill('8');
      await threshold.getByLabel('Predict survivors if answer 1', { exact: true }).fill('0');
      await threshold.getByRole('button', { name: 'Record prediction', exact: true }).click();
      await threshold.getByRole('button', { name: 'Acquire this answer', exact: true }).click();
      await checkResult(threshold, 'This legal query eliminated no hypothesis.');
      await threshold.getByRole('button', { name: 'Reset run', exact: true }).click();
      await threshold.getByLabel('Oracle mode', { exact: true }).selectOption('manual');
      await threshold.getByRole('button', { name: 'Start run', exact: true }).click();
      await threshold.getByLabel('Unused query', { exact: true }).selectOption('8');
      await threshold.getByLabel('Predict survivors if answer 0', { exact: true }).fill('0');
      await threshold.getByLabel('Predict survivors if answer 1', { exact: true }).fill('8');
      await threshold.getByRole('button', { name: 'Record prediction', exact: true }).click();
      await threshold.getByRole('button', { name: 'Acquire this answer', exact: true }).click();
      assert.ok((await threshold.innerText()).includes('No listed threshold is consistent.'));
      assert.equal(await threshold.getByRole('button', { name: 'Acquire this answer', exact: true }).count(), 0);
      results.push({ width, check: 'threshold keyboard acquisition, sequential state, null, contradiction and reset' });

      const committee = lesson.locator('[data-active-lab="committee"]');
      await assertNoResult(committee);
      assert.ok(!(await committee.innerText()).includes('Calculated D'));
      await committee.getByRole('button', { name: 'Shared ambiguity', exact: true }).click();
      await committee.getByLabel('Predict D relative to reference', { exact: true }).selectOption('smaller');
      await committee.getByLabel('Predict D in nats', { exact: true }).fill('0');
      await committee.getByRole('button', { name: 'Record prediction', exact: true }).click();
      await committee.getByRole('button', { name: 'Reveal decomposition', exact: true }).click();
      await checkResult(committee, 'Prediction matches.');
      if ([1440, 390].includes(width)) await capture(committee, `${width}-committee-null`);
      await committee.getByLabel('Member 1 class 0', { exact: true }).fill('0.9');
      await assertNoResult(committee);
      await committee.getByRole('button', { name: 'Record prediction', exact: true }).click();
      assert.ok((await committee.getByRole('alert').innerText()).includes('sum to 1'));
      await committee.getByRole('button', { name: 'Normalize member 1', exact: true }).click();
      await committee.getByRole('button', { name: 'Add member', exact: true }).click();
      assert.equal(await committee.locator('.active-editor-member').count(), 3);
      await committee.getByRole('button', { name: 'Reset committee', exact: true }).click();
      assert.equal(await committee.locator('.active-editor-member').count(), 2);
      await committee.getByLabel('Predict D relative to reference', { exact: true }).selectOption('same');
      await committee.getByLabel('Predict D in nats', { exact: true }).fill('0.494632');
      await committee.getByRole('button', { name: 'Record prediction', exact: true }).click();
      await committee.getByRole('button', { name: 'Reveal decomposition', exact: true }).click();
      await checkResult(committee, 'Prediction matches.');
      results.push({ width, check: 'committee contrast and null grading, invalidation, normalized row and member reset' });

      const batch = lesson.locator('[data-active-lab="batch"]');
      await assertNoResult(batch);
      await batch.getByLabel('A', { exact: true }).check();
      await batch.getByLabel('B', { exact: true }).check();
      await batch.getByLabel("Predict your batch's covering radius", { exact: true }).fill('4');
      await batch.getByRole('button', { name: 'Record prediction', exact: true }).click();
      await batch.getByRole('button', { name: 'Reveal batch comparison', exact: true }).click();
      await checkResult(batch, 'Prediction matches.');
      await batch.getByRole('button', { name: 'Next geometric step', exact: true }).click();
      await batch.getByRole('button', { name: 'Next geometric step', exact: true }).click();
      assert.ok((await batch.innerText()).includes('radius becomes 1.004988'));
      const paintedLabels = await batch.locator('svg text').evaluateAll(labels => labels.map(label => getComputedStyle(label).fill));
      assert.ok(paintedLabels.length >= 3 && paintedLabels.every(fill => fill === 'rgb(203, 214, 206)'), 'Every lab axis label uses the readable intended fill');
      const probabilityWidth = await batch.getByLabel('A probability', { exact: true }).evaluate(input => input.getBoundingClientRect().width);
      assert.ok(probabilityWidth >= 66, 'Probability input has room for its value and spinner');
      if ([1440, 390].includes(width)) await capture(batch, `${width}-batch-complete`);
      await batch.getByLabel('C y', { exact: true }).fill('2');
      await assertNoResult(batch);
      await batch.getByRole('button', { name: 'Coincident null case', exact: true }).click();
      await batch.getByLabel('A', { exact: true }).check();
      await batch.getByLabel('B', { exact: true }).check();
      await batch.getByLabel("Predict your batch's covering radius", { exact: true }).fill('0');
      await batch.getByRole('button', { name: 'Record prediction', exact: true }).click();
      await batch.getByRole('button', { name: 'Reveal batch comparison', exact: true }).click();
      await checkResult(batch, 'Prediction matches.');
      await checkResult(batch, 'equal coverage');
      await batch.getByRole('button', { name: 'Reset batch', exact: true }).click();
      await assertNoResult(batch);
      results.push({ width, check: 'geometric entity edits, chosen-batch grading, full step trace, coincident null and reset' });

      const curves = lesson.locator('[data-active-figure="measured-acquisition-curves"]');
      await curves.getByLabel('Curve view', { exact: true }).selectOption('2');
      await curves.locator('summary').filter({ hasText: 'Replay acquired labels' }).click();
      assert.equal(await curves.getByRole('table', { name: 'Only acquired oracle responses' }).count(), 0);
      await curves.getByRole('button', { name: 'Acquire next recorded label', exact: true }).click();
      assert.equal(await curves.getByRole('table', { name: 'Only acquired oracle responses' }).locator('tbody tr').count(), 1);
      await curves.getByRole('button', { name: 'Reset replay', exact: true }).click();
      assert.equal(await curves.getByRole('table', { name: 'Only acquired oracle responses' }).count(), 0);
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), `Post-interaction overflow ${width}`);
      results.push({ width, check: 'measured trace selection, oracle replay boundary and post-state bounds' });
      if (width === 780) {
        await lesson.evaluate(element => { element.style.fontSize = '20px'; });
        assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
        results.push({ width, check: 'increased text size at intermediate width' });
      }
      await context.close();
    }
    assert.deepEqual(errors, []);
    const paths = ['src/learn/data/topics/active-learning.jsx', 'src/learn/components/lesson-labs/ActiveLearningFigures.jsx', 'src/learn/components/lesson-labs/ActiveLearningLabs.jsx', 'src/learn/components/lesson-labs/active-learning.css', 'src/learn/data/active-learning-models.js'];
    const record = { status: 'passed', checkedAt: new Date().toISOString(), base, browser: 'Microsoft Edge / Playwright', results, errors, screenshots, sourceHashes: Object.fromEntries(paths.map(path => [path, createHash('sha256').update(fs.readFileSync(path)).digest('hex')])) };
    fs.writeFileSync('docs/teaching/evidence/active-learning-browser.json', JSON.stringify(record, null, 2) + '\n');
    console.log(`PASS: ${results.length} browser groups across four widths; ${screenshots.length} informative captures await visual inspection.`);
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

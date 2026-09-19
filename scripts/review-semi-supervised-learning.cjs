const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const root = path.resolve(__dirname, '..');
const base = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4186';
const topic = 'semi-supervised-learning-label-propagation-self-training-co-training';
const destination = path.join(root, 'scratch/semi-supervised-review');
fs.mkdirSync(destination, { recursive: true });
const checks = [];
const captures = [];
const pageErrors = [];
const label = (lab, text) => lab.getByLabel(text, { exact: true });
async function capture(locator, name) {
  await locator.scrollIntoViewIfNeeded();
  const target = path.join(destination, `${name}.png`);
  await locator.screenshot({ path: target, style: '.learn-nav { visibility: hidden !important; }' });
  captures.push({ path: path.relative(root, target).replaceAll('\\', '/'), sha256: crypto.createHash('sha256').update(fs.readFileSync(target)).digest('hex') });
}
async function runGraph(page) {
  const lab = page.locator('[data-ssl-lab="graph"]');
  const commit = lab.getByRole('button', { name: 'Commit prediction and calculate', exact: true });
  assert.equal(await commit.isDisabled(), true);
  await label(lab, 'My final score prediction').selectOption('below');
  await commit.click();
  await lab.getByRole('button', { name: 'One synchronous update', exact: true }).click();
  assert.ok((await lab.innerText()).includes('0.25'));
  await lab.getByRole('button', { name: 'Reveal equilibrium and feedback', exact: true }).click();
  assert.ok((await lab.innerText()).includes('Prediction matched.'));
  assert.ok((await lab.innerText()).includes('0.333333'));
  await lab.getByRole('button', { name: 'B–D shortcut', exact: true }).click();
  assert.equal(await lab.locator('.ssl-result').count(), 0);
  assert.equal(await label(lab, 'My final score prediction').inputValue(), '');
  await label(lab, 'My final score prediction').selectOption('above');
  await commit.click();
  await lab.getByRole('button', { name: 'Reveal equilibrium and feedback', exact: true }).click();
  assert.ok((await lab.innerText()).includes('0.714286'));
  await capture(lab, `graph-shortcut-${page.viewportSize().width}`);
  await lab.getByRole('button', { name: 'Unanchored E–F', exact: true }).click();
  await label(lab, 'My final score prediction').selectOption('unavailable');
  await commit.click();
  await lab.getByRole('button', { name: 'Reveal equilibrium and feedback', exact: true }).click();
  assert.ok((await lab.innerText()).includes('E is unavailable'));
  await lab.getByRole('button', { name: 'Original chain', exact: true }).click();
  await label(lab, 'Algorithm').selectOption('soft');
  await label(lab, 'Alpha (0–0.99)').fill('0');
  await label(lab, 'My final score prediction').selectOption('unavailable');
  await commit.click();
  await lab.getByRole('button', { name: 'Reveal equilibrium and feedback', exact: true }).click();
  assert.ok((await lab.innerText()).includes('B is unavailable'));
  await label(lab, 'Alpha (0–0.99)').fill('0.8');
  assert.equal(await lab.locator('.ssl-result').count(), 0);
  await label(lab, 'My final score prediction').selectOption('below');
  await commit.click();
  await lab.getByRole('button', { name: 'Reveal equilibrium and feedback', exact: true }).click();
  assert.ok((await lab.innerText()).includes('0.37037'));
  await lab.getByRole('button', { name: 'Reset graph', exact: true }).click();
  assert.equal(await commit.isDisabled(), true);
  checks.push({ width: page.viewportSize().width, case: 'Graph committed prediction, synchronous trace, shortcut reversal, island, alpha-zero null, normalization, invalidation and reset' });
}
async function runSelfTraining(page) {
  const lab = page.locator('[data-ssl-lab="self-training"]');
  const commit = lab.getByRole('button', { name: 'Commit prediction and calculate', exact: true });
  assert.equal(await commit.isDisabled(), true);
  await label(lab, 'My boundary movement prediction').selectOption('right');
  await label(lab, 'Does the query class change?').selectOption('no');
  await commit.click();
  await lab.getByRole('button', { name: 'Promote one batch and refit', exact: true }).click();
  assert.ok((await lab.innerText()).includes('accepted 3'));
  await lab.getByRole('button', { name: 'Reveal final model and feedback', exact: true }).click();
  assert.ok((await lab.innerText()).includes('Both predictions matched.'));
  assert.ok((await lab.innerText()).includes('Final boundary: 0.5'));
  await lab.getByRole('button', { name: 'Move 3 to 9', exact: true }).click();
  assert.equal(await lab.locator('.ssl-result').count(), 0);
  await label(lab, 'My boundary movement prediction').selectOption('right');
  await label(lab, 'Does the query class change?').selectOption('yes');
  await commit.click();
  await lab.getByRole('button', { name: 'Reveal final model and feedback', exact: true }).click();
  assert.ok((await lab.innerText()).includes('Final boundary: 1.5'));
  await capture(lab, `self-training-contrast-${page.viewportSize().width}`);
  await lab.getByRole('button', { name: 'No-acceptance pool', exact: true }).click();
  await label(lab, 'My boundary movement prediction').selectOption('unchanged');
  await label(lab, 'Does the query class change?').selectOption('no');
  await commit.click();
  await lab.getByRole('button', { name: 'Reveal final model and feedback', exact: true }).click();
  assert.ok((await lab.innerText()).includes('No candidate passed'));
  await label(lab, 'Observed points · one x,class row per line').fill('-2,0\n2,0');
  assert.equal(await lab.locator('.ssl-result').count(), 0);
  assert.ok((await lab.getByRole('alert').innerText()).includes('each class'));
  await lab.getByRole('button', { name: 'Reset self-training', exact: true }).click();
  assert.equal(await commit.isDisabled(), true);
  checks.push({ width: page.viewportSize().width, case: 'Prototype feedback default/edited/no-acceptance cases, final refit, separate predictions, invalid input and reset' });
}
async function runCoTraining(page) {
  const lab = page.locator('[data-ssl-lab="co-training"]');
  const commit = lab.getByRole('button', { name: 'Commit prediction and calculate', exact: true });
  assert.equal(await commit.isDisabled(), true);
  await label(lab, 'My final rule prediction').selectOption('0');
  await commit.click();
  for (let i = 0; i < 3; i += 1) await lab.getByRole('button', { name: 'Next transfer stage', exact: true }).click();
  assert.ok((await lab.innerText()).includes('triangle'));
  assert.ok((await lab.innerText()).includes('Conflicting rows deferred: 6'));
  await capture(lab, `co-training-transfer-${page.viewportSize().width}`);
  await lab.getByRole('button', { name: 'Reveal final rules and feedback', exact: true }).click();
  assert.ok((await lab.innerText()).includes('Prediction matched.'));
  await lab.getByRole('button', { name: 'Change bridge to blue', exact: true }).click();
  assert.equal(await lab.locator('.ssl-result').count(), 0);
  await label(lab, 'My final rule prediction').selectOption('1');
  await commit.click();
  await lab.getByRole('button', { name: 'Reveal final rules and feedback', exact: true }).click();
  assert.ok((await lab.innerText()).includes('green → 1'));
  await lab.getByRole('button', { name: 'Duplicate view 1', exact: true }).click();
  await label(lab, 'My final rule prediction').selectOption('unknown');
  await commit.click();
  await lab.getByRole('button', { name: 'Reveal final rules and feedback', exact: true }).click();
  assert.ok((await lab.innerText()).includes('Row 3, view 1: unknown'));
  await lab.getByRole('button', { name: 'Remove all anchors', exact: true }).click();
  await label(lab, 'My final rule prediction').selectOption('unknown');
  await commit.click();
  await lab.getByRole('button', { name: 'Reveal final rules and feedback', exact: true }).click();
  assert.ok((await lab.innerText()).includes('No offers: the loop stops.'));
  await lab.getByRole('button', { name: 'Reset co-training', exact: true }).click();
  assert.equal(await commit.isDisabled(), true);
  checks.push({ width: page.viewportSize().width, case: 'Co-training staged recipient transfers, conflicts, changed bridge, duplicate-view/no-anchor nulls, independent arrays and reset' });
}
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    for (const width of [1366, 390, 320]) {
      const context = await browser.newContext({ viewport: { width, height: 900 }, reducedMotion: 'reduce' });
      const page = await context.newPage();
      page.on('pageerror', error => pageErrors.push(error.message));
      await page.goto(`${base}/learn/path/full-curriculum/${topic}?module=classical-ml`, { waitUntil: 'domcontentloaded', timeout: 60000 });
      await page.locator('[data-ssl-lab="graph"]').waitFor({ timeout: 60000 });
      assert.equal(await page.locator('.ssl-lesson h2').count(), 10);
      assert.equal(await page.locator('.ssl-lesson .katex-error').count(), 0);
      assert.equal(await page.locator('[data-ssl-figure]').count(), 7);
      assert.equal(await page.locator('[data-ssl-lab]').count(), 3);
      const body = await page.locator('.ssl-lesson').innerText();
      assert.ok(body.includes('selected supervised_lr test 72 / 80'));
      assert.ok((await page.locator('.ssl-lesson').textContent()).includes('duplicate-view unresolved [3, 4, 5]'));
      assert.ok(body.includes('H. A small independent investigation'));
      for (const detail of await page.locator('.ssl-lesson details').all()) assert.equal(await detail.getAttribute('open'), null);
      const programToggle = page.getByText('Inspect the complete categorical co-training program and its recorded output', { exact: true });
      await programToggle.click();
      assert.ok((await page.locator('.ssl-lesson').innerText()).includes('duplicate-view unresolved [3, 4, 5]'));
      await programToggle.click();
      if (width === 1366) {
        for (const figure of await page.locator('[data-ssl-figure]').all()) await capture(figure, `figure-${await figure.getAttribute('data-ssl-figure')}`);
      }
      await runGraph(page);
      await runSelfTraining(page);
      await runCoTraining(page);
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1);
      assert.equal(overflow, false, `Page overflow at ${width}`);
      const graph = page.locator('[data-ssl-lab="graph"]');
      const prediction = label(graph, 'My final score prediction');
      await prediction.focus();
      await page.keyboard.press('ArrowDown');
      await page.keyboard.press('Enter');
      assert.equal(await prediction.inputValue(), 'below');
      await page.keyboard.press('Escape');
      await page.keyboard.press('Tab');
      assert.equal(await page.evaluate(() => document.activeElement.textContent), 'Commit prediction and calculate');
      await page.keyboard.press('Enter');
      assert.equal(await graph.locator('.ssl-result').count(), 1);
      checks.push({ width, case: 'Complete manuscript/code output, seven figures, closed solutions, no KaTeX error, no document overflow, reduced-motion and keyboard commit' });
      if (width === 1366) {
        const links = await page.locator('.ssl-lesson a[href^="/learn-assets/"]').evaluateAll(nodes => [...new Set(nodes.map(node => node.getAttribute('href')))]);
        for (const link of links) assert.equal((await page.request.get(`${base}${link}`)).status(), 200);
        checks.push({ width, case: 'All topic-owned dataset, provenance and executable download links resolve' });
      }
      await context.close();
    }
    assert.deepEqual(pageErrors, []);
    const files = [
      `src/learn/data/topics/${topic}.jsx`, 'src/learn/data/semi-supervised-models.js', 'src/learn/data/semi-supervised-examples.js',
      'src/learn/components/lesson-labs/SemiSupervisedFigures.jsx', 'src/learn/components/lesson-labs/SemiSupervisedLabs.jsx',
      'src/learn/components/lesson-labs/semi-supervised.css', 'scripts/review-semi-supervised-learning.cjs',
    ];
    fs.writeFileSync(path.join(root, 'docs/teaching/evidence/semi-supervised-browser.json'), JSON.stringify({ status: 'passed', base, checks, pageErrors, captures, sourceHashes: Object.fromEntries(files.map(file => [file, crypto.createHash('sha256').update(fs.readFileSync(path.join(root, file))).digest('hex')])), visualInspection: 'Screenshots require separate author inspection; automated checks do not establish visual quality.' }, null, 2) + '\n');
    console.log(`Passed ${checks.length} browser groups at 1366, 390 and 320 px; ${captures.length} informative captures.`);
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

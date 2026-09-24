// Complementary review: full BIO supports, conditional sampling, relative ties,
// and fresh-versus-revealed prediction state. Does not refit the author's model.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const crypto = require('node:crypto');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const sha256 = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const close = (a, b) => assert.ok(Math.abs(a - b) < 1e-11, `${a} versus ${b}`);
const paths = (length, labels) => length ? paths(length - 1, labels).flatMap(prefix => Array.from({ length: labels }, (_, i) => [...prefix, i])) : [[]];
(async () => {
  const { BIO_LABELS, legalBioPath, logPartition, chainDistribution } = await import('../src/learn/data/crf-models.js');
  const checks = [], screenshots = [], errors = [];
  let supportCases = 0;
  // Independently phrase the BIO condition over a complete path, then compare
  // exhaustive supports with the implementation's edge-local legality engine.
  const admissible = sequence => sequence.every((label, i) => label[0] !== 'I' || (i > 0 && sequence[i - 1] !== 'O' && sequence[i - 1].slice(2) === label.slice(2)));
  const transition = Array.from({ length: 5 }, () => Array(5).fill(0));
  // A previous I-state is permitted when it already has a compatible prefix.
  for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) transition[i][j] = BIO_LABELS[j][0] !== 'I' || (BIO_LABELS[i] !== 'O' && BIO_LABELS[i].slice(2) === BIO_LABELS[j].slice(2)) ? 0 : -Infinity;
  const counts = [];
  for (let length = 1; length <= 5; length++) {
    let count = 0;
    for (const indices of paths(length, 5)) {
      const sequence = indices.map(index => BIO_LABELS[index]);
      const expected = admissible(sequence);
      assert.equal(legalBioPath(sequence), expected); count += Number(expected); supportCases++;
    }
    close(logPartition(Array.from({ length }, () => Array(5).fill(0)), transition, BIO_LABELS.map(label => label[0] !== 'I')), Math.log(count));
    counts.push(count);
  }
  checks.push({ check: 'independent complete-path BIO support and uniform constrained partition', supportCases, counts });
  // Check the manuscript's backward-sampling factorization against direct
  // joint products for every path of an asymmetric 3-state, 4-position chain.
  const unary = [[2, 3, 5], [7, 2, 4], [3, 6, 1], [4, 1, 8]], pair = [[2, 1, 4], [3, 5, 2], [1, 3, 2]];
  const prefix = [unary[0]];
  for (let t = 1; t < unary.length; t++) prefix.push(unary[t].map((value, j) => value * prefix[t - 1].reduce((sum, mass, i) => sum + mass * pair[i][j], 0)));
  const enumerated = paths(4, 3).map(path => ({ path, mass: path.reduce((mass, label, t) => mass * unary[t][label] * (t ? pair[path[t - 1]][label] : 1), 1) }));
  const total = enumerated.reduce((sum, row) => sum + row.mass, 0);
  for (const { path, mass } of enumerated) {
    let reconstructed = prefix[3][path[3]] / total;
    for (let t = 2; t >= 0; t--) reconstructed *= prefix[t][path[t]] * pair[path[t]][path[t + 1]] / prefix[t].reduce((sum, value, i) => sum + value * pair[i][path[t + 1]], 0);
    close(reconstructed, mass / total);
  }
  checks.push({ check: 'backward conditional-sampling product equals the complete normalized joint', paths: enumerated.length });
  const nearTie = [16, 16, 16, 16, 16, 16, 15.999999999999, 15.999999999999];
  assert.deepEqual(chainDistribution(nearTie).parents, ['A, B', 'A, B']);
  assert.deepEqual(chainDistribution(nearTie).winners, ['AA', 'AB', 'BA', 'BB']);
  checks.push({ check: 'large-factor near-tie uses identical relative policy for winners and predecessors', factors: nearTie });
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const directory = 'scratch/crf-independent-review'; fs.mkdirSync(directory, { recursive: true });
  try {
    const page = await browser.newPage({ viewport: { width: 390, height: 1000 }, reducedMotion: 'reduce' });
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(`${process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4184'}/learn/path/full-curriculum/conditional-random-fields-crf?module=classical-ml`);
    await page.locator('.crf-lesson').waitFor(); await page.evaluate(() => document.fonts.ready);
    for (const type of ['trellis', 'bias']) {
      const lab = page.locator(`[data-crf-lab="${type}"]`), select = lab.getByRole('combobox');
      assert.equal(await lab.locator('.crf-result').count(), 0);
      assert.ok(!(await lab.innerText()).includes(type === 'trellis' ? '0.800000' : '0.009901'));
      await select.selectOption(type === 'trellis' ? 'same' : 'decrease');
      assert.ok(await lab.getByRole('button', { name: 'Commit prediction', exact: true }).isDisabled());
      // Type equivalent numeric values: a formatting change is not new evidence.
      const inputs = lab.getByRole('spinbutton'); await inputs.first().fill(type === 'trellis' ? '3.00' : '0.500');
      await select.selectOption(type === 'trellis' ? 'same' : 'decrease');
      assert.ok(await lab.getByRole('button', { name: 'Commit prediction', exact: true }).isDisabled());
      const tuple = type === 'trellis' ? nearTie.map(String) : ['0.25', '3', '1'];
      for (let i = 0; i < tuple.length; i++) await inputs.nth(i).fill(tuple[i]);
      await select.selectOption(type === 'trellis' ? 'same' : 'increase');
      assert.equal(await lab.locator('.crf-result').count(), 0);
      assert.ok(!(await lab.innerText()).includes(type === 'trellis' ? '16384' : 'Global route A0.5'));
      await lab.getByRole('button', { name: 'Commit prediction', exact: true }).click();
      await lab.getByRole('button', { name: type === 'trellis' ? 'Calculate' : 'Apply', exact: true }).click();
      assert.match(await lab.getByRole('status').innerText(), /Prediction matched/);
      if (type === 'trellis') assert.equal((await lab.locator('tbody tr').allTextContents()).filter(text => text.endsWith('A, B')).length, 2);
      const file = `${directory}/${type}-changed-390.png`;
      await lab.locator('.crf-result').screenshot({ path: file, style: '.learn-nav { visibility: hidden !important; }' }); screenshots.push({ file, sha256: sha256(file) });
      await inputs.first().fill(type === 'trellis' ? '15' : '.3');
      assert.equal(await lab.locator('.crf-result').count(), 0);
      await inputs.first().fill(tuple[0]);
      await select.selectOption(type === 'trellis' ? 'same' : 'increase');
      assert.ok(await lab.getByRole('button', { name: 'Commit prediction', exact: true }).isDisabled());
      const explore = lab.getByRole('button', { name: 'Explore without a prediction', exact: true });
      await explore.focus(); await page.keyboard.press('Enter');
      assert.match(await lab.getByRole('status').innerText(), /No prediction (was )?graded/);
      await lab.getByRole('button', { name: 'Reset', exact: true }).click();
      assert.equal(await lab.locator('.crf-result').count(), 0);
      assert.equal(await select.inputValue(), '');
      checks.push({ check: `${type}: hidden numeric result; reference/formatting/revealed-return cannot grade; fresh changed prediction; keyboard exploration; reset` });
    }
    for (const [name, label] of [['factor-chain', 'Observed words and the output factor chain'], ['neural-flow', 'Neural encoder and CRF training versus decoding']]) {
      const file = `${directory}/${name}-390.png`;
      await page.getByRole('figure', { name: label, exact: true }).screenshot({ path: file, style: '.learn-nav { visibility: hidden !important; }' }); screenshots.push({ file, sha256: sha256(file) });
    }
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
    assert.deepEqual(errors, []);
    checks.push({ check: '390px informative states: no page overflow or runtime errors; four captures for actual review' });
  } finally { await browser.close(); }
  const sourcePaths = [...Object.keys(JSON.parse(fs.readFileSync('docs/teaching/evidence/crf-author-review.json')).reviewedFiles), 'scripts/review-crf-independent.cjs'];
  fs.writeFileSync('docs/teaching/evidence/crf-independent.json', JSON.stringify({ status: 'passed', checks, screenshots, source: Object.fromEntries(sourcePaths.map(file => [file, sha256(file)])), visualInspection: 'Capture inspection recorded separately in the independent review.' }, null, 2) + '\n');
  console.log(JSON.stringify({ status: 'passed', groups: checks.length, supportCases, samplingPaths: enumerated.length, captures: screenshots.length }));
})().catch(error => { console.error(error); process.exitCode = 1; });

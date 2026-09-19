// Complementary reviewer cases; intentionally separate from the author's wider campaign.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const crypto = require('node:crypto');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const base = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4184';
const captureDirectory = 'scratch/evaluation-metrics-independent';
const sourceFiles = ['src/learn/data/topics/evaluation-metrics-precision-recall-f1-auc-roc-ap-r-mae.jsx', 'src/learn/components/lesson-labs/EvaluationMetricsFigures.jsx', 'src/learn/components/lesson-labs/EvaluationMetricsLabs.jsx', 'src/learn/components/lesson-labs/evaluation-metrics.css', 'src/learn/data/evaluation-metrics-models.js', 'src/learn/data/evaluation-metrics-data.json', 'scripts/review-evaluation-metrics-independent.cjs'];
const digest = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const record = { reviewer: 'semi_supervised_implementation', passed: false, base, widths: [390], checks: [], source: Object.fromEntries(sourceFiles.map(file => [file, digest(file)])), screenshots: [], errors: [] };
const check = (name, condition) => { assert.ok(condition, name); record.checks.push(name); };
async function fill(lab, values) {
  for (const [name, value] of Object.entries(values)) {
    const field = lab.getByLabel(name, { exact: true });
    if (await field.evaluate(node => node.tagName === 'SELECT')) await field.selectOption(value);
    else await field.fill(value);
  }
  await lab.getByLabel('Explain your prediction', { exact: true }).fill('Track the exact contributions, preserve the denominator, and retain the same comparison under equivalent inputs.');
}
async function commit(lab, values) {
  await fill(lab, values);
  await lab.getByRole('button', { name: 'Commit prediction', exact: true }).click();
  await lab.getByRole('button', { name: 'Reveal comparison', exact: true }).click();
  check('Changed-input prediction accepted with mechanism feedback', (await lab.locator('.metric-response .metric-status').innerText()).includes('agrees with your predictions'));
}
async function capture(locator, name) {
  const file = `${captureDirectory}/${name}.png`;
  await locator.screenshot({ path: file, animations: 'disabled', style: '.learn-nav{visibility:hidden!important}' });
  record.screenshots.push({ file, sha256: digest(file), bytes: fs.statSync(file).size });
}
(async () => {
  fs.mkdirSync(captureDirectory, { recursive: true });
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 390, height: 900 }, reducedMotion: 'reduce' });
    page.on('pageerror', error => record.errors.push(error.message));
    await page.goto(`${base}/learn/path/full-curriculum/evaluation-metrics-precision-recall-f1-auc-roc-ap-r-mae?module=classical-ml`, { waitUntil: 'domcontentloaded', timeout: 60000 });
    await page.locator('.evaluation-metrics-lesson').waitFor();
    await page.evaluate(() => document.fonts.ready);
    if (process.argv.includes('--measured-followup')) {
      const previous = JSON.parse(fs.readFileSync('docs/teaching/evidence/evaluation-metrics-independent-browser.json', 'utf8'));
      assert.ok(previous.passed);
      const affected = ['src/learn/components/lesson-labs/EvaluationMetricsFigures.jsx', 'src/learn/components/lesson-labs/evaluation-metrics.css', 'scripts/review-evaluation-metrics-independent.cjs'];
      for (const file of sourceFiles.filter(file => !affected.includes(file))) assert.equal(record.source[file], previous.source[file], `Unexpected behavioral change: ${file}`);
      const selected = JSON.parse(fs.readFileSync('src/learn/data/evaluation-metrics-data.json', 'utf8')).selected_development_record;
      for (const width of [1440, 390, 320]) {
        await page.setViewportSize({ width, height: 900 });
        const figure = page.locator('[data-metric-figure="measured"]');
        const geometry = await figure.locator('svg').evaluate(svg => ({
          width: svg.getBoundingClientRect().width,
          viewWidth: svg.viewBox.baseVal.width,
          labels: [...svg.querySelectorAll('text')].map(label => ({ size: parseFloat(getComputedStyle(label).fontSize), bounds: label.getBoundingClientRect().toJSON() })),
          bounds: svg.getBoundingClientRect().toJSON(),
          selectedX: Number(svg.querySelector('circle').getAttribute('cx')),
        }));
        check(`${width}: measured cost ticks are readable at painted scale`, geometry.labels.length === 5 && geometry.labels.every(label => label.size * geometry.width / geometry.viewWidth >= 11.5));
        check(`${width}: measured tick labels remain inside SVG`, geometry.labels.every(label => label.bounds.left >= geometry.bounds.left - 1 && label.bounds.right <= geometry.bounds.right + 1 && label.bounds.top >= geometry.bounds.top - 1 && label.bounds.bottom <= geometry.bounds.bottom + 1));
        check(`${width}: selected threshold uses the new common x mapping`, Math.abs(geometry.selectedX - (42 + 250 * selected.threshold)) < 1e-10);
        check(`${width}: finite curve explicitly excludes no-alert cost195`, (await figure.innerText()).includes('separate no-alert candidate costs 195'));
        check(`${width}: updated measured figure retains page containment`, !await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1));
        if (width === 390) await capture(figure, 'measured-followup-390');
      }
      check('No runtime errors in measured-figure follow-up', record.errors.length === 0);
      for (const file of sourceFiles) assert.equal(digest(file), record.source[file], `Source changed during targeted follow-up: ${file}`);
      previous.followupChecks = [...(previous.followupChecks || []), { reason: 'Only measured-cost SVG coordinates/readable tick size/caption and related CSS changed; all behavior sources unchanged.', previousSource: previous.source, checks: record.checks, errors: record.errors }];
      previous.source = record.source;
      previous.screenshots.push(...record.screenshots);
      fs.writeFileSync('docs/teaching/evidence/evaluation-metrics-independent-browser.json', `${JSON.stringify(previous, null, 2)}\n`);
      console.log(JSON.stringify({ passed: true, reusedBehaviorGroups: previous.checks.length, affectedFigureChecks: record.checks.length, addedCaptures: record.screenshots.length }));
      return;
    }
    check('All four investigations initially hide response content', await page.locator('[data-metric-lab] .metric-response').count() === 0);
    const ruler = page.locator('[data-metric-figure="threshold"] .metric-score-ruler svg');
    const tiedLocations = await ruler.locator('g').evaluateAll(groups => groups.filter(group => /^[BC]:/.test(group.querySelector('title').textContent)).map(group => {
      const circle = group.querySelector('circle'), rectangle = group.querySelector('rect');
      return circle ? Number(circle.getAttribute('cx')) : Number(rectangle.getAttribute('x')) + 5;
    }));
    check('Tied B and C occupy identical quantitative score positions', tiedLocations.length === 2 && tiedLocations[0] === tiedLocations[1]);
    const gateLocation = Number(await ruler.locator('.metric-gate-line').getAttribute('x1'));
    check('Threshold ruler positions and dashed gate use the declared common scale', Math.abs(gateLocation - (12 + 376 * (0.8 - 0.1) / (0.95 - 0.1))) < 1e-10 && gateLocation === tiedLocations[0]);

    const ranking = page.locator('[data-metric-lab="ranking"]');
    await ranking.locator('details').first().locator('summary').click();
    await ranking.getByLabel('C score', { exact: true }).fill('0.45');
    const rankingAnswers = { 'Additional true positives': '0', 'Additional false positives': '1', 'New FPR coordinate': '0.25', 'New recall / TPR coordinate': '0.25' };
    await commit(ranking, rankingAnswers);
    await ranking.getByRole('button', { name: 'Explore complete ranking (ends graded steps for these inputs)', exact: true }).click();
    await ranking.getByRole('button', { name: 'Use this point; predict the next block', exact: true }).click();
    await fill(ranking, rankingAnswers);
    check('Full reveal prevents grading later exposed steps', await ranking.getByRole('button', { name: 'Commit prediction', exact: true }).isDisabled());
    await ranking.getByLabel('C score', { exact: true }).fill('0.44');
    await ranking.getByLabel('C score', { exact: true }).fill('0.45');
    await fill(ranking, rankingAnswers);
    check('Edit and restore retains knowledge that the full ranking was exposed', await ranking.getByRole('button', { name: 'Commit prediction', exact: true }).isDisabled());
    check('Restoring inputs does not resurrect an old response', await ranking.locator('.metric-response').count() === 0);
    await ranking.getByRole('button', { name: 'Swap B/C display order', exact: true }).click();
    await fill(ranking, rankingAnswers);
    check('Display-only row permutation cannot restore grading after full reveal', await ranking.getByRole('button', { name: 'Commit prediction', exact: true }).isDisabled());
    await capture(ranking, 'ranking-exposure-memory-390');

    const gate = page.locator('[data-metric-lab="threshold"]');
    await gate.getByLabel('Current threshold', { exact: true }).fill('0.73');
    await gate.getByLabel('Proposed threshold', { exact: true }).fill('0.74');
    await commit(gate, { 'Precision will…': 'same', 'Recall will…': 'same', 'IDs that change decisions': 'none' });
    check('Changed threshold inside an empty interval moves no items', (await gate.locator('.metric-response').innerText()).includes('Moved: none'));

    const residual = page.locator('[data-metric-lab="regression"]');
    await residual.getByRole('button', { name: 'Copy targets to both predictions', exact: true }).click();
    await residual.getByLabel('T1 A', { exact: true }).fill('1.00000000001');
    const tieAnswers = { 'MAE will prefer…': 'same', 'RMSE will prefer…': 'same' };
    await commit(residual, tieAnswers);
    await residual.getByLabel('Display unit', { exact: true }).selectOption('60');
    await commit(residual, tieAnswers);
    check('Numerical tie policy remains invariant under minutes-to-seconds conversion', (await residual.locator('.metric-response .metric-status').innerText()).includes('base minutes'));
    await residual.getByRole('button', { name: 'All targets and predictions = 4', exact: true }).click();
    await commit(residual, tieAnswers);
    check('Perfect constant predictions retain zero error and undefined R2 explanation', (await residual.locator('.metric-response').innerText()).includes('R² undefined: the evaluation target has zero variance'));
    await capture(residual.locator('.metric-response'), 'constant-target-result-390');

    const retrieval = page.locator('[data-metric-lab="retrieval"]');
    for (let i = 0; i < 4; i++) await retrieval.getByRole('button', { name: 'Move D5 up', exact: true }).click();
    for (let i = 0; i < 3; i++) await retrieval.getByRole('button', { name: 'Move D1 down', exact: true }).click();
    await commit(retrieval, { 'Precision@K will…': 'same', 'Reciprocal rank will…': 'same', 'Full-list AP will…': 'same', 'NDCG@K will…': 'same' });
    check('Swapping equally judged documents preserves all four reported metrics', (await retrieval.locator('.metric-response').innerText()).includes('0.523434 → 0.523434'));
    await retrieval.getByRole('button', { name: 'Set all proposed grades to zero', exact: true }).click();
    await commit(retrieval, { 'Precision@K will…': 'down', 'Reciprocal rank will…': 'down', 'Full-list AP will…': 'undefined', 'NDCG@K will…': 'undefined' });
    check('No-relevance state distinguishes zero decision measures from undefined normalized scores', (await retrieval.locator('.metric-response').innerText()).includes('No candidate has positive gain; the raw ratio is undefined'));
    await capture(retrieval.locator('.metric-response'), 'no-relevance-result-390');

    const curve = page.locator('[data-metric-figure="curves"] .metric-curve svg').first();
    const coordinates = await curve.locator('circle').evaluateAll(nodes => nodes.map(node => [Number(node.getAttribute('cx')), Number(node.getAttribute('cy'))]));
    check('ROC selected tie point uses shared quantitative axes', coordinates.some(([x, y]) => x === 104.5 && y === 115));
    check('ROC initial and final endpoints match zero and one axes', coordinates[0][0] === 42 && coordinates[0][1] === 210 && coordinates.at(-1)[0] === 292 && coordinates.at(-1)[1] === 20);
    check('No page overflow in complementary changed-input states', !await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1));
    check('No runtime errors in complementary browser scenarios', record.errors.length === 0);
    for (const file of sourceFiles) assert.equal(digest(file), record.source[file], `Source changed during independent browser check: ${file}`);
    record.passed = true;
    fs.writeFileSync('docs/teaching/evidence/evaluation-metrics-independent-browser.json', `${JSON.stringify(record, null, 2)}\n`);
    console.log(JSON.stringify({ passed: true, checks: record.checks.length, captures: record.screenshots.length }));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

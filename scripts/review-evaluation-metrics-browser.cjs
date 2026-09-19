const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const crypto = require('node:crypto');

(async () => {
  const base = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4184';
  const directory = 'scratch/evaluation-metrics-implementation';
  const checks = [], screenshots = [], errors = [];
  fs.mkdirSync(directory, { recursive: true });
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  async function capture(locator, name, width) {
    if (width === 320) return;
    const path = `${directory}/${name}-${width}.png`;
    await locator.screenshot({ path, style: '.learn-nav { visibility: hidden !important; }' });
    screenshots.push(path);
  }
  async function commit(lab, answers) {
    for (const [label, value] of Object.entries(answers)) {
      const field = lab.getByLabel(label, { exact: true });
      if (await field.evaluate(el => el.tagName === 'SELECT')) await field.selectOption(value); else await field.fill(value);
    }
    await lab.getByLabel('Explain your prediction', { exact: true }).fill('The changed entities alter the stated counts or contributions.');
    await lab.getByRole('button', { name: 'Commit prediction', exact: true }).click();
    assert.equal(await lab.locator('.metric-response').count(), 0);
    await lab.getByRole('button', { name: 'Reveal comparison', exact: true }).click();
    assert.match(await lab.locator('.metric-response .metric-status').innerText(), /agrees with your predictions/);
  }
  try {
    for (const width of [1440, 390, 320]) {
      const context = await browser.newContext({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      const page = await context.newPage();
      page.on('pageerror', error => errors.push(error.message));
      await page.goto(`${base}/learn/path/full-curriculum/evaluation-metrics-precision-recall-f1-auc-roc-ap-r-mae?module=classical-ml`, { waitUntil: 'domcontentloaded', timeout: 60000 });
      const lesson = page.locator('.evaluation-metrics-lesson');
      await lesson.waitFor({ timeout: 60000 });
      await page.evaluate(() => Promise.race([document.fonts.ready, new Promise(resolve => setTimeout(resolve, 5000))]));
      console.log('Loaded metrics at', width);
      assert.equal(await lesson.locator('.katex-error').count(), 0);
      assert.equal(await lesson.locator('h2').count(), 11);
      assert.equal(await lesson.locator('[data-metric-figure]').count(), 9);
      assert.equal(await lesson.locator('[data-metric-lab]').count(), 4);
      for (const anchor of await lesson.locator('.lesson-intro nav a').evaluateAll(nodes => nodes.map(node => node.getAttribute('href')))) assert.equal(await lesson.locator(`[id="${anchor.slice(1)}"]`).count(), 1, anchor);
      checks.push({ width, check: 'complete content, nine figures/four investigations, math and route anchors' });

      const gate = lesson.locator('[data-metric-lab="threshold"]');
      assert.equal(await gate.locator('.metric-response').count(), 0);
      assert.equal(await gate.getByRole('button', { name: 'Reveal comparison', exact: true }).isDisabled(), true);
      await gate.getByLabel('Proposed threshold', { exact: true }).fill('.3');
      await commit(gate, { 'Precision will…': 'same', 'Recall will…': 'up', 'IDs that change decisions': 'D E F' });
      assert.match(await gate.locator('.metric-response').innerText(), /TP = 4/);
      await capture(gate.locator('.metric-response'), 'gate-result', width);
      await gate.getByLabel('Proposed threshold', { exact: true }).fill('.4');
      assert.equal(await gate.locator('.metric-response').count(), 0);
      await gate.getByLabel('Proposed threshold', { exact: true }).fill('.3');
      assert.equal(await gate.locator('.metric-response').count(), 0);
      assert.equal(await gate.getByRole('button', { name: 'Commit prediction', exact: true }).isDisabled(), true);
      await gate.getByLabel('Proposed rule: above the maximum score (no alerts)').check();
      await gate.getByRole('button', { name: 'Explore without grading', exact: true }).click();
      assert.match(await gate.locator('.metric-response').innerText(), /Precision undefined: there are no predicted positives/);
      await gate.getByLabel('Current threshold', { exact: true }).fill('');
      assert.equal(await gate.getByRole('button', { name: 'Explore without grading', exact: true }).isDisabled(), true);
      await gate.getByRole('button', { name: 'Reset gate investigation' }).click();
      checks.push({ width, check: 'threshold membership/denominators, changed commit, no-alert undefined, edit-return retirement, invalid/reset' });

      const ranking = lesson.locator('[data-metric-lab="ranking"]');
      await ranking.getByText('Edit the actual items (8/16)', { exact: true }).click();
      await ranking.getByLabel('C score', { exact: true }).fill('.45');
      await commit(ranking, { 'Additional true positives': '0', 'Additional false positives': '1', 'New FPR coordinate': '.25', 'New recall / TPR coordinate': '.25' });
      await capture(ranking.locator('.metric-response'), 'ranking-step', width);
      await ranking.getByRole('button', { name: 'Explore complete ranking (ends graded steps for these inputs)' }).click();
      assert.match(await ranking.locator('.metric-response').innerText(), /AUC 0.6875/);
      await ranking.getByRole('button', { name: 'Use this point; predict the next block' }).click();
      assert.equal(await ranking.getByRole('button', { name: 'Commit prediction', exact: true }).isDisabled(), true);
      await ranking.getByLabel('C score', { exact: true }).fill('.46');
      await ranking.getByLabel('C score', { exact: true }).fill('.45');
      await ranking.getByRole('button', { name: 'Swap B/C display order' }).click();
      assert.match(await ranking.getByRole('status').innerText(), /worked or previously revealed/);
      await ranking.getByRole('button', { name: 'No observed positives' }).click();
      await ranking.getByRole('button', { name: 'Explore without grading', exact: true }).click();
      assert.match(await ranking.locator('.metric-response').innerText(), /ROC operating points undefined/);
      checks.push({ width, check: 'edited-score grouped step, full ranking, exposed-answer history across steps/round-trip/permutation and no-positive domain' });

      const residual = lesson.locator('[data-metric-lab="regression"]');
      await residual.getByLabel('T5 A', { exact: true }).fill('8');
      await commit(residual, { 'MAE will prefer…': 'a', 'RMSE will prefer…': 'a' });
      assert.match(await residual.locator('.metric-response').innerText(), /MAE: A 0.4, B 2/);
      await capture(residual.locator('.metric-response'), 'residual-result', width);
      await residual.getByLabel('Display unit', { exact: true }).selectOption('60');
      assert.equal(await residual.locator('.metric-response').count(), 0);
      await residual.getByRole('button', { name: 'Explore without grading', exact: true }).click();
      assert.match(await residual.locator('.metric-response').innerText(), /MAE: A 24, B 120/);
      await residual.getByRole('button', { name: 'All targets and predictions = 4' }).click();
      await residual.getByRole('button', { name: 'Explore without grading', exact: true }).click();
      assert.match(await residual.locator('.metric-response').innerText(), /R² undefined: the evaluation target has zero variance/);
      checks.push({ width, check: 'edited residual preference, proportional squares, unit invalidation/conversion and zero-variance R²' });

      const retrieval = lesson.locator('[data-metric-lab="retrieval"]');
      await retrieval.getByRole('button', { name: 'Move D2 up', exact: true }).focus();
      await page.keyboard.press('Enter');
      await page.waitForTimeout(40);
      assert.equal(await page.evaluate(() => document.activeElement?.getAttribute('aria-label')), 'Move D2 down');
      await commit(retrieval, { 'Precision@K will…': 'same', 'Reciprocal rank will…': 'up', 'Full-list AP will…': 'up', 'NDCG@K will…': 'up' });
      assert.match(await retrieval.locator('.metric-response').innerText(), /0.798485/);
      await capture(retrieval.locator('.metric-response'), 'retrieval-result', width);
      await retrieval.getByLabel('Gain convention', { exact: true }).selectOption('linear');
      assert.equal(await retrieval.locator('.metric-response').count(), 0);
      await retrieval.getByRole('button', { name: 'Set all proposed grades to zero' }).click();
      await retrieval.getByRole('button', { name: 'Explore without grading', exact: true }).click();
      assert.match(await retrieval.locator('.metric-response').innerText(), /No candidate has positive gain/);
      await retrieval.getByLabel('D1 relevance grade', { exact: true }).fill('1.5');
      assert.equal(await retrieval.getByRole('button', { name: 'Explore without grading', exact: true }).isDisabled(), true);
      await retrieval.getByRole('button', { name: 'Reset retrieval investigation' }).click();
      checks.push({ width, check: 'keyboard stable-document movement/focus, committed changed order, gain invalidation, empty relevance and invalid grade/reset' });

      for (const id of ['contracts', 'threshold', 'prevalence', 'curves', 'pairs', 'probabilities', 'residuals', 'retrieval', 'measured']) await capture(lesson.locator(`[data-metric-figure="${id}"]`), id, width);
      const overflow = await page.evaluate(() => ({ document: document.documentElement.scrollWidth, viewport: innerWidth, lesson: document.querySelector('.evaluation-metrics-lesson').getBoundingClientRect().right }));
      assert.ok(overflow.document <= width + 1 && overflow.lesson <= width + 1, JSON.stringify(overflow));
      const badGeometry = await lesson.locator('.metric-error-square').evaluateAll(nodes => nodes.some(node => Math.abs(node.getBoundingClientRect().width - node.getBoundingClientRect().height) > .01));
      assert.equal(badGeometry, false);
      checks.push({ width, check: 'viewport containment, actual square geometry, informative-state captures and no runtime/math errors' });
      await context.close();
    }
    for (const file of ['metrics-calculations.py', 'banknote-evaluation.py', 'banknote-subset.csv', 'data-provenance.md']) {
      const response = await fetch(`${base}/learn-assets/evaluation-metrics/${file}`);
      assert.equal(response.status, 200);
      assert.deepEqual(Buffer.from(await response.arrayBuffer()), fs.readFileSync(`public/learn-assets/evaluation-metrics/${file}`));
    }
    assert.deepEqual(errors, []);
    const report = { status: 'passed', checks, errors, screenshots: screenshots.map(path => ({ path, sha256: crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex') })), visualInspection: 'Captures require separate human/agent opening; the browser assertions do not claim image inspection.' };
    fs.writeFileSync('docs/teaching/evidence/evaluation-metrics-browser.json', JSON.stringify(report, null, 2)+'\n');
    console.log(JSON.stringify({ status: 'passed', groups: checks.length, captures: screenshots.length }));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

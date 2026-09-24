const fs = require('fs'),
  crypto = require('crypto');
const {
  chromium
} = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const names = ['feature-selection-importance-shap-permutation-mutual-info', 'bias-variance-tradeoff-learning-curves', 'imbalanced-learning-smote-cost-sensitive-learning', 'automl-neural-architecture-search-nas', 'hidden-markov-models-hmm', 'bayesian-networks-causal-graphical-models', 'conditional-random-fields-crf', 'gaussian-processes-gp', 'semi-supervised-learning-label-propagation-self-training-co-training', 'active-learning', 'evaluation-metrics-precision-recall-f1-auc-roc-ap-r-mae', 'pac-learning-vc-dimension', 'calibration-conformal-prediction', 'rademacher-complexity-generalization-bounds', 'ml-problem-formulation-baselines-data-leakage', 'time-series-validation-forecasting-baselines', 'end-to-end-supervised-learning-error-analysis'];
let browser;
(async () => {
  browser = await chromium.launch({
    channel: 'msedge',
    headless: true
  });
  const context = await browser.newContext({
    viewport: {
      width: 1366,
      height: 900
    }
  });
  const fonts = JSON.parse(fs.readFileSync('scratch/kmeans-revision-review/fonts/manifest.json', 'utf8').replace(/^\uFEFF/, ''));
  await context.route('https://fonts.googleapis.com/**', route => route.fulfill({
    path: fonts.stylesheet,
    contentType: 'text/css'
  }));
  await context.route('https://fonts.gstatic.com/**', route => {
    const file = fonts.files[route.request().url()];
    return file ? route.fulfill({
      path: file,
      contentType: 'font/ttf',
      headers: {
        'access-control-allow-origin': '*'
      }
    }) : route.abort();
  });
  const page = await context.newPage();
  const all = [];
  let errors = [];
  page.on('pageerror', e => errors.push(e.message));
  for (const id of names) {
    errors = [];
    await page.goto((process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4190') + '/learn/path/full-curriculum/' + id + '?module=classical-ml', { waitUntil: 'domcontentloaded' });
    await page.locator('[data-live-exploration]').first().waitFor({
      timeout: 30000
    });
    await page.evaluate(() => document.fonts.ready);
    if ((await page.locator('[data-live-exploration]').count()) === 0) throw new Error(id + ': no live lab mounted');
    if (id === 'active-learning') await page.getByRole('button', {
      name: 'Start run',
      exact: true
    }).click();
    const forbidden = await page.getByRole('button', {
      name: /record prediction|commit prediction|explore without (a prediction|grading)/i
    }).count();
    if (forbidden) throw new Error(id + ': obsolete prediction button');
    const outer = await page.evaluate(() => [...document.querySelectorAll('section')].filter(e => e.querySelector('[data-live-exploration]') || e.hasAttribute('data-live-exploration')).filter(e => !e.querySelector('section')).map(e => e.className));
    const records = [];
    for (let i = 0; i < outer.length; i++) {
      const lab = page.locator('section.' + outer[i].split(' ').join('.')).nth(outer.slice(0, i).filter(x => x === outer[i]).length);
      await lab.scrollIntoViewIfNeeded();
      const state = () => lab.evaluate(e => [...e.querySelectorAll('svg,table,[role="status"]')].map(x => x.outerHTML).join(''));
      const before = await state();
      let control = lab.locator('input[type="number"]:enabled').first();
      if (await control.count()) {
        await control.locator('xpath=ancestor::details').evaluateAll(es => es.forEach(e => e.open = true));
        const value = await control.inputValue();
        const bounds = await control.evaluate(e => ({
          min: Number(e.min),
          max: Number(e.max),
          step: e.step,
          label: e.getAttribute('aria-label') || e.closest('label')?.innerText
        }));
        let next = Number(value) + (bounds.step && bounds.step !== 'any' ? Number(bounds.step) : 1);
        if (next > bounds.max) next = Number(value) - 1;
        if (next < bounds.min) next = (bounds.min + bounds.max) / 2;
        await control.fill(String(next));
        await page.waitForTimeout(100);
        records.push({
          lab: await lab.locator('h3').first().innerText(),
          control: bounds.label,
          before: value,
          after: await control.inputValue(),
          changed: (await state()) !== before
        });
      } else {
        control = lab.locator('select:enabled').first();
        if (await control.count()) {
          const vals = await control.locator('option').evaluateAll(es => es.map(x => x.value));
          const value = await control.inputValue(),
            next = vals.find(x => x !== value);
          if (next !== undefined) {
            await control.selectOption(next);
            await page.waitForTimeout(100);
            records.push({
              lab: await lab.locator('h3').first().innerText(),
              control: 'select',
              before: value,
              after: next,
              changed: (await state()) !== before
            });
          }
        }
      }
    }
    const focusedChecks = [];
    if (id === 'conditional-random-fields-crf') {
      const lab = page.locator('[data-crf-lab="trellis"]');
      const field = lab.locator('input[type="number"]').first();
      const validValue = await field.inputValue();
      await field.fill('');
      if (await lab.locator('.crf-result').count()) throw new Error('CRF invalid factors retained a current result');
      await field.fill(validValue);
      await lab.locator('.crf-result').waitFor();
      const explanation = await lab.locator('.crf-result > p').first().innerText();
      if (explanation.includes("? '") || !/The (largest-path sets|pair contributions)/.test(explanation)) throw new Error('CRF explanation is malformed');
      await lab.getByRole('button', { name: 'Reset', exact: true }).click();
      await lab.locator('.crf-result').waitFor();
      focusedChecks.push('Invalid factor suppresses result; valid recovery and reset restore trellis; conditional explanation renders as prose.');
    }
    if (id === 'gaussian-processes-gp') {
      const lab = page.locator('[data-gp-lab="conditioning"]');
      const slider = lab.locator('input[type="range"]').first();
      const before = await lab.locator('[data-gp-result]').innerHTML();
      const previousValue = await slider.inputValue();
      await slider.focus();
      await slider.press('ArrowRight');
      if (await slider.inputValue() === previousValue) throw new Error('GP keyboard slider did not change');
      const output = await lab.locator('[data-gp-result]').innerHTML();
      if (output === before) throw new Error('GP slider did not update connected output');
      focusedChecks.push('Keyboard range control changes a real GP input without a prediction field; matching calculation remains available.');
      await lab.screenshot({ path: 'docs/teaching/evidence/live-exploration-classical-late-gp-desktop.png' });
    }
    if (id === 'rademacher-complexity-generalization-bounds') {
      const button = page.getByRole('button', { name: 'Inspect the frozen assessment report', exact: true });
      const table = page.locator('.rad-table').filter({ has: page.locator('.rad-caption', { hasText: 'Assessment of the frozen validation-selected study' }) });
      if (await table.count()) throw new Error('Rademacher assessment opened before explicit inspection');
      await button.click();
      await table.waitFor();
      const lab = table.locator('xpath=ancestor::section[1]');
      await lab.getByRole('button', { name: 'Reset', exact: true }).click();
      if (await table.count() !== 1) throw new Error('Reset incorrectly erased assessment exposure');
      focusedChecks.push('Live validation selection does not expose assessment; explicit inspection opens it without a guess; exposure persists after reset.');
    }
    await page.setViewportSize({
      width: 390,
      height: 844
    });
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - innerWidth);
    if (overflow > 1) throw new Error(id + ': page overflow ' + overflow);
    await page.setViewportSize({
      width: 1366,
      height: 900
    });
    if (errors.length) throw new Error(id + ': ' + errors.join(';'));
    all.push({
      id,
      records,
      focusedChecks,
      errors: [...errors],
      phonePageOverflow: overflow
    });
    console.log(id + ': ' + records.length + ' control edits; no errors');
  }
  const families = ['Selection', 'BiasVariance', 'Imbalance', 'Automl', 'Hmm', 'BayesNet', 'Crf', 'GaussianProcess', 'SemiSupervised', 'ActiveLearning', 'EvaluationMetrics', 'Pac', 'Calibration', 'Rademacher', 'Formulation', 'TimeSeries', 'EndToEnd'];
  const sourceFiles = ['src/learn/components/lesson-labs/LiveInvestigationState.js', ...families.flatMap(name => ['Labs', 'Shared'].map(suffix => 'src/learn/components/lesson-labs/' + name + suffix + '.jsx')).filter(file => fs.existsSync(file)), ...names.map(id => 'src/learn/data/topics/' + id + '.jsx'), ...names.map(id => 'src/learn/data/curriculum/blueprints/' + id + '.js')];
  const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
  const receipt = {
    status: 'passed',
    checkedAt: new Date().toISOString(),
    baseUrl: process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4190',
    topicCount: all.length,
    controlChanges: all.reduce((n, t) => n + t.records.length, 0),
    scope: 'Actual controls and current diagrams/tables at desktop; phone document overflow; retained exact font fixtures. Unchanged outputs can be legitimate nulls. Not a fresh numerical audit of unchanged model implementations.',
    sourceHashes: Object.fromEntries(sourceFiles.map(file => [file, hash(file)])),
    verifierHash: hash(__filename),
    topics: all
  };
  fs.writeFileSync('docs/teaching/evidence/live-exploration-classical-late-browser.json', JSON.stringify(receipt, null, 2) + '\n');
  await browser.close();
})().catch(async e => {
  await browser?.close();
  console.error(e);
  process.exitCode = 1;
});

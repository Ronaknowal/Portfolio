const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const base = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4185';
const output = 'scratch/gaussian-process-implementation';
fs.mkdirSync(output, { recursive: true });
const evidence = { base, passed: false, widths: [], checks: [], screenshots: [], pageErrors: [] };
const check = (name, condition) => { assert.ok(condition, name); evidence.checks.push(name); };
const sourcePaths = ['src/learn/data/topics/gaussian-processes-gp.jsx', 'src/learn/components/lesson-labs/GaussianProcessFigures.jsx', 'src/learn/components/lesson-labs/GaussianProcessLabs.jsx', 'src/learn/components/lesson-labs/gaussian-process.css', 'src/learn/data/gaussian-process-model.js', 'src/learn/data/gaussian-process-data.js'];
const digest = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
async function capture(locator, filename) {
  await locator.scrollIntoViewIfNeeded();
  await locator.screenshot({ path: filename, animations: 'disabled', style: '.learn-nav { visibility: hidden !important; }' });
  evidence.screenshots.push({ file: filename, sha256: digest(filename), bytes: fs.statSync(filename).size });
}
async function predict(lab, name, reason = 'I am comparing the covariance connection and fixed inputs.') {
  await lab.getByRole('radio', { name, exact: true }).check();
  await lab.getByLabel('Reason before the result', { exact: true }).fill(reason);
  await lab.getByRole('button', { name: 'Commit prediction and reveal', exact: true }).click();
}
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    for (const width of [1366, 390, 320]) {
      const context = await browser.newContext({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      const page = await context.newPage();
      page.on('pageerror', error => evidence.pageErrors.push(error.message));
      await page.goto(`${base}/learn/path/full-curriculum/gaussian-processes-gp?module=classical-ml`, { waitUntil: 'domcontentloaded', timeout: 60000 });
      await page.locator('.gp-lesson').waitFor();
      await page.evaluate(() => document.fonts.ready);
      check(`${width}: full lesson and all six inline figures`, await page.locator('[data-gp-figure]').count() === 6);
      check(`${width}: all four investigation mounts`, await page.locator('[data-gp-lab]').count() === 4);
      check(`${width}: no formula render errors`, await page.locator('.katex-error').count() === 0);
      check(`${width}: no result shown before commitment`, await page.locator('[data-gp-result]').count() === 0);
      check(`${width}: changed-practice solutions closed`, await page.locator('.gp-lesson details[open]').count() === 0);
      check(`${width}: two complete native stdout blocks`, await page.locator('.python-example').count() === 2 && (await page.locator('.python-example').last().innerText()).includes('seasonal_naive MAE 3.813333'));
      const download = await context.request.get(`${base}/learn/examples/gaussian-processes-gp/mauna-loa-monthly.csv`);
      check(`${width}: offline source CSV downloadable`, download.ok() && (await download.text()).trim().split('\n').length === 121);
      const paint = await page.locator('.gp-band').first().evaluate(element => ({ fill: getComputedStyle(element).fill, opacity: getComputedStyle(element).fillOpacity }));
      check(`${width}: latent band actually painted`, paint.fill !== 'none' && Number(paint.opacity) > 0);
      const geometry = await page.locator('.gp-lesson').evaluate(root => {
        const collisions = [], clipped = [], pathCrossings = [];
        for (const svg of root.querySelectorAll('svg')) {
          const labels = [...svg.querySelectorAll('text')].map(element => ({ text: element.textContent, box: element.getBoundingClientRect() }));
          const bounds = svg.getBoundingClientRect();
          for (let i = 0; i < labels.length; i++) {
            const a = labels[i];
            if (a.box.left < bounds.left - 1 || a.box.right > bounds.right + 1 || a.box.top < bounds.top - 1 || a.box.bottom > bounds.bottom + 1) clipped.push(a.text);
            for (let j = i + 1; j < labels.length; j++) { const b = labels[j]; if (Math.min(a.box.right, b.box.right) - Math.max(a.box.left, b.box.left) > 1 && Math.min(a.box.bottom, b.box.bottom) - Math.max(a.box.top, b.box.top) > 1) collisions.push([a.text, b.text]); }
          }
          for (const element of svg.querySelectorAll('path')) {
            const painted = getComputedStyle(element);
            if (painted.stroke === 'none' || element.classList.contains('gp-grid')) continue;
            const transform = element.getScreenCTM();
            const length = element.getTotalLength();
            for (let distance = 0; distance <= length; distance += 2) {
              const point = element.getPointAtLength(distance);
              const screen = new DOMPoint(point.x, point.y).matrixTransform(transform);
              const hit = labels.find(({ box }) => screen.x > box.left && screen.x < box.right && screen.y > box.top && screen.y < box.bottom);
              if (hit) { pathCrossings.push({ label: hit.text, chart: svg.getAttribute('aria-label') }); break; }
            }
          }
        }
        return { collisions, clipped, pathCrossings, overflow: document.documentElement.scrollWidth > innerWidth + 1 };
      });
      check(`${width}: SVG labels neither collide nor clip`, !geometry.collisions.length && !geometry.clipped.length);
      check(`${width}: foreground paths avoid chart labels`, !geometry.pathCrossings.length);
      check(`${width}: no page-level horizontal overflow`, !geometry.overflow);
      if (width !== 320) for (const figure of ['vector', 'slice', 'matrix', 'kernels', 'forecast', 'probes']) await capture(page.locator(`[data-gp-figure="${figure}"]`), `${output}/${figure}-${width}.png`);

      const direct = page.locator('[data-gp-lab="direct"]');
      await direct.getByLabel('Covariance ρ', { exact: true }).fill('0');
      await predict(direct, 'Both');
      check(`${width}: direct rho-null comparison correctly graded`, (await direct.innerText()).includes('Prediction matches.'));
      await direct.getByRole('button', { name: 'Save this state as baseline' }).click();
      await direct.getByLabel('Observed value', { exact: true }).fill('-2');
      check(`${width}: value edit clears previous result and choice`, await direct.locator('[data-gp-result]').count() === 0 && await direct.locator('input[type=radio]:checked').count() === 0);
      await predict(direct, 'Neither');
      check(`${width}: zero-covariance changed-value null correctly graded`, (await direct.innerText()).includes('Prediction matches.'));
      await direct.getByLabel('Covariance ρ', { exact: true }).focus();
      await page.keyboard.press('Tab');
      check(`${width}: keyboard advances to observed-value field`, await direct.getByLabel('Observed value', { exact: true }).evaluate(element => element === document.activeElement));
      const lab = page.locator('[data-gp-lab="conditioning"]');
      await lab.getByLabel('Reading 1 value', { exact: true }).fill('3');
      await lab.getByLabel('Reading 2 value', { exact: true }).fill('-2');
      await predict(lab, 'Mean only');
      check(`${width}: edited readings produce expected mean and fixed variance`, (await lab.innerText()).includes('0.437822') && (await lab.innerText()).includes('Prediction matches.'));
      if (width !== 320) await capture(lab, `${output}/conditioning-result-${width}.png`);
      await lab.getByRole('button', { name: 'Reset investigation', exact: true }).click();
      await lab.getByLabel('RBF length', { exact: true }).fill('0.3');
      await predict(lab, 'Latent variance only');
      check(`${width}: symmetric length edit changes variance alone`, (await lab.innerText()).includes('0.999976') && (await lab.innerText()).includes('Prediction matches.'));
      await lab.getByLabel('RBF length', { exact: true }).fill('');
      await predict(lab, 'Neither');
      check(`${width}: invalid numeric input shows error with no stale result`, await lab.getByRole('alert').count() === 1 && await lab.locator('[data-gp-result]').count() === 0);
      await lab.getByRole('button', { name: 'Reset investigation', exact: true }).click();
      await lab.getByRole('button', { name: 'Remove all readings' }).click();
      await predict(lab, 'Latent variance only');
      check(`${width}: zero-reading prior remains usable`, (await lab.innerText()).includes('Prediction matches.'));

      const forecast = page.locator('[data-gp-lab="forecast"]');
      check(`${width}: future numerical scores not leaked before commitment`, !(await forecast.innerText()).includes('0.320601') && !(await forecast.innerText()).includes('13/24'));
      await predict(forecast, 'Trend + periodic + RBF');
      check(`${width}: development result revealed and test still concealed`, await forecast.locator('[data-gp-result="development"]').count() === 1 && await forecast.locator('[data-gp-result="test"]').count() === 0);
      const testGate = forecast.locator('.gp-commit').last();
      await testGate.getByRole('radio', { name: 'No, the interval claim needs its own assessment', exact: true }).check();
      await testGate.getByLabel('Reason before the result', { exact: true }).fill('MAE measures error magnitude, whereas coverage compares errors with interval widths.');
      await testGate.getByRole('button', { name: 'Commit interval prediction and reveal test' }).click();
      check(`${width}: test failure remains visible`, (await forecast.locator('[data-gp-result="test"]').innerText()).includes('13/24'));
      if (width !== 320) await capture(forecast.locator('[data-gp-result="test"]'), `${output}/forecast-test-${width}.png`);
      await forecast.getByLabel(/^Experiment mode/).selectOption('explore');
      await forecast.getByLabel('Conditioning cutoff (months from January 1990)', { exact: true }).fill('84');
      await forecast.getByLabel('Forecast horizon (months)', { exact: true }).fill('6');
      await forecast.getByLabel('Your first forecast-month estimate (ppm)', { exact: true }).fill('366');
      await predict(forecast, 'It remains unchanged');
      check(`${width}: editable forecast performs fresh conditioning`, await forecast.locator('[data-gp-result="exploration"]').count() === 1 && (await forecast.innerText()).includes('Prefix mean:'));
      await forecast.getByLabel('Forecast horizon (months)', { exact: true }).fill('7');
      check(`${width}: horizon edit invalidates estimate result`, await forecast.locator('[data-gp-result]').count() === 0);
      await forecast.getByRole('button', { name: 'Reset forecast investigation' }).click();
      check(`${width}: reset returns to unsolved reported protocol`, await forecast.getByLabel(/^Experiment mode/).inputValue() === 'reported' && await forecast.locator('input[type=radio]:checked').count() === 0);

      const probe = page.locator('[data-gp-lab="probe"]');
      check(`${width}: candidate gains hidden initially`, !(await probe.innerText()).includes('0.305834'));
      await predict(probe, 'Candidate 1 at 1');
      check(`${width}: target-specific candidate gain`, (await probe.innerText()).includes('0.305834') && (await probe.innerText()).includes('Your choice is a maximum.'));
      await probe.getByLabel('Target to clarify', { exact: true }).fill('4');
      await predict(probe, 'Candidate 2 at 4');
      check(`${width}: moving target changes the selected candidate`, (await probe.innerText()).includes('Your choice is a maximum.'));
      if (width !== 320) await capture(probe, `${output}/probe-result-${width}.png`);
      await probe.getByRole('button', { name: 'Reset probe investigation' }).click();
      await probe.getByLabel(/^Probe covariance model/).selectOption('independent');
      await probe.getByLabel('Candidate 1 position', { exact: true }).fill('3');
      await predict(probe, 'Candidate 2 at 4');
      check(`${width}: all-zero-gain tie accepts either candidate`, (await probe.innerText()).includes('The maximum is tied.') && (await probe.innerText()).includes('Your choice is a maximum.'));
      check(`${width}: no page overflow after informative states`, await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
      evidence.widths.push(width);
      await context.close();
    }
    check('No browser runtime errors', evidence.pageErrors.length === 0);
    evidence.source = Object.fromEntries(sourcePaths.map(file => [file, digest(file)]));
    evidence.passed = true;
    fs.writeFileSync('docs/teaching/evidence/gaussian-process-browser.json', JSON.stringify(evidence, null, 2) + '\n');
    console.log(JSON.stringify({ passed: true, checks: evidence.checks.length, widths: evidence.widths, screenshots: evidence.screenshots.length }));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

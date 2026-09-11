const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const directory = 'scratch/ito-sde-browser';
fs.mkdirSync(directory, { recursive: true });

async function capture(page, target, name, width, full = false) {
  await target.evaluate(element => window.scrollTo(0, window.scrollY + element.getBoundingClientRect().top - 175));
  if (full) {
    await page.addStyleTag({ content: '.learn-nav { visibility:hidden !important; }' });
    await target.screenshot({ path: `${directory}/${name}-${width}.png` });
    await page.addStyleTag({ content: '.learn-nav { visibility:visible !important; }' });
  } else await page.screenshot({ path: `${directory}/${name}-${width}.png` });
}
async function metric(region, label) {
  return region.locator('.ito-metrics>div').filter({ has: region.page().locator('dt', { hasText: label }) }).first().locator('dd').innerText();
}
async function slider(locator, value) {
  await locator.evaluate((element, next) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(element, String(next));
    element.dispatchEvent(new Event('input', { bubbles: true }));
    element.dispatchEvent(new Event('change', { bubbles: true }));
  }, value);
}

(async () => {
  const { itoSdeExamples: examples } = await import('../src/learn/data/ito-sde-examples.js');
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 } });
      await page.routeWebSocket('**', socket => socket.close());
      const errors = [], warnings = [], failedRequests = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => {
        if (['warning', 'error'].includes(message.type()) && !message.text().startsWith('[vite]')) warnings.push(message.text());
      });
      page.on('requestfailed', request => failedRequests.push(request.url()));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/it-calculus-stochastic-differential-equations?module=math-foundations', { waitUntil: 'networkidle' });
      const lesson = page.locator('.ito-sde-lesson');
      await lesson.waitFor();
      await page.evaluate(() => document.fonts.ready);
      const anchors = await lesson.locator('.lesson-intro a[href^="#"]').evaluateAll(elements => elements.map(element => ({
        href: element.getAttribute('href'), exists: !!document.getElementById(element.getAttribute('href').slice(1)),
      })));
      assert.equal(anchors.length, 10); assert.ok(anchors.every(anchor => anchor.exists));
      assert.equal(await lesson.locator('.katex-error').count(), 0);
      const programs = await lesson.locator('.python-example').evaluateAll(elements => elements.map(element => ({
        title: element.querySelector('h3').textContent,
        question: element.previousElementSibling.textContent.replace(/^Before running:\s*/, ''),
        outputs: [...element.children].filter(child => getComputedStyle(child).whiteSpace === 'pre').map(child => [...child.childNodes].filter(node => node.nodeType === Node.TEXT_NODE).map(node => node.textContent).join('')),
      })));
      assert.equal(programs.length, 13);
      for (const program of programs) {
        const example = Object.values(examples).find(value => value.title === program.title);
        assert.ok(example); assert.equal(program.question, example.question);
        assert.equal(program.outputs.length, 2);
        assert.equal(program.outputs[0].trim(), example.code.trim());
        assert.equal(program.outputs[1].trim(), example.expected.trim());
      }
      for (const target of await lesson.locator('.lesson-check details').all()) {
        await target.locator('summary').click(); assert.equal(await target.getAttribute('open'), '');
      }
      assert.equal(await lesson.getByText('Try it independently.', { exact: true }).count(), 11);
      for (let i = 0; i < 10; i += 1) await capture(page, lesson.locator('h2').nth(i), `reading-${i + 1}`, width);
      for (let i = 0; i < 4; i += 1) await capture(page, lesson.locator('.ito-figure').nth(i), `inline-${i + 1}`, width, true);
      let states = 0;
      const integral = page.getByRole('region', { name: 'Adapted integral investigation', exact: true });
      assert.equal(await metric(integral, 'Left sum'), '-0.4375');
      for (const value of ['left', 'right', 'symmetric']) {
        await integral.getByLabel('Coefficient choice', { exact: true }).selectOption({ value });
        await integral.getByRole('button', { name: 'Next interval', exact: true }).click(); states++;
      }
      assert.equal(await metric(integral, 'Symmetric sum'), '0.125');
      await integral.getByLabel('Increment source', { exact: true }).selectOption({ value: 'brownian' });
      const terminal = await metric(integral, 'Terminal W');
      for (const value of ['64', '16', '4', '1']) {
        await integral.getByLabel('Observed intervals', { exact: true }).selectOption({ value });
        assert.equal(await metric(integral, 'Terminal W'), terminal); states++;
      }
      await integral.getByLabel('Integral seed', { exact: true }).fill('0');
      await integral.getByRole('button', { name: 'Apply seed', exact: true }).click();
      assert.equal(await metric(integral, 'Terminal W'), terminal); assert.equal(await integral.getByRole('alert').count(), 1);
      await capture(page, integral, 'integral-invalid', width, true);
      await integral.getByLabel('Integral seed', { exact: true }).fill('19');
      await integral.getByLabel('Integral seed', { exact: true }).press('Enter');
      assert.notEqual(await metric(integral, 'Terminal W'), terminal);
      await integral.getByRole('button', { name: 'Reset integral', exact: true }).focus(); await page.keyboard.press('Enter');
      assert.equal(await metric(integral, 'Left sum'), '-0.4375'); states += 3;

      const growth = page.getByRole('region', { name: 'Growth law investigation', exact: true });
      for (const mu of ['-0.5', '0.2', '0.4', '1']) for (const sigma of ['0', '0.3', '0.6', '1']) {
        await growth.getByLabel('Growth drift mu', { exact: true }).selectOption({ value: mu });
        await growth.getByLabel('Growth noise sigma', { exact: true }).selectOption({ value: sigma });
        for (const index of [0, 4, 64, 256]) {
          await slider(growth.getByLabel('Growth time index (64 per unit time)', { exact: true }), index);
          const actual = Number(await metric(growth, 'Selected time')); assert.equal(actual, index / 64); states++;
        }
      }
      await growth.getByLabel('Growth drift mu', { exact: true }).selectOption({ value: '0.2' });
      await growth.getByLabel('Growth noise sigma', { exact: true }).selectOption({ value: '1' });
      await capture(page, growth, 'growth-mean-median', width, true);
      await growth.getByLabel('Growth seed', { exact: true }).fill('NaN');
      await growth.getByRole('button', { name: 'Apply seed', exact: true }).click();
      await growth.getByRole('button', { name: 'Reset growth law', exact: true }).focus(); await page.keyboard.press('Enter');
      assert.equal(await growth.getByRole('alert').count(), 0); assert.equal(await growth.getByLabel('Growth seed', { exact: true }).inputValue(), '5');
      await growth.getByLabel('Growth time index (64 per unit time)', { exact: true }).focus(); await page.keyboard.press('ArrowRight');
      assert.equal(await metric(growth, 'Selected time'), '1.0625'); states += 3;

      const ou = page.getByRole('region', { name: 'OU restoring flow investigation', exact: true });
      for (const theta of ['0', '0.3', '1', '2']) for (const eta of ['0', '0.4', '0.8']) {
        await ou.getByLabel('OU reversion theta', { exact: true }).selectOption({ value: theta });
        await ou.getByLabel('OU noise eta', { exact: true }).selectOption({ value: eta });
        for (const initial of theta === '0' ? ['fixed'] : ['fixed', 'stationary']) {
          await ou.getByLabel('OU initial law', { exact: true }).selectOption({ value: initial });
          for (const time of [0, 0.125, 1, 4]) {
            await slider(ou.getByLabel('OU elapsed time', { exact: true }), time);
            assert.ok(Number(await metric(ou, 'Variance v')) >= 0);
            if (initial === 'stationary') assert.ok(Math.abs(Number(await metric(ou, 'Variance change rate'))) < 1e-12);
            states++;
          }
        }
      }
      await capture(page, ou, 'ou-stationary', width, true);
      await ou.getByLabel('OU reversion theta', { exact: true }).selectOption({ value: '0' });
      assert.equal(await ou.getByLabel('OU initial law', { exact: true }).inputValue(), 'fixed');
      await ou.getByLabel('OU noise eta', { exact: true }).selectOption({ value: '0' });
      await capture(page, ou, 'ou-atom', width, true);
      await ou.getByRole('button', { name: 'Reset OU', exact: true }).click(); states += 3;

      const solver = page.getByRole('region', { name: 'Coupled SDE solver investigation', exact: true });
      for (const sigma of ['0', '0.3', '0.6', '1']) {
        await solver.getByLabel('Solver noise scale', { exact: true }).selectOption({ value: sigma });
        const endpoint = await metric(solver, 'Exact terminal value');
        for (const group of ['32', '8', '2', '1']) {
          await solver.getByLabel('Solver coarse steps', { exact: true }).selectOption({ value: group });
          assert.equal(await metric(solver, 'Exact terminal value'), endpoint);
          await solver.getByRole('button', { name: 'Next solver step', exact: true }).click(); states++;
        }
      }
      await solver.getByLabel('Solver fixture', { exact: true }).selectOption({ value: 'stress' });
      assert.equal(await metric(solver, 'Euler nonpositive?'), 'Yes'); assert.ok((await solver.innerText()).includes('new: -0.6'));
      assert.ok(await solver.getByLabel('Solver coarse steps', { exact: true }).isDisabled());
      assert.equal(await solver.getByLabel('Solver coarse steps', { exact: true }).inputValue(), '1');
      assert.equal(await solver.getByLabel('Solver coarse steps', { exact: true }).locator('option:checked').innerText(), '1');
      assert.equal(await solver.getByLabel('Solver noise scale', { exact: true }).inputValue(), '1');
      await capture(page, solver, 'solver-negative', width, true);
      await solver.getByRole('button', { name: 'Previous solver step', exact: true }).click();
      assert.ok((await solver.innerText()).includes('Initial state: no increment used yet.'));
      await solver.getByRole('button', { name: 'Reset coupled solvers', exact: true }).click();
      await solver.getByLabel('Solver seed', { exact: true }).fill('9999999999999999');
      await solver.getByRole('button', { name: 'Apply seed', exact: true }).click();
      await solver.getByRole('button', { name: 'Reset coupled solvers', exact: true }).focus(); await page.keyboard.press('Enter');
      assert.equal(await solver.getByRole('alert').count(), 0); states += 5;

      const error = page.getByRole('region', { name: 'SDE error budget investigation', exact: true });
      for (const mu of ['-0.5', '0', '0.4', '1']) for (const sigma of ['0', '0.05', '0.6', '1']) {
        await error.getByLabel('Error drift mu', { exact: true }).selectOption({ value: mu });
        await error.getByLabel('Error noise sigma', { exact: true }).selectOption({ value: sigma });
        for (const method of ['euler', 'milstein']) for (const step of ['4', '16', '64', '256']) {
          await error.getByLabel('Error method', { exact: true }).selectOption({ value: method });
          await error.getByLabel('Error step count', { exact: true }).selectOption({ value: step });
          const rms = Number(await metric(error, 'Analytic terminal RMS')); assert.ok(Number.isFinite(rms) && rms >= 0);
          if (mu === '0') assert.equal(await metric(error, 'Analytic first-moment bias'), '0'); states++;
        }
      }
      await error.getByLabel('Error drift mu', { exact: true }).selectOption({ value: '0' });
      await error.getByLabel('Error noise sigma', { exact: true }).selectOption({ value: '0' });
      await capture(page, error, 'error-zero', width, true);
      await error.getByRole('button', { name: 'Reset error experiment', exact: true }).click();
      for (const count of ['64', '512', '4096']) {
        await error.getByLabel('Monte Carlo path count', { exact: true }).selectOption({ value: count });
        assert.equal(await error.locator('.ito-sample-result').count(), 0);
        await error.getByRole('button', { name: 'Run paired Monte Carlo', exact: true }).click();
        assert.ok((await error.locator('.ito-sample-result').innerText()).includes(`${count} paired paths`)); states++;
      }
      await capture(page, error, 'error-sampling', width, true);
      await error.getByRole('button', { name: 'Reset error experiment', exact: true }).focus(); await page.keyboard.press('Enter');
      assert.equal(await error.locator('.ito-sample-result').count(), 0); states += 2;
      for (const target of await lesson.locator('.ito-lab details').all()) await target.locator('summary').click();
      if (width < 500) {
        const plot = lesson.locator('.ito-plot:not(.compact)').first();
        await plot.focus();
        for (let i = 0; i < 5; i++) await page.keyboard.press('ArrowRight');
        await page.waitForTimeout(150);
        assert.ok(await plot.evaluate(element => element.scrollLeft > 0));
        states++;
      }
      const geometry = await lesson.evaluate(element => ({
        documentWidth: document.documentElement.scrollWidth,
        fonts: [...document.fonts].some(font => font.family === 'Space Grotesk' && font.status === 'loaded'),
        equations: [...element.querySelectorAll('.katex-display')].map(node => ({ width: node.getBoundingClientRect().width,
          content: node.scrollWidth, tex: node.querySelector('annotation')?.textContent })),
        svgOverflow: [...element.querySelectorAll('.ito-plot svg')].flatMap(svg => [...svg.querySelectorAll('text')].filter(node => {
          const box = node.getBBox(); return box.x < -0.5 || box.x + box.width > svg.viewBox.baseVal.width + 0.5;
        }).map(node => node.textContent)),
        controls: [...element.querySelectorAll('.ito-lab button,.ito-lab input,.ito-lab select')].map(node => ({
          text: node.getAttribute('aria-label') || node.textContent, height: node.getBoundingClientRect().height,
        })),
      }));
      assert.equal(geometry.documentWidth, width); assert.ok(geometry.fonts);
      assert.deepEqual(errors, []); assert.deepEqual(warnings, []); assert.deepEqual(failedRequests, []);
      assert.ok(geometry.controls.every(control => control.height >= 43));
      results.push({ width, states, anchors, programs: programs.map(({ title, question }) => ({ title, question })),
        independentPractice: 11, geometry, errors, warnings, failedRequests });
      fs.writeFileSync(directory + '/in-progress-results.json', JSON.stringify(results, null, 2));
      console.log(width, 'passed behavior', states, 'states; equation overflows', geometry.equations.filter(e => e.content > e.width + 1).map(e => e.tex));
      await page.close();
    }
  } finally { await browser.close(); }
  fs.writeFileSync(directory + '/results.json', JSON.stringify({ checkedAt: new Date().toISOString(), results }, null, 2));
  assert.ok(results.every(row => row.geometry.equations.every(e => e.content <= e.width + 1)), 'Equation layout needs repair');
  assert.ok(results.every(row => row.geometry.svgOverflow.length === 0), 'SVG labels need repair');
})().catch(error => { console.error(error); process.exitCode = 1; });

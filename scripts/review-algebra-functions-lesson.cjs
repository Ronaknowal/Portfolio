const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const directory = path.resolve('scratch/algebra-functions-browser');
fs.mkdirSync(directory, { recursive: true });

(async () => {
  const { algebraFunctionsExamples: examples } = await import('../src/learn/data/algebra-functions-examples.js');
  const model = await import('../src/learn/data/algebra-functions-models.js');
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1080 }, reducedMotion: 'reduce' });
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/algebra-functions-exponentials-logarithms', { waitUntil: 'domcontentloaded', timeout: 60000 });
      const lesson = page.locator('.algebra-lesson');
      await lesson.waitFor({ timeout: 60000 });
      await page.evaluate(() => document.fonts.ready);
      assert(await page.evaluate(() => document.fonts.check('16px "Space Grotesk"')));
      const capture = async (locator, name) => {
        await locator.first().evaluate(node => window.scrollTo({ top: node.getBoundingClientRect().top + scrollY - 84, behavior: 'instant' }));
        await page.waitForTimeout(150);
        await page.screenshot({ path: path.join(directory, `${name}-${width}.png`) });
      };
      const range = async (label, value) => {
        const control = lesson.getByRole('slider', { name: label, exact: true });
        await control.fill(String(value)); await control.dispatchEvent('input');
      };
      const lab = name => lesson.locator(`.algebra-lab[aria-label="${name}"]`);
      assert.equal(await lesson.locator('h2').count(), 11);
      assert.equal(await lesson.locator('.algebra-lab').count(), 6);
      const broken = await lesson.locator('nav a').evaluateAll(links => links.map(a => a.hash).filter(hash => !document.getElementById(hash.slice(1))));
      assert.deepEqual(broken, []);
      for (let i = 0; i < 11; i++) await capture(lesson.locator('h2').nth(i), `reading-${i + 1}`);
      for (let i = 0; i < 2; i++) await capture(lesson.locator('.algebra-figure').nth(i), `inline-${i + 1}`);

      const equation = lab('Equation steps investigation');
      await equation.getByRole('button', { name: 'Next step', exact: true }).click();
      await equation.getByRole('button', { name: 'Next step', exact: true }).click();
      assert((await equation.innerText()).includes('x ≈ 5'));
      await equation.getByRole('button', { name: 'Previous step' }).click();
      assert((await equation.innerText()).includes('Step 2 of 3'));
      await equation.getByRole('button', { name: 'Restart steps' }).click();
      await equation.getByRole('textbox', { name: 'Coefficient a' }).fill('');
      await equation.getByRole('button', { name: 'Apply equation' }).click();
      assert.equal(await equation.getByRole('alert').count(), 1);
      assert((await equation.locator('.algebra-equation').innerText()).includes('3x'));
      await capture(equation, 'equation-invalid');
      for (const [preset, text] of [['Every x', 'All real x'], ['No x', 'No real x'], ['Negative slope', 'x ≈ -3']]) {
        await equation.getByRole('button', { name: preset, exact: true }).click();
        await equation.getByRole('button', { name: 'Next step', exact: true }).click();
        await equation.getByRole('button', { name: 'Next step', exact: true }).click();
        assert((await equation.innerText()).includes(text));
      }
      await capture(equation, 'equation-negative');
      await equation.getByRole('button', { name: 'Ordinary', exact: true }).focus();
      await page.keyboard.press('Enter');
      assert((await equation.innerText()).includes('Step 1 of 3'));

      const probe = lab('Function and inverse investigation');
      await range('Function input', -2);
      assert((await probe.innerText()).includes('Inputs returning this output: -2 and 2'));
      await capture(probe, 'function-two-preimages');
      await probe.getByRole('checkbox').check();
      assert((await probe.innerText()).includes('outside the selected domain'));
      await range('Function input', 2);
      assert((await probe.innerText()).includes('Inputs returning this output: 2.'));
      await capture(probe, 'function-branch');
      await probe.getByRole('combobox').selectOption('reciprocal');
      await range('Function input', 0);
      assert((await probe.innerText()).includes('There is no output; it is not zero'));
      assert.equal(await probe.locator('polyline').count(), 2);
      await capture(probe, 'function-zero');
      await probe.getByRole('combobox').selectOption('affine');
      const slider = lesson.getByRole('slider', { name: 'Function input', exact: true });
      await slider.focus(); await page.keyboard.press('ArrowRight');
      assert.equal(await slider.inputValue(), '0.25');

      const composition = lab('Function composition investigation');
      for (const x of [-4, -1, 0, 2, 4]) {
        await range('Composition input', x);
        const state = model.compositionState(x);
        assert((await composition.innerText()).includes(`Square after affine: ${model.algebraNumber(state.squareAfterAffine)}.`));
        assert((await composition.innerText()).includes(`Affine after square: ${model.algebraNumber(state.affineAfterSquare)}.`));
      }
      await capture(composition, 'composition-order');

      const quadratic = lab('Quadratic roots investigation');
      await range('Vertex h', -1); await range('Vertex k', -9);
      assert((await quadratic.innerText()).includes('approximately -4 and 2'));
      await capture(quadratic, 'quadratic-two');
      await range('Vertex k', 0);
      assert((await quadratic.innerText()).includes('One root x=-1'));
      await range('Vertex k', 4);
      assert((await quadratic.innerText()).includes('No real roots'));
      await capture(quadratic, 'quadratic-none');

      const growth = lab('Growth and scale investigation');
      let growthStates = 0;
      for (const rate of [-.5, -.2, 0, .1, .2, .5, 1]) {
        await growth.getByRole('combobox', { name: 'Growth rate' }).selectOption(String(rate));
        for (const n of [0, 4, 8]) {
          await range('Observation period', n);
          const expected = model.algebraNumber(model.growthState(rate, n).active.growth);
          assert((await growth.locator('.algebra-result').innerText()).includes(`gives ${expected} units`));
          growthStates++;
        }
        await growth.getByRole('checkbox').check();
        assert.equal(await growth.locator('polyline').count(), 2);
        await growth.getByRole('checkbox').uncheck();
      }
      await growth.getByRole('combobox', { name: 'Growth rate' }).selectOption('-0.2');
      await growth.getByRole('combobox', { name: 'Growth target factor' }).selectOption('0.5');
      assert((await growth.innerText()).includes('3.10628'));
      await growth.getByRole('checkbox').check();
      await capture(growth, 'growth-log-decay');
      await capture(growth.locator('.algebra-plot'), 'growth-log-plot');
      await growth.getByRole('combobox', { name: 'Growth rate' }).selectOption('0');
      assert((await growth.innerText()).includes('never reaches'));
      await growth.getByRole('combobox', { name: 'Growth target factor' }).selectOption('1');
      assert((await growth.innerText()).includes('Every time'));

      const logs = lab('Logarithmic ruler investigation');
      for (const base of [.5, 2, 10]) {
        await logs.getByRole('combobox').selectOption(String(base));
        for (const exponent of [-3, 0, 1.5, 3]) {
          await range('Exponent position', exponent);
          const state = model.logarithmState(base, exponent);
          assert((await logs.locator('.algebra-result').innerText()).includes(model.algebraNumber(state.value)));
        }
      }
      await logs.getByRole('combobox').selectOption('0.5');
      await range('Exponent position', -1);
      assert((await logs.innerText()).includes('Larger ratios lie farther left'));
      await capture(logs, 'log-base-below-one');
      await capture(logs.locator('svg'), 'log-ruler');

      const checkpoints = lesson.locator('div.lesson-check');
      assert.equal(await checkpoints.count(), 2);
      for (let i = 0; i < 2; i++) {
        const checkpoint = checkpoints.nth(i);
        assert((await checkpoint.locator('p').first().innerText()).length > 80);
        await checkpoint.getByText('Show explanation', { exact: true }).click();
        assert((await checkpoint.locator('details').innerText()).length > 140);
      }
      const practice = lesson.locator('section.lesson-check');
      assert.equal(await practice.count(), 12);
      for (let i = 0; i < 12; i++) {
        await practice.nth(i).getByText('Hint', { exact: true }).click();
        await practice.nth(i).getByText('Explained solution', { exact: true }).click();
        assert((await practice.nth(i).locator('details').nth(1).innerText()).length > 160);
      }
      await capture(practice.nth(3), 'practice-inverse');
      await capture(practice.nth(10), 'practice-capstone');
      for (const example of examples) {
        const container = lesson.locator('.python-example').filter({ has: page.getByRole('heading', { name: example.title, exact: true }) });
        const text = (await container.innerText()).replace(/\r\n/g, '\n');
        assert(text.includes(example.code.trim()), 'Exact actual code: '+example.id);
        assert(text.includes(example.expected), 'Exact output: '+example.id);
        assert((await lesson.innerText()).includes(example.question), 'Visible question: '+example.id);
      }
      assert.equal(await lesson.locator('.python-example').count(), 9);
      await capture(lesson.locator('.python-example').last(), 'precision-program');
      await capture(lesson.locator('.lesson-sources'), 'sources');
      const mathWidths = await lesson.locator('.katex-display').evaluateAll(nodes => nodes.map((node, index) => ({ index, width: node.clientWidth, content: node.scrollWidth })));
      fs.writeFileSync(path.join(directory, `math-${width}.json`), JSON.stringify(mathWidths, null, 2));
      assert.deepEqual(mathWidths.filter(row => row.content > row.width + 2), [], 'All formulas fit ordinary reading width');
      for (let i = 0; i < mathWidths.length; i++) await capture(lesson.locator('.katex-display').nth(i), `equation-${i}`);
      assert.equal(await lesson.locator('.katex-error').count(), 0);
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'No document overflow');
      assert.deepEqual(errors, []);
      records.push({ width, fonts: true, labs: 6, figures: 2, sections: 11, examples: 9, practice: 12, checkpoints: 2, growthStates, keyboard: true, invalidRetained: true, mathWidths, errors });
      await page.close();
    }
  } finally { await browser.close(); }
  const result = { at: new Date().toISOString(), passed: true, records };
  fs.writeFileSync(path.join(directory, 'results.json'), JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result));
})().catch(error => { console.error(error); process.exit(1); });

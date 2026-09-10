const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const requireFonts = process.argv.includes('--require-fonts');
const directory = `scratch/mutual-information-browser-review${requireFonts ? '-fonts' : ''}`;
fs.mkdirSync(directory, { recursive: true });
const report = [];

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  for (const width of [1440, 390, 320]) {
    const page = await browser.newPage({ viewport: { width, height: 1000 }, deviceScaleFactor: 1 });
    await page.routeWebSocket('**', socket => socket.close());
    const errors = [], requests = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('requestfailed', request => requests.push({ url: request.url(), error: request.failure()?.errorText }));
    await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/mutual-information-information-bottleneck?module=math-foundations', { waitUntil: 'networkidle' });
    await page.locator('.mutual-information-lesson').waitFor();
    await page.evaluate(() => document.fonts.ready);
    const fonts = await page.evaluate(() => [...document.fonts].map(font => ({ family: font.family, status: font.status })));
    if (requireFonts) {
      assert.ok(fonts.some(font => font.family === 'Space Grotesk' && font.status === 'loaded'), 'Original site font did not load');
      assert.equal(requests.length, 0, 'Failed request in final font-enabled review');
    }
    const checks = [];
    const region = name => page.getByRole('region', { name, exact: true });
    const readouts = async scope => scope.locator('.mi-readouts > div').evaluateAll(elements => elements.map(element => ({ label: element.querySelector('dt').textContent, value: Number(element.querySelector('dd').textContent) })));
    const near = (actual, expected) => assert.ok(Math.abs(actual - expected) < .000011, `${actual} != ${expected}`);
    const range = async (scope, name, value) => {
      const input = scope.getByRole('slider', { name, exact: true });
      const [minimum, step] = await input.evaluate(element => [Number(element.min), Number(element.step)]);
      await input.focus();
      await input.press('Home');
      for (let i = 0; i < Math.round((value - minimum) / step); i += 1) await input.press('ArrowRight');
      near(Number(await input.inputValue()), value);
      assert.equal(await input.evaluate(element => element === document.activeElement), true);
      checks.push(`keyboard slider ${name}=${value}`);
    };
    const capture = async (scope, name) => {
      await scope.scrollIntoViewIfNeeded();
      await page.addStyleTag({ content: '.learn-nav{visibility:hidden!important}' });
      await scope.screenshot({ path: `${directory}/${name}-${width}.png` });
      await page.evaluate(() => [...document.querySelectorAll('style')].filter(element => element.textContent === '.learn-nav{visibility:hidden!important}').forEach(element => element.remove()));
    };
    const joint = region('Joint information investigation');
    near((await readouts(joint))[2].value, .2780719051126377);
    await joint.getByRole('button', { name: 'Inspect X=0, Y=1', exact: true }).click();
    assert.match(await joint.getByRole('status').innerText(), /-1.32193/);
    await capture(joint, 'joint-negative-cell');
    await range(joint, 'Channel flip probability', .5);
    near((await readouts(joint))[2].value, 0);
    await range(joint, 'Channel flip probability', 1);
    near((await readouts(joint))[2].value, 1);
    await range(joint, 'Probability of X=1', 0);
    near((await readouts(joint))[2].value, 0);
    assert.match(await joint.innerText(), /conditional law is not determined/);
    await joint.getByRole('button', { name: 'Reset joint investigation', exact: true }).click();
    near((await readouts(joint))[2].value, .2780719051126377);
    checks.push('joint negative contribution, independent, inverted, zero-row and reset');

    const conditional = region('Conditional information investigation');
    near((await readouts(conditional))[0].value, 0);
    await conditional.getByRole('combobox').selectOption('both');
    near((await readouts(conditional))[0].value, 1);
    await capture(conditional, 'conditional-pair');
    await conditional.getByRole('combobox').selectOption('a');
    await range(conditional, 'Probability of B=1', .1);
    near((await readouts(conditional))[0].value, .5310044064107189);
    await conditional.getByRole('combobox').selectOption('b');
    near((await readouts(conditional))[0].value, 0);
    await conditional.getByText('Inspect the two conditional groups', { exact: true }).click();
    assert.match(await conditional.innerText(), /weighted average is 1 bit/);
    await conditional.getByRole('button', { name: 'Reset conditional investigation', exact: true }).click();
    checks.push('conditional reveal, changed bias, explicit slices and reset');

    const representation = region('Information bottleneck representation investigation');
    await representation.getByRole('combobox').selectOption('both');
    near((await readouts(representation))[0].value, 2);
    near((await readouts(representation))[1].value, .5310044064107189);
    await capture(representation, 'representation-both');
    await representation.getByRole('combobox').selectOption('nuisance');
    near((await readouts(representation))[1].value, 0);
    await representation.getByRole('combobox').selectOption('constant');
    near((await readouts(representation))[0].value, 0);
    await representation.getByRole('combobox').selectOption('noisy');
    await range(representation, 'Encoder flip probability', .2);
    near((await readouts(representation))[0].value, .2780719051126377);
    near((await readouts(representation))[1].value, .17325362750738216);
    await range(representation, 'Relevance weight beta', .5);
    near((await readouts(representation))[2].value, .19144509135894664);
    await capture(representation, 'representation-noisy');
    await representation.getByRole('button', { name: 'Reset representation investigation', exact: true }).click();
    checks.push('all representations, joint bars, changing rate/relevance/objective, reset');

    const iteration = region('Bottleneck update investigation');
    assert.ok(await iteration.getByRole('button', { name: 'Previous update', exact: true }).isDisabled());
    const next = iteration.getByRole('button', { name: 'Next complete update', exact: true });
    await next.focus();
    await next.press('Enter');
    near((await readouts(iteration))[2].value, -.346865);
    await iteration.getByText('Inspect the predictive KL values used by this update', { exact: true }).click();
    await capture(iteration, 'iteration-first');
    await iteration.getByRole('button', { name: 'Inspect update 40', exact: true }).click();
    near((await readouts(iteration))[2].value, -.6008346473703947);
    assert.ok(await next.isDisabled());
    await iteration.getByRole('button', { name: 'Previous update', exact: true }).click();
    await iteration.getByRole('combobox').selectOption('symmetric');
    await iteration.getByRole('button', { name: 'Inspect update 40', exact: true }).click();
    near((await readouts(iteration))[2].value, 0);
    await capture(iteration, 'iteration-symmetric');
    await iteration.getByRole('combobox').selectOption('nuisance');
    await next.click();
    near((await readouts(iteration))[0].value, 0);
    await range(iteration, 'Update relevance weight beta', 0);
    await next.click();
    near((await readouts(iteration))[0].value, 0);
    await iteration.getByRole('button', { name: 'Reset bottleneck updates', exact: true }).click();
    checks.push('keyboard stepping, previous/40 bounds, symmetric/nuisance stationarity, beta zero, reset');

    const bound = region('Variational information bound investigation');
    near((await readouts(bound))[0].value, -.24168897740950874);
    await bound.getByRole('button', { name: 'Try mismatched approximations', exact: true }).click();
    near((await readouts(bound))[1].value, .267168);
    await capture(bound, 'bounds-mismatched');
    await range(bound, 'Decoder flipped-label probability', .98);
    assert.match(await bound.getByRole('status').innerText(), /negative lower bound/);
    near((await readouts(bound))[0].value, -.24168897740950874);
    await capture(bound, 'bounds-negative');
    await bound.getByRole('button', { name: 'Match both true distributions', exact: true }).click();
    near((await readouts(bound))[0].value, (await readouts(bound))[1].value);
    await range(bound, 'Reference probability of Z=1', .98);
    await capture(bound, 'bounds-reference-extreme');
    await bound.getByRole('button', { name: 'Match both true distributions', exact: true }).click();
    checks.push('fixed true information, separately moved gaps, negative lower bound, exact reset');

    const estimation = region('Mutual information estimation investigation');
    near((await readouts(estimation))[0].value, .06826202518089218);
    await estimation.getByRole('combobox', { name: 'Categories per variable', exact: true }).selectOption('8');
    await estimation.getByRole('combobox', { name: 'Observation count', exact: true }).selectOption('20');
    near((await readouts(estimation))[1].value, 0);
    await capture(estimation, 'sample-sparse');
    const before = await readouts(estimation);
    await estimation.getByRole('textbox', { name: 'Sampling seed', exact: true }).fill('not a seed');
    await estimation.getByRole('button', { name: 'Generate this sample', exact: true }).click();
    assert.match(await estimation.getByRole('alert').innerText(), /unchanged/);
    assert.deepEqual(await readouts(estimation), before);
    await estimation.getByRole('textbox', { name: 'Sampling seed', exact: true }).fill('123');
    await estimation.getByRole('textbox', { name: 'Sampling seed', exact: true }).press('Enter');
    assert.equal(await estimation.getByRole('alert').count(), 0);
    await estimation.getByRole('combobox', { name: 'Population law', exact: true }).selectOption('channel');
    near((await readouts(estimation))[1].value, 1.7166000799651506);
    await estimation.getByRole('combobox', { name: 'Observation count', exact: true }).selectOption('2000');
    await capture(estimation, 'sample-channel');
    await estimation.getByRole('button', { name: 'Reset estimation investigation', exact: true }).click();
    near((await readouts(estimation))[0].value, .06826202518089218);
    checks.push('sparse/large/known-channel samples, draft invalid unchanged, keyboard valid apply, reset');

    const geometry = await page.evaluate(() => {
      const lesson = document.querySelector('.mutual-information-lesson');
      const overflow = [...lesson.querySelectorAll('.katex-display')].map(element => ({ text: element.textContent, width: element.scrollWidth, available: element.parentElement.clientWidth })).filter(row => row.width > row.available + 1);
      const svgText = [...lesson.querySelectorAll('svg text')].map(element => {
        const bounds = element.getBBox(), view = element.ownerSVGElement.viewBox.baseVal;
        const scale = element.ownerSVGElement.getBoundingClientRect().width / view.width;
        return { text: element.textContent, font: parseFloat(getComputedStyle(element).fontSize)*scale, outside: bounds.x < -1 || bounds.x+bounds.width > view.width+1 || bounds.y < -1 || bounds.y+bounds.height > view.height+1 };
      });
      return { pageWidth: document.documentElement.clientWidth, contentWidth: document.documentElement.scrollWidth, mathOverflow: overflow, smallSvgText: svgText.filter(row => row.font < 13.5), outsideSvgText: svgText.filter(row => row.outside), katexErrors: [...lesson.querySelectorAll('.katex-error')].map(element => element.textContent) };
    });
    // Save diagnostic measurements before asserting so an actual layout defect
    // has inspectable evidence even when this first pass fails.
    fs.writeFileSync(`${directory}/geometry-${width}.json`, JSON.stringify(geometry, null, 2));
    const headings = page.locator('.mutual-information-lesson > h2');
    for (let i = 0; i < await headings.count(); i += 1) {
      await headings.nth(i).evaluate(element => window.scrollTo(0, window.scrollY + element.getBoundingClientRect().top - 175));
      await page.screenshot({ path: `${directory}/reading-${i+1}-${width}.png` });
    }
    const practice = page.locator('.mi-practice');
    for (let i = 0; i < await practice.count(); i += 1) {
      await practice.nth(i).getByText('Show explained solution', { exact: true }).click();
    }
    await capture(practice.nth(4), 'practice-exponential-units');
    await capture(page.locator('.mi-inline').first(), 'nonlinear');
    await capture(page.locator('.mi-inline').last(), 'gaussian');
    assert.equal(errors.length, 0, errors.join('\n'));
    report.push({ width, checks, errors, requests, fonts, geometry, headings: await headings.count(), examples: await page.locator('.python-example').count(), independentPractice: await practice.count() });
    fs.writeFileSync(`${directory}/results.json`, JSON.stringify(report, null, 2));
    console.log(`Reviewed ${width}px: ${checks.length} grouped interaction checks; ${geometry.mathOverflow.length} equation overflow, ${geometry.outsideSvgText.length} clipped SVG labels, ${geometry.smallSvgText.length} small SVG labels; page ${geometry.contentWidth}/${geometry.pageWidth}.`);
    assert.equal(geometry.contentWidth, geometry.pageWidth, 'Page overflow');
    assert.equal(geometry.mathOverflow.length, 0, 'Equation overflow');
    assert.equal(geometry.outsideSvgText.length, 0, 'SVG label outside its viewBox');
    assert.equal(geometry.smallSvgText.length, 0, 'SVG labels too small');
    assert.equal(geometry.katexErrors.length, 0, 'KaTeX parse error');
    await page.close();
  }
  await browser.close();
})().catch(error => { console.error(error); process.exit(1); });

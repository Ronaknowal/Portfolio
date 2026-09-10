const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const directory = 'scratch/divergence-ipm-browser';
fs.mkdirSync(directory, { recursive: true });

async function readShot(page, target, name, width) {
  await target.evaluate(element => window.scrollTo(0, window.scrollY + element.getBoundingClientRect().top - 175));
  await page.screenshot({ path: directory + '/' + name + '-' + width + '.png' });
}

async function regionShot(page, target, name, width) {
  await page.addStyleTag({ content: '.learn-nav { visibility:hidden !important; }' });
  await target.screenshot({ path: directory + '/' + name + '-' + width + '.png' });
  await page.addStyleTag({ content: '.learn-nav { visibility:visible !important; }' });
}

(async () => {
  const { divergenceIpmExamples: examples } = await import('../src/learn/data/divergence-ipm-examples.js');
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
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/f-divergences-integral-probability-metrics?module=math-foundations', { waitUntil: 'networkidle' });
      const lesson = page.locator('.divergence-ipm-lesson');
      await lesson.waitFor();
      await page.evaluate(() => document.fonts.ready);
      const anchors = await lesson.locator('.lesson-intro a[href^="#"]').evaluateAll(elements => elements.map(element => ({
        href: element.getAttribute('href'), exists: !!document.getElementById(element.getAttribute('href').slice(1)),
      })));
      assert.equal(anchors.length, 10);
      assert.ok(anchors.every(anchor => anchor.exists));
      for (const example of await lesson.locator('.python-example').all()) {
        await example.evaluate(element => element.closest('details')?.setAttribute('open', ''));
        await example.scrollIntoViewIfNeeded();
        await page.waitForTimeout(100);
      }
      const programs = await lesson.locator('.python-example').evaluateAll(elements => elements.map(element => ({
        title: element.querySelector('h3').textContent,
        question: element.previousElementSibling.textContent.replace(/^Before running:\s*/, ''),
        preCount: [...element.children].filter(child => getComputedStyle(child).whiteSpace === 'pre').length,
      })));
      assert.equal(programs.length, 10);
      for (const program of programs) {
        const example = Object.values(examples).find(candidate => candidate.title === program.title);
        assert.ok(example);
        assert.equal(program.question, example.question);
        assert.equal(program.preCount, 2);
      }
      const headings = lesson.locator('h2');
      for (let index = 0; index < await headings.count(); index += 1) {
        await readShot(page, headings.nth(index), 'section-' + (index + 1), width);
      }
      const mechanisms = [];
      const ratio = page.getByRole('region', { name: 'Probability ratio investigation', exact: true });
      const ratioDefault = await ratio.locator('output').innerText();
      assert.ok(ratioDefault.includes('0.20847'));
      await ratio.getByLabel(/^Penalty rule/).selectOption('tv');
      assert.ok((await ratio.locator('output').innerText()).endsWith('0.3'));
      await ratio.getByLabel(/^Penalty rule/).selectOption('kl');
      await ratio.getByRole('button', { name: 'Remove Q’s support at A' }).click();
      assert.ok((await ratio.locator('output').innerText()).includes('∞'));
      await regionShot(page, ratio, 'ratio-support', width);
      await ratio.getByRole('button', { name: 'Reset ratios' }).click();
      await ratio.getByLabel('P weights', { exact: true }).fill('1, 0, 0, 0');
      assert.equal(await ratio.locator('output').innerText(), ratioDefault);
      await ratio.getByRole('button', { name: 'Apply weights', exact: true }).click();
      assert.ok((await ratio.locator('output').innerText()).includes('0.91629'));
      const appliedRatio = await ratio.locator('output').innerText();
      await ratio.getByLabel('P weights', { exact: true }).fill('1e-9999, 0, 0, 0');
      await ratio.getByRole('button', { name: 'Apply weights', exact: true }).click();
      assert.ok((await ratio.getByRole('alert').innerText()).includes('unchanged'));
      assert.equal(await ratio.locator('output').innerText(), appliedRatio);
      await ratio.getByRole('button', { name: 'Reset ratios' }).focus();
      await page.keyboard.press('Enter');
      assert.equal(await ratio.locator('output').innerText(), ratioDefault);
      await ratio.getByRole('button', { name: 'Swap P and Q' }).click();
      assert.ok((await ratio.locator('output').innerText()).includes('0.2343'));
      await ratio.getByRole('button', { name: 'Reset ratios' }).click();
      for (const kind of ['reverse', 'js', 'hellinger', 'chi', 'tv', 'kl']) {
        await ratio.getByLabel(/^Penalty rule/).selectOption(kind);
        assert.ok((await ratio.locator('output').innerText()).length > 10);
      }
      mechanisms.push('ratio: six generators; support infinity; draft/apply; rejected literal preserves state; swap; keyboard reset');

      const observer = page.getByRole('region', { name: 'Observable class investigation', exact: true });
      await observer.getByLabel('Allowed observer').selectOption('linear');
      assert.ok((await observer.locator('output').innerText()).endsWith('0'));
      await observer.getByLabel('Allowed observer').selectOption('lipschitz');
      assert.ok((await observer.locator('output').innerText()).endsWith('1.2'));
      await regionShot(page, observer, 'observer-lipschitz', width);
      await observer.getByLabel('Distribution pair').selectOption('shift');
      assert.ok((await observer.locator('output').innerText()).endsWith('2'));
      await observer.getByRole('button', { name: 'Reset observers' }).click();
      assert.ok((await observer.locator('output').innerText()).endsWith('0.6'));
      mechanisms.push('observer: zero linear gap versus positive TV/W1; changed pair; reset');

      const moving = page.getByRole('region', { name: 'Moving point mass investigation', exact: true });
      const h = moving.getByLabel(/^Displacement h:/);
      await h.focus();
      await page.keyboard.press('Home');
      assert.ok((await moving.locator('output').innerText()).includes('every listed comparison is zero'));
      await page.keyboard.press('ArrowRight');
      assert.ok((await moving.locator('output').innerText()).includes('h=0.01'));
      assert.ok((await moving.innerText()).includes('∞'));
      await regionShot(page, moving, 'moving-small-disjoint', width);
      await h.focus(); await page.keyboard.press('End');
      assert.equal(await h.inputValue(), '3');
      await moving.getByLabel(/^Gaussian kernel bandwidth:/).focus();
      await page.keyboard.press('Home');
      await moving.getByRole('button', { name: 'Reset moving mass' }).click();
      assert.equal(await h.inputValue(), '0.5');
      mechanisms.push('moving atoms: keyboard zero versus positive small displacement; extrema; bandwidth; reset');

      const kernel = page.getByRole('region', { name: 'Kernel witness investigation', exact: true });
      await kernel.getByLabel(/^Kernel/).selectOption('linear');
      assert.ok((await kernel.locator('output').innerText()).startsWith('Biased squared MMD 0;'));
      assert.ok(await kernel.getByLabel(/^Gaussian bandwidth/).isDisabled());
      await kernel.getByLabel(/^Kernel/).selectOption('quadratic');
      assert.ok((await kernel.locator('output').innerText()).startsWith('Biased squared MMD 9;'));
      await regionShot(page, kernel, 'kernel-two-features', width);
      await kernel.getByLabel(/^Kernel/).selectOption('rbf');
      await kernel.getByRole('button', { name: 'Identical two-point empirical laws' }).click();
      assert.ok((await kernel.locator('output').innerText()).includes('0; unbiased squared estimate -0.86466'));
      await regionShot(page, kernel, 'kernel-negative-unbiased', width);
      const appliedKernel = await kernel.locator('output').innerText();
      await kernel.getByLabel('X observations', { exact: true }).fill('4, 5');
      assert.equal(await kernel.locator('output').innerText(), appliedKernel);
      await kernel.getByRole('button', { name: 'Apply samples' }).click();
      assert.ok((await kernel.getByRole('alert').innerText()).includes('unchanged'));
      assert.equal(await kernel.locator('output').innerText(), appliedKernel);
      await kernel.getByLabel('X observations', { exact: true }).fill('-3, -1, 0');
      await kernel.getByLabel('Y observations', { exact: true }).fill('0.5, 2');
      await kernel.getByRole('button', { name: 'Apply samples' }).click();
      assert.equal(await kernel.getByRole('alert').count(), 0);
      assert.equal(await kernel.getByRole('region', { name: 'Signed kernel contribution matrix' }).locator('tbody tr').count(), 5);
      await kernel.getByRole('button', { name: 'Shifted samples' }).click();
      await kernel.getByLabel(/^Gaussian bandwidth/).focus(); await page.keyboard.press('End');
      const matrix = kernel.getByRole('region', { name: 'Signed kernel contribution matrix' });
      const beforeScroll = await matrix.evaluate(element => element.scrollLeft);
      await matrix.focus(); await page.keyboard.press('ArrowRight'); await page.waitForTimeout(200);
      const scroll = await matrix.evaluate(element => ({ left: element.scrollLeft, max: element.scrollWidth - element.clientWidth }));
      if (scroll.max > 0) assert.ok(scroll.left > beforeScroll);
      await kernel.getByRole('button', { name: 'Reset kernel comparison' }).click();
      assert.ok((await kernel.locator('output').innerText()).includes('0.4502'));
      mechanisms.push('kernel: linear blindness; quadratic feature gap; Gaussian witness; negative U; invalid preservation; unequal samples; keyboard Gram scroll; bandwidth; reset');

      const permutation = page.getByRole('region', { name: 'Exact permutation investigation', exact: true });
      const originalP = await permutation.locator('output').innerText();
      assert.ok(originalP.includes('2/70'));
      assert.ok(await permutation.getByRole('button', { name: 'Previous allocation' }).isDisabled());
      await permutation.getByRole('button', { name: 'Next allocation' }).click();
      assert.equal(await permutation.locator('output').innerText(), originalP);
      const allocation = permutation.getByLabel(/^Inspect allocation:/);
      await allocation.focus(); await page.keyboard.press('End');
      assert.equal(await allocation.inputValue(), '69');
      assert.ok(await permutation.getByRole('button', { name: 'Next allocation' }).isDisabled());
      await permutation.getByLabel(/^Prespecified bandwidth σ/).selectOption('0.3');
      assert.equal(await allocation.inputValue(), '0');
      await regionShot(page, permutation, 'permutation-small-bandwidth', width);
      await permutation.getByLabel(/^Prespecified bandwidth σ/).selectOption('3');
      await permutation.getByRole('button', { name: 'Reset permutations' }).click();
      assert.equal(await permutation.locator('output').innerText(), originalP);
      mechanisms.push('permutation: exact reference invariant to inspected split; keyboard final allocation; three bandwidths; reset');

      const critic = page.getByRole('region', { name: 'Variational divergence investigation', exact: true });
      const optimum = await critic.locator('output').innerText();
      assert.ok(optimum.includes('critic bound 0.20847'));
      await critic.getByRole('button', { name: 'Use a constant critic' }).click();
      assert.ok(!(await critic.locator('output').innerText()).includes('critic bound 0.20847'));
      await critic.getByRole('button', { name: 'Restore optimal critic' }).click();
      const offset = critic.getByLabel(/^Score offset:/);
      await offset.focus(); await page.keyboard.press('End');
      assert.ok((await critic.locator('output').innerText()).includes('critic bound -'));
      await regionShot(page, critic, 'critic-negative-bound', width);
      await critic.getByLabel(/^Log-ratio scale:/).focus(); await page.keyboard.press('End');
      await critic.getByRole('button', { name: 'Restore optimal critic' }).click();
      assert.equal(await critic.locator('output').innerText(), optimum);
      mechanisms.push('variational: exact optimum; constant restriction; negative suboptimal bound; keyboard range extremes; restore');

      const practice = lesson.locator('.divergence-practice');
      assert.equal(await practice.count(), 10);
      for (let index = 0; index < 10; index += 1) {
        for (const summary of await practice.nth(index).locator('summary').all()) {
          if (await summary.evaluate(element => element.parentElement.open)) await summary.evaluate(element => element.parentElement.removeAttribute('open'));
          await summary.focus(); await page.keyboard.press('Enter');
          assert.ok(await summary.evaluate(element => element.parentElement.open));
        }
      }
      await readShot(page, practice.last(), 'practice-complete-project', width);
      for (const details of await lesson.locator('details').all()) {
        if (!(await details.evaluate(element => element.open))) {
          await details.locator(':scope > summary').focus();
          await page.keyboard.press('Enter');
        }
        assert.ok(await details.evaluate(element => element.open));
      }
      for (let index = 0; index < await lesson.locator('.divergence-inline').count(); index += 1) {
        await regionShot(page, lesson.locator('.divergence-inline').nth(index), 'inline-figure-' + (index + 1), width);
      }
      const geometry = await page.evaluate(() => {
        const root = document.querySelector('.divergence-ipm-lesson');
        return {
          page: document.documentElement.scrollWidth,
          loadedOriginalFont: [...document.fonts].some(font => font.family === 'Space Grotesk' && font.status === 'loaded'),
          equationOverflow: [...root.querySelectorAll('.katex-display')].filter(element => element.scrollWidth > element.parentElement.clientWidth + 1).map(element => ({ text: element.textContent, width: element.scrollWidth, available: element.parentElement.clientWidth })),
          svgIssues: [...root.querySelectorAll('svg text')].flatMap(element => {
            if (element.hasAttribute('transform')) return [];
            const bounds = element.getBBox(), svg = element.ownerSVGElement, view = svg.viewBox.baseVal;
            const effectiveFont = parseFloat(getComputedStyle(element).fontSize) * svg.getBoundingClientRect().width / view.width;
            const outside = bounds.x < -1 || bounds.y < -1 || bounds.x + bounds.width > view.width + 1 || bounds.y + bounds.height > view.height + 1;
            return outside || effectiveFont < 13.5 ? [{ text: element.textContent, bounds: { x: bounds.x, y: bounds.y, width: bounds.width, height: bounds.height }, effectiveFont, outside }] : [];
          }),
          shortControls: [...root.querySelectorAll('.divergence-lab button, .divergence-lab input, .divergence-lab select')].filter(element => element.getBoundingClientRect().height < 43).map(element => element.outerHTML),
          clippedSelectLabels: [...root.querySelectorAll('.divergence-lab select')].flatMap(element => {
            const style = getComputedStyle(element), canvas = document.createElement('canvas'), context = canvas.getContext('2d');
            context.font = style.font;
            const needed = context.measureText(element.selectedOptions[0].textContent).width + parseFloat(style.paddingLeft) + parseFloat(style.paddingRight) + 20;
            return needed > element.clientWidth + 1 ? [{ label: element.selectedOptions[0].textContent, needed, available: element.clientWidth }] : [];
          }),
        };
      });
      results.push({ width, anchors, programs, mechanisms, practiceOpened: 20, geometry, errors, warnings, failedRequests });
      fs.writeFileSync(directory + '/results.json', JSON.stringify({ at: new Date().toISOString(), results }, null, 2));
      assert.equal(geometry.page, width);
      assert.ok(geometry.loadedOriginalFont);
      assert.equal(geometry.shortControls.length, 0);
      assert.equal(geometry.equationOverflow.length, 0);
      assert.equal(geometry.svgIssues.length, 0);
      assert.equal(geometry.clippedSelectLabels.length, 0);
      assert.deepEqual(errors, []);
      assert.deepEqual(warnings, []);
      assert.deepEqual(failedRequests, []);
      await page.close();
    }
  } finally {
    await browser.close();
  }
  console.log(JSON.stringify({ at: new Date().toISOString(), results }, null, 2));
})().catch(error => { console.error(error); process.exitCode = 1; });

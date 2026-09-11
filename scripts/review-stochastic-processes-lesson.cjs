const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const directory = 'scratch/stochastic-processes-browser';
fs.mkdirSync(directory, { recursive: true });

async function screenshot(page, target, name, width, full = false) {
  await target.evaluate(element => window.scrollTo(0, window.scrollY + element.getBoundingClientRect().top - 175));
  if (full) {
    await page.addStyleTag({ content: '.learn-nav { visibility:hidden !important; }' });
    await target.screenshot({ path: directory + '/' + name + '-' + width + '.png' });
    await page.addStyleTag({ content: '.learn-nav { visibility:visible !important; }' });
  } else {
    await page.screenshot({ path: directory + '/' + name + '-' + width + '.png' });
  }
}

(async () => {
  const { stochasticProcessesExamples: examples } = await import('../src/learn/data/stochastic-processes-examples.js');
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
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/stochastic-processes-markov-chains-brownian-motion-poisson?module=math-foundations', { waitUntil: 'networkidle' });
      const lesson = page.locator('.stochastic-processes-lesson');
      await lesson.waitFor();
      await page.evaluate(() => document.fonts.ready);
      const anchors = await lesson.locator('.lesson-intro a[href^="#"]').evaluateAll(elements => elements.map(element => ({
        href: element.getAttribute('href'), exists: !!document.getElementById(element.getAttribute('href').slice(1)),
      })));
      assert.equal(anchors.length, 10);
      assert.ok(anchors.every(anchor => anchor.exists));
      assert.equal(await lesson.locator('.katex-error').count(), 0);
      const programs = await lesson.locator('.python-example').evaluateAll(elements => elements.map(element => ({
        title: element.querySelector('h3').textContent,
        question: element.previousElementSibling.textContent.replace(/^Before running:\s*/, ''),
        preCount: [...element.children].filter(child => getComputedStyle(child).whiteSpace === 'pre').length,
      })));
      assert.equal(programs.length, 12);
      for (const program of programs) {
        const example = Object.values(examples).find(value => value.title === program.title);
        assert.ok(example);
        assert.equal(program.question, example.question);
        assert.equal(program.preCount, 2);
      }
      for (const target of await lesson.locator('.lesson-check details').all()) {
        await target.locator('summary').click();
        assert.equal(await target.getAttribute('open'), '');
      }
      assert.equal(await lesson.getByText('Try it independently.', { exact: true }).count(), 13);
      for (let i = 0; i < 10; i += 1) await screenshot(page, lesson.locator('h2').nth(i), 'reading-section-' + (i + 1), width);
      for (let i = 0; i < await lesson.locator('.process-inline').count(); i += 1) {
        await screenshot(page, lesson.locator('.process-inline').nth(i), 'inline-' + (i + 1), width, true);
      }

      let states = 0;
      const markov = page.getByRole('region', { name: 'Markov probability propagation', exact: true });
      assert.ok((await markov.innerText()).includes('0.8'));
      await markov.getByRole('button', { name: 'Next step', exact: true }).click();
      assert.ok((await markov.innerText()).includes('Sunny 0.64+0.06=0.7'));
      states += 1;
      for (const preset of ['weather', 'sticky', 'independent', 'alternating', 'identity', 'absorbing']) {
        await markov.getByLabel(/^Transition preset/).selectOption(preset);
        await markov.getByRole('button', { name: 'Next step', exact: true }).click();
        states += 1;
        if (preset === 'alternating') {
          assert.ok((await markov.innerText()).includes('nonstationary start alternates forever'));
          await markov.getByLabel(/^Initial sunny probability/).selectOption('0.5');
          assert.ok((await markov.innerText()).includes('mass 0.5'));
          await screenshot(page, markov, 'markov-periodic', width, true);
        }
        if (preset === 'identity') assert.ok((await markov.innerText()).includes('Every distribution is stationary'));
      }
      await markov.getByRole('button', { name: 'Reset weather', exact: true }).click();
      await markov.getByLabel('Sunny → Rainy, a', { exact: true }).fill('1e-9999');
      await markov.getByRole('button', { name: 'Apply transition edits' }).click();
      assert.ok((await markov.getByRole('alert').innerText()).includes('prior applied matrix remains active'));
      assert.ok((await markov.innerText()).includes('a=0.2'));
      await markov.getByRole('button', { name: 'Reset weather', exact: true }).focus();
      await page.keyboard.press('Enter');
      assert.equal(await markov.getByRole('alert').count(), 0);
      await markov.getByRole('button', { name: 'Previous step', exact: true }).click();
      assert.ok((await markov.innerText()).includes('No transition has happened'));
      assert.ok(await markov.getByRole('button', { name: 'Previous step', exact: true }).isDisabled());
      states += 3;

      const reserve = page.getByRole('region', { name: 'First passage and absorption', exact: true });
      assert.ok((await reserve.innerText()).includes('Lower boundary: 0.375'));
      await reserve.getByLabel(/^Upward probability/).selectOption('0.25');
      assert.ok((await reserve.innerText()).includes('eventual success 0.1'));
      assert.ok((await reserve.innerText()).includes('boundary 3.2 steps'));
      await reserve.getByLabel(/^Upper reserve boundary/).selectOption('8');
      await reserve.getByLabel(/^Starting reserve/).selectOption('8');
      assert.ok((await reserve.innerText()).includes('eventual success 1'));
      await reserve.getByLabel(/^Inspect first-step/).selectOption('8');
      assert.ok((await reserve.innerText()).includes('already a boundary'));
      await screenshot(page, reserve, 'reserve-boundary', width, true);
      await reserve.getByRole('button', { name: 'Reset reserve walk' }).click();
      states += 4;

      const arrival = page.getByRole('region', { name: 'Arrival clocks and event counts', exact: true });
      const eventBefore = await arrival.getByLabel(/^Inspect arrival/).locator('option').allTextContents();
      await arrival.getByLabel(/^Routing rule/).selectOption('alternating');
      assert.deepEqual(await arrival.getByLabel(/^Inspect arrival/).locator('option').allTextContents(), eventBefore);
      assert.ok(await arrival.getByLabel(/^Independent probability/).isDisabled());
      assert.ok((await arrival.innerText()).includes('not Poisson'));
      await arrival.getByLabel(/^Arrival rate preset/).selectOption('0,0');
      assert.ok((await arrival.innerText()).includes('observed count 0'));
      assert.ok(await arrival.getByLabel(/^Inspect arrival/).isDisabled());
      await arrival.getByLabel(/^Arrival rate preset/).selectOption('0,4');
      assert.ok((await arrival.innerText()).includes('mean count 0'));
      await arrival.getByLabel(/^Count interval/).selectOption('2,3');
      assert.ok((await arrival.innerText()).includes('mean count 4'));
      await arrival.getByLabel('Rate before minute 2', { exact: true }).fill('bad');
      await arrival.getByRole('button', { name: 'Apply rate edits' }).click();
      assert.ok(await arrival.getByRole('alert').count());
      assert.ok((await arrival.innerText()).includes('mean count 4'));
      await arrival.getByRole('button', { name: 'Reset arrival clock' }).click();
      await arrival.getByLabel(/^Arrival rate preset/).selectOption('1,4');
      await arrival.getByLabel(/^Count interval/).selectOption('0,3');
      assert.ok((await arrival.innerText()).includes('mean count 6'));
      await screenshot(page, arrival, 'arrival-schedule', width, true);
      await arrival.getByRole('button', { name: 'Draw another event stream' }).click();
      await arrival.getByRole('button', { name: 'Reset arrival clock' }).focus();
      await page.keyboard.press('Enter');
      assert.deepEqual(await arrival.getByLabel(/^Inspect arrival/).locator('option').allTextContents(), eventBefore);
      states += 8;

      const clock = page.getByRole('region', { name: 'Continuous time holding clocks', exact: true });
      await clock.getByLabel(/^Initial device state/).selectOption('1');
      await clock.getByLabel(/^Observation horizon/).selectOption('12');
      await clock.getByLabel(/^On → Off rate/).fill('1');
      await clock.getByLabel(/^Off → On rate/).fill('3');
      await clock.getByRole('button', { name: 'Apply holding rates' }).click();
      assert.ok((await clock.innerText()).includes('equilibrium 0.75'));
      await screenshot(page, clock, 'holding-time-vs-visits', width, true);
      await clock.getByLabel(/^On → Off rate/).fill('0');
      await clock.getByRole('button', { name: 'Apply holding rates' }).click();
      assert.ok(await clock.getByRole('alert').count());
      assert.ok((await clock.innerText()).includes('equilibrium 0.75'));
      await clock.getByRole('button', { name: 'Reset holding clock' }).click();
      await clock.getByRole('button', { name: 'Draw another holding path' }).click();
      states += 5;

      const brownian = page.getByRole('region', { name: 'Brownian paths and coupled refinement', exact: true });
      const endpoint = (await brownian.innerText()).match(/endpoint X\(T\)=([^ ]+)/)[1];
      for (const level of ['2', '3', '4', '5', '6', '7', '8']) {
        await brownian.getByLabel(/^Number of grid intervals/).selectOption({ value: level });
        assert.equal((await brownian.innerText()).match(/endpoint X\(T\)=([^ ]+)/)[1], endpoint);
        states += 1;
      }
      await brownian.getByLabel(/^Diffusion σ/).selectOption('2');
      await brownian.getByLabel(/^Drift μ/).selectOption('0.5');
      await brownian.getByLabel(/^Brownian horizon/).selectOption('2');
      await brownian.getByLabel(/^Highlighted Brownian path/).selectOption('3');
      await brownian.getByLabel(/^Inspect Brownian increment/).selectOption('255');
      await screenshot(page, brownian, 'brownian-refined', width, true);
      await brownian.getByRole('button', { name: 'Reset Brownian paths' }).focus();
      await page.keyboard.press('Enter');
      assert.equal((await brownian.innerText()).match(/endpoint X\(T\)=([^ ]+)/)[1], endpoint);
      states += 5;

      const geometry = await lesson.evaluate(element => ({
        documentWidth: document.documentElement.scrollWidth,
        fonts: [...document.fonts].some(font => font.family === 'Space Grotesk' && font.status === 'loaded'),
        equations: [...element.querySelectorAll('.katex-display')].map(node => ({
          width: node.getBoundingClientRect().width, content: node.scrollWidth,
          tex: node.querySelector('annotation')?.textContent,
        })),
        svgOverflow: [...element.querySelectorAll('.process-figure')].flatMap(svg =>
          [...svg.querySelectorAll('text')].filter(node => {
            const box = node.getBBox();
            return box.x < -0.5 || box.x+box.width > svg.viewBox.baseVal.width+0.5;
          }).map(node => node.textContent)),
        controls: [...element.querySelectorAll('.process-lab button,.process-lab input,.process-lab select')]
          .map(node => ({ text: node.textContent, height: node.getBoundingClientRect().height })),
      }));
      fs.writeFileSync(directory + '/geometry-' + width + '.json', JSON.stringify(geometry, null, 2));
      assert.equal(geometry.documentWidth, width);
      assert.ok(geometry.fonts);
      assert.deepEqual(errors, []);
      assert.deepEqual(warnings, []);
      assert.deepEqual(failedRequests, []);
      assert.ok(geometry.controls.every(control => control.height >= 43));
      assert.ok(geometry.equations.every(equation => equation.content <= equation.width + 1));
      assert.deepEqual(geometry.svgOverflow, []);
      results.push({ width, states, anchors: anchors.length, programs, independentPractice: 13,
        geometry, errors, warnings, failedRequests });
      fs.writeFileSync(directory + '/in-progress-results.json', JSON.stringify(results, null, 2));
      await page.close();
    }
  } finally { await browser.close(); }
  const report = { checkedAt: new Date().toISOString(), results };
  fs.writeFileSync(directory + '/results.json', JSON.stringify(report, null, 2));
  console.log(JSON.stringify(report, null, 2));
})().catch(error => { console.error(error); process.exitCode = 1; });

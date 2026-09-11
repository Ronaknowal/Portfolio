const {
  chromium
} = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('fs');
const path = require('path');
const assert = require('assert/strict');
const output = path.resolve('scratch/queueing-browser');
fs.mkdirSync(output, {
  recursive: true
});
(async () => {
  const {
    queueingExamples: examples
  } = await import('../src/learn/data/queueing-examples.js');
  const m = await import('../src/learn/data/queueing-models.js');
  const browser = await chromium.launch({
    channel: 'msedge',
    headless: true
  });
  const records = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({
        viewport: {
          width,
          height: 1050
        },
        reducedMotion: 'reduce'
      });
      const errors = [];
      page.on('pageerror', e => errors.push(e.message));
      page.on('console', e => {
        if (['error', 'warning'].includes(e.type())) errors.push(e.text());
      });
      page.on('requestfailed', r => errors.push(r.url() + ': ' + r.failure().errorText));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/queueing-theory-m-m-1-m-g-1-little-s-law');
      const lesson = page.locator('.queueing-lesson');
      await lesson.locator('h2').last().waitFor();
      await page.evaluate(() => document.fonts.ready);
      let states = 0;
      const region = name => lesson.getByRole('region', {
        name,
        exact: true
      });
      const slider = async (parent, name, value) => parent.getByRole('slider', {
        name,
        exact: true
      }).fill(String(value));
      const metric = async (parent, name) => parent.locator('.queueing-metrics > div').filter({
        has: page.locator('dt', {
          hasText: name
        })
      }).locator('dd').innerText();
      async function shot(locator, name) {
        await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(n => n.style.visibility = 'hidden'));
        await locator.screenshot({
          path: path.join(output, `${name}-${width}.png`)
        });
        await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(n => n.style.visibility = ''));
      }
      const anchors = await lesson.locator('nav a').evaluateAll(nodes => nodes.map(node => ({
        href: node.hash,
        present: !!document.getElementById(node.hash.slice(1))
      })));
      assert.equal(anchors.length, 10);
      assert(anchors.every(a => a.present), JSON.stringify(anchors));
      await lesson.locator(':scope > details').evaluateAll(nodes => nodes.forEach(n => n.open = true));
      assert.equal(await lesson.locator('.python-example').count(), 10);
      for (const example of examples) {
        const block = lesson.locator('.python-example').filter({
          has: page.getByRole('heading', {
            name: example.title,
            exact: true
          })
        });
        const text = (await block.innerText()).replace(/\r\n/g, '\n');
        assert(text.includes(example.code.replace(/\r\n/g, '\n').trim()), example.id);
        assert(text.includes(example.expected.trim()), example.id);
        assert(await block.evaluate(node => node.previousElementSibling.textContent.includes('Before running:')));
      }
      await lesson.locator(':scope > details').evaluateAll(nodes => nodes.forEach(n => n.open = false));
      const checks = lesson.locator('section.lesson-check, div.lesson-check');
      assert.equal(await checks.count(), 12);
      for (const check of await checks.all()) {
        assert((await check.locator('p').first().innerText()).length > 60);
        for (const detail of await check.locator('details').all()) {
          await detail.locator('summary').click();
          assert((await detail.innerText()).length > 65);
          await detail.locator('summary').click();
        }
      }
      const area = region('Occupancy area investigation');
      for (const boundary of ['system', 'queue']) for (const horizon of [1, 2.5, 5, 7, 8]) {
        await area.getByRole('combobox', {
          name: 'Population boundary'
        }).selectOption(boundary);
        await slider(area, 'Observation horizon', horizon);
        const state = m.occupancyWindow(m.queueExampleJobs, horizon, boundary);
        assert.equal(await metric(area, 'Mean population'), `${m.queueNumber(state.meanOccupancy)} jobs`);
        assert.equal(await metric(area, 'Step area = clipped job sum'), `${m.queueNumber(state.stepArea)} = ${m.queueNumber(state.clippedArea)} job·s`);
        states++;
      }
      await area.getByRole('button', {
        name: 'Reset area'
      }).click();
      await slider(area, 'Observation horizon', 5);
      await area.getByRole('button', {
        name: 'Job B',
        exact: true
      }).click();
      assert.equal(await area.locator('tr.queueing-selected th').innerText(), 'B');
      await shot(area, 'area-censored');
      await area.getByRole('button', {
        name: 'Reset area'
      }).click();
      const load = region('Stationary load investigation');
      for (const arrival of [.5, 4, 8, 9.5, 10, 11]) {
        await slider(load, 'M/M/1 arrival rate', arrival);
        if (arrival < 10) {
          const state = m.mm1State(arrival, 10);
          assert.equal(await metric(load, 'Mean total W'), `${m.queueNumber(state.meanTotal)} s`);
          const bars = await load.locator('svg rect').evaluateAll(nodes => nodes.map(n => Number(n.getAttribute('height')) / 150));
          assert(Math.abs(bars.reduce((a, b) => a + b, 0) - 1) < 1e-12);
        } else assert((await load.getByRole('alert').innerText()).includes('no stationary'));
        states++;
      }
      await shot(load, 'unstable-load');
      await load.getByRole('button', {
        name: 'Reset load'
      }).click();
      const tails = region('Waiting and response tails investigation');
      for (const arrival of [.1, .2, .5, .9, .95]) for (const percentile of [.5, .9, .99]) {
        await slider(tails, 'Tail arrival rate', arrival);
        await tails.getByRole('combobox', {
          name: 'Duration percentile'
        }).selectOption(String(percentile));
        const state = m.mm1State(arrival, 1, percentile);
        assert.equal(await metric(tails, `${percentile * 100}th wait percentile`), `${m.queueNumber(state.waitQuantile)} s`);
        assert.equal(await metric(tails, `${percentile * 100}th total percentile`), `${m.queueNumber(state.totalQuantile)} s`);
        states++;
      }
      await tails.getByRole('button', {
        name: 'Reset tails'
      }).click();
      await shot(tails, 'tails');
      const variability = region('Service inspection and residual work investigation');
      for (const kind of Object.keys(m.serviceMixtureNames)) for (const arrival of [.5, 8, 9.5]) {
        await variability.getByRole('combobox', {
          name: 'Service distribution'
        }).selectOption(kind);
        await slider(variability, 'M/G/1 arrival rate', arrival);
        const state = m.serviceVariabilityState(arrival, kind);
        assert.equal(await metric(variability, 'Mean queue wait'), `${m.queueNumber(state.meanWait)} s`);
        if (kind === 'exponential') assert.equal(await variability.locator('.queueing-strip').count(), 0);else {
          const strips = await variability.locator('.queueing-strip').evaluateAll(nodes => nodes.map(n => [...n.children].map(c => c.getBoundingClientRect().width / n.getBoundingClientRect().width)));
          assert(Math.abs(strips[1].at(-1) - state.atoms.at(-1).busyShare) < .012);
        }
        states++;
      }
      await variability.getByRole('combobox', {
        name: 'Service distribution'
      }).selectOption('rareLong');
      await slider(variability, 'M/G/1 arrival rate', 8);
      await shot(variability, 'rare-long');
      await variability.getByRole('button', {
        name: 'Reset variability'
      }).click();
      const pool = region('Pooling comparison investigation');
      for (const c of [1, 2, 4]) for (const arrival of [1, 8, 14]) {
        await slider(pool, 'Number of workers', c);
        await slider(pool, 'Pooled arrival rate', arrival);
        const state = m.pooledQueueState(arrival, 5, c);
        if (state.stable) assert.equal(await metric(pool, 'Common-queue total mean'), `${m.queueNumber(state.meanTotal)} s`);else assert((await pool.getByRole('alert').innerText()).includes('unavailable'));
        states++;
      }
      await pool.getByRole('button', {
        name: 'Reset pooling'
      }).click();
      const finite = region('Finite capacity and admission investigation');
      for (const K of [1, 2, 3, 6]) for (const arrival of [1, 10, 12, 20]) {
        await slider(finite, 'Total system capacity', K);
        await slider(finite, 'Offered arrival rate', arrival);
        const state = m.finiteBufferState(arrival, 10, K);
        assert.equal(await metric(finite, 'Admitted-customer queue wait'), `${m.queueNumber(state.meanWait)} s`);
        assert.equal(await finite.locator('.queueing-state-list>div').count(), K + 1);
        states++;
      }
      await finite.getByRole('button', {
        name: 'Reset admission'
      }).click();
      await shot(finite, 'admission');
      let keyboard = 0,
        scrolls = 0;
      for (const control of await lesson.locator('.queueing-lab input,.queueing-lab select,.queueing-lab button').all()) {
        await control.focus();
        assert(await control.evaluate(n => n === document.activeElement));
        keyboard++;
      }
      const horizon = area.getByRole('slider', {
        name: 'Observation horizon'
      });
      await horizon.focus();
      await horizon.press('ArrowLeft');
      assert.equal(await horizon.inputValue(), '6.75');
      await area.getByRole('button', {
        name: 'Reset area'
      }).focus();
      await page.keyboard.press('Enter');
      assert.equal(await horizon.inputValue(), '7');
      for (const scroller of await lesson.locator('.queueing-scroll,.queueing-table-wrap').all()) {
        if (await scroller.evaluate(n => n.scrollWidth > n.clientWidth + 2)) {
          await scroller.focus();
          await scroller.press('ArrowRight');
          await page.waitForTimeout(150);
          assert(await scroller.evaluate(n => n.scrollLeft > 0));
          await scroller.evaluate(n => n.scrollLeft = 0);
          scrolls++;
        }
      }
      await lesson.locator(':scope > details').evaluateAll(nodes => nodes.forEach(n => n.open = true));
      const geometry = await page.evaluate(() => ({
        pageWidth: document.documentElement.scrollWidth,
        viewport: innerWidth,
        errors: document.querySelectorAll('.queueing-lesson .katex-error').length,
        math: [...document.querySelectorAll('.queueing-lesson .katex-display')].map(n => ({
          scroll: n.scrollWidth,
          client: n.clientWidth
        })),
        font: getComputedStyle(document.querySelector('.queueing-lesson p')).fontFamily,
        svgLabels: [...document.querySelectorAll('.queueing-lesson svg text')].map(n => parseFloat(getComputedStyle(n).fontSize) * n.getScreenCTM().a)
      }));
      assert(geometry.pageWidth <= width + 1, JSON.stringify(geometry));
      assert.equal(geometry.errors, 0);
      assert(geometry.math.every(x => x.scroll <= x.client + 3), JSON.stringify(geometry.math));
      assert(geometry.svgLabels.every(size => size >= 14), Math.min(...geometry.svgLabels));
      await lesson.locator(':scope > details').evaluateAll(nodes => nodes.forEach(n => n.open = false));
      for (const section of [0, 2, 5, 7, 9]) {
        const heading = lesson.locator('h2').nth(section);
        await heading.evaluate(node => window.scrollTo({
          top: node.getBoundingClientRect().top + scrollY - 90,
          behavior: 'instant'
        }));
        await page.evaluate(() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r))));
        await page.waitForTimeout(200);
        await page.screenshot({
          path: path.join(output, `reading-${section + 1}-${width}.png`)
        });
      }
      await shot(lesson.locator('.lesson-sources'), 'sources');
      assert.deepEqual(errors, []);
      records.push({
        width,
        states,
        anchors: anchors.length,
        examples: 10,
        practiceAndCheckpoints: 12,
        keyboardControls: keyboard,
        keyboardScrolls: scrolls,
        geometry,
        errors
      });
      console.log('Passed width', width, 'states', states);
      await page.close();
    }
    fs.writeFileSync(path.join(output, 'results.json'), JSON.stringify({
      at: new Date().toISOString(),
      passed: true,
      records
    }, null, 2));
  } finally {
    await browser.close();
  }
})().catch(error => {
  console.error(error);
  process.exit(1);
});

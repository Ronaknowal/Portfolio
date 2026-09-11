const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs'), assert = require('node:assert/strict'), crypto = require('node:crypto');
const directory = 'scratch/linked-extension-independent/browser';
fs.mkdirSync(directory, { recursive: true });
const hash = path => crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex');
(async () => {
  const native = JSON.parse(fs.readFileSync('scratch/linked-extension-independent/native-results.json'));
  const sources = native.sources.map(({ path, sha256 }) => ({ path, sha256 }));
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [], images = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1080 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => { if (['error','warning'].includes(message.type()) && !message.text().startsWith('[vite]')) errors.push(message.text()); });
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/linked-lists-stacks-queues?module=data-structures-algorithms', { waitUntil: 'domcontentloaded', timeout: 60000 });
      const lesson = page.locator('.lesson-pilot').first(); await lesson.waitFor({ timeout: 60000 });
      await page.evaluate(() => document.fonts.ready);
      assert(await page.evaluate(() => document.fonts.check('16px "Space Grotesk"')));
      let states = 0;
      const capture = async (locator, name) => {
        await locator.first().evaluate(node => scrollTo({ top: scrollY + node.getBoundingClientRect().top - 112, behavior: 'instant' }));
        await page.waitForTimeout(100);
        const bounds = await locator.first().boundingBox();
        assert(bounds && bounds.y > 50 && bounds.y < 700);
        const path = `${directory}/${name}-${width}.png`; await page.screenshot({ path });
        images.push({ path, sha256: hash(path), opened: false });
      };
      const range = async (locator, value) => {
        await locator.focus(); await page.keyboard.press('Home');
        for (let index = 0; index < value; index += 1) await page.keyboard.press('ArrowRight');
        assert.equal(await locator.inputValue(), String(value)); states += 1;
      };
      const finish = async button => {
        let count = 0;
        while (await button.isEnabled()) { await button.click(); if (++count > 100) throw Error('Unexpected unbounded trace'); }
        states += 1;
      };
      assert.equal(await lesson.locator('h2').count(), 14);
      assert.deepEqual(await lesson.locator('nav a').evaluateAll(nodes => nodes.map(node => node.hash).filter(hash => !document.getElementById(hash.slice(1)))), []);
      for (const index of [8, 9, 10, 11, 12]) await capture(lesson.locator('h2').nth(index), `reading-${index+1}`);
      const cycle = lesson.locator('[data-traversal-lab="cycle"]');
      await range(cycle.getByLabel('Cycle node count', { exact: true }), 9);
      await cycle.getByLabel('Cycle tail target', { exact: true }).selectOption('7'); states += 1;
      const nextCycle = cycle.getByRole('button', { name: 'Next cycle step', exact: true });
      while (!(await cycle.locator('.traversal-state').innerText()).includes('meeting')) await nextCycle.click();
      assert((await cycle.innerText()).includes('slow → n8; fast → n8')); states += 1;
      await capture(cycle.locator('.traversal-state'), 'changed-cycle-meeting');
      await finish(nextCycle);
      assert((await cycle.getByRole('status').innerText()).includes('entry n7'));
      assert((await cycle.getByRole('status').innerText()).includes('Prefix length 7; cycle length 2'));
      const cycleRegion = cycle.getByRole('region');
      if (width < 800) {
        await cycleRegion.focus(); await page.keyboard.press('ArrowRight'); await page.waitForTimeout(150);
        assert(await cycleRegion.evaluate(node => node.scrollLeft > 0));
        await cycleRegion.evaluate(node => { node.scrollLeft = node.scrollWidth - node.clientWidth; });
      }
      await capture(cycle.locator('.traversal-state'), 'changed-cycle-entry');
      await cycle.getByLabel('All values equal 7', { exact: true }).focus(); await page.keyboard.press('Space');
      assert((await cycle.getByRole('status').innerText()).includes('entry n7')); states += 1;
      await cycle.getByRole('button', { name: 'Reset cycle', exact: true }).click(); states += 1;

      const middle = lesson.locator('[data-traversal-lab="middle"]');
      await range(middle.getByLabel('Middle chain length', { exact: true }), 8);
      assert((await middle.getByRole('status').innerText()).includes('returns n4'));
      await middle.getByLabel('Show the left-heavy cut', { exact: true }).focus(); await page.keyboard.press('Space');
      assert((await middle.innerText()).includes('Return left head n0 and right head n4: 4 and 4')); states += 1;
      await middle.getByLabel('Middle policy', { exact: true }).selectOption('first');
      assert((await middle.getByRole('status').innerText()).includes('returns n3'));
      assert((await middle.innerText()).includes('right head n4: 4 and 4')); states += 1;
      if (width < 800) await middle.getByRole('region').nth(1).evaluate(node => { node.scrollLeft = 190; });
      await capture(middle.getByRole('status'), 'changed-even-cut');
      await range(middle.getByLabel('Middle chain length', { exact: true }), 7);
      assert((await middle.innerText()).includes('right head n4: 4 and 3'));
      await middle.getByRole('button', { name: 'Reset middle and split', exact: true }).click(); states += 1;

      const greater = lesson.locator('[data-monostack-lab="greater"]');
      await greater.getByLabel('Next-greater readings', { exact: true }).fill('-5, -5, 0, -1, 0, 1');
      await greater.getByRole('button', { name: 'Apply readings', exact: true }).click(); states += 1;
      const nextStack = greater.getByRole('button', { name: 'Next stack event', exact: true });
      while (!(await greater.getByRole('status').innerText()).includes('Index 1 is answered by 2')) await nextStack.click();
      await capture(greater.getByRole('region').first(), 'signed-resolution');
      await finish(nextStack);
      assert.deepEqual(await greater.locator('tbody td').allTextContents(), ['2','1','3','1','1','0']);
      await greater.getByLabel('Future comparison', { exact: true }).selectOption('inclusive'); states += 1;
      await greater.getByRole('button', { name: 'Finish future search', exact: true }).click(); states += 1;
      assert.deepEqual(await greater.locator('tbody td').allTextContents(), ['1','1','2','1','1','0']);
      await capture(greater.locator('.monostack-memory'), 'changed-inclusive-result');
      const distances = await greater.locator('tbody td').allTextContents();
      await greater.getByLabel('Next-greater readings', { exact: true }).fill('1,,2');
      await greater.getByRole('button', { name: 'Apply readings', exact: true }).click();
      assert(await greater.getByRole('alert').isVisible()); assert.deepEqual(await greater.locator('tbody td').allTextContents(), distances); states += 1;

      const histogram = lesson.locator('[data-monostack-lab="histogram"]');
      await histogram.getByLabel('Histogram heights', { exact: true }).fill('0, 3, 3, 1, 4, 4, 0');
      await histogram.getByRole('button', { name: 'Apply heights', exact: true }).click(); states += 1;
      await histogram.getByRole('button', { name: 'Finish boundary scan', exact: true }).click();
      assert.deepEqual(await histogram.locator('table').first().locator('tbody td').allTextContents(), ['-1','0','0','0','3','3','-1']); states += 1;
      await histogram.getByLabel('Smaller boundary direction', { exact: true }).selectOption('right');
      await histogram.getByRole('button', { name: 'Finish boundary scan', exact: true }).click();
      assert.deepEqual(await histogram.locator('table').first().locator('tbody td').allTextContents(), ['7','3','3','6','6','6','7']); states += 1;
      for (const [index, area] of [['1',6],['3',5],['4',8]]) {
        await histogram.getByLabel('Limiting histogram bar', { exact: true }).selectOption(index);
        assert((await histogram.locator('.monostack-candidate').innerText()).endsWith(`= ${area}.`)); states += 1;
      }
      assert((await histogram.innerText()).includes('Largest area: 8'));
      const geometry = histogram.getByRole('region', { name: 'Chosen histogram rectangle and its excluded smaller boundary bars', exact: true });
      const rect = geometry.locator('[data-rectangle="candidate"]');
      assert.equal(await rect.getAttribute('width'), '104');
      assert.equal(await rect.getAttribute('height'), '145');
      const zeroBars = geometry.locator('[data-bar="0"] rect, [data-bar="6"] rect');
      assert.deepEqual(await zeroBars.evaluateAll(nodes => nodes.map(node => Number(node.getAttribute('height')))), [0,0]);
      if (width < 800) await geometry.evaluate(node => { node.scrollLeft = node.scrollWidth - node.clientWidth; });
      await capture(geometry, 'changed-histogram-geometry');
      await histogram.getByRole('button', { name: 'Reset boundaries', exact: true }).click(); states += 1;

      for (const task of await lesson.locator('.linked-extension-practice').all()) for (const summary of await task.locator('summary').all()) {
        await summary.focus(); await page.keyboard.press('Enter'); assert(await summary.evaluate(node => node.parentElement.open));
      }
      await capture(lesson.locator('.linked-extension-practice').nth(1), 'alias-practice');
      await capture(lesson.locator('.linked-extension-practice').last(), 'tied-rectangle-practice');
      const examples = (await import('../src/learn/data/linked-traversal-examples.js')).linkedTraversalExamples;
      const stackExamples = (await import('../src/learn/data/monotonic-stack-examples.js')).monotonicStackExamples;
      const displayed = await lesson.locator('.python-example').allTextContents();
      for (const example of [...Object.values(examples), ...Object.values(stackExamples)]) {
        const found = displayed.find(text => text.includes(example.code.trim())); assert(found && found.includes(example.output.trim()));
      }
      assert.equal(await lesson.locator('.dsa-practice__problem').count(), 14);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 2), false);
      assert.deepEqual(errors, []);
      results.push({ width, states, actualFonts: true, changedCycle: '9 nodes, prefix7/cycle2, meetingn8/entryn7', changedCut: '8→4+4 and7→4+3', signedFuture: true, rectangleHeightAndWidth: true, codeOutputs: 4, practicePlacements: 14, keyboardSolutions: 8, errors, documentOverflow: false });
      await page.close();
    }
    assert.deepEqual(sources, sources.map(({path}) => ({path,sha256:hash(path)})));
    fs.writeFileSync(`${directory}/results.json`, JSON.stringify({ checkedAt: new Date().toISOString(), passed: true, sources, results, images }, null, 2));
    console.log(JSON.stringify(results, null, 2));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

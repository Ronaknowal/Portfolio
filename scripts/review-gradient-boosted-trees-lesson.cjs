const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const { pathToFileURL } = require('node:url');
const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const directory = path.resolve('scratch/gradient-boosted-trees-verification/browser');
fs.mkdirSync(directory, { recursive: true });
const normalize = text => text.replace(/\s+/g, ' ').trim();
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const sources = [
  'src/learn/data/topics/gradient-boosted-trees-xgboost-lightgbm-catboost.jsx',
  'src/learn/data/gradient-boosted-trees-models.js',
  'src/learn/data/gradient-boosted-trees-examples.js',
  'src/learn/components/lesson-labs/GradientBoostedTreeLabs.jsx',
  'src/learn/components/lesson-labs/gradient-boosted-trees-labs.css',
  'src/learn/data/curriculum/blueprints/gradient-boosted-trees-xgboost-lightgbm-catboost.js',
];

(async () => {
  const startedAt = new Date().toISOString();
  const sourceHashes = sources.map(file => ({ path: file, sha256: hash(file) }));
  const { gradientBoostedTreeExamples: examples } = await import(pathToFileURL(path.resolve(sources[2])));
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1050 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      const record = { width, states: [], screenshots: [], errors: [], failedRequests: [], keyboard: [], programs: [] };
      page.on('pageerror', error => record.errors.push(String(error)));
      page.on('console', message => { if (message.type() === 'error' && !message.text().includes('[vite]')) record.errors.push(message.text()); });
      page.on('requestfailed', request => record.failedRequests.push({ url: request.url(), failure: request.failure() }));
      try {
        await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/gradient-boosted-trees-xgboost-lightgbm-catboost?module=classical-ml', { waitUntil: 'domcontentloaded' });
        const lesson = page.locator('.gradient-boosted-trees-lesson');
        await lesson.waitFor();
        await page.evaluate(() => document.fonts.ready);
        record.fonts = await page.evaluate(() => [...document.fonts].filter(font => font.status === 'loaded').map(font => font.family));
        assert(record.fonts.some(font => font.includes('Space Grotesk')));
        assert(record.fonts.some(font => font.includes('JetBrains Mono')));
        await page.addStyleTag({ content: 'html{scroll-behavior:auto!important}' });
        const lab = name => lesson.locator(`[data-gbt-lab="${name}"]`);
        async function shot(target, name) {
          await target.evaluate(node => scrollTo({ top: node.getBoundingClientRect().top + scrollY - 82, behavior: 'instant' }));
          await page.evaluate(() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))));
          const file = path.join(directory, `${name}-${width}.png`);
          await page.screenshot({ path: file });
          record.screenshots.push({ path: path.relative(process.cwd(), file).replaceAll('\\', '/'), sha256: hash(file) });
        }
        async function range(region, label, value) {
          const input = region.getByLabel(label, { exact: true });
          await input.fill(String(value));
          await input.dispatchEvent('input');
        }
        async function state(region, name, expected) {
          await page.evaluate(() => new Promise(resolve => requestAnimationFrame(resolve)));
          const text = normalize(await region.innerText());
          if (expected) assert(text.includes(expected), `${name}: ${text}`);
          const geometry = await region.locator('svg').evaluateAll(nodes => nodes.map(svg => {
            const bounds = svg.getBoundingClientRect();
            const labels = [...svg.querySelectorAll('text')].filter(label => {
              const box = label.getBoundingClientRect();
              return box.left < bounds.left - 2 || box.right > bounds.right + 2 || box.top < bounds.top - 2 || box.bottom > bounds.bottom + 2;
            }).map(label => label.textContent);
            const points = [...svg.querySelectorAll('polyline')].flatMap(line => [...line.points].map(point => [point.x, point.y]));
            const view = svg.viewBox.baseVal;
            return { labels, pointCount: points.length, pointsInside: points.every(([x, y]) => x >= 44 - 1e-6 && x <= view.width - 14 + 1e-6 && y >= 20 - 1e-6 && y <= 192 + 1e-6) };
          }));
          assert(geometry.every(item => item.labels.length === 0 && item.pointsInside), `${name}: ${JSON.stringify(geometry)}`);
          if (await region.getAttribute('data-gbt-lab') === 'validation') {
            const stepShape = await region.locator('figure').nth(1).locator('polyline').evaluate(node => {
              const points = [...node.points];
              return points.slice(1).every((point, index) => Math.abs(point.x-points[index].x) < 1e-8 || Math.abs(point.y-points[index].y) < 1e-8);
            });
            assert(stepShape, `${name}: saved predictor must contain horizontal pieces and vertical discontinuity markers`);
          }
          assert.deepEqual(record.errors, []);
          record.states.push({ name, geometry });
        }
        await shot(lesson.locator('.lesson-intro'), 'orientation');
        await shot(lesson.locator('.gbt-additive'), 'additive-reading');
        const correction = lab('correction');
        await state(correction, 'default correction', '7.33333 → 2');
        await shot(correction.locator('.gbt-plots'), 'residual-default');
        for (let i = 1; i < 8; i += 1) { await correction.getByRole('button', { name: 'Next tree', exact: true }).click(); await state(correction, `correction round ${i + 1}`); }
        assert(await correction.getByRole('button', { name: 'Next tree', exact: true }).isDisabled());
        await correction.getByRole('button', { name: 'Reset correction' }).click();
        for (const values of ['0,0,0,0,0,0', '3,-1,2,5,-2,1', '-30,30,-30,30,-30,30']) {
          await correction.getByLabel('Six target values').fill(values);
          await correction.getByRole('button', { name: 'Apply targets and restart' }).click();
          await state(correction, `targets ${values}`);
        }
        await range(correction, 'Correction learning rate', 1.5);
        await state(correction, 'extreme draft + over-one step');
        await shot(correction.locator('.gbt-plots'), 'residual-changed');
        const activeBefore = await correction.locator('.gbt-readouts').innerText();
        for (const invalid of ['1,2,3', '1,2,3,4,5,1e-9999', '1,2,3,4,5,31']) {
          await correction.getByLabel('Six target values').fill(invalid);
          await correction.getByRole('button', { name: 'Apply targets and restart' }).click();
          assert.equal(await correction.locator('.gbt-readouts').innerText(), activeBefore);
          assert(await correction.getByRole('alert').isVisible());
          await state(correction, `invalid preserves ${invalid}`);
        }
        await correction.getByRole('button', { name: 'Reset correction' }).focus();
        await page.keyboard.press('Enter');
        record.keyboard.push('Correction reset');
        const newton = lab('newton');
        for (const kind of ['square', 'logistic', 'confident']) {
          await newton.getByLabel('Loss fixture').selectOption(kind);
          for (const split of [1, 2, 3, 4]) {
            await range(newton, 'Rows sent left', split);
            await state(newton, `Newton ${kind}, left ${split}`);
          }
        }
        await range(newton, 'Rows sent left', 3);
        await range(newton, 'Leaf L2 lambda', 0);
        await range(newton, 'Newton learning rate', 1);
        await state(newton, 'mixed confident Newton overshoot');
        await shot(newton.locator('.gbt-row-strip'), 'newton-overshoot');
        await newton.getByRole('button', { name: 'Reset split' }).click();
        await range(newton, 'New-leaf gamma', 3);
        await state(newton, 'gamma rejection', 'No positive net gain');
        await range(newton, 'Leaf L1 alpha', 2);
        await state(newton, 'L1 zero leaf');
        await shot(newton.locator('.gbt-leaf-bands'), 'newton-penalties');
        await newton.getByRole('button', { name: 'Reset split' }).click();
        const histogram = lab('histogram');
        for (const coarse of ['coarse', 'fine']) {
          await histogram.getByLabel('Candidate boundaries').selectOption(coarse);
          for (const target of [0, 2, 5, 8, 10]) {
            await range(histogram, 'Missing-row target', target);
            await state(histogram, `${coarse} missing ${target}`);
          }
        }
        await range(histogram, 'Missing-row target', 2);
        await shot(histogram.locator('.gbt-histogram'), 'bins-fine');
        await histogram.getByRole('button', { name: 'Reset histogram' }).click();
        await shot(histogram.locator('.gbt-histogram'), 'bins-coarse');
        await shot(lesson.locator('.gbt-inline').nth(1), 'growth-reading');
        const sampling = lab('goss');
        for (let i = 0; i < 6; i += 1) {
          await state(sampling, `GOSS sample ${i}`);
          if (i < 5) await sampling.getByRole('button', { name: 'Next sample' }).click();
        }
        await shot(sampling.locator('.gbt-sample-strip'), 'goss-expectations');
        await sampling.getByRole('button', { name: 'Reset sample' }).click();
        await shot(sampling.locator('figure'), 'goss-contributions');
        await shot(lesson.locator('.gbt-inline').nth(2), 'bundling-reading');
        const ordered = lab('ordered');
        for (let row = 0; row < 6; row += 1) {
          await ordered.getByLabel('Categorical row').selectOption(String(row));
          const before = await ordered.locator('.gbt-readouts dd').first().innerText();
          await ordered.getByLabel("Flip the selected row's target").check();
          assert.equal(await ordered.locator('.gbt-readouts dd').first().innerText(), before);
          await state(ordered, `own-label invariance row ${row + 1}`);
        }
        await ordered.getByLabel('Reverse the permutation').check();
        for (const strength of [.25, 4]) { await range(ordered, 'Prior strength', strength); await state(ordered, `reversed prior ${strength}`); }
        await shot(ordered.locator('.gbt-prefix-strip'), 'prefix-reversed');
        await ordered.getByRole('button', { name: 'Reset categories' }).click();
        await shot(ordered.locator('.gbt-prefix-strip'), 'prefix-default');
        const validation = lab('validation');
        for (const depth of [1, 2, 3]) {
          await range(validation, 'Correction tree depth', depth);
          for (const rate of [.05, .3, 1]) {
            await range(validation, 'Validation learning rate', rate);
            for (const round of [0, 10, 60]) {
              await range(validation, 'Inspect boosting round', round);
              await state(validation, `validation depth${depth} rate${rate} round${round}`);
            }
          }
        }
        await shot(validation.locator('.gbt-plots'), 'validation-changed');
        await validation.getByRole('button', { name: 'Reset validation' }).click();
        await validation.getByRole('button', { name: 'Show stopping round' }).click();
        await state(validation, 'patience stop', '11');
        await validation.getByRole('button', { name: 'Show validation choice' }).click();
        await shot(validation.locator('.gbt-plots'), 'validation-default');
        for (const slider of await lesson.locator('input[type=range]').all()) {
          const old = await slider.inputValue();
          await slider.focus();
          await page.keyboard.press(Number(old) < Number(await slider.getAttribute('max')) ? 'ArrowRight' : 'ArrowLeft');
          assert.notEqual(await slider.inputValue(), old);
          await slider.fill(old); await slider.dispatchEvent('input');
          record.keyboard.push('Actual slider arrow');
        }
        const checkbox = ordered.getByLabel('Reverse the permutation');
        await checkbox.focus(); await page.keyboard.press('Space'); assert(await checkbox.isChecked());
        record.keyboard.push('Actual checkbox space');
        for (const details of await lesson.locator('details.gbt-data').all()) {
          await details.locator('summary').focus();
          await page.keyboard.press('Enter');
          assert.notEqual(await details.getAttribute('open'), null);
          const region = details.getByRole('region');
          await region.focus();
          assert(await region.evaluate(node => document.activeElement === node));
          await details.locator('summary').focus();
          await page.keyboard.press('Enter');
          assert.equal(await details.getAttribute('open'), null);
          record.keyboard.push('Data disclosure and focusable table');
        }
        for (const practice of await lesson.locator('div.gbt-practice').all()) {
          const details = practice.locator(':scope > details');
          assert.equal(await details.count(), 2);
          await details.nth(0).locator('summary').focus(); await page.keyboard.press('Enter');
          assert.notEqual(await details.nth(0).getAttribute('open'), null);
          assert.equal(await details.nth(1).getAttribute('open'), null);
          await details.nth(1).locator('summary').focus(); await page.keyboard.press('Enter');
          assert.notEqual(await details.nth(1).getAttribute('open'), null);
          record.keyboard.push('Hint before answer');
        }
        const blocks = await lesson.locator('.python-example').all();
        assert.equal(blocks.length, Object.keys(examples).length);
        for (let index = 0; index < blocks.length; index += 1) {
          const block = blocks[index];
          const title = await block.locator('h3').innerText();
          const example = Object.values(examples).find(item => item.title === title);
          assert(example, title);
          const code = await block.evaluate(node => [...node.children]
            .filter(child => getComputedStyle(child).whiteSpace === 'pre')
            .map(child => [...child.childNodes].filter(part => part.nodeType === Node.TEXT_NODE).map(part => part.textContent).join('')));
          assert.equal(normalize(code[0]), normalize(example.code));
          assert.equal(normalize(code[1]), normalize(example.expected));
          const preceding = await block.evaluate(node => node.previousElementSibling.textContent);
          assert(normalize(preceding).includes(normalize(example.question)));
          record.programs.push(example.title);
        }
        record.anchors = await lesson.locator('.lesson-intro a').evaluateAll(links => links.map(link => ({ href: link.getAttribute('href'), exists: !!document.getElementById(link.hash.slice(1)) })));
        assert(record.anchors.every(anchor => anchor.exists), JSON.stringify(record.anchors));
        record.equations = await lesson.locator('.katex-display').evaluateAll(nodes => nodes.map(node => ({ text: node.textContent, width: node.getBoundingClientRect().width, scrollWidth: node.scrollWidth })));
        assert(record.equations.every(item => item.scrollWidth <= item.width + 2), JSON.stringify(record.equations.filter(item => item.scrollWidth > item.width + 2)));
        for (const index of [1, 3, 7, 8, 10, 12]) {
          const equation = lesson.locator('.katex-display').nth(index);
          if (await equation.count()) await shot(equation, `equation-reading-${index}`);
        }
        await shot(lesson.getByText('A minimum Hessian mass is also different', { exact: false }), 'hessian-reading');
        await shot(blocks[9], 'xgboost-program-reading');
        await shot(lesson.locator('h2').last(), 'practice-reading');
        record.overflow = await page.evaluate(() => ({ viewport: innerWidth, document: document.documentElement.scrollWidth }));
        assert(record.overflow.document <= width + 1, JSON.stringify(record.overflow));
        assert.deepEqual(record.errors, []);
        assert.deepEqual(record.failedRequests, []);
        record.passed = true;
      } catch (error) {
        record.failure = String(error.stack || error);
        const file = path.join(directory, `failure-${width}.png`);
        await page.screenshot({ path: file });
        record.screenshots.push({ path: path.relative(process.cwd(), file).replaceAll('\\', '/'), sha256: hash(file) });
        throw error;
      } finally {
        records.push(record);
        fs.writeFileSync(path.join(directory, 'results.json'), JSON.stringify({ startedAt, checkedAt: new Date().toISOString(), sourceHashes, records }, null, 2));
        await page.close();
      }
    }
    assert.deepEqual(sources.map(file => ({ path: file, sha256: hash(file) })), sourceHashes, 'Source changed during browser review');
    console.log(JSON.stringify({ checkedAt: new Date().toISOString(), widths: records.map(record => ({ width: record.width, states: record.states.length, keyboard: record.keyboard.length, programs: record.programs.length, equations: record.equations.length, screenshots: record.screenshots.length })) }, null, 2));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

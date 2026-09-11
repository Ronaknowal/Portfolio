const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const { pathToFileURL } = require('node:url');
const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const directory = path.resolve('scratch/support-vector-machines-verification/browser');
fs.mkdirSync(directory, { recursive: true });
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const normalize = text => text.replace(/\s+/g, ' ').trim();
const sources = [
  'src/learn/data/topics/support-vector-machines-svm.jsx',
  'src/learn/data/support-vector-machines-models.js',
  'src/learn/data/support-vector-machines-examples.js',
  'src/learn/data/svm-validation-fixtures.js',
  'src/learn/components/lesson-labs/SupportVectorMachineLabs.jsx',
  'src/learn/components/lesson-labs/support-vector-machines-labs.css',
  'src/learn/data/curriculum/blueprints/support-vector-machines-svm.js',
];

(async () => {
  const startedAt = new Date().toISOString();
  const { svmExamples } = await import(pathToFileURL(path.resolve(sources[2])));
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [];
  try {
    for (const width of Number.isFinite(Number(process.argv[2])) ? [Number(process.argv[2])] : [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1050 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      const record = { width, states: [], screenshots: [], errors: [], failedRequests: [], keyboard: [] };
      records.push(record);
      page.on('pageerror', error => record.errors.push(String(error)));
      page.on('console', message => { if (message.type() === 'error' && !message.text().includes('[vite]')) record.errors.push(message.text()); });
      page.on('requestfailed', request => record.failedRequests.push({ url: request.url(), failure: request.failure() }));
      try {
        await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/support-vector-machines-svm?module=classical-ml', { waitUntil: 'domcontentloaded' });
        const lesson = page.locator('.support-vector-machines-lesson');
        await lesson.waitFor();
        await page.evaluate(() => document.fonts.ready);
        record.fonts = await page.evaluate(() => [...document.fonts].filter(font => font.status === 'loaded').map(font => font.family));
        assert(record.fonts.some(font => font.includes('Space Grotesk')));
        assert(record.fonts.some(font => font.includes('JetBrains Mono')));
        await page.addStyleTag({ content: 'html{scroll-behavior:auto!important}' });
        const lab = name => lesson.locator(`[data-svm-lab="${name}"]`);
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
            const numeric = [...svg.querySelectorAll('*')].flatMap(node => [...node.attributes].filter(attr => ['points', 'd', 'cx', 'cy', 'x', 'y', 'width', 'height'].includes(attr.name)).map(attr => attr.value));
            return { labels, finite: numeric.every(value => !/NaN|Infinity/.test(value)) };
          }));
          assert(geometry.every(item => item.labels.length === 0 && item.finite), `${name}: ${JSON.stringify(geometry)}`);
          assert.deepEqual(record.errors, []);
          record.states.push({ name, geometry });
        }
        assert.equal(await lesson.locator('[data-svm-lab]').count(), 7);
        if (process.argv[3] !== 'reading') {
        const margin = lab('margin');
        await state(margin, 'margin default', 'Score-level corridor width');
        for (const scale of [.5, 3]) { await range(margin, 'Score scale', scale); await state(margin, `scale ${scale}`); }
        await range(margin, 'Normal angle', 45);
        await range(margin, 'Boundary offset', .75);
        await state(margin, 'rotated misclassification', 'No');
        if (width === 1440 || width === 320) await shot(margin.locator('figure'), 'margin-geometry');
        await margin.getByRole('button', { name: 'Reset margin', exact: true }).click();
        const support = lab('support');
        for (const value of [3, 1, .5, .25]) { await range(support, 'Point D coordinate', value); await state(support, `support ${value}`); }
        if (width === 390) await shot(support.locator('figure'), 'support-motion');
        await support.getByRole('button').click();
        const soft = lab('soft');
        for (const fixture of ['separate', 'conflict']) {
          await soft.getByLabel('Two-point fixture', { exact: true }).selectOption(fixture);
          for (const c of ['0.1', '0.25', '0.5', '1', '2']) {
            await soft.getByLabel('Penalty C', { exact: true }).selectOption(c);
            await state(soft, `${fixture} C${c}`);
          }
        }
        await range(soft, 'Position in optimal bias interval', 1);
        await state(soft, 'conflict shifted bias');
        if (width === 320) await shot(soft.locator('figure'), 'conflicting-bias');
        await soft.getByRole('button').click();
        const kernel = lab('kernel');
        for (const kind of ['linear', 'poly', 'rbf']) {
          await kernel.getByLabel('Kernel', { exact: true }).selectOption(kind);
          for (const c of ['0.05', '1', '4']) {
            await kernel.getByLabel('Kernel penalty C', { exact: true }).selectOption(c);
            await state(kernel, `${kind} C${c}`);
          }
        }
        for (const gamma of [.05, 4]) { await range(kernel, 'RBF gamma', gamma); await state(kernel, `gamma ${gamma}`); }
        await kernel.getByLabel('Kernel', { exact: true }).selectOption('poly');
        await range(kernel, 'Query x1', 2); await range(kernel, 'Query x2', 2);
        await state(kernel, 'expanded polynomial contributions', '±2 scale');
        assert(await kernel.locator('.svm-contribution-track i').evaluateAll(nodes => nodes.every(node => parseFloat(node.style.width) <= 50)));
        if (width === 1440) await shot(kernel.locator('.svm-plots'), 'kernel-lift');
        if (width === 390) await shot(kernel.locator('.svm-contributions'), 'kernel-contributions');
        await kernel.getByRole('button').click();
        const pair = lab('pair');
        for (const value of ['1,2', '0,2', '0,1']) {
          await pair.getByLabel('Selected coefficient pair', { exact: true }).selectOption(value);
          await state(pair, `pair ${value}`);
          if (await pair.getByRole('button', { name: 'Commit segment optimum', exact: true }).isEnabled()) await pair.getByRole('button', { name: 'Commit segment optimum', exact: true }).click();
        }
        await pair.getByLabel('Pair fixture', { exact: true }).selectOption('duplicate');
        await state(pair, 'duplicate nonzero gain');
        if (width === 320) await shot(pair.locator('.svm-plots'), 'zero-curvature-pair');
        await pair.getByRole('button', { name: 'Commit segment optimum', exact: true }).click();
        await state(pair, 'duplicate endpoint committed', '2.4');
        assert(await pair.getByRole('button', { name: 'Commit segment optimum', exact: true }).isDisabled());
        await pair.getByRole('button', { name: 'Reset pair moves', exact: true }).click();
        const validation = lab('validation');
        const buttons = validation.locator('.svm-selection button');
        for (let index = 0; index < 9; index += 1) {
          await buttons.nth(index).click();
          assert.equal(await buttons.nth(index).getAttribute('aria-pressed'), 'true');
          await state(validation, `actual saved model ${index}`);
        }
        await validation.getByRole('checkbox').check();
        await state(validation, 'actual training rows');
        if (width === 1440) await shot(validation, 'measured-selection');
        await validation.getByRole('button', { name: 'Reset validation', exact: true }).click();
        const tube = lab('tube');
        await range(tube, 'Target amplitude a', 3);
        for (const epsilon of [0, 2]) for (const c of ['0.1', '2']) {
          await range(tube, 'Tube epsilon', epsilon); await tube.getByLabel('Regression penalty C', { exact: true }).selectOption(c);
          await state(tube, `tube epsilon${epsilon} C${c}`);
        }
        if (width === 390) await shot(tube.locator('figure'), 'target-units-tube');
        await tube.getByRole('button').click();
        for (const control of await lesson.locator('.svm-lab button:enabled,.svm-lab select,.svm-lab input:enabled').all()) {
          await control.focus(); assert(await control.evaluate(node => node === document.activeElement));
          record.keyboard.push(await control.getAttribute('aria-label') || await control.innerText());
        }
        const angle = margin.getByLabel('Normal angle', { exact: true });
        await angle.focus(); await angle.press('ArrowRight'); assert.equal(await angle.inputValue(), '5');
        await margin.getByRole('button').focus(); await page.keyboard.press('Enter'); assert.equal(await angle.inputValue(), '0');
        const check = validation.getByRole('checkbox'); await check.focus(); await check.press('Space'); assert(await check.isChecked()); await check.press('Space');
        for (const details of await lesson.locator('details').all()) {
          const summary = details.locator('summary').first(); await summary.focus(); await summary.press('Enter');
          assert(await details.evaluate(node => node.open)); await summary.press('Enter');
        }
        record.disclosures = await lesson.locator('details').count();
        }
        if (process.argv[3] === 'reading') {
          const tied = await page.evaluate(async () => {
            const model = await import('/src/learn/data/support-vector-machines-models.js');
            return model.svmPairStep([[-1, 0], [1, 0], [1, 0]], [-1, 1, 1], [.4, .1, .3], 1, 1, 2);
          });
          assert.equal(tied.bestDelta, 0); assert.deepEqual(tied.after, [.4, .1, .3]);
          const pair = lab('pair');
          await pair.getByLabel('Pair fixture', { exact: true }).selectOption('duplicate');
          await state(pair, 'final duplicate linear direction');
          await pair.getByRole('button', { name: 'Commit segment optimum', exact: true }).click();
          await state(pair, 'final duplicate committed', '2.4');
          await pair.getByRole('button', { name: 'Reset pair moves', exact: true }).click();
          record.flatTieLoadedModule = { delta: tied.bestDelta, alpha: tied.after };
        }
        const programs = lesson.locator('.python-example');
        assert.equal(await programs.count(), 14);
        for (let index = 0; index < 14; index += 1) {
          const example = Object.values(svmExamples)[index], program = programs.nth(index);
          const code = await program.evaluate(node => [...node.children]
            .filter(child => getComputedStyle(child).whiteSpace === 'pre')
            .map(child => [...child.childNodes].filter(part => part.nodeType === Node.TEXT_NODE).map(part => part.textContent).join('')));
          assert.equal(code[0].trim(), example.code);
          assert.equal(code[1].trim(), example.expected);
          assert(normalize(await program.evaluate(node => node.previousElementSibling.textContent)).includes(example.question));
        }
        record.programs = 14;
        const ids = await lesson.locator('.lesson-intro nav a').evaluateAll(nodes => nodes.map(node => node.hash));
        for (const id of ids) assert.equal(await lesson.locator(`[id="${id.slice(1)}"]`).count(), 1, `missing anchor ${id}`);
        record.anchors = ids.length;
        record.equations = await lesson.locator('.katex-display').evaluateAll(nodes => nodes.map(node => ({ width: node.getBoundingClientRect().width, content: node.scrollWidth, text: node.textContent.slice(0, 90) })));
        assert(record.equations.every(item => item.content <= item.width + 2), JSON.stringify(record.equations));
        record.overflow = await page.evaluate(() => ({ page: document.documentElement.scrollWidth, viewport: innerWidth }));
        assert(record.overflow.page <= width + 2, JSON.stringify(record.overflow));
        assert.equal(await page.locator('vite-error-overlay').count(), 0);
        assert.deepEqual(record.errors, []); assert.deepEqual(record.failedRequests, []);
        if (width === 320) await shot(lesson.getByText('The coefficient of b must vanish:', { exact: false }).first(), 'dual-reading');
        if (width === 390) await shot(lesson.getByText('The unit of the target changes the objective', { exact: true }), 'regression-reading');
        console.log(`${width}: ${record.states.length} states, ${record.keyboard.length} controls, ${record.programs} actual programs passed`);
      } catch (error) {
        record.failure = String(error.stack || error);
        await page.screenshot({ path: path.join(directory, `failure-${width}.png`) });
        throw error;
      } finally { await page.close(); }
    }
  } finally {
    await browser.close();
    fs.writeFileSync(path.join(directory, 'results.json'), JSON.stringify({ startedAt, completedAt: new Date().toISOString(), sourceHashes: sources.map(file => ({ path: file, sha256: hash(file) })), records }, null, 2));
  }
})().catch(error => { console.error(error); process.exit(1); });

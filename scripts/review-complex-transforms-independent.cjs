const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const directory = 'scratch/complex-transforms-independent-review';
fs.mkdirSync(directory, { recursive: true });
const digest = path => crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex');
async function slide(locator, value) {
  await locator.evaluate((node, next) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(node, String(next));
    node.dispatchEvent(new Event('input', { bubbles: true }));
    node.dispatchEvent(new Event('change', { bubbles: true }));
  }, value);
}
async function capture(page, locator, name, width) {
  await locator.evaluate(node => scrollTo({ top: scrollY + node.getBoundingClientRect().top - 155, behavior: 'instant' }));
  await page.waitForTimeout(100);
  const path = `${directory}/${name}-${width}.png`;
  await page.screenshot({ path });
  return { path, sha256: digest(path), opened: false };
}
async function dataBounds(svg, kind) {
  return svg.evaluate((node, type) => {
    const points = type === 'dots'
      ? [...node.querySelectorAll('circle')].map(circle => [Number(circle.getAttribute('cx')), Number(circle.getAttribute('cy'))])
      : [...node.querySelectorAll(':scope > path')].flatMap(path => [...path.getAttribute('d').matchAll(/[ML]([+-]?[\d.]+),([+-]?[\d.]+)/g)].map(m => [Number(m[1]), Number(m[2])]));
    return { count: points.length, minY: Math.min(...points.map(p => p[1])), maxY: Math.max(...points.map(p => p[1])), escaped: points.filter(([x,y]) => x < 41.99 || x > 284.01 || y < 23.99 || y > 164.01) };
  }, kind);
}
(async () => {
  const { complexTransformExamples: examples } = await import('../src/learn/data/complex-transforms-examples.js');
  const packet = JSON.parse(fs.readFileSync('docs/teaching/evidence/complex-transforms-author-review.json', 'utf8'));
  const sources = packet.productionSources.map(({ path }) => ({ path, sha256: digest(path) }));
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [], images = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 } });
      await page.routeWebSocket('**', socket => socket.close());
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => { if (['error', 'warning'].includes(message.type()) && !message.text().startsWith('[vite]')) errors.push(message.text()); });
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/complex-numbers-fourier-laplace-transforms?module=math-foundations', { waitUntil: 'domcontentloaded', timeout: 60000 });
      const lesson = page.locator('.complex-transforms-lesson');
      await lesson.waitFor({ timeout: 60000 });
      await page.evaluate(() => document.fonts.ready);
      assert(await page.evaluate(() => document.fonts.check('16px "Space Grotesk"')));
      assert.equal(await page.locator('vite-error-overlay').count(), 0);
      assert.equal(await lesson.locator('.katex-error').count(), 0);
      assert((await lesson.innerText()).includes('square-integrable'));
      const programs = await lesson.locator('.python-example').evaluateAll(nodes => nodes.map(node => ({ title: node.querySelector('h3').textContent, question: node.previousElementSibling.textContent.replace(/^Before running\.\s*/, ''), blocks: [...node.children].filter(child => getComputedStyle(child).whiteSpace === 'pre').map(child => [...child.childNodes].filter(n => n.nodeType === Node.TEXT_NODE).map(n => n.textContent).join('')) })));
      assert.equal(programs.length, 17);
      for (const program of programs) {
        const example = Object.values(examples).find(e => e.title === program.title);
        assert.equal(program.question, example.question);
        assert.equal(program.blocks[0].trim(), example.code.trim());
        assert.equal(program.blocks[1].trim(), example.expected.trim());
      }
      const dft = lesson.locator('[data-transform-lab="dft"]');
      await dft.getByLabel('DFT sample draft').fill('4,-4,4,4,4,4,4,-4');
      await dft.getByRole('button', { name: 'Apply sample list' }).focus();
      await page.keyboard.press('Enter');
      await slide(dft.getByLabel('DFT bin k', { exact: true }), 1);
      await dft.getByRole('checkbox').focus(); await page.keyboard.press('Space');
      assert(await dft.getByRole('checkbox').isChecked());
      const dftBounds = await dataBounds(dft.locator('.transform-chart svg'), 'dots');
      assert.equal(dftBounds.escaped.length, 0, JSON.stringify(dftBounds));
      await dft.locator('summary').click();
      assert((await dft.locator('table').innerText()).includes('6.828'));
      images.push(await capture(page, dft.locator('.transform-chart'), 'changed-filtered-dft', width));
      const filter = lesson.locator('[data-transform-lab="filter"]');
      await slide(filter.getByLabel('Decay rate a', { exact: true }), 20);
      await slide(filter.getByLabel('Initial output y(0)', { exact: true }), -3);
      await slide(filter.getByLabel('Filter third-tone phase', { exact: true }), Math.PI/4);
      const filterBounds = await dataBounds(filter.locator('.transform-chart svg'), 'paths');
      assert.equal(filterBounds.escaped.length, 0, JSON.stringify(filterBounds));
      assert((await filter.innerText()).includes('-5.548'));
      images.push(await capture(page, filter.locator('.transform-chart'), 'changed-transient', width));
      const laplace = lesson.locator('[data-transform-lab="laplace"]');
      await slide(laplace.getByLabel('Real part sigma', { exact: true }), -1);
      await slide(laplace.getByLabel('Angular frequency omega', { exact: true }), 0);
      assert((await laplace.innerText()).includes('Boundary: no ordinary limit'));
      assert((await laplace.innerText()).includes('Pole: undefined'));
      await laplace.getByLabel('Exponential support', { exact: true }).selectOption('left');
      await slide(laplace.getByLabel('Real part sigma', { exact: true }), -2);
      assert((await laplace.innerText()).includes('Inside the ROC'));
      images.push(await capture(page, laplace, 'left-roc', width));
      const laplaceProgram = lesson.locator('.python-example').filter({ has: page.getByRole('heading', { name: examples.laplace.title, exact: true }) });
      images.push(await capture(page, laplaceProgram, 'stable-laplace-program', width));
      const intro = lesson.locator('h3').filter({ hasText: 'Projection is also the best finite least-squares fit' });
      assert((await intro.evaluate(node => node.nextElementSibling.textContent)).includes('square-integrable'));
      images.push(await capture(page, intro, 'projection-hypothesis', width));
      const convergence = lesson.locator('[data-transform-lab="convergence"]');
      await slide(convergence.getByLabel('Number of odd harmonics', { exact: true }), 64);
      assert((await convergence.innerText()).includes('1.179'));
      images.push(await capture(page, convergence.locator('.transform-chart').nth(1), 'gibbs-scaled-neighborhood', width));
      const equations = await lesson.locator('.katex-display').evaluateAll(nodes => nodes.map(node => ({ client: node.clientWidth, scroll: node.scrollWidth })));
      assert(equations.every(row => row.scroll <= row.client+2));
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 2);
      assert.equal(overflow, false);
      assert.deepEqual(errors, []);
      results.push({ width, publicFont: true, actualProgramsMatched: 17, changedDft: dftBounds, changedTransient: filterBounds, controls: ['Enter apply', 'Space checkbox', 'bin selection', 'rate20', 'initial-3', 'phasepi/4', 'ROC boundary', 'left ROC', '64-term moving peak'], equationCount: equations.length, errors, overflow });
      await page.close();
    }
    assert.deepEqual(sources, sources.map(({ path }) => ({ path, sha256: digest(path) })));
    fs.writeFileSync(`${directory}/browser-results.json`, JSON.stringify({ checkedAt: new Date().toISOString(), passed: true, productionSources: sources, results, images }, null, 2));
    console.log(JSON.stringify(results, null, 2));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

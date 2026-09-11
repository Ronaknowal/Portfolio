const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const directory = 'scratch/abstract-algebra-browser';
const digest = path => crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex');
(async () => {
  const { abstractAlgebraExamples: examples } = await import('../src/learn/data/abstract-algebra-examples.js');
  const initial = JSON.parse(fs.readFileSync(`${directory}/results.json`, 'utf8'));
  const productionSources = initial.productionSources.map(({ path }) => ({ path, sha256: digest(path) }));
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [], images = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1080 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => { if (['error', 'warning'].includes(message.type()) && !message.text().startsWith('[vite]')) errors.push(message.text()); });
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/abstract-algebra-groups-symmetry-actions?module=math-foundations', { waitUntil: 'domcontentloaded', timeout: 60000 });
      const lesson = page.locator('.abstract-algebra-lesson');
      await lesson.waitFor({ timeout: 60000 });
      await page.evaluate(() => document.fonts.ready);
      assert(await page.evaluate(() => document.fonts.check('16px "Space Grotesk"')));
      const capture = async (locator, name) => {
        await locator.first().evaluate(node => scrollTo({ top: scrollY + node.getBoundingClientRect().top - 115, behavior: 'instant' }));
        await page.waitForTimeout(90);
        const path = `${directory}/final-${name}-${width}.png`;
        await page.screenshot({ path });
        images.push({ path, sha256: digest(path), opened: false });
      };
      for (let index = 0; index < 11; index += 1) await capture(lesson.locator('h2').nth(index), `reading-${index + 1}`);
      assert.deepEqual(await lesson.locator('nav a').evaluateAll(nodes => nodes.map(node => node.hash).filter(hash => !document.getElementById(hash.slice(1)))), []);
      const figure = lesson.locator('.algebra-figure').first();
      assert((await figure.innerText()).includes('Scroll sideways'));
      assert.equal(await figure.locator('.algebra-action-graph circle').count(), 2);
      assert.equal(await figure.locator('.algebra-action-graph .algebra-s').count(), 2);
      const labelBounds = await figure.locator('.algebra-action-graph text').evaluateAll(nodes => nodes.map(node => { const box = node.getBBox(); return { x: box.x, y: box.y, right: box.x + box.width, bottom: box.y + box.height }; }));
      assert(labelBounds.every(box => box.x >= 0 && box.y >= 0 && box.right <= 220 && box.bottom <= 150));
      await capture(figure, 'cayley');
      await capture(figure.locator('.algebra-action-pair>div'), 'action-graph');
      if (width < 600) {
        const scroll = figure.getByRole('region');
        await scroll.focus(); await page.keyboard.press('ArrowRight'); await page.waitForTimeout(150);
        assert(await scroll.evaluate(node => node.scrollLeft > 0));
        await scroll.evaluate(node => { node.scrollLeft = node.scrollWidth; });
        await capture(scroll, 'cayley-scrolled');
      }
      await capture(lesson.locator('[data-algebra-lab="composition"] .algebra-route').first(), 'composition');
      await capture(lesson.locator('.algebra-figure').last(), 'modular');
      const coset = lesson.locator('[data-algebra-lab="cosets"]');
      await coset.getByLabel('Subgroup H', { exact: true }).selectOption('rotations');
      await coset.getByLabel('Representative from H', { exact: true }).selectOption('3');
      assert((await coset.getByRole('status').innerText()).includes('same output block'));
      await capture(coset, 'normal-coset');
      await coset.getByLabel('Subgroup H', { exact: true }).selectOption('reflection');
      await coset.getByLabel('Representative from H', { exact: true }).selectOption('1');
      await capture(coset.locator('.algebra-product-witness'), 'bad-coset');
      const map = lesson.locator('[data-algebra-lab="equivariance"]');
      await map.getByLabel('Processing map', { exact: true }).selectOption('averaged');
      await capture(map.locator('.algebra-route').first(), 'averaged-map');
      const positionOverlap = await map.locator('.algebra-square svg g').evaluateAll(nodes => nodes.some(node => {
        const texts = node.querySelectorAll('text'); const a = texts[0].getBBox(), b = texts[1].getBBox();
        return a.x < b.x + b.width && a.x + a.width > b.x && a.y < b.y + b.height && a.y + a.height > b.y;
      }));
      assert.equal(positionOverlap, false);
      const deep = lesson.locator('details').filter({ has: page.locator('summary').filter({ hasText: 'Deeper: why this is the nearest tied matrix' }) });
      await deep.locator('summary').click();
      await capture(deep, 'projection-proof');
      for (const summary of await lesson.locator('.lesson-check summary').all()) await summary.click();
      const practice = lesson.locator('.lesson-check').filter({ has: page.getByRole('heading', { name: 'H. Produce a changed sensor-map audit', exact: true }) });
      await capture(practice.locator('details').last(), 'changed-solution');
      assert((await practice.innerText()).includes('133/4'));
      const programs = await lesson.locator('.python-example').evaluateAll(nodes => nodes.map(node => ({ title: node.querySelector('h3').textContent, question: node.previousElementSibling.textContent.replace(/^Before running:\s*/, ''), blocks: [...node.children].filter(child => getComputedStyle(child).whiteSpace === 'pre').map(child => [...child.childNodes].filter(n => n.nodeType === Node.TEXT_NODE).map(n => n.textContent).join('')) })));
      for (const program of programs) { const example = examples.find(row => row.title === program.title); assert.equal(program.question, example.question); assert.equal(program.blocks[0].trim(), example.code.trim()); assert.equal(program.blocks[1].trim(), example.expected.trim()); }
      assert.equal(programs.length, 10);
      await capture(lesson.locator('.python-example').last(), 'program');
      await capture(lesson.locator('.lesson-sources'), 'references');
      const optionWidths = await lesson.locator('select').evaluateAll(nodes => nodes.map(node => {
        const context = document.createElement('canvas').getContext('2d');
        const style = getComputedStyle(node);
        context.font = `${style.fontSize} ${style.fontFamily}`;
        return { label: node.getAttribute('aria-label'), available: node.clientWidth - parseFloat(style.paddingLeft) - parseFloat(style.paddingRight) - 25, maximum: Math.max(...[...node.options].map(option => context.measureText(option.textContent).width)) };
      }));
      assert(optionWidths.every(row => row.maximum <= row.available), JSON.stringify(optionWidths));
      const equations = await lesson.locator('.katex-display').evaluateAll(nodes => nodes.map(node => ({ width: node.clientWidth, scroll: node.scrollWidth })));
      assert(equations.every(row => row.scroll <= row.width + 2));
      assert.equal(await lesson.locator('.katex-error').count(), 0);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 2), false);
      assert.deepEqual(errors, []);
      results.push({ width, publicFont: true, anchors: 11, actualPrograms: programs.length, equations: equations.length, actionGraphLabelsInside: true, sensorValuesSeparateFromPositions: true, optionWidths, errors, documentOverflow: false });
      await page.close();
    }
    assert.deepEqual(productionSources, productionSources.map(({ path }) => ({ path, sha256: digest(path) })));
    fs.writeFileSync(`${directory}/final-reading-results.json`, JSON.stringify({ checkedAt: new Date().toISOString(), status: 'passed', productionSources, results, images, scope: 'Final reading and narrow display amendment closure after the comprehensive 44-state-per-width interaction pass. Adds visible scroll instruction, explicit column-header scope, separates action-loop labels from their curves, and fits every dropdown option by shortening labels and stacking phone controls; models/programs unchanged.' }, null, 2));
    console.log(JSON.stringify(results, null, 2));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

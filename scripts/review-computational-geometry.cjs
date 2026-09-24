const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const directory = path.resolve('scratch/computational-geometry-browser');
fs.mkdirSync(directory, { recursive: true });
const normalize = value => value.replace(/\s+/g, ' ').trim();
const coordinates = value => `(${value.join(', ')})`;

(async () => {
  const models = await import(pathToFileURL(path.resolve('src/learn/data/computational-geometry-models.js')));
  const { computationalGeometryExamples: examples } = await import(pathToFileURL(path.resolve('src/learn/data/computational-geometry-examples.js')));
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const results = [], errors = [];
  try {
    for (const width of [1440, 390]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 }, reducedMotion: 'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      page.on('pageerror', error => errors.push(`${width}: ${error.message}`));
      await page.goto((process.env.LEARNING_BASE_URL || 'http://127.0.0.1:5173') + '/learn/path/full-curriculum/computational-geometry-robust-predicates-convex-hulls?module=data-structures-algorithms');
      const lesson = page.locator('.computational-geometry-lesson');
      await lesson.waitFor();
      assert.equal((await lesson.locator('.geometry-calculation').innerText()).split('\n').length, 4);
      assert.equal(await lesson.locator('.geometry-calculation').evaluate(element => element.scrollWidth > element.clientWidth + 1), false);
      const record = { width, anchors: [], orientations: 0, segments: 0, precisions: 0, hullFrames: 0, hullContracts: 0, polygonEdges: 0, examples: [], images: [] };
      const keyButton = async (region, name, key = 'Enter') => {
        await region.getByRole('button', { name, exact: true }).focus();
        await page.keyboard.press(key);
      };
      const setRange = async (region, name, value) => {
        const control = region.getByRole('slider', { name, exact: true });
        const minimum = Number(await control.getAttribute('min'));
        await control.focus();
        await page.keyboard.press('Home');
        for (let step = minimum; step < value; step++) await page.keyboard.press('ArrowRight');
        assert.equal(await control.inputValue(), String(value));
      };
      const fact = (region, label) => region.locator('.geometry-facts > div').filter({ has: page.locator('dt', { hasText: label }) }).locator('dd');
      const capture = async (locator, name) => {
        const style = await page.addStyleTag({ content: '.learn-nav { visibility: hidden !important; }' });
        await locator.screenshot({ path: path.join(directory, `${name}-${width}.png`) });
        await style.evaluate(element => element.remove());
        record.images.push(`${name}-${width}.png`);
      };
      const anchors = lesson.locator('nav[aria-label="In this lesson"] a');
      assert.equal(await anchors.count(), 8);
      for (let index = 0; index < await anchors.count(); index++) {
        const href = await anchors.nth(index).getAttribute('href');
        await anchors.nth(index).focus();
        await page.keyboard.press('Enter');
        await page.waitForFunction(hash => location.hash === hash, href);
        await page.waitForTimeout(100);
        const box = await page.locator(`[id="${href.slice(1)}"]`).boundingBox();
        assert(box.y >= 50 && box.y < 200, `anchor ${href} arrived at ${box.y}`);
        record.anchors.push({ href, top: box.y });
        await page.screenshot({ path: path.join(directory, `reading-${index + 1}-${width}.png`) });
      }
      const turn = lesson.getByRole('region', { name: 'Orientation and signed area', exact: true });
      for (const query of [[4, 5], [4, 2], [4, 1], [1, 1], [8, 8]]) {
        await setRange(turn, 'Point C x', query[0]);
        await setRange(turn, 'Point C y', query[1]);
        const determinant = models.orientation([1, 1], [7, 3], query);
        assert.equal(await fact(turn, 'D = product 1 − product 2').innerText(), String(determinant));
        await keyButton(turn, 'Reverse A and B');
        assert.equal(await fact(turn, 'D = product 1 − product 2').innerText(), String(-determinant));
        await keyButton(turn, 'Reverse A and B');
        record.orientations += 2;
      }
      await keyButton(turn, 'Reset orientation', 'Space');
      assert.equal(await fact(turn, 'D = product 1 − product 2').innerText(), '18');
      await capture(turn, 'orientation');
      const segments = lesson.getByRole('region', { name: 'Closed segment intersection', exact: true });
      for (const [name, points] of Object.entries(models.segmentPresets)) {
        await keyButton(segments, name === 'crossing' ? 'Reset crossing' : `Try ${name}`);
        assert.equal(await fact(segments, 'Classification').innerText(), models.segmentState(...points).kind);
        record.segments++;
        if (name === 'overlap' || name === 'point') await capture(segments, `segments-${name}`);
      }
      await keyButton(segments, 'Reset crossing');
      await segments.getByLabel('Endpoint to move').selectOption('0');
      await setRange(segments, 'Selected endpoint x', 8);
      await setRange(segments, 'Selected endpoint y', 0);
      assert.equal(await fact(segments, 'Classification').innerText(), models.segmentState([8, 0], ...models.segmentPresets.crossing.slice(1)).kind);
      record.segments++;
      await keyButton(segments, 'Reset crossing');
      assert.equal(await segments.getByLabel('Endpoint to move').inputValue(), '3');
      const precision = lesson.getByRole('region', { name: 'Exact and floating predicate comparison', exact: true });
      for (const exponent of [20, 26, 27, 30]) {
        await setRange(precision, 'Precision exponent', exponent);
        const state = models.precisionState(exponent);
        assert.deepEqual(await precision.locator('.geometry-arithmetic-paths strong').allTextContents(), [`D = ${state.exactDeterminant}`, `D = ${state.floatDeterminant}`]);
        record.precisions++;
      }
      await keyButton(precision, 'Reset precision comparison');
      await capture(precision, 'precision-products');
      await precision.getByLabel('Failure stage').selectOption('input');
      assert((await precision.getByRole('status').innerText()).includes('A and B became the same represented point'));
      const inputState = models.precisionState(27, 'input');
      assert.deepEqual(await precision.locator('.geometry-arithmetic-paths strong').allTextContents(), [`D = ${inputState.exactDeterminant}`, `D = ${inputState.floatDeterminant}`]);
      record.precisions++;
      await capture(precision, 'precision-input');
      await keyButton(precision, 'Reset precision comparison');
      await capture(lesson.locator('.geometry-inline'), 'concave-envelope');
      const hull = lesson.getByRole('region', { name: 'Monotone convex hull construction', exact: true });
      for (const [name, points] of Object.entries(models.hullPresets)) for (const boundary of [false, true]) {
        await keyButton(hull, name === 'fence' ? 'Reset fence points' : `Load ${name}`);
        await hull.getByRole('checkbox', { name: 'Include every boundary point' }).setChecked(boundary);
        const state = models.hullTrace(points, boundary);
        if (name === 'turns' && !boundary) {
          for (let index = 0; index < state.frames.length; index++) {
            const frame = state.frames[index];
            assert.equal(await fact(hull, 'Active stack').innerText(), frame.stack.map(coordinates).join(' → ') || 'empty');
            assert((await hull.locator('.geometry-current').innerText()).includes(`${frame.phase}: ${frame.action}`));
            if (frame.removed && !record.images.includes(`hull-pop-${width}.png`)) await capture(hull, 'hull-pop');
            record.hullFrames++;
            if (index + 1 < state.frames.length) await keyButton(hull, 'Next hull step');
          }
        } else await keyButton(hull, 'Show completed hull');
        assert.equal(await fact(hull, 'Active stack').innerText(), state.hull.map(coordinates).join(' → ') || 'empty');
        assert(await hull.getByRole('button', { name: 'Next hull step', exact: true }).isDisabled());
        record.hullContracts++;
        if ((name === 'fence' && boundary) || name === 'line' && !boundary) await capture(hull, `hull-${name}`);
        await keyButton(hull, 'Restart hull trace');
        assert(await hull.getByRole('button', { name: 'Previous hull step', exact: true }).isDisabled());
      }
      await keyButton(hull, 'Reset fence points');
      await keyButton(hull, 'Next hull step');
      await keyButton(hull, 'Previous hull step');
      const activeBeforeInvalid = await hull.locator('.geometry-current').innerText();
      const pointInput = hull.getByLabel('Point records, one x,y per line');
      for (const invalid of ['1.5,3', '9,2', Array(21).fill('1,2').join('\n')]) {
        await pointInput.fill(invalid);
        await keyButton(hull, 'Apply point records');
        assert(await hull.getByRole('alert').isVisible());
        assert.equal(await hull.locator('.geometry-current').innerText(), activeBeforeInvalid);
      }
      await pointInput.fill('0,0\n8,0\n8,8\n0,8\n4,4');
      await keyButton(hull, 'Apply point records');
      assert.equal(await hull.getByRole('alert').count(), 0);
      await keyButton(hull, 'Show completed hull');
      assert.equal(await fact(hull, 'Active stack').innerText(), '(0, 0) → (8, 0) → (8, 8) → (0, 8)');
      await capture(hull, 'hull-edge-coordinates');
      await keyButton(hull, 'Reset fence points');
      const polygon = lesson.getByRole('region', { name: 'Polygon boundary and crossing parity', exact: true });
      for (const [shape, vertices] of Object.entries(models.polygonPresets)) for (const query of [[4, 5], [4, 3], [3, 4], [2, 5], [8, 8], [1, 1]]) {
        await polygon.getByLabel('Simple polygon').selectOption(shape);
        await setRange(polygon, 'Query Q x', query[0]);
        await setRange(polygon, 'Query Q y', query[1]);
        const state = models.polygonState(vertices, query);
        assert.equal(await fact(polygon, 'Query classification').innerText(), state.classification);
        assert.equal(await fact(polygon, 'Total right crossings').innerText(), String(state.crossings));
        for (let index = 0; index < state.edges.length; index++) {
          const edge = state.edges[index];
          assert.equal(await fact(polygon, 'Selected edge / its determinant').innerText(), `${index + 1} / ${edge.determinant}`);
          assert.equal(await fact(polygon, 'Counts on this edge?').innerText(), edge.rightCrossing ? 'yes' : 'no');
          record.polygonEdges++;
          if (index + 1 < state.edges.length) await keyButton(polygon, 'Next polygon edge');
        }
        assert(await polygon.getByRole('button', { name: 'Next polygon edge', exact: true }).isDisabled());
        await keyButton(polygon, 'Previous polygon edge');
      }
      await keyButton(polygon, 'Reset polygon query');
      await keyButton(polygon, 'Next polygon edge');
      await capture(polygon, 'polygon-opening');
      await setRange(polygon, 'Query Q y', 3);
      assert.equal(await fact(polygon, 'Query classification').innerText(), 'boundary');
      await capture(polygon, 'polygon-boundary');
      await keyButton(polygon, 'Reset polygon query', 'Space');
      const codeBlocks = lesson.locator('.python-example');
      const entries = Object.entries(examples);
      assert.equal(await codeBlocks.count(), entries.length);
      for (let index = 0; index < entries.length; index++) {
        const title = await codeBlocks.nth(index).getByRole('heading').innerText();
        const [name, example] = entries.find(([, fixture]) => fixture.title === title);
        const displayed = await codeBlocks.nth(index).locator(':scope > div').evaluateAll(elements => elements.map(element => [...element.childNodes].filter(child => child.nodeType === Node.TEXT_NODE).map(child => child.textContent).join('')));
        assert.deepEqual(displayed.map(normalize), [normalize(example.code), normalize(example.expected)], name);
        record.examples.push(name);
      }
      const practice = lesson.locator('.dsa-practice');
      const links = practice.locator('a[href^="https://leetcode.com/problems/"]');
      assert.equal(await links.count(), 6);
      assert.equal(await practice.locator('details[open]').count(), 0);
      for (let index = 0; index < await links.count(); index++) {
        assert.equal(await links.nth(index).getAttribute('target'), '_blank');
        assert((await links.nth(index).getAttribute('rel')).includes('noreferrer'));
      }
      const exercises = lesson.locator('.lesson-check').filter({ has: page.locator('p > strong', { hasText: /^[1-6]\. / }) });
      assert.equal(await exercises.count(), 6);
      assert.equal(await exercises.locator('details[open]').count(), 0);
      for (let index = 0; index < 6; index++) {
        const answers = exercises.nth(index).locator('summary');
        assert.equal(await answers.count(), 2);
        await answers.nth(1).focus();
        await page.keyboard.press('Enter');
        assert.equal(await exercises.nth(index).locator('details[open]').count(), 1);
        await page.keyboard.press('Enter');
      }
      await practice.locator('.dsa-practice__extension > summary').focus();
      await page.keyboard.press('Enter');
      assert(await practice.getByRole('link', { name: /Max Points on a Line/ }).isVisible());
      await page.keyboard.press('Enter');
      await capture(lesson.locator('.lesson-intro'), 'intro');
      await capture(lesson.locator('.lesson-sources'), 'sources');
      const overflow = await lesson.locator('.geometry-lab, .geometry-inline').evaluateAll(elements => elements.filter(element => element.scrollWidth > element.clientWidth + 1).map(element => element.getAttribute('aria-label')));
      assert.deepEqual(overflow, []);
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth + 1), false);
      record.svgTextOutsideViewBox = await lesson.locator('.geometry-plane').evaluateAll(figures => figures.flatMap((figure, index) => [...figure.querySelectorAll('text')].flatMap(text => {
        const box = text.getBBox();
        return box.x < 0 || box.y < 0 || box.x + box.width > 316 || box.y + box.height > 320 ? [{ index, text: text.textContent, box: { x: box.x, y: box.y, width: box.width, height: box.height } }] : [];
      })));
      assert.deepEqual(record.svgTextOutsideViewBox, []);
      results.push(record);
      await page.close();
    }
    assert.deepEqual(errors, []);
    const result = { checkedAt: new Date().toISOString(), results, errors };
    fs.writeFileSync(path.join(directory, 'results.json'), JSON.stringify(result, null, 2));
    console.log(JSON.stringify(result, null, 2));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

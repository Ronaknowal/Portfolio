const assert = require('node:assert/strict');
const fs = require('node:fs');
const { createHash } = require('node:crypto');
const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const output = 'scratch/geometry-trigonometry-independent';
fs.mkdirSync(output, { recursive: true });
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1050 }, reducedMotion: 'reduce' });
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/geometry-trigonometry-coordinate-reasoning?module=math-foundations', { waitUntil: 'domcontentloaded' });
      const lesson = page.locator('.geometry-lesson');
      await lesson.locator('h2').last().waitFor();
      await page.evaluate(() => document.fonts.ready);
      const region = name => lesson.getByRole('region', { name, exact: true });
      const captures = [];
      async function shot(target, name) {
        await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = 'hidden'));
        const file = `${output}/${name}-${width}.png`;
        await target.screenshot({ path: file }); captures.push(file);
        await page.locator('.learn-nav').evaluateAll(nodes => nodes.forEach(node => node.style.visibility = ''));
      }
      async function press(target, name) {
        const button = target.getByRole('button', { name, exact: true });
        await button.focus();
        assert(await button.evaluate(node => document.activeElement === node && getComputedStyle(node).outlineStyle !== 'none'));
        await page.keyboard.press('Enter');
      }
      const arc = region('Angle and arc investigation');
      await arc.getByRole('slider', { name: 'Radius', exact: true }).fill('3');
      await arc.getByRole('slider', { name: 'Sweep', exact: true }).fill('270');
      const geometry = await arc.locator('path.gold').evaluate(node => {
        const length = node.getTotalLength();
        return { length, samples: Array.from({ length: 25 }, (_, i) => { const p = node.getPointAtLength(length * i / 24); return [p.x, p.y]; }) };
      });
      assert(Math.abs(geometry.length - 117 * 3 * Math.PI / 2) < 0.2);
      geometry.maximumRadiusError = Math.max(...geometry.samples.map(([x, y]) => Math.abs(Math.hypot(x - 160, y - 160) - 117)));
      fs.writeFileSync(`${output}/arc-sampling-${width}.json`, JSON.stringify(geometry, null, 2));
      assert(geometry.maximumRadiusError < 0.05, JSON.stringify(geometry));
      assert(geometry.samples[1][1] < 160, 'positive sweep begins counterclockwise');
      assert(Math.abs(geometry.samples.at(-1)[0] - 160) < 0.001 && Math.abs(geometry.samples.at(-1)[1] - 277) < 0.001);
      await shot(arc, 'reflex-sector'); await press(arc, 'Reset arc');
      const similarity = region('Similar triangles investigation');
      await similarity.getByRole('combobox').selectOption('equal-legs');
      await similarity.getByRole('slider').fill('0.75');
      assert((await similarity.locator('.readout').innerText()).includes('Area: 4.5 → 2.5313'));
      await shot(similarity, 'smaller-similar-triangle'); await press(similarity, 'Reset similarity');
      const circle = region('Circle components investigation');
      await circle.getByRole('slider').fill('225');
      assert((await circle.locator('.readout').innerText()).includes('(-0.7071, -0.7071)'));
      const endpoint = await circle.locator('svg').first().locator('circle.point').evaluate(node => [Number(node.getAttribute('cx')), Number(node.getAttribute('cy'))]);
      assert(Math.abs(endpoint[0] - (165 - 108 / Math.sqrt(2))) < 1e-10);
      assert(Math.abs(endpoint[1] - (155 + 108 / Math.sqrt(2))) < 1e-10);
      await shot(circle, 'third-quadrant-components');
      await circle.getByRole('slider').fill('-90');
      assert((await circle.locator('.readout').innerText()).includes('undefined: cosine is zero'));
      await press(circle, 'Reset circle');
      const bearing = region('Bearing investigation');
      await bearing.getByLabel('Bearing x').fill('-5'); await bearing.getByLabel('Bearing y').fill('-5');
      await press(bearing, 'Apply coordinates');
      assert((await bearing.locator('.readout').innerText()).includes('Bearing = -135°'));
      await bearing.getByLabel('Bearing x').fill('bad'); await press(bearing, 'Apply coordinates');
      assert((await bearing.getByRole('alert').innerText()).includes('last valid point is retained'));
      assert((await bearing.locator('.readout').innerText()).includes('Active point: (-5, -5)'));
      await shot(bearing, 'invalid-input-retains-bearing');
      await press(bearing, '(0, 0)');
      assert((await bearing.locator('.readout').innerText()).includes('origin has no direction'));
      await press(bearing, '(-3, 4)');

      const ssa = lesson.locator('.geometry-figure').nth(2);
      const samples = await ssa.locator('path.axis.dashed').evaluate(node => {
        const length = node.getTotalLength();
        return Array.from({ length: 21 }, (_, i) => { const point = node.getPointAtLength(length * i / 20); return [point.x, point.y]; });
      });
      const center = [35 + 90 * Math.sqrt(3), 85];
      for (const point of samples) assert(Math.abs(Math.hypot(point[0] - center[0], point[1] - center[1]) - 126) < 0.04, JSON.stringify(point));
      assert((await ssa.innerText()).includes('3.7613') && (await ssa.innerText()).includes('13.5592'));
      await shot(ssa, 'actual-ssa-circle');
      const frame = region('Coordinate frame investigation');
      for (const [name, value] of [['Point x', 3], ['Point y', 1], ['Origin x', 1], ['Origin y', 2], ['Rotation angle', 90]]) await frame.getByRole('slider', { name, exact: true }).fill(String(value));
      assert((await frame.locator('.readout').innerText()).includes('Local coordinates q = (-1, -2)'));
      await shot(frame, 'passive-quarter-turn');
      await frame.getByRole('combobox').selectOption('active');
      assert((await frame.locator('.readout').innerText()).includes('Rotated world point P′ = (2, 4)'));
      const bluePoint = await frame.locator('circle[fill="#8db9df"]').evaluate(node => [Number(node.getAttribute('cx')), Number(node.getAttribute('cy'))]);
      assert.deepEqual(bluePoint, [191, 108]);
      await shot(frame, 'active-quarter-turn');
      await frame.getByRole('slider', { name: 'Point x', exact: true }).fill('1');
      await frame.getByRole('slider', { name: 'Point y', exact: true }).fill('2');
      assert((await frame.locator('svg').textContent()).includes('P = P′ = O'));
      await press(frame, 'Reset frame');
      for (const index of [0, 1, 3, 4]) await shot(lesson.locator('.geometry-figure').nth(index), 'inline-' + index);
      const practice = lesson.locator('.lesson-check').filter({ has: page.getByRole('heading', { name: 'G. An independently changed ambiguous triangle', exact: true }) });
      await practice.locator('summary').last().focus(); await page.keyboard.press('Enter');
      assert((await practice.innerText()).includes('4√3±3'));
      await shot(practice, 'changed-ssa-practice');
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1));
      const fonts = await page.evaluate(() => [...document.fonts].filter(font => font.status === 'loaded').map(font => font.family));
      assert(fonts.some(font => font.includes('Space Grotesk')));
      assert.equal(await lesson.locator('.katex-error').count(), 0);
      assert.deepEqual(errors, []);
      records.push({ width, captures, arcGeometry: geometry, ssaCircleSamples: samples, fonts, errors });
      console.log('Geometry independent actual diagram check passed', width);
      await page.close();
    }
    const paths = ['src/learn/data/topics/geometry-trigonometry-coordinate-reasoning.jsx', 'src/learn/data/geometry-trigonometry-models.js', 'src/learn/data/geometry-trigonometry-examples.js', 'src/learn/components/lesson-labs/GeometryTrigonometryLabs.jsx', 'src/learn/components/lesson-labs/geometry-trigonometry-labs.css'];
    const sources = paths.map(path => ({ path, sha256: createHash('sha256').update(fs.readFileSync(path)).digest('hex') }));
    fs.writeFileSync(output + '/browser-results.json', JSON.stringify({ reviewedAt: new Date().toISOString(), passed: true, sources, records }, null, 2) + '\n');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

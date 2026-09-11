const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const directory = path.resolve('scratch/geometry-trigonometry-browser');
fs.mkdirSync(directory, { recursive: true });

(async () => {
  const model = await import('../src/learn/data/geometry-trigonometry-models.js');
  const { geometryTrigonometryExamples: examples } = await import('../src/learn/data/geometry-trigonometry-examples.js');
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  const records = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1080 }, reducedMotion: 'reduce' });
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/geometry-trigonometry-coordinate-reasoning', { waitUntil: 'domcontentloaded', timeout: 60000 });
      const lesson = page.locator('.geometry-lesson');
      await lesson.waitFor({ timeout: 60000 });
      await page.evaluate(() => document.fonts.ready);
      assert(await page.evaluate(() => document.fonts.check('16px "Space Grotesk"')));
      const capture = async (locator, name) => {
        await locator.first().evaluate(node => scrollTo({ top: node.getBoundingClientRect().top + scrollY - 84, behavior: 'instant' }));
        await page.screenshot({ path: path.join(directory, `${name}-${width}.png`) });
      };
      const slider = async (label, value) => {
        const control = lesson.getByRole('slider', { name: label, exact: true });
        await control.fill(String(value)); await control.dispatchEvent('input');
      };
      const lab = name => lesson.locator(`.geometry-lab[aria-label="${name}"]`);
      assert.equal(await lesson.locator('h2').count(), 10);
      assert.equal(await lesson.locator('.geometry-lab').count(), 5);
      assert.equal(await lesson.locator('.geometry-figure').count(), 5);
      const broken = await lesson.locator('nav a').evaluateAll(links => links.map(a => a.hash).filter(hash => !document.getElementById(hash.slice(1))));
      assert.deepEqual(broken, []);
      for (let i = 0; i < 10; i++) await capture(lesson.locator('h2').nth(i), `reading-${i + 1}`);
      for (let i = 0; i < 5; i++) await capture(lesson.locator('.geometry-figure').nth(i), `inline-${i + 1}`);
      for (let i = 0; i < 5; i++) {
        await capture(lesson.locator('.geometry-lab').nth(i), `initial-lab-${i + 1}`);
        await capture(lesson.locator('.geometry-lab').nth(i).locator('svg').first(), `initial-figure-${i + 1}`);
      }

      let controlStates = 0;
      const arc = lab('Angle and arc investigation');
      for (const radius of [1, 2, 3]) for (const degrees of [15, 60, 180, 270, 330]) {
        await slider('Radius', radius); await slider('Sweep', degrees);
        const state = model.arcState(radius, degrees);
        assert((await arc.locator('.readout').innerText()).includes(`Sector area = r²θ/2 = ${model.formatGeometry(state.area)}`));
        const geometry = await arc.locator('svg').evaluate(svg => ({ radius: Number(svg.querySelector('circle').getAttribute('r')), endpoint: [...svg.querySelectorAll('circle')].at(-1).getAttribute('cy'), path: svg.querySelector('path.gold').getAttribute('d') }));
        assert.equal(geometry.radius, radius * 39);
        assert(Math.abs(Number(geometry.endpoint) - (160 - radius * 39 * state.sine)) < 1e-10);
        controlStates++;
      }
      await capture(arc.locator('svg'), 'arc-large-positive');
      await arc.getByRole('button', { name: 'Reset arc' }).focus(); await page.keyboard.press('Enter');
      assert.equal(await arc.getByRole('slider', { name: 'Radius', exact: true }).inputValue(), '2');

      const similar = lab('Similar triangles investigation');
      for (const shape of ['3-4-5', '5-12-13', 'equal-legs']) {
        await similar.getByRole('combobox').selectOption(shape);
        for (const scale of [.5, 1, 1.75, 3]) {
          await slider('Positive scale', scale);
          const state = model.similarityState(shape, scale);
          assert((await similar.locator('.readout').innerText()).includes(`${model.formatGeometry(state.originalArea)} → ${model.formatGeometry(state.scaledArea)}`));
          controlStates++;
        }
      }
      await capture(similar, 'similar-equal-scaled');
      await similar.getByRole('button', { name: 'Reset similarity' }).click();

      const circle = lab('Circle components investigation');
      for (let degrees = -360; degrees <= 360; degrees += 15) {
        await slider('Signed angle', degrees);
        const state = model.circleComponents(degrees);
        assert((await circle.locator('.readout').innerText()).includes(`(${model.formatGeometry(state.cosine)}, ${model.formatGeometry(state.sine)})`));
        if (degrees % 180 === 90 || degrees % 180 === -90) assert((await circle.locator('.readout').innerText()).includes('undefined: cosine is zero'));
        controlStates++;
      }
      await slider('Signed angle', 150); await capture(circle.locator('svg').first(), 'circle-quadrant-two');
      await capture(circle.locator('svg').nth(1), 'circle-linked-traces');
      await circle.getByRole('button', { name: 'Reset circle' }).click();
      await circle.getByRole('slider').focus(); await page.keyboard.press('ArrowRight');
      assert.equal(await circle.getByRole('slider').inputValue(), '45');

      const bearing = lab('Bearing investigation');
      for (const pair of [[3, 4], [-3, 4], [-3, -4], [3, -4], [-4, 0], [0, 4], [0, 0]]) {
        await bearing.getByRole('button', { name: `(${pair.join(', ')})`, exact: true }).click();
        const state = model.bearingState(...pair);
        assert((await bearing.locator('.readout').innerText()).includes(state.quadrant)); controlStates++;
      }
      await capture(bearing, 'bearing-origin');
      await bearing.getByRole('button', { name: '(-3, 4)', exact: true }).click();
      for (const value of ['', '2.5', 'Infinity', '1e-200', '7']) {
        await bearing.getByRole('textbox', { name: 'Bearing x' }).fill(value);
        await bearing.getByRole('button', { name: 'Apply coordinates' }).click();
        assert.equal(await bearing.getByRole('alert').count(), 1);
        assert((await bearing.locator('.readout').innerText()).includes('Active point: (-3, 4)')); controlStates++;
      }
      await capture(bearing, 'bearing-invalid-retained');
      await bearing.getByRole('textbox', { name: 'Bearing x' }).fill('-5');
      await bearing.getByRole('textbox', { name: 'Bearing y' }).fill('-5');
      await bearing.getByRole('button', { name: 'Apply coordinates' }).focus(); await page.keyboard.press('Enter');
      assert((await bearing.locator('.readout').innerText()).includes('-135°'));
      assert.equal(await bearing.getByRole('alert').count(), 0);

      const frame = lab('Coordinate frame investigation');
      for (const mode of ['passive', 'active']) {
        await frame.getByRole('combobox').selectOption(mode);
        for (const degrees of [-180, -90, -30, 0, 30, 90, 180]) {
          await slider('Rotation angle', degrees);
          const state = model.frameState(4, 2, 1, -1, degrees, mode);
          const readout = await frame.locator('.readout').innerText();
          assert(readout.includes(`= (${state.result.map(value => model.formatGeometry(value)).join(', ')})`));
          assert(readout.includes('reconstructed in world = (4, 2)')); controlStates++;
        }
      }
      await slider('Point x', 3); await slider('Point y', 1); await slider('Origin x', 1); await slider('Origin y', 2); await slider('Rotation angle', 90);
      assert((await frame.locator('.readout').innerText()).includes('Rotated world point P′ = (2, 4)'));
      await capture(frame.locator('svg'), 'frame-active-changed');
      await frame.getByRole('combobox').selectOption('passive');
      assert((await frame.locator('.readout').innerText()).includes('Local coordinates q = (-1, -2)'));
      await capture(frame.locator('svg'), 'frame-passive-changed');
      await frame.getByRole('combobox').selectOption('active');
      await slider('Point x', -4); await slider('Point y', -4); await slider('Origin x', 2); await slider('Origin y', 2); await slider('Rotation angle', 135);
      await capture(frame.locator('svg'), 'frame-extreme-label');
      const labelBounds = await frame.locator('svg text').evaluateAll(nodes => nodes.map(node => { const box=node.getBBox(); return { text:node.textContent,x:box.x,y:box.y,right:box.x+box.width,bottom:box.y+box.height }; }));
      assert.deepEqual(labelBounds.filter(box => box.x < 0 || box.right > 330 || box.y < 0 || box.bottom > 325), []);
      await slider('Point x', 2); await slider('Point y', 2);
      assert((await frame.locator('svg').textContent()).includes('P = P′ = O'));
      await capture(frame.locator('svg'), 'frame-coincident');
      await frame.getByRole('button', { name: 'Reset frame' }).click();
      assert.equal(await frame.getByRole('combobox').inputValue(), 'passive');
      assert.equal(await frame.getByRole('slider', { name: 'Origin y', exact: true }).inputValue(), '-1');

      const checkpoints = lesson.locator('div.lesson-check');
      assert.equal(await checkpoints.count(), 2);
      for (let i = 0; i < 2; i++) {
        assert((await checkpoints.nth(i).locator('p').first().innerText()).length > 100);
        await checkpoints.nth(i).getByText('Show explanation', { exact: true }).click();
        assert((await checkpoints.nth(i).locator('details').innerText()).includes(i === 0 ? '12 cm²' : 'cancels the −2'));
      }
      const practices = lesson.locator('section.lesson-check');
      assert.equal(await practices.count(), 12);
      for (let i = 0; i < 12; i++) {
        await practices.nth(i).getByText('Hint', { exact: true }).click();
        await practices.nth(i).getByText('Explained solution', { exact: true }).click();
        assert((await practices.nth(i).locator('details').last().innerText()).length > 120);
      }
      await capture(practices.nth(6), 'practice-ssa'); await capture(practices.nth(7), 'practice-frame'); await capture(practices.nth(10), 'practice-measurement');
      await lesson.getByText('Deeper connection: angle addition and composition', { exact: true }).click();
      await capture(lesson.locator('details').filter({ has: page.getByText('Deeper connection: angle addition and composition', { exact: true }) }), 'deeper-composition');
      for (const example of examples) {
        const container = lesson.locator('.python-example').filter({ has: page.getByRole('heading', { name: example.title, exact: true }) });
        const text = (await container.innerText()).replace(/\r\n/g, '\n');
        assert(text.includes(example.code.trim()), 'Actual complete code '+example.id);
        assert(text.includes(example.expected), 'Actual exact output '+example.id);
        assert((await lesson.innerText()).includes(example.question), 'Visible question '+example.id);
      }
      assert.equal(await lesson.locator('.python-example').count(), 9);
      const directRejections = await page.evaluate(async () => {
        const m = await import('/src/learn/data/geometry-trigonometry-models.js');
        return [1e-200,1e-323].map(angle => { try { m.circleComponents(angle); return false; } catch(error) { return error instanceof RangeError; } });
      });
      assert.deepEqual(directRejections,[true,true]);
      await capture(lesson.locator('.python-example').nth(5), 'native-ssa');
      await capture(lesson.locator('.python-example').nth(8), 'native-links');
      const scrollCode = lesson.locator('.python-example').nth(5).locator('div[style*="white-space: pre"]').first();
      await scrollCode.focus(); await page.keyboard.press('ArrowRight'); await page.waitForTimeout(120);
      const codeGeometry = await scrollCode.evaluate(node => ({ width: node.clientWidth, content: node.scrollWidth, left: node.scrollLeft, focused: node === document.activeElement }));
      if (codeGeometry.content > codeGeometry.width) assert(codeGeometry.focused && codeGeometry.left > 0);
      await capture(lesson.locator('.lesson-sources'), 'sources');
      const mathWidths = await lesson.locator('.katex-display').evaluateAll(nodes => nodes.map((node,index) => ({ index, width: node.clientWidth, content: node.scrollWidth })));
      fs.writeFileSync(path.join(directory, `math-${width}.json`), JSON.stringify(mathWidths, null, 2));
      for (let i=0; i<mathWidths.length; i++) await capture(lesson.locator('.katex-display').nth(i), `equation-${i}`);
      assert.deepEqual(mathWidths.filter(row => row.content > row.width + 2), [], 'Math fits reading width');
      assert.equal(await lesson.locator('.katex-error').count(), 0);
      assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'No document overflow');
      assert.deepEqual(errors, []);
      records.push({ width, fonts: true, sections: 10, labs: 5, figures: 5, controlStates, actualKeyboard: true, invalidRetention: true, examples: 9, checkpoints: 2, independentPractice: 12, mathWidths, codeGeometry, labelBounds, directRejections, errors });
      await page.close();
    }
  } finally { await browser.close(); }
  const result = { at: new Date().toISOString(), passed: true, records };
  fs.writeFileSync(path.join(directory, 'results.json'), JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result));
})().catch(error => { console.error(error); process.exit(1); });

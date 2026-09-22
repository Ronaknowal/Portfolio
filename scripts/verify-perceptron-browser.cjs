// Topic-owned production checks. The increment owner supplies the built preview.
// PLAYWRIGHT_PACKAGE=<package path> LEARNING_BASE_URL=http://127.0.0.1:4194
// node scripts/verify-perceptron-browser.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');

async function runPerceptronChecks(page, { capture = async () => {}, touch = false } = {}) {
  const groups = [];
  const text = locator => locator.innerText();
  const contains = async (locator, expected) => assert.ok((await text(locator)).includes(expected), `Missing ${expected}`);
  const edit = async (panel, name, value) => {
    const field = panel.getByRole('textbox', { name: `${name} exact value`, exact: true });
    await field.fill(String(value));
    await field.press('Tab');
  };
  const geometry = page.locator('#perceptron-geometry-lab');
  const xor = page.locator('#perceptron-xor-lab');
  const sensitivity = page.locator('#perceptron-activation-lab');
  await geometry.waitFor();
  await contains(geometry.getByTestId('geometry-result'), 'Unscaled score 4 → scaled score 4');
  await contains(xor.getByTestId('xor-result'), 'Outputs: 0, 1, 1, 1');
  await contains(sensitivity.getByTestId('sensitivity-result'), 'weight × slope = 0.5');
  const geometryBar = geometry.locator('.perceptron-bar-fill').first();
  const xorBar = xor.locator('.perceptron-bar-fill').nth(1);
  const barWidth = locator => locator.evaluate(element => parseFloat(element.style.width));
  await contains(geometry, 'scale runs from −16 to 16');
  await contains(xor, 'scale runs from −8 to 8');
  assert.equal(await barWidth(geometryBar), 9.375);
  await edit(geometry, 'Weight w₁', 3);
  assert.equal(await barWidth(geometryBar), 18.75);
  await edit(geometry, 'Weight w₁', 1.5);
  assert.equal(await barWidth(xorBar), 6.25);
  await edit(xor, 'Second output coefficient', -2);
  assert.equal(await barWidth(xorBar), 12.5);
  await edit(xor, 'Second output coefficient', -1);
  assert.equal(await page.locator('.katex-error').count(), 0);
  for (const panel of [geometry, xor, sensitivity]) {
    assert.equal(await panel.getByRole('button', { name: /predict|commit|reveal|submit/i }).count(), 0);
    assert.equal(await panel.locator('input[placeholder*="predict" i]').count(), 0);
  }
  groups.push('Initial results, rendered mathematics and no learner-prediction controls');

  await edit(geometry, 'Common scale c', 2);
  await contains(geometry.getByTestId('geometry-result'), 'Unscaled score 4 → scaled score 8');
  await contains(geometry.getByTestId('geometry-result'), 'distance 1.6; sigmoid 0.999665');
  await geometry.getByRole('button', { name: 'Pin current scaled case', exact: true }).click();
  await edit(geometry, 'Input x₁', 1);
  await contains(geometry.getByTestId('geometry-pinned'), 'Score 8, distance 1.6');
  await geometry.getByRole('button', { name: 'Start at boundary tie', exact: true }).click();
  await contains(geometry.getByTestId('geometry-result'), 'Hard output 0; distance 0; sigmoid 0.5');
  await edit(geometry, 'Input x₂', .5);
  await contains(geometry.getByTestId('geometry-result'), 'Hard output 1');
  await edit(geometry, 'Input x₂', 1.5);
  await contains(geometry.getByTestId('geometry-result'), 'Hard output 0');
  await geometry.getByRole('button', { name: 'Zero weight vector', exact: true }).click();
  await contains(geometry.getByTestId('geometry-result'), 'distance undefined');
  await capture('geometry-zero-weight', geometry);
  await geometry.getByRole('button', { name: 'Reset evidence', exact: true }).click();
  const before = await text(geometry.getByTestId('geometry-result'));
  await edit(geometry, 'Weight w₁', 'bad');
  assert.equal(await text(geometry.getByTestId('geometry-result')), before);
  assert.equal(await geometry.locator('[aria-invalid=true]').count(), 1);
  await contains(geometry, 'Last valid result');
  await edit(geometry, 'Weight w₁', 1.5);
  groups.push('Geometry rescale, pinned identity, crossing, null, invalid text and reset');

  // These are real keyboard events on the native slider, including every endpoint.
  for (const panel of [geometry, xor, sensitivity]) {
    for (const slider of await panel.getByRole('slider').all()) {
      const min = Number(await slider.getAttribute('min'));
      const max = Number(await slider.getAttribute('max'));
      await slider.focus();
      await slider.press('Home');
      assert.equal(Number(await slider.inputValue()), min);
      await slider.press('End');
      assert.equal(Number(await slider.inputValue()), max);
      await slider.press('ArrowLeft');
      const interior = Number(await slider.inputValue());
      assert.ok(interior < max && interior > min);
      const control = slider.locator('..');
      assert.ok(Math.abs(Number(await control.getByRole('textbox').inputValue()) - interior) < 1e-12);
    }
  }
  await geometry.getByRole('button', { name: 'Reset evidence', exact: true }).click();
  const slider = geometry.getByRole('slider', { name: 'Input x₁', exact: true });
  await slider.evaluate(element => element.scrollIntoView({ block: 'center', behavior: 'instant' }));
  await page.waitForTimeout(300);
  const box = await slider.boundingBox();
  assert.ok(await slider.evaluate(element => {
    const rect = element.getBoundingClientRect();
    return document.elementFromPoint(rect.x + rect.width * .75, rect.y + rect.height / 2) === element;
  }), 'The real slider must be the pointer target after scrolling settles.');
  await page.mouse.move(box.x + box.width * .75, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width * .27, box.y + box.height / 2, { steps: 8 });
  await page.mouse.up();
  assert.ok(Number(await slider.inputValue()) < 0, `Pointer drag ended at ${await slider.inputValue()}`);
  await contains(geometry.getByTestId('geometry-result'), 'Hard output 0');
  if (touch) {
    await page.touchscreen.tap(box.x + box.width * .65, box.y + box.height / 2);
    assert.ok(Number(await slider.inputValue()) > 0, 'A touch tap updates the native range control.');
    await contains(geometry.getByTestId('geometry-result'), 'Hard output 1');
  }
  await geometry.getByRole('button', { name: 'Reset evidence', exact: true }).click();
  groups.push('Every native slider endpoint/interior, exact numeric twins and actual pointer gesture');

  await xor.getByRole('button', { name: 'Reset XOR', exact: true }).click();
  await edit(xor, 'Second output coefficient', -2);
  await contains(xor.getByTestId('xor-result'), 'Outputs: 0, 1, 1, 0');
  await edit(xor, 'Second hidden bias', -.5);
  await edit(xor, 'Second output coefficient', '-4/3');
  await contains(xor.getByTestId('xor-result'), 'Outputs: 0, 0.333333, 0.333333, 0');
  assert.ok(Math.abs(Number(await xor.getByRole('slider', { name: 'Second output coefficient', exact: true }).inputValue()) + 4/3) < 1e-12);
  await capture('xor-one-corner-repair', xor);
  await edit(xor, 'Second hidden bias', -3);
  await edit(xor, 'Second output coefficient', -1);
  const nullBefore = await text(xor.getByTestId('xor-result'));
  await edit(xor, 'Second output coefficient', -2);
  assert.equal(await text(xor.getByTestId('xor-result')), nullBefore);
  await xor.getByRole('button', { name: 'Reset XOR', exact: true }).click();
  groups.push('XOR full repair, rational one-corner failure and inactive-feature null');

  await sensitivity.getByRole('button', { name: 'Reset sensitivity', exact: true }).click();
  await edit(sensitivity, 'Incoming scalar weight', -2);
  await contains(sensitivity.getByTestId('sensitivity-result'), 'weight × slope = −2 (negative)');
  await sensitivity.getByRole('combobox', { name: 'Activation', exact: true }).selectOption('sigmoid');
  await edit(sensitivity, 'Operating point z', 0);
  await edit(sensitivity, 'Incoming scalar weight', 4);
  await contains(sensitivity.getByTestId('sensitivity-result'), 'weight × slope = 1 (at least 1)');
  await sensitivity.getByRole('checkbox', { name: 'Include deeper functions from §7', exact: true }).check();
  await sensitivity.getByRole('combobox', { name: 'Activation', exact: true }).selectOption('silu');
  await edit(sensitivity, 'Operating point z', -2);
  await edit(sensitivity, 'Incoming scalar weight', 1);
  await contains(sensitivity.getByTestId('sensitivity-result'), 'slope = −0.090784');
  await capture('silu-negative-slope', sensitivity);
  await sensitivity.getByRole('combobox', { name: 'Activation', exact: true }).selectOption('relu');
  await edit(sensitivity, 'Operating point z', 0);
  await contains(sensitivity, 'no ordinary derivative at zero');
  await sensitivity.getByRole('button', { name: 'Reset sensitivity', exact: true }).click();
  groups.push('Activation weight/sign distinction, sigmoid compensation, SiLU negative slope and ReLU corner');

  const digits = page.locator('#perceptron-digits-figure');
  await digits.getByRole('button', { name: /Digit 7/ }).click();
  await contains(digits, 'Digit 7 · source ID 8');
  const comparison = page.locator('#perceptron-comparison-figure');
  await comparison.getByRole('combobox', { name: 'Recorded initialization seed', exact: true }).selectOption('3');
  await contains(comparison, '116/120');
  await comparison.getByRole('checkbox', { name: /Zoom count axis/ }).check();
  await contains(comparison, 'Count scale: 116–120');
  await capture('recorded-comparison-seed-3-zoom', comparison);
  await comparison.getByRole('checkbox', { name: /Zoom count axis/ }).uncheck();
  await comparison.getByRole('combobox', { name: 'Recorded initialization seed', exact: true }).selectOption('1');
  groups.push('Observed digit identity and recorded-seed/count inspection');

  for (const id of ['perceptron-weighted-figure','perceptron-geometry-lab','perceptron-xor-figure','perceptron-xor-lab','perceptron-activation-lab','perceptron-digits-figure','perceptron-comparison-figure','perceptron-swiglu-figure','perceptron-triangle-figure']) {
    const panel=page.locator(`#${id}`);
    await capture(id,panel);
    const layout=await panel.evaluate(element=>({width:element.getBoundingClientRect().width,overflow:element.scrollWidth-element.clientWidth}));
    assert.ok(layout.width>200);
    assert.ok(layout.overflow<=2,`${id} overflows by ${layout.overflow}px`);
  }
  const duplicateIds=await page.evaluate(()=>{const ids=[...document.querySelectorAll('[id]')].map(e=>e.id);return ids.filter((id,index)=>ids.indexOf(id)!==index);});
  assert.deepEqual(duplicateIds,[]);
  assert.ok(await page.locator('.python-example').count()===3);
  groups.push('All representation bounds, unique IDs and complete displayed programs');
  return groups;
}
module.exports={runPerceptronChecks};

if(require.main===module){
  (async()=>{
    const {chromium}=require(process.env.PLAYWRIGHT_PACKAGE||'playwright');
    const browser=await chromium.launch({channel:'msedge',headless:true});
    const evidence='docs/teaching/evidence/perceptron-browser',screenshots=[];fs.mkdirSync(evidence,{recursive:true});
    const results=[];
    try{
      for(const width of [1366,390,320]){
        const page=await browser.newPage({viewport:{width,height:1000},reducedMotion:'reduce',hasTouch:width<600}),errors=[];
        page.on('pageerror',error=>errors.push(error.message));
        const fontManifest='scratch/kmeans-revision-review/fonts/manifest.json';
        if(fs.existsSync(fontManifest)){
          const fonts=JSON.parse(fs.readFileSync(fontManifest,'utf8'));
          await page.route('https://fonts.googleapis.com/**',route=>route.fulfill({path:fonts.stylesheet,contentType:'text/css'}));
          await page.route('https://fonts.gstatic.com/**',route=>fonts.files[route.request().url()]?route.fulfill({path:fonts.files[route.request().url()],contentType:'font/ttf'}):route.continue());
        }
        await page.goto(`${process.env.LEARNING_BASE_URL||'http://127.0.0.1:4194'}/learn/path/full-curriculum/perceptrons-neurons-activation-functions?module=deep-learning-fundamentals`);
        await page.locator('#perceptron-geometry-lab').waitFor();await page.evaluate(()=>document.fonts.ready);
        const groups=await runPerceptronChecks(page,{touch:width<600,capture:async(name,locator)=>{
          const bounds=await locator.boundingBox();await page.setViewportSize({width,height:Math.max(1000,Math.ceil(bounds.height)+220)});
          await locator.evaluate(element=>element.scrollIntoView({block:'center',behavior:'instant'}));await page.waitForTimeout(120);
          const filename=path.join(evidence,`${width}-${name}.png`);await locator.screenshot({path:filename});screenshots.push(filename);await page.setViewportSize({width,height:1000});
        }});
        assert.deepEqual(errors,[]);results.push({width,groups,errors});await page.close();
      }
      const files=['src/learn/data/topics/perceptrons-neurons-activation-functions.jsx','src/learn/data/perceptron-models.js','src/learn/components/lesson-labs/PerceptronLabs.jsx','src/learn/components/lesson-labs/PerceptronFigures.jsx','src/learn/components/lesson-labs/PerceptronShared.jsx','src/learn/components/lesson-labs/perceptron-labs.css'];
      const result={status:'passed',baseUrl:process.env.LEARNING_BASE_URL||'http://127.0.0.1:4194',results,screenshots,sourceHashes:Object.fromEntries(files.map(file=>[file,crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex')])),visualReview:'Captures retained for separate human/agent visual inspection; passing assertions do not certify every label collision.'};
      fs.writeFileSync('docs/teaching/evidence/perceptron-browser.json',JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify({status:'passed',widths:results.map(r=>r.width),groupsPerWidth:results[0].groups.length,screenshots:screenshots.length}));
    }finally{await browser.close();}
  })().catch(error=>{console.error(error);process.exitCode=1;});
}

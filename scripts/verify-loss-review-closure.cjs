// Independent, targeted browser probes for repaired Loss lesson relationships.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const crypto = require('node:crypto');
const {chromium} = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const sources = ['src/learn/components/lesson-labs/LossFunctionsLabs.jsx', 'src/learn/components/lesson-labs/NeuralLessonElements.jsx', 'src/learn/components/lesson-labs/loss-functions-labs.css', 'src/learn/components/lesson-labs/neural-lesson-elements.css'];
const report = {passed:false, base:process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4197', checks:[], captures:[], sourceHashes:Object.fromEntries(sources.map(file=>[file,hash(file)]))};
const receipt = 'docs/teaching/evidence/loss-independent-browser-closure.json';
fs.writeFileSync(receipt, JSON.stringify(report,null,2));
(async()=>{
  const {default: measurements} = await import('../src/learn/data/loss-functions-measurements.js');
  const browser = await chromium.launch({channel:'msedge',headless:true});
  const fonts=JSON.parse(fs.readFileSync('scratch/kmeans-revision-review/fonts/manifest.json','utf8'));
  try {
    const page=await browser.newPage({viewport:{width:1366,height:1000},reducedMotion:'reduce'});
    const errors=[];page.on('pageerror',error=>errors.push(error.message));
    await page.route('https://fonts.googleapis.com/**',route=>route.fulfill({path:fonts.stylesheet,contentType:'text/css'}));
    await page.route('https://fonts.gstatic.com/**',route=>fonts.files[route.request().url()]?route.fulfill({path:fonts.files[route.request().url()],contentType:'font/ttf'}):route.continue());
    await page.goto(`${report.base}/learn/path/full-curriculum/loss-functions-ce-mse-focal-contrastive-triplet?module=deep-learning-fundamentals`,{timeout:60000});
    await page.locator('[data-lab="loss-focal"]').waitFor();await page.evaluate(()=>document.fonts.ready);
    const focal=page.locator('[data-lab="loss-focal"]');
    const field=focal.getByRole('spinbutton',{name:'Number of negatives',exact:true});
    const before=await focal.locator('[data-result="focal"]').innerText();
    await field.fill('1.4');assert.equal(await field.getAttribute('aria-invalid'),'true');
    assert.equal(await focal.locator('[data-result="focal"]').innerText(),before);
    assert.ok((await focal.innerText()).includes('a whole number'));
    await field.press('Tab');assert.equal(await field.inputValue(),'1000');
    await field.fill('90');await focal.getByRole('spinbutton',{name:'Focusing gamma',exact:true}).fill('0');
    assert.ok((await focal.locator('[data-result="focal"]').innerText()).includes('no first-order bias update'));
    report.checks.push('Fractional negative count is rejected locally, retains the last valid derived result, restores on blur; integer 90 reaches BCE cancellation');

    const decision=page.locator('[data-lab="loss-decisions"]');
    const probabilities=measurements.records[0].validation_probabilities;
    const labels=measurements.validation_source_ids.map(id=>measurements.specimens.find(row=>row.source_id===id).digit===9);
    const sorted=[...new Set(probabilities)].sort((a,b)=>a-b), lower=sorted[70],upper=sorted[71];
    const threshold=decision.getByRole('spinbutton',{name:'Decision threshold',exact:true});
    const counts=async()=>decision.locator('.loss-confusion strong').allTextContents();
    await threshold.fill(String((lower+upper)/2));const middle=await counts();
    assert.ok((await decision.innerText()).includes('The same decisions hold for thresholds in ('));
    await threshold.fill(String(upper));assert.deepEqual(await counts(),middle);
    await threshold.fill(String(lower));assert.notDeepEqual(await counts(),middle);
    await threshold.fill('0');assert.ok((await decision.innerText()).includes('The same decisions hold for thresholds in [0,'));
    await threshold.fill('1');assert.equal(Number((await counts())[1])+Number((await counts())[3]),0);
    await threshold.fill('.5');
    for(const cell of ['tn','fp','fn','tp']){
      const index=probabilities.findIndex((p,i)=>(labels[i]?(p>=.5?'tp':'fn'):(p>=.5?'fp':'tn'))===cell);
      assert.ok(index>=0);await decision.getByRole('combobox',{name:'Inspect validation specimen',exact:true}).selectOption(String(index));
      const selected=await decision.locator('.loss-confusion-selected').innerText();
      assert.ok(selected.startsWith({tn:'True negative',fp:'False positive',fn:'False negative',tp:'True positive'}[cell]));
    }
    report.checks.push('Adjacent threshold interval excludes its lower specimen, includes its upper specimen, handles 0/1 endpoints, and all four specimen selections highlight their correct confusion cells');

    const regression=page.locator('[data-lab="loss-regression"]'),dot=regression.locator('.loss-location-dot').last();
    const position=()=>dot.evaluate(element=>parseFloat(element.style.left));
    assert.equal(await position(),55);await regression.getByRole('spinbutton',{name:'Observation 7 value',exact:true}).fill('70');assert.equal(await position(),85);
    await regression.getByRole('button',{name:'All measurements are 3',exact:true}).click();
    assert.deepEqual(await regression.locator('.loss-location-dot').evaluateAll(elements=>elements.map(element=>parseFloat(element.style.left))),Array(7).fill(51.5));
    await regression.getByRole('button',{name:'Reset measurements',exact:true}).click();
    report.checks.push('The selected observation moves from 55% to 85% on a fixed quantitative axis; the equal-observation null aligns every dot');
    const targets=[['locations',regression.locator('figure').first()],['triplet',page.locator('[data-lab="loss-triplet"] figure')],['reductions',page.locator('figure').filter({hasText:'Four times the matrix side'})],['confusion',decision.locator('.loss-confusion')]];
    fs.mkdirSync('docs/teaching/evidence/loss-independent-browser',{recursive:true});
    for(const width of [1366,320])for(const[name,locator]of targets){
      await page.setViewportSize({width,height:1000});const box=await locator.boundingBox();await page.setViewportSize({width,height:Math.max(1000,Math.ceil(box.height)+220)});
      await locator.evaluate(element=>element.scrollIntoView({block:'center',behavior:'instant'}));await page.waitForTimeout(120);
      const path=`docs/teaching/evidence/loss-independent-browser/${width}-${name}.png`;await locator.screenshot({path});report.captures.push(path);
    }
    assert.deepEqual(errors,[]);for(const file of sources)assert.equal(hash(file),report.sourceHashes[file],`${file} changed during the probe`);
    report.passed=true;fs.writeFileSync(receipt,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({passed:true,checks:report.checks.length,captures:report.captures.length}));
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});

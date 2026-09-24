const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const directory='scratch/sampling-measurement-independent';
const fingerprint=path=>({path,sha256:crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex')});
(async()=>{
  const {samplingMeasurementExamples:examples}=await import('../src/learn/data/sampling-measurement-examples.js');
  const sourceRecord=JSON.parse(fs.readFileSync(`${directory}/results.json`,'utf8'));
  const result={at:new Date().toISOString(),sources:sourceRecord.sources,records:[],images:[]};
  result.sources.forEach(row=>assert.equal(fingerprint(row.path).sha256,row.sha256));
  const browser=await chromium.launch({channel:'msedge',headless:true}); result.browser=browser.version();
  try{for(const width of [1440,390,320]){
    const page=await browser.newPage({viewport:{width,height:1050}});await page.routeWebSocket('**',socket=>socket.close());
    const errors=[],failedRequests=[];
    page.on('pageerror',error=>errors.push(error.message));page.on('requestfailed',request=>failedRequests.push(request.url()));
    await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/sampling-measurement-experimental-design?module=math-foundations',{waitUntil:'networkidle'});
    const lesson=page.locator('.sampling-lesson');await lesson.waitFor();await page.evaluate(()=>document.fonts.ready);
    assert(await page.evaluate(()=>document.fonts.check('16px "Space Grotesk"')));
    assert.equal(await page.locator('vite-error-overlay').count(),0);
    assert.equal(await page.locator('.lesson-guide').count(),0);
    const labs=lesson.locator('.sampling-lab');assert.equal(await labs.count(),5);
    assert.equal(await lesson.locator('.sampling-figure').count(),5);
    const programs=await lesson.locator('.python-example').evaluateAll(nodes=>nodes.map(node=>({title:node.querySelector('h3').textContent,
      question:node.previousElementSibling.textContent.replace(/^Before running:\s*/,''),
      blocks:[...node.children].filter(child=>getComputedStyle(child).whiteSpace==='pre').map(child=>[...child.childNodes].filter(n=>n.nodeType===Node.TEXT_NODE).map(n=>n.textContent).join(''))})));
    assert.equal(programs.length,12);
    for(const program of programs){const example=examples.find(row=>row.title===program.title);assert(example);assert.equal(program.question,example.question);assert.equal(program.blocks[0].trim(),example.code.trim());assert.equal(program.blocks[1].trim(),example.expected.trim());}
    const equations=await lesson.locator('.katex-display').evaluateAll(nodes=>nodes.map(node=>({width:node.clientWidth,content:node.scrollWidth})));
    assert.equal(equations.length,9);assert(equations.every(row=>row.content<=row.width+1));
    const anchors=await lesson.locator('.lesson-intro a[href^="#"]').evaluateAll(nodes=>nodes.map(node=>!!document.getElementById(node.hash.slice(1))));
    assert.equal(anchors.length,10);assert(anchors.every(Boolean));
    async function capture(target,name){
      const height=Math.max(1050,Math.ceil((await target.boundingBox()).height)+160);await page.setViewportSize({width,height});
      await target.evaluate(node=>scrollTo({top:scrollY+node.getBoundingClientRect().top-84,behavior:'instant'}));
      const path=`${directory}/${name}-${width}.png`;await target.screenshot({path});result.images.push({...fingerprint(path),captureViewport:{width,height}});
      const bad=await target.locator('svg text').evaluateAll(nodes=>nodes.filter(node=>{const b=node.getBBox(),v=node.ownerSVGElement.viewBox.baseVal;return b.x < -1||b.y < -1||b.x+b.width>v.width+1||b.y+b.height>v.height+1;}).map(node=>node.textContent));
      assert.deepEqual(bad,[],name);await page.setViewportSize({width,height:1050});
    }
    const sample=labs.nth(0);await sample.getByLabel('Available frame',{exact:true}).selectOption('partial');await sample.getByLabel('Sample size',{exact:true}).fill('3');
    assert((await sample.locator('.sampling-readout').innerText()).includes('Variance = 0.55556; bias = -4; MSE = 16.55556'));
    const probabilities=await sample.locator('rect[data-probability]').evaluateAll(nodes=>nodes.map(node=>Number(node.dataset.probability)));
    assert.deepEqual(probabilities,[.25,.25,.25,.25]);await capture(sample,'incomplete-frame-three');
    const weights=labs.nth(1);await weights.getByLabel('Inspect subset',{exact:true}).selectOption('5');
    assert((await weights.locator('.sampling-readout').innerText()).includes('HT mean: sum / fixed N = 13.33333'));
    await capture(weights,'changed-inclusion-contributions');
    await weights.getByLabel('Selection design',{exact:true}).selectOption('uncovered');assert(await weights.getByRole('status').isVisible());
    const units=labs.nth(2);await units.getByLabel('Independent units G',{exact:true}).fill('3');await units.getByLabel('Readings per unit m',{exact:true}).fill('5');await units.getByLabel('Fixed offset b',{exact:true}).fill('0.5');
    assert((await units.locator('.sampling-readout').innerText()).includes('Variance = 1.4;'));
    const bars=await units.locator('svg rect').evaluateAll(nodes=>nodes.map(node=>Number(node.getAttribute('width'))));
    [200*4/3/9,200/15/9].forEach((expected,i)=>assert(Math.abs(bars[i]-expected)<1e-11));
    await capture(units,'changed-units-and-readings');
    const assignment=labs.nth(3);await assignment.getByLabel('Assignment rule',{exact:true}).selectOption('mixed');
    await assignment.getByLabel('Allocation number',{exact:true}).fill('6');assert((await assignment.locator('.sampling-readout').innerText()).includes('Exact assignment variance = 22.55556'));
    assert((await assignment.innerText()).includes('unobserved'));
    const observedTable=assignment.locator('.sampling-table-scroll');
    const hasOverflow=await observedTable.evaluate(node=>node.scrollWidth>node.clientWidth);
    await observedTable.focus();await page.keyboard.press('ArrowRight');
    if(hasOverflow)await page.waitForFunction(()=>document.activeElement.scrollLeft>0);
    const reveal=assignment.getByRole('checkbox');await reveal.focus();await page.keyboard.press('Space');assert(await reveal.isChecked());
    const table=assignment.getByRole('region',{name:'Synthetic science table: both potential outcomes are specified',exact:true});
    assert(await table.isVisible());
    await capture(assignment,'unlike-pairs-observation');
    const factorial=labs.nth(4);await factorial.getByLabel('Interaction contrast',{exact:true}).fill('-3');await factorial.getByLabel('Share with B at one',{exact:true}).fill('0.75');
    await factorial.getByRole('button',{name:'Reveal fourth cell',exact:true}).focus();await page.keyboard.press('Enter');
    assert((await factorial.locator('.sampling-readout').innerText()).includes('A effect in the declared B mixture: -0.25'));
    await capture(factorial,'changed-negative-interaction');
    const resets=['Reset sampling','Reset weights','Reset readings','Reset assignment','Reset factorial'];
    for(let i=0;i<5;i++){await labs.nth(i).getByRole('button',{name:resets[i],exact:true}).focus();await page.keyboard.press('Enter');}
    for(let i=0;i<5;i++)await capture(lesson.locator('.sampling-figure').nth(i),`inline-${i}`);
    const practice=lesson.locator('.lesson-check').filter({hasText:'K. Complete a changed protocol and run'});
    await practice.getByText('Explained solution',{exact:true}).focus();await page.keyboard.press('Enter');
    assert((await practice.innerText()).includes('120 students'));assert((await practice.innerText()).includes('1/6, 1/3 and 1/2'));
    await capture(practice,'student-weighted-protocol');
    const checks=await page.evaluate(async()=>{const m=await import('/src/learn/data/sampling-measurement-models.js');
      const invalid=[()=>m.finiteMoments([,1],[.5,.5]),()=>m.boundedMissingMean([,1]),()=>m.finiteSampleState([0,1e-200],[0],1)].map(fn=>{try{fn();return false;}catch{return true;}});
      return {invalid,constant:m.finiteMoments([1e9,1e9],[.5,.5000000000005]),closeValues:m.finiteMoments([1e9-2**-23,1e9])};});
    assert(checks.invalid.every(Boolean));assert.equal(checks.constant.variance,0);assert.equal(checks.closeValues.variance,2**-48);
    assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));assert.deepEqual(errors,[]);assert.deepEqual(failedRequests,[]);
    result.records.push({width,programs:12,equations:9,investigations:5,inlineFigures:5,keyboardActions:9,modelAmendments:checks,errors,failedRequests});await page.close();
  }result.passed=true;fs.writeFileSync(`${directory}/browser-results.json`,JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify({passed:true,widths:result.records.map(row=>row.width),images:result.images.length}));}
  finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});

const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const directory = 'scratch/random-variables-independent';
const fingerprint = path => ({ path, sha256: crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex') });
const close = (actual, expected) => assert(Math.abs(Number(actual) - expected) < 6e-6, `${actual} != ${expected}`);
async function metric(region, label) {
  return region.locator('.rv-metrics > div').filter({has: region.page().getByText(label, {exact:true})}).locator('dd').innerText();
}

(async () => {
  const { randomVariableExamples } = await import('../src/learn/data/random-variables-examples.js');
  const baseline = JSON.parse(fs.readFileSync(`${directory}/author-baseline.json`, 'utf8'));
  const result = { at:new Date().toISOString(), records:[], images:[], sources:baseline.sources.map(row=>fingerprint(row.path)) };
  const browser = await chromium.launch({channel:'msedge', headless:true});
  result.browser = browser.version();
  try {
    for(const width of [1440,390,320]) {
      const page = await browser.newPage({viewport:{width,height:1050}});
      await page.routeWebSocket('**', socket=>socket.close());
      const errors=[], failedRequests=[];
      page.on('pageerror', error=>errors.push(error.message));
      page.on('requestfailed', request=>failedRequests.push(request.url()));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/random-variables-expectation-covariance?module=math-foundations',{waitUntil:'networkidle'});
      const lesson=page.locator('.random-variable-lesson');
      await lesson.waitFor(); await page.evaluate(()=>document.fonts.ready);
      assert(await page.evaluate(()=>document.fonts.check('16px "Space Grotesk"')));
      assert.equal(await page.locator('vite-error-overlay').count(),0);
      assert.equal(await page.locator('.lesson-guide').count(),0);
      assert.equal(await lesson.locator('.lesson-intro').count(),1);
      const labs=lesson.locator('.rv-lab'); assert.equal(await labs.count(),7);
      const setup=lesson.getByText('save', {exact:false}).filter({hasText:'random_variables.py'});
      assert.equal(await setup.count(),1);
      assert(await setup.isVisible());
      assert(await setup.evaluate(node=>!!(node.compareDocumentPosition(document.querySelector('.python-example')) & Node.DOCUMENT_POSITION_FOLLOWING)));
      const programs=await lesson.locator('.python-example').evaluateAll(nodes=>nodes.map(node=>({
        title:node.querySelector('h3').textContent,
        question:node.previousElementSibling.textContent.replace(/^Before running\.\s*/,''),
        blocks:[...node.children].filter(child=>getComputedStyle(child).whiteSpace==='pre').map(child=>[...child.childNodes].filter(n=>n.nodeType===Node.TEXT_NODE).map(n=>n.textContent).join(''))
      })));
      assert.equal(programs.length,14);
      for(const program of programs) {
        const example=Object.values(randomVariableExamples).find(row=>row.title===program.title);
        assert(example); assert.equal(program.question,example.question);
        assert.equal(program.blocks[0].trim(),example.code.trim()); assert.equal(program.blocks[1].trim(),example.expected.trim());
      }
      const anchors=await lesson.locator('.lesson-intro a[href^="#"]').evaluateAll(nodes=>nodes.map(node=>!!document.getElementById(node.hash.slice(1))));
      assert.equal(anchors.length,12); assert(anchors.every(Boolean));
      assert.equal(await lesson.locator('.katex-error').count(),0);
      const equations=await lesson.locator('.katex-display').evaluateAll(nodes=>nodes.map(node=>({width:node.clientWidth,content:node.scrollWidth})));
      assert.equal(equations.length,17); assert(equations.every(row=>row.content<=row.width+1));
      async function capture(target,name) {
        const height=Math.max(1050,Math.ceil((await target.boundingBox()).height)+160);
        await page.setViewportSize({width,height});
        await target.evaluate(node=>scrollTo({top:scrollY+node.getBoundingClientRect().top-84,behavior:'instant'}));
        const path=`${directory}/${name}-${width}.png`; await target.screenshot({path});
        result.images.push({...fingerprint(path),captureViewport:{width,height}});
        const bad=await target.locator('svg text').evaluateAll(nodes=>nodes.filter(node=>{
          const box=node.getBBox(), vb=node.ownerSVGElement.viewBox.baseVal;
          return box.x < -1 || box.y < -1 || box.x+box.width > vb.width+1 || box.y+box.height > vb.height+1;
        }).map(node=>node.textContent));
        assert.deepEqual(bad,[],`${name} labels`);
        await page.setViewportSize({width,height:1050});
      }
      const mapping=labs.nth(0);
      await mapping.getByLabel('First coin head chance (%)',{exact:true}).fill('17');
      await mapping.getByLabel('Second coin head chance (%)',{exact:true}).fill('35');
      const barMasses=await mapping.locator('.rv-distribution rect').evaluateAll(nodes=>nodes.map(node=>Number(node.getAttribute('height'))/140));
      [.5395,.401,.0595].forEach((mass,i)=>close(barMasses[i],mass));
      await capture(mapping,'changed-outcome-law');
      const loss=labs.nth(1);
      await loss.getByLabel('Constant prediction c',{exact:true}).fill('0.5');
      close(await metric(loss,'Total expected squared loss'),3.25);
      await capture(loss,'changed-squared-loss');
      const joint=labs.nth(2);
      await joint.getByLabel('Joint law',{exact:true}).selectOption('nonlinear');
      await joint.getByLabel('Scale Y',{exact:true}).fill('-2');
      await joint.getByLabel('Shift Y',{exact:true}).fill('1');
      await joint.getByRole('button',{name:'Inspect X 0 Y -1',exact:true}).focus(); await page.keyboard.press('Enter');
      assert((await joint.locator('.rv-result').innerText()).includes('joint mass 0; marginal product 0.222222'));
      close(await metric(joint,'Covariance'),0);
      await capture(joint,'changed-nonlinear-witness');
      const noise=labs.nth(3);
      for(const [label,value] of [['Common amplitude (mV)','1.5'],['Local amplitude (mV)','0.5'],['Coefficient a','0.25'],['Coefficient b','0.75']]) await noise.getByLabel(label,{exact:true}).fill(value);
      close(await metric(noise,'Mean aA+bB (mV)'),17.5);
      close(await metric(noise,'Variance aA+bB (mV²)'),2.40625);
      await capture(noise,'changed-noise-combination');
      const conditional=labs.nth(4);
      await conditional.getByLabel('P(G=2), percent',{exact:true}).fill('25');
      assert.deepEqual((await conditional.locator('.rv-decomposition strong').allTextContents()).map(Number),[4,2]);
      await capture(conditional,'changed-group-mixture');
      await conditional.getByLabel('P(G=2), percent',{exact:true}).fill('0');
      await conditional.getByLabel('Inspect group',{exact:true}).selectOption('2');
      assert((await conditional.locator('.rv-result').innerText()).startsWith('There is no identified'));
      const squared=labs.nth(5);
      await squared.getByLabel('Lower Y endpoint (%)',{exact:true}).fill('4');
      await squared.getByLabel('Upper Y endpoint (%)',{exact:true}).fill('49');
      assert((await squared.locator('.rv-result').innerText()).includes('= 0.5.'));
      const intervals=await squared.locator('svg line.rv-interval').evaluateAll(nodes=>nodes.map(node=>Number(node.getAttribute('x2'))-Number(node.getAttribute('x1'))));
      [57.5,57.5,103.5].forEach((length,i)=>close(intervals[i],length));
      await capture(squared,'changed-two-preimages');
      const sample=labs.nth(6);
      await sample.getByLabel('Readings per dataset n',{exact:true}).fill('9');
      await sample.getByLabel('Population success chance (%)',{exact:true}).fill('35');
      close(await metric(sample,'Independent mean variance'),.35*.65/9);
      close(await metric(sample,'Copied mean variance'),.35*.65);
      await capture(sample,'changed-independent-versus-copies');
      let keyboardChecks=1;
      for(const lab of await labs.all()) {
        await lab.getByRole('button',{name:'Reset investigation',exact:true}).focus(); await page.keyboard.press('Enter');
        const slider=lab.getByRole('slider').first(), before=Number(await slider.inputValue());
        const step=Number(await slider.getAttribute('step')) || 1;
        await slider.focus(); await page.keyboard.press('ArrowRight');
        close(await slider.inputValue(),before+step); keyboardChecks+=2;
      }
      const practice=lesson.locator('.lesson-check').filter({hasText:'Two dice, two new variables'});
      await practice.getByText('Reasoned solution',{exact:true}).focus(); await page.keyboard.press('Enter');
      assert((await practice.innerText()).includes('1/216')); keyboardChecks++;
      await capture(practice,'changed-dice-practice');
      await capture(setup,'python-setup');
      const invalid=await page.evaluate(async()=>{
        const {finiteMoments,pairedMoments}=await import('/src/learn/data/random-variables-models.js');
        return [()=>finiteMoments([,1],[.5,.5]),()=>finiteMoments([0,1],[,1]),()=>pairedMoments([,{x:1,y:2,mass:1}])].map(fn=>{try{fn();return false;}catch{return true;}});
      });
      assert(invalid.every(Boolean));
      assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
      assert.deepEqual(errors,[]); assert.deepEqual(failedRequests,[]);
      result.records.push({width,actualPrograms:programs.length,equations:equations.length,investigations:7,keyboardChecks,invalidModelInputs:invalid.length,errors,failedRequests});
      await page.close();
    }
    result.passed=true;
    fs.writeFileSync(`${directory}/browser-results.json`,JSON.stringify(result,null,2)+'\n');
    console.log(JSON.stringify({passed:true,records:result.records,images:result.images.length}));
  } finally { await browser.close(); }
})().catch(error=>{console.error(error);process.exitCode=1;});

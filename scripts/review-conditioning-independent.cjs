const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const folder = 'scratch/conditioning-independent-review';
const author = JSON.parse(fs.readFileSync('docs/teaching/evidence/conditioning-stability-author-review.json','utf8'));
function verifySources() {
  for (const file of author.productionSources) assert.equal(crypto.createHash('sha256').update(fs.readFileSync(file.path)).digest('hex'), file.sha256);
}
async function range(locator, value) {
  await locator.evaluate((node,next)=>{
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(node,String(next));
    node.dispatchEvent(new Event('input',{bubbles:true}));
    node.dispatchEvent(new Event('change',{bubbles:true}));
  },value);
}
async function capture(page, locator, name, width) {
  const height=Math.max(1000,Math.ceil((await locator.boundingBox()).height)+180);
  await page.setViewportSize({width,height});
  await locator.evaluate(node=>scrollTo({top:scrollY+node.getBoundingClientRect().top-140,behavior:'instant'}));
  await locator.screenshot({path:`${folder}/${name}-${width}.png`});
  await page.setViewportSize({width,height:1000});
}
(async()=>{
  verifySources();
  fs.mkdirSync(folder,{recursive:true});
  const browser=await chromium.launch({channel:'msedge',headless:true});
  const results=[];
  try {
    for (const width of [1440,390,320]) {
      const page=await browser.newPage({viewport:{width,height:1000}});
      await page.routeWebSocket('**',socket=>socket.close());
      const errors=[],warnings=[],failedRequests=[];
      page.on('pageerror',e=>errors.push(e.message));
      page.on('console',m=>{if(['error','warning'].includes(m.type())&&!m.text().startsWith('[vite]'))warnings.push(m.text());});
      page.on('requestfailed',r=>failedRequests.push(r.url()));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/conditioning-stability-numerical-analysis?module=math-foundations',{waitUntil:'networkidle'});
      const lesson=page.locator('.conditioning-lesson');
      await lesson.waitFor();
      await page.evaluate(()=>document.fonts.ready);
      assert(await page.evaluate(()=>document.fonts.check('16px "Space Grotesk"')));
      assert.equal(await lesson.locator('.python-example').count(),11);
      assert.equal(await lesson.locator('.conditioning-lab').count(),6);
      assert.equal(await lesson.locator('.katex-error').count(),0);
      const anchors=await lesson.locator('.lesson-intro a[href^="#"]').evaluateAll(nodes=>nodes.map(node=>!!document.getElementById(node.getAttribute('href').slice(1))));
      assert(anchors.length===10&&anchors.every(Boolean));
      await capture(page,lesson.locator('.conditioning-figure').first(),'reference-branches',width);
      let states=0;
      const rounding=lesson.getByRole('region',{name:'Exact rounding grid and halfway input; scroll horizontally if needed'});
      await page.getByLabel('Exponent bin',{exact:true}).selectOption('1');
      await range(page.getByLabel('Half-step position',{exact:true}),3);
      assert.match(await lesson.getByLabel('Rounding cells investigation',{exact:true}).innerText(),/stored = 2.5/); states++;
      await rounding.focus(); await page.keyboard.press('ArrowRight');
      await page.getByRole('button',{name:'Reset rounding',exact:true}).focus();await page.keyboard.press('Enter');states++;
      await page.getByLabel('Input sign',{exact:true}).selectOption('-1');
      await range(page.getByLabel('Dyadic exponent k',{exact:true}),0);
      const cancellation=lesson.getByLabel('Cancellation investigation',{exact:true});
      assert.match(await cancellation.innerText(),/At x = −1/);states++;
      await page.getByLabel('Input sign',{exact:true}).selectOption('0');
      assert.match(await cancellation.innerText(),/relative output error is undefined/);states++;
      await page.getByLabel('Input sign',{exact:true}).selectOption('1');
      await range(page.getByLabel('Dyadic exponent k',{exact:true}),60);states++;
      await capture(page,cancellation,'tiny-cancellation',width);
      const measurement=lesson.getByLabel('Measurement sensitivity investigation',{exact:true});
      await range(page.getByLabel('Separation exponent',{exact:true}),7);
      await range(page.getByLabel('Second reading change in units of 1/256',{exact:true}),-2);
      assert.match(await measurement.innerText(),/Answer = \(2, 0\)/);states++;
      const identical=page.getByRole('checkbox',{name:'Make the rows identical: ε = 0'});
      await identical.focus();await page.keyboard.press('Space');
      assert.match(await measurement.innerText(),/no solution/);states++;
      await range(page.getByLabel('Second reading change in units of 1/256',{exact:true}),0);
      assert.match(await measurement.innerText(),/infinitely many/);states++;
      await capture(page,measurement,'nonunique-measurements',width);
      const backward=lesson.getByLabel('Backward error investigation',{exact:true});
      await range(page.getByLabel('Small row exponent',{exact:true}),9);
      await page.getByRole('checkbox',{name:'Rescale the second equation to unit coefficient'}).check();
      assert.match(await backward.innerText(),/Normwise η = 0.5/);states++;
      await capture(page,backward,'scaled-backward-witness',width);
      const summation=lesson.getByLabel('Summation investigation',{exact:true});
      await page.getByLabel('Input order',{exact:true}).selectOption('positive');states++;
      assert.match(await summation.innerText(),/9007199254740996/);
      const addition=page.getByLabel('Inspect addition',{exact:true});
      await addition.focus();await page.keyboard.press('ArrowRight');states++;
      assert.equal(await addition.inputValue(),'2');
      const tree=summation.getByRole('region',{name:'Balanced addition tree; scroll horizontally if needed'});
      await tree.focus();await page.keyboard.press('ArrowRight');states++;
      await capture(page,summation,'positive-addition-tree',width);
      await page.getByLabel('Input order',{exact:true}).selectOption('zero');states++;
      assert.match(await summation.innerText(),/no relative output-error denominator/);
      const propagation=lesson.getByLabel('Error propagation investigation',{exact:true});
      await page.getByLabel('Multiplier q',{exact:true}).selectOption('-0.5');
      await page.getByLabel('Disturbance pattern',{exact:true}).selectOption('alternating');
      await range(page.getByLabel('Propagation steps',{exact:true}),7);states++;
      const point=propagation.locator('circle[data-step="7"]');
      assert(Math.abs(Number(await point.getAttribute('data-error'))+0.01984375)<1e-14);
      await page.getByLabel('Fixed-time refinement',{exact:true}).selectOption('32');states++;
      assert.equal(Number(await propagation.locator('[data-refinement-error]').getAttribute('data-refinement-error')),4294967295/1024);
      await capture(page,propagation,'signed-propagation',width);
      const practice=lesson.locator('section.lesson-check').filter({has:page.getByRole('heading',{name:'Complete the changed measurement report',exact:true})});
      const answer=practice.getByText('Explained solution',{exact:true});
      await answer.focus();await page.keyboard.press('Enter');states++;
      assert.match(await practice.innerText(),/1\/65536/);
      await capture(page,practice,'changed-report-answer',width);
      const heading=lesson.getByRole('heading',{name:'5. Interpret a residual through a nearby problem',exact:true});
      await heading.evaluate(node=>scrollTo({top:scrollY+node.getBoundingClientRect().top-125,behavior:'instant'}));
      await page.screenshot({path:`${folder}/ordinary-residual-reading-${width}.png`});
      const equations=await lesson.locator('.katex-display').evaluateAll(nodes=>nodes.map(n=>({client:n.clientWidth,scroll:n.scrollWidth})));
      const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+2);
      assert(!overflow);assert(equations.length===17&&equations.every(n=>n.scroll<=n.client+2));
      assert.deepEqual(errors,[]);assert.deepEqual(warnings,[]);assert.deepEqual(failedRequests,[]);
      results.push({width,states,equations:equations.length,anchors:anchors.length,fonts:true,overflow,errors,warnings,failedRequests});
      await page.close();
    }
    verifySources();
    const output={checkedAt:new Date().toISOString(),passed:true,productionSources:author.productionSources,results,scope:'Reviewer-operated changed states and keyboard/ordinary-reading captures; separate from author comprehensive82-state runs.'};
    fs.writeFileSync(`${folder}/browser-results.json`,JSON.stringify(output,null,2));
    console.log(JSON.stringify(results));
  } finally { await browser.close(); }
})().catch(e=>{console.error(e);process.exitCode=1;});

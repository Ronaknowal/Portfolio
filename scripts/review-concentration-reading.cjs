const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const {chromium}=require(process.env.PLAYWRIGHT_PACKAGE||'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const directory=path.resolve('scratch/concentration-browser');
fs.mkdirSync(directory,{recursive:true});
(async()=>{
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const records=[];
 try{
  for(const width of [1440,390,320]){
   const page=await browser.newPage({viewport:{width,height:1000},reducedMotion:'reduce'});
   await page.routeWebSocket('**',socket=>socket.close());
   await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/concentration-inequalities-hoeffding-bernstein-chernoff?module=mathematical-statistical-foundations');
   const lesson=page.locator('.concentration-lesson');await lesson.waitFor();
   const record={width,captures:[],math:[],code:[]};
   const shot=async(locator,name)=>{
    await locator.evaluate(element=>window.scrollTo(0,element.getBoundingClientRect().top+window.scrollY-90));
    const filename=`detail-${name}-${width}.png`;await page.screenshot({path:path.join(directory,filename)});record.captures.push(filename);
   };
   await shot(lesson.locator('.lesson-intro'),'route');
   await shot(lesson.locator('.concentration-proof-flow'),'proof-chain');
   await shot(lesson.locator('.concentration-zero-variance'),'zero-variance');
   await shot(lesson.locator('.concentration-program').nth(0),'native-program');
   await shot(lesson.locator('.python-example').nth(0).locator('.lesson-note'),'native-output');
   await shot(lesson.locator('.lesson-check').nth(4),'changed-variance-task');
   const exercise=lesson.locator('.concentration-exercise').nth(3);
   await exercise.locator('.concentration-hint summary').focus();await page.keyboard.press('Enter');
   await shot(exercise,'changed-variance-hint');
   await exercise.locator('.concentration-solution summary').focus();await page.keyboard.press('Enter');
   await shot(exercise.locator('.concentration-solution'),'changed-variance-solution');
   await shot(lesson.locator('.lesson-sources'),'sources');
   const math=await lesson.locator('.katex-display').all();
   for(let index=0;index<math.length;index++){
    const item=math[index];
    const measurement=await item.evaluate(element=>({client:element.clientWidth,scroll:element.scrollWidth}));record.math.push(measurement);
    if(width===320||[0,5,6,9].includes(index))await shot(item,`equation-${index+1}`);
   }
   if(width===320){
    const figures=await lesson.locator('.concentration-lab figure').all();
    for(let index=0;index<figures.length;index++)await shot(figures[index],`plot-${index+1}`);
    const witness=lesson.getByRole('region',{name:'Exponential Markov witness',exact:true});
    await witness.getByRole('button',{name:'Use exact optimum',exact:true}).focus();await page.keyboard.press('Enter');
    assert(Math.abs(Number(await witness.getByRole('slider').inputValue())-Math.log(3))<1e-12);
    await witness.getByRole('button',{name:'Reset exponential witness',exact:true}).focus();await page.keyboard.press('Space');
    assert.equal(await witness.getByRole('slider').inputValue(),'1');
   }
   for(const example of await lesson.locator('.python-example').all()){
    const block=example.locator(':scope > div').first();
    const state=await block.evaluate(element=>({client:element.clientWidth,scroll:element.scrollWidth,overflow:getComputedStyle(element).overflowX}));
    assert(state.scroll<=state.client+1||['auto','scroll'].includes(state.overflow));record.code.push(state);
   }
   record.boxOverflow=await lesson.locator('.concentration-lab,.concentration-inline,.concentration-proof,.katex-display').evaluateAll(elements=>elements.filter(element=>element.scrollWidth>element.clientWidth+2).map(element=>({className:element.className,client:element.clientWidth,scroll:element.scrollWidth})));
   record.pageOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1);
   record.mathErrors=await lesson.locator('.katex-error').count();
   record.svgTextOverflow=await lesson.locator('svg text').evaluateAll(elements=>elements.flatMap(element=>{const bounds=element.getBBox(),box=element.ownerSVGElement.viewBox.baseVal;return bounds.x<-.5||bounds.x+bounds.width>box.width+.5||bounds.y<-.5||bounds.y+bounds.height>box.height+.5?[{text:element.textContent,x:bounds.x,width:bounds.width}]:[]}));
   assert.deepEqual(record.svgTextOverflow,[]);
   records.push(record);await page.close();
  }
  const result={checkedAt:new Date().toISOString(),records};fs.writeFileSync(path.join(directory,'reading-results.json'),JSON.stringify(result,null,2));console.log(JSON.stringify(result,null,2));
  assert(records.every(record=>!record.pageOverflow&&!record.mathErrors&&!record.boxOverflow.length),JSON.stringify(records.map(row=>({width:row.width,overflow:row.boxOverflow}))));
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});

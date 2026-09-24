const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('fs'),path=require('path'),assert=require('node:assert/strict');
const out=path.resolve('scratch/combinatorial-optimization-review/browser');
(async()=>{
  const b=await chromium.launch({channel:'msedge',headless:true}),results=[];
  try {
    for(const width of [1440,390,320]){
      const p=await b.newPage({viewport:{width,height:1000},reducedMotion:'reduce'}),errors=[],failedRequests=[];
      p.on('pageerror',e=>errors.push(e.message));p.on('requestfailed',r=>failedRequests.push({url:r.url(),error:r.failure()}));
      await p.goto('http://127.0.0.1:5173/learn/path/full-curriculum/combinatorial-optimization-approximation-algorithms?module=math-foundations');
      const lesson=p.locator('.combinatorial-lesson');await lesson.locator('h2').last().waitFor();await p.evaluate(()=>document.fonts.ready);
      const capture=async(region,name)=>{await region.scrollIntoViewIfNeeded();await p.waitForTimeout(180);await p.locator('.learn-nav').evaluateAll(ns=>ns.forEach(n=>n.style.visibility='hidden'));await region.screenshot({path:path.join(out,name+'-'+width+'.png')});await p.locator('.learn-nav').evaluateAll(ns=>ns.forEach(n=>n.style.visibility=''));};
      await capture(lesson.locator('.combinatorial-inline').nth(1),'bound-final');
      for(const index of [3,6,13])await capture(lesson.locator('.katex-display').nth(index),'proof-final-'+index);
      const program=lesson.locator('.python-example').filter({has:p.locator('h3',{hasText:'Stop an exact search with an honest bound'})});
      await program.locator('h3').evaluate(n=>scrollTo(0,n.getBoundingClientRect().top+scrollY-110));await p.waitForTimeout(180);
      await p.screenshot({path:path.join(out,'native-program-final-'+width+'.png')});
      const narrow=await lesson.locator('.combinatorial-bound').evaluate(n=>({direction:getComputedStyle(n).flexDirection,overflow:n.scrollWidth>n.clientWidth+1}));
      if(width<480)assert.equal(narrow.direction,'column');assert.equal(narrow.overflow,false);
      const equations=await lesson.locator('.katex-display').evaluateAll(ns=>ns.map((n,index)=>({index,width:n.clientWidth,scroll:n.scrollWidth})));
      assert(equations.every(n=>n.scroll<=n.width+2));assert.equal(await lesson.locator('.katex-error').count(),0);
      assert((await lesson.innerText()).includes('A covers {1,2,3,4}, B covers {1,2,5}, C covers {3,4,6}'));
      const intervalAnswer=lesson.locator('section.lesson-check').nth(1);await intervalAnswer.locator('details').last().evaluate(n=>n.open=true);assert((await intervalAnswer.innerText()).includes('different, unweighted objective'));
      const fonts=await p.evaluate(()=>[...document.fonts].filter(f=>f.status==='loaded').map(f=>f.family));
      assert(fonts.length>0);assert.deepEqual(errors,[]);assert.deepEqual(failedRequests,[]);
      results.push({width,narrow,equations,fonts,errors,failedRequests});await p.close();
    }
  }finally{await b.close();}
  fs.writeFileSync(path.join(out,'final-reading-results.json'),JSON.stringify({checkedAt:new Date().toISOString(),results},null,2));console.log(JSON.stringify(results.map(({width,narrow,fonts})=>({width,narrow,fonts})),null,2));
})().catch(e=>{console.error(e);process.exitCode=1});

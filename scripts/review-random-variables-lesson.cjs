const { chromium }=require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('node:fs');
const assert=require('node:assert/strict');
const directory='scratch/random-variables-browser';fs.mkdirSync(directory,{recursive:true});
const readingOnly=process.argv.includes('--reading-only');
async function slider(locator,value){
  await locator.evaluate((node,next)=>{Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(node,String(next));node.dispatchEvent(new Event('input',{bubbles:true}));node.dispatchEvent(new Event('change',{bubbles:true}));},value);
}
async function capture(page,target,name,width,whole=false){
  await target.evaluate(node=>window.scrollTo(0,window.scrollY+node.getBoundingClientRect().top-140));
  if(whole){await page.addStyleTag({content:'.learn-nav{visibility:hidden!important}'});await target.screenshot({path:`${directory}/${name}-${width}.png`});await page.addStyleTag({content:'.learn-nav{visibility:visible!important}'});}
  else await page.screenshot({path:`${directory}/${name}-${width}.png`});
}
async function metric(region,label){return region.locator('.rv-metrics>div').filter({has:region.page().getByText(label,{exact:true})}).locator('dd').innerText();}
function near(actual,expected){assert.ok(Math.abs(Number(actual)-expected)<2e-5,`${actual} != ${expected}`);}
(async()=>{
  const {randomVariableExamples:examples}=await import('../src/learn/data/random-variables-examples.js');
  const browser=await chromium.launch({channel:'msedge',headless:true});const results=[];
  try{for(const width of [1440,390,320]){
    const page=await browser.newPage({viewport:{width,height:1000}});await page.routeWebSocket('**',socket=>socket.close());
    const errors=[],warnings=[],failedRequests=[];
    page.on('pageerror',error=>errors.push(error.message));
    page.on('console',message=>{if(['warning','error'].includes(message.type())&&!message.text().startsWith('[vite]'))warnings.push(message.text());});
    page.on('requestfailed',request=>failedRequests.push(request.url()));
    await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/random-variables-expectation-covariance?module=math-foundations',{waitUntil:'networkidle'});
    const lesson=page.locator('.random-variable-lesson');await lesson.waitFor();await page.evaluate(()=>document.fonts.ready);
    assert.equal(await page.locator('vite-error-overlay').count(),0);
    const anchors=await lesson.locator('.lesson-intro a[href^="#"]').evaluateAll(nodes=>nodes.map(node=>({href:node.getAttribute('href'),exists:!!document.getElementById(node.getAttribute('href').slice(1))})));
    assert.equal(anchors.length,12);assert(anchors.every(row=>row.exists));assert.equal(await lesson.locator('.katex-error').count(),0);
    const programs=await lesson.locator('.python-example').evaluateAll(nodes=>nodes.map(node=>({title:node.querySelector('h3').textContent,question:node.previousElementSibling.textContent.replace(/^Before running\.\s*/,''),blocks:[...node.children].filter(child=>getComputedStyle(child).whiteSpace==='pre').map(child=>[...child.childNodes].filter(n=>n.nodeType===Node.TEXT_NODE).map(n=>n.textContent).join(''))})));
    assert.equal(programs.length,14);for(const program of programs){const example=Object.values(examples).find(row=>row.title===program.title);assert(example);assert.equal(program.question,example.question);assert.equal(program.blocks[0].trim(),example.code.trim());assert.equal(program.blocks[1].trim(),example.expected.trim());}
    for(let i=0;i<12;i++)await capture(page,lesson.locator('h2').nth(i),'reading-'+(i+1),width);
    for(let i=0;i<7;i++)await capture(page,lesson.locator('.rv-lab').nth(i),'default-'+(i+1),width,true);
    let states=0;
    if(!readingOnly){
      const mapping=page.getByRole('region',{name:'Outcome mapping investigation',exact:true});
      for(const p of [0,25,50,100])for(const q of [0,50,100])for(const rule of ['heads','first','equal']){
        await slider(mapping.getByLabel('First coin head chance (%)',{exact:true}),p);await slider(mapping.getByLabel('Second coin head chance (%)',{exact:true}),q);await mapping.getByLabel('Numerical rule',{exact:true}).selectOption(rule);
        const masses=await mapping.locator('.rv-outcomes>div span:first-of-type').allTextContents();near(masses.map(s=>Number(s.replace('mass ',''))).reduce((a,b)=>a+b,0),1);states++;
      }
      await slider(mapping.getByLabel('CDF threshold t',{exact:true}),-1);assert((await mapping.locator('.rv-result').innerText()).startsWith('F(-1) = 0.'));states++;
      await capture(page,mapping,'mapping-degenerate',width,true);await mapping.getByRole('button',{name:'Reset investigation',exact:true}).focus();await page.keyboard.press('Enter');
      await mapping.getByLabel('First coin head chance (%)',{exact:true}).focus();await page.keyboard.press('ArrowRight');assert.equal(await mapping.getByLabel('First coin head chance (%)',{exact:true}).inputValue(),'51');states+=2;
      const loss=page.getByRole('region',{name:'Mean and squared loss investigation',exact:true});
      for(const preset of ['asymmetric','symmetric','constant'])for(const c of [-3,0,.25,2,4]){
        await loss.getByLabel('Population law',{exact:true}).selectOption(preset);await slider(loss.getByLabel('Constant prediction c',{exact:true}),c);
        const mean={asymmetric:.25,symmetric:0,constant:2}[preset],variance={asymmetric:51/16,symmetric:1,constant:0}[preset];
        near(await metric(loss,'Mean'),mean);near(await metric(loss,'Total expected squared loss'),variance+(c-mean)**2);states++;
      }
      await capture(page,loss,'loss-constant',width,true);await loss.getByRole('button',{name:'Reset investigation',exact:true}).click();
      const joint=page.getByRole('region',{name:'Joint covariance investigation',exact:true});
      for(const preset of ['matching','opposite','independent','nonlinear'])for(const scale of [-2,0,1,2])for(const shift of [-3,0,3]){
        await joint.getByLabel('Joint law',{exact:true}).selectOption(preset);await slider(joint.getByLabel('Scale Y',{exact:true}),scale);await slider(joint.getByLabel('Shift Y',{exact:true}),shift);
        near(await metric(joint,'Covariance'),(preset==='matching'?2/3:preset==='opposite'?-2/3:0)*scale);
        assert.equal(await metric(joint,'Correlation')==='undefined',scale===0);states++;
      }
      await joint.getByRole('button',{name:'Reset investigation',exact:true}).click();await joint.getByLabel('Joint law',{exact:true}).selectOption('nonlinear');await joint.getByRole('button',{name:'Inspect X 0 Y 1',exact:true}).focus();await page.keyboard.press('Enter');assert((await joint.locator('.rv-result').innerText()).includes('joint mass 0; marginal product 0.222222'));states++;
      await capture(page,joint,'joint-nonlinear-witness',width,true);await slider(joint.getByLabel('Scale Y',{exact:true}),0);await capture(page,joint,'joint-constant',width,true);
      const noise=page.getByRole('region',{name:'Shared noise investigation',exact:true});
      for(const common of [0,1,3])for(const local of [0,1,2])for(const [a,b] of [[.5,.5],[-1,1],[.25,.75],[0,0]]){
        for(const [label,value] of [['Common amplitude (mV)',common],['Local amplitude (mV)',local],['Coefficient a',a],['Coefficient b',b]])await slider(noise.getByLabel(label,{exact:true}),value);
        near(await metric(noise,'Mean aA+bB (mV)'),10*a+20*b);near(await metric(noise,'Variance aA+bB (mV²)'),(a+b)**2*common**2+(a*a+b*b)*local**2);states++;
      }
      await noise.getByRole('button',{name:'Difference B−A',exact:true}).click();await capture(page,noise,'noise-cancellation',width,true);states++;
      await noise.getByRole('button',{name:'Average (A+B)/2',exact:true}).click();states++;
      const conditional=page.getByRole('region',{name:'Conditional moments investigation',exact:true});
      for(const p of [0,1,25,50,75,99,100])for(const group of ['-2','2']){
        await slider(conditional.getByLabel('P(G=2), percent',{exact:true}),p);await conditional.getByLabel('Inspect group',{exact:true}).selectOption(group);
        const values=await conditional.locator('.rv-decomposition strong').allTextContents();near(values[0],1+16*(p/100)*(1-p/100));near(values[1],-1+16*(p/100)*(1-p/100));
        assert.equal((await conditional.locator('.rv-result').innerText()).startsWith('There is no identified'),(p===0&&group==='2')||(p===100&&group==='-2'));states++;
      }
      await capture(page,conditional,'conditional-null-group',width,true);
      const squared=page.getByRole('region',{name:'Squared uniform transformation investigation',exact:true});
      for(const [a,b] of [[0,0],[0,1],[0,100],[4,49],[25,81],[99,100],[100,100]]){
        await slider(squared.getByLabel('Lower Y endpoint (%)',{exact:true}),a);await slider(squared.getByLabel('Upper Y endpoint (%)',{exact:true}),b);
        const masses=await squared.locator('tbody tr td:last-child').allTextContents();near(masses.reduce((sum,text)=>sum+Number(text),0),Math.sqrt(b/100)-Math.sqrt(a/100));states++;
      }
      await capture(page,squared,'squared-zero-width',width,true);await squared.getByRole('button',{name:'Reset investigation',exact:true}).click();await capture(page,squared,'squared-two-branches',width,true);
      const sample=page.getByRole('region',{name:'Sample mean distribution investigation',exact:true});
      for(const n of [1,2,8,16])for(const p of [0,1,25,50,99,100]){
        await slider(sample.getByLabel('Readings per dataset n',{exact:true}),n);await slider(sample.getByLabel('Population success chance (%)',{exact:true}),p);
        near(await metric(sample,'Independent mean variance'),(p/100)*(1-p/100)/n);near(await metric(sample,'Copied mean variance'),(p/100)*(1-p/100));states++;
      }
      await slider(sample.getByLabel('Population success chance (%)',{exact:true}),50);await capture(page,sample,'sample-independent-copies',width,true);
      for(const region of await lesson.locator('.rv-lab').all()){
        const before=await region.locator('input,select').evaluateAll(nodes=>nodes.map(n=>({label:n.getAttribute('aria-label'),value:n.value})));
        await region.getByRole('button',{name:'Reset investigation',exact:true}).focus();await page.keyboard.press('Enter');states++;
        for(const input of await region.locator('input[type="range"]').all()){const old=Number(await input.inputValue());await input.focus();await page.keyboard.press('ArrowRight');assert(Number(await input.inputValue())!==old);states++;}
        await region.getByRole('button',{name:'Reset investigation',exact:true}).click();
        for(const select of await region.locator('select').all()){const old=await select.inputValue();await select.focus();await page.keyboard.press('ArrowDown');await page.keyboard.press('Enter');assert.notEqual(await select.inputValue(),old);states++;}
        await region.getByRole('button',{name:'Reset investigation',exact:true}).click();
      }
      const practice=lesson.locator('section.lesson-check');assert.equal(await practice.count(),11);
      for(const summary of await practice.locator('summary').all()){await summary.focus();await page.keyboard.press('Enter');assert.equal(await summary.locator('..').getAttribute('open'),'');states++;}
      await capture(page,practice.nth(4),'changed-practice-noise',width,true);
    }
    let boundaryCheck=null;
    if(readingOnly){
      boundaryCheck=await page.evaluate(async()=>{
        const model=await import('/src/learn/data/random-variables-models.js');
        let rejected=0;
        for(const value of [1e-160,1e-200]){try{model.finiteMoments([0,value],[.5,.5]);}catch(error){if(error instanceof RangeError)rejected++;}}
        return {rejected,constantVariance:model.finiteMoments([.1,.1,.1],[.1,.2,.7]).variance,tinyCorrelation:model.pairedMoments([{x:0,y:0,mass:.5},{x:1e-130,y:1e-130,mass:.5}]).correlation};
      });
      assert.deepEqual(boundaryCheck,{rejected:2,constantVariance:0,tinyCorrelation:1});
      assert((await lesson.innerText()).includes('Their count has a binomial law.'));
    }
    for(const details of await lesson.locator('details').all())await details.evaluate(n=>n.open=true);
    const geometry=await lesson.evaluate(node=>({documentWidth:document.documentElement.scrollWidth,fontReady:[...document.fonts].some(f=>f.family==='Space Grotesk'&&f.status==='loaded'),equations:[...node.querySelectorAll('.katex-display')].map(n=>({width:n.getBoundingClientRect().width,content:n.scrollWidth,tex:n.querySelector('annotation')?.textContent})),svgOverflow:[...node.querySelectorAll('.rv-lab svg')].flatMap(svg=>[...svg.querySelectorAll('text')].filter(n=>{const b=n.getBBox();return b.x<-.5||b.x+b.width>svg.viewBox.baseVal.width+.5||b.y<-.5||b.y+b.height>svg.viewBox.baseVal.height+2;}).map(n=>n.textContent)),controls:[...node.querySelectorAll('.rv-lab input,.rv-lab select,.rv-lab button')].map(n=>({name:n.getAttribute('aria-label')||n.textContent,height:n.getBoundingClientRect().height})),internalLinks:[...node.querySelectorAll('a[href^="/learn/"]')].map(n=>n.getAttribute('href'))}));
    fs.writeFileSync(directory+'/geometry-'+width+'.json',JSON.stringify(geometry,null,2));
    assert.equal(geometry.documentWidth,width);assert(geometry.fontReady);assert(geometry.controls.every(row=>row.height>=43));assert.deepEqual(errors,[]);assert.deepEqual(warnings,[]);assert.deepEqual(failedRequests,[]);
    if(readingOnly)for(let i=0;i<geometry.equations.length;i++)await capture(page,lesson.locator('.katex-display').nth(i),'equation-'+i,width);
    results.push({width,states,anchors,programs:programs.length,boundaryCheck,geometry,errors,warnings,failedRequests});fs.writeFileSync(directory+'/in-progress'+(readingOnly?'-reading':'')+'.json',JSON.stringify(results,null,2));
    console.log(width,'passed',states,'states; equation overflow',geometry.equations.filter(x=>x.content>x.width+1).map(x=>x.tex),'SVG labels',geometry.svgOverflow);await page.close();
  }}finally{await browser.close();}
  fs.writeFileSync(directory+(readingOnly?'/reading-results.json':'/results.json'),JSON.stringify({checkedAt:new Date().toISOString(),results},null,2));
  assert(results.every(row=>row.geometry.equations.every(x=>x.content<=x.width+1)),'Equation overflow');assert(results.every(row=>!row.geometry.svgOverflow.length),'SVG label overflow');
})().catch(error=>{console.error(error);process.exitCode=1;});

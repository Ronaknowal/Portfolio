const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('fs'),assert=require('assert/strict');
(async()=>{
  const browser=await chromium.launch({channel:'msedge',headless:true});
  const records=[];
  try{
    for(const width of [1440,390,320]){
      const page=await browser.newPage({viewport:{width,height:1050},reducedMotion:'reduce'});
      const errors=[];
      page.on('pageerror',e=>errors.push(e.message));
      page.on('console',e=>{if(['error','warning'].includes(e.type()))errors.push(e.text())});
      page.on('requestfailed',r=>errors.push(r.url()+': '+r.failure().errorText));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/spectral-graph-theory');
      const lesson=page.locator('.spectral-lesson');
      await lesson.locator('h2').last().waitFor();
      await page.evaluate(()=>document.fonts.ready);
      assert(!/\\(?:u[0-9a-fA-F]{4}|x[0-9a-fA-F]{2})/.test(await lesson.innerText()), 'No literal escaped Unicode in rendered prose');
      assert((await lesson.innerText()).includes('x=[1,1,1,−1,−1,−1]'));
      assert((await lesson.innerText()).includes('A—B—C—D'));
      const anchor=lesson.locator('.spectral-anchor');
      const paragraphs=await lesson.locator(':scope > p').evaluateAll(nodes=>nodes.map(node=>({left:node.getBoundingClientRect().left,width:node.getBoundingClientRect().width,scroll:node.scrollWidth,text:node.textContent.slice(0,30)})));
      assert(paragraphs.every(item=>item.left>=19&&item.width+2>=item.scroll),JSON.stringify(paragraphs.filter(item=>item.left<19||item.width+2<item.scroll)));
      assert.equal(await anchor.locator('line').count(),7);
      assert.deepEqual(await anchor.locator('.anchor-name').allTextContents(),['A','B','C','D','E','F']);
      assert.deepEqual(await anchor.locator('.anchor-value').allTextContents(),['+1','+1','+1','−1','−1','−1']);
      const edgeGeometry=await anchor.locator('line').evaluateAll(lines=>lines.map(line=>['x1','y1','x2','y2'].map(key=>Number(line.getAttribute(key)))));
      assert.deepEqual(edgeGeometry,[[55,45,205,45],[55,45,130,140],[205,45,130,140],[130,140,130,245],[130,245,55,340],[130,245,205,340],[55,340,205,340]]);
      for(const [selector,name] of [['.spectral-anchor','anchor-final'],['.laplacian-row','row-final'],['.spectral-correspondence','coordinates-final'],['.spectral-circuit','circuit-final'],['.lesson-sources','sources-final']]){
        await page.locator('.learn-nav').evaluateAll(nodes=>nodes.forEach(n=>n.style.visibility='hidden'));
        await lesson.locator(selector).screenshot({path:`scratch/spectral-browser/${name}-${width}.png`});
        await page.locator('.learn-nav').evaluateAll(nodes=>nodes.forEach(n=>n.style.visibility=''));
      }
      const proof=lesson.locator('details').filter({has:page.locator('summary',{hasText:'Deeper derivation: why some sweep threshold'})});
      await proof.locator('summary').click();
      assert((await proof.innerText()).includes('one side may contain only zero-valued nodes'));
      const geometry=await lesson.evaluate(root=>({pageWidth:document.documentElement.clientWidth,scrollWidth:document.documentElement.scrollWidth,math:[...root.querySelectorAll('.katex-display')].map(n=>({width:n.clientWidth,scroll:n.scrollWidth})),svgText:[...root.querySelectorAll('.spectral-anchor svg text')].map(n=>{const b=n.getBBox(),v=n.ownerSVGElement.viewBox.baseVal;return {label:n.textContent,size:parseFloat(getComputedStyle(n).fontSize)*n.ownerSVGElement.getBoundingClientRect().width/v.width,inside:b.x>=0&&b.y>=0&&b.x+b.width<=v.width&&b.y+b.height<=v.height}})}));
      assert(geometry.scrollWidth<=geometry.pageWidth+1);
      assert(geometry.math.every(n=>n.width===0||n.scroll<=n.width+2));
      assert(geometry.svgText.every(n=>n.inside&&n.size>=15));
      const labText=await lesson.locator('.spectral-graph-scroll svg text,.spectral-embedding-scroll svg text').evaluateAll(nodes=>nodes.filter(n=>n.ownerSVGElement.getBoundingClientRect().width>0).map(n=>{const b=n.getBBox(),v=n.ownerSVGElement.viewBox.baseVal;return {label:n.textContent,size:parseFloat(getComputedStyle(n).fontSize)*n.ownerSVGElement.getBoundingClientRect().width/v.width,inside:b.x>=0&&b.y>=0&&b.x+b.width<=v.width&&b.y+b.height<=v.height}}));
      assert(labText.every(n=>n.inside&&n.size>=14.5),JSON.stringify(labText.filter(n=>!n.inside||n.size<14.5)));
      const spectrum=lesson.getByRole('region',{name:'Graph spectrum lab',exact:true});
      const spectrumGeometry=await spectrum.locator('svg:visible').evaluate(svg=>({width:svg.getBoundingClientRect().width,parentWidth:svg.parentElement.clientWidth,points:[...svg.querySelectorAll('circle')].map(n=>[Number(n.getAttribute('cx')),Number(n.getAttribute('cy'))]),lines:[...svg.querySelectorAll('line')].map(n=>['x1','y1','x2','y2'].map(k=>Number(n.getAttribute(k))))}));
      assert(spectrumGeometry.width<=spectrumGeometry.parentWidth+1);
      for(const line of spectrumGeometry.lines){assert(spectrumGeometry.points.some(p=>p[0]===line[0]&&p[1]===line[1]));assert(spectrumGeometry.points.some(p=>p[0]===line[2]&&p[1]===line[3]));}
      assert.equal(spectrumGeometry.lines.length,7);
      assert(await lesson.locator('.spectral-embedding-scroll').evaluate(n=>n.scrollWidth<=n.clientWidth+1));
      for(const [selector,name] of [['.spectral-graph-scroll','spectrum-graph-final'],['.spectral-embedding-scroll','embedding-plot-final'],['.spectral-mode-bank','filter-gains-final']]){
        await page.locator('.learn-nav').evaluateAll(nodes=>nodes.forEach(n=>n.style.visibility='hidden'));
        await lesson.locator(selector).first().screenshot({path:`scratch/spectral-browser/${name}-${width}.png`});
        await page.locator('.learn-nav').evaluateAll(nodes=>nodes.forEach(n=>n.style.visibility=''));
      }
      await page.locator('.learn-nav').evaluateAll(nodes=>nodes.forEach(n=>n.style.visibility='hidden'));
      await proof.screenshot({path:`scratch/spectral-browser/cheeger-proof-final-${width}.png`});
      await page.locator('.learn-nav').evaluateAll(nodes=>nodes.forEach(n=>n.style.visibility=''));
      await proof.locator('summary').click();
      for(let index=0;index<10;index++){
        if(width!==390&&![0,3,5,7,9].includes(index))continue;
        await lesson.locator('h2').nth(index).evaluate(n=>window.scrollTo({top:window.scrollY+n.getBoundingClientRect().top-100,behavior:'instant'}));
        await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
        await page.waitForTimeout(200);
        await page.screenshot({path:`scratch/spectral-browser/reading-final-${index+1}-${width}.png`});
      }
      assert.deepEqual(errors,[]);
      records.push({width,edgeGeometry,geometry,labText,paragraphs,spectrumGeometry,errors});
      console.log('final reading passed',width);
      await page.close();
    }
    fs.writeFileSync('scratch/spectral-browser/final-reading-results.json',JSON.stringify({verifiedAt:new Date().toISOString(),records},null,2));
  }finally{await browser.close()}
})().catch(e=>{console.error(e);process.exit(1)});

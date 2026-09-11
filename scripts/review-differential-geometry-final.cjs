const { chromium }=require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('node:fs');
const assert=require('node:assert/strict');
const directory='scratch/differential-geometry-browser';
(async()=>{
  const {differentialGeometryExamples:examples}=await import('../src/learn/data/differential-geometry-examples.js');
  const browser=await chromium.launch({channel:'msedge',headless:true});
  const results=[];
  try{
    for(const width of [1440,390,320]){
      const page=await browser.newPage({viewport:{width,height:1000}});
      await page.routeWebSocket('**',socket=>socket.close());
      const errors=[],failedRequests=[];
      page.on('pageerror',error=>errors.push(error.message));
      page.on('requestfailed',request=>failedRequests.push(request.url()));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/differential-geometry-riemannian-manifolds?module=math-foundations',{waitUntil:'networkidle'});
      const lesson=page.locator('.differential-geometry-lesson');
      await lesson.waitFor();await page.evaluate(()=>document.fonts.ready);
      const programs=await lesson.locator('.python-example').evaluateAll(nodes=>nodes.map(node=>({
        title:node.querySelector('h3').textContent,
        question:node.previousElementSibling.textContent.replace(/^Before running:\s*/,''),
        blocks:[...node.children].filter(child=>getComputedStyle(child).whiteSpace==='pre').map(child=>[...child.childNodes].filter(n=>n.nodeType===Node.TEXT_NODE).map(n=>n.textContent).join('')),
      })));
      assert.equal(programs.length,14);
      for(const row of programs){
        const example=Object.values(examples).find(e=>e.title===row.title);
        assert.equal(row.question,example.question);
        assert.equal(row.blocks[0].trim(),example.code.trim());assert.equal(row.blocks[1].trim(),example.expected.trim());
      }
      const capstone=lesson.locator('.python-example').last();
      const crossProduct=lesson.getByText('In three-dimensional space, the cross product',{exact:false});
      await crossProduct.evaluate(node=>window.scrollTo(0,window.scrollY+node.getBoundingClientRect().top-140));
      await page.screenshot({path:directory+'/final-cross-product-reading-'+width+'.png'});
      const output=capstone.locator(':scope > div').last();
      const actualText=await capstone.innerText();
      assert.ok(actualText.includes('gradient norm: 8.641e-09')&&actualText.includes('changed gradient norm: 3.050e-08'));
      const interpretation=lesson.getByText('The changed matrix is Q diag(2,5,9)Qᵀ.',{exact:false});
      assert.equal(await interpretation.count(),1);
      await interpretation.evaluate(node=>window.scrollTo(0,window.scrollY+node.getBoundingClientRect().top-400));
      await page.screenshot({path:directory+'/final-capstone-reading-'+width+'.png'});
      const measurements=await lesson.evaluate(node=>({
        fontReady:[...document.fonts].some(f=>f.family==='Space Grotesk'&&f.status==='loaded'),
        documentWidth:document.documentElement.scrollWidth,
        equations:[...node.querySelectorAll('.katex-display')].map(e=>({width:e.clientWidth,content:e.scrollWidth})),
        codeCopyButtons:node.querySelectorAll('.python-example button').length,
      }));
      assert.ok(measurements.fontReady&&measurements.documentWidth===width);
      assert.ok(measurements.equations.every(e=>e.content<=e.width+1));
      assert.equal(await lesson.locator('.katex-error').count(),0);
      const summary=lesson.locator('.dg-practice').last().getByText('Show explained solution',{exact:true});
      await summary.focus();await page.keyboard.press('Enter');
      assert.equal(await summary.locator('..').getAttribute('open'),'');
      await page.getByRole('button',{name:'Reset transport',exact:true}).focus();await page.keyboard.press('Enter');
      assert.equal(await lesson.getByLabel('Transport progress',{exact:true}).inputValue(),'3');
      const source=await lesson.locator('a[href^="/learn/"]').evaluateAll(nodes=>nodes.map(node=>node.getAttribute('href')));
      assert.equal(source.length,3);
      assert.deepEqual(errors,[]);assert.deepEqual(failedRequests,[]);
      results.push({width,programs:programs.length,visibleResiduals:true,keyboardStates:2,measurements,reviewLinks:source,errors,failedRequests});
      console.log(width,'final programs and reading passed');
      await page.close();
    }
  }finally{await browser.close();}
  fs.writeFileSync(directory+'/final-program-results.json',JSON.stringify({checkedAt:new Date().toISOString(),results},null,2));
})().catch(error=>{console.error(error);process.exitCode=1;});

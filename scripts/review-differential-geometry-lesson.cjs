const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const directory = 'scratch/differential-geometry-browser';
const readingOnly = process.argv.includes('--reading-only');
fs.mkdirSync(directory, { recursive: true });
async function slider(locator, value) {
  await locator.evaluate((node,next) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype,'value').set.call(node,String(next));
    node.dispatchEvent(new Event('input',{bubbles:true}));
    node.dispatchEvent(new Event('change',{bubbles:true}));
  },value);
}
async function metric(region,label) {
  return region.locator('.dg-readouts>div').filter({has:region.page().getByText(label,{exact:true})}).locator('dd').innerText();
}
async function capture(page,target,name,width,full=false) {
  await target.evaluate(node=>window.scrollTo(0,window.scrollY+node.getBoundingClientRect().top-140));
  if (full) {
    await page.addStyleTag({content:'.learn-nav { visibility:hidden !important; }'});
    await target.screenshot({path:directory+'/'+name+'-'+width+'.png'});
    await page.addStyleTag({content:'.learn-nav { visibility:visible !important; }'});
  } else await page.screenshot({path:directory+'/'+name+'-'+width+'.png'});
}
(async()=>{
  const { differentialGeometryExamples:examples }=await import('../src/learn/data/differential-geometry-examples.js');
  const browser=await chromium.launch({channel:'msedge',headless:true});
  const results=[];
  try {
    for(const width of [1440,390,320]) {
      const page=await browser.newPage({viewport:{width,height:1000}});
      await page.routeWebSocket('**',socket=>socket.close());
      const errors=[],warnings=[],failedRequests=[];
      page.on('pageerror',error=>errors.push(error.message));
      page.on('console',msg=>{if(['warning','error'].includes(msg.type())&&!msg.text().startsWith('[vite]'))warnings.push(msg.text());});
      page.on('requestfailed',request=>failedRequests.push(request.url()));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/differential-geometry-riemannian-manifolds?module=math-foundations',{waitUntil:'networkidle'});
      const lesson=page.locator('.differential-geometry-lesson');
      await lesson.waitFor();
      await page.evaluate(()=>document.fonts.ready);
      assert.equal(await page.locator('vite-error-overlay').count(),0);
      const anchors=await lesson.locator('.lesson-intro a[href^="#"]').evaluateAll(nodes=>nodes.map(node=>({href:node.getAttribute('href'),exists:!!document.getElementById(node.getAttribute('href').slice(1))})));
      assert.equal(anchors.length,11);assert.ok(anchors.every(row=>row.exists));
      assert.equal(await lesson.locator('.katex-error').count(),0);
      const programs=await lesson.locator('.python-example').evaluateAll(nodes=>nodes.map(node=>({
        title:node.querySelector('h3').textContent,
        question:node.previousElementSibling.textContent.replace(/^Before running:\s*/,''),
        blocks:[...node.children].filter(child=>getComputedStyle(child).whiteSpace==='pre').map(child=>[...child.childNodes].filter(n=>n.nodeType===Node.TEXT_NODE).map(n=>n.textContent).join('')),
      })));
      assert.equal(programs.length,14);
      for(const program of programs){
        const example=Object.values(examples).find(row=>row.title===program.title);
        assert.ok(example);assert.equal(program.question,example.question);
        assert.equal(program.blocks[0].trim(),example.code.trim());
        assert.equal(program.blocks[1].trim(),example.expected.trim());
      }
      for(let i=0;i<11;i++)await capture(page,lesson.locator('h2').nth(i),'reading-'+(i+1),width);
      for(let i=0;i<10;i++)await capture(page,lesson.locator('.dg-lab,.dg-figure').nth(i),'default-visual-'+(i+1),width,true);
      let states=0;
      if (!readingOnly) {
      const atlas=page.getByRole('region',{name:'Circle atlas investigation',exact:true});
      for(const angle of [0,45,179,180,181,225,315,360]){
        await slider(atlas.getByLabel('Circle position',{exact:true}),angle);
        assert.equal(await atlas.getByText('Not in this chart’s domain.',{exact:true}).count(),[0,180,360].includes(angle)?1:0);
        states++;
      }
      await atlas.getByRole('button',{name:'Inspect α seam',exact:true}).click();
      await capture(page,atlas,'atlas-seam',width,true);
      await atlas.getByRole('button',{name:'Reset atlas',exact:true}).focus();await page.keyboard.press('Enter');
      assert.equal(await atlas.getByLabel('Circle position',{exact:true}).inputValue(),'225');
      await atlas.getByLabel('Circle position',{exact:true}).focus();await page.keyboard.press('ArrowRight');
      assert.equal(await atlas.getByLabel('Circle position',{exact:true}).inputValue(),'226');states+=3;

      const differential=page.getByRole('region',{name:'Metric and differential investigation',exact:true});
      for(const cost of [.5,1,2,3])for(const shear of [-1.5,0,1,1.5]){
        await slider(differential.getByLabel('Physical y-motion cost',{exact:true}),cost);
        await slider(differential.getByLabel('Coordinate shear',{exact:true}),shear);
        assert.equal(await metric(differential,'Gradient in x,y'),'(2, '+Number((-1/cost**2).toFixed(4))+')');
        for(const angle of [0,90,270]){
          await slider(differential.getByLabel('Chosen direction angle',{exact:true}),angle);
          const actual=Number(await metric(differential,'df on your unit direction'));
          const expected=2*Math.cos(angle*Math.PI/180)-Math.sin(angle*Math.PI/180)/cost;
          assert.ok(Math.abs(actual-expected)<1e-4);states++;
        }
      }
      await capture(page,differential,'metric-changed',width,true);
      await differential.getByRole('button',{name:'Reset metric',exact:true}).click();

      const paths=page.getByRole('region',{name:'Sphere paths investigation',exact:true});
      for(const radius of [.5,1,3])for(const angle of [0,45,90,180]){
        await slider(paths.getByLabel('Sphere radius',{exact:true}),radius);
        await slider(paths.getByLabel('Endpoint separation',{exact:true}),angle);
        for(const fraction of [0,.5,1]){
          await slider(paths.getByLabel('Route fraction',{exact:true}),fraction);
          assert.ok(Math.abs(Number(await metric(paths,'Chord length'))-2*radius*Math.sin(angle*Math.PI/360))<1e-4);states++;
        }
      }
      assert.equal(await metric(paths,'Shortest Log'),'nonunique at antipodes');
      await capture(page,paths,'paths-antipodal',width,true);
      await paths.getByRole('button',{name:'Reset paths',exact:true}).click();

      const step=page.getByRole('region',{name:'Sphere update investigation',exact:true});
      for(const angle of [-180,-90,0,90,180])for(const rate of [0,.1,.8,1.5]){
        await slider(step.getByLabel('Starting direction',{exact:true}),angle);
        await slider(step.getByLabel('Update step size',{exact:true}),rate);
        assert.ok(Number(await metric(step,'Candidate norm'))>=1);
        assert.ok(!/NaN|Infinity/.test(await step.innerText()));states++;
      }
      await capture(page,step,'step-large',width,true);
      await step.getByRole('button',{name:'Reset original update',exact:true}).click();

      const polar=page.getByRole('region',{name:'Moving polar basis investigation',exact:true});
      for(const t of [-2,-.1,0,1,2])for(const b of [.25,1,2]){
        await slider(polar.getByLabel('Position along straight path',{exact:true}),t);
        await slider(polar.getByLabel('Line height b',{exact:true}),b);
        const velocity=(await metric(polar,'Reconstructed Cartesian velocity')).slice(1,-1).split(',').map(Number);
        assert.ok(Math.abs(velocity[0]-1)<1e-10&&Math.abs(velocity[1])<1e-10);states++;
      }
      await capture(page,polar,'polar-changed',width,true);
      await polar.getByRole('button',{name:'Reset moving basis',exact:true}).click();

      const transport=page.getByRole('region',{name:'Parallel transport investigation',exact:true});
      for(const wedge of [15,60,90,150])for(const radius of [.5,1,3]){
        await slider(transport.getByLabel('Longitude wedge',{exact:true}),wedge);
        await slider(transport.getByLabel('Transport sphere radius',{exact:true}),radius);
        for(const initial of [-180,0,90]){
          await slider(transport.getByLabel('Initial tangent angle',{exact:true}),initial);
          for(const progress of [0,.5,1,2.5,3]){
            await slider(transport.getByLabel('Transport progress',{exact:true}),progress);
            assert.equal(await metric(transport,'Final oriented turn'),wedge+'°');
            const readout=await metric(transport,'Arrow norm / radial dot');
            assert.ok(readout.startsWith('1 /'));assert.ok(!/NaN|Infinity/.test(readout));states++;
          }
        }
      }
      await transport.getByRole('button',{name:'Route: N → A → B → N',exact:true}).click();
      assert.equal(await metric(transport,'Final oriented turn'),'-150°');states++;
      await capture(page,transport,'transport-reversed',width,true);
      await transport.getByRole('button',{name:'Reset transport',exact:true}).click();
      await slider(transport.getByLabel('Transport progress',{exact:true}),.5);
      await capture(page,transport,'transport-half-leg',width,true);

      const curvature=page.getByRole('region',{name:'Intrinsic curvature investigation',exact:true});
      for(const radius of [.5,1,3])for(const distance of [.05,.75,1.5]){
        await slider(curvature.getByLabel('Curvature length scale R',{exact:true}),radius);
        await slider(curvature.getByLabel('Geodesic radius s over R',{exact:true}),distance);
        const bars=await curvature.locator('.dg-area-strip i').evaluateAll(nodes=>nodes.map(n=>parseFloat(n.style.width)));
        assert.ok(Math.abs(bars[0]-bars[1])<1e-10&&bars[2]<bars[0]&&bars[3]>bars[0]);states++;
      }
      await capture(page,curvature,'curvature-wide-disk',width,true);
      await curvature.getByRole('button',{name:'Reset curvature',exact:true}).focus();await page.keyboard.press('Enter');
      assert.equal(await curvature.getByLabel('Geodesic radius s over R',{exact:true}).inputValue(),'0.75');states++;
      const practice=lesson.locator('.dg-practice');
      assert.equal(await practice.count(),11);
      for(const item of await practice.all())for(const summary of await item.locator('summary').all()){
        await summary.focus();await page.keyboard.press('Enter');
        assert.equal(await summary.locator('..').getAttribute('open'),'');states++;
      }
      await capture(page,practice.last(),'changed-practice',width,true);
      } else {
        const equations=await lesson.locator('.katex-display').all();
        for (let index=0;index<equations.length;index++) {
          await capture(page,equations[index],'final-equation-'+index,width);
        }
        const button=lesson.getByRole('button',{name:'Reset atlas',exact:true});
        await button.focus();await page.keyboard.press('Enter');
        await lesson.getByLabel('Circle position',{exact:true}).focus();await page.keyboard.press('ArrowRight');
        assert.equal(await lesson.getByLabel('Circle position',{exact:true}).inputValue(),'226');
        states=2;
      }
      const geometry=await lesson.evaluate(node=>({
        documentWidth:document.documentElement.scrollWidth,
        fontReady:[...document.fonts].some(f=>f.family==='Space Grotesk'&&f.status==='loaded'),
        equations:[...node.querySelectorAll('.katex-display')].map(n=>({width:n.getBoundingClientRect().width,content:n.scrollWidth,tex:n.querySelector('annotation')?.textContent})),
        svgOverflow:[...node.querySelectorAll('.dg-plot')].flatMap(svg=>[...svg.querySelectorAll('text')].filter(n=>{const b=n.getBBox();return b.x<-.5||b.x+b.width>svg.viewBox.baseVal.width+.5;}).map(n=>n.textContent)),
        controls:[...node.querySelectorAll('.dg-lab input,.dg-lab button')].map(n=>({name:n.getAttribute('aria-label')||n.textContent,height:n.getBoundingClientRect().height})),
      }));
      assert.equal(geometry.documentWidth,width);assert.ok(geometry.fontReady);
      assert.ok(geometry.controls.every(row=>row.height>=43));
      assert.deepEqual(errors,[]);assert.deepEqual(warnings,[]);assert.deepEqual(failedRequests,[]);
      results.push({width,states,anchors,programCount:programs.length,independentPractice:11,geometry,errors,warnings,failedRequests});
      fs.writeFileSync(directory+(readingOnly?'/final-reading-in-progress.json':'/in-progress-results.json'),JSON.stringify(results,null,2));
      console.log(width,'behavior passed',states,'states; equation overflow',geometry.equations.filter(e=>e.content>e.width+1).map(e=>e.tex));
      await page.close();
    }
  }finally{await browser.close();}
  fs.writeFileSync(directory+(readingOnly?'/final-reading-results.json':'/results.json'),JSON.stringify({checkedAt:new Date().toISOString(),readingOnly,results},null,2));
  assert.ok(results.every(row=>row.geometry.equations.every(e=>e.content<=e.width+1)),'Equation layout needs repair');
  assert.ok(results.every(row=>row.geometry.svgOverflow.length===0),'SVG labels need repair');
})().catch(error=>{console.error(error);process.exitCode=1;});

const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const directory = 'scratch/single-variable-calculus-browser';
(async () => {
  const browser = await chromium.launch({ channel:'msedge', headless:true });
  const result = { at:new Date().toISOString(), records:[], images:[], purpose:'Final equation and reading review after line-break-only body amendments; complete behavioral states remain recorded separately.' };
  try {
    const { singleVariableCalculusExamples: examples } = await import('../src/learn/data/single-variable-calculus-examples.js');
    for (const width of [1440,390,320]) {
      const page = await browser.newPage({viewport:{width,height:1050}});
      const errors=[]; page.on('pageerror', e=>errors.push(e.message));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/single-variable-calculus-limits-derivatives-integrals?module=math-foundations',{waitUntil:'domcontentloaded'});
      const lesson=page.locator('.single-calculus-lesson'); await lesson.waitFor(); await page.evaluate(()=>document.fonts.ready);
      assert(await page.evaluate(()=>document.fonts.check('16px "Space Grotesk"')));
      // Reading preparation exposes the optional capstone program as well.
      await lesson.locator('details').evaluateAll(nodes=>nodes.forEach(node=>{node.open=true;}));
      for (let i=0;i<examples.length;i++) {
        const program=lesson.locator('.python-example').filter({has:page.getByRole('heading',{name:examples[i].title,exact:true})});
        const code=await program.innerText();
        assert(code.includes(examples[i].code.trim()),examples[i].id+' source'); assert(code.includes(examples[i].expected.trim()),examples[i].id+' output');
        assert((await program.evaluate(node=>node.previousElementSibling.textContent)).includes(examples[i].question));
      }
      // Prepare all optional prose for a reading/fit pass, not a disclosure-interaction claim.
      await lesson.locator('details').evaluateAll(nodes=>nodes.forEach(node=>{node.open=true;}));
      const equations=await lesson.locator('.katex-display').evaluateAll(nodes=>nodes.map((n,i)=>({i,client:n.clientWidth,scroll:n.scrollWidth})));
      assert.equal(equations.length,24); assert.equal(await lesson.locator('.katex-error').count(),0);
      assert(equations.every(e=>e.scroll<=e.client+2),JSON.stringify(equations.filter(e=>e.scroll>e.client+2)));
      const capture=async(locator,name)=>{
        const height=Math.max(1050,Math.ceil((await locator.boundingBox()).height)+160);
        await page.setViewportSize({width,height});
        await locator.evaluate(node=>scrollTo({top:node.getBoundingClientRect().top+scrollY-84,behavior:'instant'}));
        const file=`${directory}/final-${name}-${width}.png`; await locator.screenshot({path:file});
        result.images.push({path:file,captureViewport:{width,height},sha256:crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex')});
        await page.setViewportSize({width,height:1050});
      };
      for(let i=0;i<24;i++) if(width===320) await capture(lesson.locator('.katex-display').nth(i),`equation-${i}`);
      for(let i=0;i<4;i++) await capture(lesson.locator('.calculus-figure').nth(i),`inline-${i}`);
      await capture(lesson.locator('.calculus-practice').nth(8),'changed-decay-practice');
      await capture(lesson.locator('.python-example').nth(14),'changed-motion-program');
      const motion=lesson.locator('.calculus-lab').first();
      const geometry=await motion.locator('svg').evaluate(svg=>{
        const rect=svg.querySelector('clipPath rect');
        const points=svg.querySelector('polyline.gold').points;
        return {left:+rect.getAttribute('x'),top:+rect.getAttribute('y'),width:+rect.getAttribute('width'),height:+rect.getAttribute('height'),points:Array.from({length:points.numberOfItems},(_,i)=>({x:points.getItem(i).x,y:points.getItem(i).y}))};
      });
      assert.equal(geometry.points.length,161);
      let maxError=0;
      for(let i=0;i<geometry.points.length;i++) {
        const t=4*i/160, expected=t*t*t-6*t*t+9*t;
        const actual=6-(geometry.points[i].y-geometry.top)*8/geometry.height;
        maxError=Math.max(maxError,Math.abs(actual-expected));
        assert(Math.abs(geometry.points[i].x-(geometry.left+t*geometry.width/4))<.00003);
      }
      assert(maxError<.000002);
      assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)); assert.deepEqual(errors,[]);
      result.records.push({width,equations:equations.length,actualPrograms:examples.length,actualMotionCurvePoints:geometry.points.length,maxSvgFloatError:maxError,errors});
      fs.writeFileSync(`${directory}/final-reading-results.json`,JSON.stringify(result,null,2));
      await page.close();
    }
    result.passed=true; fs.writeFileSync(`${directory}/final-reading-results.json`,JSON.stringify(result,null,2)); console.log(JSON.stringify({passed:true,records:result.records,images:result.images.length}));
  } finally { await browser.close(); }
})().catch(e=>{console.error(e);process.exitCode=1;});

const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');
(async () => {
  const browser = await chromium.launch({channel:'msedge',headless:true});
  const results=[];
  try {
    for(const width of [1440,390,320]) {
      const page=await browser.newPage({viewport:{width,height:1000}});
      await page.routeWebSocket('**',socket=>socket.close());
      const errors=[],failed=[];
      page.on('pageerror',e=>errors.push(e.message));page.on('requestfailed',r=>failed.push(r.url()));
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/it-calculus-stochastic-differential-equations?module=math-foundations',{waitUntil:'networkidle'});
      const lesson=page.locator('.ito-sde-lesson');await lesson.waitFor();await page.evaluate(()=>document.fonts.ready);
      const geometry=await lesson.evaluate(element=>({
        fonts:[...document.fonts].some(font=>font.family==='Space Grotesk'&&font.status==='loaded'),
        documentWidth:document.documentElement.scrollWidth,
        equations:[...element.querySelectorAll('.katex-display')].map(node=>({width:node.getBoundingClientRect().width,content:node.scrollWidth,tex:node.querySelector('annotation')?.textContent})),
        svgOverflow:[...element.querySelectorAll('.ito-plot svg')].flatMap(svg=>[...svg.querySelectorAll('text')].filter(node=>{const box=node.getBBox();return box.x<-.5||box.x+box.width>svg.viewBox.baseVal.width+.5;}).map(node=>node.textContent)),
        compact:[...element.querySelectorAll('.compact svg')].map(svg=>({width:svg.getBoundingClientRect().width,container:svg.parentElement.clientWidth,labelSize:parseFloat(getComputedStyle(svg.querySelector('text')).fontSize)*svg.getBoundingClientRect().width/svg.viewBox.baseVal.width})),
      }));
      for(let i=0;i<await lesson.locator('.katex-display').count();i++){
        const target=lesson.locator('.katex-display').nth(i);
        await target.evaluate(e=>window.scrollTo(0,scrollY+e.getBoundingClientRect().top-180));
        await page.screenshot({path:`scratch/ito-sde-browser/equation-${i+1}-${width}.png`});
      }
      const figure=lesson.locator('.ito-figure').nth(1);
      await figure.evaluate(e=>window.scrollTo(0,scrollY+e.getBoundingClientRect().top-180));
      await page.screenshot({path:`scratch/ito-sde-browser/compact-curvature-${width}.png`});
      results.push({width,geometry,errors,failed});
      console.log(width,JSON.stringify(geometry.equations.filter(e=>e.content>e.width+1)),JSON.stringify(geometry.svgOverflow));
      assert.ok(geometry.fonts);assert.equal(geometry.documentWidth,width);assert.deepEqual(errors,[]);assert.deepEqual(failed,[]);
      await page.close();
    }
  }finally{await browser.close();}
  fs.writeFileSync('scratch/ito-sde-browser/reading-results.json',JSON.stringify({checkedAt:new Date().toISOString(),results},null,2));
  assert.ok(results.every(r=>r.geometry.equations.every(e=>e.content<=e.width+1)));
  assert.ok(results.every(r=>r.geometry.svgOverflow.length===0));
  assert.ok(results.every(r=>r.geometry.compact.every(c=>c.width<=c.container+1&&c.labelSize>=14)));
})().catch(error=>{console.error(error);process.exitCode=1;});

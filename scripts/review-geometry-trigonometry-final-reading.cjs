const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');
(async () => {
  const browser = await chromium.launch({channel:'msedge',headless:true});
  const records=[];
  try {
    for (const width of [1440,390,320]) {
      const page=await browser.newPage({viewport:{width,height:1080}});
      const errors=[];
      page.on('pageerror',error=>errors.push(error.message));
      page.on('console',message=>{if(message.type()==='error') errors.push(message.text());});
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/geometry-trigonometry-coordinate-reasoning',{waitUntil:'domcontentloaded',timeout:60000});
      const lesson=page.locator('.geometry-lesson');
      await lesson.waitFor(); await page.evaluate(()=>document.fonts.ready);
      const arc=lesson.locator('[aria-label="Angle and arc investigation"]');
      assert((await arc.innerText()).includes('The drawing keeps a fixed scale while you change the radius.'));
      assert(!(await arc.innerText()).includes('39 pixels'));
      await arc.evaluate(node=>scrollTo({top:node.getBoundingClientRect().top+scrollY-84,behavior:'instant'}));
      await page.screenshot({path:`scratch/geometry-trigonometry-browser/final-arc-copy-${width}.png`});
      const radii=[];
      for(const radius of [1,2]) {
        await arc.getByRole('slider',{name:'Radius',exact:true}).fill(String(radius));
        radii.push(await arc.locator('svg circle').first().evaluate(node=>node.getBoundingClientRect().width/2));
      }
      assert(Math.abs(radii[1]/radii[0]-2)<0.001, JSON.stringify(radii));
      const math=await lesson.locator('.katex-display').evaluateAll(nodes=>nodes.map(node=>({width:node.clientWidth,content:node.scrollWidth})));
      assert(math.every(item=>item.content<=item.width+2));
      assert.equal(await lesson.locator('.katex-error').count(),0);
      assert(await page.evaluate(()=>document.fonts.check('16px "Space Grotesk"')));
      assert(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
      assert.deepEqual(errors,[]);
      records.push({width,actualCSSRadii:radii,radiusRatio:radii[1]/radii[0],finalCaptionVerified:true,formulas:math.length,errors});
      await page.close();
    }
  } finally {await browser.close();}
  const result={at:new Date().toISOString(),passed:true,records};
  fs.writeFileSync('scratch/geometry-trigonometry-browser/final-reading-results.json',JSON.stringify(result,null,2));
  console.log(JSON.stringify(result));
})().catch(error=>{console.error(error);process.exit(1);});

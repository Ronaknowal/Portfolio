// Bounded independent capture and geometry check after graph-label repairs.
const fs=require('node:fs'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const {chromium}=require(process.env.PLAYWRIGHT_PACKAGE||'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const sources=['src/learn/components/lesson-labs/NormalizationLabs.jsx','src/learn/components/lesson-labs/normalization-labs.css','src/learn/components/lesson-labs/NeuralLessonElements.jsx','src/learn/components/lesson-labs/neural-lesson-elements.css'];
const hash=file=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const report={passed:false,base:process.env.LEARNING_BASE_URL||'http://127.0.0.1:4197',sourceHashes:Object.fromEntries(sources.map(file=>[file,hash(file)])),captures:[],checks:[]};
const receipt='docs/teaching/evidence/normalization-independent-visual-closure.json';
fs.writeFileSync(receipt,JSON.stringify(report,null,2));
(async()=>{
 const browser=await chromium.launch({channel:'msedge',headless:true});
 try{
  const page=await browser.newPage({viewport:{width:1366,height:1000},reducedMotion:'reduce'}),fonts=JSON.parse(fs.readFileSync('scratch/kmeans-revision-review/fonts/manifest.json','utf8'));
  await page.route('https://fonts.googleapis.com/**',route=>route.fulfill({path:fonts.stylesheet,contentType:'text/css'}));
  await page.route('https://fonts.gstatic.com/**',route=>fonts.files[route.request().url()]?route.fulfill({path:fonts.files[route.request().url()],contentType:'font/ttf'}):route.continue());
  await page.goto(`${report.base}/learn/path/full-curriculum/batch-layer-group-rms-normalization?module=deep-learning-fundamentals`,{timeout:60000});
  await page.locator('[data-lab="normalization-gradients"]').waitFor();await page.evaluate(()=>document.fonts.ready);
  const bounds=await page.locator('.normalization-dependency-graph text,.normalization-residual-graph text').evaluateAll(elements=>elements.map(element=>{const b=element.getBBox(),v=element.ownerSVGElement.viewBox.baseVal;return{text:element.textContent,x:b.x,y:b.y,right:b.x+b.width,bottom:b.y+b.height,width:v.width,height:v.height};}));
  for(const b of bounds)assert.ok(b.x>=0&&b.y>=0&&b.right<=b.width&&b.bottom<=b.height,JSON.stringify(b));
  report.checks.push('Every gradient and residual graph text box fits its actual SVG viewBox');
  const directory='docs/teaching/evidence/normalization-independent-browser';fs.mkdirSync(directory,{recursive:true});
  for(const width of [1366,320]){
   await page.setViewportSize({width,height:1000});
   for(const name of ['gradients','placement']){
    const panel=page.locator(`[data-lab="normalization-${name}"]`),locator=name==='gradients'?panel.locator('figure'):panel;
    const box=await locator.boundingBox();await page.setViewportSize({width,height:Math.max(1000,Math.ceil(box.height)+220)});
    await locator.evaluate(element=>element.scrollIntoView({block:'center',behavior:'instant'}));await page.waitForTimeout(120);
    const path=`${directory}/${width}-${name}.png`;await locator.screenshot({path});report.captures.push(path);
    if(width===320&&name==='gradients'){
     const region=panel.locator('.normalization-graph-scroll');await region.focus();await region.press('End');
     await region.evaluate(element=>element.scrollLeft=element.scrollWidth-element.clientWidth);
     assert.ok(await region.evaluate(element=>element.scrollLeft>0));
     const path=`${directory}/${width}-${name}-scroll-end.png`;await locator.screenshot({path});report.captures.push(path);
    }
   }
   assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
  }
  report.checks.push('320 px dependency graph exposes its remaining branch in its own scroll region; page stays contained at 1366 and 320 px');
  for(const file of sources)assert.equal(hash(file),report.sourceHashes[file],`${file} changed during capture`);
  report.passed=true;fs.writeFileSync(receipt,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({passed:true,checks:report.checks.length,captures:report.captures.length}));
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});

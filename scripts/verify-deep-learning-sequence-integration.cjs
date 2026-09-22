const fs = require('node:fs');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const installFonts = require('./lib/lesson-browser-fonts.cjs');
const { inspectLessonVisualLayout } = require('./lib/lesson-visual-layout.cjs');
const base = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4194';
const read = file => JSON.parse(fs.readFileSync(file, 'utf8'));
const hash = file => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const manifest = read('dist/.vite/manifest.json');
const lessons = read('src/learn/data/lesson-manifest.json');
const ids = read('docs/teaching/evidence/deep-learning-sequence-baseline.json').scope;
const next = [...ids.slice(1), 'attention-mechanism-bahdanau-luong'];
const chunks = Object.entries(lessons).map(([id, path]) => ({id, file:manifest[`src/learn/data/${path.slice(2)}`].file}));
const output = 'docs/teaching/evidence/deep-learning-sequence-production-integration.json';
const report = {passed:false, checkedAt:new Date().toISOString(), base, manifestHash:hash('dist/.vite/manifest.json'), routes:[], recovery:[]};
const save = () => fs.writeFileSync(output, JSON.stringify(report,null,2)+'\n');
save();
(async()=>{
 const browser = await chromium.launch({channel:'msedge',headless:true});
 try {
  for (const [i,id] of ids.entries()) {
   const context = await browser.newContext({viewport:{width:1366,height:1000},reducedMotion:'reduce'});
   await installFonts(context);
   const page = await context.newPage(), requests = new Set(), errors=[];
   page.on('request',r=>requests.add(new URL(r.url()).pathname.slice(1)));
   page.on('pageerror',e=>errors.push(e.message));
   await page.goto(`${base}/learn/path/full-curriculum/${id}?module=deep-learning-fundamentals`);
   await page.locator('.reader-article[aria-busy=false] .lesson-intro, .reader-article[aria-busy=false] h2').first().waitFor();
   await page.evaluate(()=>document.fonts.ready);
   const body = page.locator('.reader-article').last();
   assert.equal(await page.locator('.lesson-load-error,.katex-error').count(),0);
   const loaded = chunks.filter(row=>requests.has(row.file)).map(row=>row.id);
   assert.deepEqual(loaded,[id]);
   const anchors = await body.locator('a[href^="#"]').evaluateAll(nodes=>nodes.map(node=>({href:node.hash,count:document.querySelectorAll(`[id="${CSS.escape(decodeURIComponent(node.hash.slice(1)))}"]`).length})));
   assert.ok(anchors.length>=8 && anchors.every(row=>row.count===1),JSON.stringify(anchors));
   const math = await body.locator('.katex .sqrt svg,.katex .accent svg').evaluateAll(nodes=>nodes.map(node=>({width:node.getBoundingClientRect().width,height:node.getBoundingClientRect().height})));
   assert.ok(math.every(box=>box.width>1&&box.height>1),`${id}: collapsed math`);
   const colors = await body.locator('a[href]').evaluateAll(nodes=>[...new Set(nodes.map(node=>getComputedStyle(node).color))]);
   assert.ok(colors.every(color=>{const rgb=color.match(/[\d.]+/g).map(Number); return rgb[0]>=rgb[1]&&rgb[1]>rgb[2];}),`${id}: non-theme links ${colors}`);
   const javascriptBytes = [...requests].filter(file=>file.endsWith('.js')&&fs.existsSync(`dist/${file}`)).reduce((sum,file)=>sum+fs.statSync(`dist/${file}`).size,0);
   const downloads = await body.locator('a[href^="/learn-assets/"],a[href^="/learn-code/"]').evaluateAll(nodes=>[...new Set(nodes.map(node=>node.getAttribute('href')))]);
   assert.ok(downloads.some(url=>url.endsWith('.py')));
   for (const url of downloads) {
    const response = await page.request.get(base+url);
    assert.equal(response.status(),200,`${id}: ${url}`);
    assert.equal(createHash('sha256').update(await response.body()).digest('hex'),hash('public'+url));
   }
   const geometry=[];
   for(const width of [1366,390,320]) {
    await page.setViewportSize({width,height:1000});
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),`${id}: overflow ${width}`);
    const findings=await page.evaluate(inspectLessonVisualLayout,'.reader-article[aria-busy="false"]');
    geometry.push({width,findings});
   }
   await page.locator('.reader-footer__next').click();
   await page.waitForURL(`**/${next[i]}?module=deep-learning-fundamentals`);
   assert.deepEqual(errors,[]);
   report.routes.push({id,loaded,anchors:anchors.length,mathSvg:math.length,colors,next:next[i],javascriptBytes,exactDownloads:downloads.length,geometry});
   await context.close();
  }
  // Network-only failures: no filesystem source mutation or old evidence rewrite.
  for(const kind of ['import','render']) {
   const context = await browser.newContext(); await installFonts(context);
   const page=await context.newPage(), id=ids[0], file=chunks.find(row=>row.id===id).file;
   let requests=0;
   await page.route(`**/${file}*`,async route=>{
    requests++;
    if(requests===1) {
     if(kind==='import') await route.abort('failed');
     else await route.fulfill({status:200,contentType:'text/javascript',body:'export default { content() { throw new Error("Scoped DL sequence render failure"); } };'});
    } else await route.continue();
   });
   await page.goto(`${base}/learn/path/full-curriculum/${id}?module=deep-learning-fundamentals`);
   const error=page.locator('.lesson-load-error'); await error.waitFor();
   assert.ok(await page.locator('.reader-complete').isDisabled());
   assert.equal(await page.locator('.planned-lesson').count(),0);
   if(kind==='import') {
    await error.getByRole('button',{name:'Try again',exact:true}).focus(); await page.keyboard.press('Enter');
    await page.waitForFunction(()=>document.querySelector('.depthwise-lesson')||document.querySelector('.lesson-load-error'));
   }
   if(await error.count()) await error.getByRole('button',{name:'Reload page',exact:true}).click();
   await page.locator('.depthwise-lesson').waitFor();
   assert.ok(requests>1); assert.ok(await page.locator('.reader-complete').isEnabled());
   report.recovery.push({kind,requests,recovered:true}); await context.close();
  }
  report.passed=true;
 } catch(error) {report.failure=error.stack;process.exitCode=1;console.error(error);}
 finally {await browser.close();report.verifierHash=hash(__filename);save();console.log(JSON.stringify({passed:report.passed,routes:report.routes.length,recovery:report.recovery,failure:report.failure}));}
})();

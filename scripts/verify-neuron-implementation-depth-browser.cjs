const fs=require('node:fs'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const {chromium}=require(process.env.PLAYWRIGHT_PACKAGE||'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const base=process.env.LEARNING_BASE_URL||'http://127.0.0.1:4197';
const cases=[['perceptrons-neurons-activation-functions','perceptron-code-route','perceptron'],['backpropagation-automatic-differentiation','backprop-code-route','backprop'],['transfer-learning-fine-tuning-strategies','transfer-code-route','transfer']];
const report={status:'incomplete',base,checks:[],captures:[]},receipt='docs/teaching/evidence/neuron-implementation-depth-browser.json';
fs.writeFileSync(receipt,JSON.stringify(report,null,2));
(async()=>{
 const metadataFiles={perceptron:'perceptron-mechanism-program.js',backprop:'backprop-mechanism-program.js',transfer:'transfer-learning-mechanism-program.js'};
 const examples=Object.fromEntries(await Promise.all(Object.entries(metadataFiles).map(async([key,file])=>[key,(await import(`../src/learn/data/${file}`)).default])));
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const fonts=JSON.parse(fs.readFileSync('scratch/kmeans-revision-review/fonts/manifest.json','utf8'));
 const directory='docs/teaching/evidence/neuron-implementation-depth-browser';fs.mkdirSync(directory,{recursive:true});
 try{
  for(const [topic,id,key]of cases){
   const page=await browser.newPage({viewport:{width:1366,height:1000},reducedMotion:'reduce',acceptDownloads:true});
   const errors=[];page.on('pageerror',error=>errors.push(error.message));
   await page.route('https://fonts.googleapis.com/**',route=>route.fulfill({path:fonts.stylesheet,contentType:'text/css'}));
   await page.route('https://fonts.gstatic.com/**',route=>fonts.files[route.request().url()]?route.fulfill({path:fonts.files[route.request().url()],contentType:'font/ttf'}):route.continue());
   let requests=0;page.on('request',request=>{if(request.url().endsWith(examples[key].source))requests++;});
   await page.goto(`${base}/learn/path/full-curriculum/${topic}?module=deep-learning-fundamentals`,{timeout:60000});
   const section=page.locator(`#${id}`),program=section.locator('.mechanism-program');await program.waitFor();await page.evaluate(()=>document.fonts.ready);
   assert.equal(requests,0);assert.equal(await program.locator('pre').count(),0);
   const summary=program.locator('summary');await summary.focus();await summary.press('Enter');
   const code=program.getByRole('region',{name:'Complete Python program; scroll horizontally when needed',exact:true});await code.waitFor();
   const source=fs.readFileSync('public'+examples[key].source,'utf8').replace(/\r\n/g,'\n');
   assert.equal((await code.textContent()).replace(/\r\n/g,'\n'),source);
   assert.equal(await program.getByRole('region',{name:'Recorded program output; scroll horizontally when needed',exact:true}).textContent(),examples[key].output);
   const theme=await code.evaluate(element=>{const style=getComputedStyle(element);return {color:style.color,background:style.backgroundColor,fontFamily:style.fontFamily};});
   assert.equal(theme.color,'rgb(214, 223, 217)');assert.equal(theme.background,'rgb(16, 21, 18)');assert.match(theme.fontFamily,/JetBrains Mono/);
   assert.equal(requests,1);
   await summary.click();await page.waitForTimeout(50);assert.equal(await program.locator('pre').count(),0);
   await summary.click();await code.waitFor();assert.equal(requests,1);
   const downloadPromise=page.waitForEvent('download');await program.getByRole('link',{name:/Download/}).click();const download=await downloadPromise;
   assert.equal(download.suggestedFilename(),examples[key].source.split('/').at(-1));
   assert.equal(fs.readFileSync(await download.path(),'utf8').replace(/\r\n/g,'\n'),source);
   report.checks.push(`${key}: deferred source fetch, keyboard open, exact complete code/output, unmount on close, cached reopen and byte-matched download`);
   for(const width of [1366,320]){
    await page.setViewportSize({width,height:1000});
    await program.evaluate(element=>element.scrollIntoView({block:'start',behavior:'instant'}));await page.waitForTimeout(120);
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),`${key} ${width} page overflow`);
    if(width===320){await code.focus();await code.press('ArrowRight');await page.waitForTimeout(120);assert.ok(await code.evaluate(element=>element.scrollLeft>0));await code.evaluate(element=>element.scrollLeft=0);}
    await program.getByRole('link',{name:/Download/}).evaluate(element=>window.scrollTo({top:scrollY+element.getBoundingClientRect().top-90,behavior:'instant'}));await page.mouse.move(1,1);await page.waitForTimeout(250);
    const path=`${directory}/${key}-${width}.png`;await page.screenshot({path});report.captures.push(path);
   }
   assert.deepEqual(errors,[]);assert.equal(await page.locator('.katex-error').count(),0);
   report.checks.push(`${key}: 1366/320 containment, focused code scroll, dark-theme code colors and mono font, rendered math and error-free page`);await page.close();
  }
  const page=await browser.newPage({viewport:{width:1366,height:1000},reducedMotion:'reduce'});
  await page.route('https://fonts.googleapis.com/**',route=>route.fulfill({path:fonts.stylesheet,contentType:'text/css'}));
  await page.route('https://fonts.gstatic.com/**',route=>fonts.files[route.request().url()]?route.fulfill({path:fonts.files[route.request().url()],contentType:'font/ttf'}):route.continue());
  let attempts=0;await page.route('**/learn-assets/perceptrons/activation-mechanisms.py',route=>++attempts===1?route.fulfill({status:503,body:'temporarily unavailable'}):route.continue());
  await page.goto(`${base}/learn/path/full-curriculum/${cases[0][0]}?module=deep-learning-fundamentals`,{timeout:60000});
  const program=page.locator('#perceptron-code-route .mechanism-program');await program.locator('summary').click();
  await program.getByRole('alert').waitFor();await program.getByRole('button',{name:'Retry code view'}).click();await program.getByRole('region',{name:'Complete Python program; scroll horizontally when needed',exact:true}).waitFor();assert.equal(attempts,2);await page.close();
  report.checks.push('Shared program disclosure: failed source request shows a local error, real Retry fetch recovers');
  const sources=['src/learn/data/topics/perceptrons-neurons-activation-functions.jsx','src/learn/data/topics/backprop.jsx','src/learn/data/topics/transfer-learning-fine-tuning-strategies.jsx','src/learn/components/lesson-labs/MechanismProgram.jsx','src/learn/components/lesson-labs/mechanism-program.css','src/learn/components/lesson-labs/perceptron-labs.css',...Object.values(metadataFiles).map(file=>`src/learn/data/${file}`)];
  report.sourceHashes=Object.fromEntries(sources.map(path=>[path,crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex')]));
  report.status='passed';fs.writeFileSync(receipt,JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify({status:report.status,groups:report.checks.length,captures:report.captures.length}));
 }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});

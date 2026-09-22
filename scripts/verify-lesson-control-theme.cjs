// Production check for native-looking lesson controls, including inline figures.
// The default scope is the 135 reviewed live-exploration topics; --topics=id,id narrows it.
const fs = require('node:fs');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const read = path => JSON.parse(fs.readFileSync(path, 'utf8').replace(/^\uFEFF/, ''));
const hash = path => createHash('sha256').update(fs.readFileSync(path)).digest('hex');
const base = process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4193';
const requested = process.argv.find(arg => arg.startsWith('--topics='))?.slice(9).split(',');
const topics = read('docs/teaching/evidence/live-exploration-runtime-scope.json').topics.filter(t => !requested || requested.includes(t.id));
assert.equal(topics.length, requested?.length ?? 135, 'The intended topic scope must resolve exactly');
const receipt = requested ? `scratch/control-theme-audit/theme-${createHash('sha256').update(requested.join(',')).digest('hex').slice(0,8)}.json` : 'docs/teaching/evidence/lesson-control-theme.json';
const evidence = { capturedAt:new Date().toISOString(), base, scope:topics.map(t=>t.id), sourceHashes:Object.fromEntries(['src/index.css','src/learn/components/topic-content.css','src/learn/components/lesson-labs/calibration-labs.css','src/learn/components/lesson-labs/CalibrationFigures.jsx','src/learn/data/calibration-models.js','scripts/verify-lesson-control-theme.cjs'].map(path=>[path,hash(path)])), manifestHash:hash('dist/.vite/manifest.json'), records:[], failures:[], passed:false };
const fonts = read(process.env.LEARNING_FONT_FIXTURES || 'scratch/kmeans-revision-review/fonts/manifest.json');
// Browser-side inspection: CSSOM source matches plus computed colors. Inspecting
// a root class alone misses controls whose markup is outside the styled lab.
function inspectControls() {
 const rules=[];
 function collect(list) { for(const rule of list) {
  if(rule.media && !matchMedia(rule.conditionText).matches) continue;
  if(rule.selectorText && rule.style) rules.push(rule);
  else if(rule.cssRules) collect(rule.cssRules);
 }}
 for(const sheet of document.styleSheets) { try { collect(sheet.cssRules); } catch {} }
 return [...document.querySelectorAll('.reader-article button,.reader-article input,.reader-article select,.reader-article textarea')].filter(n=>n.checkVisibility()).map(n=>{
  const style=getComputedStyle(n),rect=n.getBoundingClientRect();
  const matches=rules.filter(rule=>{try{return n.matches(rule.selectorText);}catch{return false;}});
  const authoredBackground=!!(n.style.background||n.style.backgroundColor||matches.some(r=>r.style.background||r.style.backgroundColor));
  const isField=['SELECT','TEXTAREA'].includes(n.tagName)||(n.tagName==='INPUT'&&!['range','checkbox','radio','hidden'].includes(n.type));
  const isButton=n.tagName==='BUTTON';
  const problems=[];
  if((isField||isButton)&&!authoredBackground)problems.push('no authored background');
  if(isButton&&/^rgb\((\d+), \1, \1\)$/.test(style.backgroundColor)&&Number(style.backgroundColor.match(/\d+/)[0])>65)problems.push('bright native-like grey button');
  if(['range','checkbox','radio'].includes(n.type)&&style.accentColor==='auto')problems.push('native accent');
  if(n.type==='range'&&rect.height<24)problems.push('range hit area below 24px');
  return {tag:n.tagName,type:n.type,text:(n.getAttribute('aria-label')||n.textContent||[...(n.labels||[])].map(l=>l.textContent).join(' ')).slice(0,140),background:style.backgroundColor,foreground:style.color,accent:style.accentColor,width:rect.width,height:rect.height,problems};
 });
}
if(!process.argv.includes('--no-evidence')){fs.mkdirSync(require('node:path').dirname(receipt),{recursive:true});fs.writeFileSync(receipt,JSON.stringify(evidence,null,2)+'\n');}
(async()=>{
 let browser;
 try {
  browser=await chromium.launch({channel:'msedge',headless:true});
  const context=await browser.newContext({viewport:{width:1366,height:1000}});
  await context.route('https://fonts.googleapis.com/**',r=>r.fulfill({path:fonts.stylesheet,contentType:'text/css',headers:{'access-control-allow-origin':'*'}}));
  await context.route('https://fonts.gstatic.com/**',r=>fonts.files[r.request().url()]?r.fulfill({path:fonts.files[r.request().url()],contentType:'font/ttf',headers:{'access-control-allow-origin':'*'}}):r.abort());
  let index=0;
  await Promise.all(Array.from({length:4},async()=>{
   const page=await context.newPage();
   while(index<topics.length){const topic=topics[index++];try{
    await page.goto(`${base}/learn/path/full-curriculum/${topic.id}?module=${topic.module}`,{waitUntil:'domcontentloaded'});
    await page.locator('.reader-article h2,.reader-article h3').first().waitFor();
    await page.evaluate(()=>document.fonts.ready);
    const controls=await page.evaluate(inspectControls);
    assert(controls.length>0, `${topic.id} must actually expose controls`);
    const failures=controls.filter(c=>c.problems.length);
    const documentOverflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1);
    assert.equal(documentOverflow,false,`${topic.id} must not overflow the desktop document`);
    evidence.records.push({id:topic.id,controls:controls.length,buttons:controls.filter(c=>c.tag==='BUTTON').length,ranges:controls.filter(c=>c.type==='range').length,failures});
    evidence.failures.push(...failures.map(c=>({topic:topic.id,...c})));
   }catch(error){evidence.failures.push({topic:topic.id,error:error.message});}}
   await page.close();
  }));
  // Measure the fallback under the real application cascade: index.css also
  // styles buttons, so an authored amber rule can exist without actually winning.
  const fallbackPage=await context.newPage();
  await fallbackPage.goto(`${base}/learn/path/full-curriculum/${topics[0].id}?module=${topics[0].module}`,{waitUntil:'domcontentloaded'});
  await fallbackPage.locator('.reader-article h2,.reader-article h3').first().waitFor();
  const fallback=await fallbackPage.locator('.reader-article').first().evaluate(root=>{const button=document.createElement('button');button.type='button';button.textContent='Theme probe';root.append(button);const style=getComputedStyle(button);const result={color:style.color,background:style.backgroundColor};button.remove();return result;});
  assert.deepEqual(fallback,{color:'rgb(226, 181, 90)',background:'rgb(20, 19, 15)'},'A new inline control must receive dark/amber defaults under the actual cascade');
  evidence.fallbackProbe=fallback;
  await fallbackPage.close();
  // Falsify the authored-background check in an isolated DOM. This has no file,
  // asset, or evidence mutation and cannot affect any other browser page.
  const probe=await context.newPage();
  await probe.setContent('<main class="reader-article"><button>Control probe</button></main>');
  const broken=await probe.evaluate(inspectControls);
  assert(broken[0].problems.includes('no authored background'),'The checker must detect a genuinely unstyled control');
  await probe.addStyleTag({content:'.reader-article button { background:#14130f; color:#e2b55a; border:1px solid #746138; }'});
  const repaired=await probe.evaluate(inspectControls);
  assert.equal(repaired[0].problems.length,0,'The same control must pass with an authored theme');
  evidence.guardFalsification='An unstyled isolated button fails; the same button with authored dark/amber styles passes.';
  assert.equal(evidence.records.length,topics.length,'Every route must complete its control inspection');
  assert.deepEqual(evidence.failures,[],'Every rendered lesson control needs its intended theme and usable range target');
  evidence.passed=true;
 }catch(error){evidence.failure=error.message;process.exitCode=1;}finally{
  await browser?.close();
  evidence.records.sort((a,b)=>a.id.localeCompare(b.id));
  if(!process.argv.includes('--no-evidence')){fs.mkdirSync(require('node:path').dirname(receipt),{recursive:true});fs.writeFileSync(receipt,JSON.stringify(evidence,null,2)+'\n');}
  console.log(JSON.stringify({passed:evidence.passed,topics:evidence.records.length,controls:evidence.records.reduce((n,r)=>n+r.controls,0),failures:evidence.failures,receipt},null,2));
 }
})();

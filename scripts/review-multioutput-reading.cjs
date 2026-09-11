const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const { chromium } = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const directory = path.resolve('scratch/multioutput/browser');
(async () => {
  const behavior = JSON.parse(fs.readFileSync(path.join(directory,'behavior-results.json'),'utf8'));
  const sourceHashes = Object.fromEntries(Object.keys(behavior.sourceHashes).map(file => [file,crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex')]));
  const formatting=JSON.parse(fs.readFileSync('scratch/multioutput/format-conservation.json','utf8'));
  for (const [file,hash] of Object.entries(sourceHashes)) if (!file.includes('/topics/')&&!file.endsWith('.css')) assert.equal(hash,behavior.sourceHashes[file]);
  assert.equal(formatting.originalCssSha256,behavior.sourceHashes[formatting.cssFile]);
  assert.equal(formatting.finalCssSha256,sourceHashes[formatting.cssFile]);
  assert(formatting.cssAstConserved);
  const browser = await chromium.launch({ channel:'msedge',headless:true });
  const records=[],errors=[];
  try {
    for (const width of [1440,390,320]) {
      const page=await browser.newPage({ viewport:{width,height:1000},reducedMotion:'reduce' });
      await page.routeWebSocket('**',socket=>socket.close());
      page.on('pageerror',error=>errors.push(error.message));
      page.on('console',message=>{ if(message.type()==='error'&&!message.text().includes('[vite]'))errors.push(message.text()); });
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/multi-label-multi-output-learning?module=classical-ml-supervised',{waitUntil:'domcontentloaded'});
      const lesson=page.locator('.multioutput-lesson');await lesson.waitFor();await page.evaluate(()=>document.fonts.ready);
      const fonts=await page.evaluate(()=>[...document.fonts].filter(font=>font.status==='loaded').map(font=>font.family));
      assert(fonts.some(font=>font.includes('Space Grotesk'))&&fonts.some(font=>font.includes('JetBrains Mono')));
      await page.addStyleTag({content:'html{scroll-behavior:auto!important}'});
      for(const summary of await lesson.locator('details > summary').all()){await summary.focus();await page.keyboard.press('Enter');}
      const equations=await lesson.locator('.katex-display').evaluateAll(nodes=>nodes.map(node=>({text:node.textContent,scroll:node.scrollWidth,client:node.clientWidth})));
      assert.equal(equations.length,12);
      assert.deepEqual(equations.filter(row=>row.scroll>row.client+2),[]);
      assert.equal(await lesson.locator('.katex-error').count(),0);
      const overflow=await lesson.evaluate(node=>{const box=node.getBoundingClientRect();return [...node.querySelectorAll('p,h2,h3,summary,svg,figure,input,select,button,table')].filter(element=>element.getClientRects().length&&!element.closest('.lesson-table-wrap,.mo-table-scroll')).filter(element=>{const rect=element.getBoundingClientRect();return rect.right>box.right+3||rect.left<box.left-3;}).map(element=>element.textContent.slice(0,100));});
      assert.deepEqual(overflow,[]);
      assert((await lesson.innerText()).includes('added. Empirical precision'));
      const captures=[];
      async function shot(target,name){await target.evaluate(node=>window.scrollTo({top:node.getBoundingClientRect().top+scrollY-90,behavior:'instant'}));await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));await page.waitForTimeout(150);const filename=`${name}-${width}.png`;await page.screenshot({path:path.join(directory,filename)});captures.push(filename);}
      if(width===1440){await shot(lesson.locator('.mo-inline-figure').last(),'labelset-figure');await shot(lesson.locator('h2').nth(6),'association-reading');const regression=lesson.getByRole('region',{name:'A shared split must serve two measurements',exact:true});await regression.getByLabel('Energy divisor',{exact:true}).selectOption('100');await shot(regression,'shared-split-units');}
      if(width===390){await shot(lesson.locator('h2').nth(9),'continuous-reading');await shot(lesson.locator('.katex-display').nth(8),'ridge-equation');}
      if(width===320){await shot(lesson.locator('.katex-display').nth(8),'ridge-equation');await shot(lesson.locator('.katex-display').nth(9),'shared-penalty-equation');await shot(lesson.locator('.mo-practice').nth(2),'changed-joint-reading');}
      const references=await lesson.locator('.lesson-sources a').count();
      records.push({width,fonts,equations,overflow,references,captures});await page.close();
    }
    assert.deepEqual(errors,[]);
    fs.writeFileSync(path.join(directory,'final-reading-results.json'),JSON.stringify({timestamp:new Date().toISOString(),sourceHashes,records,errors,passed:true,scope:'Final actual-font reading, disclosure, formula and page geometry after one remaining aligned-equation line-layout repair; preceding complete behavior run reused for unchanged interactions.'},null,2));
    console.log(JSON.stringify(records.map(record=>({width:record.width,equations:record.equations.length,captures:record.captures})),null,2));
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exit(1);});

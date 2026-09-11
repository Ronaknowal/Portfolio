const fs=require('node:fs');
const {chromium}=require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
(async()=>{
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const results=[];
 for(const width of [390,320]) {
  const page=await browser.newPage({viewport:{width,height:1000}});
  await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/numerical-pdes-grids-finite-elements-stability?module=math-foundations');
  await page.locator('.npde-lesson').waitFor();await page.evaluate(()=>document.fonts.ready);
  await page.locator('.npde-lesson details').evaluateAll(nodes=>nodes.forEach(node=>node.open=true));
  results.push(await page.evaluate(()=>{const texts=[];const walker=document.createTreeWalker(document.querySelector('.npde-lesson'),NodeFilter.SHOW_TEXT);let node;while(node=walker.nextNode()){if(node.parentElement.closest('.katex,.python-example,table,svg'))continue;const range=document.createRange();range.selectNodeContents(node);for(const rect of range.getClientRects())if(rect.right>innerWidth+1||rect.left< -1)texts.push({text:node.textContent,right:rect.right});}return {width:innerWidth,scroll:document.documentElement.scrollWidth,texts,elements:[...document.querySelectorAll('.npde-lesson *')].flatMap(node=>{const box=node.getBoundingClientRect();return box.right>innerWidth+1||box.left< -1?[{tag:node.tagName,cls:node.className,width:box.width,left:box.left,right:box.right,text:node.textContent.slice(0,130),overflow:getComputedStyle(node).overflowX}]:[];})};}));
  await page.close();
 }
 fs.writeFileSync('scratch/numerical-pde-browser/layout-diagnostic.json',JSON.stringify(results,null,2));
 console.log(JSON.stringify(results.map(r=>({...r,elements:r.elements.filter(x=>x.overflow==='visible').slice(0,35)})),null,2));
 await browser.close();
})().catch(error=>{console.error(error);process.exitCode=1;});

const {chromium}=require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('node:fs');
(async()=>{
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const page=await browser.newPage({viewport:{width:320,height:1100}});
 await page.routeWebSocket('**',socket=>socket.close());
 await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/dynamic-programming-states-transitions-optimization?module=data-structures-algorithms');
 await page.locator('.dynamic-programming-lesson').waitFor();
 await page.evaluate(()=>document.fonts.ready);
 const measure=async()=>await page.evaluate(()=>({document:document.documentElement.scrollWidth,innerWidth,elements:[...document.querySelectorAll('body *')].filter(node=>{
  const rect=node.getBoundingClientRect();return rect.width&&rect.right>innerWidth+1;
 }).map(node=>({tag:node.tagName,class:node.className,overflow:getComputedStyle(node).overflowX,left:node.getBoundingClientRect().left,right:node.getBoundingClientRect().right,width:node.getBoundingClientRect().width,text:node.textContent.slice(0,100)}))}));
 const initial=await measure();
 const probes=await page.evaluate(()=>{
  const out=[];
  for(const selector of ['.dp-scroll','.lesson-table-wrap','.dpf-lab','.dpf-inline','.lesson-intro','.reader-article','.dp-lab','svg','table','code','.python-example','.reader-header','.reader-content']){
   const nodes=[...document.querySelectorAll(selector)];
   const previous=nodes.map(node=>node.style.display);
   nodes.forEach(node=>node.style.display='none');
   out.push({selector,width:document.documentElement.scrollWidth,count:nodes.length});
   nodes.forEach((node,index)=>node.style.display=previous[index]);
  }
  return out;
 });
 console.log('PROBES',JSON.stringify(probes));
 const textOverflow=await page.evaluate(()=>{
  const walker=document.createTreeWalker(document.querySelector('.reader-article'),NodeFilter.SHOW_TEXT);const result=[];
  for(let node=walker.nextNode();node;node=walker.nextNode()){
   let parent=node.parentElement,clipped=false;
   while(parent&&parent!==document.body){if(['auto','scroll','hidden','clip'].includes(getComputedStyle(parent).overflowX)){clipped=true;break;}parent=parent.parentElement;}
   if(clipped)continue;
   const range=document.createRange();range.selectNodeContents(node);
   const rectangles=[...range.getClientRects()].filter(rectangle=>rectangle.width&&rectangle.right>innerWidth+1);
   if(rectangles.length)result.push({text:node.textContent,tag:node.parentElement.tagName,class:node.parentElement.className,rectangles:rectangles.map(rectangle=>rectangle.toJSON())});
  }return result;
 });
 console.log('TEXT',JSON.stringify(textOverflow));
 const command=page.locator('.lesson-check').filter({hasText:'Explain a recurrence and one full implementation'}).locator('summary');
 await command.click();
 await page.locator('.dsa-practice summary').first().click();
 const opened=await measure();
 console.log(JSON.stringify({initial,opened},null,2));
 fs.writeFileSync('scratch/dp-state-families-browser/overflow-debug.json',JSON.stringify({initial,opened},null,2));
 await browser.close();
})();

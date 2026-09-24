const {chromium}=require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('node:fs');
(async()=>{
 const browser=await chromium.launch({channel:'msedge',headless:true});
 const page=await browser.newPage({viewport:{width:1440,height:1100}});
 await page.routeWebSocket('**',socket=>socket.close());
 page.on('pageerror',error=>console.log('PAGE ERROR',error.message));
 await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/dynamic-programming-states-transitions-optimization?module=data-structures-algorithms');
 await page.locator('.dynamic-programming-lesson').waitFor();
 const {matrixChainPlan}=await import('../src/learn/data/dp-state-families-models.js');
 const interval=page.getByRole('region',{name:'Interval split investigation',exact:true});
 try{
 for(const dimensions of [[8,2,12,3,6],[3,7,2,5],[2,2,2,2],[3,7],[2,3,4,5,6,7,8]]){
 await interval.getByLabel('Matrix-chain dimensions',{exact:true}).fill(dimensions.join(','));
 await interval.getByRole('button',{name:'Apply dimensions',exact:true}).click();
 for(const cell of matrixChainPlan(dimensions).cells){
 console.log('CELL',dimensions,cell.left,cell.right,cell.candidates.length);
 await interval.getByRole('button',{name:`Inspect interval ${cell.left} to ${cell.right}`,exact:true}).click();
 console.log((await interval.innerText()).slice(-1500));
 for(const candidate of cell.candidates) await interval.getByLabel('Candidate final split',{exact:true}).selectOption(String(candidate.split),{timeout:3000});
 }
 }
 }catch(error){console.log(error.message);fs.writeFileSync('scratch/dp-state-families-browser/debug.html',await page.content());await page.screenshot({path:'scratch/dp-state-families-browser/debug.png'});}
 await browser.close();
})();

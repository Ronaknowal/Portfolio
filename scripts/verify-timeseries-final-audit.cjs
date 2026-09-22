// Complementary checks for the final Classical ML audit. Uses the running site;
// never builds, changes source, or replaces the earlier author evidence.
const fs = require('node:fs');
const assert = require('node:assert/strict');
const {createHash} = require('node:crypto');
const {chromium} = require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const base=process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4184';
const output='docs/teaching/evidence/classical-final-timeseries-browser.json';
const report={startedAt:new Date().toISOString(),passed:false,base,cases:[],screenshots:[],limitations:['Targeted final audit on a dev server; production integration is separate.','Automated checks plus separately recorded image inspection; no observed beginner study.']};
const hash=p=>createHash('sha256').update(fs.readFileSync(p)).digest('hex');
function check(name,fn){fn();report.cases.push(name);}
const save=()=>fs.writeFileSync(output,JSON.stringify(report,null,2)+'\n');
save();
(async()=>{
 const browser=await chromium.launch({channel:'msedge',headless:true});
 try {
 const page=await browser.newPage({viewport:{width:1366,height:900}});
 const errors=[]; page.on('pageerror',e=>errors.push(e.message));
 await page.goto(`${base}/learn/path/full-curriculum/time-series-validation-forecasting-baselines?module=classical-ml`,{waitUntil:'domcontentloaded'});
 await page.locator('.ts-investigation').last().waitFor();
 const lab=page.locator('.ts-investigation').nth(2);
 const initial=[];
 for(const origin of ['364','371','476','609']) {
  await lab.getByLabel(/forecast origin/).selectOption(origin);
  for(let h=1;h<=7;h++) {
   await lab.getByLabel(/replace a future outcome/).selectOption(String(h));
   initial.push(await lab.getByLabel(/with this count/).nth(1).inputValue());
  }
 }
 check('28 origin/horizon selections keep the future editor independent of concealed source outcomes',()=>assert.deepEqual(initial,Array(28).fill('0')));
 const capture=async(locator,name)=>{const p=`docs/teaching/evidence/screenshots/final-audit-timeseries-${name}.png`;await locator.screenshot({path:p,style:'.learn-nav {visibility:hidden !important}'});report.screenshots.push({path:p,sha256:hash(p)});};
 await lab.getByRole('button',{name:'Reset',exact:true}).click();
 report.cases.push('No outcome table or verdict before issuing');
 assert.equal(await lab.locator('.ts-verdict').count(),0);
 assert.equal(await lab.getByText('What actually happened.',{exact:false}).count(),0);
 await lab.locator('input[type=radio][value=higher]').check();
 await lab.locator('.ts-numeric-guess input').fill('1162');
 await lab.getByRole('button',{name:'Issue the forecast',exact:true}).click();
 assert.equal(await lab.locator('.ts-verdict').count(),0);
 assert.equal(await lab.getByLabel(/with this count/).nth(1).inputValue(),'0');
 await capture(lab,'issued-desktop');
 await lab.getByRole('button',{name:/Reveal.*outcomes/}).click();
 assert.match(await lab.locator('.ts-verdict').innerText(),/matches/);
 assert.match(await lab.innerText(),/1,042\.571429|1042\.571429/);
 report.cases.push('Issue/reveal gates and exact seasonal forecast1162 and MAE7298/7');
 const forecasts=await lab.locator('.ts-table').first().locator('tbody').innerText();
 await lab.getByRole('button',{name:'Show the saved ridge evidence',exact:true}).click();
 await lab.getByRole('button',{name:'Reset',exact:true}).click();
 await lab.getByRole('button',{name:'Set the first outcome to 3294',exact:true}).click();
 await lab.locator('input[type=radio][value=higher]').check();
 await lab.locator('.ts-numeric-guess input').fill('1162');
 await lab.getByRole('button',{name:'Issue the forecast',exact:true}).click();
 assert.equal(await lab.getByRole('button',{name:'Show the saved ridge evidence',exact:true}).count(),1);
 assert.equal(await lab.locator('.ts-table').first().locator('tbody').innerText(),forecasts);
 await lab.getByRole('button',{name:/Reveal.*outcomes/}).click();
 assert.match(await lab.innerText(),/1185\.428571/);
 report.cases.push('Future replacement changes MAE by1000/7 but leaves issued forecasts identical; reset closes saved-ridge gate');
 const donor=page.locator('.ts-investigation').first();
 await donor.getByLabel(/^history 1$/).fill('98');
 await donor.getByLabel(/^history 2$/).focus();
 assert.equal(await donor.getByRole('button',{name:'Add 5 to every value',exact:true}).isDisabled(),true);
 await donor.getByLabel(/^history 1$/).fill('95');
 await donor.getByLabel(/^history 2$/).focus();
 await donor.getByRole('button',{name:'Add 5 to every value',exact:true}).click();
 assert.equal(await donor.getByLabel(/^history 1$/).inputValue(),'100');
 assert.equal(await donor.getByLabel(/^history 2$/).inputValue(),'25');
 assert.equal(await donor.getByLabel(/^outcome h1$/).inputValue(),'17');
 report.cases.push('Uniform shift refuses98 and admits95 without silently clipping any input');
 await donor.getByRole('button',{name:'Reset',exact:true}).click();
 await donor.getByRole('button',{name:'Ask for eight horizons',exact:true}).click();
 await donor.locator('input[type=radio][value=source-5]').check();
 await donor.locator('.ts-numeric-guess input').fill('12');
 await donor.getByRole('button',{name:'Apply and reveal the donor',exact:true}).click();
 assert.match(await donor.locator('.ts-verdict').innerText(),/matches/);
 assert.match(await donor.innerText(),/4 of 8/);
 report.cases.push('Eight-horizon wrap copies observed position5 and scores only four supplied outcomes');
 const eligibility=page.locator('.ts-investigation').nth(1);
 await eligibility.getByRole('button',{name:'Cutoff 10, h 3, delay 2',exact:true}).click();
 await eligibility.getByLabel(/none of them qualify/).check();
 await eligibility.getByRole('button',{name:'Record this set and check it',exact:true}).click();
 assert.match(await eligibility.locator('.ts-verdict').innerText(),/matches: no offered origin qualifies/);
 report.cases.push('Empty eligible set is expressible and correctly graded');
 for(const width of [1366,390,320]) {
  await page.setViewportSize({width,height:900});
  const figs=page.locator('.ts-figure svg.ts-diagram');
  assert.equal(await figs.count(),6);
  for(let i=0;i<6;i++) await capture(figs.nth(i),`figure-diagram-${i+1}-${width}`);
  await capture(donor,`donor-eight-${width}`);
  await capture(lab,`counterfactual-${width}`);
  const layout=await page.evaluate(()=>({width:innerWidth,document:document.documentElement.scrollWidth,errors:document.querySelectorAll('.katex-error').length,radicals:[...document.querySelectorAll('.ts-lesson .katex .sqrt svg')].filter(e=>e.getBoundingClientRect().width).map(e=>e.getBoundingClientRect().height)}));
  assert.ok(layout.document<=width+1); assert.equal(layout.errors,0); assert.ok(layout.radicals.every(h=>h>=1));
  report.cases.push({name:`Reading geometry at${width}`,layout});
 }
 const compare=page.locator('.ts-figure').nth(4);
 await page.setViewportSize({width:1366,height:900});
 await compare.getByRole('button',{name:'development: choose',exact:true}).click();
 await capture(compare.locator('svg.ts-diagram'),'development-desktop');
 const advance=page.locator('.ts-figure').nth(3);
 await advance.getByRole('button',{name:'sliding, last 90 rows',exact:true}).click();
 await advance.getByRole('button',{name:'h = 7',exact:true}).click();
 await advance.getByRole('button',{name:'Advance the origin',exact:true}).click();
 await advance.getByRole('button',{name:'Advance the origin',exact:true}).click();
 await capture(advance,'sliding-h7-third-origin');
 await donor.getByRole('button',{name:'Reset',exact:true}).focus();
 await page.keyboard.press('Enter');
 assert.equal(await donor.locator('.ts-verdict').count(),0);
 assert.equal(await donor.getByLabel(/^horizons requested/).inputValue(),'4');
 report.cases.push('Keyboard Enter resets the committed donor state and dependent horizon');
 assert.deepEqual(errors,[]);
 report.sources=Object.fromEntries(['src/learn/data/topics/time-series-validation-forecasting-baselines.jsx','src/learn/components/lesson-labs/TimeSeriesLabs.jsx','src/learn/components/lesson-labs/TimeSeriesFigures.jsx','src/learn/components/lesson-labs/TimeSeriesShared.jsx','src/learn/components/lesson-labs/timeseries-labs.css','src/learn/data/timeseries-models.js'].map(p=>[p,hash(p)]));
 report.passed=true;
 }finally{await browser.close();report.completedAt=new Date().toISOString();save();}
 console.log(JSON.stringify({passed:report.passed,cases:report.cases.length,screenshots:report.screenshots.length,output}));
})().catch(e=>{report.error=e.stack;save();console.error(e);process.exitCode=1;});

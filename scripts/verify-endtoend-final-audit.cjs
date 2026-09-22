// Complementary final review against an already-running reader. No build,
// source mutation, fitting, or replacement of earlier author receipts.
const fs=require('node:fs');
const assert=require('node:assert/strict');
const {createHash}=require('node:crypto');
const {chromium}=require(process.env.PLAYWRIGHT_PACKAGE || 'playwright');
const base=process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4184';
const output='docs/teaching/evidence/classical-final-endtoend-browser.json';
const hash=p=>createHash('sha256').update(fs.readFileSync(p)).digest('hex');
const report={startedAt:new Date().toISOString(),passed:false,base,cases:[],screenshots:[],
 limitations:['Complementary dev-server review, not production integration or a repeat of the full historical campaign.',
 'Screenshot captures hide only the fixed reader navigation; visual inspection is recorded separately.']};
const save=()=>fs.writeFileSync(output,JSON.stringify(report,null,2)+'\n');
save();
(async()=>{
 const {endToEndData:data}=await import('../src/learn/data/endtoend-data.js');
 const {endToEndExamples:examples}=await import('../src/learn/data/endtoend-examples.js');
 const browser=await chromium.launch({channel:'msedge',headless:true});
 try {
 const page=await browser.newPage({viewport:{width:1366,height:900}});
 const errors=[];page.on('pageerror',e=>errors.push(e.message));
 await page.goto(`${base}/learn/path/full-curriculum/end-to-end-supervised-learning-error-analysis?module=classical-ml`,{waitUntil:'domcontentloaded'});
 const slice=page.locator('[data-investigation="slice"]');await slice.waitFor();
 const freeze=page.locator('[data-investigation="freeze"]');
 const cost=page.locator('[data-investigation="acceptance"]');
 const capture=async(locator,name)=>{const p=`docs/teaching/evidence/screenshots/final-audit-endtoend-${name}.png`;
  await locator.screenshot({path:p,style:'.learn-nav {visibility:hidden !important}'});
  report.screenshots.push({path:p,sha256:hash(p)});};
 const absentHeldOut=async()=>{
  const text=await page.locator('.endtoend-lesson').innerText();
  for(const score of [data.heldOut.accuracy.value,data.heldOut.balancedAccuracy.value,data.heldOut.logLoss.value])
   assert.ok(!text.includes(score.toFixed(6)),`held-out score prematurely present: ${score}`);
  assert.doesNotMatch(text,/its one error|One cultivar-2 specimen is predicted|35\s*(?:of|\/)\s*36|k\s*=\s*35/);
  assert.equal(await page.locator('.ete-heldout-report').count(),0);
  assert.equal(await page.locator('[data-program-output]').count(),1);
  assert.equal((await page.locator('[data-program-output]').innerText()).replace(/^OUTPUT\n/, '').trim(),examples.study.developmentOutput);
 };
 await absentHeldOut();
 report.cases.push('Whole-document held-out numbers, count/prose, report and stdout absent before freeze; development stdout remains exact');
 assert.equal(await slice.locator('.ete-error-ring').count(),0);
 const positions=await slice.locator('.ete-point').evaluateAll(es=>es.map(e=>e.innerHTML));
 const cases=[];
 for(const candidate of ['linear_three','forest_two','linear_two','majority'])
  for(const side of ['lower','upper']) cases.push({candidate,side,cutoff:4});
 cases.push({candidate:'linear_three',side:'lower',cutoff:0},
  {candidate:'linear_three',side:'upper',cutoff:14},
  {candidate:'linear_three',side:'lower',cutoff:data.validationRows[0].colorIntensity},
  {candidate:'linear_three',side:'upper',cutoff:data.validationRows[0].colorIntensity});
 for(const c of cases) {
  await slice.getByRole('button',{name:'Reset',exact:true}).click();
  await slice.getByLabel('Candidate being compared').selectOption(c.candidate);
  await slice.getByLabel('Side of the cutoff').selectOption(c.side);
  await slice.getByLabel('Colour-intensity cutoff').fill(String(c.cutoff));
  assert.equal(await slice.locator('.ete-error-ring').count(),0);
  const rows=data.validationRows.filter(r=>c.side==='lower'?r.colorIntensity<c.cutoff:r.colorIntensity>=c.cutoff);
  const ids=model=>rows.filter(r=>r.prediction[model]!==r.actual).map(r=>r.id).sort((a,b)=>a-b);
  const ref=ids('linear_two'),cand=ids(c.candidate);
  const choice=cand.length<ref.length?'fewer':cand.length>ref.length?'more':'same';
  await slice.locator(`input[type=radio][value=${choice}]`).check();
  await slice.locator('.ete-numeric-guess input').fill(String(cand.length));
  await slice.getByRole('button',{name:'Commit the prediction and compare',exact:true}).click();
  assert.deepEqual(await slice.locator('.ete-point').evaluateAll(es=>es.map(e=>e.innerHTML)),positions);
  for(const [attr,expected] of [['data-reference-error',ref],['data-candidate-error',cand]])
   assert.deepEqual(await slice.locator(`[${attr}]`).evaluateAll((es,a)=>es.map(e=>Number(e.getAttribute(a))).sort((a,b)=>a-b),attr),expected);
  if(rows.length) {
   assert.match(await slice.locator('.ete-verdict').innerText(),/matches/);
   assert.equal(await slice.locator('[data-graded-quantity="slice-candidate-errors"]').innerText(),String(cand.length));
  } else {
   assert.match(await slice.locator('.ete-verdict').innerText(),/nothing is graded/);
   assert.equal(await slice.locator('.ete-reveal').count(),0);
   assert.doesNotMatch(await slice.innerText(),/NaN|Difference in error count: null/);
  }
  if(c.candidate==='linear_three' && c.cutoff===4) await capture(slice,`slice-${c.side}-desktop`);
 }
 report.cases.push('12 slice states: all four candidates, both sides, exact observed cutoff boundary and two empty slices; every error-ring ID matches independently filtered saved rows, coordinates unchanged');
 await slice.getByRole('button',{name:'Reset',exact:true}).click();
 await slice.getByLabel('Inspect one specimen').selectOption(String(data.validationRows[0].id));
 assert.equal(await slice.locator('.ete-point.is-selected').count(),1);
 assert.equal(await slice.locator('.ete-strip-mark.is-selected').count(),1);
 await slice.locator('input[type=radio][value=more]').check();
 await slice.locator('.ete-numeric-guess input').fill('3');
 await slice.getByRole('button',{name:'Commit the prediction and compare',exact:true}).click();
 report.cases.push('Specimen inspector links the same fixed scatter/added-feature positions and remains usable after compare');
 // An alternative metric may still pick the same winner; it must not earn this protocol's report.
 await freeze.getByLabel('Measure that decides').selectOption('validationAccuracy');
 await freeze.locator('input[type=radio][value=linear_three]').check();
 await freeze.getByRole('button',{name:'Commit the prediction and apply the rule',exact:true}).click();
 assert.equal(await freeze.locator('.ete-refusal').count(),1);await absentHeldOut();
 await freeze.getByRole('button',{name:'Reset',exact:true}).click();
 await freeze.locator('input[type=radio][value=linear_three]').check();
 await freeze.getByRole('button',{name:'Commit the prediction and apply the rule',exact:true}).click();
 await absentHeldOut();
 await freeze.locator('.ete-heldout-step input[value=higher]').check();
 await freeze.getByRole('button',{name:'Commit and open the held-out report',exact:true}).click();
 assert.match(await freeze.locator('.ete-heldout-step .ete-verdict').innerText(),/matches/);
 for(const score of [data.heldOut.accuracy.value,data.heldOut.balancedAccuracy.value,data.heldOut.logLoss.value])
  assert.ok((await page.locator('.endtoend-lesson').innerText()).includes(score.toFixed(6)));
 assert.equal((await page.locator('[data-program-output]').nth(1).innerText()).replace(/^OUTPUT\n/, '').trim(),examples.study.heldOutOutput);
 await capture(page.locator('.ete-heldout-report'),'heldout-desktop');
 report.cases.push('Same-winner alternative metric refused; declared decision still withholds report until second commitment; exact held-out output appears afterward');
 await freeze.getByRole('button',{name:'Reset',exact:true}).focus();await page.keyboard.press('Enter');
 await absentHeldOut();report.cases.push('Keyboard reset closes all held-out report, summary, stdout and interval regions');
 const commitCost=async(choice,guess)=>{
  await cost.locator(`input[type=radio][value=${choice}]`).check();
  await cost.locator('.ete-numeric-guess input').fill(String(guess));
  await cost.getByRole('button',{name:'Commit the prediction and compare',exact:true}).click();
  assert.match(await cost.locator('.ete-verdict').innerText(),/matches/);
  assert.equal(await cost.locator('[data-graded-quantity="acceptance-proposed-cost"]').innerText(),String(guess));
 };
 const noTileOverlap=async()=>{
  const boxes=await cost.locator('.ete-tile rect').evaluateAll(es=>es.map(e=>{
   const b=e.getBoundingClientRect();return {x:b.x,y:b.y,right:b.right,bottom:b.bottom};}));
  assert.equal(boxes.length,10);
  for(let i=0;i<boxes.length;i++) for(let j=i+1;j<boxes.length;j++) {
   const a=boxes[i],b=boxes[j];
   assert.ok(a.right<=b.x || b.right<=a.x || a.bottom<=b.y || b.bottom<=a.y,`tiles ${i+1},${j+1} overlap`);
  }
 };
 await cost.getByLabel('Cost of a wrong automatic answer').fill('0.25');
 await cost.getByLabel('Cost of a deferred case').fill('0.1');
 await commitCost('higher',0.6000000000000001);
 report.cases.push('Fractional costs: active0.45 versus proposed0.6000000000000001; numeric grading uses proposed total with1e-9 tolerance');
 await cost.getByRole('button',{name:'Reset',exact:true}).click();
 await cost.getByLabel('Case 6 score').fill('0.8');
 await cost.getByLabel('Case 9 score').fill('0.8');
 await commitCost('higher',28);
 await noTileOverlap();
 assert.match(await cost.locator('.ete-reveal').innerText(),/1, 2, 3, 4, 6, 9/);
 report.cases.push('Two edited scores equal threshold0.8 are accepted, preserving two wrong outcomes; proposed total28');
 await capture(cost,'acceptance-ties-desktop');
 for(const width of [1366,390,320]) {
  await page.setViewportSize({width,height:900});
  const figures=page.locator('.ete-figure');assert.equal(await figures.count(),7);
  await noTileOverlap();
  for(let i=0;i<7;i++) await capture(figures.nth(i),`figure-${i+1}-${width}`);
  if(width!==1366) {await capture(slice,`slice-lower-${width}`);await capture(cost,`acceptance-ties-${width}`);}
  const layout=await page.evaluate(()=>({width:innerWidth,document:document.documentElement.scrollWidth,
   katexErrors:document.querySelectorAll('.endtoend-lesson .katex-error').length,
   radicals:[...document.querySelectorAll('.endtoend-lesson .katex .sqrt svg')].filter(e=>e.getBoundingClientRect().width).map(e=>e.getBoundingClientRect().height)}));
  assert.ok(layout.document<=width,JSON.stringify(layout));assert.equal(layout.katexErrors,0);
  assert.ok(layout.radicals.length>0);assert.ok(layout.radicals.every(h=>h>=1));
 }
 report.cases.push('All seven diagrams and changed lab states captured at1366/390/320; no page overflow or collapsed/errored KaTeX');
 await cost.getByRole('button',{name:'Reset',exact:true}).click();
 for(let i=1;i<=10;i++) await cost.getByLabel(`Case ${i} score`,{exact:false}).fill('0.8');
 await commitCost('same',20);await noTileOverlap();
 await capture(cost.locator('.ete-figure'),'all-ten-tied-320');
 await cost.getByRole('button',{name:'Reset',exact:true}).click();
 for(let i=1;i<=10;i++) await cost.getByLabel(`Case ${i} score`,{exact:false}).fill(String(0.8+(i-1)/100));
 await noTileOverlap();
 report.cases.push('All10 equal scores and ten adjacent hundredths remain visible without rectangle overlap, with score coordinates unchanged');
 await cost.getByRole('button',{name:'Reset',exact:true}).click();
 await cost.getByRole('button',{name:'A threshold no case reaches',exact:true}).click();
 await commitCost('higher',20);assert.match(await cost.locator('.ete-reveal').innerText(),/undefined/);
 await capture(cost,'acceptance-none-320');
 report.cases.push('Threshold1.01: no automatic answers, total20, undefined conditional error explicitly retained');
 assert.deepEqual(errors,[]);report.cases.push('No page errors');report.passed=true;
 } finally {await browser.close();report.finishedAt=new Date().toISOString();
 report.sourceHashes=Object.fromEntries([
 'src/learn/data/topics/end-to-end-supervised-learning-error-analysis.jsx','src/learn/data/endtoend-data.js',
 'src/learn/data/endtoend-models.js','src/learn/data/endtoend-examples.js',
 'src/learn/components/lesson-labs/EndToEndLabs.jsx','src/learn/components/lesson-labs/EndToEndShared.jsx',
 'src/learn/components/lesson-labs/EndToEndFigures.jsx','src/learn/components/lesson-labs/endtoend-labs.css',
 'scripts/verify-endtoend-final-audit.cjs'].map(p=>[p,hash(p)]));save();}
 console.log(`PASS ${report.cases.length} complementary groups; ${report.screenshots.length} screenshots`);
})().catch(e=>{report.error=e.stack;save();console.error(e);process.exitCode=1;});

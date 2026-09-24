// Complementary supported-state regressions for the final PAC/calibration audit.
// Read-only against the shared preview; writes only this audit's evidence/captures.
const fs = require('node:fs');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const {pathToFileURL} = require('node:url');
const path = require('node:path');
const {chromium} = require(process.env.PLAYWRIGHT_PACKAGE || 'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const reportPath='docs/teaching/evidence/classical-final-audit-pac-calibration.json';
const hash=file=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const base=process.env.LEARNING_BASE_URL || 'http://127.0.0.1:4184';
const checks=[]; const captures=[];
const record=(name,details)=>checks.push({name,passed:true,details});
async function capture(locator,name){const file=`docs/teaching/evidence/screenshots/classical-final-${name}.png`;await locator.screenshot({path:file,style:'.learn-nav { visibility: hidden !important; }'});captures.push({file,sha256:hash(file)});}
async function pac(page){
 const m=await import(pathToFileURL(path.resolve('src/learn/data/pac-models.js')));
 const duplicate=m.manualIntervalRequest({points:[.1,.1,.6],labels:[0,0,1]});
 assert.equal(duplicate.feasible,true);assert.deepEqual(duplicate.predictions.map(p=>p.predicted),[0,0,1]);
 assert.equal(m.manualIntervalRequest({points:[.1,.1,.6],labels:[0,1,1]}).feasible,false);
 record('Repeated manual inputs preserve consistent labels and refute conflicting ones');
 for(const [right,expected] of [[.301,true],[.3,true],[.299,false]]){
  const r=m.intervalExperiment({points:[.2,right],target:[.1,.4],epsilon:.2});
  assert.equal(r.meetsTarget,expected);record(`Risk boundary at fitted [.2,${right}]`,{risk:r.risk,epsilon:.2,meetsTarget:r.meetsTarget});
 }
 assert.equal(m.intervalExperiment({points:[.12],target:[.1,.15],epsilon:.9}).strips.length,0);
 record('Target narrower than epsilon has no invalid coverage-strip geometry');
 await page.goto(`${base}/learn/path/full-curriculum/pac-learning-vc-dimension?module=classical-ml`,{waitUntil:'domcontentloaded'});
 await page.locator('.pac-investigation').last().waitFor();
 const world=page.locator('.pac-investigation').nth(0), witness=page.locator('.pac-investigation').nth(1), lab=page.locator('.pac-investigation').nth(2);
 const reset=async locator=>locator.getByRole('button',{name:'Reset',exact:true}).click();
 const commit=async(choice,value)=>{await lab.getByRole('radio',{name:choice,exact:true}).check();if(value!==undefined)await lab.locator('.pac-numeric-guess input').fill(String(value));await lab.getByRole('button',{name:'Apply and check',exact:true}).click();};
 await witness.getByRole('button',{name:'Translate everything by +2',exact:true}).click();
 await witness.getByRole('button',{name:'Translate everything by +2',exact:true}).click();
 assert.equal(await witness.getByRole('button',{name:'Translate everything by +2',exact:true}).isDisabled(),true);
 await witness.getByLabel('C coordinate',{exact:true}).fill('5');await witness.getByRole('button',{name:'Add a point',exact:true}).click();
 assert.equal(await witness.locator('input[type=number]').evaluateAll(nodes=>nodes.some(n=>n.validity.rangeOverflow||n.validity.rangeUnderflow)),false);
 record('Witness translation and Add preserve coordinate range');
 await lab.getByLabel('RNG seed',{exact:true}).fill('99');await lab.getByLabel('how many points to draw',{exact:true}).fill('24');await reset(lab);
 assert.equal(await lab.getByLabel('RNG seed',{exact:true}).inputValue(),'7');assert.equal(await lab.getByLabel('how many points to draw',{exact:true}).inputValue(),'8');
 record('Reset restores seed and draw size');
 await lab.locator('select').selectOption('manual');await lab.getByLabel('observation 2',{exact:true}).fill('.1');await commit('Yes — some interval fits them');
 assert.match(await lab.locator('.pac-verdict').innerText(),/Your prediction matches/);
 record('Repeated input accepted in the actual manual-label UI');
 await lab.locator('select').selectOption('generated');await commit('Establish a new generated-label baseline','.1');
 assert.match(await lab.locator('.pac-verdict').innerText(),/new generated-label baseline/);
 assert.doesNotMatch(await lab.locator('.pac-verdict').innerText(),/does not move|in both cases/);
 await capture(lab,'pac-baseline-desktop');record('Manual-to-generated transition establishes a baseline without inventing a direction');
 await reset(lab);await lab.getByLabel('how many points to draw',{exact:true}).fill('24');await lab.getByRole('button',{name:'Draw a fresh independent sample',exact:true}).click();
 await commit('It falls',m.intervalExperiment({points:m.seededUniformSample(7,24),target:[.3,.7],epsilon:.1}).risk);
 assert.equal(await lab.getByRole('button',{name:'Add two negatives far outside: .01 and .99',exact:true}).isDisabled(),true);
 record('Interval presets respect the 24-observation cap');
 await reset(lab);await lab.getByLabel('target left endpoint',{exact:true}).fill('.1');await lab.getByLabel('target right endpoint',{exact:true}).fill('.15');await lab.getByLabel('ε, the risk target',{exact:true}).fill('.9');await commit('It falls','.05');
 assert.equal(await lab.locator('.pac-band.is-coverage').count(),0);assert.match(await lab.innerText(),/every sample meets the risk target/);
 record('Narrow-target UI explains the trivial success event instead of drawing strips outside the target');
 await reset(lab);await lab.getByLabel('target left endpoint',{exact:true}).fill('0');await lab.getByLabel('target right endpoint',{exact:true}).fill('1');await commit('It rises','.2');
 assert.equal(await lab.getByRole('button',{name:'Add two negatives far outside: .01 and .99',exact:true}).count(),0);
 await lab.getByRole('button',{name:'Add observations at .01 and .99 (labels follow the current target)',exact:true}).click();await commit('It falls','.02');
 assert.match(await lab.locator('.pac-verdict').innerText(),/Your prediction matches/);record('Changed targets relabel fixed-coordinate presets truthfully');
 for(const width of [1366,390,320]){await page.setViewportSize({width,height:900});await capture(lab,`pac-changed-target-${width}`);assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),true);}
 await reset(world);for(let i=0;i<22;i++)await world.getByRole('button',{name:'add an observation at 0',exact:true}).click();
 assert.equal(await world.getByRole('button',{name:'add an observation at 0',exact:true}).isDisabled(),true);assert.equal(await world.getByRole('button',{name:'Add input 2, the first positive',exact:true}).isDisabled(),true);
 record('Finite-world buttons respect their own 24-observation cap');
}
async function calibration(page){
 const m=await import(pathToFileURL(path.resolve('src/learn/data/calibration-models.js')));
 for(const label of [0,1]){
  const scores=[-2,-1,0,1,2,3],labels=scores.map(()=>label),t=m.plattTargets(labels)[0],b=Math.log(t/(1-t));
  const residual=m.sigmoid(b)-t;
  assert(Math.abs(residual)<1e-14);assert(Math.abs(residual*scores.reduce((s,x)=>s+x,0)/6)<1e-14);
  const curvature=t*(1-t),mean=scores.reduce((s,x)=>s+x,0)/6,second=scores.reduce((s,x)=>s+x*x,0)/6;
  assert(curvature*curvature*(second-mean*mean)>0);
  assert.match(m.fitSigmoid(scores,labels).because,/evidence policy/);
  record(`One-class ${label} smoothed optimum exists; lab refusal is an explicit evidence policy`,{target:t,a:0,b});
 }
 await page.goto(`${base}/learn/path/full-curriculum/calibration-conformal-prediction?module=classical-ml`,{waitUntil:'domcontentloaded'});
 await page.locator('.cal-investigation').last().waitFor();
 await capture(page.locator('.cal-figure-block').nth(9),'calibration-probability-layers-desktop');
 const labs=page.locator('.cal-investigation'),r=labs.nth(0),mono=labs.nth(1),rank=labs.nth(2),interval=labs.nth(3);
 const reset=async l=>l.getByRole('button',{name:'Reset',exact:true}).click();
 const commit=async(l,choice,value)=>{await l.getByRole('radio',{name:choice,exact:true}).check();if(value!==undefined)await l.locator('.cal-numeric-guess input').fill(String(value));await l.getByRole('button',{name:'Apply and check',exact:true}).click();await l.locator('.cal-verdict').waitFor();};
 const matches=async l=>assert.match(await l.locator('.cal-verdict').innerText(),/Your prediction matches/);
 await r.getByRole('button',{name:'Make every outcome 1 — AUC then has no value',exact:true}).click();await commit(r,'higher than the applied state');
 await r.getByRole('button',{name:'Merge the two bins into one',exact:true}).click();await commit(r,'unchanged — unchanged means within 10⁻¹²','.5');await matches(r);
 assert.match(await r.locator('.cal-verdict').innerText(),/zero only when those averages agree/);assert.doesNotMatch(await r.locator('.cal-verdict').innerText(),/zero whatever/);
 await capture(r.locator('.cal-prediction'),'calibration-one-bin-feedback');record('One occupied bin correctly retains nonzero ECE .5');
 await r.getByRole('button',{name:'Original ten cards with forecasts .4 and .6',exact:true}).click();
 assert.equal(await r.locator('.cal-cards input[type=number]').count(),10);
 assert.deepEqual(await r.locator('.cal-cards input[type=number]').evaluateAll(es=>es.map(e=>Number(e.value))),[.4,.4,.4,.4,.4,.6,.6,.6,.6,.6]);
 record('Named original-card preset resets the complete fixture after changed outcomes');
 const scoreInputs=mono.locator('.cal-controls input[type=number]');assert.equal(await scoreInputs.count(),6);
 for(let i=0;i<6;i++)await scoreInputs.nth(i).fill('0');
 await commit(mono,'no merge is required — the blocks already increase');await matches(mono);
 assert.match(await mono.locator('.cal-graded').innerText(),/one distinct score/);record('All tied input scores produce one finished block without interpolating a nonexistent second knot');
 await mono.getByRole('button',{name:'Make every label 1 — a legitimate constant fit',exact:true}).click();await commit(mono,'no merge is required — the blocks already increase');
 await mono.getByRole('button',{name:'Overlay the two-parameter sigmoid fit',exact:true}).click();assert.match(await mono.locator('.cal-graded').innerText(),/evidence policy/);
 await capture(mono.locator('.cal-graded'),'calibration-single-knot-one-class');record('One-class overlay refusal matches corrected lesson explanation');
 for(let i=0;i<6;i++)await rank.getByRole('button',{name:'Remove the last card',exact:true}).click();
 for(let i=0;i<3;i++)await rank.getByLabel(`calibration card ${i+1}`,{exact:true}).fill(String(i*.001));
 await rank.locator('.cal-controls input[type=number]').first().fill('.5');
 await commit(rank,'no class passes — the set is empty',2);await matches(rank);
 await rank.getByRole('button',{name:'Lower the smallest card to 0 — an exact null',exact:true}).click();
 await commit(rank,'no class passes — the set is empty',2);await matches(rank);
 assert.match(await rank.locator('.cal-verdict').innerText(),/which is 0.001/);record('Smallest-card null remains null when the applied minimum is already zero');
 await reset(interval);await interval.locator('select').first().selectOption('absolute');
 assert.match(await interval.locator('p.cal-caption').filter({hasText:'You are predicting against'}).innerText(),/width is 12/);
 record('Baseline caption follows the selected score branch');
 for(const branch of ['absolute','normalized','cqr']){
  await reset(interval);await interval.locator('select').first().selectOption(branch);
  await interval.getByLabel('alpha',{exact:true}).fill('.01');await commit(interval,'the interval becomes the whole line — unbounded width',99);await matches(interval);
  const text=await interval.locator('.cal-verdict').innerText();assert.match(text,/Your prediction matches the category/);assert.match(text,/no finite guess matches the whole line/);assert.doesNotMatch(text,/width from .*no width/);
  assert.match(await interval.locator('.cal-graded').innerText(),/unbounded/);
  record(`${branch}: alpha .01 produces the whole line with unbounded width`);
  if(branch==='cqr')for(const width of [1366,390,320]){await page.setViewportSize({width,height:900});await capture(interval.locator('.cal-prediction'),`calibration-cqr-unbounded-${width}`);assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),true);}
  await interval.getByLabel('alpha',{exact:true}).fill(branch==='cqr'?'.4':'.2');await commit(interval,'the interval gets narrower');await matches(interval);
  assert.match(await interval.locator('.cal-verdict').innerText(),/whole line becomes a finite interval/);record(`${branch}: returning to an available rank restores a finite width`);
 }
 await interval.getByRole('button',{name:'CQR base interval 10 to 12 with the original shrinking scores',exact:true}).click();
 await commit(interval,'the endpoints cross, so the set is empty and has no width');await matches(interval);assert.match(await interval.locator('.cal-graded').innerText(),/empty — the shrunken endpoints cross/);
 record('Crossed CQR endpoints remain a distinct empty set');
 await interval.getByLabel('initial upper endpoint U(x)',{exact:true}).fill('20');await commit(interval,'it has a finite width now; the applied state was an empty set',6);await matches(interval);
 assert.doesNotMatch(await interval.locator('.cal-verdict').innerText(),/half-width is the threshold multiplied/);
 await capture(interval.locator('.cal-graded'),'calibration-cqr-restored-320');record('Returning from an empty CQR set produces finite width 6 and CQR-specific feedback');
 await page.setViewportSize({width:1366,height:900});await reset(interval);
 await interval.locator('.cal-controls input[type=number]').nth(1).fill('3');await commit(interval,'the interval gets wider');
 await interval.getByRole('button',{name:'Change the last two residuals to 12 and 16',exact:true}).click();
 const residuals=await interval.getByLabel('residual, response units',{exact:true}).evaluateAll(es=>es.map(e=>Number(e.value)));
 assert.deepEqual(residuals,[3,1,1.5,2,3,4,5,12,16]);record('Last-two-residual preset preserves prior changes to the other residuals');
}
(async()=>{const topic=process.argv[2]||'pac';const old=fs.existsSync(reportPath)?JSON.parse(fs.readFileSync(reportPath,'utf8')):{topics:{}};
 const browser=await chromium.launch({channel:'msedge',headless:true});const page=await browser.newPage({viewport:{width:1366,height:900}});page.setDefaultTimeout(12000);const errors=[];page.on('pageerror',e=>errors.push(e.message));
 try{if(topic==='pac')await pac(page);else if(topic==='calibration')await calibration(page);else throw Error(`Unknown topic ${topic}`);assert.deepEqual(errors,[]);record('No page errors throughout supported-state checks');
 const files=topic==='pac'?['src/learn/data/topics/pac-learning-vc-dimension.jsx','src/learn/data/pac-models.js','src/learn/data/pac-data.js','src/learn/data/pac-examples.js','src/learn/components/lesson-labs/PacFigures.jsx','src/learn/components/lesson-labs/PacLabs.jsx','src/learn/components/lesson-labs/PacShared.jsx','src/learn/components/lesson-labs/pac-labs.css']:['src/learn/data/topics/calibration-conformal-prediction.jsx','src/learn/data/calibration-models.js','src/learn/data/calibration-data.js','src/learn/data/calibration-examples.js','src/learn/components/lesson-labs/CalibrationFigures.jsx','src/learn/components/lesson-labs/CalibrationLabs.jsx','src/learn/components/lesson-labs/CalibrationShared.jsx','src/learn/components/lesson-labs/calibration-labs.css'];
 old.topics[topic]={checkedAt:new Date().toISOString(),passed:true,checks,captures,sourceHashes:Object.fromEntries(files.map(file=>[file,hash(file)])),environment:{browser:'Edge Chromium',base,viewports:[1366,390,320]},limits:'Targeted checks on the shared development server. Root owns final production integration. Existing source-matching native and unchanged figure evidence are reused.'};
 fs.writeFileSync(reportPath,JSON.stringify(old,null,2)+'\n');console.log(`PASS ${topic}: ${checks.length} targeted checks; ${captures.length} current screenshots.`);
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1});

const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs = require('node:fs');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const directory = 'scratch/linked-traversal-extension-browser';
fs.mkdirSync(directory, { recursive: true });
const hash = path => crypto.createHash('sha256').update(fs.readFileSync(path)).digest('hex');

(async () => {
  const { linkedExamples } = await import('../src/learn/data/linked-foundations-examples.js');
  const { linkedTraversalExamples } = await import('../src/learn/data/linked-traversal-examples.js');
  const { monotonicStackExamples } = await import('../src/learn/data/monotonic-stack-examples.js');
  const native = JSON.parse(fs.readFileSync('scratch/linked-traversal-extension-verification/results.json','utf8'));
  const productionSources = native.productionSources.map(({path}) => ({path,sha256:hash(path)}));
  const browser = await chromium.launch({ channel:'msedge', headless:true });
  const results = [], images = [];
  try {
    for (const width of [1440,390,320]) {
      const page = await browser.newPage({ viewport:{width,height:1080}, reducedMotion:'reduce' });
      await page.routeWebSocket('**', socket => socket.close());
      const errors = [];
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => { if (['error','warning'].includes(message.type()) && !message.text().startsWith('[vite]')) errors.push(message.text()); });
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/linked-lists-stacks-queues?module=data-structures-algorithms',{waitUntil:'domcontentloaded',timeout:60000});
      const lesson = page.locator('.lesson-pilot').first();
      await lesson.waitFor({timeout:60000});
      await page.evaluate(() => document.fonts.ready);
      assert(await page.evaluate(() => document.fonts.check('16px "Space Grotesk"')));
      const capture = async (locator,name) => {
        await locator.first().evaluate(node => scrollTo({top:scrollY+node.getBoundingClientRect().top-112,behavior:'instant'}));
        await page.waitForTimeout(80);
        const path = `${directory}/${name}-${width}.png`;
        await page.screenshot({path});
        images.push({path,sha256:hash(path),opened:false});
      };
      const range = async (locator,value) => { await locator.focus(); await page.keyboard.press('Home'); for(let i=0;i<value;i++) await page.keyboard.press('ArrowRight'); };
      const finish = async button => { let count=0; while(await button.isEnabled()){ await button.click(); if(++count>100)throw Error('Unbounded trace');} return count; };
      let operatedStates=0;
      assert.equal(await lesson.locator('h2').count(),14);
      assert.deepEqual(await lesson.locator('nav a').evaluateAll(nodes => nodes.map(node=>node.hash).filter(hash=>!document.getElementById(hash.slice(1)))),[]);
      for(let index=0;index<13;index++) await capture(lesson.locator('h2').nth(index),`reading-${index+1}`);
      // The original mechanism components still render and accept their established input changes.
      for(const [id,label,value] of [['linked-reversal','List length','1'],['bracket-stack','Bracket input','([)]'],['circular-queue','Buffer capacity','3']]) {
        const original=lesson.locator(`[data-investigation="${id}"]`);
        await original.getByRole('combobox',{name:label,exact:true}).selectOption(value);
        assert.equal(await original.getByRole('combobox',{name:label,exact:true}).inputValue(),value);
        operatedStates++;
      }
      const cycle=lesson.locator('[data-traversal-lab="cycle"]');
      await capture(cycle,'cycle-initial');
      const nextCycle=cycle.getByRole('button',{name:'Next cycle step',exact:true});
      while(!(await cycle.locator('.traversal-state').innerText()).includes('meeting')) {await nextCycle.click();operatedStates++;}
      assert((await cycle.innerText()).includes('slow → n5; fast → n5'));
      await capture(cycle.locator('.traversal-state'),'cycle-meeting');
      operatedStates+=await finish(nextCycle);
      assert((await cycle.getByRole('status').innerText()).includes('entry n2'));
      assert((await cycle.getByRole('status').innerText()).includes('cycle length 5'));
      await capture(cycle.locator('.traversal-state'),'cycle-entry');
      await cycle.getByRole('button',{name:'Previous cycle step',exact:true}).focus();await page.keyboard.press('Enter');operatedStates++;
      await cycle.getByLabel('Cycle tail target',{exact:true}).selectOption('4');
      await cycle.getByRole('button',{name:'Show cycle result',exact:true}).click();
      assert((await cycle.getByRole('status').innerText()).includes('entry n4'));operatedStates++;
      await cycle.getByLabel('All values equal 7',{exact:true}).focus();await page.keyboard.press('Space');
      assert.equal(await cycle.getByLabel('All values equal 7',{exact:true}).isChecked(),false);
      assert((await cycle.innerText()).includes('entry n4'));operatedStates++;
      await cycle.getByRole('button',{name:'Self-loop',exact:true}).click();operatedStates+=await finish(nextCycle);
      assert((await cycle.getByRole('status').innerText()).includes('cycle length 1'));
      await capture(cycle.locator('.traversal-state'),'cycle-self-loop');
      await cycle.getByRole('button',{name:'No cycle',exact:true}).click();operatedStates+=await finish(nextCycle);
      assert((await cycle.getByRole('status').innerText()).includes('No reachable cycle'));
      await cycle.getByRole('button',{name:'Empty chain',exact:true}).click();operatedStates+=await finish(nextCycle);
      assert((await cycle.innerText()).includes('head → None'));
      await range(cycle.getByLabel('Cycle node count',{exact:true}),9);
      assert.equal(await cycle.getByLabel('Cycle tail target',{exact:true}).inputValue(),'-1');
      if(width<800){const region=cycle.getByRole('region');await region.focus();await page.keyboard.press('ArrowRight');await page.waitForTimeout(150);assert(await region.evaluate(node=>node.scrollLeft>0));}
      await cycle.getByRole('button',{name:'Reset cycle',exact:true}).click();operatedStates+=2;

      const middle=lesson.locator('[data-traversal-lab="middle"]');
      assert((await middle.getByRole('status').innerText()).includes('returns n3'));
      await middle.getByLabel('Middle policy',{exact:true}).selectOption('first');
      assert((await middle.getByRole('status').innerText()).includes('returns n2'));
      await middle.getByLabel('Show the left-heavy cut',{exact:true}).focus();await page.keyboard.press('Space');
      assert((await middle.innerText()).includes('3 and 3 original nodes'));
      assert((await middle.innerText()).includes('right head n3'));
      await capture(middle,'middle-first');
      await capture(middle.getByRole('region').last(),'middle-cut');
      if(width<800){const region=middle.getByRole('region').last();await region.focus();for(let i=0;i<8;i++)await page.keyboard.press('ArrowRight');await page.waitForTimeout(150);assert(await region.evaluate(node=>node.scrollLeft>0));await capture(region,'middle-cut-scrolled');}

      await range(middle.getByLabel('Middle chain length',{exact:true}),7);
      assert((await middle.innerText()).includes('4 and 3 original nodes'));
      assert((await middle.getByRole('status').innerText()).includes('returns n3'));
      await range(middle.getByLabel('Middle chain length',{exact:true}),0);
      assert((await middle.getByRole('status').innerText()).includes('returns None'));
      await middle.getByRole('button',{name:'Reset middle and split',exact:true}).click();operatedStates+=6;

      const greater=lesson.locator('[data-monostack-lab="greater"]');
      const nextStack=greater.getByRole('button',{name:'Next stack event',exact:true});
      while(!(await greater.getByRole('status').innerText()).includes('Index 0 is answered by 3')){await nextStack.click();operatedStates++;}
      await capture(greater.locator('.monostack-scroll').first(),'next-greater-multipop');
      await greater.getByRole('button',{name:'Finish future search',exact:true}).click();
      assert.deepEqual(await greater.locator('tbody td').allTextContents(),['3','2','1','2','1','0']);
      await capture(greater.locator('.monostack-memory'),'next-greater-result');
      const before=await greater.getByRole('status').innerText();
      await greater.getByLabel('Next-greater readings',{exact:true}).fill('1,,2');
      await greater.getByRole('button',{name:'Apply readings',exact:true}).click();
      assert(await greater.getByRole('alert').isVisible());assert.equal(await greater.getByRole('status').innerText(),before);
      await capture(greater,'invalid-readings');
      await greater.getByLabel('Next-greater readings',{exact:true}).fill('-2, 0, -2, 2');
      await greater.getByRole('button',{name:'Apply readings',exact:true}).click();
      assert.equal(await greater.getByRole('alert').count(),0);
      await greater.getByRole('button',{name:'Finish future search',exact:true}).click();
      assert.deepEqual(await greater.locator('tbody td').allTextContents(),['1','2','1','0']);
      await capture(greater.locator('.monostack-scroll').first(),'signed-readings');
      await greater.getByRole('button',{name:'Equal readings',exact:true}).click();
      await greater.getByLabel('Future comparison',{exact:true}).selectOption('inclusive');
      await greater.getByRole('button',{name:'Finish future search',exact:true}).click();
      assert.deepEqual(await greater.locator('tbody td').allTextContents(),['1','1','0']);
      await greater.getByRole('button',{name:'Many pops at once',exact:true}).click();
      await greater.getByLabel('Future comparison',{exact:true}).selectOption('strict');
      await greater.getByRole('button',{name:'Finish future search',exact:true}).click();
      assert((await greater.innerText()).includes('Pushes 5; pops 4'));
      await greater.getByRole('button',{name:'Previous stack event',exact:true}).focus();await page.keyboard.press('Enter');
      await greater.getByRole('button',{name:'Empty readings',exact:true}).click();operatedStates+=await finish(nextStack);
      assert.equal(await greater.locator('tbody td').count(),0);
      await greater.getByRole('button',{name:'Reset future search',exact:true}).click();operatedStates+=10;

      const histogram=lesson.locator('[data-monostack-lab="histogram"]');
      const nextBoundary=histogram.getByRole('button',{name:'Next boundary event',exact:true});
      assert((await histogram.innerText()).includes('Largest area: 9'));
      await capture(histogram.locator('.monostack-candidate'),'histogram-candidate');
      await capture(histogram.getByRole('region',{name:'Chosen histogram rectangle and its excluded smaller boundary bars',exact:true}),'histogram-rectangle');
      const rectangle=await histogram.locator('[data-rectangle="candidate"]').evaluate(node=>Object.fromEntries(['x','y','width','height'].map(key=>[key,Number(node.getAttribute(key))])));
      assert.deepEqual(rectangle,{x:192,y:61.25,width:156,height:108.75});
      if(width<800){const region=histogram.getByRole('region',{name:'Chosen histogram rectangle and its excluded smaller boundary bars',exact:true});await region.focus();for(let i=0;i<8;i++)await page.keyboard.press('ArrowRight');await page.waitForTimeout(150);assert(await region.evaluate(node=>node.scrollLeft>0));await capture(region,'histogram-scrolled');}

      while(!(await histogram.getByRole('status').innerText()).includes('Discard candidate 0')){await nextBoundary.click();operatedStates++;}
      await capture(histogram.locator('.monostack-memory'),'boundary-equal-pop');
      await histogram.getByRole('button',{name:'Finish boundary scan',exact:true}).click();
      assert.deepEqual(await histogram.locator('table').first().locator('tbody td').allTextContents(),['-1','-1','-1','2','2','2']);
      await histogram.getByLabel('Smaller boundary direction',{exact:true}).selectOption('right');operatedStates+=await finish(nextBoundary);
      assert.deepEqual(await histogram.locator('table').first().locator('tbody td').allTextContents(),['2','2','6','5','5','6']);
      await histogram.getByRole('button',{name:'Previous boundary event',exact:true}).click();
      await histogram.getByLabel('Limiting histogram bar',{exact:true}).selectOption('3');
      assert((await histogram.locator('.monostack-candidate').innerText()).includes('area = 4 × 2 = 8'));
      const oldCandidate=await histogram.locator('.monostack-candidate').innerText();
      await histogram.getByLabel('Histogram heights',{exact:true}).fill('-1, 2');await histogram.getByRole('button',{name:'Apply heights',exact:true}).click();
      assert(await histogram.getByRole('alert').isVisible());assert.equal(await histogram.locator('.monostack-candidate').innerText(),oldCandidate);
      await histogram.getByRole('button',{name:'Equal plateau',exact:true}).click();assert((await histogram.innerText()).includes('Largest area: 4'));
      await histogram.getByRole('button',{name:'Tied maxima',exact:true}).click();assert((await histogram.innerText()).includes('Largest area: 6'));
      await capture(histogram.getByRole('region',{name:'Chosen histogram rectangle and its excluded smaller boundary bars',exact:true}),'histogram-tied');
      await histogram.getByRole('button',{name:'Zero heights',exact:true}).click();assert((await histogram.innerText()).includes('No positive-area rectangle'));assert.deepEqual(await histogram.locator('[data-bar] rect').evaluateAll(nodes=>nodes.map(node=>Number(node.getAttribute('height')))),[0,0,0,0]);await capture(histogram.locator('.monostack-candidate'),'zero-heights');
      await histogram.getByRole('button',{name:'Empty histogram',exact:true}).click();assert(await histogram.getByLabel('Limiting histogram bar',{exact:true}).isDisabled());
      await histogram.getByLabel('Histogram heights',{exact:true}).fill('1, 3, 2');await histogram.getByRole('button',{name:'Apply heights',exact:true}).click();
      assert((await histogram.innerText()).includes('Largest area: 4'));
      await histogram.getByRole('button',{name:'Reset boundaries',exact:true}).click();operatedStates+=11;

      // Open native solutions and all optional guidance, exercising summary keyboards.
      for(const summary of await lesson.locator('summary').all()) {
        if(await summary.isVisible() && !(await summary.evaluate(node=>node.parentElement.open))) {
          await summary.focus();await page.keyboard.press('Enter');
          assert(await summary.evaluate(node=>node.parentElement.open));
        }
      }
      assert.equal(await lesson.locator('.dsa-practice__problem').count(),14);
      for(const number of [142,876,739,84]) assert.equal(await lesson.locator(`.dsa-practice__number`).filter({hasText:new RegExp(`^${number}\\.$`)}).count(),1);
      for(const practice of await lesson.locator('.linked-extension-practice').all()) {
        assert.equal(await practice.locator('details[open]').count(),2);
        assert((await practice.innerText()).length>350);
      }
      await capture(lesson.locator('.linked-extension-practice').nth(2),'changed-stack-practice');
      await capture(lesson.locator('.linked-extension-practice').nth(3),'changed-histogram-practice');
      const programs=await lesson.locator('.python-example').evaluateAll(nodes=>nodes.map(node=>({
        code:[...node.querySelectorAll('pre')].map(item=>item.textContent),
        text:node.innerText,
      })));
      // CodeBlock currently renders styled divs rather than native pre: inspect direct text leaves.
      const displayed=await lesson.locator('.python-example').evaluateAll(nodes=>nodes.map(node=>({text:node.textContent, preceding:node.previousElementSibling?.textContent||''})));
      const allPrograms=[...Object.values(linkedExamples),...Object.values(linkedTraversalExamples),...Object.values(monotonicStackExamples)];
      assert.equal(displayed.length,10);
      for(const example of allPrograms){const match=displayed.find(row=>row.text.includes(example.code.trim()));assert(match,'Displayed complete code mismatch');assert(match.text.includes(example.output.trim()),'Displayed complete output mismatch');if(example.question)assert(match.preceding.includes(example.question));}
      await capture(lesson.locator('.python-example').last(),'new-code-output');
      await capture(lesson.locator('.lesson-sources'),'references');
      const proof=lesson.locator('p').filter({hasText:"The meeting node is also the node reached after slow's t hops"});assert.equal(await proof.count(),1);await capture(proof,'reset-proof');
      const optionWidths=await lesson.locator('.traversal-lab select,.monostack-lab select').evaluateAll(nodes=>nodes.map(node=>{const context=document.createElement('canvas').getContext('2d');const style=getComputedStyle(node);context.font=`${style.fontSize} ${style.fontFamily}`;return {label:node.getAttribute('aria-label'),available:node.clientWidth-parseFloat(style.paddingLeft)-parseFloat(style.paddingRight)-25,maximum:Math.max(...[...node.options].map(option=>context.measureText(option.textContent).width))};}));
      assert(optionWidths.every(row=>row.maximum<=row.available),JSON.stringify(optionWidths));
      assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+2),false);
      assert.deepEqual(errors,[]);
      results.push({width,actualFont:true,operatedStates,allPrograms:displayed.length,originalLabs:3,newLabs:4,practicePlacements:14,originalTeachingPreservation:'61 original AST subtrees separately verified',anchors:14,optionWidths,errors,documentOverflow:false});
      await page.close();
    }
    assert.deepEqual(productionSources,productionSources.map(({path})=>({path,sha256:hash(path)})));
    fs.writeFileSync(`${directory}/results.json`,JSON.stringify({checkedAt:new Date().toISOString(),status:'passed',productionSources,results,images,limits:'Actual Edge browser at three widths with fonts/keyboard/reading and targeted original-lab preservation; no broader production integration or screen-reader session claimed.'},null,2));
    console.log(JSON.stringify(results,null,2));
  } finally {await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});

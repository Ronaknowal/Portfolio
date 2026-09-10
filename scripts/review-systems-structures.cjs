const {chromium}=require(process.env.PLAYWRIGHT_PACKAGE||'playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');
(async()=>{
  const {topicCatalogue}=await import('../src/learn/data/curriculum/topic-catalogue.js');
  const {learningPaths,getLearningRoute}=await import('./lib/authoring-curriculum.mjs');
  const {scheduleTrace,translationModel,sharingTrace}=await import('../src/learn/data/os-foundations-model.js');
  const {arrayMovementTrace,textModel,hashTrace}=await import('../src/learn/data/array-map-foundations-model.js');
  const {reverseTrace,bracketTrace,ringTrace}=await import('../src/learn/data/linked-foundations-model.js');
  const route=getLearningRoute(learningPaths.find(p=>p.id==='full-curriculum')).topicIds;
  const ids=['os-processes-virtual-memory-isolation','arrays-strings-hash-maps','linked-lists-stacks-queues'];
  ids.forEach(id=>assert.ok(route.includes(id)));
  const dir='scratch/systems-three-review';fs.mkdirSync(dir,{recursive:true});
  const browser=await chromium.launch({channel:'msedge',headless:true});
  const errors=[],results=[];
  try {
    for(const width of [1440,390]) {
      const page=await browser.newPage({viewport:{width,height:1000}});
      page.on('pageerror',e=>errors.push(e.message));
      page.on('console',m=>{if(m.type()==='error'&&/validateDOMNesting|cannot be a child|Each child/.test(m.text()))errors.push(m.text());});
      const capture=async(lab,name)=>{
        await page.locator('.learn-nav').evaluate(n=>n.style.visibility='hidden');
        try{await lab.screenshot({path:`${dir}/${name}-${width}.png`});}
        finally{await page.locator('.learn-nav').evaluate(n=>n.style.visibility='');}
      };
      const geometry=async()=>{
        assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'horizontal page overflow');
        const bad=await page.locator('.reader-article svg text').evaluateAll(ns=>ns.filter(n=>n.getBoundingClientRect().width).flatMap(n=>{const a=n.getBoundingClientRect(),s=n.ownerSVGElement.getBoundingClientRect();return a.left<s.left-2||a.right>s.right+2||a.top<s.top-2||a.bottom>s.bottom+2?[n.textContent]:[];}));
        assert.deepEqual(bad,[],'SVG text outside viewBox');
      };
      const trace=async(lab,states,check)=>{
        for(let i=0;i<states.length;i++){
          await check(states[i],i);
          await geometry();
          if(i<states.length-1)await lab.getByRole('button',{name:'Next step',exact:true}).click();
        }
        assert.ok(await lab.getByRole('button',{name:'Next step',exact:true}).isDisabled());
        if(states.length>1){await lab.getByRole('button',{name:'Back',exact:true}).click();await check(states.at(-2),states.length-2);}
        await lab.getByRole('button',{name:'Reset',exact:true}).click();await check(states[0],0);
        assert.ok(await lab.getByRole('button',{name:'Back',exact:true}).isDisabled());
      };
      for(const [i,id]of ids.entries()) {
        if(process.env.LESSON_ID&&process.env.LESSON_ID!==id)continue;
        await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/'+id);
        await page.locator('.reader-article .lesson-intro').waitFor();
        assert.equal(await page.locator('.reader-header h1').innerText(),topicCatalogue[id].title);
        assert.equal((await page.locator('.reader-header .topic-status').innerText()).toLowerCase(),'published');
        assert.equal(await page.locator('.reader-article .lesson-lab').count(),3);
        assert.ok((await page.locator('.reader-footer__next').innerText()).includes(topicCatalogue[route[route.indexOf(id)+1]].title));
        for(const target of await page.locator('.reader-article a[href^="#"]').evaluateAll(ns=>ns.map(n=>n.hash.slice(1))))assert.ok(await page.evaluate(id=>!!document.getElementById(id),target),'missing anchor '+target);
        for(const target of await page.locator('.reader-article a[href^="/learn/topic/"]').evaluateAll(ns=>ns.map(n=>n.getAttribute('href').split('/').pop())))assert.ok(topicCatalogue[target],'unknown topic '+target);
        assert.ok(await page.locator('.nt-resources a').count()>=3);
        assert.equal(await page.locator('.lesson-sources ul > :not(li)').count(),0);
        await page.screenshot({path:`${dir}/lesson-${10+i}-${width}.png`});
        if(i===0){
          const scheduler=page.locator('[data-investigation="os-scheduler"]');
          for(const quantum of [1,2])for(const io of [false,true]){
            await scheduler.getByLabel('Time slice').selectOption(String(quantum));await scheduler.getByLabel("A's middle instruction").selectOption(String(io));
            await trace(scheduler,scheduleTrace(quantum,io),async s=>{
              const rows=await scheduler.locator('tbody tr').allTextContents();
              assert.deepEqual(rows,Object.entries(s.jobs).map(([id,j])=>[id,j.value,j.code[j.pc]??'finished',j.state].join('')));
              assert.equal(await scheduler.locator('.nt-feedback').innerText(),s.note);
            });
          }
          await scheduler.getByRole('button',{name:'Next step',exact:true}).focus();await page.keyboard.press('Enter');
          assert.match(await scheduler.locator('.foundation-state').first().innerText(),/Time 1/);
          assert.equal(await scheduler.getByRole('button',{name:'Next step',exact:true}).evaluate(n=>getComputedStyle(n).outlineStyle),'solid');
          await capture(scheduler,'os-scheduler');
          const translate=page.locator('[data-investigation="os-translation"]');
          for(const process of ['A','B'])for(const address of [6,22,38,54])for(const access of ['read','write']){
            await translate.getByLabel('Selected process').selectOption(process);await translate.getByLabel('Virtual byte address').selectOption(String(address));await translate.getByLabel('Requested access').selectOption(access);
            const m=translationModel(process,address,access);
            await trace(translate,[0,1,2,3],async step=>{
              assert.deepEqual(await translate.locator('tbody tr').allTextContents(),m.table.map(p=>[p.page,m.kind==='demand'&&step>=2&&p.page===m.page?m.frame:p.frame??'not resident',p.rights].join('')));
              if(step===3){const label=await translate.locator('svg').getAttribute('aria-label');assert.ok(label.includes(m.kind));assert.ok(label.includes(m.physical===null?'no completed access yet':'physical '+m.physical));}
            });
          }
          await translate.getByLabel('Virtual byte address').selectOption('38');for(let k=0;k<3;k++)await translate.getByRole('button',{name:'Next step',exact:true}).click();
          await capture(translate,'os-translation');
          const sharing=page.locator('[data-investigation="os-sharing"]');
          for(const shared of [false,true]){
            await sharing.getByLabel('Mapping contract').selectOption(String(shared));
            await trace(sharing,sharingTrace(shared),async s=>assert.equal(await sharing.locator('svg').getAttribute('aria-label'),`A maps frame ${s.aFrame} and reads ${s.aValue}; B maps frame ${s.bFrame} and reads ${s.bValue}`));
          }
          await sharing.getByLabel('Mapping contract').selectOption('false');for(let k=0;k<3;k++)await sharing.getByRole('button',{name:'Next step',exact:true}).click();
          await capture(sharing,'os-sharing');
        }else if(i===1){
          const movement=page.locator('[data-investigation="array-movement"]');
          for(const op of ['front','insert','append','delete'])for(const full of [false,true]){
            await movement.getByLabel('Sequence operation').selectOption(op);await movement.getByLabel('Starting capacity').selectOption(String(full));
            await trace(movement,arrayMovementTrace(op,full),async s=>{assert.equal(await movement.locator('.nt-feedback').innerText(),s.note);assert.match(await movement.locator('.foundation-state').innerText(),new RegExp('element writes '+s.writes));assert.ok((await movement.locator('svg').getAttribute('aria-label')).includes(s.cells.map(v=>v??'unused').join(', ')));});
          }
          await movement.getByLabel('Sequence operation').selectOption('front');await movement.getByLabel('Starting capacity').selectOption('true');for(let k=0;k<3;k++)await movement.getByRole('button',{name:'Next step',exact:true}).click();await capture(movement,'array-movement');
          const text=page.locator('[data-investigation="text-units"]');
          for(const kind of ['ascii','composed','decomposed','emoji'])for(const normalize of [false,true]){
            await text.getByLabel('Text sample').selectOption(kind);await text.getByLabel('Text operation').selectOption(String(normalize));
            const m=textModel(kind,normalize);assert.ok((await text.innerText()).includes(`${m.length} code points · ${m.bytes.length} UTF-8 bytes`));
            for(const [index,unit]of m.units.entries()){
              const button=text.getByRole('button',{name:new RegExp('index '+index+' ')});await button.click();assert.equal(await button.getAttribute('aria-pressed'),'true');
              assert.ok((await text.locator('.nt-feedback').innerText()).includes(unit.code));
            }
          }
          await text.getByRole('button',{name:'Reset text',exact:true}).click();assert.equal(await text.getByLabel('Text sample').inputValue(),'decomposed');await capture(text,'text-units');
          const hash=page.locator('[data-investigation="hash-buckets"]');
          for(const key of [10,14,18,22])for(const capacity of [4,5])for(const op of ['get','set']){
            await hash.getByLabel('Event key').selectOption(String(key));await hash.getByLabel('Bucket count').selectOption(String(capacity));await hash.getByLabel('Map operation').selectOption(op);
            await trace(hash,hashTrace(key,capacity,op),async s=>{assert.equal(await hash.locator('.nt-feedback').innerText(),s.note);assert.equal(await hash.locator('.foundation-state').innerText(),`Key comparisons: ${s.comparisons} · result: ${s.result??'not decided'}`);});
          }
          await hash.getByLabel('Event key').selectOption('18');await hash.getByLabel('Bucket count').selectOption('4');await hash.getByLabel('Map operation').selectOption('get');for(let k=0;k<2;k++)await hash.getByRole('button',{name:'Next step',exact:true}).click();await capture(hash,'hash-buckets');
        }else{
          const reverse=page.locator('[data-investigation="linked-reversal"]');
          for(const size of [3,1,0])for(const broken of [false,true]){
            await reverse.getByLabel('List length').selectOption(String(size));await reverse.getByLabel('Rewiring order').selectOption(String(broken));
            await trace(reverse,reverseTrace(size,broken),async s=>{assert.equal(await reverse.locator('.nt-feedback').innerText(),s.note);const refs=await reverse.locator('.foundation-reference').innerText();for(const name of ['head','previous','current','saved'])assert.ok(refs.includes(`${name} → ${s[name]??'∅'}`));});
          }
          await reverse.getByLabel('List length').selectOption('3');await reverse.getByLabel('Rewiring order').selectOption('false');for(let k=0;k<5;k++)await reverse.getByRole('button',{name:'Next step',exact:true}).click();await capture(reverse,'linked-reversal');
          const brackets=page.locator('[data-investigation="bracket-stack"]');
          for(const input of ['([])','([)]',')','(()','']){
            await brackets.getByLabel('Bracket input').selectOption(input);
            await trace(brackets,bracketTrace(input),async s=>{assert.equal(await brackets.locator('.nt-feedback').innerText(),s.note);assert.equal(await brackets.locator('.foundation-state').innerText(),s.result===null?'Still scanning':s.result?'Accept: nested and closed':'Reject: mismatch or missing partner');});
          }
          await brackets.getByLabel('Bracket input').selectOption('([)]');for(let k=0;k<3;k++)await brackets.getByRole('button',{name:'Next step',exact:true}).click();await capture(brackets,'bracket-stack');
          const ring=page.locator('[data-investigation="circular-queue"]');
          for(const capacity of [3,4]){
            await ring.getByLabel('Buffer capacity').selectOption(String(capacity));
            await trace(ring,ringTrace(capacity),async s=>{assert.equal(await ring.locator('.nt-feedback').innerText(),s.note);assert.ok((await ring.innerText()).includes(`Logical FIFO order: ${s.logical.join(' → ')||'empty'}`));assert.deepEqual(await ring.locator('.foundation-buffer strong').allTextContents(),s.cells.map(v=>v??'·'));});
          }
          for(let k=0;k<6;k++)await ring.getByRole('button',{name:'Next step',exact:true}).click();await capture(ring,'circular-queue');
        }
        // Open nested solutions too; check their code/output and responsive containment.
        await page.locator('.reader-article details').evaluateAll(ns=>ns.forEach(n=>n.open=true));
        assert.equal(await page.locator('.reader-article .python-example').count(),i===0?4:6);
        await geometry();
        const smallControls=await page.locator('.nt-lab button,.nt-lab select').evaluateAll(ns=>ns.filter(n=>n.getBoundingClientRect().height<43).map(n=>n.textContent));assert.deepEqual(smallControls,[]);
        for(const s of await page.locator('.reader-article details').all()){assert.ok(await s.locator(':scope > summary').isVisible());}
        results.push({id,width,labs:3,examples:i===0?4:6,allControlStates:true,expandedSolutions:true,overflow:false});
      }
      await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/linux-basics-filesystems-processes');
      await page.locator('.reader-article .lesson-intro').waitFor();
      assert.equal(await page.locator('.reader-article .lesson-lab').count(),4);
      assert.ok((await page.locator('.reader-footer__next').innerText()).includes('Bash Scripting'));
      assert.equal(await page.locator('.nt-resources a[href*="youtube.com"]').count(),1);
      assert.ok((await page.locator('.reader-article').innerText()).includes('Next in this module, Bash Scripting'));
      await geometry();await capture(page.locator('.lesson-sources'),'linux-resources');
      await page.close();
    }
    assert.deepEqual(errors,[]);fs.writeFileSync(`${dir}/browser-results${process.env.LESSON_ID?'-'+process.env.LESSON_ID:''}.json`,JSON.stringify({results,errors},null,2));
    console.log(`PASS: ${process.env.LESSON_ID||'all nine labs'} across all controls/steps at 1440 and 390px; keyboard, reset/back, expanded solutions, anchors/topic links, resources, exact next sequence, SVG bounds, no page overflow; Linux keeps four labs and correct Bash bridge.`);
  }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});

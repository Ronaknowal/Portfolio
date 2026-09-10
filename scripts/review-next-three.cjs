const {chromium}=require(process.env.PLAYWRIGHT_PACKAGE||'playwright');
const assert=require('node:assert/strict');
const fs=require('node:fs');

(async()=>{
  const {learningPaths,getPathTopicIds}=await import('./lib/authoring-curriculum.mjs');
  const {topicCatalogue}=await import('../src/learn/data/curriculum/topic-catalogue.js');
  const {modelJoin}=await import('../src/learn/data/pandas-join-model.js');
  const {alignmentModel,groupingModel}=await import('../src/learn/data/pandas-foundations-model.js');
  const {coordinateModel,histogramModel,intervalModel}=await import('../src/learn/data/plotting-foundations-model.js');
  const {stagingTrace,branchTrace,remoteTrace,conflictTrace}=await import('../src/learn/data/git-foundations-model.js');
  const ids=['pandas-data-wrangling-joins-grouping','matplotlib-scientific-plotting','git-github-collaborative-version-control'];
  const path=getPathTopicIds(learningPaths.find(p=>p.id==='full-curriculum'));
  ids.forEach(id=>assert.ok(path.includes(id)));
  const dir='scratch/next-three-review';fs.mkdirSync(dir,{recursive:true});
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
      for(const [i,id] of ids.entries()) {
        await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/'+id);
        await page.locator('.reader-article .lesson-intro').waitFor();
        // Stabilize layout before tall lab captures; lazy plots above a lab can
        // otherwise load during scrolling and move the element being captured.
        await page.locator('.reader-article img').evaluateAll(async ns=>{ns.forEach(n=>n.loading='eager');await Promise.all(ns.map(n=>n.decode()));});
        assert.equal(await page.locator('.reader-header h1').innerText(),topicCatalogue[id].title);
        assert.equal((await page.locator('.reader-header .topic-status').innerText()).toLowerCase(),'published');
        assert.equal(await page.locator('.reader-article .lesson-lab').count(),[4,3,4][i]);
        const nextId=path[path.indexOf(id)+1];
        assert.ok((await page.locator('.reader-footer__next').innerText()).includes(topicCatalogue[nextId].title));
        for(const target of await page.locator('.reader-article a[href^="#"]').evaluateAll(ns=>ns.map(n=>n.hash.slice(1)))) assert.ok(await page.evaluate(id=>!!document.getElementById(id),target),'Missing anchor '+target);
        for(const target of await page.locator('.reader-article a[href^="/learn/topic/"]').evaluateAll(ns=>ns.map(n=>n.getAttribute('href').split('/').pop()))) assert.ok(topicCatalogue[target],'Unknown topic '+target);
        assert.equal(await page.locator('.nt-resources a[href*="youtube.com"]').count(),1);
        assert.ok(await page.locator('.nt-resources a:not([href*="youtube.com"])').count()>=1);
        assert.equal(await page.locator('.lesson-sources ul > :not(li)').count(),0);
        assert.ok(await page.locator('.reader-complete').isEnabled());
        await page.screenshot({path:`${dir}/lesson-${i+6}-${width}.png`});

        if(i===0) {
          const align=page.locator('[data-investigation="pandas-alignment"]');
          for(const mode of ['labels','positions'])for(const reversed of [false,true])for(const incomplete of [false,true]) {
            await align.getByLabel('Assignment rule').selectOption(mode);
            await align.getByLabel('Order row arrangement').selectOption(String(reversed));
            await align.getByLabel('Incoming fee labels').selectOption(String(incomplete));
            const model=alignmentModel(mode,reversed,incomplete);
            for(const [index,r] of model.rows.entries()) {
              await align.getByLabel('Inspect order row').selectOption(String(index));
              assert.match(await align.locator('svg').getAttribute('aria-label'),new RegExp('Order '+r.label+' receives '+(r.fee===null?'no matching fee':'fee '+r.fee)));
            }
            const table=await align.locator('tbody tr').allTextContents();
            model.rows.forEach((r,j)=>assert.equal(table[j],[r.label,r.value,r.fee??'NA',r.total??'NA'].join('')));
          }
          await align.getByRole('button',{name:'Reset alignment'}).click();
          assert.equal(await align.getByLabel('Assignment rule').inputValue(),'labels');
          await capture(align,'pandas-alignment');
          const clean=page.locator('[data-investigation="pandas-cleaning"]');
          for(const policy of ['known','positive']) {
            await clean.getByLabel('Selection policy').selectOption(policy);
            await clean.getByRole('button',{name:'Next step',exact:true}).focus();await page.keyboard.press('Enter');
            for(let j=0;j<2;j++)await clean.getByRole('button',{name:'Next step',exact:true}).click();
            assert.match(await clean.locator('.nt-feedback').innerText(),policy==='known'?/Selected source rows: 0, 3/:/Selected source rows: 0\./);
            await capture(clean,'pandas-cleaning-'+policy);
            await clean.getByRole('button',{name:'Back',exact:true}).click();
            await clean.getByRole('button',{name:'Reset',exact:true}).click();assert.ok(await clean.getByRole('button',{name:'Back',exact:true}).isDisabled());
          }
          const join=page.locator('.pandas-join-lab');
          for(const how of ['left','inner','outer'])for(const duplicate of [false,true])for(const validate of [false,true]) {
            await join.getByLabel('Join type').selectOption(how);await join.getByLabel('Add a second C1 lookup row').setChecked(duplicate);await join.getByLabel('Enforce many-to-one validation').setChecked(validate);
            const model=modelJoin(how,duplicate,validate);
            if(model.error)assert.ok((await join.locator('.lesson-results').innerText()).includes(model.error));
            else {
              const buttons=join.locator('.nt-records button');assert.equal(await buttons.count(),model.rows.length);
              for(const [j,row]of model.rows.entries()) {await buttons.nth(j).click();assert.equal(await buttons.nth(j).getAttribute('aria-pressed'),'true');assert.ok((await join.locator('.nt-columns').innerText()).includes(row.customer));}
            }
          }
          await join.getByRole('button',{name:'Reset join'}).click();assert.equal(await join.locator('.nt-records button').first().getAttribute('aria-pressed'),'true');await capture(join,'pandas-join');
          const group=page.locator('[data-investigation="pandas-grouping"]');
          for(const keepMissing of [false,true])for(const fillZero of [false,true]) {
            await group.getByLabel('Missing group key').selectOption(String(keepMissing));await group.getByLabel('Unknown amount policy').selectOption(String(fillZero));
            for(const g of groupingModel(keepMissing,fillZero).groups) {
              await group.getByLabel('Trace group').selectOption(g.key??'missing');
              assert.equal(await group.locator('.nt-reducer .nt-big-value').innerText(),`${g.sum} ÷ ${g.count} = ${g.mean}`);
              assert.equal(await group.locator('.nt-records .is-linked').count(),g.members.length);
            }
          }
          await group.getByRole('button',{name:'Reset grouping'}).click();await capture(group,'pandas-grouping');
        }
        if(i===1) {
          const coord=page.locator('[data-investigation="plot-coordinates"]');
          for(const view of ['full','zoom','log'])for(const selected of [0,1]) {
            await coord.getByLabel('Axis mapping').selectOption(view);await coord.getByLabel('Trace measurement').selectOption(String(selected));
            assert.ok((await coord.locator('.nt-feedback').innerText()).includes(coordinateModel(view).points[selected].fraction.toFixed(3)));
          }
          await capture(coord,'plot-coordinates-log');await coord.getByRole('button',{name:'Reset mapping'}).click();assert.equal(await coord.getByLabel('Axis mapping').inputValue(),'full');
          const hist=page.locator('[data-investigation="plot-histogram"]');
          for(const layout of ['three','two'])for(const density of [false,true]) {
            await hist.getByLabel('Bin edges').selectOption(layout);await hist.getByLabel('Rectangle height').selectOption(String(density));
            for(const [j,b]of histogramModel(layout,density).entries()) {
              await hist.getByLabel('Trace bin').selectOption(String(j));assert.match(await hist.locator('svg').getAttribute('aria-label'),new RegExp('contains '+b.count+' of six'));
            }
          }
          await capture(hist,'plot-histogram-density');await hist.getByRole('button',{name:'Reset histogram'}).click();
          const interval=page.locator('[data-investigation="plot-interval"]');
          for(const kind of ['sd','sem'])for(const repeated of [false,true]) {
            await interval.getByLabel('Interval shown').selectOption(kind);await interval.getByLabel('Dataset').selectOption(String(repeated));
            assert.ok((await interval.locator('.nt-feedback').innerText()).includes('Half-width = '+intervalModel(kind,repeated).halfWidth.toFixed(3)));
          }
          await capture(interval,'plot-interval-copies');await interval.getByRole('button',{name:'Reset interval'}).click();
          // Open all optional examples before checking lazy images and actual exported figures.
          await page.locator('.reader-article details').evaluateAll(ns=>ns.forEach(n=>n.open=true));
          const plots=page.locator('.reader-article img[src$=".svg"]');assert.equal(await plots.count(),9);
          for(const img of await plots.all()) {await img.scrollIntoViewIfNeeded();await img.evaluate(n=>n.decode());assert.ok(await img.evaluate(n=>n.naturalWidth>0));}
          const scroll=page.locator('.lesson-plot__scroll').first();
          assert.ok(await scroll.evaluate(n=>n.scrollWidth<=n.clientWidth+1));
          await page.getByRole('button',{name:'Enlarge chart',exact:true}).first().click();
          assert.ok(await scroll.evaluate(n=>n.scrollWidth>n.clientWidth));
          await scroll.focus();await page.keyboard.press('ArrowRight');
          await page.waitForFunction(()=>document.querySelector('.lesson-plot__scroll').scrollLeft>0);
          await page.getByRole('button',{name:'Fit chart',exact:true}).first().focus();await page.keyboard.press('Enter');
          assert.ok(await scroll.evaluate(n=>n.scrollWidth<=n.clientWidth+1));
        }
        if(i===2) {
          const configs=[['git-staging','Before committing',['false','true'],v=>stagingTrace(v==='true')],['git-branches','Main branch activity',['false','true'],v=>branchTrace(v==='true')],['git-remotes','Your local work',['false','true'],v=>remoteTrace(v==='true')],['git-conflict','Chosen resolution',['combined','ours'],conflictTrace]];
          for(const [name,label,options,getTrace]of configs) {
            const lab=page.locator(`[data-investigation="${name}"]`);
            for(const option of options) {
              await lab.getByLabel(label).selectOption(option);
              const trace=getTrace(option);
              for(const [j,state]of trace.entries()) {
                assert.equal(await lab.locator('.nt-feedback').innerText(),state.note);
                if(name==='git-staging')assert.deepEqual(await lab.locator('.nt-snapshot pre').allTextContents(),state.versions.map(v=>'version '+v));
                if(name==='git-branches')assert.ok((await lab.locator('svg').getAttribute('aria-label')).includes('main at '+state.main));
                if(name==='git-remotes')assert.deepEqual(await lab.locator('.nt-big-value').allTextContents(),[state.local,state.tracking,state.shared]);
                if(name==='git-conflict')assert.equal(await lab.locator('.nt-code').innerText(),state.working);
                if(j<trace.length-1){await lab.getByRole('button',{name:'Next step',exact:true}).focus();await page.keyboard.press('Enter');}
              }
              await capture(lab,name+'-'+option);
              await lab.getByRole('button',{name:'Back',exact:true}).click();await lab.getByRole('button',{name:'Reset',exact:true}).click();assert.ok(await lab.getByRole('button',{name:'Back',exact:true}).isDisabled());
            }
          }
        }
        const disclosures=page.locator('.reader-article details');
        await disclosures.evaluateAll(ns=>ns.forEach(n=>n.open=true));
        for(const detail of await disclosures.all()){await detail.locator(':scope > summary').click();await detail.locator(':scope > summary').click();}
        assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),'Document overflow '+id+' '+width);
        const svgOverflow=await page.locator('.nt-diagram').evaluateAll(ns=>ns.flatMap(svg=>[...svg.querySelectorAll('text')].filter(n=>{const b=n.getBBox(),v=svg.viewBox.baseVal;return b.x < -1 || b.x+b.width>v.width+1 || b.y< -1 || b.y+b.height>v.height+1;}).map(n=>n.textContent)));
        assert.deepEqual(svgOverflow,[],'SVG text bounds '+id+' '+width);
        await page.locator('.reader-footer__next').click();await page.waitForURL(url=>url.pathname.endsWith('/'+nextId));
        results.push({id,width,labs:[4,3,4][i],passed:true});
      }
      await page.close();
    }
    assert.deepEqual(errors,[]);
    fs.writeFileSync(dir+'/browser-results.json',JSON.stringify({results,errors},null,2));
    console.log('PASS: 3 lessons, 11 labs, desktop/mobile, control states, keyboard steps, disclosures, assets, links and actual adjacent navigation.');
  } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});

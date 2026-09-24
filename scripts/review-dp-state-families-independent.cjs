const fs=require('node:fs');
const path=require('node:path');
const crypto=require('node:crypto');
const assert=require('node:assert/strict');
const {chromium}=require(process.env.PLAYWRIGHT_PACKAGE||'C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const sourceHashes=require('./dp-state-families-source-hashes.cjs');
const directory='scratch/dp-state-families-independent/browser';
fs.mkdirSync(directory,{recursive:true});
const initial=sourceHashes();
const normalized=text=>text.replace(/\s+/g,' ').trim();
(async()=>{
  const browser=await chromium.launch({channel:'msedge',headless:true});
  const records=[],errors=[];
  try{
    for(const width of [1440,390,320]){
      const page=await browser.newPage({viewport:{width,height:1050},reducedMotion:'reduce'});
      await page.routeWebSocket('**',socket=>socket.close());
      page.on('pageerror',error=>errors.push(error.message));
      const record={width,states:[],images:[],anchors:[],hintStates:[]};
      try{
        await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/dynamic-programming-states-transitions-optimization?module=data-structures-algorithms',{waitUntil:'domcontentloaded'});
        const lesson=page.locator('.dynamic-programming-lesson');
        await lesson.waitFor();
        await page.evaluate(()=>document.fonts.ready);
        record.fonts=await page.evaluate(()=>[...document.fonts].filter(face=>face.status==='loaded').map(face=>face.family));
        assert(record.fonts.includes('Space Grotesk')&&record.fonts.includes('JetBrains Mono'));
        await page.addStyleTag({content:'html{scroll-behavior:auto!important}'});
        async function shot(target,name){
          await target.evaluate(node=>window.scrollTo({top:node.getBoundingClientRect().top+scrollY-95,behavior:'instant'}));
          await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
          const filename=name+'-'+width+'.png';
          await page.screenshot({path:path.join(directory,filename)});
          record.images.push(filename);
        }
        async function key(button){await button.focus();await page.keyboard.press('Enter');}
        async function state(region,name,fragment){
          const text=normalized(await region.innerText());
          assert(text.includes(fragment),name+'\n'+text);
          record.states.push(name);
        }
        for(const id of ['interval-states-split-the-last-operation','tree-states-remember-the-parent-boundary','digit-states-count-constrained-continuations']){
          const anchor=lesson.locator('nav a[href="#'+id+'"]');
          await key(anchor);
          await page.waitForFunction(id=>{const box=document.getElementById(id).getBoundingClientRect();return box.top>=35&&box.top<=155;},id);
          record.anchors.push(id);await shot(page.locator('#'+id),'reading-'+id);
        }
        const interval=lesson.getByRole('region',{name:'Interval split investigation',exact:true});
        await interval.getByLabel('Matrix-chain dimensions',{exact:true}).fill('6,2,5,3');
        await key(interval.getByRole('button',{name:'Apply dimensions',exact:true}));
        await state(interval,'changed shape optimum','Best interval cost: 66');
        const select=interval.getByLabel('Candidate final split',{exact:true});
        await select.focus();await page.keyboard.press('ArrowDown');await page.keyboard.press('Enter');
        await state(interval,'candidate differs from whole-chain optimum','= 150. Best interval cost: 66');
        await shot(interval.getByRole('region',{name:'Candidate expression tree',exact:true}),'matrix-candidate');
        await key(interval.getByRole('button',{name:'Next multiplication',exact:true}));
        await shot(interval.locator('.dpf-operation-list'),'matrix-postorder');
        await key(interval.getByRole('button',{name:'Inspect interval 1 to 2',exact:true}));
        await state(interval,'single matrix cell','One existing matrix: cost 0');
        await interval.getByLabel('Matrix-chain dimensions',{exact:true}).fill('4,0,3');
        await key(interval.getByRole('button',{name:'Apply dimensions',exact:true}));
        assert.equal(await interval.getByRole('alert').count(),1);
        await state(interval,'rejected dimensions retain old table','Active dimensions: [6, 2, 5, 3]');
        await key(interval.getByRole('button',{name:'Reset interval lab',exact:true}));
        await state(interval,'interval reset','All operation costs sum to 204');
        await shot(lesson.locator('.dpf-inline').nth(1),'balloon-order');

        const tree=lesson.getByRole('region',{name:'Tree parent boundary investigation',exact:true});
        await tree.getByLabel('Node weights in ID order',{exact:true}).fill('7,-2,4,6,0,8');
        await key(tree.getByRole('button',{name:'Apply tree weights',exact:true}));
        await tree.getByLabel('Tree shape',{exact:true}).selectOption('star');
        await tree.getByLabel('Subtree root',{exact:true}).selectOption('0');
        await state(tree,'changed star free witness','Requested F(0,0) = 18; selected IDs: 2, 3, 5');
        const parent=tree.getByLabel('External parent selected',{exact:true});
        await parent.focus();await page.keyboard.press('Space');
        await state(tree,'changed star conditional witness','Requested F(0,1) = 18');
        await shot(tree.getByRole('region',{name:'Weighted tree and conditional witness',exact:true}),'star-conditional');
        await tree.getByLabel('Tree shape',{exact:true}).selectOption('chain');
        await tree.getByLabel('Subtree root',{exact:true}).selectOption('0');
        await parent.focus();await page.keyboard.press('Space');
        await state(tree,'same weights chain changes feasible sets','Requested F(0,0) = 21; selected IDs: 0, 3, 5');
        await shot(tree.locator('.dpf-boundary-values'),'chain-values');
        await key(tree.getByRole('button',{name:'Previous postorder node',exact:true}));
        await state(tree,'cursor is dependency inspection','These are already computed answers');
        await tree.getByLabel('Node weights in ID order',{exact:true}).fill('');
        await key(tree.getByRole('button',{name:'Apply tree weights',exact:true}));
        await state(tree,'empty tree','Empty tree: value 0');
        await key(tree.getByRole('button',{name:'Reset tree lab',exact:true}));
        await state(tree,'tree reset','Requested F(1,0) = 9');

        const digit=lesson.getByRole('region',{name:'Digit prefix investigation',exact:true});
        await digit.getByLabel('Inclusive upper bound',{exact:true}).fill('1023');
        await key(digit.getByRole('button',{name:'Apply digit bound',exact:true}));
        async function append(value){await key(digit.getByRole('button',{name:new RegExp('^Append digit '+value+':')}));}
        await append(1);await append(0);
        await state(digit,'real zero consumes zero','Prefix 10 has 1 valid positive completions. One is 1023');
        assert(await digit.getByRole('button',{name:/^Append digit 0:/}).isDisabled());
        await shot(digit.locator('.dpf-digit-tray'),'real-zero-state');
        await key(digit.getByRole('button',{name:'Return to empty prefix',exact:true}));
        await append(0);await append(1);
        await state(digit,'padding zero leaves zero available','Prefix 01 has 72 valid positive completions');
        assert(!(await digit.getByRole('button',{name:/^Append digit 0:/}).isDisabled()));
        await append(0);
        await state(digit,'same numerical zero now actual','Prefix 010 has 8 valid positive completions');
        await shot(digit.locator('.dpf-prefix-strip'),'padding-then-real-zero');
        await key(digit.getByRole('button',{name:'Previous prefix',exact:true}));
        await state(digit,'prefix backward exact count','Prefix 01 has 72');
        await digit.getByLabel('Inclusive upper bound',{exact:true}).fill('0001');
        await key(digit.getByRole('button',{name:'Apply digit bound',exact:true}));
        assert.equal(await digit.getByRole('alert').count(),1);
        await state(digit,'invalid bound preserves cache','Active bound 1023');
        await digit.getByLabel('Inclusive upper bound',{exact:true}).fill('0');
        await key(digit.getByRole('button',{name:'Apply digit bound',exact:true}));
        await append(0);
        await state(digit,'all padding is excluded','this leaf contributes 0');
        await key(digit.getByRole('button',{name:'Reset digit lab',exact:true}));
        await state(digit,'digit reset','Active bound 213; positive integers with all digits distinct: 172');

        const checkpoints=lesson.locator('[data-dpf-checkpoint]');
        assert.equal(await checkpoints.count(),5);
        for(const [index,checkpoint] of (await checkpoints.all()).entries()){
          const details=checkpoint.locator(':scope > details');
          assert.equal(await details.count(),2);
          assert.equal(await details.nth(0).getAttribute('open'),null);
          assert.equal(await details.nth(1).getAttribute('open'),null);
          await key(details.nth(0).locator('summary'));
          assert.notEqual(await details.nth(0).getAttribute('open'),null);
          assert.equal(await details.nth(1).getAttribute('open'),null);
          await key(details.nth(1).locator('summary'));
          assert.notEqual(await details.nth(1).getAttribute('open'),null);
          record.hintStates.push(index);
          if(index===3)await shot(checkpoint,'nonempty-task-hint');
        }
        const practiceLinks=await lesson.locator('.dsa-practice a[href*="leetcode.com/problems/"]').evaluateAll(nodes=>nodes.map(node=>({href:node.href,target:node.target,rel:node.rel})));
        assert.equal(practiceLinks.length,15);
        assert(practiceLinks.every(link=>link.target==='_blank'&&link.rel.includes('noreferrer')));
        const badLabels=await lesson.locator('.dpf-lab svg').evaluateAll(nodes=>nodes.flatMap(svg=>[...svg.querySelectorAll('text')].filter(text=>{const box=text.getBBox();return box.x<-30||box.width>58;}).map(text=>text.textContent)));
        assert.deepEqual(badLabels,[]);
        assert(!(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth+1)));
        assert.deepEqual(errors,[]);
        records.push(record);
      }catch(error){await page.screenshot({path:path.join(directory,'failure-'+width+'.png')});throw error;}
      await page.close();
    }
    assert.deepEqual(sourceHashes(),initial,'Exact reviewed sources stayed stable');
    fs.writeFileSync(directory+'/results.json',JSON.stringify({checkedAt:new Date().toISOString(),passed:true,sourceHashes:initial,records,errors},null,2));
    console.log('Independent DP changed three-width original-font/keyboard/hint/read checks passed.');
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1;});

const fs = require('node:fs');
const { chromium } = require('C:/Users/ronak/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: 320, height: 1000 } });
    await page.routeWebSocket('**', socket => socket.close());
    await page.goto('http://127.0.0.1:5173/learn/path/full-curriculum/real-analysis-sequences-modes-of-convergence?module=math-foundations');
    await page.locator('.real-analysis-lesson').waitFor();
    await page.evaluate(() => document.fonts.ready);
    await page.locator('.real-analysis-lesson details').evaluateAll(nodes => nodes.forEach(node => { node.open = true; }));
    const result = await page.locator('.real-analysis-lesson *').evaluateAll(nodes => nodes.flatMap(node => {
      const rect = node.getBoundingClientRect();
      const style = getComputedStyle(node);
      if (node.closest('.katex-mathml')) return [];
      for (let ancestor = node.parentElement; ancestor && !ancestor.classList.contains('real-analysis-lesson'); ancestor = ancestor.parentElement) {
        if (['auto', 'scroll', 'hidden', 'clip'].includes(getComputedStyle(ancestor).overflowX)) return [];
      }
      if (rect.right <= innerWidth + 1 || rect.width === 0) return [];
      return [{ tag: node.tagName, class: node.getAttribute('class'), text: node.textContent.slice(0, 180), right: rect.right, width: rect.width, overflow: style.overflowX }];
    }).slice(0, 45));
    fs.writeFileSync('scratch/real-analysis-browser/layout-diagnostic.json', JSON.stringify(result, null, 2));
    console.log(JSON.stringify(result, null, 2));
    console.log(await page.evaluate(() => ({documentWidth:document.documentElement.scrollWidth,innerWidth})));
    console.log(await page.evaluate(() => {
      const walker=document.createTreeWalker(document.body,NodeFilter.SHOW_TEXT);
      const results=[];
      while(walker.nextNode()) {
        const node=walker.currentNode, parent=node.parentElement;
        if(!node.textContent.trim() || parent.closest('.katex-mathml,script,style')) continue;
        if(!parent.getClientRects().length) continue;
        let clipped=false;
        for(let ancestor=parent;ancestor&&ancestor!==document.body;ancestor=ancestor.parentElement) if(['auto','scroll','hidden','clip'].includes(getComputedStyle(ancestor).overflowX)){clipped=true;break;}
        if(clipped) continue;
        const range=document.createRange();range.selectNodeContents(node);
        const box=range.getBoundingClientRect();
        if(box.right>innerWidth+1) results.push({tag:parent.tagName,class:parent.className,right:box.right,text:node.textContent});
      }
      return results.slice(0,20);
    }));
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });

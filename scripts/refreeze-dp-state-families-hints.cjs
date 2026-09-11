const fs=require('node:fs'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const hashes=require('./dp-state-families-source-hashes.cjs');
const read=path=>JSON.parse(fs.readFileSync(path,'utf8'));
const packetPath='docs/teaching/evidence/dp-state-families-author-review.json';
const previousPath='docs/teaching/evidence/dp-state-families-author-review-before-hints.json';
const before=read(previousPath),amendment=read('docs/teaching/evidence/dp-state-families-hint-amendment.json');
const browser=read('docs/teaching/evidence/dp-state-families-hint-browser.json');
const sourceHashes=hashes();assert.deepEqual(browser.sourceHashes,sourceHashes);
for(const source of sourceHashes){const old=before.sourceHashes.find(item=>item.path===source.path);assert.equal(source.sha256,source.path===amendment.path?amendment.afterSha256:old.sha256);}
assert.equal(before.sourceHashes.find(source=>source.path===amendment.path).sha256,amendment.beforeSha256);
for(const result of browser.results){assert.equal(result.expandedDisclosures,55);assert.deepEqual(result.errors,[]);assert.deepEqual(result.failedRequests,[]);}
const hash=bytes=>crypto.createHash('sha256').update(bytes).digest('hex');
assert.equal(hash(fs.readFileSync(amendment.archive)),amendment.beforeSha256);
const superseded=new Set(['scratch/dp-state-families-reading/tree-nonempty-solution-390.png','scratch/dp-state-families-reading/balloon-signed-solution-1440.png']);
const newImages=['hint-0-1440.png','hint-1-320.png','hint-2-320.png','hint-3-390.png','hint-4-320.png'].map(name=>{const path=`scratch/dp-state-families-hints/${name}`;return {path,sha256:hash(fs.readFileSync(path)),openedByAuthor:true,sourceStage:'final hint amendment'};});
const record={...before,
 frozenAt:new Date().toISOString(),sourceHashes,
 previousAuthorPacket:previousPath,previousFrozenAt:before.frozenAt,
 finalHintAmendment:'docs/teaching/evidence/dp-state-families-hint-amendment.json',
 finalHintBrowser:'docs/teaching/evidence/dp-state-families-hint-browser.json',
 disclosures:browser.results.map(({width,expandedDisclosures})=>({width,disclosures:expandedDisclosures})),
 actuallyOpenedImages:[...before.actuallyOpenedImages.map(image=>({...image,superseded:superseded.has(image.path),sourceStage:superseded.has(image.path)?'historical answer-only layout':'unchanged representation captured before hint-only amendment'})),...newImages],
 limitations:[...before.limitations,'Final body-only hint scaffold amendment adds five disclosures. It receives its own three-width keyboard/reading checks; the previous author packet and body bytes remain preserved. Two superseded answer-only images remain explicitly historical.'],
};
fs.writeFileSync(packetPath,JSON.stringify(record,null,2)+'\n');
console.log(JSON.stringify({frozenAt:record.frozenAt,sourceHashes,finalApplicableImages:record.actuallyOpenedImages.filter(image=>!image.superseded).length},null,2));

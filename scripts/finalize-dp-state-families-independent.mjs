import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import crypto from 'node:crypto';
import {parse} from '@babel/parser';

const hash=file=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const read=file=>JSON.parse(fs.readFileSync(file,'utf8'));
const packetPath='docs/teaching/evidence/dp-state-families-author-review.json';
const author=read(packetPath);
const startPath='docs/teaching/evidence/dp-state-families-independent-start.json';
const start=read(startPath);
const nativePath='scratch/dp-state-families-independent/native-results.json';
const browserPath='scratch/dp-state-families-independent/browser/results.json';
const native=read(nativePath),browser=read(browserPath);
assert(native.passed&&browser.passed);
assert.deepEqual(author.sourceHashes,native.sourceHashes);
assert.deepEqual(author.sourceHashes,browser.sourceHashes);
author.sourceHashes.forEach(item=>assert.equal(hash(item.path),item.sha256));
assert.equal(author.sourceHashes.length,7);
const body='src/learn/data/topics/dynamic-programming-states-transitions-optimization.jsx';
const original='docs/teaching/archive/dp-state-families-independent-start/'+body+'.txt';
const before=parse(fs.readFileSync(original,'utf8'),{sourceType:'module',plugins:['jsx']});
const after=parse(fs.readFileSync(body,'utf8'),{sourceType:'module',plugins:['jsx']});
const hintSites=[];
function normalize(value){
  if(Array.isArray(value))return value.map(normalize);
  if(value&&typeof value==='object')return Object.fromEntries(Object.entries(value)
    .filter(([key])=>!['start','end','loc','extra','leadingComments','trailingComments','innerComments'].includes(key))
    .map(([key,child])=>[key,normalize(child)]));
  return value;
}
function reverseHints(node){
  if(!node||typeof node!=='object')return;
  if(node.type==='JSXElement'&&node.openingElement.name.name==='StateFamilyCheckpoint'){
    const hint=node.openingElement.attributes.find(attribute=>attribute.name?.name==='hint');
    assert(hint&&hint.value.value.length>30);
    hintSites.push(hint.value.value);
    node.openingElement.name.name='Checkpoint';
    node.closingElement.name.name='Checkpoint';
    node.openingElement.attributes=node.openingElement.attributes.filter(attribute=>attribute!==hint);
  }
  Object.values(node).forEach(value=>Array.isArray(value)?value.forEach(reverseHints):reverseHints(value));
}
const wrappers=after.program.body.filter(node=>node.type==='FunctionDeclaration'&&node.id.name==='StateFamilyCheckpoint');
assert.equal(wrappers.length,1);
after.program.body=after.program.body.filter(node=>node!==wrappers[0]);
reverseHints(after);
assert.equal(hintSites.length,5);
assert.deepEqual(normalize(after),normalize(before),'Only reviewed wrapper/hints changed from preliminary body');
start.sourceHashes.filter(item=>item.path!==body).forEach(item=>assert.equal(hash(item.path),item.sha256));
const opened=[
  'reading-interval-states-split-the-last-operation-1440','matrix-candidate-320',
  'matrix-postorder-390','balloon-order-320','star-conditional-390',
  'chain-values-1440','padding-then-real-zero-320','real-zero-state-390',
  'nonempty-task-hint-320','reading-tree-states-remember-the-parent-boundary-390',
  'reading-digit-states-count-constrained-continuations-320','nonempty-task-hint-1440',
];
const captures=new Set(browser.records.flatMap(record=>record.images));
assert.deepEqual(browser.records.map(record=>record.width),[1440,390,320]);
assert(browser.records.every(record=>record.hintStates.length===5&&record.anchors.length===3));
assert(opened.every(name=>captures.has(name+'.png')));
const evidenceFiles=[packetPath,startPath,nativePath,browserPath,
  'docs/teaching/evidence/dp-state-families-original.json',
  'docs/teaching/evidence/dp-state-families-preservation.json',
  'scripts/verify-dp-state-families-independent.mjs',
  'scripts/verify-dp-state-families-independent.py',
  'scripts/review-dp-state-families-independent.cjs',
  'scripts/finalize-dp-state-families-independent.mjs',
];
const record={
  reviewedAt:new Date().toISOString(),
  topicId:'dynamic-programming-states-transitions-optimization',
  status:'independent review complete; no unresolved material finding',
  authorFrozenAt:author.frozenAt,
  sourceHashes:author.sourceHashes,
  startingSourceRead:start,
  sourceConservation:{
    unchangedOriginalSupport:native.unchangedOriginalSupport,
    originalProgramsExecuted:16,addedProgramsExecuted:4,originalPracticeObjectsPreserved:12,
    addedPractice:3,
    oldLessonElements:'144 ordered normalized elements retained by inspected author preservation checker; one original inline formula gained spaces, without changing its identifiers/operators.',
    subsequentDifference:'Reversing exactly the five added hint attributes/component names and removing the reviewed wrapper reproduces the entire preliminary normalized body AST. Other six source hashes unchanged.',
  },
  findings:[{
    category:'teaching scaffold',
    initial:'Five new changed reasoning tasks exposed only the full explanation, despite the design promise of hints.',
    resolution:'Author added a topic-owned wrapper with prompt, optional hint and separate hidden explanation; original checkpoints and all numerical code retained.',
    verification:'Complete normalized-AST reverse comparison plus independent keyboard tests of each hint/answer at all three widths, with final screenshots opened.',
    remaining:false,
  }],
  native,
  browser,
  actuallyOpenedImages:opened.map(name=>{
    const file='scratch/dp-state-families-independent/browser/'+name+'.png';
    return {path:file,sha256:hash(file),openedByIndependentReviewer:true};
  }),
  sourceReview:[
    'Complete interval/tree/digit body additions, mechanism proofs, base/reconstruction/cost contracts, all seven changed/new source files, four actual added programs and all new changed practice.',
    'Matrix child replacement preserves fixed shape; last-balloon reconstruction keeps boundaries alive. Counts optimize the declared arithmetic objective, not runtime or numerical accuracy.',
    'Tree decomposition requires one parent edge and no cross-child conflict; parent weight stays outside the subtree, signed weights permit the empty set, and reconstructed choices propagate the actual parent permission.',
    'Digit states distinguish bound equality, padding and used real zero. Prefix groups partition answers, caches are bound-specific, and the inclusive zero/endpoint conventions match native range behavior.',
    'The original source archive and old support/program/practice conservation were inspected separately from the new extension evidence. The old entire numerical suite was not rerun by this reviewer; author regression remains attributed to its author.',
  ],
  freshOfficialResources:[
    {url:'https://leetcode.com/problems/burst-balloons/',scope:'Public statement: ID312, Hard, nonnegative current-neighbor rewards, all removed, 1 sentinels, n<=300.'},
    {url:'https://leetcode.com/problems/house-robber-iii/',scope:'Public statement: ID337, Medium, binary-tree objects, no adjacent selected nodes, up to10000 nodes.'},
    {url:'https://leetcode.com/problems/count-special-integers/',scope:'Public statement: ID2376, Hard, positive all-distinct decimal integers, inclusive n<=2e9.'},
    {url:'https://cses.fi/problemset/task/2220/',scope:'Public statement: adjacent digits unequal, nonnegative inclusive range through1e18. No difficulty label inferred.'},
  ],
  evidenceHashes:Object.fromEntries(evidenceFiles.map(file=>[file,hash(file)])),
  limits:[
    'Finite complementary cases support the read proofs and concrete implementations, not every possible future DP problem.',
    'Browser comparison is a changed subset and final-hint review, not a duplicate claim to have independently rerun the author full639-state suite.',
    'No official judge submission, editorial access or full audiovisual playback is claimed.',
    'Historical author packet and pre-amendment source remain preserved; later wording/layout evidence is bound to its exact final source.',
    'Integrated production build/loading checks and user acceptance belong to separate records.',
  ],
};
const destination='docs/teaching/evidence/dp-state-families-independent-review.json';
assert(!fs.existsSync(destination),'Preserve an existing independent record and append a specific amendment');
fs.writeFileSync(destination,JSON.stringify(record,null,2)+'\n');
console.log(JSON.stringify({reviewedAt:record.reviewedAt,sourceCount:7,openedImages:opened.length,native:native.checkedAt,browser:browser.checkedAt}));

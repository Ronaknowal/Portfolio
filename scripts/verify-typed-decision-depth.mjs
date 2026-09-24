import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import { parse } from '@babel/parser';
import { topicMap } from '../src/learn/data/catalogue.js';
import { encodeRequestTrace, attentionMixture, scoringHeadStep, stableProbabilities } from '../src/learn/data/projects/typed-decision-model/mechanism-models.js';

const directory = 'src/learn/data/projects/typed-decision-model/';
const assets = 'public/learn-projects/typed-decision-model/';
const fixture = JSON.parse(fs.readFileSync(assets + 'trace-fixture.json', 'utf8'));
const checked = [];
function check(name, run) { run(); checked.push(name); }
function close(actual, expected, tolerance = 1e-9) { assert.ok(Math.abs(actual - expected) <= tolerance, `${actual} != ${expected}`); }
const files = ['content.jsx','project-elements.jsx','mechanism-depth.jsx','mechanism-labs.jsx','research-depth.jsx'];
const links = new Set();
const excerpts = [];
function visit(node) {
  if (!node || typeof node !== 'object') return;
  if (node.type === 'JSXOpeningElement' && node.name.name === 'ConceptLinks') {
    const items = node.attributes.find(attribute => attribute.name?.name === 'items').value.expression;
    for (const item of items.elements) {
      const id = item.properties.find(property => property.key.name === 'id').value.value;
      assert.ok(topicMap[id], `Broken concept ${id}`);
      links.add(id);
    }
  }
  if (node.type === 'JSXOpeningElement' && node.name.name === 'Source') {
    const props = Object.fromEntries(node.attributes.map(attribute => [attribute.name.name, attribute.value.value]));
    const source = fs.readFileSync(assets + (props.file || 'typed_decision.py'), 'utf8');
    const start = props.start ? source.indexOf(props.start) : 0;
    const end = props.end ? source.indexOf(props.end, start + props.start.length) : source.length;
    assert.ok(start >= 0 && end > start, `Missing source excerpt ${props.title}`);
    excerpts.push(props.title);
  }
  for (const value of Object.values(node)) if (Array.isArray(value)) value.forEach(visit); else if (value && typeof value === 'object') visit(value);
}
check('JSX, exact concept links and canonical source excerpt boundaries', () => {
  for (const file of files) visit(parse(fs.readFileSync(directory + file, 'utf8'), { sourceType: 'module', plugins: ['jsx'] }));
});
check('Browser encoding agrees exactly with native refund trace', () => {
  const result = encodeRequestTrace(fixture.request, fixture.vocabulary);
  assert.deepEqual(result.cells.map(cell => cell.id), fixture.ids);
  assert.deepEqual(result.markers, fixture.markers);
  assert.equal(result.cells.length, 35);
  assert.equal(result.target, 0);
});
check('Semantic target follows reordering; length and unknown-token limits are explicit', () => {
  const row = {...fixture.request, options: [...fixture.request.options.slice(1), fixture.request.options[0]]};
  assert.equal(encodeRequestTrace(row,fixture.vocabulary).target,2);
  const short = encodeRequestTrace({...fixture.request,options:fixture.request.options.slice(0,2)},fixture.vocabulary);
  assert.equal(short.cells.length,28);
  assert.deepEqual(short.markers,[8,15]);
  assert.ok(encodeRequestTrace({...fixture.request,state:'money left my bank twice'},fixture.vocabulary).unknownCount>0);
  assert.match(encodeRequestTrace({...fixture.request,state:'123'},fixture.vocabulary).error,/a–z/);
  assert.match(encodeRequestTrace({...fixture.request,state:'word '.repeat(150)},fixture.vocabulary).error,/128/);
});
check('Attention mask, normalized rows, zero-query uniform case and weighted-value identity', () => {
  for (const query of [-2,-.7,0,.3,1,2]) {
    for (const mask of [true,false]) {
      const result=attentionMixture(query,mask);
      close(result.probabilities.reduce((sum,value)=>sum+value,0),1);
      if(mask) assert.equal(result.probabilities[3],0);
      for(let column=0;column<2;column++)close(result.output[column],result.contributions.reduce((sum,row)=>sum+row[column],0));
    }
  }
  attentionMixture(0,true).probabilities.slice(0,3).forEach(value=>close(value,1/3));
  assert.ok(attentionMixture(1,false).output[0]>attentionMixture(1,true).output[0]+5);
});
check('Shared scoring-head analytic gradient agrees with central finite differences', () => {
  const epsilon=1e-5;
  for(const weights of [[2,1],[-2,.7],[0,0],[3,-1]])for(const target of [0,1,2]) {
    const result=scoringHeadStep(weights,target,.1);
    for(const coordinate of [0,1]) {
      const plus=weights.map((value,index)=>value+(index===coordinate?epsilon:0));
      const minus=weights.map((value,index)=>value-(index===coordinate?epsilon:0));
      close(result.gradient[coordinate],(scoringHeadStep(plus,target,0).loss-scoringHeadStep(minus,target,0).loss)/(2*epsilon),1e-8);
    }
    assert.ok(result.nextLoss<=result.loss+1e-10);
    assert.deepEqual(scoringHeadStep(weights,target,0).nextWeights,weights);
  }
  close(scoringHeadStep([2,1],0,.5).nextLoss,.26246618357755613);
});
check('Worked lexical and scalar examples match executable arithmetic', () => {
  const p=stableProbabilities([1,0,0]);
  close(-Math.log(p[0]),.5514447139320511);
  const brier=p.reduce((sum,value,index)=>sum+(value-Number(index===0))**2,0);
  close(brier,.2695153416422547,1e-8);
  close(-Math.log(stableProbabilities([2,1,0])[0]),.40760596444438046);
});
const sourceHashes=Object.fromEntries([...files.map(file=>directory+file),directory+'mechanism-models.js',directory+'project.css',assets+'trace-fixture.json'].map(file=>[file,crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex')]));
const report={passed:true,checkedAt:new Date().toISOString(),checks:checked,linkedTopics:[...links],sourceExcerpts:excerpts,sourceHashes};
fs.writeFileSync('docs/teaching/projects/evidence/typed-decision-depth-author.json',JSON.stringify(report,null,2)+'\n');
console.log(`PASS: ${checked.length} substantive groups; ${links.size} resolved concepts; ${excerpts.length} canonical excerpts.`);

import assert from 'node:assert/strict';
import fs from 'node:fs';
import crypto from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { parse } from '@babel/parser';
import { dynamicProgrammingExamples } from '../src/learn/data/dynamic-programming-examples.js';
import practice from '../src/learn/data/practice/dynamic-programming-states-transitions-optimization.js';

const baseline = JSON.parse(fs.readFileSync('docs/teaching/evidence/dp-state-families-original.json', 'utf8'));
const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const bodyPath = 'src/learn/data/topics/dynamic-programming-states-transitions-optimization.jsx';
const oldBody = baseline.files.find(file => file.source === bodyPath);
function rootElement(value) {
  if (!value || typeof value !== 'object') return null;
  if (value.type === 'JSXElement' && value.openingElement.attributes.some(attribute => attribute.name?.name === 'className' && attribute.value?.value?.includes('dynamic-programming-lesson'))) return value;
  for (const child of Object.values(value)) {
    const candidates = Array.isArray(child) ? child : [child];
    for (const candidate of candidates) {
      const result = rootElement(candidate);
      if (result) return result;
    }
  }
  return null;
}
function normalized(value) {
  if (Array.isArray(value)) return value.filter(child => child?.type !== 'JSXText' || child.value.trim()).map(normalized);
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value)
    .filter(([key]) => !['start','end','loc','extra','leadingComments','trailingComments','innerComments'].includes(key))
    .map(([key, child]) => [key, value.type === 'JSXText' && key === 'value' ? child.replace(/\s+/g, ' ').trim() : normalized(child)]));
  return value;
}
const roots = [oldBody.archive, bodyPath].map(path => rootElement(parse(fs.readFileSync(path, 'utf8'), {sourceType: 'module',plugins:['jsx']})));
const children = roots.map(root => root.children.filter(node => node.type !== 'JSXText'));
let position = 0;
let retainedContentElements = 0;
for (const old of children[0]) {
  const name = old.openingElement?.name?.name;
  if (name === 'LessonIntro') {
    const current = children[1].find(node => node.openingElement?.name?.name === name);
    assert.deepEqual(normalized(current.children), normalized(old.children));
    continue;
  }
  if (name === 'Sources') {
    const current = children[1].find(node => node.openingElement?.name?.name === name);
    assert.deepEqual(normalized(current.openingElement), normalized(old.openingElement));
    const oldReferences = old.children.filter(node => node.type !== 'JSXText');
    const currentReferences = current.children.filter(node => node.type !== 'JSXText');
    assert.deepEqual(normalized(currentReferences.slice(0,oldReferences.length)),normalized(oldReferences));
    continue;
  }
  // One original prose formula needed legal line-break opportunities at 320px.
  // Its identifiers, operators and mathematical meaning are preserved.
  const signature = JSON.stringify(normalized(old)).replace('current=max(next_one,reward+next_two)', 'current = max(next_one, reward + next_two)');
  const index = children[1].findIndex((node, index) => index >= position && JSON.stringify(normalized(node)) === signature);
  assert(index >= 0, `Original lesson element not retained: ${name} at ${old.start}`);
  position = index + 1;
  retainedContentElements++;
}
const retainedPaths = [
  'src/learn/data/dynamic-programming-models.js',
  'src/learn/data/dynamic-programming-examples.js',
  'src/learn/components/lesson-labs/DynamicProgrammingLabs.jsx',
  'src/learn/components/lesson-labs/dynamic-programming-labs.css',
];
for (const source of retainedPaths) {
  const previous = baseline.files.find(file => file.source === source);
  assert.equal(hash(fs.readFileSync(source)), previous.sha256, `Unchanged original support: ${source}`);
}
const programs = [];
for (const original of baseline.programs) {
  const example = dynamicProgrammingExamples[original.id];
  assert.equal(example.title, original.title);
  assert.equal(hash(example.code), original.codeSha256);
  assert.equal(example.expected, original.expected);
  const actual = spawnSync('scratch/lesson-tools/Scripts/python.exe', ['-'], {
    input: example.code, encoding: 'utf8', timeout: 15000,
  });
  assert.equal(actual.status, 0, `${original.id}: ${actual.stderr}`);
  assert.equal(actual.stdout.replace(/\r\n/g, '\n').trimEnd(), original.expected);
  programs.push({ id: original.id, stdout: actual.stdout.replace(/\r\n/g, '\n').trimEnd(), codeSha256: original.codeSha256 });
}
const actualProblems = practice.groups.flatMap(group => group.problems.map(problem => ({ group: group.id, ...problem })));
for (const original of baseline.problems) {
  assert.deepEqual(actualProblems.find(problem => problem.number === original.number && problem.group === original.group), original);
}
const result = {
  checkedAt: new Date().toISOString(),
  scope: 'Exact original support/program/practice conservation, fresh execution of 16 original stdout fixtures, and ordered normalized-AST conservation of original lesson elements, allowing one documented original prose formula spacing repair. Intro prerequisite/anchor attributes and read time are intentionally updated; intro teaching, original references and alternate-resource contents are retained. Line endings/source locations/formatting whitespace are normalized. No browser claim.',
  retainedContentElements,
  originalReadingAmendment: { before: 'current=max(next_one,reward+next_two)', after: 'current = max(next_one, reward + next_two)', reason: 'Actual 320px text-range measurement showed the original unbroken formula extended to x=349.1875. Natural spaces provide legal breaks without changing operators, identifiers or meaning.' },
  retainedSupport: retainedPaths.map(source => ({ source, sha256: hash(fs.readFileSync(source)) })),
  programs, retainedPractice: baseline.problems.length,
};
fs.writeFileSync('docs/teaching/evidence/dp-state-families-preservation.json', JSON.stringify(result, null, 2) + '\n');
console.log(`Preserved and executed ${programs.length} original programs; retained ${baseline.problems.length} original practice entries and four support-file hashes.`);

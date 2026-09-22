import assert from 'node:assert/strict';
import fs from 'node:fs';
import { parse } from '@babel/parser';
import { projects } from '../src/learn/data/projects/catalogue.js';
import { loadProject } from '../src/learn/data/projects/loader.js';
import { topicMap } from '../src/learn/data/catalogue.js';

const ids = new Set();
let stages = 0;
const loader = fs.readFileSync('src/learn/data/projects/loader.js', 'utf8');
for (const project of projects) {
  assert.match(project.id, /^[a-z][a-z0-9-]+$/);
  assert.ok(!ids.has(project.id), `Duplicate project ID: ${project.id}`);
  ids.add(project.id);
  for (const field of ['title', 'description', 'kind', 'level', 'status', 'outcome']) {
    assert.ok(typeof project[field] === 'string' && project[field].trim(), `${project.id}: missing ${field}`);
  }
  assert.ok(project.stages.length > 0, `${project.id}: no stages`);
  assert.ok(loader.includes(`./${project.id}/content.jsx`), `${project.id}: missing dynamic import`);
  for (const topicId of [...project.prerequisiteIds, ...(project.relatedTopicIds || [])]) {
    assert.ok(Object.hasOwn(topicMap, topicId), `${project.id}: unknown topic ${topicId}`);
  }
  const stageIds = project.stages.map(stage => stage.id);
  assert.equal(new Set(stageIds).size, stageIds.length, `${project.id}: duplicate stage`);
  for (const stage of project.stages) {
    assert.match(stage.id, /^[a-z][a-z0-9-]*$/);
    for (const field of ['title', 'summary', 'deliverable']) assert.ok(stage[field]?.trim(), `${stage.id}: missing ${field}`);
  }
  const metadata = fs.readFileSync(`src/learn/data/projects/${project.id}/metadata.js`, 'utf8');
  const metadataAst = parse(metadata, { sourceType: 'module' });
  assert.equal(metadataAst.program.body.some(node => node.type === 'ImportDeclaration'), false, 'Metadata must not import project bodies');
  const source = fs.readFileSync(`src/learn/data/projects/${project.id}/content.jsx`, 'utf8');
  const ast = parse(source, { sourceType: 'module', plugins: ['jsx'] });
  const exported = ast.program.body.find(node => node.type === 'ExportDefaultDeclaration')?.declaration;
  assert.equal(exported?.type, 'ObjectExpression', 'Use an explicit stage-to-component default export');
  const exportedIds = exported.properties.map(property => property.key.name || property.key.value);
  assert.deepEqual(exportedIds, stageIds, `${project.id}: stage map/order differs from metadata`);
  assert.ok(fs.existsSync(`docs/teaching/projects/${project.id}.md`), `${project.id}: missing author handoff`);
  stages += stageIds.length;
}
await assert.rejects(loadProject('__proto__'), /Unknown project/);
await assert.rejects(loadProject('not-a-project'), /Unknown project/);
console.log(`PASS: ${projects.length} project registry, ${stages} ordered stages, all topic links, metadata import isolation and unknown-ID rejection.`);

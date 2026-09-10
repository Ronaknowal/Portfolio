import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { tracks } from './lib/authoring-curriculum.mjs';
import { topicCatalogue } from '../src/learn/data/curriculum/topic-catalogue.js';

// This is the active rollout's conservation/evidence check. Earlier dated
// increment snapshots remain unchanged and are not widened to fit new work.
const readJson = filename => JSON.parse(fs.readFileSync(filename, 'utf8'));
const progress = readJson('docs/teaching/dsa-math-foundations-progress.json');
const baseline = readJson(progress.baseline);
const publications = readJson('src/learn/data/lesson-manifest.json');
const hashFile = filename => createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
const expectedScope = baseline.modules.flatMap(module => module.sections.flatMap(section => section.topics)
  .slice(module.id === 'data-structures-algorithms' ? 5 : 0)
  .map(topic => topic.id));
assert.deepEqual(progress.scope, { remainingDsa: 17, allMathFoundations: 57, total: 74 });
assert.equal(new Set(progress.topics.map(topic => topic.id)).size, 74);
assert.deepEqual(progress.topics.map(topic => topic.id).sort(), expectedScope.slice().sort(), 'Scoped topics were removed, replaced or duplicated');
assert.deepEqual(Object.keys(topicCatalogue).sort(), baseline.allTopicIds.slice().sort(), 'Catalogue identities changed without a scope review');
for (const module of baseline.modules) {
  const previousIds = module.sections.flatMap(section => section.topics.map(topic => topic.id));
  const current = tracks.find(track => track.id === module.id);
  assert.deepEqual(current.topicIds, previousIds, `${module.id}: module reading order or membership changed`);
  for (const entry of progress.topics.filter(topic => topic.moduleId === module.id)) {
    assert.equal(entry.modulePosition, previousIds.indexOf(entry.id) + 1);
  }
}
for (const [id, source] of Object.entries(baseline.publicationManifest)) {
  assert.equal(publications[id], source, `Prior publication removed or remapped: ${id}`);
}
const newPublications = Object.keys(publications).filter(id => !Object.hasOwn(baseline.publicationManifest, id));
for (const id of newPublications) assert.ok(expectedScope.includes(id), `Unscoped publication: ${id}`);

const allowedStatuses = new Set(['pending', 'in-progress', 'author-verified', 'implementation-reviewed']);
for (const topic of progress.topics) {
  assert.ok(allowedStatuses.has(topic.status), `${topic.id}: unknown evidence state`);
  if (!['author-verified', 'implementation-reviewed'].includes(topic.status)) continue;
  assert.ok(publications[topic.id], `${topic.id}: verified lesson is not published`);
  assert.equal(topic.source, publications[topic.id]);
  const source = path.join('src/learn/data', topic.source);
  assert.equal(hashFile(source), topic.reviewedSourceSha256, `${topic.id}: source changed since review`);
  for (const field of ['designRecord', 'verificationRecord']) {
    assert.ok(topic[field] && fs.statSync(topic[field]).size > 500, `${topic.id}: missing ${field}`);
  }
  assert.ok(topic.reviewedFiles && Object.keys(topic.reviewedFiles).length > 0, `${topic.id}: missing owned-source hashes`);
  for (const [filename, expectedHash] of Object.entries(topic.reviewedFiles)) {
    assert.equal(hashFile(filename), expectedHash, `${topic.id}: reviewed dependency changed: ${filename}`);
  }
  if (topic.status === 'implementation-reviewed') {
    assert.ok(topic.integrationRecord && fs.statSync(topic.integrationRecord).size > 500, `${topic.id}: missing integration record`);
  }
  if (topic.moduleId === 'data-structures-algorithms') {
    const practice = (await import(`../src/learn/data/practice/${topic.id}.js`)).default;
    assert.equal(practice.topicId, topic.id);
    assert.ok(practice.verifiedOn);
    assert.ok(practice.groups.length > 0);
    for (const group of practice.groups) for (const problem of group.problems) {
      for (const field of ['number', 'title', 'slug', 'difficulty', 'focus', 'hint', 'transfer']) {
        assert.ok(problem[field], `${topic.id}: incomplete problem ${problem.number}: ${field}`);
      }
      assert.match(problem.slug, /^[a-z0-9]+(?:-[a-z0-9]+)*$/);
    }
  }
}
if (process.argv.includes('--complete')) {
  assert.ok(progress.topics.every(topic => topic.status === 'implementation-reviewed'), 'The full 74-topic goal is not complete');
}
const counts = Object.fromEntries([...allowedStatuses].map(status => [status, progress.topics.filter(topic => topic.status === status).length]));
console.log(JSON.stringify({ preservedCatalogueIds: baseline.allTopicIds.length, preservedPublications: Object.keys(baseline.publicationManifest).length,
  newPublications, scope: progress.scope, evidenceStates: counts, complete: counts['implementation-reviewed'] === 74 }, null, 2));

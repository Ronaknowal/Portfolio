import assert from 'node:assert/strict';
import fs from 'node:fs';
import { tracks } from './lib/authoring-curriculum.mjs';

// This increment's baseline is captured before its first publication, not from
// HEAD: the user's substantial existing uncommitted publication work is retained.
const beforePath = 'docs/teaching/evidence/dsa-publication-before.json';
assert.ok(fs.existsSync(beforePath), 'Missing pre-increment publication snapshot');
const before = JSON.parse(fs.readFileSync(beforePath, 'utf8'));
const after = JSON.parse(fs.readFileSync('src/learn/data/lesson-manifest.json', 'utf8'));
const newTopics = ['trees-binary-search-trees', 'heaps-priority-queues-tries', 'graphs-representations-bfs-dfs'];
for (const [id, source] of Object.entries(before)) assert.equal(after[id], source, `Prior publication changed or removed: ${id}`);
assert.deepEqual(Object.keys(after).filter(id => !Object.hasOwn(before, id)).sort(), [...newTopics].sort());
const firstFive = ['arrays-strings-hash-maps', 'linked-lists-stacks-queues', ...newTopics];
const dsa = tracks.find(track => track.id === 'data-structures-algorithms');
assert.deepEqual(dsa.topicIds.slice(0, 5), firstFive, 'DSA reading order changed');
const seenProblems = new Set();
for (const id of firstFive) {
  const practice = (await import(`../src/learn/data/practice/${id}.js`)).default;
  assert.equal(practice.topicId, id);
  const lesson = fs.readFileSync(`src/learn/data/${after[id].slice(2)}`, 'utf8');
  assert.ok(lesson.includes(`../practice/${id}.js`), `${id}: missing scoped practice import`);
  assert.ok(lesson.includes('<DsaPractice practice='), `${id}: practice not rendered`);
  assert.ok(lesson.includes('guided-dsa-practice'), `${id}: missing route anchor`);
  for (const group of practice.groups) for (const problem of group.problems) {
    for (const key of ['number', 'title', 'slug', 'difficulty', 'focus', 'hint', 'transfer']) assert.ok(problem[key], `${id}: missing ${key}`);
    assert.match(problem.slug, /^[a-z0-9]+(?:-[a-z0-9]+)*$/);
    assert.ok(['Easy', 'Medium', 'Hard'].includes(problem.difficulty));
    assert.ok(!seenProblems.has(problem.number), `Duplicate selected problem ${problem.number}; assign an owner or explicitly review overlap`);
    seenProblems.add(problem.number);
  }
}
console.log(`Preserved all ${Object.keys(before).length} prior publications; added exactly three in DSA module order. Verified ${seenProblems.size} curated practice entries across the five published DSA lessons.`);

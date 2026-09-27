import fs from 'node:fs';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { parseLesson, walkJsx, attributeValue, endingLayoutOf, preserveLessonEndingLayout, groupPracticeExercises, applyEndingLayout } from './lib/lesson-ending-layout.mjs';
import { renderPreparedLesson } from './lib/prepared-lesson-renderer.mjs';
import { filesMatch } from './lib/lesson-delivery.mjs';

const directory = 'docs/teaching/lesson-endings';
const hash = value => createHash('sha256').update(value).digest('hex');
// Erase ONLY the new transparent layout containers; everything inside remains.
// Flatten at JSX child-array boundaries rather than serializing away expressions.
function semanticTree(value) {
  if (value === null || typeof value !== 'object') return value;
  if (Array.isArray(value)) return value.flatMap(item => {
    if (item?.type === 'JSXElement' && (attributeValue(item, 'data-lesson-ending') || attributeValue(item, 'data-lesson-exercise') === '')) {
      assert.ok(item.openingElement.attributes.every(a => ['className', 'data-lesson-ending', 'data-lesson-exercise', 'data-lesson-resource-list'].includes(a.name?.name)), 'Unreviewed wrapper attribute');
      return semanticTree(item.children);
    }
    const result = semanticTree(item);
    return result === undefined ? [] : [result];
  });
  if (value.type === 'JSXText' && !value.value.trim()) return undefined;
  return Object.fromEntries(Object.entries(value)
    .filter(([key]) => !['start', 'end', 'loc', 'extra', 'comments', 'leadingComments', 'trailingComments', 'innerComments', 'tokens'].includes(key))
    .map(([key, item]) => [key, semanticTree(item)]));
}
const semanticHash = source => hash(JSON.stringify(semanticTree(parseLesson(source))));
// Read-only regression check. Never refresh a source-bound baseline or receipt
// merely because the current implementation differs from its reviewed state.
const baseline = JSON.parse(fs.readFileSync(`${directory}/baseline.json`));
assert.deepEqual(JSON.parse(fs.readFileSync('src/learn/data/lesson-manifest.json')), baseline.manifest, 'Publication mappings changed');
const checks = [];
for (const [file, before] of Object.entries(baseline.files)) {
  const source = fs.readFileSync(file, 'utf8');
  assert.equal(semanticHash(source), before.semanticSha256, `Lesson content changed: ${file}`);
  assert.ok(source.includes('data-lesson-ending='), `Missing ending structure: ${file}`);
  walkJsx(parseLesson(source), node => {
    if (attributeValue(node, 'data-lesson-ending')) assert.ok(node.children.some(child => child.type === 'JSXElement'), `Empty ending container: ${file}`);
  });
  assert.equal(groupPracticeExercises(source), source, `Exercise grouping is not idempotent: ${file}`);
  checks.push({ file, before: before.sha256, after: hash(fs.readFileSync(file)), semanticSha256: before.semanticSha256 });
}
// Exact-title registration survives generation; ambiguity or missing headings
// fails rather than silently discarding reviewed presentation.
const manuscript = '# Example\n\n## Practice\n\n### Change the input\n\nWork it out.\n\n## Next\n\nContinue.\n\n## References\n\n- [Docs](https://example.org)\n';
const endings = [{ level: 'H2', title: 'Practice', kind: 'practice' }, { level: 'H2', title: 'Next', kind: 'next' }, { level: 'H2', title: 'References', kind: 'resources' }];
const rendered = renderPreparedLesson(manuscript, { endings }).jsx;
assert.deepEqual(endingLayoutOf(`const lesson=<>${rendered}</>;`), endings);
assert.match(rendered, /data-lesson-exercise/);
assert.equal(preserveLessonEndingLayout(rendered, undefined, endings), rendered);
assert.throws(() => renderPreparedLesson(manuscript, { endings: [{ level: 'H2', title: 'Missing', kind: 'practice' }] }), /one exact heading/);
assert.throws(() => renderPreparedLesson(manuscript + '\n## Practice\n\nDuplicate\n', { endings }), /one exact heading/);
const adjacent = applyEndingLayout('const lesson=<><H2>Practice</H2><Prose>Task</Prose><H2>Next</H2><Prose>Link</Prose></>;', endings.slice(0,2));
assert.deepEqual(endingLayoutOf(adjacent), endings.slice(0,2), 'Adjacent JSX sections lost their headings');
const subheading = groupPracticeExercises(applyEndingLayout('const lesson=<><H3>Independent practice</H3><Practice/><Practice/></>;', [{ level:'H3', title:'Independent practice', kind:'practice' }]));
assert.equal((subheading.match(/data-lesson-exercise=/g) || []).length, 2, 'Practice section heading swallowed individual exercises');
assert.throws(() => preserveLessonEndingLayout(rendered, 'src/learn/data/topics/numerical-pdes-grids-finite-elements-stability.jsx'), /custom ending range explicitly/);
const receiptPath = `${directory}/presentation-review.json`;
if (fs.existsSync(receiptPath)) {
  const receipt = JSON.parse(fs.readFileSync(receiptPath));
  const ledger = JSON.parse(fs.readFileSync('docs/teaching/lesson-delivery-progress.json'));
  assert.equal(receipt.passed, true);
  assert.ok(filesMatch(process.cwd(), receipt.reviewedFiles), 'Reviewed presentation source changed');
  assert.ok(filesMatch(process.cwd(), receipt.evidenceHashes), 'Presentation evidence changed');
  assert.equal(hash(fs.readFileSync(receipt.previousReview)), receipt.previousReviewSha256, 'Historical presentation receipt was altered');
  for (const id of receipt.preservedCurrentTopics) {
    assert.ok(filesMatch(process.cwd(), ledger.topics[id].content.files), `Stale content: ${id}`);
    assert.ok(filesMatch(process.cwd(), ledger.topics[id].implementation.reviewedFiles), `Stale implementation: ${id}`);
  }
  for (const [id, expected] of Object.entries(receipt.rowHashesBefore)) {
    const row = structuredClone(ledger.topics[id]), delta = receipt.checkpointDeltas[id];
    if (delta) {
      assert.equal(row.implementation.presentationReviews.pop(), receiptPath);
      for (const [phase, key, changes] of [['content', 'files', delta.contentFiles], ['implementation', 'reviewedFiles', delta.implementationFiles]]) {
        for (const [file, before] of Object.entries(changes)) {
          if (before === null) delete row[phase][key][file];
          else row[phase][key][file] = before;
        }
      }
    }
    assert.equal(hash(JSON.stringify(row)), expected, `Unreviewed ledger change: ${id}`);
  }
}
console.log(JSON.stringify({ passed: true, publishedLessons: checks.length, authoringFixtures: 8 }));

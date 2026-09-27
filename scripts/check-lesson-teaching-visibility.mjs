import fs from 'node:fs';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { parseTeachingJsx, teachingTag, teachingAttribute, teachingDisclosures, preserveVisibleTeachingSections, visibleTeachingEntries } from './lib/lesson-teaching-disclosures.mjs';
import { renderPreparedLesson } from './lib/prepared-lesson-renderer.mjs';

export const sourceHash = source => createHash('sha256').update(source).digest('hex');
function removeClass(node, token) {
  node.openingElement.attributes = node.openingElement.attributes.filter(attribute => {
    if (attribute.name?.name !== 'className') return true;
    assert.equal(attribute.value?.type, 'StringLiteral');
    attribute.value.value = attribute.value.value.split(/\s+/).filter(value => value !== token).join(' ');
    return Boolean(attribute.value.value);
  });
}
// Reverse only the declared presentation substitution; all original JSX,
// expressions, attributes, content ordering and practice feedback still compare.
export function teachingSemanticTree(value) {
  if (value === null || typeof value !== 'object') return value;
  if (Array.isArray(value)) return value.map(teachingSemanticTree);
  if (value.type === 'JSXElement' && teachingAttribute(value, 'data-lesson-teaching')) {
    assert.equal(teachingTag(value), 'section');
    value.openingElement.name.name = value.closingElement.name.name = 'details';
    value.openingElement.attributes = value.openingElement.attributes.filter(attribute => attribute.name?.name !== 'data-lesson-teaching');
    removeClass(value, 'lesson-teaching-section');
    const headings = value.children.filter(child => teachingAttribute(child, 'className')?.value?.value?.split(/\s+/).includes('lesson-teaching-section__title'));
    assert.equal(headings.length, 1, 'Visible teaching needs one preserved title');
    headings[0].openingElement.name.name = headings[0].closingElement.name.name = 'summary';
    removeClass(headings[0], 'lesson-teaching-section__title');
  }
  return Object.fromEntries(Object.entries(value)
    .filter(([key]) => !['start', 'end', 'loc', 'extra'].includes(key))
    .map(([key, child]) => [key, teachingSemanticTree(child)]));
}
export const teachingSemanticHash = source => sourceHash(JSON.stringify(teachingSemanticTree(parseTeachingJsx(source))));

if (process.argv[1]?.replaceAll('\\', '/').endsWith('/check-lesson-teaching-visibility.mjs')) {
  const path = 'docs/teaching/lesson-code-access/topic-disclosures.json';
  const inventory = JSON.parse(fs.readFileSync(path, 'utf8'));
  assert.deepEqual(JSON.parse(fs.readFileSync('src/learn/data/lesson-manifest.json', 'utf8')), inventory.manifest, 'Published mappings changed');
  let converted = 0, kept = 0;
  const checked = [];
  for (const topic of inventory.topics) {
    const source = fs.readFileSync(topic.file, 'utf8');
    assert.equal(teachingSemanticHash(source), topic.semanticSha256, `Authored content changed: ${topic.file}`);
    const practice = topic.disclosures.filter(item => item.action === 'preserve-practice');
    assert.deepEqual(teachingDisclosures(source).map(node => sourceHash(source.slice(node.start, node.end))), practice.map(item => item.sha256), `Practice disclosures changed: ${topic.file}`);
    converted += topic.disclosures.length - practice.length;
    kept += practice.length;
    checked.push({ file: topic.file, before: topic.sha256, after: sourceHash(source), semanticSha256: topic.semanticSha256 });
  }
  const fixture = '<details id="theory" className="local"><summary id="title">Derivation</summary><Prose>Visible proof</Prose><details><summary>Hint</summary><Prose>Practice feedback</Prose></details></details>';
  const visible = preserveVisibleTeachingSections(fixture, undefined, [{ summary: 'Derivation' }]);
  assert.ok(visible.includes('data-lesson-teaching=""'));
  assert.ok(visible.includes('<h3 id="title"'));
  assert.ok(visible.includes('<details><summary>Hint</summary>'));
  assert.equal(preserveVisibleTeachingSections(visible, undefined, [{ summary: 'Derivation' }]), visible);
  assert.throws(() => preserveVisibleTeachingSections(fixture, undefined, [{ summary: 'Missing' }]), /one exact summary/);
  assert.throws(() => preserveVisibleTeachingSections(fixture + fixture, undefined, [{ summary: 'Derivation' }]), /one exact summary/);
  assert.throws(() => preserveVisibleTeachingSections('<details onToggle={toggle}><summary>Dynamic</summary></details>', undefined, [{ summary: 'Dynamic' }]), /stateful disclosure/);
  assert.throws(() => visibleTeachingEntries('const lesson=<section data-lesson-teaching=""><h3 className="lesson-teaching-section__title">{title}</h3>{children}</section>'), /dynamic teaching section explicitly/);
  const prepared = renderPreparedLesson('# Example\n\n<details><summary>Derivation</summary>\n\nVisible proof.\n\n</details>\n', { teaching: [{ summary: 'Derivation' }] }).jsx;
  assert.ok(prepared.includes('data-lesson-teaching=""') && !prepared.includes('<details'));
  const result = { passed: true, checkedAt: new Date().toISOString(), topics: checked.length, teachingDisclosuresRevealed: converted, practiceDisclosuresBytePreserved: kept, conservation: checked, scope: 'Presentation-only native disclosure conversion; no scientific execution repeated' };
  if (process.argv.includes('--record')) fs.writeFileSync('docs/teaching/lesson-code-access/topic-conservation.json', JSON.stringify(result, null, 2) + '\n');
  console.log(JSON.stringify({ passed: true, topics: checked.length, teachingDisclosuresRevealed: converted, practiceDisclosuresBytePreserved: kept, fixtures: 9 }));
}

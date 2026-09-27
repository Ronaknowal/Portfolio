import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { parse, parseExpression } from '@babel/parser';
import { renderPreparedLesson } from './lib/prepared-lesson-renderer.mjs';
import { preserveLessonOpeningAnnotations } from './lib/lesson-opening-annotations.mjs';

function annotatedParagraphs(source, fragment = false) {
  const nodes = [];
  function visit(node) {
    if (!node || typeof node !== 'object') return;
    if (node.type === 'JSXElement' && node.openingElement.name.name === 'Prose') {
      const attribute = node.openingElement.attributes.find(item => item.type === 'JSXAttribute' && item.name.name === 'opening');
      if (attribute) nodes.push({ node, attribute });
    }
    for (const [key, value] of Object.entries(node)) {
      if (['loc', 'start', 'end', 'extra', 'comments'].includes(key)) continue;
      if (Array.isArray(value)) value.forEach(visit);
      else if (value && typeof value === 'object') visit(value);
    }
  }
  visit(fragment ? parseExpression(`<>${source}</>`, { plugins: ['jsx'] }) : parse(source, { sourceType: 'module', plugins: ['jsx'] }));
  return nodes;
}

const directory = mkdtempSync(join(tmpdir(), 'lesson-opening-authoring-'));
let cases = 0, topics = 0, paragraphs = 0;
try {
  const sourcePath = join(directory, 'lesson.jsx');
  const fixture = `export default {content:()=> <div>
    <Prose opening="route"><strong>First pass:</strong> Read <a href="/learn/topic/example">one example</a>.</Prose>
    <Prose opening="exploration">Change <code>x</code> and inspect the result.</Prose>
  </div>};`;
  writeFileSync(sourcePath, fixture);
  const manuscript = '**First pass:** Read [one example](/learn/topic/example).\n\nChange `x` and inspect the result.\n\n## 3. Keep the authored number\n\nFirst pass is a phrase in this explanation.';
  const rendered = renderPreparedLesson(manuscript, { preserveOpeningFrom: sourcePath });
  assert.equal(annotatedParagraphs(rendered.jsx, true).length, 2);
  assert.match(rendered.jsx, /<Prose opening="route">/);
  assert.match(rendered.jsx, /<Prose opening="exploration">/);
  assert.match(rendered.jsx, /href={"\/learn\/topic\/example"}/);
  assert.deepEqual(rendered.sections, [['3-keep-the-authored-number', '3. Keep the authored number']]);
  assert.match(rendered.jsx, /<Prose>{"First pass is a phrase in this explanation\."}<\/Prose>/);
  cases++;

  assert.throws(() => renderPreparedLesson(manuscript.replace('one example', 'a different example'), { preserveOpeningFrom: sourcePath }), /found 0/); cases++;
  assert.throws(() => renderPreparedLesson(manuscript + '\n\nChange `x` and inspect the result.', { preserveOpeningFrom: sourcePath }), /found 2/); cases++;
  assert.throws(() => renderPreparedLesson(manuscript, { preserveOpeningFrom: sourcePath, opening: [['**First pass:**', 'summary']] }), /Conflicting opening/); cases++;
  const explicit = renderPreparedLesson('**Your chosen route:** Work the example.', { opening: [['**Your chosen route:**', 'route']] });
  assert.match(explicit.jsx, /<Prose opening="route">/); cases++;
  assert.throws(() => renderPreparedLesson('Ordinary prose.', { opening: [['Absent.', 'route']] }), /found 0/); cases++;
  assert.throws(() => renderPreparedLesson('Route.\n\nRoute.', { opening: [['Route.', 'route']] }), /found 2/); cases++;
  assert.throws(() => renderPreparedLesson('Route.', { opening: [['Route', 'route'], ['Route.', 'summary']] }), /Ambiguous opening/); cases++;
  assert.throws(() => renderPreparedLesson('Route.', { opening: [['Route.', 'unknown']] }), /Invalid/); cases++;
  assert.throws(() => renderPreparedLesson('Route.', { opening: [['Route.', 'route']], replacements: [['Route.', '<Example />']] }), /Ambiguous opening/); cases++;
  assert.equal(preserveLessonOpeningAnnotations('<Prose>New lesson.</Prose>', join(directory, 'new.jsx')), '<Prose>New lesson.</Prose>'); cases++;
  writeFileSync(sourcePath, 'export default () => <Prose opening="route">{dynamicGuidance}</Prose>;');
  assert.throws(() => preserveLessonOpeningAnnotations('<Prose>Guidance.</Prose>', sourcePath), /supported static kind and paragraph/); cases++;

  // Exercise every current explicit annotation without invoking a generator,
  // copying public assets, or changing any published lesson or manuscript.
  const manifest = JSON.parse(readFileSync('src/learn/data/lesson-manifest.json', 'utf8'));
  for (const source of Object.values(manifest)) {
    const sourcePath = join('src/learn/data', source);
    const code = readFileSync(sourcePath, 'utf8');
    const annotated = annotatedParagraphs(code);
    if (!annotated.length) continue;
    const jsx = annotated.map(({ node, attribute }) => code.slice(node.start, attribute.start) + code.slice(attribute.end, node.end)).join('\n');
    const recovered = preserveLessonOpeningAnnotations(jsx, sourcePath);
    const restored = annotatedParagraphs(recovered, true);
    assert.equal(restored.length, annotated.length, sourcePath);
    assert.deepEqual(restored.map(item => item.attribute.value.value), annotated.map(item => item.attribute.value.value), sourcePath);
    topics++; paragraphs += annotated.length;
  }
  console.log(JSON.stringify({ fixtureCases: cases, currentTopics: topics, currentOpeningParagraphs: paragraphs, generatorsExecuted: 0, passed: true }));
} finally {
  // Only remove the exact temp directory created above, after resolving it.
  const resolved = realpathSync(directory);
  if (dirname(resolved) !== realpathSync(tmpdir())) throw new Error(`Unexpected temporary directory parent: ${resolved}`);
  rmSync(resolved, { recursive: true });
}

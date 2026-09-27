// Authoring-time layout only. Explicit titles, never a browser prose classifier.
import fs from 'node:fs';
import { parse } from '@babel/parser';

export const endingKinds = new Set(['practice', 'next', 'resources', 'further-learning', 'references']);
export const elementName = node => node?.openingElement?.name?.name;
export const attributeValue = (node, name) => node?.openingElement?.attributes.find(a => a.name?.name === name)?.value?.value;
export function walkJsx(node, visit) {
  if (!node || typeof node !== 'object') return;
  if (node.type) visit(node);
  for (const [key, value] of Object.entries(node)) {
    if (['loc', 'extra', 'comments'].includes(key)) continue;
    if (Array.isArray(value)) value.forEach(child => walkJsx(child, visit));
    else if (value && typeof value === 'object') walkJsx(value, visit);
  }
}
export function parseLesson(source) {
  return parse(source, { sourceType: 'module', plugins: ['jsx'] });
}
export function textOf(node, bindings = {}) {
  if (!node) return '';
  if (['JSXText', 'StringLiteral', 'NumericLiteral'].includes(node.type)) return String(node.value);
  if (node.type === 'JSXExpressionContainer') return textOf(node.expression, bindings);
  if (node.type === 'MemberExpression' && node.computed && bindings[node.object.name]) return textOf(bindings[node.object.name][node.property.value], bindings);
  if (node.type === 'JSXElement' || node.type === 'JSXFragment') return node.children.map(child => textOf(child, bindings)).join('').replace(/\s+/g, ' ').trim();
  return '';
}
export function lessonBindings(ast) {
  const bindings = {};
  walkJsx(ast, node => {
    if (node.type === 'VariableDeclarator' && node.init?.type === 'ArrayExpression') bindings[node.id.name] = node.init.elements;
  });
  return bindings;
}

// Entries identify an authored H2/H3 and its semantic range. Other headings and
// source/practice components are boundaries, so unlike a tail wrapper this does
// not absorb references or next steps into an exercise.
export function applyEndingLayout(source, entries) {
  const ast = parseLesson(source), bindings = lessonBindings(ast), ranges = [];
  for (const entry of entries) {
    if (!endingKinds.has(entry.kind) || !['H2', 'H3'].includes(entry.level) || !entry.title) throw new Error('Invalid ending layout entry');
    const matches = [];
    walkJsx(ast, parent => {
      if (!parent.children) return;
      parent.children.forEach((node, index) => {
        if (elementName(node) === entry.level && textOf(node, bindings) === entry.title) matches.push({ parent, node, index });
      });
    });
    if (matches.length !== 1) throw new Error(`Ending layout needs one exact heading: ${entry.title}; found ${matches.length}`);
    const { parent, node, index } = matches[0];
    if (attributeValue(parent, 'data-lesson-ending')) continue;
    const siblings = parent.children;
    let end = node.end;
    for (let i = index + 1; i < siblings.length; i++) {
      const next = siblings[i], tag = elementName(next);
      if (tag === 'H2' || (entry.level === 'H3' && tag === 'H3') || ['Sources', 'DsaPractice'].includes(tag)) break;
      // A separately registered subheading ends the preceding practice range.
      if (tag === 'H3' && entries.some(e => e.level === 'H3' && e.title === textOf(next, bindings))) break;
      if (next.type !== 'JSXText' || next.value.trim()) end = next.end;
    }
    ranges.push({ start: node.start, end, open: `<section className="lesson-ending lesson-ending--${entry.kind}" data-lesson-ending="${entry.kind}"${entry.resourceList ? ' data-lesson-resource-list=""' : ''}>`, close: '</section>' });
  }
  return insertRanges(source, ranges);
}

export function insertRanges(source, ranges) {
  const edits = [];
  for (const range of ranges) {
    edits.push({ at: range.start, text: range.open, order: 0 });
    edits.push({ at: range.end, text: range.close, order: 1 });
  }
  // Adjacent ranges may share an offset (generated JSX need not have whitespace).
  // Insert the new opening first so the previous closing ends up before it.
  for (const edit of edits.sort((a, b) => b.at - a.at || a.order - b.order)) source = source.slice(0, edit.at) + edit.text + source.slice(edit.at);
  parseLesson(source); // Refuse overlapping/invalid edits before a caller writes.
  return source;
}

export function endingLayoutOf(source) {
  const ast = parseLesson(source), bindings = lessonBindings(ast), entries = [];
  walkJsx(ast, node => {
    const kind = attributeValue(node, 'data-lesson-ending');
    if (!kind) return;
    const heading = node.children.find(child => ['H2', 'H3'].includes(elementName(child)));
    if (heading) entries.push({ title: textOf(heading, bindings), level: elementName(heading), kind,
      ...(attributeValue(node, 'data-lesson-resource-list') === '' ? { resourceList: true } : {}) });
  });
  return entries;
}

export function groupPracticeExercises(source) {
  const ast = parseLesson(source), ranges = [];
  walkJsx(ast, parent => {
    if (attributeValue(parent, 'data-lesson-ending') !== 'practice') return;
    const children = parent.children;
    for (let i = 0; i < children.length; i++) {
      const node = children[i], tag = elementName(node);
      if (tag === 'H3' && node === children.find(child => child.type === 'JSXElement')) continue;
      if (tag !== 'H3' && !/^(?:Practice|Exercise|ConcentrationExercise|FamilyExercise|CausalExercise|StateFamilyCheckpoint|PracticeTask)$/.test(tag || '')) continue;
      let end = node.end;
      if (tag === 'H3') {
        for (let j = i + 1; j < children.length; j++) {
          const next = children[j];
          if (['H2', 'H3'].includes(elementName(next)) || attributeValue(next, 'data-lesson-ending')) break;
          if (next.type !== 'JSXText' || next.value.trim()) end = next.end;
          i = j;
        }
      }
      ranges.push({ start: node.start, end, open: '<div className="lesson-exercise" data-lesson-exercise="">', close: '</div>' });
    }
  });
  return insertRanges(source, ranges);
}

export function preserveLessonEndingLayout(jsx, sourcePath, explicit = []) {
  let entries = explicit;
  if (sourcePath && fs.existsSync(sourcePath)) {
    const source = fs.readFileSync(sourcePath, 'utf8');
    walkJsx(parseLesson(source), node => {
      if (attributeValue(node, 'data-lesson-ending') && !node.children.some(child => ['H2', 'H3'].includes(elementName(child)))) {
        throw new Error(`Preserve the custom ending range explicitly before regenerating ${sourcePath}; no static H2/H3 identifies this range.`);
      }
    });
    const previous = endingLayoutOf(source);
    entries = [...previous, ...explicit.filter(item => !previous.some(old => old.title === item.title && old.level === item.level))];
    for (const item of explicit) {
      if (previous.some(old => old.title === item.title && old.kind !== item.kind)) throw new Error(`Conflicting ending purpose: ${item.title}`);
    }
  }
  if (!entries.length) return jsx;
  return groupPracticeExercises(applyEndingLayout(`const lesson = <>${jsx}</>;`, entries)).slice('const lesson = <>'.length, -4);
}

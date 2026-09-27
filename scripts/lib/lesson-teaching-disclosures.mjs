// Authoring-only source transforms. The browser receives ordinary visible JSX.
import fs from 'node:fs';
import { parse } from '@babel/parser';

export const teachingTag = node => node?.openingElement?.name?.name;
export const teachingAttribute = (node, name) => node?.openingElement?.attributes.find(attribute => attribute.name?.name === name);
export function walkTeachingJsx(node, visit) {
  if (!node || typeof node !== 'object') return;
  if (node.type) visit(node);
  for (const [key, value] of Object.entries(node)) {
    if (['loc', 'extra', 'comments'].includes(key)) continue;
    if (Array.isArray(value)) value.forEach(child => walkTeachingJsx(child, visit));
    else if (value && typeof value === 'object') walkTeachingJsx(value, visit);
  }
}
export const parseTeachingJsx = source => parse(source, { sourceType: 'module', plugins: ['jsx'] });
export function staticTeachingLabel(node) {
  if (!node) return null;
  if (['JSXText', 'StringLiteral', 'NumericLiteral'].includes(node.type)) return String(node.value);
  if (node.type === 'JSXExpressionContainer') return staticTeachingLabel(node.expression);
  if (node.type === 'JSXElement' || node.type === 'JSXFragment') {
    const children = node.children.map(staticTeachingLabel);
    return children.some(value => value === null) ? null : children.join(' ').replace(/\s+/g, ' ').trim();
  }
  return null;
}
export function teachingDisclosures(source) {
  const result = [];
  walkTeachingJsx(parseTeachingJsx(source), node => {
    if (teachingTag(node) === 'details') result.push(node);
  });
  return result;
}

function classEdit(source, element, token) {
  const attribute = teachingAttribute(element, 'className');
  if (!attribute) return { start: element.openingElement.end - 1, end: element.openingElement.end - 1, value: ` className="${token}"` };
  if (attribute.value?.type !== 'StringLiteral') throw new Error('A dynamic disclosure class needs an explicit authoring adapter');
  if (attribute.value.value.split(/\s+/).includes(token)) throw new Error(`Unexpected existing teaching class: ${token}`);
  return { start: attribute.value.end - 1, end: attribute.value.end - 1, value: ` ${token}` };
}

// A reviewed source inventory supplies exact disclosure ordinals. This function
// never guesses whether words such as "explanation" denote teaching or feedback.
export function revealTeachingDisclosures(source, entries) {
  const nodes = teachingDisclosures(source), edits = [], seen = new Set();
  for (const { index, headingLevel = 'h3' } of entries) {
    if (seen.has(index) || !['h2', 'h3', 'h4'].includes(headingLevel)) throw new Error('Invalid or duplicate teaching disclosure entry');
    seen.add(index);
    const node = nodes[index];
    if (!node) throw new Error(`Missing teaching disclosure ordinal ${index}`);
    const summaries = node.children.filter(child => teachingTag(child) === 'summary');
    if (summaries.length !== 1) throw new Error(`Teaching disclosure ${index} needs exactly one direct summary`);
    if (node.openingElement.attributes.some(attribute => ['open', 'onToggle', 'onClick'].includes(attribute.name?.name))) throw new Error('A stateful disclosure needs an explicit visible-section adapter');
    const summary = summaries[0];
    for (const [element, tag] of [[node, 'section'], [summary, headingLevel]]) {
      edits.push({ start: element.openingElement.name.start, end: element.openingElement.name.end, value: tag });
      edits.push({ start: element.closingElement.name.start, end: element.closingElement.name.end, value: tag });
    }
    edits.push(classEdit(source, node, 'lesson-teaching-section'));
    edits.push(classEdit(source, summary, 'lesson-teaching-section__title'));
    edits.push({ start: node.openingElement.end - 1, end: node.openingElement.end - 1, value: ' data-lesson-teaching=""' });
  }
  for (const edit of edits.sort((a, b) => b.start - a.start)) source = source.slice(0, edit.start) + edit.value + source.slice(edit.end);
  parseTeachingJsx(source);
  return source;
}

export function visibleTeachingEntries(source, sourcePath = 'source') {
  const entries = [];
  walkTeachingJsx(parseTeachingJsx(source), node => {
    if (!teachingAttribute(node, 'data-lesson-teaching')) return;
    const heading = node.children.find(child => teachingAttribute(child, 'className')?.value?.value?.split(/\s+/).includes('lesson-teaching-section__title'));
    const summary = staticTeachingLabel(heading);
    if (!summary) throw new Error(`Preserve the dynamic teaching section explicitly before regenerating ${sourcePath}`);
    entries.push({ summary, headingLevel: teachingTag(heading) });
  });
  return entries;
}

export function preserveVisibleTeachingSections(jsx, sourcePath, explicit = []) {
  const previous = sourcePath && fs.existsSync(sourcePath) ? visibleTeachingEntries(fs.readFileSync(sourcePath, 'utf8'), sourcePath) : [];
  const entries = [...previous];
  for (const entry of explicit) {
    const old = entries.find(item => item.summary === entry.summary);
    if (old && (old.headingLevel || 'h3') !== (entry.headingLevel || 'h3')) throw new Error(`Conflicting teaching heading: ${entry.summary}`);
    if (!old) entries.push(entry);
  }
  if (!entries.length) return jsx;
  const prefix = 'const lesson = <>', suffix = '</>;', source = prefix + jsx + suffix;
  const disclosures = teachingDisclosures(source), visible = visibleTeachingEntries(source), changes = [];
  for (const entry of entries) {
    if (!entry.summary || !['h2', 'h3', 'h4'].includes(entry.headingLevel || 'h3')) throw new Error('Invalid visible teaching entry');
    const matches = disclosures.flatMap((node, index) => staticTeachingLabel(node.children.find(child => teachingTag(child) === 'summary')) === entry.summary ? [index] : []);
    const existing = visible.filter(item => item.summary === entry.summary);
    if (matches.length + existing.length !== 1) throw new Error(`Visible teaching needs one exact summary: ${entry.summary}; found ${matches.length + existing.length}`);
    if (existing.length && existing[0].headingLevel !== (entry.headingLevel || 'h3')) throw new Error(`Conflicting teaching heading: ${entry.summary}`);
    if (matches.length) changes.push({ index: matches[0], headingLevel: entry.headingLevel || 'h3' });
  }
  return revealTeachingDisclosures(source, changes).slice(prefix.length, -suffix.length);
}

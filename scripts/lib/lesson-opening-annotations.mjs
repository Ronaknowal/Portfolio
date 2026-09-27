// Authoring-only preservation of explicit JSX annotations. No prose classifier
// or source parser is included in the browser lesson runtime.
import { readFileSync } from 'node:fs';
import { parse, parseExpression } from '@babel/parser';

export const openingKinds = new Set(['summary', 'route', 'prerequisites', 'exploration']);

function visit(node, callback) {
  if (!node || typeof node !== 'object') return;
  if (node.type) callback(node);
  for (const [key, value] of Object.entries(node)) {
    if (['loc', 'start', 'end', 'extra', 'comments'].includes(key)) continue;
    if (Array.isArray(value)) value.forEach(child => visit(child, callback));
    else if (value && typeof value === 'object') visit(value, callback);
  }
}

function staticText(node) {
  if (node.type === 'JSXText' || node.type === 'StringLiteral' || node.type === 'NumericLiteral') return String(node.value);
  if (node.type === 'JSXEmptyExpression') return '';
  if (node.type === 'JSXExpressionContainer') return staticText(node.expression);
  if (node.type === 'TemplateLiteral' && node.expressions.length === 0) return node.quasis[0].value.cooked;
  if (node.type === 'JSXElement' || node.type === 'JSXFragment') {
    const values = node.children.map(staticText);
    return values.some(value => value === null) ? null : values.join('');
  }
  return null;
}

function proseNodes(ast) {
  const result = [];
  visit(ast, node => {
    if (node.type !== 'JSXElement' || node.openingElement.name.name !== 'Prose') return;
    const attribute = node.openingElement.attributes.find(item => item.type === 'JSXAttribute' && item.name.name === 'opening');
    const value = attribute?.value;
    const kind = value?.type === 'StringLiteral' ? value.value : value?.type === 'JSXExpressionContainer' && value.expression.type === 'StringLiteral' ? value.expression.value : null;
    const text = staticText(node);
    result.push({ node, attribute, kind, text: text === null ? null : text.replace(/\s+/g, ' ').trim() });
  });
  return result;
}

export function preserveLessonOpeningAnnotations(jsx, sourcePath) {
  if (!sourcePath) return jsx;
  let source;
  try { source = readFileSync(sourcePath, 'utf8'); }
  catch (error) {
    if (error.code === 'ENOENT') return jsx; // A new lesson has no previous annotations.
    throw error;
  }
  const previous = proseNodes(parse(source, { sourceType: 'module', plugins: ['jsx'] })).filter(item => item.attribute);
  if (!previous.length) return jsx;
  const generated = proseNodes(parseExpression(`<>${jsx}</>`, { plugins: ['jsx'] }));
  const edits = [], seen = new Set();
  for (const item of previous) {
    const label = `${sourcePath}: ${item.text?.slice(0,90) || 'dynamic opening paragraph'}`;
    if (!openingKinds.has(item.kind) || !item.text) throw new Error(`Opening annotation must have a supported static kind and paragraph: ${label}`);
    if (seen.has(item.text)) throw new Error(`Opening paragraph is ambiguous in the previous source: ${label}`);
    seen.add(item.text);
    const matches = generated.filter(candidate => candidate.text === item.text);
    if (matches.length !== 1) throw new Error(`Cannot preserve opening="${item.kind}": expected one exact paragraph match, found ${matches.length}: ${label}`);
    const match = matches[0];
    if (match.attribute && match.kind !== item.kind) throw new Error(`Conflicting opening annotation for: ${label}`);
    if (!match.attribute) edits.push({ at: match.node.openingElement.name.end - 2, text: ` opening="${item.kind}"` });
  }
  // Only insert the annotation; keep generated paragraph JSX and all links intact.
  for (const edit of edits.sort((a, b) => b.at - a.at)) jsx = jsx.slice(0, edit.at) + edit.text + jsx.slice(edit.at);
  return jsx;
}

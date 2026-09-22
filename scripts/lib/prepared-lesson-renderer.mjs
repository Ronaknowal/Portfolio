// Authoring-only Markdown subset -> static JSX. No parser ships to the browser.
export const jsxString = value => `{${JSON.stringify(value)}}`;
export function renderInline(source, assetBase = '') {
  const pattern = /(\\\([\s\S]*?\\\)|`[^`]+`|\*\*[^*]+\*\*|\[[^\]]+\]\([^\s]+\))/g;
  let result = '', end = 0;
  for (const match of source.matchAll(pattern)) {
    result += jsxString(source.slice(end, match.index));
    const token = match[0];
    if (token.startsWith('\\(')) result += `<InlineMath>${jsxString(token.slice(2, -2))}</InlineMath>`;
    else if (token.startsWith('`')) result += `<code>${jsxString(token.slice(1, -1))}</code>`;
    else if (token.startsWith('**')) result += `<strong>${renderInline(token.slice(2, -2), assetBase)}</strong>`;
    else {
      const [, label, url] = token.match(/^\[([^\]]+)\]\((.+)\)$/);
      const target = /^(?:https?:|\/|#)/.test(url) ? url : assetBase + url;
      result += `<a href=${jsxString(target)}>${renderInline(label, assetBase)}</a>`;
    }
    end = match.index + token.length;
  }
  return result + jsxString(source.slice(end));
}
function tableCells(line) {
  // A math absolute-value bar is not a Markdown table separator.
  const protectedTokens = [];
  const protectedLine = line.replace(/\\\([\s\S]*?\\\)|`[^`]*`/g, text => {
    protectedTokens.push(text); return `\u0000${protectedTokens.length - 1}\u0000`;
  });
  return protectedLine.trim().replace(/^\||\|$/g, '').split('|').map(cell => cell.trim().replace(/\u0000(\d+)\u0000/g, (_, n) => protectedTokens[n]));
}
export function renderPreparedLesson(manuscript, { assetBase, replacements = [], additions = [] } = {}) {
  const lines = manuscript.replaceAll('\r\n', '\n').split('\n');
  const parts = [], sections = [], used = new Map();
  const inline = text => renderInline(text, assetBase);
  let paragraph = [], heading = 'Lesson overview';
  const emitParagraph = () => {
    if (!paragraph.length) return;
    const value = paragraph.join(' '), replacement = replacements.find(([prefix]) => value.startsWith(prefix));
    if (replacement) { parts.push(replacement[1]); used.set(replacement[0], (used.get(replacement[0]) || 0) + 1); }
    else parts.push(`<Prose>${inline(value)}</Prose>`);
    for (const [prefix, jsx] of additions) if (value.startsWith(prefix)) { parts.push(jsx); used.set(prefix, (used.get(prefix) || 0) + 1); }
    paragraph = [];
  };
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.startsWith('# ')) continue;
    if (!line.trim()) { emitParagraph(); continue; }
    if (line.startsWith('```')) {
      emitParagraph(); const language = line.slice(3), code = [];
      while (++i < lines.length && lines[i] !== '```') code.push(lines[i]);
      if (i === lines.length) throw new Error('Unclosed code fence');
      parts.push(`<CodeBlock language=${jsxString(language)}>${jsxString(code.join('\n'))}</CodeBlock>`); continue;
    }
    if (line === '\\[') {
      emitParagraph(); const math = [];
      while (++i < lines.length && lines[i] !== '\\]') math.push(lines[i]);
      if (i === lines.length) throw new Error('Unclosed mathematical expression');
      parts.push(`<div className="neural-equation"><MathBlock>${jsxString(math.join('\n'))}</MathBlock></div>`); continue;
    }
    if (/^##+ /.test(line)) {
      emitParagraph(); heading = line.replace(/^#+ /, '');
      const level = line.startsWith('### ') ? 'H3' : 'H2';
      if (level === 'H2') sections.push([heading.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, ''), heading]);
      parts.push(`<${level}>${jsxString(heading)}</${level}>`); continue;
    }
    if (line.startsWith('|')) {
      emitParagraph(); const table = [];
      while (i < lines.length && lines[i].startsWith('|')) { if (!/^\|[\s:|-]+\|$/.test(lines[i])) table.push(tableCells(lines[i])); i++; } i--;
      if (table.some(row => row.length !== table[0].length)) throw new Error(`Table width mismatch: ${heading}`);
      parts.push(`<NeuralTable caption=${jsxString(heading)} headers={[${table[0].map(v => `<>${inline(v)}</>`).join(',')}]} rows={[${table.slice(1).map(row => `[${row.map(v => `<>${inline(v)}</>`).join(',')}]`).join(',')}]} />`); continue;
    }
    if (/^\d+\. /.test(line) && line.includes('**Solution:**')) {
      emitParagraph(); const [questionHint, solution] = line.replace(/^\d+\. /, '').split('**Solution:**');
      const [question, hint] = questionHint.split('Hint:');
      parts.push(`<section className="neural-practice"><Prose>${inline(question.trim())}</Prose>${hint ? `<details><summary>Hint</summary><Prose>${inline(hint.trim())}</Prose></details>` : ''}<details><summary>Worked solution</summary><Prose>${inline(solution.trim())}</Prose></details></section>`); continue;
    }
    if (/^(?:- |\d+\. )/.test(line)) {
      emitParagraph(); const tag = line.startsWith('- ') ? 'ul' : 'ol', items = [];
      while (i < lines.length && /^(?:- |\d+\. )/.test(lines[i])) { items.push(lines[i].replace(/^(?:- |\d+\. )/, '')); i++; } i--;
      parts.push(`<${tag}>${items.map(value => `<li>${inline(value)}</li>`).join('')}</${tag}>`); continue;
    }
    if (/^<\/?(?:details|summary)/.test(line)) { emitParagraph(); parts.push(line); continue; }
    paragraph.push(line);
  }
  emitParagraph();
  for (const [prefix] of [...replacements, ...additions]) if (used.get(prefix) !== 1) throw new Error(`Expected one insertion anchor: ${prefix}; found ${used.get(prefix) || 0}`);
  return { jsx: parts.join('\n\n'), sections };
}

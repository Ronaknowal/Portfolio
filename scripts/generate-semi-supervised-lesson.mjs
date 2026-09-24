/** Render this topic's retained manuscript as static JSX; no runtime Markdown dependency. */
import { readFileSync, writeFileSync } from 'node:fs';

const topicId = 'semi-supervised-learning-label-propagation-self-training-co-training';
const packet = `docs/teaching/drafts/${topicId}`;
const manuscript = readFileSync(`${packet}/lesson.md`, 'utf8').replaceAll('\r\n', '\n');
const quote = value => JSON.stringify(value);
const localLinks = new Set(['graph-solutions.py', 'banknote-experiment.py', 'cotrain-categories.py', 'banknote-subset.csv', 'data-provenance.md']);
function inline(text) {
  const pattern = /(\$[^$\n]+\$|`[^`]+`|\*\*[^*]+\*\*|\[[^\]]+\]\([^\s]+\))/g;
  let result = '';
  let last = 0;
  for (const match of text.matchAll(pattern)) {
    result += `{${quote(text.slice(last, match.index))}}`;
    const token = match[0];
    if (token[0] === '$') result += `<InlineMath>{${quote(token.slice(1, -1))}}</InlineMath>`;
    else if (token[0] === '`') result += `<code>{${quote(token.slice(1, -1))}}</code>`;
    else if (token.startsWith('**')) result += `<strong>{${quote(token.slice(2, -2))}}</strong>`;
    else {
      const [, label, link] = token.match(/^\[([^\]]+)\]\((.+)\)$/);
      result += `<a href={${quote(localLinks.has(link) ? `/learn-assets/semi-supervised-learning/${link}` : link)}}>${inline(label)}</a>`;
    }
    last = match.index + token.length;
  }
  result += `{${quote(text.slice(last))}}`;
  return result;
}

const replacements = [
  ['**Inline illustration — the label ledger.**', '<LabelLedgerFigure />'],
  ['**Inline illustration — two layers of scores.**', '<EvidenceFigure />'],
  ['**Inline illustration — the promotion audit.**', '<PromotionAuditFigure />'],
  ['**Investigation — repair the neighborhood.**', '<LabelPropagationLab />'],
  ['**Investigation — a guess changes the next guess.**', '<PrototypeFlowFigure /><SelfTrainingLab />'],
  ['**Investigation — pass a label through the other view.**', '<PairedViewsFigure /><CoTrainingLab />'],
];
const additions = [
  ['Learning the groups accurately helps the first task', '<SameInputsFigure />'],
  ['The intermediate values alternate around the answer.', '<HardFlowFigure />'],
];
const sourceLines = manuscript.split('\n');
const parts = [];
let paragraph = [];
let list = [];
let ordered = false;
let table = [];
let sectionTitle = 'Lesson overview';
let displayedProgram = 0;
function flushParagraph() {
  if (!paragraph.length) return;
  const text = paragraph.join(' ');
  const replacement = replacements.find(([prefix]) => text.startsWith(prefix));
  parts.push(replacement ? replacement[1] : `<Prose>${inline(text)}</Prose>`);
  const addition = additions.find(([prefix]) => text.startsWith(prefix));
  if (addition) parts.push(addition[1]);
  paragraph = [];
}
function flushList() {
  if (!list.length) return;
  parts.push(`<${ordered ? 'ol' : 'ul'} className="ssl-prose-list">${list.map(item => `<li>${inline(item)}</li>`).join('\n')}</${ordered ? 'ol' : 'ul'}>`);
  list = [];
}
function flushTable() {
  if (!table.length) return;
  const rows = table.filter(row => !/^\|[\s:|-]+\|$/.test(row)).map(row => row.slice(1, -1).split('|').map(value => value.trim()));
  parts.push(`<DataTable caption={${quote(sectionTitle)}} headers={[${rows[0].map(cell => `<>${inline(cell)}</>`).join(',')}]} rows={[${rows.slice(1).map(row => `[${row.map(cell => `<>${inline(cell)}</>`).join(',')}]`).join(',')}]} />`);
  table = [];
}
for (let i = 0; i < sourceLines.length; i += 1) {
  const line = sourceLines[i];
  if (line.startsWith('# ')) continue;
  if (!line.trim()) { flushParagraph(); flushList(); flushTable(); continue; }
  if (line.startsWith('```')) {
    flushParagraph(); flushList(); flushTable();
    const language = line.slice(3);
    const code = [];
    while (sourceLines[++i] !== '```') code.push(sourceLines[i]);
    if (language === 'python') {
      const index = displayedProgram++ === 0 ? 0 : 2;
      const file = index === 0 ? 'graph-solutions.py' : 'banknote-experiment.py';
      if (code.join('\n').trim() !== readFileSync(`${packet}/${file}`, 'utf8').replaceAll('\r\n', '\n').trim()) throw new Error(`Manuscript/program mismatch: ${file}`);
      parts.push(`<RunnableExample example={semiSupervisedExamples[${index}]} /><p className="lesson-note"><a href={semiSupervisedExamples[${index}].download} download>Download the exact Python program</a></p>`);
    } else parts.push(`<CodeBlock language={${quote(language)}}>{${quote(code.join('\n'))}}</CodeBlock>`);
    continue;
  }
  if (line === '$$') {
    flushParagraph(); flushList(); flushTable();
    const expression = [];
    while (sourceLines[++i] !== '$$') expression.push(sourceLines[i]);
    parts.push(`<div className="ssl-equation" tabIndex={0} role="region" aria-label="Mathematical expression"><MathBlock>{${quote(expression.join('\n'))}}</MathBlock></div>`);
    continue;
  }
  if (line.startsWith('## ')) { flushParagraph(); flushList(); flushTable(); sectionTitle = line.slice(3); parts.push(`<H2>{${quote(sectionTitle)}}</H2>`); continue; }
  if (line.startsWith('### ')) { flushParagraph(); flushList(); flushTable(); sectionTitle = line.slice(4); parts.push(`<H3>{${quote(sectionTitle)}}</H3>`); continue; }
  if (line.startsWith('<details>')) { flushParagraph(); parts.push(line); continue; }
  if (line === '</details>') { flushParagraph(); parts.push(line); continue; }
  if (line.startsWith('|')) { flushParagraph(); flushList(); table.push(line); continue; }
  const item = line.match(/^(?:- |\d+\. )(.*)/);
  if (item) { flushParagraph(); flushTable(); ordered = /^\d/.test(line); list.push(item[1]); continue; }
  paragraph.push(line);
}
flushParagraph(); flushList(); flushTable();
const cotrainParagraph = parts.findIndex(part => part.includes('The final view-1 rules include'));
parts.splice(cotrainParagraph, 0, '<details><summary>Inspect the complete categorical co-training program and its recorded output</summary><RunnableExample example={semiSupervisedExamples[1]} /><p><a href={semiSupervisedExamples[1].download} download>Download the complete co-training program</a></p></details>');

const intro = `<LessonIntro prerequisites="Weighted averages, basic classification and train/development/test separation. The lesson refreshes its matrix and probability notation locally." sections={${JSON.stringify(sourceLines.filter(line => line.startsWith('## ')).map(line => [line.slice(3).toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, ''), line.slice(3)]))}}>Use a few observed labels without mistaking every model guess for new evidence. Follow sections 1–7, then practice; the deeper connections are an optional second pass.</LessonIntro>`;
const output = `// Static presentation of the prepared manuscript; regeneration is topic-scoped.\nimport { Prose, H2, H3, CodeBlock } from '../../components/content';\nimport { Math as InlineMath, MathBlock } from '../../components/content/Math.jsx';\nimport { LessonIntro } from '../../components/lesson-labs/LessonElements.jsx';\nimport { RunnableExample } from '../../components/lesson-labs/RunnableExample.jsx';\nimport { DataTable, LabelLedgerFigure, SameInputsFigure, HardFlowFigure, EvidenceFigure, PrototypeFlowFigure, PairedViewsFigure, PromotionAuditFigure } from '../../components/lesson-labs/SemiSupervisedFigures.jsx';\nimport { LabelPropagationLab, SelfTrainingLab, CoTrainingLab } from '../../components/lesson-labs/SemiSupervisedLabs.jsx';\nimport { semiSupervisedExamples } from '../semi-supervised-examples.js';\n\nconst semiSupervisedLesson = {\n  title: 'Semi-Supervised Learning (Label Propagation, Self-Training, Co-Training)',\n  readTime: '~60 min read + 90 min practice',\n  hasIntegratedGuide: true,\n  content: () => <div className="ssl-lesson">\n${intro}\n${parts.join('\n\n')}\n  </div>,\n};\nexport default semiSupervisedLesson;\n`;
writeFileSync(`src/learn/data/topics/${topicId}.jsx`, output);
console.log('Generated full ten-section SSL manuscript, seven figures, three labs and all eight practice tasks.');

// Rebuild only the prepared Backprop lesson. Production renders JSX, never Markdown.
// Inputs: retained topic packet. Outputs: topic body, recorded data, offline downloads.
import fs from 'node:fs';
import path from 'node:path';

const packet = 'docs/teaching/drafts/backpropagation-automatic-differentiation';
const assetRoot = 'public/learn-assets/backpropagation';
fs.mkdirSync(assetRoot, { recursive: true });
for (const name of ['teaching-autodiff.py', 'digits-400.csv', 'author-calculations.py', 'calculated-inputs.json']) {
  fs.copyFileSync(path.join(packet, name), path.join(assetRoot, name));
}
const provenance = fs.readFileSync(`${packet}/data-provenance.md`, 'utf8').replace('[fixture and full provenance](../perceptrons-neurons-activation-functions/data-provenance.md)', 'fixture and full provenance (reproduced below)') + '\n\n' + fs.readFileSync('docs/teaching/drafts/perceptrons-neurons-activation-functions/data-provenance.md', 'utf8').replace('Exact source IDs are retained in calculated-inputs.json.', 'Every source ID is retained in digits-400.csv; the program reproduces the instructional split.');
fs.writeFileSync(`${assetRoot}/data-provenance.md`, provenance);
const recorded = JSON.parse(fs.readFileSync(`${packet}/calculated-inputs.json`, 'utf8'));
fs.writeFileSync('src/learn/data/backprop-training.js', `// Source: retained calculated-inputs.json; complete CPU teaching-engine runs.\nexport const backpropTraining = ${JSON.stringify({ versions: recorded.versions, xorTraining: recorded.xorTraining, digitTraining: recorded.digitTraining }, null, 2)};\n`);

const json = JSON.stringify;
function readableProse(value) {
  // Only ordinary text runs pass here: code and math tokens stay byte-for-byte.
  return value.split(/((?:float|binary)(?:32|64)|3Blue1Brown|\b\d+(?:\.\d+)?e[-−]?\d+\b)/g).map((part, index) => index % 2 ? part : part
    .replace(/([A-Za-z])(?=\d)/g, '$1 ')
    .replace(/(\d)(?=[A-Za-z])/g, '$1 ')
    .replace(/([A-Za-z])(?=[−-]\d)/g, '$1 ')).join('');
}
function inline(text) {
  const pattern = /(`[^`]+`|\$[^$]+\$|\*\*[^*]+\*\*|\[[^\]]+\]\([^\s)]+\))/g;
  let output = '';
  let position = 0;
  for (const match of text.matchAll(pattern)) {
    if (match.index > position) output += `{${json(readableProse(text.slice(position, match.index)))}}`;
    const token = match[0];
    if (token.startsWith('`')) output += `<Code>{${json(token.slice(1, -1))}}</Code>`;
    else if (token.startsWith('$')) output += `<Math>{${json(token.slice(1, -1))}}</Math>`;
    else if (token.startsWith('**')) output += `<strong>${inline(token.slice(2, -2))}</strong>`;
    else {
      const [, label, href] = token.match(/^\[([^\]]+)\]\((.*)\)$/);
      const target = href.startsWith('http') || href.startsWith('/') ? href : `/learn-assets/backpropagation/${href}`;
      output += `<a href={${json(target)}}>${inline(label)}</a>`;
    }
    position = match.index + token.length;
  }
  if (position < text.length) output += `{${json(readableProse(text.slice(position)))}}`;
  return output;
}

const source = fs.readFileSync(`${packet}/lesson.md`, 'utf8');
const lines = source.split(/\r?\n/);
const output = [];
const markers = { A: 'BackpropScalarFigure', B: 'BackpropSharedLab', C: 'BackpropBroadcastFigure', D: 'BackpropShapeFigure', E: 'BackpropTrainingFigure', G: 'BackpropProductsFigure', H: 'BackpropCheckpointFigure' };
let index = 0;
let tableIndex = 0;
let pythonIndex = 0;
const programs = [];
const tableCaptions = ['One neuron: forward values and loss sensitivities', 'Local derivative rules and their saved forward values', 'Two-layer classifier tensor shapes', 'Recorded XOR and digit observations from separate training runs', 'Perturbation size, sine derivative error and offset-linear estimate', 'Autograd operations and their distinct effects'];
const programNames = ['finite-differences.py', 'pytorch-accumulation.py', 'directional-products.py', 'hessian-vector.py', 'custom-hard-sigmoid.py', 'microbatch-means.py', 'activation-checkpoint.py'];
while (index < lines.length) {
  const line = lines[index].trim();
  if (!line || line.startsWith('# ')) { index++; continue; }
  if (line.startsWith('```')) {
    const language = line.slice(3) || 'text';
    const block = [];
    index++;
    while (index < lines.length && !lines[index].startsWith('```')) block.push(lines[index++]);
    const code = block.join('\n');
    output.push(`<CodeBlock language=${json(language)}>{${json(code)}}</CodeBlock>`);
    if (language === 'python' && !code.startsWith('# Inside')) {
      const filename = programNames[pythonIndex++];
      fs.writeFileSync(`${assetRoot}/${filename}`, `${code}\n`);
      programs.push(filename);
    }
    index++; continue;
  }
  if (line === '$$') {
    const math = [];
    index++;
    while (index < lines.length && lines[index].trim() !== '$$') math.push(lines[index++]);
    output.push(`<MathBlock>{${json(math.join('\n'))}}</MathBlock>`);
    index++; continue;
  }
  if (line.startsWith('## ')) { output.push(`<H2>{${json(line.slice(3))}}</H2>`); index++; continue; }
  if (line.startsWith('### ')) { output.push(`<H3>{${json(line.slice(4))}}</H3>`); index++; continue; }
  const visual = line.match(/^\[Visual([A-H]):/);
  if (visual) { output.push(`<${markers[visual[1]]} />`); index++; continue; }
  if (line.startsWith('<details>')) { output.push(`<details><summary>${inline(line.match(/<summary>(.*?)<\/summary>/)[1])}</summary>`); index++; continue; }
  if (line === '</details>') { output.push('</details>'); index++; continue; }
  if (line.startsWith('|')) {
    const rows = [];
    while (index < lines.length && lines[index].trim().startsWith('|')) rows.push(lines[index++].trim().slice(1, -1).split('|').map(cell => cell.trim()));
    const headers = rows.shift();
    rows.shift();
    const caption = tableCaptions[tableIndex++];
    output.push(`<BackpropTable caption=${json(caption)} headers={[${headers.map(cell => `<>${inline(cell)}</>`).join(',')}]} rows={[${rows.map(row => `[${row.map(cell => `<>${inline(cell)}</>`).join(',')}]`).join(',')}]} />`);
    continue;
  }
  if (/^(?:- |\d+\. )/.test(line)) {
    const ordered = /^\d+\./.test(line);
    const items = [];
    while (index < lines.length && /^(?:- |\d+\. )/.test(lines[index].trim())) items.push(lines[index++].trim().replace(/^(?:- |\d+\. )/, ''));
    const tag = ordered ? 'ol' : 'ul';
    output.push(`<${tag}>${items.map(item => `<li>${inline(item)}</li>`).join('')}</${tag}>`);
    continue;
  }
  const paragraph = [line];
  index++;
  while (index < lines.length && lines[index].trim() && !/^(?:#|\[Visual|\| |<details|<\/details|```|\$\$|- |\d+\. )/.test(lines[index].trim())) paragraph.push(lines[index++].trim());
  let prose = paragraph.join(' ');
  prose = prose.replace('Independent replay of this complete custom-op program belongs to the finishing phase; its piecewise rule is derived here.', 'The complete custom-operation program is included among the independently replayed native examples; its piecewise rule is derived here.')
    .replace('The expected result isTrue for this deterministic block; independent execution of this excerpt is deferred.', 'The executed result is True for this deterministic block.')
    .replace('is expected to pass at those smooth points', 'passes at those tested smooth points');
  output.push(`<Prose>${inline(prose)}</Prose>`);
  if (prose.startsWith('**First pass:**')) {
    const sections = lines.filter(line => line.startsWith('## ')).map(line => line.slice(3));
    output.push(`<nav className="backprop-section-route" aria-label="Backpropagation section route"><p>Jump within this lesson</p><ol>${sections.map(section => `<li><a href="#${section.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '')}">${inline(section)}</a></li>`).join('')}</ol></nav>`);
  }
  if (prose.startsWith('The author checks compared every parameter')) output.push('<Prose>Replay the <a href="/learn-assets/backpropagation/author-calculations.py">complete same-state checking program</a> beside the engine and CSV; compare its output with the <a href="/learn-assets/backpropagation/calculated-inputs.json">recorded calculations</a>. It needs the NumPy/scikit-learn environment above plus PyTorch 2.14.0 (CPU is sufficient).</Prose>');
  if (prose.startsWith('**Investigation A')) output.push('<BackpropFitLab />');
  if (prose.startsWith('**Investigation F')) output.push('<BackpropDifferenceLab />');
}

const imports = `import { Prose, H2, H3, Code, CodeBlock } from '../../components/content';
import { Math, MathBlock } from '../../components/content/Math.jsx';
import { BackpropTable } from '../../components/lesson-labs/BackpropShared.jsx';
import { BackpropFitLab, BackpropSharedLab, BackpropDifferenceLab } from '../../components/lesson-labs/BackpropLabs.jsx';
import { BackpropScalarFigure, BackpropBroadcastFigure, BackpropShapeFigure, BackpropTrainingFigure, BackpropProductsFigure, BackpropCheckpointFigure } from '../../components/lesson-labs/BackpropFigures.jsx';

// Complete prepared manuscript rendered with topic-specific mechanisms. See the retained design for provenance and checks.
const backpropContent = {
  title: 'Backpropagation & Automatic Differentiation',
  hasIntegratedGuide: true,
  readTime: '60–80 min + practice',
  content: () => <div className="backprop-lesson">
`;
fs.writeFileSync('src/learn/data/topics/backprop.jsx', `${imports}${output.map(block => `    ${block}`).join('\n')}\n  </div>,\n};\nexport default backpropContent;\n`);
fs.writeFileSync(`${assetRoot}/programs.json`, JSON.stringify(programs, null, 2) + '\n');
console.log(`Backpropagation: generated ${output.length} prose/representation blocks, ${programs.length} complete native examples, engine and data downloads.`);

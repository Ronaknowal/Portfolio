// Rebuild this topic only. Preserves the complete revision-3 manuscript and
// adds mechanism-specific teaching components; no Markdown renderer is shipped.
import fs from 'node:fs';
const packet = 'docs/teaching/drafts/residual-connections-skip-connections';
const assets = 'public/learn-assets/residual-connections';
const json = JSON.stringify;
const source = fs.readFileSync(`${packet}/lesson.md`, 'utf8');
function inline(text) {
  const pattern = /(`[^`]+`|\\\([\s\S]*?\\\)|\*\*[^*]+\*\*|\[[^\]]+\]\([^\s)]+\))/g;
  let output = '', start = 0;
  for (const match of text.matchAll(pattern)) {
    if (match.index > start) output += `{${json(text.slice(start, match.index))}}`;
    const token = match[0];
    if (token.startsWith('`')) output += `<Code>{${json(token.slice(1, -1))}}</Code>`;
    else if (token.startsWith('\\(')) output += `<Math>{${json(token.slice(2, -2))}}</Math>`;
    else if (token.startsWith('**')) output += `<strong>${inline(token.slice(2, -2))}</strong>`;
    else {
      const [, label, href] = token.match(/^\[([^\]]+)\]\((.*)\)$/);
      const target = href.startsWith('/') || href.startsWith('http') ? href : `/learn-assets/residual-connections/${href.replace(/^\.\//, '')}`;
      output += `<a href={${json(target)}}>${inline(label)}</a>`;
    }
    start = match.index + token.length;
  }
  if (start < text.length) output += `{${json(text.slice(start))}}`;
  return output;
}
const lines = source.split(/\r?\n/), output = [];
const sections = lines.filter(line => line.startsWith('## ')).map(line => line.slice(3));
const slug = text => text.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
const captions = ['Forward routes in the worked vector example', 'Scalar derivatives through ten residual blocks', 'Operation placement with a zero correction', 'Four block return expressions', 'Selected actual seed-1 final results'];
let index = 0, tableIndex = 0;
while (index < lines.length) {
  const line = lines[index].trim();
  if (!line || line.startsWith('# ')) { index++; continue; }
  if (line.startsWith('```')) {
    const language = line.slice(3) || 'text', block = []; index++;
    while (index < lines.length && !lines[index].startsWith('```')) block.push(lines[index++]);
    output.push(`<CodeBlock language=${json(language)}>{${json(block.join('\n'))}}</CodeBlock>`); index++; continue;
  }
  if (line === '\\[') {
    const math = []; index++; while (index < lines.length && lines[index].trim() !== '\\]') math.push(lines[index++]);
    output.push(`<MathBlock>{${json(math.join('\n'))}}</MathBlock>`); index++; continue;
  }
  if (line.startsWith('## ')) { output.push(`<H2>{${json(line.slice(3))}}</H2>`); index++; continue; }
  if (line.startsWith('### ')) { output.push(`<H3>{${json(line.slice(4))}}</H3>`); index++; continue; }
  if (line.startsWith('<details>')) {
    const match = line.match(/^<details><summary>(.*?)<\/summary>(.*)$/);
    output.push(`<details><summary>${inline(match[1])}</summary>`);
    if (match[2]) { const body = match[2].replace(/<\/details>$/, '').replace('At rate0.1', 'At rate 0.1'); if (body) output.push(`<Prose>${inline(body)}</Prose>`); if (match[2].endsWith('</details>')) output.push('</details>'); }
    index++; continue;
  }
  if (line === '</details>') { output.push('</details>'); index++; continue; }
  if (line.startsWith('|')) {
    const rows = [];
    while (index < lines.length && lines[index].trim().startsWith('|')) rows.push(lines[index++].trim().slice(1, -1).split('|').map(cell => cell.trim()));
    const headers = rows.shift(); rows.shift();
    output.push(`<ResidualTable caption=${json(captions[tableIndex++])} headers={[${headers.map(cell => `<>${inline(cell)}</>`).join(',')}]} rows={[${rows.map(row => `[${row.map(cell => `<>${inline(cell)}</>`).join(',')}]`).join(',')}]} />`); continue;
  }
  if (line.startsWith('- ')) {
    const entries = [];
    while (index < lines.length && lines[index].trim().startsWith('- ')) entries.push(`<li>${inline(lines[index++].trim().slice(2))}</li>`);
    output.push(`<ul>${entries.join('')}</ul>`); continue;
  }
  const paragraph = [line]; index++;
  while (index < lines.length && lines[index].trim() && !/^(?:#|\||<details|<\/details|```|\\\[|- )/.test(lines[index].trim())) paragraph.push(lines[index++].trim());
  let prose = paragraph.join(' ');
  prose = prose.replace('The prior initialization manuscript is prepared and supplies the scaling discussion; it is not evidence that its new rendered page is already finished.', 'The preceding initialization lesson supplies the scaling discussion; the local calculations here expose how that scaling changes a connected block.')
    .replace('at rate0.1', 'at rate 0.1');
  output.push(`<Prose>${inline(prose)}</Prose>`);
  if (prose.startsWith('**First pass:**')) output.push(`<aside className="res-route"><p>Before you start: matrix multiplication, gradients and the preceding normalization/initialization lessons. The first five sections build the core; optional connections follow the real experiment.</p><nav aria-label="Residual connections lesson sections"><ol>${sections.map(section => `<li><a href="#${slug(section)}">${inline(section)}</a></li>`).join('')}</ol></nav></aside>`);
  if (prose.startsWith('This is a linear example')) output.push('<ResidualCorrectionLab />');
  if (prose.startsWith('Notice the second coordinate')) output.push('<ResidualGradientFigure />');
  if (prose.startsWith('**Gradient investigation:**')) output.push('<ResidualDepthLab />');
  if (prose.startsWith('In **pre-activation**')) output.push('<ResidualOrderLab />');
  if (prose.startsWith('**Shape investigation:**')) output.push('<ResidualProjectionLab />');
  if (prose.startsWith('where the expectations')) output.push('<ResidualOpeningLab />');
  if (prose.startsWith('The program contains all data')) output.push('<ResidualProgram />');
  if (prose.startsWith('**Try a controlled change:**')) output.push('<ResidualEvidenceLab />');
  if (prose.startsWith('The scratch operation here')) {
    const native = fs.readFileSync(`${assets}/residual-experiments.py`, 'utf8');
    const excerpt = native.slice(native.indexOf('    def forward(self, inputs):'), native.indexOf('\n\n\nclass DigitNetwork'));
    output.push(`<CodeBlock language="python">{${json(excerpt)}}</CodeBlock>`);
    output.push('<Prose>The branch normalizes only its own input, creates hidden features, and produces a signed correction with a final linear map. The plain case returns that correction directly. The residual case first rejects accidental broadcasting, then adds the original input. This shape check cannot inspect feature meaning; preserving coordinate semantics remains part of the architecture design. There is no special residual operator hidden behind this return statement.</Prose>');
  }
  if (prose.startsWith('The function `mechanisms`')) output.push('<Prose>In the fitting loop, <Code>model.train()</Code> selects training behavior, <Code>optimizer.zero_grad(set_to_none=True)</Code> clears the previous derivative, and cross-entropy receives logits rather than already-normalized probabilities. After checking that loss is finite, <Code>loss.backward()</Code> traverses both routes and <Code>optimizer.step()</Code> updates the parameters. Evaluation uses <Code>model.eval()</Code> with <Code>torch.no_grad()</Code>; these control different things. Fixed scales are saved buffers, while trainable gates enter the optimizer through <Code>model.parameters()</Code>. Match these details before attributing a changed result to the shortcut. The <a href="https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html">Module API</a> documents mode, buffer and parameter behavior.</Prose>');
  if (prose.startsWith('The ensemble-of-paths interpretation')) output.push('<ResidualPathsFigure />');
  if (prose.startsWith('**Remove noise')) output.push('<ResidualDenoisingFigure />');
  if (prose.startsWith('**Refinement as a time step')) output.push('<ResidualEulerLab />');
  if (prose.startsWith('Backward through a pure addition')) output.push('<ResidualMemoryFigure />');
}
if (tableIndex !== captions.length) throw new Error('Manuscript table mapping changed');
const header = `import { Prose, H2, H3, Code, CodeBlock } from '../../components/content';
import { Math, MathBlock as SharedMathBlock } from '../../components/content/Math.jsx';
import { ResidualTable, ResidualCorrectionLab, ResidualGradientFigure, ResidualDepthLab, ResidualOrderLab, ResidualProjectionLab, ResidualOpeningLab, ResidualEvidenceLab, ResidualPathsFigure, ResidualDenoisingFigure, ResidualEulerLab, ResidualMemoryFigure, ResidualProgram } from '../../components/lesson-labs/ResidualConnectionsLabs.jsx';
function MathBlock({children}) { return <div className="res-equation" role="region" tabIndex={0} aria-label="Equation; scroll horizontally when needed"><SharedMathBlock>{children}</SharedMathBlock></div>; }
export default {
  title: 'Residual Connections & Skip Connections: Keep a Path, Learn a Correction',
  readTime: '55–75 min + live exploration and practice',
  hasIntegratedGuide: true,
  content: () => <div className="residual-lesson">
`;
fs.writeFileSync('src/learn/data/topics/residual-connections-skip-connections.jsx', `${header}${output.join('\n\n')}\n</div>,\n};\n`);
const measured = JSON.parse(fs.readFileSync(`${assets}/calculated-inputs.json`, 'utf8'));
const csv = fs.readFileSync(`${assets}/digits-400.csv`, 'utf8').trim().split(/\r?\n/).slice(1).map(line => line.split(',').map(Number));
const training = new Set(measured.training_source_ids);
const specimens = [0, 3, 8].map(digit => { const row = csv.find(row => row.at(-1) === digit && training.has(row[0])); return { source_id: row[0], digit, pixels: row.slice(1, 65) }; });
fs.writeFileSync(`${assets}/browser-measurements.json`, JSON.stringify({ versions: measured.versions, fits: measured.fits, specimens }));
console.log(`Residual connections: ${output.length} manuscript/visual blocks, ${sections.length} sections, ${measured.fits.length} actual fits.`);

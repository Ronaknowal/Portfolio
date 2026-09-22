// Authoring-time conversion: no Markdown parser or model weights in the route body.
import fs from 'node:fs';
import { renderPreparedLesson } from './lib/prepared-lesson-renderer.mjs';

const id = 'rnns-lstms-grus', folder = `docs/teaching/drafts/${id}/`, destination = `public/learn-assets/${id}/`;
fs.mkdirSync(destination, { recursive: true });
for (const name of ['pen-sequence-learning.py', 'recurrent-mechanics.py', 'pen-trajectories.csv', 'data-provenance.md', 'data-extraction.json', 'calculated-inputs.json', 'mechanics-results.json']) fs.copyFileSync(folder + name, destination + name);
fs.writeFileSync(destination + 'data-provenance.md', '> Implementation update, 22 September 2026: this packet now powers the implemented recurrent lesson. The historical content-first boundary below describes its preparation date; current native and browser evidence is recorded in docs/teaching/RECURRENT-IMPLEMENTATION.md. Dataset origin, split and experiment limits remain unchanged.\n\n' + fs.readFileSync(folder + 'data-provenance.md', 'utf8'));
const report = JSON.parse(fs.readFileSync(folder + 'calculated-inputs.json'));
const mechanics = JSON.parse(fs.readFileSync(folder + 'mechanics-results.json'));
const [header, ...csv] = fs.readFileSync(folder + 'pen-trajectories.csv', 'utf8').trim().split(/\r?\n/);
const columns = header.split(',');
const specimens = ['pendigits.tes:6', 'pendigits.tes:1', 'pendigits.tes:2'].map(id => {
  const row = csv.map(line => Object.fromEntries(line.split(',').map((v, i) => [columns[i], v]))).find(row => row.source_id === id);
  if (!row) throw new Error('Missing specimen ' + id);
  return { sourceId: id, digit: Number(row.digit), points: Array.from({ length: 8 }, (_, i) => ['x', 'y'].map(axis => Number((Number(row[axis + (i + 1)]) / 50 - 1).toFixed(2)))) };
});
fs.writeFileSync(destination + 'pen-models.json', JSON.stringify({ weights: report.saved_models, specimens }) + '\n');
fs.writeFileSync(destination + 'recurrent-evidence.json', JSON.stringify({ boundaries: mechanics.state_and_padding, runs: report.runs.map(({ first_two_examples, final, reversed, swapped_points_3_4, ...rest }) => ({ ...rest, final: { correct: final.correct, cross_entropy: final.cross_entropy }, reversed: { correct: reversed.correct, cross_entropy: reversed.cross_entropy }, swapped_points_3_4: { correct: swapped_points_3_4.correct, cross_entropy: swapped_points_3_4.cross_entropy } })) }) + '\n');

let manuscript = fs.readFileSync(folder + 'lesson.md', 'utf8').replaceAll('\r\n', '\n');
const code = fs.readFileSync(folder + 'pen-sequence-learning.py', 'utf8').replaceAll('\r\n', '\n').trim();
if (!manuscript.includes(code)) throw new Error('Displayed learner program differs from canonical source');
manuscript = manuscript.replace('```python\n' + code + '\n```', 'COMPLETE_RECURRENT_PROGRAM');
// Handle CRLF packets without changing their preserved checkpoint.
if (!manuscript.includes('COMPLETE_RECURRENT_PROGRAM')) {
  manuscript = manuscript.replaceAll('\r\n', '\n').replace('```python\n' + code.replaceAll('\r\n', '\n') + '\n```', 'COMPLETE_RECURRENT_PROGRAM');
}
manuscript = manuscript.replace(/(<details>|<\/details>|<summary>[^<]*<\/summary>)/g, '\n\n$1\n\n');
const { jsx, sections } = renderPreparedLesson(manuscript, {
  assetBase: `/learn-assets/${id}/`,
  replacements: [
    ['COMPLETE_RECURRENT_PROGRAM', '<RecurrentProgram file="pen-sequence-learning.py" title="Read the complete real-data training and evaluation program" />'],
    ['**Visual: the same trace', '<PenRepresentationLab />'],
    ['**Investigation: edit an observation', '<RecurrentCreditLab />'],
    ['**Visual: an accounting', '<LstmMemoryLab />'],
    ['**Investigation: design a memory interval', '<LstmRetentionLab />'],
    ['**Visual: two reset placements', '<GruResetLab />'],
    ['**Investigation: change the actual pen path', '<RecurrentPenLab />'],
    ['**Visual: two arrows', '<RecurrentBoundaryLab />'],
  ],
  additions: [
    ['All nine runs remain visible', '<RecurrentMeasuredLab />'],
    ['Packing tells the native recurrent', '<RecurrentPaddingLab />'],
    ['The complete [recurrent-mechanics.py]', '<RecurrentProgram start="def manual_sequence" end="def scalar_credit" title="Read the explicit NumPy RNN, LSTM and GRU cell equations" />'],
    ['Read the gate ordering', '<RecurrentProgram start="def scalar_credit" end="def state_and_padding" title="Read the scratch temporal gradient and matched autograd check" />'],
    ['For T steps, batch B', '<RecurrentProgram title="Read all native parity, chunk, padding and derivative checks" />'],
  ],
});
const imports = ['PenRepresentationLab', 'RecurrentCreditLab', 'LstmMemoryLab', 'LstmRetentionLab', 'GruResetLab', 'RecurrentPenLab', 'RecurrentBoundaryLab', 'RecurrentPaddingLab', 'RecurrentMeasuredLab', 'RecurrentProgram'];
fs.writeFileSync(`src/learn/data/topics/${id}.jsx`, `// Complete prepared revision-3 manuscript with live mechanism investigations.
import { Prose, H2, H3, CodeBlock } from '../../components/content';
import { Math as InlineMath, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro } from '../../components/lesson-labs/LessonElements.jsx';
import { NeuralTable } from '../../components/lesson-labs/NeuralLessonElements.jsx';
import { ${imports.join(', ')} } from '../../components/lesson-labs/RecurrentLabs.jsx';
export default {
  title: 'RNNs, LSTMs & GRUs',
  readTime: '~70 min read + experiments and practice',
  hasIntegratedGuide: true,
  content: () => <div className="neural-lesson recurrent-lesson">
    <LessonIntro prerequisites="Vectors, affine transforms, activations and the chain rule. Sequence axes, gates and state ownership are introduced here." sections={${JSON.stringify(sections)}}>Trace a sequence into state, follow how it learns, and control what crosses a boundary.</LessonIntro>
${jsx}
  </div>,
};
`);
console.log('Rendered complete recurrent manuscript with nine investigations and canonical lazy source.');

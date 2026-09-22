import fs from 'node:fs';
import assert from 'node:assert/strict';
import { renderPreparedLesson } from './lib/prepared-lesson-renderer.mjs';
const id = 'sequence-to-sequence-encoder-decoder';
const draft = `docs/teaching/drafts/${id}/`, assets = `public/learn-code/${id}/`;
fs.mkdirSync(assets, { recursive: true });
for (const file of ['inflection-seq2seq.py', 'sequence-mechanics.py', 'english-inflections.csv', 'calculated-inputs.json', 'mechanics-results.json', 'data-provenance.md', 'data-extraction.json', 'prepare-inflection-data.py']) fs.copyFileSync(draft + file, assets + file);
const measured = JSON.parse(fs.readFileSync(draft + 'calculated-inputs.json'));
const mechanics = JSON.parse(fs.readFileSync(draft + 'mechanics-results.json'));
fs.writeFileSync(assets + 'seed-one-inference.json', JSON.stringify({ weights: measured.runs[0].weights }) + '\n');
fs.writeFileSync('src/learn/data/seq2seq-measurements.json', JSON.stringify({ runs: measured.runs.map(({ seed, checkpoints }) => ({ seed, checkpoints: checkpoints.map(({ update, teacher_forced_nll, exact }) => ({ update, teacher_forced_nll, exact })) })), slices: mechanics.slices }) + '\n');
let manuscript = fs.readFileSync(draft + 'lesson.md', 'utf8').replaceAll('\r\n', '\n');
const blocks = [...manuscript.matchAll(/```python\n([\s\S]*?)\n```/g)];
assert.equal(blocks[0][1].trim(), fs.readFileSync(draft + 'inflection-seq2seq.py', 'utf8').trim());
fs.writeFileSync(assets + 'saved-inflection.py', blocks[1][1] + '\n');
manuscript = manuscript.replace(/<details>\n<summary>Complete runnable program<\/summary>\n\n```python\n[\s\S]*?\n```\n\n<\/details>/, '[Complete runnable program placement]');
manuscript = manuscript.replace(/<details><summary>(.*?)<\/summary>(.*?)<\/details>/gs, (_, title, contents) => `<details>\n<summary>${title}</summary>\n\n${contents.trim()}\n\n</details>`);
manuscript = manuscript.replace(/\$\$([\s\S]*?)\$\$/g, (_, math) => `\\[\n${math.trim()}\n\\]`).replace(/\$([^$\n]+)\$/g, (_, math) => `\\(${math}\\)`);
manuscript = manuscript.replace('The prepared [recurrent cell owner](../rnns-lstms-grus/recurrent-mechanics.py) already maps gate order and biases to `nn.GRU`; until its improved page is published, that exact packet remains the honest prerequisite source.', 'The [recurrent cell implementation](/learn-assets/rnns-lstms-grus/recurrent-mechanics.py) maps gate order and biases to `nn.GRU`. Its [RNN, LSTM and GRU lesson](/learn/path/full-curriculum/rnns-lstms-grus?module=deep-learning-fundamentals) teaches the scratch cell and matched library route; reuse that mechanism here.');
const rendered = renderPreparedLesson(manuscript, {
  assetBase: `/learn-code/${id}/`,
  replacements: [
    ['**Visual: two token tracks.**', '<Seq2SeqTimelines />'],
    ['**Investigation: repair the token tracks.**', '<Seq2SeqAlignmentLab />'],
    ['**Visual: state bridge with a numerical expansion.**', '<Seq2SeqBridgeLab />'],
    ['**Investigation: edit a probability tree.**', '<Seq2SeqTreeLab />'],
    ['**Visual: actual learning curves and paired outputs.**', '<Seq2SeqEvidenceFigure />'],
    ['[Complete runnable program placement]', '<Seq2SeqProgram file="inflection-seq2seq.py" title="Read the complete runnable inflection model" />'],
    ['**Investigation: change the source and the decoder prefix.**', '<Seq2SeqFittedLab />'],
  ],
  additions: [
    ['Save `english-inflections.csv`', '<Prose>Download the <a href={seq2seqAsset + "english-inflections.csv"}>offline inflection CSV</a>, <a href={seq2seqAsset + "inflection-seq2seq.py"}>complete program</a> and <a href={seq2seqAsset + "data-provenance.md"}>attribution and split provenance</a>. Preserve the UniMorph English contributors, source revision, adaptation and CC BY-SA 3.0 license when sharing the extract. To inspect the fixed model without rerunning training, also save <a href={seq2seqAsset + "calculated-inputs.json"}>the measured record and weights</a> plus <a href={seq2seqAsset + "saved-inflection.py"}>the standalone inference example</a>.</Prose>'],
    ['This lesson\'s new program', '<Seq2SeqProgram file="sequence-mechanics.py" title="Read the scalar joint derivative, manual GRU protocol and exact search checks" />'],
  ],
});
fs.writeFileSync(`src/learn/data/topics/${id}.jsx`, `// Full revision-3 manuscript preserved; long canonical source loads on demand.
import { Prose, H2, H3, CodeBlock } from '../../components/content';
import { Math as InlineMath, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro } from '../../components/lesson-labs/LessonElements.jsx';
import { NeuralTable } from '../../components/lesson-labs/NeuralLessonElements.jsx';
import { Seq2SeqTimelines, Seq2SeqAlignmentLab, Seq2SeqBridgeLab, Seq2SeqTreeLab, Seq2SeqEvidenceFigure, Seq2SeqProgram, Seq2SeqFittedLab, seq2seqAsset } from '../../components/lesson-labs/Seq2SeqLabs.jsx';
export default {
 title: 'Sequence-to-Sequence & Encoder-Decoder',
 readTime: '~65 min read + experiments and practice',
 hasIntegratedGuide: true,
 content: () => <div className="neural-lesson seq2seq-lesson"><LessonIntro prerequisites="Recurrent state updates and cross-entropy. Source/target tokens, shapes, context and loss normalization are refreshed before composing the model." sections={${JSON.stringify(rendered.sections)}}>Read an input, learn a conditional output and inspect the difference between memorizing examples, generating an answer and finding a better route.</LessonIntro>
${rendered.jsx}
 </div>,
};
`);
console.log(`Seq2seq: ${rendered.sections.length} sections; complete code and four live investigations.`);

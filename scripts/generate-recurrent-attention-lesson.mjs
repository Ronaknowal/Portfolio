import fs from 'node:fs';
import assert from 'node:assert/strict';
import { renderPreparedLesson } from './lib/prepared-lesson-renderer.mjs';

const id = 'attention-mechanism-bahdanau-luong';
const draft = `docs/teaching/drafts/${id}/`;
const manuscriptDirectory = `docs/teaching/concept-intuition/${id}/`;
const assets = `public/learn-code/${id}/`;
fs.mkdirSync(assets, { recursive: true });
for (const file of ['attentive-inflection.py', 'attention-calculations.py', 'attention-mechanics.py', 'english-inflections.csv', 'calculated-inputs.json', 'mechanics-results.json', 'analytic-results.json', 'data-provenance.md', 'data-extraction.json', 'prepare-inflection-data.py']) {
  fs.copyFileSync(draft + file, assets + file);
}
fs.copyFileSync('docs/teaching/drafts/sequence-to-sequence-encoder-decoder/split-integrity-repair.json', assets + 'split-integrity-repair.json');
fs.writeFileSync(assets + 'data-provenance.md', fs.readFileSync(draft + 'data-provenance.md', 'utf8').replace('../sequence-to-sequence-encoder-decoder/split-integrity-repair.json', 'split-integrity-repair.json'));
const report = JSON.parse(fs.readFileSync(draft + 'calculated-inputs.json'));
const mechanics = JSON.parse(fs.readFileSync(draft + 'mechanics-results.json'));
for (const kind of ['additive', 'general']) {
  const run = report.runs.find(row => row.kind === kind && row.seed === 1);
  fs.writeFileSync(assets + `${kind}-seed-one.json`, JSON.stringify({ kind, weights: run.weights }) + '\n');
}
const compact = run => ({ kind: run.kind ?? 'fixed', seed: run.seed, checkpoints: run.checkpoints.map(({ update, exact }) => ({ update, exact })) });
fs.writeFileSync('src/learn/data/recurrent-attention-measurements.json', JSON.stringify({
  runs: [...mechanics.prior_baseline.runs.map(compact), ...report.runs.map(compact)],
  worked: mechanics.traces.additive.worked.rows.map(row => ({ attention: row.attention, emitted: row.emitted_token })),
}) + '\n');

let manuscript = fs.readFileSync(manuscriptDirectory + 'lesson.md', 'utf8').replaceAll('\r\n', '\n');
const blocks = [...manuscript.matchAll(/```python\n([\s\S]*?)\n```/g)];
const training = fs.readFileSync(assets + 'attentive-inflection.py', 'utf8').replaceAll('\r\n', '\n').trim();
assert.equal(blocks.filter(block => block[1].trim() === training).length, 1, 'Retain the complete canonical training program');
const savedExample = blocks.find(block => block[1].includes('spec_from_file_location'));
assert.ok(savedExample, 'Retain the runnable saved-model example');
fs.writeFileSync(assets + 'saved-inflection.py', savedExample[1] + '\n');
const coreRead = blocks.find(block => block[1].includes('def attention_read('));
assert.ok(coreRead, 'Publish the locally explained core read');
fs.writeFileSync(assets + 'attention-read.py', coreRead[1] + '\n');
manuscript = manuscript.replace(/<details>\n<summary>Complete CPU training, generation and measurement program<\/summary>\n\n```python\n[\s\S]*?\n```\n\n<\/details>/, '[Complete attention training program]');
manuscript = manuscript.replace(/<details><summary>(.*?)<\/summary>(.*?)<\/details>/gs, (_, title, body) => `<details>\n<summary>${title}</summary>\n\n${body.trim()}\n\n</details>`);
const rendered = renderPreparedLesson(manuscript, { preserveOpeningFrom: `src/learn/data/topics/${id}.jsx`,
  assetBase: `/learn-code/${id}/`,
  replacements: [
    ['**Visual: summary versus revisitable notes.**', '<AttentionRecallContrast />'],
    ['**Visual: lookup becomes a weighted read.**', '<AttentionLookupBridge />'],
    ['**Visual: scores become shares.**', '<AttentionSoftmaxSteps />'],
    ['**Visual: build a learned comparison.**', '<AttentionAdditiveSteps />'],
    ['**Visual: padding changes the denominator.**', '<AttentionPaddingShares />'],
    ["**Visual: a memory's effect depends on its content.**", '<AttentionLearningDirection />'],
    ['**Visual: the source-memory shelf.**', '<AttentionMemoryShelf />'],
    ['**Visual: three weighted contributions.**', '<AttentionWorkedRead />'],
    ['**Investigation: edit the memory, then inspect the read.**', '<AttentionReadLab />'],
    ['**Visual: synchronized decoder timelines.**', '<AttentionSchedules />'],
    ['**Investigation: repair a padded read.**', '<AttentionFittedLab mode="padding" />'],
    ['**Visual: signed credit through the read.**', '<AttentionGradientFlow />'],
    ['**Investigation: one update and a null case.**', '<AttentionReadLab learning />'],
    ['**Measured figure: all run checkpoints.**', '<AttentionLearningCurves />'],
    ['[Complete attention training program]', '<AttentionProgram file="attentive-inflection.py" title="Read the complete CPU training and generation program" />'],
    ['**Visual: the complete alignment matrix and one linked read.**', '<AttentionWorkedAlignment />'],
    ['**Investigation: test an alignment hypothesis.**', '<AttentionFittedLab />'],
    ['**Visual: window on a source ruler.**', '<AttentionWindowLab />'],
    ['**Visual: two routes into one vocabulary.**', '<AttentionCopyFlow />'],
  ],
  additions: [
    ['The weights no longer depend on the question.', '<AttentionCancellationLab />'],
    ['Both \\(\\alpha=', '<AttentionAmbiguityFigure />'],
    ['The convolution \\(F*', '<AttentionLocationFlow />'],
    ['The complete [attentive-inflection.py]', '<AttentionScratchRoute />'],
  ],
});
fs.writeFileSync('src/learn/data/topics/attention.jsx', `// Generated from the active concept-intuition manuscript; canonical numerical programs retained.
import { Prose, H2, H3, CodeBlock } from '../../components/content';
import { Math as InlineMath, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro } from '../../components/lesson-labs/LessonElements.jsx';
import { NeuralTable } from '../../components/lesson-labs/NeuralLessonElements.jsx';
import { AttentionMemoryShelf, AttentionWorkedRead, AttentionReadLab, AttentionSchedules, AttentionFittedLab, AttentionGradientFlow, AttentionLearningCurves, AttentionProgram, AttentionWorkedAlignment, AttentionWindowLab, AttentionCopyFlow, AttentionCancellationLab, AttentionAmbiguityFigure, AttentionLocationFlow, AttentionScratchRoute } from '../../components/lesson-labs/RecurrentAttentionLabs.jsx';
import { AttentionRecallContrast, AttentionLookupBridge, AttentionSoftmaxSteps, AttentionAdditiveSteps, AttentionPaddingShares, AttentionLearningDirection } from '../../components/lesson-labs/RecurrentAttentionIntuition.jsx';
export default {
  title: 'Attention Mechanism (Bahdanau, Luong)',
  readTime: '~65 min read + 90 min experiments and practice',
  hasIntegratedGuide: true,
  content: () => <div className="neural-lesson attention-lesson"><LessonIntro prerequisites="Recurrent state updates, encoder–decoder generation and cross-entropy. Queries, vector reads, shapes and masks are introduced here." sections={${JSON.stringify(rendered.sections)}}>Let each output read the input it needs. Follow an exact memory read, build it from scratch, and inspect real trained spelling models.</LessonIntro>
${rendered.jsx}
  </div>,
};
`);
console.log(`Attention: ${rendered.sections.length} full sections; seven live investigations and explanatory figures.`);

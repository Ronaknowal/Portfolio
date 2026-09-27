import fs from 'node:fs';
import path from 'node:path';
import { renderPreparedLesson } from './lib/prepared-lesson-renderer.mjs';

const id = 'transformer-block-architecture';
const directory = `docs/teaching/drafts/${id}`;
const destination = `public/learn-code/${id}`;
fs.mkdirSync(destination, { recursive: true });
for (const filename of ['author-calculations.py', 'author-results.json', 'block-models.json', 'movement_libras.data', 'movement_libras.names', 'data-provenance.md']) {
  fs.copyFileSync(path.join(directory, filename), path.join(destination, filename));
}
const models = JSON.parse(fs.readFileSync(`${directory}/block-models.json`, 'utf8'));
const runtime = Object.fromEntries(Object.entries(models).map(([placement, model]) => [placement, {
  preNorm: placement === 'pre-norm', state_dict: model.state_dict, points: model.points,
  times: model.times, original: model.original,
}]));
fs.writeFileSync(`${destination}/movement-runtime.json`, JSON.stringify(runtime));
const evidence = JSON.parse(fs.readFileSync(`${directory}/author-results.json`, 'utf8'));
fs.writeFileSync(`${destination}/evidence-runtime.json`, JSON.stringify({ fits: evidence.fits, gradientDepth: evidence.fixtures.gradient_depth }));
let manuscript = fs.readFileSync(`${directory}/lesson.md`, 'utf8').replaceAll('\r\n', '\n');
manuscript = manuscript.replaceAll('../self-attention-multi-head-attention/lesson.md#5-implement-the-operation-you-just-traced', '/learn/path/full-curriculum/self-attention-multi-head-attention?module=deep-learning-fundamentals');
const rendered = renderPreparedLesson(manuscript, { preserveOpeningFrom: `src/learn/data/topics/${id}.jsx`,
  assetBase: `/learn-code/${id}/`,
  replacements: [
    ['**Inline figure — communication and revision.', '<BlockCommunicationFigure />'],
    ['**Inline figure — two circuits, same components.', '<BlockWiringFigure />'],
    ['**Investigation — choose the ruler.', '<BlockNormalizationLab />'],
    ['**Inline figure — feature responses and write directions.', '<BlockFeedforwardLab />'],
    ['**Investigation — build the block.', '<BlockTraceLab />'],
    ['**Inline figure — observed learning, not a preferred winner.', '<BlockEvidenceFigure />'],
    ['**Investigation — what did the model use?', '<BlockMovementLab />'],
    ['**Investigation — choose a meaningful probe.', '<BlockProbeLab /><BlockEvidenceFigure gradients />'],
    ['**Inline figure — one block interface, three information boundaries.', '<BlockTaskFigure />'],
    ['**Inline figure — where the count grows.', '<BlockCostsLab />'],
  ],
  additions: [['A practical architecture description', '<BlockVariantsFigure />'], ['There are two useful levels of partitioning.', '<BlockSystemsFigure />']],
});
const componentNames = ['BlockCommunicationFigure', 'BlockWiringFigure', 'BlockNormalizationLab', 'BlockFeedforwardLab', 'BlockTraceLab', 'BlockMovementLab', 'BlockProbeLab', 'BlockEvidenceFigure', 'BlockTaskFigure', 'BlockVariantsFigure', 'BlockCostsLab', 'BlockSystemsFigure'];
fs.writeFileSync(`src/learn/data/topics/${id}.jsx`, `// Generated from the complete prepared manuscript by scripts/generate-transformer-block-lesson.mjs.
import { Prose, H2, H3, CodeBlock } from '../../components/content';
import { Math as InlineMath, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro } from '../../components/lesson-labs/LessonElements.jsx';
import { NeuralTable } from '../../components/lesson-labs/NeuralLessonElements.jsx';
import { ${componentNames.join(', ')} } from '../../components/lesson-labs/TransformerBlockLabs.jsx';
export default {
  title: 'Transformer Block Architecture',
  readTime: '~85 min read + experiments and practice',
  hasIntegratedGuide: true,
  content: () => <div className="neural-lesson neural-lesson-neutral transformer-block-lesson"><LessonIntro prerequisites="Self-attention, residual additions and per-position feature normalization. Their essential operations are refreshed where used." sections={${JSON.stringify(rendered.sections)}}>Follow one representation through communication, feature processing and the saved residual path.</LessonIntro>
${rendered.jsx}
  </div>,
};
`);
console.log(`Generated ${id}: ${rendered.sections.length} sections, complete manuscript and topic-owned runtime assets.`);

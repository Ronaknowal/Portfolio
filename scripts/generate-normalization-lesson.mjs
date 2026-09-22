import { readFileSync, writeFileSync } from 'node:fs';
import { renderPreparedLesson } from './lib/prepared-lesson-renderer.mjs';
const id = 'batch-layer-group-rms-normalization';
writeFileSync('src/learn/data/normalization-backward-program.js', '// Generated from the canonical downloadable Python source.\nexport default ' + JSON.stringify(readFileSync(`public/learn-assets/${id}/normalization-backward.py`, 'utf8')) + ';\n');
const rendered = renderPreparedLesson(readFileSync(`docs/teaching/drafts/${id}/lesson.md`, 'utf8'), {
  assetBase: `/learn-assets/${id}/`,
  replacements: [
    ['**Visual — shared ruler:**', '<NormalizationRulerFigure />'],
    ['**Investigation — who shares my ruler?**', '<NormalizationMembershipLab />'],
    ['**Investigation — change brightness or contrast?**', '<NormalizationGeometryLab />'],
    ['**Investigation — follow the state:**', '<BatchNormalizationStateLab />'],
    ['**Visual — measured learning trajectories:**', '<NormalizationMeasuredLab />'],
    ['**Visual — zero-branch contrast:**', '<NormalizationPlacementLab />'],
  ],
  additions: [
    ['Download [normalization-backward.py]', '<NeuralProgram topic="normalization-backward" title="Read the manual normalization derivatives and library checks" />'],
    ['The tiny input gradient', '<NormalizationGradientLab />'],
    ['Download [normalization-experiments.py]', '<NeuralProgram topic="normalization" /><p><a href="/learn-assets/batch-layer-group-rms-normalization/recorded-output.txt">Recorded complete-program output</a> · <a href="/learn-assets/batch-layer-group-rms-normalization/snippet-output.txt">Precision snippet output</a></p>'],
  ],
});
writeFileSync(`src/learn/data/topics/${id}.jsx`, `// Complete prepared manuscript rendered at authoring time.
import { Prose, H2, H3, CodeBlock } from '../../components/content';
import { Math as InlineMath, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro } from '../../components/lesson-labs/LessonElements.jsx';
import { NeuralTable } from '../../components/lesson-labs/NeuralLessonElements.jsx';
import { NormalizationRulerFigure, NormalizationMembershipLab, NormalizationGeometryLab, BatchNormalizationStateLab, NormalizationGradientLab, NormalizationMeasuredLab, NormalizationPlacementLab } from '../../components/lesson-labs/NormalizationLabs.jsx';
import NeuralProgram from '../../components/lesson-labs/NeuralProgram.jsx';
export default {
  title: 'Batch, Layer, Group & RMS Normalization',
  readTime: '~65 min read + experiments and practice',
  hasIntegratedGuide: true,
  content: () => <div className="neural-lesson normalization-lesson">
  <LessonIntro prerequisites="Neural activations, a loss and a gradient. Mean, variance, tensor axes, trainable parameters and stored buffers are explained as they arise." sections={${JSON.stringify(rendered.sections)}}>Name the values that share a statistic before choosing a normalization layer.</LessonIntro>
  ${rendered.jsx}
  <p><a href="/learn/path/full-curriculum/transfer-learning-fine-tuning-strategies?module=deep-learning-fundamentals">Continue to Transfer Learning and Fine-Tuning Strategies</a></p>
  </div>
};
`);
console.log(`Rendered ${rendered.sections.length} complete normalization sections with manual derivatives and independent practice.`);

import { readFileSync, writeFileSync } from 'node:fs';
import { renderPreparedLesson } from './lib/prepared-lesson-renderer.mjs';
const id = 'loss-functions-ce-mse-focal-contrastive-triplet';
const packet = `docs/teaching/drafts/${id}`;
writeFileSync('src/learn/data/loss-mechanisms-program.js', '// Generated from the canonical downloadable Python source.\nexport default ' + JSON.stringify(readFileSync(`public/learn-assets/${id}/loss-mechanisms.py`, 'utf8')) + ';\n');
const rendered = renderPreparedLesson(readFileSync(`${packet}/lesson.md`, 'utf8'), {
  assetBase: `/learn-assets/${id}/`,
  replacements: [
    ['**Visual — three linked views:**', '<LossUpdateFigure />'],
    ['**Visual — probability-to-decision audit:**', '<LossDecisionLab />'],
    ['**Visual — candidate competition:**', '<InfoNceLab />'],
  ],
  additions: [
    ['Save `loss-mechanisms.py`,', '<NeuralProgram topic="loss-mechanisms" title="Read the scratch objectives and matched library checks" />'],
    ['**Investigation — move one measurement:**', '<RegressionInfluenceLab />'],
    ['Focal loss was introduced for the many easy', '<FocalContributionLab />'],
    ['**Investigation — choose a useful negative:**', '<TripletGeometryLab />'],
    ['Full CE over', '<LossScalingFigure />'],
    ['Read the implementation in three parts.', '<NeuralProgram topic="loss" /><p><a href="/learn-assets/loss-functions-ce-mse-focal-contrastive-triplet/recorded-output.txt">Recorded full-program output</a> · <a href="/learn-assets/loss-functions-ce-mse-focal-contrastive-triplet/snippet-output.txt">API snippet outputs</a></p>'],
  ],
});
writeFileSync(`src/learn/data/topics/${id}.jsx`, `// Complete prepared manuscript rendered at authoring time.
import { Prose, H2, H3, CodeBlock } from '../../components/content';
import { Math as InlineMath, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro } from '../../components/lesson-labs/LessonElements.jsx';
import { NeuralTable } from '../../components/lesson-labs/NeuralLessonElements.jsx';
import { LossUpdateFigure, RegressionInfluenceLab, FocalContributionLab, LossDecisionLab, TripletGeometryLab, InfoNceLab, LossScalingFigure } from '../../components/lesson-labs/LossFunctionsLabs.jsx';
import NeuralProgram from '../../components/lesson-labs/NeuralProgram.jsx';
export default {
  title: 'Loss Functions (CE, MSE, Focal, Contrastive, Triplet)',
  readTime: '~65 min read + experiments and practice',
  hasIntegratedGuide: true,
  content: () => <div className="neural-lesson loss-functions-lesson">
  <LessonIntro prerequisites="A model maps inputs to outputs; backpropagation propagates derivatives. Residuals, probabilities and embedding coordinates are introduced locally." sections={${JSON.stringify(rendered.sections)}}>Choose what errors should change, trace their gradients, then judge the decisions produced by a real model.</LessonIntro>
  ${rendered.jsx}
  <p><a href="/learn/path/full-curriculum/batch-layer-group-rms-normalization?module=deep-learning-fundamentals">Continue to Batch, Layer, Group and RMS Normalization</a></p>
  </div>
};
`);
console.log(`Rendered ${rendered.sections.length} complete loss sections with the mechanism/library route and independent practice.`);

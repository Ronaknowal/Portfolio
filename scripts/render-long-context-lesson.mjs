// Deterministic authoring conversion; no Markdown parser or fitted weights in the route body.
import fs from 'node:fs';
import { renderPreparedLesson } from './lib/prepared-lesson-renderer.mjs';

const id = 'long-context-sequence-models-transformer-xl-griffin-perceiver';
let manuscript = fs.readFileSync(`docs/teaching/concept-intuition/${id}/lesson.md`, 'utf8').replaceAll('\r\n', '\n');
// Presentation-only normalization. Scientific source and prior revision stay conserved.
manuscript = manuscript.replace(/^\* /gm, '- ')
  .replace(/(<details>|<\/details>|<summary>[^<]*<\/summary>)/g, '\n\n$1\n\n');
const replacements = [
  ['**Inline figure: the entrance note leaves', '<EntranceMessageFigure />'],
  ['**Inline figure: six shares form', '<WeightedReadSharesFigure />'],
  ['**Inline figure: three ways to pass', '<SilentStepGateFigure />'],
  ['**Inline figure: two paths collide', '<PathMeanCollisionFigure />'],
  ['**Inline figure: three memory workspaces.', '<MemoryWorkspacesFigure />'],
  ['**Inline figure: a query, three key/value pairs', '<MaskedReadFigure />'],
  ['**Inline figure: a two-dimensional segment/layer', '<SegmentLayersFigure />'],
  ['**Investigation 1 — build the retained context.', '<MemoryCacheLab />'],
  ['**Inline figure: repeated local labels', '<PositionAddressFigure />'],
  ['**Inline figure: retention and injection', '<RetentionFigure />'],
  ['**Investigation 2 — preserve or replace a memory.', '<RecurrentMemoryLab />'],
  ['**Inline figure: recurrent, recurrent, local-attention', '<GriffinPathsFigure />'],
  ['**Inline figure: input array →', '<PerceiverReadFigure />'],
  ['**Investigation 3 — choose what the workspace', '<LatentWorkspaceLab />'],
  ['**Inline figure: paired permutation', '<PositionAttachmentFigure />'],
  ['**Inline figure: the two masks', '<CausalLatentsFigure />'],
  ['**Inline figure: the actual trajectory classifier.', '<TrajectoryArchitectureFigure />'],
  ['**Inline figure: individual measured error marks.', '<TrajectoryEvidenceFigure />'],
  ['**Investigation 4 — inspect a learned read.', '<LongContextTrajectoryLab />'],
  ['**Inline figure: attention interaction budgets.', '<MemoryBudgetsFigure />'],
];
const additions = [
  ['For many queries, do one such read', '<AttentionShapesFigure />'],
  ['If there are O output queries', '<OutputQueriesFigure />'],
  ['Every query must have at least one legal key.', '<LongContextProgram file="sequence_mechanisms.py" title="Read the complete scratch attention, cache, recurrence and latent program" />'],
  ['The complete file defines the projections', '<LongContextProgram file="latent_trajectory_classifier.py" title="Read the complete trainable classifier, baselines and evaluation program" />'],
  ['Run `python memory_library_bridge.py`', '<LongContextProgram file="memory_library_bridge.py" title="Read the matched library operators, RG-LRU reference and gradient comparisons" />'],
  ['For a scalar example, let old state', '<DetachedMemoryFigure />'],
  ['Composition is associative.', '<AffineScanFigure />'],
];
const { jsx, sections } = renderPreparedLesson(manuscript, { preserveOpeningFrom: `src/learn/data/topics/${id}.jsx`, assetBase: `/learn-code/${id}/`, replacements, additions });
const figures = ['MemoryWorkspacesFigure', 'MaskedReadFigure', 'AttentionShapesFigure', 'SegmentLayersFigure', 'PositionAddressFigure', 'RetentionFigure', 'GriffinPathsFigure', 'PerceiverReadFigure', 'PositionAttachmentFigure', 'OutputQueriesFigure', 'CausalLatentsFigure', 'TrajectoryArchitectureFigure', 'TrajectoryEvidenceFigure', 'DetachedMemoryFigure', 'MemoryBudgetsFigure', 'AffineScanFigure'];
fs.writeFileSync(`src/learn/data/topics/${id}.jsx`, `// Generated from the active concept-intuition manuscript; revision-4 evidence remains historical.
import { Prose, H2, H3, CodeBlock } from '../../components/content';
import { Math as InlineMath, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro } from '../../components/lesson-labs/LessonElements.jsx';
import { NeuralTable } from '../../components/lesson-labs/NeuralLessonElements.jsx';
import { MemoryCacheLab, RecurrentMemoryLab, LatentWorkspaceLab, LongContextProgram } from '../../components/lesson-labs/LongContextLabs.jsx';
import { LongContextTrajectoryLab } from '../../components/lesson-labs/LongContextTrajectoryLab.jsx';
import { ${figures.join(', ')} } from '../../components/lesson-labs/LongContextFigures.jsx';
import { EntranceMessageFigure, WeightedReadSharesFigure, SilentStepGateFigure, PathMeanCollisionFigure } from '../../components/lesson-labs/LongContextIntuitionFigures.jsx';
export default {
  title: 'Long-Context Sequence Models (Transformer-XL, Griffin, Perceiver)',
  readTime: '~80 min read + experiments and practice',
  hasIntegratedGuide: true,
  content: () => <div className="neural-lesson long-context-lesson">
    <LessonIntro prerequisites="Basic algebra and the preceding attention lesson help. We rebuild weighted reads, stored state, queries, masks and array shapes through small examples before using them." sections={${JSON.stringify(sections)}}>A model can read a fact and still lose access to it later. Follow what each kind of memory keeps, calculate how it changes, and then build and inspect the mechanisms yourself.</LessonIntro>
${jsx}
  </div>,
};
`);
console.log('Rendered active long-context manuscript, 20 inline figures and four live investigations.');

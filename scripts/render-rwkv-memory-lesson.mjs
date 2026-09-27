import fs from 'node:fs';
import { renderPreparedLesson } from './lib/prepared-lesson-renderer.mjs';

const id = 'rwkv-linear-attention-models';
let manuscript = fs.readFileSync(`docs/teaching/drafts/${id}/lesson.md`, 'utf8').replaceAll('\r\n', '\n');
manuscript = manuscript.replace('includes a constant-value case', 'includes a constant-value case');
const figure = kind => `<RwkvFigure kind="${kind}" />`;
const { jsx, sections } = renderPreparedLesson(manuscript, { preserveOpeningFrom: `src/learn/data/topics/${id}.jsx`,
  assetBase: `/learn-code/${id}/`,
  replacements: [
    ['**Inline visual — a query visits', figure('transcript')],
    ['**Inline visual — accumulate rows', figure('outer')],
    ['**Inline visual — same input', figure('kernels')],
    ['**Inline visual — two routes', figure('chunk')],
    ['**Inline visual — two-stage', figure('circuit')],
    ['**Inline visual — the answer survives', figure('scale')],
    ['**Inline visual — one block', figure('block')],
    ['**Inline visual — the address', figure('correction')],
    ['**Inline visual — evolving', figure('versions')],
    ['**Inline visual — erase direction', figure('goose')],
    ['**Inline visual — one state', figure('forgetting')],
    ['**Inline visual — what grows', figure('cache')],
    ['**Inline visual — the task-shaped', figure('pipeline')],
    ['**Inline visual — honest', figure('outcomes')],
    ['**Investigation — can two summaries', '<RwkvSummaryLab />'],
    ["**Investigation — which change affects", '<RwkvWeightedLab />'],
    ['**Investigation — can a memory update', '<RwkvDeltaLab />'],
    ['**Investigation — pause a hand', '<RwkvTrajectoryLab />'],
  ],
  additions: [
    ['Updating a state at inference', figure('parameters')],
    ['After the first two writes', figure('ledger')],
    ['The identity follows from', figure('random')],
    ['During autoregressive generation', figure('timeline')],
    ['A retention of .5 halves', figure('retention')],
    ['A causal state update processes', figure('objective')],
    ['Keys interfere when', figure('interference')],
    ['A diagonal transition rescales', figure('reflection')],
    ['Now deliberately reset', figure('continuation')],
    ['The complete [linear_memory_mechanisms.py]', '<RwkvProgram file="linear_memory_mechanisms.py" title="Read the complete scratch memory operators" />'],
    ['The following loop is', '<RwkvProgram file="trajectory_memory_models.py" title="Read the complete trainable PyTorch model and study" />'],
    ['The small `TrajectoryMemoryClassifier`', '<RwkvProgram file="rwkv_checkpoint_state.py" title="Read the optional official-checkpoint continuation route" />'],
  ],
});
fs.writeFileSync(`src/learn/data/topics/${id}.jsx`, `// Generated from the complete current prepared manuscript by render-rwkv-memory-lesson.mjs.
import { Prose, H2, H3, CodeBlock } from '../../components/content';
import { Math as InlineMath, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro } from '../../components/lesson-labs/LessonElements.jsx';
import { NeuralTable } from '../../components/lesson-labs/NeuralLessonElements.jsx';
import { RwkvFigure, RwkvSummaryLab, RwkvWeightedLab, RwkvDeltaLab, RwkvTrajectoryLab, RwkvProgram } from '../../components/lesson-labs/RwkvMemoryLabs.jsx';
import '../../components/lesson-labs/neural-lesson-neutral.css';
export default {
  title: 'RWKV & Linear Attention Models',
  readTime: '~100 min read + experiments and practice',
  hasIntegratedGuide: true,
  content: () => <div className="neural-lesson neural-lesson-neutral rwkv-memory-lesson"><LessonIntro prerequisites="Vector products and a recurrent update. Query/key/value, matrix orientation and stable normalization are refreshed locally." sections={${JSON.stringify(sections)}}>Choose what a stream must remember: a weighted summary, individual records, or associations that can be corrected.</LessonIntro>
${jsx}
  </div>,
};
`);
console.log(`Rendered ${id}: ${sections.length} sections, 23 figures, 4 live investigations.`);

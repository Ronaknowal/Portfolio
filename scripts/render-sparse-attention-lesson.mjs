import fs from 'node:fs';
import { renderPreparedLesson } from './lib/prepared-lesson-renderer.mjs';
const id = 'sparse-linear-attention-variants';
const manuscript = fs.readFileSync(`docs/teaching/drafts/${id}/lesson.md`, 'utf8')
  .replaceAll('\r\n', '\n').replace(/\$([^$\n]+)\$/g, (_, math) => `\\(${math}\\)`);
const figure = kind => `<SparseFigure kind="${kind}" />`;
const { jsx, sections } = renderPreparedLesson(manuscript, { preserveOpeningFrom: `src/learn/data/topics/${id}.jsx`,
  assetBase: `/learn-code/${id}/`,
  replacements: [
    ['**Visual: four ways', figure('representations')],
    ['**Visual: an outer product', figure('memory')],
    ['**Visual: follow the future', figure('projection')],
    ['**Visual: the same trace', figure('trajectory')],
    ['**Investigation: draw a route', '<SparseGraphLab /><SparseReadLab />'],
    ['**Investigation: edit a memory', '<SparseMemoryLab />'],
    ['**Investigation: an approximation', `${figure('random')}<SparseRandomLab />`],
    ['**Investigation: repair the leak', '<SparseProjectionLab />'],
    ['**Investigation: pack the same edges', '<SparseTilesLab />'],
    ['**Investigation: inspect what an old edit', '<SparseForecastLab />'],
    ['The mechanism diagram has three', figure('nystrom')],
  ],
  additions: [
    ['**Candidate discovery**', figure('architectures')],
    ['If all values satisfy', figure('mass')],
    ['Causality changes that picture.', figure('hubs')],
    ['The earlier program owns', '<SparseProgram file="attention_compression_bridges.py" title="Read the complete scratch and SDPA compression bridges" />'],
    ['Run from the downloaded packet', '<SparseProgram file="author-calculations.py" title="Read the complete trainable PyTorch models and retained study" /><SparseProgram file="mechanism-calculations.py" title="Read the independent numerical mechanisms and samplers" />'],
  ],
});
fs.writeFileSync(`src/learn/data/topics/${id}.jsx`, `// Full prepared manuscript rendered by render-sparse-attention-lesson.mjs.
import { Prose, H2, H3, CodeBlock } from '../../components/content';
import { Math as InlineMath, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro } from '../../components/lesson-labs/LessonElements.jsx';
import { NeuralTable } from '../../components/lesson-labs/NeuralLessonElements.jsx';
import { SparseFigure } from '../../components/lesson-labs/SparseAttentionFigures.jsx';
import { SparseGraphLab, SparseReadLab, SparseMemoryLab, SparseRandomLab, SparseProjectionLab, SparseTilesLab, SparseForecastLab, SparseProgram } from '../../components/lesson-labs/SparseAttentionLabs.jsx';
import '../../components/lesson-labs/neural-lesson-neutral.css';
export default {
 title: 'Sparse & Linear Attention Variants',
 readTime: '~100 min read + experiments and practice',
 hasIntegratedGuide: true,
 content: () => <div className="neural-lesson neural-lesson-neutral sparse-attention-lesson"><LessonIntro prerequisites="Attention weights, vector products, matrix multiplication and a recurrent update; the local examples refresh the required shapes and normalization." sections={${JSON.stringify(sections)}}>Trace what reaches a query, what a memory retains, and which work an implementation actually avoids.</LessonIntro>
${jsx}
 </div>,
};
`);
console.log(`${id}: full ${sections.length}-section manuscript rendered`);

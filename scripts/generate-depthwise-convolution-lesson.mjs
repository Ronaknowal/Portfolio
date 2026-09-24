// Authoring-time conversion preserves the complete prepared manuscript.
import fs from 'node:fs';
import { renderPreparedLesson } from './lib/prepared-lesson-renderer.mjs';
const id = 'depthwise-separable-dilated-convolutions';
const draft = `docs/teaching/drafts/${id}/`;
const assets = `public/learn-code/${id}/`;
fs.mkdirSync(assets, { recursive: true });
for (const file of ['convolution-factorization.py', 'context-blocks.py', 'author-checks.py', 'digits-400.csv', 'calculated-inputs.json', 'author-check-results.json', 'block-check-results.json', 'data-provenance.md']) fs.copyFileSync(draft + file, assets + file);
const measured = JSON.parse(fs.readFileSync(draft + 'calculated-inputs.json'));
fs.writeFileSync(assets + 'digit-inference.json', JSON.stringify({ runs: measured.runs.filter(run => run.seed === 1).map(({ dilation, dense_state, factorizations, examples }) => ({ dilation, dense_state, factorizations: factorizations.map(({ multiplier, factorized_spatial_state }) => ({ multiplier, factorized_spatial_state })), examples })) }) + '\n');
let manuscript = fs.readFileSync(draft + 'lesson.md', 'utf8').replaceAll('\r\n', '\n');
manuscript = manuscript.replace(/<details><summary>(.*?)<\/summary>(.*?)<\/details>/g, '<details>\n<summary>$1</summary>\n\n$2\n\n</details>');
// Convert this packet's dollar delimiters to the renderer's explicit delimiters.
manuscript = manuscript.replace(/\$\$([\s\S]*?)\$\$/g, (_, math) => `\\[\n${math.trim()}\n\\]`).replace(/\$([^$\n]+)\$/g, (_, math) => `\\(${math}\\)`);
manuscript = manuscript.replace('its newly prepared [pullback program]', 'its published [pullback program]').replace('Those improved pages are still prepared, so the source links are the exact reuse contract, not a claim that their new website versions are already published.', 'The improved [Convolution, Pooling & Receptive Fields lesson](/learn/path/full-curriculum/convolution-pooling-receptive-fields?module=deep-learning-fundamentals) is published. Reuse those actual operators while this lesson opens their factorization and sampling choices.').replace('[convolution packet, `direct_conv2d`]', '[convolution program, `direct_conv2d`]');
const rendered = renderPreparedLesson(manuscript, {
  assetBase: `/learn-code/${id}/`,
  replacements: [
    ['**Visual: follow two colored channel lanes.**', '<Prose>Follow one channel through its own spatial filter, then combine the two channel results. The channel investigation below exposes each scalar term beside the editable input.</Prose>'],
    ['**Investigation: build two independently controllable outputs.**', '<DepthwiseRankLab />'],
    ['**Investigation: edit the signal under a movable stencil.**', '<DepthwiseStencilLab />'],
    ['**Visual: branch a single map into several stencils.**', '<DepthwiseContextLab />'],
    ['**Investigation: inspect a compressed prediction.**', '<Prose>Inspect the saved seed-one dense model alongside its rank-one replacement on the same real image. Source 299 (digit 1) and source 32 (digit 9) are deliberately selected disagreement cases for dilations 1 and 2 respectively; they are diagnostic selections, not random representatives. Change a pixel to compare the actual signed logits and inspect original/reconstructed filters.</Prose><DepthwiseDigitLab />'],
  ],
  additions: [
    ['Try changing B\'s first input', '<DepthwiseChannelLab />'],
    ['Repeated even rates', '<DepthwiseCoverageLab />'],
    ['These are exact operation counts', '<DepthwiseSupportingFigure kind="cost" />'],
    ['Starting at \\(r=j=1\\)', '<DepthwiseSupportingFigure kind="field" />'],
    ['For example, hard-swish', '<DepthwiseSupportingFigure kind="mobile" />'],
    ['The complete [direct-loop reference', '<DepthwiseProgram file="author-checks.py" title="Read the direct-loop spatial operator and verification program" />'],
    ['For a complete building-block program,', '<DepthwiseProgram file="context-blocks.py" title="Read the inverted residual and parallel-context program" />'],
    ['The program contains all imports,', '<DepthwiseProgram file="convolution-factorization.py" title="Read the complete training and SVD factorization program" />'],
  ],
});
fs.writeFileSync(`src/learn/data/topics/${id}.jsx`, `// Full prepared revision-3 manuscript; static JSX, no browser Markdown engine.
import { Prose, H2, H3, CodeBlock } from '../../components/content';
import { Math as InlineMath, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro } from '../../components/lesson-labs/LessonElements.jsx';
import { NeuralTable } from '../../components/lesson-labs/NeuralLessonElements.jsx';
import { DepthwiseChannelLab, DepthwiseRankLab, DepthwiseStencilLab, DepthwiseCoverageLab, DepthwiseContextLab, DepthwiseDigitLab, DepthwiseProgram, DepthwiseSupportingFigure } from '../../components/lesson-labs/DepthwiseConvolutionLabs.jsx';
export default {
  title: 'Depthwise Separable & Dilated Convolutions',
  readTime: '~65 min read + experiments and practice',
  hasIntegratedGuide: true,
  content: () => <div className="neural-lesson depthwise-lesson"><LessonIntro prerequisites="Weighted sums and the preceding convolution lesson; spatial and channel axes, rank and receptive-field units are refreshed locally." sections={${JSON.stringify(rendered.sections)}}>Separate spatial filtering, channel mixing and sampling reach; inspect their costs and an actual compressed model.</LessonIntro>
${rendered.jsx}
  </div>,
};
`);
console.log(`Depthwise: ${rendered.sections.length} sections; six live investigations; complete prepared content retained.`);

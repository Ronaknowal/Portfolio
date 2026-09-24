// Authoring-only: retain the full prepared manuscript; replace each specified visual home.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import { renderPreparedLesson } from './lib/prepared-lesson-renderer.mjs';
const id = 'convolution-pooling-receptive-fields';
let manuscript = fs.readFileSync(`docs/teaching/drafts/${id}/lesson.md`, 'utf8');
manuscript = manuscript.replace(/<details><summary>(.*?)<\/summary>(.*?)<\/details>/g, '<details>\n<summary>$1</summary>\n\n$2\n\n</details>').replaceAll('<br>', '; ');
const visualNames = ['ConvolutionPatchLab', 'ConvolutionChannelsLab', 'ConvolutionUpdateLab', 'ConvolutionGeometryLab', 'ConvolutionPoolingLab', 'ConvolutionReceptiveLab', 'ConvolutionMeasuredLab', 'ConvolutionInfluenceLab', 'ConvolutionTransposeLab'];
const visualParagraphs = manuscript.match(/^\[Visual placement:.*\]$/gm);
assert.equal(visualParagraphs.length, visualNames.length);
const rendered = renderPreparedLesson(manuscript, {
  assetBase: `/learn-code/${id}/`,
  replacements: visualParagraphs.map((paragraph, i) => [paragraph, `<${visualNames[i]} />`]),
  additions: [
    ['It needs no GPU,', '<ConvolutionProgram file="convolution-experiments.py" title="Read the complete convolution and digit experiment program" />'],
    ['Max pooling can tolerate', '<ConvolutionShiftLab />'],
    ['Folding those columns back', '<ConvolutionPatchMatrixFigure />'],
    ['Run `python convolution_pullbacks.py`', '<ConvolutionProgram file="convolution_pullbacks.py" title="Read the complete vectorized convolution and pooling pullbacks" />'],
  ],
});
assert.ok(!rendered.jsx.includes('[Visual placement:'));
fs.writeFileSync(`src/learn/data/topics/${id}.jsx`, `// Full prepared revision-3 manuscript rendered statically; no Markdown parser in the browser.
import { Prose, H2, H3, CodeBlock } from '../../components/content';
import { Math as InlineMath, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro } from '../../components/lesson-labs/LessonElements.jsx';
import { NeuralTable } from '../../components/lesson-labs/NeuralLessonElements.jsx';
import { ${[...visualNames, 'ConvolutionShiftLab', 'ConvolutionPatchMatrixFigure', 'ConvolutionProgram'].join(', ')} } from '../../components/lesson-labs/ConvolutionLabs.jsx';
export default {
  title: 'Convolution, Pooling & Receptive Fields',
  readTime: '~65 min read + experiments and practice; optional implementation branches ~30 min',
  hasIntegratedGuide: true,
  content: () => <div className="neural-lesson convolution-lesson">
    <LessonIntro prerequisites="Multiplication, sums and array indexing. Weighted sums, gradients and the objective are refreshed locally; earlier neural and tensor lessons provide deeper prerequisites." sections={${JSON.stringify(rendered.sections)}}>Follow one patch, one shared update and a complete digit classifier before the deeper influence, transpose and implementation branches.</LessonIntro>
${rendered.jsx}
  </div>,
};
`);
console.log(`Convolution: ${rendered.sections.length} complete sections and all ${visualParagraphs.length} specified manuscript placements rendered.`);

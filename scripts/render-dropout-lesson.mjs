// Authoring tool: conserves the prepared manuscript and inserts topic-owned labs.
import fs from 'node:fs';
import { renderPreparedLesson } from './lib/prepared-lesson-renderer.mjs';

const folder = 'docs/teaching/drafts/dropout-droppath-stochastic-depth/';
let manuscript = fs.readFileSync(folder + 'lesson.md', 'utf8');
// The prepared packet uses both supported math delimiters and compact HTML
// disclosures. Normalize those two forms at authoring time, never in the browser.
manuscript = manuscript.replace(/(?<!\\)\$([^$\n]+)\$/g, (_, math) => '\\(' + math + '\\)');
manuscript = manuscript.replace(/(<details>|<\/details>|<summary>[^<]*<\/summary>)/g, '\n\n$1\n\n');
const { jsx, sections } = renderPreparedLesson(manuscript, {
  assetBase: '/learn-assets/dropout-droppath-stochastic-depth/',
  additions: [
    ['That is not a promise', '<DropoutUpdateLab />'],
    ['At \\(p=0\\)', '<DropoutExpectationLab />'],
    ['The difference is more than appearance.', '<DropoutGeometryLab />'],
    ['For a sampled branch bit,', '<DropoutBranchLab />'],
    ['In the expression', '<DropoutDepthLab />'],
    ['For MC dropout,', '<DropoutModeLab />'],
    ['Read `mask_values`', '<DropoutProgram title="Read the scratch mask and its axis contract" start="def mask_values" end="def fixtures" />'],
    ['This standalone code needs compatible', '<DropoutProgram file="dropout-library-checks.py" title="Read the matched library checks and locked-feature solution" />'],
    ['The recorded run used', '<DropoutProgram />'],
    ['**Investigate:**', '<DropoutMeasuredLab />'],
    ['In the actual run,', '<DropoutMonteCarloLab />'],
  ],
});
const lesson = `// Full prepared revision-3 manuscript, rendered at authoring time.
import { Prose, H2, H3, CodeBlock } from '../../components/content';
import { Math as InlineMath, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro } from '../../components/lesson-labs/LessonElements.jsx';
import { NeuralTable } from '../../components/lesson-labs/NeuralLessonElements.jsx';
import { DropoutUpdateLab, DropoutExpectationLab, DropoutGeometryLab, DropoutBranchLab, DropoutDepthLab, DropoutModeLab, DropoutMeasuredLab, DropoutMonteCarloLab, DropoutProgram } from '../../components/lesson-labs/DropoutLabs.jsx';

export default {
  title: 'Dropout, DropPath & Stochastic Depth',
  readTime: '~65 min read + experiments and practice',
  hasIntegratedGuide: true,
  content: () => <div className="neural-lesson dropout-lesson">
    <LessonIntro prerequisites="Activations, multiplication, means and a loss gradient. The previous residual-connections lesson explains the direct and correction paths; mask probabilities and tensor axes are introduced here." sections={${JSON.stringify(sections)}}>Follow a mask through values, gradients, network geometry and real training evidence.</LessonIntro>
${jsx}
  </div>,
};
`;
fs.writeFileSync('src/learn/data/topics/dropout-droppath-stochastic-depth.jsx', lesson);
console.log('Rendered full Dropout manuscript with eight live investigations.');

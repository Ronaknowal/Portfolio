// Static authoring conversion; no Markdown parser or training loop ships to the reader.
import { readFileSync, writeFileSync } from 'node:fs';
import { renderPreparedLesson } from './lib/prepared-lesson-renderer.mjs';
const id = 'landmark-architectures-lenet-alexnet-vgg-resnet-efficientnet';
const assets = `public/learn-assets/${id}/`;
const packet = JSON.parse(readFileSync(`docs/teaching/drafts/${id}/calculated-inputs.json`));
const replay = JSON.parse(readFileSync(assets + 'calculated-inputs.json'));
if (JSON.stringify(packet) !== JSON.stringify(replay)) throw new Error('Native replay differs from conserved manuscript; reconcile before rendering.');
const compact = { exact: replay.exact, fits: replay.fits.map(({ observations, head_weight, head_bias, ...rest }) => rest) };
writeFileSync('src/learn/data/landmark-architecture-measurements.json', JSON.stringify(compact) + '\n');
writeFileSync(assets + 'recorded-feature-maps.json', JSON.stringify(replay.fits.filter(row => row.seed === 1).map(({ kind, observations, head_weight, head_bias }) => ({ kind, observations, head_weight, head_bias }))) + '\n');
let manuscript = readFileSync(`docs/teaching/drafts/${id}/lesson.md`, 'utf8');
manuscript = manuscript.replace(/<!--.*?-->/g, '').replace(/<details><summary>(.*?)<\/summary>(.*?)<\/details>/g, '<details>\n<summary>$1</summary>\n\n$2\n\n</details>');
manuscript = manuscript.replace(/\]\(\.\.\/([^/]+)\/lesson\.md\)/g, '](/learn/path/full-curriculum/$1?module=deep-learning-fundamentals)');
manuscript = manuscript.replace('<summary>Check your prediction</summary>', '<summary>Worked reasoning</summary>');
const rendered = renderPreparedLesson(manuscript, {
  assetBase: `/learn-assets/${id}/`,
  replacements: [
    ['**Budget investigation — where did the memory go?**', '<LandmarkHeadLab />'],
    ['**Context-gate investigation.**', '<LandmarkContextLab />'],
    ['**Scaling investigation.**', '<LandmarkScalingLab />'],
    ['**Result investigation.**', '<LandmarkComparisonLab />'],
    ['**Map investigation.**', '<LandmarkScoreLab />'],
    ['With compatible Torchvision installed,', `<Prose>With <code>torchvision==0.29.0</code> paired with <code>torch==2.14.0</code>, run <code>python landmark_builders.py --family resnet18 --compare</code>. It constructs the ordinary <code>get_model(..., weights=None)</code> route too. <code>copy_components</code> copies every convolution, linear layer and BatchNorm state in semantic order, refusing a component count/type/shape mismatch. Both models are in evaluation mode and receive the same input. Actual separate CPU comparisons for ResNet-18, EfficientNet-B0, VGG-16 and AlexNet each produced maximum absolute logit difference 0 in the recorded environment. This compares matched random weights, not trained quality. The VGG/AlexNet commands allocate both full models; run those families individually when memory permits, never in the browser.</Prose>`],
  ],
  additions: [
    ['**Reading the feature pyramid.**', '<LandmarkFeatureRoute />'],
    ['The historical model used', '<LandmarkHistoricalShapes />'],
    ['The same support does', '<LandmarkKernelFigure />'],
    ['**Add versus concatenate.**', '<LandmarkBranchFigure />'],
    ['This reverses the wide', '<LandmarkInvertedRoute />'],
    ['The model source is intentionally', '<LandmarkProgram file="landmark_builders.py" title="Read all five explicit architecture builders and the library comparison" /><Prose>The reusable mobile block accepts branch-drop probability from 0 through 1. At 1 it keeps only the identity path during training without dividing by zero; out-of-range values are rejected. Default B0 probabilities remain below 1. <a href={landmarkAsset + "native-verification.json"}>Executed construction, boundary and environment checks</a>.</Prose>'],
    ['This does not turn a random', '<Prose>The photograph/IMAGENET1K_V1 download example is a complete usage route, but it was not executed here: no external photograph or pretrained weights were downloaded. The offline matched-state construction comparisons above were executed. To reproduce this environment, create and activate a separate Python environment, then install <code>torch==2.14.0 torchvision==0.29.0 numpy==2.3.5 scikit-learn==1.9.1 pillow</code> with <code>python -m pip install</code>.</Prose>'],
    ['The complete program defines every body,', '<LandmarkProgram /><p><a href={landmarkAsset + "calculated-inputs.json"}>Complete measured experiment record</a> · <a href={landmarkAsset + "native-verification.json"}>Execution evidence and limits</a></p>'],
    ['The saved features, all ten class maps,', '<LandmarkRecordedMaps />'],
  ],
});
writeFileSync(`src/learn/data/topics/${id}.jsx`, `// Conserved revision-3 manuscript statically rendered at authoring time.
import { Prose, H2, H3, CodeBlock } from '../../components/content';
import { Math as InlineMath, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro } from '../../components/lesson-labs/LessonElements.jsx';
import { NeuralTable } from '../../components/lesson-labs/NeuralLessonElements.jsx';
import { LandmarkFeatureRoute, LandmarkHistoricalShapes, LandmarkKernelFigure, LandmarkBranchFigure, LandmarkInvertedRoute, LandmarkHeadLab, LandmarkContextLab, LandmarkScalingLab, LandmarkComparisonLab, LandmarkScoreLab, LandmarkRecordedMaps, LandmarkProgram, landmarkAsset } from '../../components/lesson-labs/LandmarkArchitectureLabs.jsx';
export default {
 title: 'Landmark Architectures: LeNet, AlexNet, VGG, ResNet & EfficientNet',
 readTime: '~80 min read + code, investigations and practice; optional historical branches',
 hasIntegratedGuide: true,
 content: () => <div className="neural-lesson landmark-lesson"><LessonIntro prerequisites="Convolution shapes and receptive fields, residual paths, normalization, dropout, initialization and the training loop. Each architectural mechanism and evidence boundary is refreshed where used." sections={${JSON.stringify(rendered.sections)}}>Read architectures as design decisions, build their complete compositions, then compare actual outcomes under explicit resource and data contracts.</LessonIntro>
${rendered.jsx}
</div>
};
`);
console.log(`Landmark: ${rendered.sections.length} full sections, ${Buffer.byteLength(JSON.stringify(compact))}B topic-only measurements; trained maps and programs load only on disclosure.`);

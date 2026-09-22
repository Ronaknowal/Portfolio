// Conserved manuscript -> static topic JSX. No Markdown parser at runtime.
import { readFileSync, writeFileSync } from 'node:fs';
import { renderPreparedLesson } from './lib/prepared-lesson-renderer.mjs';
const id = 'convnext-modern-cnn-designs';
let manuscript = readFileSync(`docs/teaching/drafts/${id}/lesson.md`, 'utf8');
manuscript = manuscript.replace(/<details><summary>(.*?)<\/summary>(.*?)<\/details>/g, '<details>\n<summary>$1</summary>\n\n$2\n\n</details>');
const rendered = renderPreparedLesson(manuscript, {
  assetBase: `/learn-assets/${id}/`,
  replacements: [
    ['**Visual: an experiment genealogy.**','<ConvNeXtGenealogy />'],
    ['**Investigation: move the reduction boundary.**','<ConvNeXtNormalizationLab />'],
    ['**Visual: unfold the hierarchy.**','<ConvNeXtHierarchy /><ConvNeXtBudgetLab />'],
    ['**Investigation: channel maps feeding a shared denominator.**','<ConvNeXtResponseLab />'],
    ['**Visual: two paths from the same image.**','<ConvNeXtMaskFigure />'],
    ['**Investigation: which side of the information boundary did you change?**','<ConvNeXtReconstructionLab />'],
    ['**Investigation: collapse the branch graph.**','<ConvNeXtFusionLab />'],
    ['Run `python convnext_library_bridge.py`','<Prose>Run <code>python convnext_library_bridge.py</code> beside <code>convnext-blocks.py</code> with compatible PyTorch/Torchvision. The offline comparison was executed with PyTorch2.14.0+cpu and Torchvision0.29.0: V1 outputs, input gradients and all trainable gradients agree under the program’s1e−12 absolute/relative tolerances. It deliberately targets V1: Torchvision’s <code>CNBlock</code> does not become V2 merely because both are called ConvNeXt. The local <code>ResponseNorm</code> exposes the complete V2 operation, independently differentiated and checked against central differences. <a href="https://github.com/pytorch/vision/blob/v0.29.0/torchvision/models/convnext.py">Torchvision block source</a>.</Prose>'],
    ['All local experimental numbers come','<Prose>All local experimental numbers come from the accompanying programs and retained results. Read the <a href={convnextAsset+"data-provenance.md"}>dataset provenance</a> and <a href={convnextAsset+"native-verification.json"}>current native verification record</a> for execution boundaries. The six original fits are conserved, with fresh native reconstruction and independent browser-model comparisons; no ImageNet training, pretrained photograph run or hardware timing is claimed.</Prose>'],
  ],
  additions: [
    ['The two linear layers are applied','<ConvNeXtBlockFigure />'],
    ['The attached [complete architecture program]','<ConvNeXtProgram file="convnext-blocks.py" title="Read the complete V1/V2 block, hierarchy, initialization and shape checks" />'],
    ['The script sets one PyTorch CPU thread','<ConvNeXtProgram file="masked-reconstruction.py" title="Read the complete masked learning, evaluation and probe program" /><Prose>To reproduce the recorded environment in a separate activated Python environment, run <code>python -m pip install torch==2.14.0 numpy==2.3.5 scipy scikit-learn==1.9.1</code>. For the separate native block bridge, also install <code>torchvision==0.29.0 pillow</code>. The large source is loaded only when you open its disclosure; the downloaded CSV runs offline.</Prose>'],
    ['Repeated seeds vary initialization','<ConvNeXtRecordedExperiment />'],
    ['The same program accepts','<ConvNeXtProgram file="convnext_library_bridge.py" title="Read the matched-state library bridge and optional pretrained photograph route" /><Prose>The offline matched-state bridge was executed. The optional photograph route was not executed: it requires a learner-supplied image and the specified external checkpoint. Its complete source shows the ordinary transform/metadata/inference contract without assigning it an unmeasured accuracy.</Prose>'],
  ],
});
writeFileSync(`src/learn/data/topics/${id}.jsx`, `// Complete revision-3 manuscript rendered statically; changes recorded in the implementation record.
import { Prose, H2, H3, CodeBlock } from '../../components/content';
import { Math as InlineMath, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro } from '../../components/lesson-labs/LessonElements.jsx';
import { NeuralTable } from '../../components/lesson-labs/NeuralLessonElements.jsx';
import { ConvNeXtGenealogy, ConvNeXtBlockFigure, ConvNeXtNormalizationLab, ConvNeXtHierarchy, ConvNeXtBudgetLab, ConvNeXtResponseLab, ConvNeXtMaskFigure, ConvNeXtReconstructionLab, ConvNeXtRecordedExperiment, ConvNeXtFusionLab, ConvNeXtProgram, convnextAsset } from '../../components/lesson-labs/ConvNeXtLabs.jsx';
export default {
 title: 'ConvNeXt & Modern CNN Designs',
 readTime: '~70 min read + live investigations, implementation and practice',
 hasIntegratedGuide: true,
 content: () => <div className="neural-lesson convnext-lesson"><LessonIntro prerequisites="Depthwise convolution, channel normalization, residual paths, tensor shapes and supervised training. The relevant axis and masking contracts are refreshed locally." sections={${JSON.stringify(rendered.sections)}}>Read a modern convolutional block, build its complete hierarchy, and investigate how a masked reconstruction model uses available evidence.</LessonIntro>
${rendered.jsx}
</div>
};
`);
console.log(`ConvNeXt: ${rendered.sections.length} complete sections, five live mechanisms, on-demand source and small saved models.`);

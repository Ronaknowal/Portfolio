// Authoring-only renderer. The revision-3 packet remains the conserved writing input.
import { readFileSync, writeFileSync } from 'node:fs';
import { renderPreparedLesson } from './lib/prepared-lesson-renderer.mjs';
const id = 'weight-initialization-xavier-kaiming-p';
const assets = `public/learn-assets/${id}/`;
const original = JSON.parse(readFileSync(`docs/teaching/drafts/${id}/calculated-inputs.json`, 'utf8'));
const executed = JSON.parse(readFileSync(assets + 'calculated-inputs.json', 'utf8'));
if (JSON.stringify(original) !== JSON.stringify(executed)) throw new Error('Recorded execution differs from the conserved manuscript; reconcile explicitly before rendering.');
const lines = readFileSync(assets + 'digits-400.csv', 'utf8').trim().split(/\r?\n/);
const headers = lines[0].split(',');
const examples = [1, 41, 81, 121].map(index => {
  const row = Object.fromEntries(lines[index].split(',').map((value, i) => [headers[i], Number(value)]));
  return { id: row.source_id, digit: row.digit, pixels: Array.from({ length: 64 }, (_, i) => row[`pixel_${i}`]) };
});
const compact = { fixtures: executed.fixtures, cotangentRms: executed.propagation.cotangent_rms,
  propagation: executed.propagation.records.map(row => ({ seed: row.seed, scheme: row.scheme, layers: row.layers.map(layer => ({ layer: layer.layer, q: layer.second_moment, g: layer.gradient_rms })) })),
  digitFits: executed.digit_fits, widthFits: executed.width_fits, specimens: examples };
writeFileSync('src/learn/data/weight-initialization-measurements.json', JSON.stringify(compact) + '\n');
let manuscript = readFileSync(`docs/teaching/drafts/${id}/lesson.md`, 'utf8');
// Convert the two inline practice disclosures using the same paragraph renderer.
manuscript = manuscript.replace(/<details><summary>(.*?)<\/summary>(.*?)<\/details>/g, '<details>\n<summary>$1</summary>\n\n$2\n\n</details>');
const rendered = renderPreparedLesson(manuscript, {
  assetBase: `/learn-assets/${id}/`,
  replacements: [
    ['**Visual investigation: follow the signal.**', '<InitializationSignalLab />'],
    ['**Geometry investigation:**', '<InitializationGeometryLab />'],
    ['**Width investigation:**', '<InitializationWidthLab />'],
    ['The supplied comparison uses widths 32 and 96,', `<Prose>The supplied comparison uses widths 32 and 96, identical inputs and targets, zero initial optimizer state and two Adam updates. The executed run with PyTorch 2.14.0+cpu and Microsoft’s <code>mup==1.0.0</code> passes output, every parameter-gradient and updated-weight agreement at absolute/relative tolerances 1e−12. Turning parameter rescaling back on <strong>after</strong> copying the custom μP weights changes the experiment. <a href="https://github.com/microsoft/mup/blob/main/mup/layer.py">Readout</a>, <a href="https://github.com/microsoft/mup/blob/main/mup/shape.py">shape registration</a> and <a href="https://github.com/microsoft/mup/blob/main/mup/optim.py">optimizer source</a> expose the mapping. <a href={initializationAsset + 'native-verification.json'}>Executed environment and comparison record</a>.</Prose><Prose>Set up a separate environment with <code>python -m venv .venv</code>, activate it using your operating system’s command, then install <code>torch==2.14.0 numpy==2.3.5 scikit-learn==1.9.1 mup==1.0.0</code> with <code>python -m pip install</code>. Keep both Python files beside the CSV; the bridge imports <code>WidthMLP</code> without launching its training sweep.</Prose>`],
  ],
  additions: [
    ['The second moment halved.', '<InitializationMomentsLab />'],
    ['Keeping the singular values', '<InitializationSpectrumFigure />'],
    ['Run `python initialization_library_bridge.py`', '<InitializationProgram file="initialization_library_bridge.py" title="Read the complete orthogonal and μP library bridge" />'],
    ['An all-zero hidden ReLU network', '<InitializationSymmetryLab />'],
    ['After the μP derivation below,', '<Prose>The Microsoft repository was marked archived on 21 September 2026. The pinned <code>mup==1.0.0</code> API is verified for this comparison; the reference link does not imply ongoing package maintenance. Preserve the tested environment and check compatibility before using a different version.</Prose>'],
    ['The program includes imports,', '<InitializationProgram /><div className="init-code-links"><a href={initializationAsset + "calculated-inputs.json"}>Complete measured record</a><a href={initializationAsset + "native-verification.json"}>Execution checks and environment</a></div>'],
    ['Across three seeds, Kaiming gives', '<InitializationTrainingLab />'],
    ['**Small numbers and precision.**', '<InitializationPrecisionFigure />'],
  ],
});
const body = rendered.jsx; // Shared H2 already derives the matching section ID.
writeFileSync(`src/learn/data/topics/${id}.jsx`, `// Complete conserved prepared manuscript, statically rendered at authoring time.
import { Prose, H2, H3, CodeBlock } from '../../components/content';
import { Math as InlineMath, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro } from '../../components/lesson-labs/LessonElements.jsx';
import { NeuralTable } from '../../components/lesson-labs/NeuralLessonElements.jsx';
import { InitializationSignalLab, InitializationMomentsLab, InitializationGeometryLab, InitializationSpectrumFigure, InitializationSymmetryLab, InitializationTrainingLab, InitializationWidthLab, InitializationPrecisionFigure, InitializationProgram, initializationAsset } from '../../components/lesson-labs/WeightInitializationLabs.jsx';
export default {
 title: 'Weight Initialization: Xavier, Kaiming, Orthogonal Methods & μP',
 readTime: '~65 min read + experiments and practice; optional μP route ~30 min',
 hasIntegratedGuide: true,
 content: () => <div className="neural-lesson initialization-lesson"><LessonIntro prerequisites="Weighted sums, activations and backpropagation. Statistical moments and singular directions are refreshed locally; the prior loss, normalization and transfer lessons own their full mechanisms." sections={${JSON.stringify(rendered.sections)}}>Choose a starting state by the signals, directions and updates it preserves. Follow the core route first; μP adds a separate width-scaling route.</LessonIntro>
${body}
</div>
};
`);
console.log(`Initialization: ${rendered.sections.length} complete sections; ${Buffer.byteLength(JSON.stringify(compact))} bytes of topic-only measurements.`);

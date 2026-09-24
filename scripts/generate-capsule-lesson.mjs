// Authoring-only conversion and precise frozen-state packaging; nothing runs in the browser.
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { renderPreparedLesson } from './lib/prepared-lesson-renderer.mjs';
const id = 'capsule-networks', packet = `docs/teaching/drafts/${id}/`, assets = `public/learn-assets/${id}/`;
const measured = JSON.parse(readFileSync(assets + 'calculated-inputs.json'));
for (const file of ['calculated-inputs.json', 'mechanics-results.json', 'author-check-results.json']) if (JSON.stringify(JSON.parse(readFileSync(assets + file))) !== JSON.stringify(JSON.parse(readFileSync(packet + file)))) throw new Error(`${file}: native replay changed; reconcile before publication.`);
const compact = { meanImageMSE: measured.training_mean_image_mse, runs: measured.runs.map(({ first_two_examples, ...run }) => ({ ...run, inference_iterations: Object.fromEntries(Object.entries(run.inference_iterations).map(([key, { lengths, ...result }]) => [key, result])) })) };
writeFileSync('src/learn/data/capsule-measurements.json', JSON.stringify(compact) + '\n');
const state = measured.saved_models['3'], order = ['transforms', 'features.weight', 'features.bias', 'primary.weight', 'primary.bias', 'decoder.0.weight', 'decoder.0.bias', 'decoder.2.weight', 'decoder.2.bias', 'decoder.4.weight', 'decoder.4.bias'];
const values = [], layout = {};
for (const name of order) { const items = state[name].flat(Infinity); layout[name] = { offset: values.length, length: items.length }; values.push(...items); }
const binary = Buffer.alloc(values.length * 4); values.forEach((value, index) => { binary.writeFloatLE(value, index * 4); if (binary.readFloatLE(index * 4) !== value) throw new Error('Packing changed an original float32 weight'); });
writeFileSync(assets + 'frozen-model.f32', binary);
const selected = measured.runs.find(run => run.seed === 1 && run.training_iterations === 3).first_two_examples;
writeFileSync(assets + 'frozen-model.json', JSON.stringify({ model: 'seed-1-routing-3', encoding: 'little-endian IEEE754 float32; exact saved CPU weights', sha256: createHash('sha256').update(binary).digest('hex'), layout, examples: selected.images.map((image, i) => ({ image: image.flat(), source_id: selected.source_ids[i], label: selected.labels[i] })) }) + '\n');
let manuscript = readFileSync(packet + 'lesson.md', 'utf8').replace(/<details><summary>(.*?)<\/summary>/g, '<details>\n<summary>$1</summary>');
manuscript = manuscript.replace('Show the current computed result and its contributing terms immediately.', 'Inspect the current computed result and its contributing terms.');
manuscript = manuscript.replace('Show coupling rows, vote contributions, squash length/direction, current parent vectors and saved/frozen-model outputs.', 'Follow the coupling rows, vote contributions, squash length/direction, current parent vectors and saved/frozen-model outputs.');
manuscript = manuscript.replace('The university index links talks and slides,', 'The historical university index links talks and slides (it returned a gateway error during the September2026 implementation check; use the accessible author slides above if unavailable),');
const rendered = renderPreparedLesson(manuscript, { assetBase: `/learn-assets/${id}/`, replacements: [
  ['**Visual — one grid cell, several arrows.**', '<CapsuleGrouping />'],
  ['**Investigation — change a vote, then inspect the route.**', '<CapsuleVoteLab />'],
  ['**Visual — a paired evidence panel.**', '<CapsuleEvidenceLab />'],
  ['**Investigation — edit pixels and inspect the evidence.**', '<CapsuleFrozenInvestigation />'],
], additions: [
  ['Do not read the output as', '<CapsuleSquashLab />'],
  ['Save [capsule-learning.py]', '<CapsuleProgram /><Prose>The complete source below is deferred until you open it. The CSV, import helper and mechanics program are available beside it; save these linked files in one directory for the documented local commands.</Prose><p><a href={capsuleAsset + "native-verification.json"}>Actual execution, environment and numerical checks</a> · <a href={capsuleAsset + "calculated-inputs.json"}>Complete measured record</a></p>'],
  ['This is an exact calculation **because**', '<CapsuleGeometryFigure />'],
  ['The resemblance to [Gaussian-mixture EM]', '<CapsuleEMLab /><CapsuleProgram file="capsule-mechanics.py" title="Read the complete NumPy routing, derivative, geometry and EM program" />'],
  ['The paired route compares', '<CapsuleProgram file="author-checks.py" title="Read the independent NumPy encoder and saved-state comparison" />'],
  ['The classic vector CapsNet uses', '<CapsuleClassicShape />'],
] });
writeFileSync(`src/learn/data/topics/${id}.jsx`, `// Complete revision-3 prepared manuscript; statically rendered at authoring time.
import { Prose, H2, H3, CodeBlock } from '../../components/content';
import { Math as InlineMath, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro } from '../../components/lesson-labs/LessonElements.jsx';
import { NeuralTable } from '../../components/lesson-labs/NeuralLessonElements.jsx';
import { CapsuleGrouping, CapsuleVoteLab, CapsuleSquashLab, CapsuleEvidenceLab, CapsuleFrozenInvestigation, CapsuleGeometryFigure, CapsuleEMLab, CapsuleClassicShape, CapsuleProgram, capsuleAsset } from '../../components/lesson-labs/CapsuleLabs.jsx';
export default {
 title: 'Capsule Networks',
 readTime: '~70 min read + code, live investigations and practice; optional deeper mechanics',
 hasIntegratedGuide: true,
 content: () => <div className="neural-lesson capsule-lesson"><LessonIntro prerequisites="Vector addition, matrix multiplication, convolution shapes and the idea of learning through a loss. Capsule-specific vocabulary and axes are introduced here." sections={${JSON.stringify(rendered.sections)}}>Follow one image from grouped properties to votes, assignments and class vectors. Build the mechanism, train a controlled model, and test what its geometry actually supports.</LessonIntro>
${rendered.jsx}
</div>
};
`);
console.log(JSON.stringify({ sections: rendered.sections.length, compactBytes: Buffer.byteLength(JSON.stringify(compact)), frozenWeightBytes: binary.length, weights: values.length }));

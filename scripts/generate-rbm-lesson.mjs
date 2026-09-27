import fs from 'node:fs';
import path from 'node:path';
import { renderPreparedLesson } from './lib/prepared-lesson-renderer.mjs';
const id = 'boltzmann-machines-restricted-boltzmann-machines-rbm';
const draft = `docs/teaching/drafts/${id}`,
  destination = `public/learn-code/${id}`;
fs.mkdirSync(destination, {
  recursive: true
});
for (const file of ['rbm-study.py', 'bernoulli_rbm_bridge.py', 'calculated-inputs.json', 'digits-400.csv', 'data-provenance.md']) fs.copyFileSync(path.join(draft, file), path.join(destination, file));
const recorded = JSON.parse(fs.readFileSync(`${draft}/calculated-inputs.json`, 'utf8'));
const rows = fs.readFileSync(`${draft}/digits-400.csv`, 'utf8').trim().split(/\r?\n/).slice(1).map(line => line.split(',').map(Number));
const images = recorded.protocol.roles.assessment.map(id => {
  const row = rows.find(r => r[0] === id);
  return {
    id,
    digit: row[65],
    pixels: row.slice(1, 65).map(x => Number(x >= 8))
  };
});
fs.writeFileSync(`${destination}/assessment-images.json`, JSON.stringify(images));
for (const fit of recorded.fits) fs.writeFileSync(`${destination}/model-${fit.method}-${fit.seed}.json`, JSON.stringify({
  model: {
    w: fit.weights,
    a: fit.visible_bias,
    b: fit.hidden_bias
  },
  samples: fit.exact_samples,
  hiddenSamples: fit.hidden_sample_states
}));
const summary = {
  baseline: recorded.baseline,
  sharedWeightExtent: Math.max(...recorded.fits.flatMap(f => f.weights.flat().map(Math.abs))),
  fits: recorded.fits.map(f => ({
    method: f.method,
    seed: f.seed,
    nll: f.metrics.assessment.nll_nats_per_image,
    reconstruction: f.metrics.assessment.mean_reconstruction_mse,
    completionMse: f.completion_mse,
    completionCorrect: f.completion_correct,
    history: f.history.map(r => ({
      epoch: r.epoch,
      fit: r.fit.nll_nats_per_image,
      development: r.development.nll_nats_per_image
    }))
  }))
};
fs.writeFileSync('src/learn/data/rbm-study-summary.js', `// Recorded outcomes; no training runs in the browser.\nexport default ${JSON.stringify(summary)};\n`);
const native = JSON.parse(fs.readFileSync(`docs/teaching/deep-learning-completion/${id}/native-fixtures.json`, 'utf8'));
fs.writeFileSync('src/learn/data/rbm-library-model.js', `// Actual sklearn 1.9.1 fitted bridge parameters, native checked.\nexport default ${JSON.stringify(native.library[0].model)};\n`);
let manuscript = fs.readFileSync(`${draft}/lesson.md`, 'utf8').replaceAll('\r\n', '\n');
let program = 0;
manuscript = manuscript.replace(/(?:```|~~~)python\n[\s\S]*?\n(?:```|~~~)/g, () => `**Complete program ${++program}.**`);
if (program !== 2) throw Error('Expected both complete canonical programs');
manuscript = manuscript.replace(/\$\$([\s\S]*?)\$\$/g, (_, math) => `\n\\[\n${math}\n\\]\n`).replace(/\$([^$]+)\$/g, '\\($1\\)');
const {
  jsx,
  sections
} = renderPreparedLesson(manuscript, { preserveOpeningFrom: `src/learn/data/topics/${id}.jsx`,
  assetBase: `/learn-code/${id}/`,
  replacements: [['**Visual: two rows of switches and an energy ledger.**', '<RbmEnergyLab />'], ['**Visual: two co-occurrence ledgers.**', '<RbmGradientLab />'], ['**Lab: probability flow through four states.**', '<RbmChainLab />'], ['**Visual: two reconstruction arrows beside four probability bars.**', '<RbmReconstructionLab />'], ['**Lab: inspect an actual completion.**', '<RbmDigitLab />'], ['**Complete program 1.**', '<RbmProgram file="rbm-study.py" />'], ['**Complete program 2.**', '<RbmProgram file="bernoulli_rbm_bridge.py" />']],
  additions: [['We will instead use hidden variables', '<RbmGeneralFigure />'], ['Use stable logarithmic calculations.', '<RbmEnumerationFigure />'], ['Persistence does not magically', '<RbmPersistenceLab />'], ['For exact-gradient seed 11, fit/development/assessment', '<RbmStudyFigure />'], ['The saved 16 samples for every fitted model', '<RbmSampleGallery />'], ['RBMs can initialize other networks', '<RbmFamilyFigure />'], ['Under the required support, initialization', '<RbmAisFigure />'], ['A small authoring probe executed', '<RbmLibraryLab />']]
});
const components = ['RbmEnergyLab', 'RbmGradientLab', 'RbmChainLab', 'RbmReconstructionLab', 'RbmDigitLab', 'RbmProgram', 'RbmGeneralFigure', 'RbmEnumerationFigure', 'RbmPersistenceLab', 'RbmStudyFigure', 'RbmSampleGallery', 'RbmFamilyFigure', 'RbmAisFigure', 'RbmLibraryLab'];
fs.writeFileSync(`src/learn/data/topics/${id}.jsx`, `// Generated from the complete prepared manuscript by scripts/generate-rbm-lesson.mjs.\nimport { Prose,H2,H3,CodeBlock } from '../../components/content';\nimport {Math as InlineMath,MathBlock} from '../../components/content/Math.jsx';\nimport {NeuralTable} from '../../components/lesson-labs/NeuralLessonElements.jsx';\nimport {${components.join(',')}} from '../../components/lesson-labs/RbmLabs.jsx';\nexport default { title:'Boltzmann Machines & Restricted Boltzmann Machines (RBM)',readTime:'~80 min read + experiments and practice',content:()=> <div className="neural-lesson neural-lesson-neutral rbm-lesson">\n${jsx}\n</div>};\n`);
console.log(`Generated RBM: ${sections.length} sections, all9 selected-model assets, assessment images and both deferred complete programs.`);

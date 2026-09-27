import fs from 'node:fs';
import { renderPreparedLesson } from './lib/prepared-lesson-renderer.mjs';
const id = 'advanced-optimizers-lion-sophia-prodigy-schedule-free';
const source = `docs/teaching/drafts/${id}`, destination = `public/learn-assets/${id}`;
fs.mkdirSync(destination, { recursive: true });
const downloads = ['optimizer_rules.py', 'optimizer_study.py', 'optimizer_calculations.py', 'optimizer_library_bridge.py', 'digits-400.csv', 'study-results.json', 'fitted-optimizer-states.json', 'calculated-inputs.json', 'data-provenance.md', 'lion_pytorch.py', 'sophia.py', 'LICENSE-lion.txt', 'LICENSE-sophia.txt'];
for (const name of downloads) fs.copyFileSync(`${source}/${name}`, `${destination}/${name}`);
const models = JSON.parse(fs.readFileSync(`${source}/fitted-optimizer-states.json`, 'utf8'));
const study = JSON.parse(fs.readFileSync(`${source}/study-results.json`, 'utf8'));
const raw = fs.readFileSync(`${source}/digits-400.csv`, 'utf8').trim().split(/\r?\n/).slice(1).map(line => line.split(',').map(Number));
const digits = Object.fromEntries([312, 277].map(id => { const row = raw.find(r => r[0] === id); return [id, { id, pixels: row.slice(1, 65), label: row[65] }]; }));
for (const model of models) fs.writeFileSync(`${destination}/model-${model.method}-${model.seed}.json`, JSON.stringify(model));
fs.writeFileSync(`${destination}/histories.json`, JSON.stringify({ selected: study.selected, runs: study.runs }));
fs.writeFileSync('src/learn/data/advanced-optimizer-study.js', `// Two actual worked/source digit records; full model state is fetched on demand.\nexport const optimizerDigits = ${JSON.stringify(digits)};\n`);
const manuscript = fs.readFileSync(`${source}/lesson.md`, 'utf8');
const { jsx: rendered, sections } = renderPreparedLesson(manuscript, { preserveOpeningFrom: `src/learn/data/topics/${id}.jsx`, assetBase: `/learn-assets/${id}/`, replacements: [
  ['**Figure 1 —', '<OptimizerAnatomy />'], ['**Figure 2 —', '<AdamHistoryFigure />'], ['**Figure 3 —', '<LionWorkedFigure />'],
  ['**Investigation: when does the new evidence win?', '<LionDirectionLab />'], ['**Figure 4 —', '<CurvatureBowlFigure />'], ['**Figure 5 —', '<CurvatureLanes />'],
  ['**Investigation: what does the label sampler estimate?', '<SophiaInstrumentsLab />'], ['**Figure 6 —', '<ProdigyScaleLab worked />'],
  ['**Investigation: adaptation without an oracle.', '<ProdigyScaleLab />'], ['**Figure 7 —', '<ScheduleFreeLab worked />'], ['**Figure 8 —', '<ScheduleWeightsFigure />'],
  ['**Investigation: which model are you measuring?', '<ScheduleFreeLab />'], ['**Figure 9 —', '<OptimizerPixelFigure />'], ['**Figure 10 —', '<OptimizerSelectionFigure />'],
  ['**Figure 11 —', '<OptimizerHistoryFigure />'], ['**Investigation: one new image, one real update.', '<OptimizerDigitLab />'], ['**Figure 12 —', '<OptimizerMemoryLab />'], ['**Figure 13 —', '<OptimizerPolarFigure />'],
], additions: [['The full Hessian additionally includes', '<CurvatureJacobianFigure />'], ['The program prints the twelve selected assessment records', '<OptimizerProgram file="optimizer_rules.py" /><OptimizerProgram file="optimizer_study.py" /><OptimizerProgram file="optimizer_calculations.py" />']] });
const jsx = rendered.replace(/<CodeBlock language=\{"python"\}>[\s\S]*?<\/CodeBlock>/g, '<details><summary>Read the complete runnable library bridge</summary>$&</details>');
fs.writeFileSync(`src/learn/data/topics/${id}.jsx`, `// Generated from the complete prepared manuscript; semantic generator retains every section.\nimport { Prose, H2, H3, CodeBlock } from '../../components/content';\nimport { Math as InlineMath, MathBlock } from '../../components/content/Math.jsx';\nimport { NeuralTable } from '../../components/lesson-labs/NeuralLessonElements.jsx';\nimport { OptimizerAnatomy, AdamHistoryFigure, LionWorkedFigure, CurvatureBowlFigure, CurvatureLanes, ScheduleWeightsFigure, OptimizerSelectionFigure, OptimizerPolarFigure, CurvatureJacobianFigure } from '../../components/lesson-labs/AdvancedOptimizerDiagrams.jsx';\nimport { LionDirectionLab, SophiaInstrumentsLab, ProdigyScaleLab, ScheduleFreeLab, OptimizerMemoryLab } from '../../components/lesson-labs/AdvancedOptimizerLabs.jsx';\nimport { OptimizerDigitLab, OptimizerPixelFigure, OptimizerHistoryFigure, OptimizerProgram } from '../../components/lesson-labs/AdvancedOptimizerStudy.jsx';\nexport default { title: 'Advanced Optimizers: Lion, Sophia, Prodigy and Schedule-Free', readTime: '~90 min read + investigations and practice', content: () => <div className="neural-lesson neural-lesson-neutral advanced-optimizer-lesson">\n${jsx}\n</div> };\n`);
console.log(`Generated ${sections.length} complete optimizer sections,12 separate model snapshots, measured histories and explicit downloads.`);

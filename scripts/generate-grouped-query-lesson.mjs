import fs from 'node:fs';
import path from 'node:path';
import { renderPreparedLesson } from './lib/prepared-lesson-renderer.mjs';

const id = 'grouped-query-attention-gqa-multi-query-attention-mqa';
const draft = `docs/teaching/drafts/${id}`;
const destination = `public/learn-assets/${id}`;
fs.mkdirSync(destination, { recursive: true });
const downloads = ['author-calculations.py', 'author-results.json', 'data-provenance.md', 'forecast-models.json', 'mechanism-calculations.py', 'mechanism-fixtures.json', 'movement_libras.data', 'movement_libras.names'];
for (const name of downloads) fs.copyFileSync(path.join(draft, name), path.join(destination, name));
const models = JSON.parse(fs.readFileSync(`${draft}/forecast-models.json`, 'utf8'));
const study = JSON.parse(fs.readFileSync(`${draft}/author-results.json`, 'utf8'));
fs.writeFileSync('src/learn/data/grouped-query-study.js', `// Measured validation histories from the prepared, source-bound study.\nexport default ${JSON.stringify(Object.fromEntries(Object.entries(study.branches).map(([heads, branch]) => [heads, { selected: branch.selected_step, history: branch.history.map(row => [row.step, row.validation_mse_transformed]) }])))};\n`);
const points = fs.readFileSync(`${draft}/movement_libras.data`, 'utf8').trim().split(/\r?\n/)[76].split(',').slice(0, 90).map(Number);
for (const [heads, model] of Object.entries(models)) {
  fs.writeFileSync(`${destination}/runtime-${heads}.json`, JSON.stringify({ weights: model.weights, points: Array.from({ length: 45 }, (_, i) => points.slice(i * 2, i * 2 + 2)) }));
}
const original = fs.readFileSync(`${draft}/lesson.md`, 'utf8');
let inCode = false;
const manuscript = original.split('\n').map(line => {
  if (line.startsWith('```')) inCode = !inCode;
  return inCode ? line : line.replace(/\$([^$]+)\$/g, '\\($1\\)');
}).join('\n');
const { jsx, sections } = renderPreparedLesson(manuscript, { preserveOpeningFrom: `src/learn/data/topics/${id}.jsx`,
  assetBase: `/learn-assets/${id}/`,
  replacements: [
    ['**Figure — write once, read again.**', '<GqaCacheTimeline />'],
    ['**Figure — wiring, then computation.**', '<GqaWiring />'],
    ['**Investigation — edit a shared memory.**', '<GqaReadLab />'],
    ['**Figure — cache growth with two readers per group.**', '<><GqaCompactWriteDiagram /><GqaMaskLab /></>'],
    ['**Investigation — account for every axis.**', '<GqaBudgetLab />'],
    ['**Investigation — merge two learned descriptions.**', '<GqaConversionLab />'],
    ['**Investigation — a shared-memory forecast workspace.**', '<GqaForecastLab />'],
    ['**Figure — several gradient contributions meet at one parameter.**', '<GqaGradientFigure />'],
  ],
  additions: [
    ['For this simplified equal partition,', '<GqaDistributedDiagram />'],
    ['**Mixture-of-experts models.**', '<GqaOwnershipDiagram />'],
    ['Validation RMSE after selection', '<GqaTrainingHistory />'],
    ['Read `CausalForecaster.forward`', '<GqaProgram />'],
    ['This is a call-site fragment;', '<GqaProgram file="mechanism-calculations.py" />'],
  ],
});
fs.writeFileSync(`src/learn/data/topics/${id}.jsx`, `// Generated from the complete prepared manuscript by scripts/generate-grouped-query-lesson.mjs.\nimport { Prose, H2, H3, CodeBlock } from '../../components/content';\nimport { Math as InlineMath, MathBlock } from '../../components/content/Math.jsx';\nimport { NeuralTable } from '../../components/lesson-labs/NeuralLessonElements';\nimport { GqaCacheTimeline, GqaWiring, GqaReadLab, GqaMaskLab, GqaBudgetLab, GqaConversionLab, GqaForecastLab, GqaGradientFigure, GqaProgram, GqaTrainingHistory } from '../../components/lesson-labs/GroupedQueryLabs';\nimport { GqaCompactWriteDiagram, GqaOwnershipDiagram, GqaDistributedDiagram } from '../../components/lesson-labs/GroupedQueryDiagrams.jsx';\nimport '../../components/lesson-labs/neural-lesson-neutral.css';\nconst lesson = { title: 'Grouped-Query Attention (GQA) & Multi-Query Attention (MQA)', readTime: '~65 min read + 90 min practice', content: () => <div className="gqa-lesson neural-lesson neural-lesson-neutral">\n${jsx}\n</div> };\nexport default lesson;\n`);
console.log(`Generated complete GQA lesson: ${sections.length} sections; three selected-model assets; source downloads preserved.`);

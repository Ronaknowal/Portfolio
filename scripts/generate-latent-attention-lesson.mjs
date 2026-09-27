import fs from 'node:fs';
import { renderPreparedLesson } from './lib/prepared-lesson-renderer.mjs';

const id = 'multi-head-latent-attention-mla', draft = `docs/teaching/drafts/${id}`, destination = `public/learn-assets/${id}`;
fs.mkdirSync(destination, { recursive: true });
const downloads = ['author-calculations.py', 'author-results.json', 'data-provenance.md', 'forecast-model.json', 'mechanism-calculations.py', 'mechanism-fixtures.json', 'mla_sdpa_bridge.py', 'movement_libras.data', 'movement_libras.names'];
for (const file of downloads) fs.copyFileSync(`${draft}/${file}`, `${destination}/${file}`);
const model = JSON.parse(fs.readFileSync(`${draft}/forecast-model.json`, 'utf8'));
const values = fs.readFileSync(`${draft}/movement_libras.data`, 'utf8').trim().split(/\r?\n/)[76].split(',').slice(0, 90).map(Number);
fs.writeFileSync(`${destination}/runtime.json`, JSON.stringify({ weights: model.weights, completeBasis: model.complete_basis, points: Array.from({ length: 45 }, (_, i) => values.slice(2 * i, 2 * i + 2)) }));
let inCode = false;
const manuscript = fs.readFileSync(`${draft}/lesson.md`, 'utf8').split('\n').map(line => {
  if (line.startsWith('```')) inCode = !inCode;
  return inCode ? line : line.replace(/\$([^$]+)\$/g, '\\($1\\)');
}).join('\n');
const { jsx, sections } = renderPreparedLesson(manuscript, { preserveOpeningFrom: `src/learn/data/topics/${id}.jsx`,
  assetBase: `/learn-assets/${id}/`,
  replacements: [
    ['**Figure — a common description, different readings.**', '<LatentStorageFigure />'],
    ['**Investigation — move a rotation through a projection.**', '<LatentRotationLab />'],
    ['**Figure — two score contributions meet.**', '<LatentScoreFigure />'],
    ['**Investigation — two paths, one answer.**', '<LatentPathsLab />'],
    ['**Investigation — construct a cache record.**', '<LatentBudgetLab />'],
    ['**Investigation — inspect the compressed history.**', '<LatentForecastLab />'],
    ['**Figure — discarded energy versus discarded information.**', '<LatentRankLab />'],
  ],
  additions: [
    ['For one head, the unnormalized content-score matrix', '<LatentSoftmaxFigure />'],
    ['Read `LatentForecaster.forward`', '<LatentProgram file="author-calculations.py" />'],
    ['Pass `scale=1/sqrt(P+R)`', '<LatentProgram file="mla_sdpa_bridge.py" />'],
    ['It produces the two head outputs', '<LatentProgram file="mechanism-calculations.py" />'],
  ],
});
fs.writeFileSync(`src/learn/data/topics/${id}.jsx`, `// Generated from the complete prepared manuscript by scripts/generate-latent-attention-lesson.mjs.\nimport { Prose, H2, H3, CodeBlock } from '../../components/content';\nimport { Math as InlineMath, MathBlock } from '../../components/content/Math.jsx';\nimport { NeuralTable } from '../../components/lesson-labs/NeuralLessonElements';\nimport { LatentStorageFigure, LatentScoreFigure, LatentPathsLab, LatentRotationLab, LatentBudgetLab, LatentRankLab, LatentSoftmaxFigure, LatentForecastLab, LatentProgram } from '../../components/lesson-labs/LatentAttentionLabs';\nimport '../../components/lesson-labs/neural-lesson-neutral.css';\nconst lesson = { title: 'Multi-Head Latent Attention (MLA)', readTime: '~65 min read + 90 min practice', content: () => <div className="mla-lesson neural-lesson neural-lesson-neutral">\n${jsx}\n</div> };\nexport default lesson;\n`);
console.log(`Generated complete MLA manuscript with ${sections.length} sections, seven diagram/lab anchors and three complete program views.`);

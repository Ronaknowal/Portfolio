import fs from 'node:fs';
import { renderPreparedLesson } from './lib/prepared-lesson-renderer.mjs';

const id = 'positional-encodings-sinusoidal-learned-rope-alibi';
const source = `docs/teaching/drafts/${id}`, destination = `public/learn-code/${id}`;
fs.mkdirSync(destination, { recursive: true });
for (const file of ['author-calculations.py', 'author-results.json', 'mechanism-calculations.py', 'mechanism-fixtures.json', 'position_library_bridge.py', 'position-models.json', 'movement_libras.data', 'movement_libras.names', 'data-provenance.md']) fs.copyFileSync(`${source}/${file}`, `${destination}/${file}`);
const models = JSON.parse(fs.readFileSync(`${source}/position-models.json`));
for (const [mode, model] of Object.entries(models)) fs.writeFileSync(`${destination}/movement-${mode}.json`, JSON.stringify({ mode, state_dict: model.state_dict, points: model.points, positions: model.positions, baseline: model.baseline }));
let manuscript = fs.readFileSync(`${source}/lesson.md`, 'utf8').replaceAll('\r\n', '\n');
// This manuscript uses single-dollar inline math; convert only outside code fences.
manuscript = manuscript.split(/(```[\s\S]*?```)/g).map((part, i) => i % 2 ? part : part.replace(/\$([^$\n]+)\$/g, (_, formula) => `\\(${formula}\\)`)).join('');
const rendered = renderPreparedLesson(manuscript, { preserveOpeningFrom: `src/learn/data/topics/${id}.jsx`,
  assetBase: `/learn-code/${id}/`,
  replacements: [
    ['**Figure — the same points, two journeys.', '<PositionJourneyFigure />'],
    ['**Figure — clock hands, traces and a coordinate table.', '<SinusoidalPositionLab />'],
    ['**Figure — two clocks with arrows inside.', '<RotaryPositionLab />'],
    ['**Investigation — a competition between evidence and distance.', '<AlibiPositionLab />'],
    ['**Investigation — reattach a trajectory to its slots.', '<PositionMovementLab />'],
    ['**Investigation — repair the cache timeline.', '<PositionCacheLab />'],
    ['**Figure — frequency ruler.', '<PositionFrequencyFigure />'],
    ['**Investigation — stretch a position system.', '<PositionExtensionLab />'],
  ],
  additions: [
    ['Do not silently replace an out-of-range index', '<AdditivePositionLab />'],
    ['These methods are alternatives', '<RelativeBucketFigure />'],
    ['The centering choice can improve numerical range', '<XposPositionFigure />'],
    ['These examples explain why there is no universal', '<PositionApplicationsFigure />'],
  ],
});
const components = ['PositionJourneyFigure', 'AdditivePositionLab', 'SinusoidalPositionLab', 'RotaryPositionLab', 'AlibiPositionLab', 'RelativeBucketFigure', 'PositionMovementLab', 'PositionCacheLab', 'PositionFrequencyFigure', 'PositionExtensionLab', 'PositionApplicationsFigure', 'XposPositionFigure'];
fs.writeFileSync(`src/learn/data/topics/${id}.jsx`, `// Generated from the complete manuscript by scripts/generate-positional-encoding-lesson.mjs.
import { Prose, H2, H3, CodeBlock } from '../../components/content';
import { Math as InlineMath, MathBlock } from '../../components/content/Math.jsx';
import { LessonIntro } from '../../components/lesson-labs/LessonElements.jsx';
import { NeuralTable } from '../../components/lesson-labs/NeuralLessonElements.jsx';
import { ${components.join(', ')} } from '../../components/lesson-labs/PositionalEncodingLabs.jsx';
export default {
  title: 'Positional Encodings: Sinusoidal, Learned, RoPE and ALiBi',
  readTime: '~90 min read + experiments and practice',
  hasIntegratedGuide: true,
  content: () => <div className="neural-lesson neural-lesson-neutral positional-encoding-lesson"><LessonIntro prerequisites="Query/key/value attention and Transformer blocks. Rotations, relative distances and cache coordinates are developed locally." sections={${JSON.stringify(rendered.sections)}}>An ordered array is useful only when the model can use its order. Follow where position enters the actual computation.</LessonIntro>
${rendered.jsx}
  </div>,
};
`);
console.log(`Generated ${id}; ${rendered.sections.length} sections, selected-model assets and complete learner downloads.`);

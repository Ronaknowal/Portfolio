import fs from 'node:fs';
import assert from 'node:assert/strict';
import { parse } from '@babel/parser';
import { alibiCompetition, alibiSlopes, cachePositionRead, extensionFrequencies, learnedPosition, positionCacheDefault, positionMovementForward, relativeBucket, rotate, sinusoidal } from '../src/learn/data/positional-encoding-models.js';
import { dot, maxDifference } from '../src/learn/data/sequence-tensor-operations.js';

const id = 'positional-encodings-sinusoidal-learned-rope-alibi';
const source = `docs/teaching/drafts/${id}`, out = `docs/teaching/deep-learning-completion/${id}`;
const fixture = JSON.parse(fs.readFileSync(`${source}/mechanism-fixtures.json`)), models = JSON.parse(fs.readFileSync(`${source}/position-models.json`));
const checks = [];
function close(name, actual, expected, tolerance = 1e-10) {
  const error = maxDifference(actual, expected);
  assert.ok(Number.isFinite(error) && error <= tolerance, `${name}: ${error} > ${tolerance}`);
  checks.push({ name, passed: true, maxAbsoluteError: error, tolerance });
}
for (const [width, length] of [[8, 4], [32, 16]]) close(`All sinusoidal entries width ${width}`, Array.from({ length }, (_, i) => sinusoidal(i, width)), fixture[`sinusoidal_width${width}_positions0to${length - 1}`]);
const geometry = fixture.geometry;
close('RoPE rotated query', rotate(geometry.query, 3), geometry.query3);
close('RoPE rotated key', rotate(geometry.key, 7), geometry.key7);
for (const shift of [0, 100, -120]) close(`RoPE shared shift ${shift}`, [dot(rotate(geometry.query, 3 + shift), rotate(geometry.key, 7 + shift))], [geometry.dot]);
for (const changed of [false, true]) {
  const values = Array.from({ length: 4 }, (_, i) => geometry.query.map(value => value * (changed && i === 1 ? 2 : 1)));
  const rotated = values.map((row, i) => rotate(row, i));
  close(`RoPE ${changed ? 'changed' : 'constant'} content`, rotated.map(row => rotated.map(key => dot(row, key))), geometry[changed ? 'changing_content' : 'constant_content']);
}
close('Original ALiBi three-head schedule', alibiSlopes(3), [.0625, .00390625, .25]);
close('ALiBi competition', alibiCompetition([2, 0, 0, 0], [0, 1, 2, 3]).weights, fixture.alibi.with_bias);
close('ALiBi no-penalty control', alibiCompetition([2, 0, 0, 0], [0, 1, 2, 3], 3, 0).weights, fixture.alibi.without_bias);
close('ALiBi equal-content', alibiCompetition([0, 0, 0, 0], [0, 1, 2, 3]).weights, fixture.alibi.equal_content);
assert.equal(alibiCompetition([8, 8], [4, 5], 3).weights, null);
close('T5 signed buckets', fixture.t5.offsets.map(relativeBucket), fixture.t5.buckets, 0);
for (const mode of ['rope', 'alibi']) {
  const settings = { ...positionCacheDefault(), mode };
  close(`${mode} cached last`, cachePositionRead(settings).output, fixture.cache[mode].cached_last[0]);
  close(`${mode} common-ID shift`, cachePositionRead({ ...settings, ids: [107, 108, 109], queryId: 109, maskId: 109, rotaryId: 109 }).output, fixture.cache[mode].cached_last[0]);
  close(`${mode} cache storage permutation`, cachePositionRead({ ...settings, ids: [...settings.ids].reverse(), keys: [...settings.keys].reverse(), values: [...settings.values].reverse() }).output, fixture.cache[mode].cached_last[0]);
}
close('Wrong query rotation with correct legal mask', cachePositionRead({ ...positionCacheDefault(), rotaryId: 0 }).output, fixture.cache.wrong_query_offset.output[0]);
assert.equal(cachePositionRead({ ...positionCacheDefault(), maskId: 0 }).output, null);
close('Mixed-frequency cache', cachePositionRead({ ...positionCacheDefault(), base: 100 }).output, fixture.cache.mixed_frequency.stale_output[0]);
close('Consistent new frequencies', cachePositionRead({ ...positionCacheDefault(), base: 100 }, true).output, fixture.cache.mixed_frequency.fresh_output[0]);
const extended = extensionFrequencies();
for (const [name, key] of [['original', 'frequency'], ['pi', 'pi_frequency'], ['baseScaled', 'ntk_frequency'], ['yarn', 'yarn_paper_ramp_frequency']]) close(`Extension ${name}`, extended[name], fixture.frequencies[key]);
for (const width of [4, 8, 64, 128]) {
  const unchanged = extensionFrequencies(width, 10000, 4096, 1);
  for (const name of ['pi', 'baseScaled', 'yarn']) close(`s=1 width ${width} ${name}`, unchanged[name], unchanged.original);
}
assert.throws(() => sinusoidal(3, 3));
assert.throws(() => learnedPosition([[1, 2]], 1));
assert.throws(() => learnedPosition([[1, 2]], .5));
for (const [mode, saved] of Object.entries(models)) {
  const model = { ...saved, mode }, original = positionMovementForward(model, saved.points, saved.positions);
  close(`${mode} saved original logits`, original.logits, saved.baseline.logits, 2e-4);
  const cases = {
    paired_permutation: [[...saved.points].reverse(), [...saved.positions].reverse(), []],
    reversed_points: [[...saved.points].reverse(), saved.positions, []],
    point_edit: [saved.points.map((row, i) => i === 22 ? [1 - row[0], row[1]] : row), saved.positions, []],
    masked_padding: [[...saved.points, ...Array.from({ length: 5 }, () => [.75, .75])], [...saved.positions, 0, 0, 0, 0, 0], [...Array(45).fill(false), ...Array(5).fill(true)]],
    unmasked_padding: [[...saved.points, ...Array.from({ length: 5 }, () => [.75, .75])], [...saved.positions, 0, 0, 0, 0, 0], []],
  };
  for (const [name, args] of Object.entries(cases)) close(`${mode} ${name}`, positionMovementForward(model, ...args).logits, saved[name].logits, 2e-4);
  const map = { rawQuery: 'raw_query', rawKey: 'raw_key', query: 'query', key: 'key', value: 'value', content: 'content_logits', attention: 'attention' };
  for (const [key, reference] of Object.entries(map)) close(`${mode} both-head ${key}`, original.heads.map(head => head[key]), saved.trace[reference], 2e-4);
  close(`${mode} pooled features`, original.pooled, saved.trace.pooled, 2e-4);
  assert.throws(() => positionMovementForward(model, [], []));
  assert.throws(() => positionMovementForward(model, saved.points, saved.positions, Array(45).fill(true)));
}
for (const file of [`src/learn/data/topics/${id}.jsx`, 'src/learn/components/lesson-labs/PositionalEncodingLabs.jsx']) {
  parse(fs.readFileSync(file, 'utf8'), { sourceType: 'module', plugins: ['jsx'] });
  checks.push({ name: `JSX parse ${file}`, passed: true });
}
fs.mkdirSync(out, { recursive: true });
fs.writeFileSync(`${out}/model-checks.json`, JSON.stringify({ topicId: id, passed: true, checks, limits: ['All five retained float32 models compared to JS float64 with bounded GELU approximation.', 'Geometry and recorded inference, not long-context performance or fresh training.'] }, null, 2) + '\n');
console.log(`${checks.length} positional mechanism, native-fixture, retained-model and syntax checks passed.`);

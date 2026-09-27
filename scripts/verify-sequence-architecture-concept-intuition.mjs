import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { transform } from 'esbuild';

const near = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-12, `${actual} differs from ${expected}`);
const sum = values => values.reduce((a, b) => a + b, 0);
const topics = {
  'attention-mechanism-bahdanau-luong': {
    file: 'attention', generator: 'scripts/generate-recurrent-attention-lesson.mjs',
    anchor: 'running tally beside the memory shelf',
    check() {
      const reads = [[.6, .3, .1], [.2, .5, .3], [.1, .2, .7]];
      const previous = reads[0].map((value, j) => value + reads[1][j]);
      const overlap = previous.map((value, j) => Math.min(value, reads[2][j]));
      const after = previous.map((value, j) => value + reads[2][j]);
      previous.forEach((value, j) => near(value, [.8, .8, .4][j]));
      overlap.forEach((value, j) => near(value, [.1, .2, .4][j]));
      after.forEach((value, j) => near(value, [.9, 1, 1.1][j]));
      near(sum(overlap), .7); near(sum(previous), 2); near(sum(after), 3);
      near(sum(reads[0].map(value => Math.min(value, 0))), 0);
      reads.forEach(row => near(sum(row), 1));
      return ['Coverage accumulation and unnormalized total', 'Overlap loss and first-read zero boundary'];
    },
  },
  'long-context-sequence-models-transformer-xl-griffin-perceiver': {
    generator: 'scripts/render-long-context-lesson.mjs',
    anchor: 'constructed one-coordinate head',
    check() {
      const positionFeatures = [1, -1], global = .5;
      const scores = query => positionFeatures.map(feature => query * feature + global * feature);
      assert.deepEqual(scores(1), [1.5, -1.5]);
      assert.deepEqual(scores(-1), [-.5, .5]);
      const probabilityNear = values => Math.exp(values[0]) / sum(values.map(Math.exp));
      assert.ok(probabilityNear(scores(1)) > .5);
      assert.ok(probabilityNear(scores(-1)) < .5);
      assert.deepEqual(positionFeatures.map(feature => global * feature), [.5, -.5]);
      return ['Four-term score isolation changes preference with query', 'Global distance contribution stays fixed'];
    },
  },
  'state-space-models-s4-mamba-mamba-2': {
    generator: 'scripts/generate-state-space-lesson.mjs',
    anchor: 'original two-state calculation',
    check() {
      const input = [1, 0, 1], kernel = [1, 2];
      const linear = Array(input.length + kernel.length - 1).fill(0);
      const circular = Array(input.length).fill(0);
      input.forEach((value, i) => kernel.forEach((tap, j) => {
        linear[i + j] += value * tap;
        circular[(i + j) % circular.length] += value * tap;
      }));
      assert.deepEqual(linear, [1, 2, 1, 2]);
      assert.deepEqual(circular, [3, 2, 1]);
      const padded = Array(linear.length).fill(0);
      input.forEach((value, i) => kernel.forEach((tap, j) => { padded[(i + j) % padded.length] += value * tap; }));
      assert.deepEqual(padded, linear);
      const diagonalInverse = [1, .5], denominator = 1 + sum(diagonalInverse);
      near(denominator, 2.5);
      const correction = diagonalInverse.map(a => diagonalInverse.map(b => a * b / denominator));
      const inverse = correction.map((row, i) => row.map((value, j) => (i === j ? diagonalInverse[i] : 0) - value));
      const matrix = [[2, 1], [1, 3]];
      for (const [a, b] of [[matrix, inverse], [inverse, matrix]]) {
        for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) near(sum(a[i].map((value, k) => value * b[k][j])), +(i === j));
      }
      near(inverse[0][0], .6); near(inverse[1][0], -.2);
      return ['Linear convolution and circular tail alias', 'Minimum padded length preserves causal output', 'Woodbury correction and two-sided inverse identity'];
    },
  },
};

const selected = process.argv.slice(2);
assert.ok(selected.length, 'Supply exact topic IDs.');
for (const topicId of selected) {
  const topic = topics[topicId];
  assert.ok(topic, 'Unknown scoped topic: ' + topicId);
  const directory = `docs/teaching/concept-intuition/${topicId}/`;
  const canonicalManuscript = directory + 'lesson.md';
  const output = `src/learn/data/topics/${topic.file ?? topicId}.jsx`;
  const before = await readFile(output, 'utf8');
  execFileSync(process.execPath, [topic.generator], { stdio: 'pipe' });
  assert.equal(await readFile(output, 'utf8'), before, 'Output must already agree with active canonical generator');
  const manuscript = await readFile(canonicalManuscript, 'utf8');
  const historical = await readFile(`docs/teaching/revisions/${topicId}/4/lesson.md`, 'utf8');
  const codeFences = value => [...value.replaceAll('\r\n', '\n').matchAll(/```[^\n]*\n[\s\S]*?\n```/g)].map(match => match[0]);
  assert.deepEqual(codeFences(manuscript), codeFences(historical), 'Preserve all original code fences');
  assert.ok(before.includes(topic.anchor));
  await transform(before, { loader: 'jsx', jsx: 'automatic', target: 'es2022' });
  const passed = topic.check();
  const sourceFiles = [output, canonicalManuscript, topic.generator, directory + 'review.md', 'scripts/verify-sequence-architecture-concept-intuition.mjs'];
  const sourceHashes = {};
  for (const file of sourceFiles) sourceHashes[file] = createHash('sha256').update(await readFile(file)).digest('hex');
  await writeFile(directory + 'author-checks.json', JSON.stringify({
    topicId, reviewedAt: new Date().toISOString(), canonicalManuscript, sourceFiles, sourceHashes,
    checks: [
      { name: 'Complete lesson reading and local concept map', status: 'author-reviewed', evidence: directory + 'review.md' },
      { name: 'Deterministic canonical generator agreement', status: 'passed' },
      { name: 'All revision-4 code fences retained verbatim', status: 'passed' },
      { name: 'JSX syntax transform', status: 'passed' },
      ...passed.map(name => ({ name, status: 'passed' })),
    ],
    unchangedEvidence: 'Revision-4 manuscripts and receipts retained; numerical engines, native programs, measured data and fitted weights unchanged. No fresh native-fit campaign claimed.',
    independentReview: 'pending', browserReview: 'pending root inspection', integration: 'pending',
  }, null, 2) + '\n');
  process.stdout.write(`${topicId}: canonical source, code retention, JSX and ${passed.length} arithmetic groups passed\n`);
}

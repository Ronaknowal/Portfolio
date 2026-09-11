// Complementary reviewer fixtures. No author-owned production source is changed.
import fs from 'node:fs';
import { spawnSync } from 'node:child_process';
import * as model from '../src/learn/data/counting-combinatorics-models.js';
import { countingCombinatoricsExamples as examples } from '../src/learn/data/counting-combinatorics-examples.js';

const directory = 'scratch/counting-combinatorics-independent-review';
fs.mkdirSync(directory, { recursive: true });
const fixtures = { examples, allocations: [], coefficients: [], choices: [], paths: [], rings: [], binomial: [] };
for (let seed = 0; seed < 144; seed += 1) {
  const length = seed % 13;
  const minimums = Array.from({ length }, (_, i) => (seed + 2 * i) % 4);
  const capacities = minimums.map((low, i) => low + (seed * 7 + i * 11) % 41);
  const total = seed % 4 === 0 ? seed : minimums.reduce((a, b) => a + b, 0) + seed % 111;
  fixtures.allocations.push({ total, capacities, minimums, count: model.allocationCount(total, capacities, minimums) });
}
for (let seed = 0; seed < 80; seed += 1) {
  const capacities = Array.from({ length: seed % 9 }, (_, i) => (seed + i * 3) % 12);
  fixtures.coefficients.push({ capacities, rows: model.coefficientStages(capacities) });
}
for (let n = 0; n <= 5; n += 1) {
  for (let length = 0; length <= 4; length += 1) {
    for (const repeats of [false, true]) fixtures.choices.push({ n, length, repeats, data: model.choiceFibers(n, length, repeats) });
  }
}
// Exercise the accepted 14-symbol path boundary, beyond the UI's five pairs.
for (let bits = 0; bits < 2 ** 14; bits += 1) {
  const word = bits.toString(2).padStart(14, '0').replaceAll('0', '(').replaceAll('1', ')');
  if ([...word].filter(c => c === '(').length === 7) fixtures.paths.push(model.parenthesisPath(word));
}
for (let n = 1; n <= 8; n += 1) fixtures.rings.push({ n, data: model.rotationOrbits(n) });
for (const n of [0, 1, 50, 1000, 9999, 10000]) {
  for (const k of [-1, 0, 1, 2, 11, 49, n - 1, n, n + 1]) fixtures.binomial.push({ n, k, value: model.binomialCount(n, k) });
}
fs.writeFileSync(`${directory}/fixtures.json`, JSON.stringify(fixtures, (_, value) => typeof value === 'bigint' ? value.toString() : value));
const result = spawnSync('scratch/lesson-tools/Scripts/python.exe', ['scripts/verify-counting-combinatorics-independent.py'], { encoding: 'utf8' });
process.stdout.write(result.stdout);
process.stderr.write(result.stderr);
if (result.status !== 0) process.exit(result.status ?? 1);

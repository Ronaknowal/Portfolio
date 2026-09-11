// Reviewer-only changed-input fixtures; no author source is rewritten.
import { mkdirSync, writeFileSync } from 'node:fs';
import { categoryTheoryExamples } from '../src/learn/data/category-theory-examples.js';
import { multiplyStochastic, pullbackPairs, tangentSnapshot } from '../src/learn/data/category-theory-models.js';

const destination = 'scratch/category-theory-independent-review';
mkdirSync(destination, { recursive: true });
const records = { channels: [], pullbacks: [], tangents: [] };
for (let seed = 0; seed < 60; seed += 1) {
  const a = Array.from({ length: 3 }, (_, i) => [((seed + 2 * i) % 7), 7 - ((seed + 2 * i) % 7)]);
  const b = Array.from({ length: 2 }, (_, i) => {
    const x = (seed + i) % 5;
    const y = (2 * seed + i) % (8 - x);
    return [x, y, 8 - x - y];
  });
  records.channels.push({ a, b, result: multiplyStochastic(a.map(row => row.map(x => x / 7)), b.map(row => row.map(x => x / 8))) });
}
for (let mask = 0; mask < 256; mask += 1) {
  const a = Array.from({ length: 3 }, (_, i) => (mask >> i) & 1);
  const b = Array.from({ length: 5 }, (_, i) => (mask >> (i + 3)) & 1);
  records.pullbacks.push({ a, b, result: pullbackPairs(a, b, 2) });
}
for (const choice of ['shiftedSquare', 'cubicAffine', 'squareSquare']) {
  for (const x of [-2.19, -0.37, 0, 0.43, 1.73, 2.81]) {
    records.tangents.push({ x, choice, result: tangentSnapshot(x, -1.37, choice, 0.83) });
  }
}
writeFileSync(destination + '/fixtures.json', JSON.stringify(records, null, 2));
writeFileSync(destination + '/actual-examples.json', JSON.stringify(categoryTheoryExamples, null, 2));
console.log(JSON.stringify({ written: destination, cases: Object.fromEntries(Object.entries(records).map(([key, rows]) => [key, rows.length])) }));

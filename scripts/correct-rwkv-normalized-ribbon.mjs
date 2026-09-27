import fs from 'node:fs';
const file = 'src/learn/components/lesson-labs/RwkvMemoryLabs.jsx';
let source = fs.readFileSync(file, 'utf8');
const before = '<MemoryContributionRibbon title="Current numerator contribution from each record" values={row.weights.map((weight, i) => weight === null ? 0 : weight * input.values[i])} />';
const after = '{row.output === null ? <p role="status">Normalized contributions are undefined because the query has zero total kernel weight.</p> : <MemoryContributionRibbon title="Contributions to the normalized answer" values={row.weights.map((weight, i) => weight * input.values[i])} />}';
if (!source.includes(before)) throw new Error('Expected ribbon label not found');
fs.writeFileSync(file, source.replace(before, after));

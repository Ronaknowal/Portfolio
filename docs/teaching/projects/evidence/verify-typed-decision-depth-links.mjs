// Run from the repository root. Validate displayed code against its canonical
// downloadable source and verify topic links without importing lesson bodies.
import fs from 'node:fs';
import crypto from 'node:crypto';
import { topicMap } from '../../../../src/learn/data/catalogue.js';

const base = 'src/learn/data/projects/typed-decision-model/';
const files = ['mechanism-depth.jsx', 'research-depth.jsx', 'content.jsx'];
const output = 'docs/teaching/projects/evidence/typed-decision-depth-links.json';
fs.writeFileSync(output, JSON.stringify({ passed: false, state: 'running' }) + '\n');
const sources = [];
const links = [];
const hashes = {};
for (const file of files) {
  const text = fs.readFileSync(base + file, 'utf8');
  hashes[base + file] = crypto.createHash('sha256').update(fs.readFileSync(base + file)).digest('hex');
  for (const match of text.matchAll(/<Source\s+([^>]+)\/>/g)) {
    const attributes = Object.fromEntries([...match[1].matchAll(/(\w+)="([^"]*)"/g)].map(item => [item[1], item[2]]));
    const filename = 'public/learn-projects/typed-decision-model/' + (attributes.file || 'typed_decision.py');
    const source = fs.readFileSync(filename, 'utf8');
    hashes[filename] = crypto.createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
    const begin = source.indexOf(attributes.start);
    const end = source.indexOf(attributes.end, begin + attributes.start.length);
    if (begin < 0 || end <= begin) throw new Error(`Missing or reversed source boundary ${JSON.stringify(attributes)}`);
    if (source.indexOf(attributes.start, begin + 1) >= 0) throw new Error(`Ambiguous source start ${attributes.start}`);
    sources.push({ component: file, title: attributes.title, source: attributes.file || 'typed_decision.py', characters: end - begin });
  }
  for (const match of text.matchAll(/\{ id: '([^']+)', reason: '([^']+)' \}/g)) {
    const topic = topicMap[match[1]];
    if (!topic) throw new Error('Broken topic link: ' + match[1]);
    links.push({ id: match[1], status: topic.status, title: topic.title });
  }
}
// A missing match must not turn an accidentally vacuous audit into a pass.
if (sources.length < 17 || links.length < 31) throw new Error('Expected source or concept nodes were not inspected');
const report = {
  passed: true, source_excerpt_count: sources.length, concept_links: links.length,
  unique_concepts: new Set(links.map(link => link.id)).size,
  planned_concepts: [...new Set(links.filter(link => link.status !== 'published').map(link => link.id))],
  sources, links, source_sha256: hashes,
};
fs.writeFileSync(output, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ ...report, sources: undefined, links: undefined, source_sha256: undefined }, null, 2));

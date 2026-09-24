// Source inventory only. Presence of a string/import is not a teaching-quality pass.
import fs from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { trackDefinitions } from '../src/learn/data/track-definitions.js';
import { slugify } from '../src/learn/data/topic-id.js';

const modules = ['programming-scientific-computing', 'data-structures-algorithms', 'math-foundations'];
const manifest = JSON.parse(fs.readFileSync('src/learn/data/lesson-manifest.json', 'utf8'));
const digest = text => createHash('sha256').update(text).digest('hex');
const locate = file => {
  const text = fs.readFileSync(file, 'utf8');
  const lines = text.split(/\r?\n/);
  return {
    path: file.replaceAll('\\', '/'), sha256: digest(text),
    headings: lines.flatMap((line, index) => /<H[23][ >]|^\s*(?:title|name):/.test(line) ? [{line:index + 1,text:line.trim().slice(0, 180)}] : []),
    programReferences: lines.flatMap((line, index) => /<RunnableExample|<PythonExample|<CodeBlock|<Practice|<Checkpoint|practice=|examples\./i.test(line) ? [{line:index + 1,text:line.trim().slice(0, 240)}] : []),
  };
};
const topics = modules.flatMap(moduleId => trackDefinitions.find(track => track.id === moduleId).sections.flatMap(section => section.topics.map(topic => {
  const title = typeof topic === 'string' ? topic : topic.title;
  const id = slugify(title), source = path.posix.normalize(`src/learn/data/${manifest[id]}`);
  const text = fs.readFileSync(source, 'utf8');
  const owned = [...text.matchAll(/from\s+['"]([^'"]+)['"]/g)].map(match => match[1]).filter(value => /examples|deeper|reference|practice\//.test(value)).map(value => {
    const base = path.posix.normalize(path.posix.join(path.posix.dirname(source), value));
    return [base, `${base}.js`, `${base}.jsx`].find(file => fs.existsSync(file));
  }).filter(Boolean);
  return {id, title, moduleId, section:section.name, source:locate(source), learnerSupportFiles:[...new Set(owned)].map(locate)};
})));
if (topics.length !== 96 || new Set(topics.map(topic => topic.id)).size !== 96) throw Error('Scope is not exactly 96 unique topics.');
const receipt = {kind:'source ownership inventory; no automatic quality classification', timestamp:new Date().toISOString(), counts:Object.fromEntries(modules.map(id => [id,topics.filter(topic => topic.moduleId === id).length])), topics};
fs.mkdirSync('docs/teaching/implementation-depth', {recursive:true});
fs.writeFileSync('docs/teaching/implementation-depth/foundations-source-inventory.json', JSON.stringify(receipt, null, 2) + '\n');
console.log(`Inventoried ${topics.length} lesson bodies and ${new Set(topics.flatMap(topic => topic.learnerSupportFiles.map(file => file.path))).size} directly imported example/reference/practice files. Manual findings belong in FOUNDATIONS.md.`);

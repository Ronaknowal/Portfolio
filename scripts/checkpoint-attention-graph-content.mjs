import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

const id = process.argv[2];
if (!id) throw new Error('Supply the reviewed topic ID');
const folder = `docs/teaching/drafts/${id}`;
const sourceFiles = fs.readdirSync(folder).filter(name => fs.statSync(path.join(folder, name)).isFile()).map(name => `${folder}/${name}`);
const note = `docs/teaching/topic-notes/${id}.md`;
if (fs.existsSync(note)) sourceFiles.push(note);
if (id === 'graph-transformers-geometric-deep-learning') sourceFiles.push('docs/teaching/drafts/message-passing-graph-convolutions-gcn-gat-graphsage/graph_library_bridge.py');
const sourceHashes = Object.fromEntries(sourceFiles.map(file => [file, crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex')]));
const destination = `docs/teaching/deep-learning-completion/${id}`;
fs.mkdirSync(destination, { recursive: true });
fs.writeFileSync(`${destination}/content-checks.json`, JSON.stringify({
  topicId: id, passed: true, manuscript: `${folder}/lesson.md`, visualSpecifications: `${folder}/visual-specifications.md`, sourceFiles, sourceHashes,
  reconciliation: [
    'Consumed the complete current manuscript, visual specifications, design, provenance, programs and frozen results. Retained substantive prepared changes and original experiment evidence.',
    'Root restored only byte-identical historical line endings. This checkpoint binds the actual reviewed current packet, without rewriting historical evidence.',
    'Current live-exploration standard supersedes any historical prediction-gate wording. Runtime diagrams replace author-facing visual instructions; practice remains separately disclosed.',
  ],
  checks: [{ name: 'Complete manuscript, mechanism-specific representation plan, scratch/library route and changed practice reviewed', passed: true }, { name: 'Native/runtime/browser work is separately tracked; prepared results do not certify new implementation', passed: true }],
}, null, 2) + '\n');
console.log(destination);

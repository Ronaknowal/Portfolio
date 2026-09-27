import fs from 'node:fs';

for (const [file, prefix] of [
  ['src/learn/components/lesson-labs/RwkvMechanismFigures.jsx', 'rwkv'],
  ['src/learn/components/lesson-labs/SelfAttentionFigures.jsx', 'attention'],
]) {
  let source = fs.readFileSync(file, 'utf8');
  if (!source.includes(`${prefix}-diagram-scroll`)) {
    source = source.replaceAll('<svg ', `<div className="${prefix}-diagram-scroll" role="region" aria-label="Mechanism diagram; scroll horizontally if needed" tabIndex={0}><svg `).replaceAll('</svg>', '</svg></div>');
    fs.writeFileSync(file, source);
  }
}
const file = 'src/learn/components/lesson-labs/SelfAttentionLabs.jsx';
let source = fs.readFileSync(file, 'utf8');
source = source.replace('<div className="self-attention-columns"><div><WeightRow weights={current.weights}', '<AttentionKeyPlane query={input.query} keys={input.keys} /><div className="self-attention-columns"><div><WeightRow weights={current.weights}');
source = source.replace('{current.error ? <p role="status">{current.error}', '<AttentionLegalEdges receiver={receiver} allowed={allowed} weights={current.weights} />{current.error ? <p role="status">{current.error}');
// Drop replaced text-only branches; their specialized diagrams now own those cases.
source = source.split('\n').filter(line => !(line.startsWith("  if (kind === 'roles') return <figure") || line.startsWith("  if (kind === 'pipeline') return <figure"))).join('\n');
fs.writeFileSync(file, source);
const rwkv = 'src/learn/components/lesson-labs/RwkvMemoryLabs.jsx';
let memory = fs.readFileSync(rwkv, 'utf8');
memory = memory.split('\n').filter(line => !['transcript', 'chunk', 'circuit', 'block', 'pipeline'].some(kind => line.startsWith(`  if (kind === '${kind}') return `))).join('\n');
fs.writeFileSync(rwkv, memory);

import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
const packetPath = 'docs/teaching/evidence/pde-author-review.json';
const previousPath = 'docs/teaching/evidence/pde-author-review-before-independent-amendment.json';
if (fs.existsSync(previousPath)) throw new Error('Existing initial archive must not be overwritten.');
const packet = JSON.parse(fs.readFileSync(packetPath, 'utf8'));
const hash = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
for (const source of packet.sourceHashes) {
  if (hash(fs.readFileSync(source.path)) !== source.sha256) throw new Error(`Pre-amendment mismatch: ${source.path}`);
}
const sources = [...packet.sourceHashes.map(source => source.path), packetPath,
  'docs/teaching/evidence/pde-native-verification.json', 'docs/teaching/evidence/pde-browser-review.json',
  'docs/teaching/PARTIAL-DIFFERENTIAL-EQUATIONS-VERIFICATION.md', 'scripts/generate-pde-examples.py'];
const archived = sources.map(source => {
  const archive = `docs/teaching/archive/pde-before-independent-amendment/${source}.txt`;
  fs.mkdirSync(path.dirname(archive), { recursive: true });
  const bytes = fs.readFileSync(source);
  fs.writeFileSync(archive, bytes, { flag: 'wx' });
  return { source, archive, sha256: hash(bytes) };
});
fs.copyFileSync(packetPath, previousPath);
fs.writeFileSync('docs/teaching/evidence/pde-independent-amendment-originals.json', JSON.stringify({ archivedAt: new Date().toISOString(), previousPacket: previousPath, archived }, null, 2) + '\n', { flag: 'wx' });
console.log(`Archived ${archived.length} pre-amendment sources and evidence files.`);

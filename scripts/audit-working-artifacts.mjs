// Read-only classification of working images; never deletes or executes lesson code.
// Usage: node scripts/audit-working-artifacts.mjs [output-json-path] [--protect scratch-folder-prefix ...]
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
const root = process.cwd();
const args = process.argv.slice(2);
const output = args[0] && !args[0].startsWith('--') ? args.shift() : 'scratch/artifact-cleanup/audit.json';
// Shared environments are always protected. Active work is supplied for this run,
// rather than inferred from a permanently hard-coded, already completed batch.
const activePrefixes = ['lesson-tools', 'classical-depth-runtime', 'endtoend-venv', 'manifold-learning-runtime', 'math-library-runtime', 'programming-three-runtime', 'survival-tools', 'artifact-cleanup'];
while (args.length) {
  if (args.shift() !== '--protect') throw new Error('Expected --protect <scratch-folder-prefix>');
  const prefix = args.shift();
  if (!prefix || !/^[a-z0-9][a-z0-9_-]*$/i.test(prefix)) throw new Error('Use a scratch folder-name prefix, not a path');
  activePrefixes.push(prefix);
}
function walk(directory, skip = () => false) {
  if (!fs.existsSync(directory)) return [];
  return fs.readdirSync(directory, {
    withFileTypes: true
  }).flatMap(entry => {
    const filename = path.join(directory, entry.name);
    if (entry.isSymbolicLink() || (entry.isDirectory() && ['node_modules', '.venv', 'venv', '__pycache__', '.git'].includes(entry.name)) || skip(filename, entry)) return [];
    return entry.isDirectory() ? walk(filename, skip) : [filename.replaceAll('\\', '/')];
  });
}
const scratchFiles = walk('scratch', filename => activePrefixes.some(prefix => filename.replaceAll('\\', '/').split('/')[1].startsWith(prefix)));
const evidenceFiles = [...walk('docs'), ...walk('scripts'), ...walk('src'), ...fs.readdirSync('.').filter(file => file.endsWith('.md')), ...scratchFiles].filter(file => /\.(md|json|cjs|mjs|js|jsx|py|txt)$/.test(file));
const imageReferences = new Set();
const referenceOrigins = new Map();
for (const file of evidenceFiles) {
  const info = fs.statSync(file);
  if (info.size > 40e6) continue;
  const content = fs.readFileSync(file, 'utf8');
  for (const match of content.matchAll(/[A-Za-z0-9_./\\-]+\.(?:png|jpe?g|webp|svg|gif)\b/gi)) {
    const value = match[0].replaceAll('\\', '/');
    const basename = path.posix.basename(value);
    imageReferences.add(basename);
    if (!referenceOrigins.has(basename)) referenceOrigins.set(basename, file);
  }
}
const hashes = new Map();
const images = scratchFiles.filter(file => /\.(png|jpe?g|webp|gif)$/i.test(file)).map(file => {
  const bytes = fs.statSync(file).size;
  const sha256 = crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
  const referenced = imageReferences.has(path.posix.basename(file));
  const item = {
    path: file,
    bytes,
    sha256,
    referenced,
    reference: referenced ? referenceOrigins.get(path.posix.basename(file)) : null
  };
  if (!hashes.has(sha256)) hashes.set(sha256, []);
  hashes.get(sha256).push(item);
  return item;
});
const duplicateCandidates = [];
for (const group of hashes.values()) {
  const keeper = group.find(item => item.referenced) || group[0];
  for (const item of group) if (item !== keeper && !item.referenced) duplicateCandidates.push({
    ...item,
    duplicateOf: keeper.path
  });
}
const unreferenced = images.filter(item => !item.referenced && !duplicateCandidates.some(candidate => candidate.path === item.path));
const report = {
  auditedAt: new Date().toISOString(),
  policy: 'Shared runtimes and explicitly supplied active prefixes excluded. Remaining documentation/source/script/scratch text checked for image basenames. A candidate is not deletion approval; dynamic paths and live owners require manual inspection. No deletions performed.',
  activePrefixes,
  imageCount: images.length,
  imageBytes: images.reduce((s, i) => s + i.bytes, 0),
  referencedCount: images.filter(i => i.referenced).length,
  duplicateCandidates,
  unreferenced,
  images
};
fs.mkdirSync(path.dirname(output), {
  recursive: true
});
fs.writeFileSync(output, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({
  output,
  imageCount: report.imageCount,
  imageBytes: report.imageBytes,
  referencedCount: report.referencedCount,
  unreferencedCount: unreferenced.length,
  unreferencedBytes: unreferenced.reduce((s, i) => s + i.bytes, 0),
  duplicates: duplicateCandidates.length,
  duplicateBytes: duplicateCandidates.reduce((s, i) => s + i.bytes, 0),
  candidateExamples: [...duplicateCandidates, ...unreferenced].slice(0, 12)
}, null, 2));

import fs from 'node:fs';
import crypto from 'node:crypto';
const id = process.argv[2];
const folder = `docs/teaching/deep-learning-completion/${id}`;
const config = JSON.parse(fs.readFileSync(`${folder}/implementation-scope.json`));
const content = JSON.parse(fs.readFileSync(`${folder}/content-checks.json`));
const downloadFolder = `public/learn-code/${id}`;
const deployed = fs.existsSync(downloadFolder) ? fs.readdirSync(downloadFolder).filter(name => fs.statSync(`${downloadFolder}/${name}`).isFile()).map(name => `${downloadFolder}/${name}`) : [];
const files = [...new Set([...content.sourceFiles, ...config.runtimeFiles, ...config.evidence, ...deployed])];
const sourceHashes = Object.fromEntries(files.map(file => [file, crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex')]));
const evidence = config.evidence.filter(file => file.endsWith('checks.json')).map(file => ({ file, ...JSON.parse(fs.readFileSync(file)) }));
if (!evidence.length || evidence.some(item => !item.passed)) throw new Error('All actual verification receipts must pass');
fs.writeFileSync(`${folder}/implementation-checks.json`, JSON.stringify({
  topicId: id, passed: true, status: 'author-ready', productionPath: `src/learn/data/topics/${id}.jsx`,
  sourceFiles: files, sourceHashes, checks: evidence.map(item => ({ name: item.file, passed: item.passed })),
  evidence: config.evidence, browserInventory: config.browserInventory,
  remaining: ['independent', 'browser', 'integration'],
  limits: config.limits,
}, null, 2) + '\n');
console.log(`${id}: author-ready, ${files.length} source/evidence files bound`);

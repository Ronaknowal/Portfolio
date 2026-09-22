import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { execFileSync } from 'node:child_process';
import assert from 'node:assert/strict';
import { getDeliveryState, filesMatch, validateDeliveryLedger } from './lib/lesson-delivery.mjs';
import { topicCatalogue } from '../src/learn/data/curriculum/topic-catalogue.js';

// Content/identity checks only. This does not claim execution or implementation review.
const root = process.cwd();
const read = file => JSON.parse(fs.readFileSync(file, 'utf8'));
const digest = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const hash = file => digest(fs.readFileSync(file));
const baseline = read('docs/teaching/evidence/prepared-depth-writing-baseline.json');
const ledger = read('docs/teaching/lesson-delivery-progress.json');
const reports = [
  ['foundations', 'docs/teaching/implementation-depth/prepared-foundations-writing.json'],
  ['attention', 'docs/teaching/implementation-depth/prepared-attention-writing.json'],
  ['specialists', 'docs/teaching/implementation-depth/prepared-specialists-writing.json'],
];
const cases = [];
const check = (name, fn) => { fn(); cases.push(name); };
const packetRows = [];

check('Author handoffs cover exactly the 37 authorized packets on matching source bytes', () => {
  for (const [group, filename] of reports) {
    const report = read(filename);
    assert.equal(report.status, 'content-prepared', filename);
    assert.deepEqual(report.topics.map(row => row.id).sort(), [...baseline.groups[group]].sort(), group);
    for (const row of report.topics) {
      assert.ok(row.coverage?.length, `Missing source-located outcome map: ${row.id}`);
      assert.ok(Object.keys(row.sourceHashes || {}).length, `Missing source binding: ${row.id}`);
      assert.ok(row.contentFiles?.length, `Missing handoff inputs: ${row.id}`);
      for (const file of row.contentFiles) assert.ok(row.sourceHashes[file], `Unbound handoff input: ${file}`);
      for (const [file, expected] of Object.entries(row.sourceHashes)) assert.equal(hash(file), expected, file);
      packetRows.push(row);
    }
  }
  assert.equal(packetRows.length, 37);
});
check('The five previously missing specialist routes have written-content dispositions', () => {
  for (const id of [
    'message-passing-graph-convolutions-gcn-gat-graphsage',
    'graph-transformers-geometric-deep-learning',
    'boltzmann-machines-restricted-boltzmann-machines-rbm',
    'advanced-optimizers-lion-sophia-prodigy-schedule-free',
    'neural-ode-continuous-depth-models',
  ]) {
    const filename = `docs/teaching/topic-notes/${id}.md`;
    const source = fs.readFileSync(filename, 'utf8');
    assert.match(source, /Content disposition:\s*(?:\*\*)?prepared/i, filename);
    assert.ok(packetRows.find(row => row.id === id).sourceHashes[filename], `Unbound note: ${filename}`);
  }
});
check('Current content checkpoints are complete; implementation remains not started', () => {
  validateDeliveryLedger(ledger, new Set(Object.keys(topicCatalogue)));
  for (const id of baseline.scope) {
    const row = ledger.topics[id], prior = baseline.priorRows[id];
    assert.equal(row.revision, prior.revision + 1, id);
    assert.equal(row.deliveryMode, 'content-first', id);
    const { previousRevisions = [], ...old } = prior;
    assert.deepEqual(row.previousRevisions, [...previousRevisions, old], id);
    const state = getDeliveryState(root, row);
    assert.equal(state.content, 'complete', id);
    assert.equal(state.implementation, 'not-started', id);
    assert.equal(state.canFinish, true, id);
    const author = packetRows.find(packet => packet.id === id);
    for (const file of author.contentFiles || []) {
      assert.equal(row.content.files[file], hash(file), `Required handoff input missing from checkpoint: ${file}`);
    }
  }
});
let unchanged = 0;
check('All 140 other phase rows and historical checkpoints are unchanged', () => {
  assert.equal(Object.keys(ledger.topics).length, Object.keys(baseline.allRowHashes).length);
  for (const [id, expected] of Object.entries(baseline.allRowHashes)) {
    if (baseline.scope.includes(id)) continue;
    assert.equal(digest(JSON.stringify(ledger.topics[id])), expected, id);
    unchanged++;
  }
  assert.equal(unchanged, 140);
});
check('All production sources/assets and publication mappings are unchanged', () => {
  const current = execFileSync('rg', ['--files', 'src', 'public'], { encoding: 'utf8' }).trim().split(/\r?\n/).map(file => file.replaceAll('\\', '/'));
  assert.deepEqual(current.sort(), Object.keys(baseline.runtimeFiles).sort());
  for (const [file, expected] of Object.entries(baseline.runtimeFiles)) assert.equal(hash(file), expected, file);
  assert.equal(hash('src/learn/data/lesson-manifest.json'), baseline.runtimeManifestSha256);
});
const historical = read('docs/teaching/dsa-math-foundations-progress.json');
const effective = Object.entries(ledger.topics).map(([id, row]) => {
  const review = row.legacyReview && historical.topics.find(topic => topic.id === id);
  const legacyCurrent = review?.status === 'implementation-reviewed' && filesMatch(root, review.reviewedFiles);
  return getDeliveryState(root, row, legacyCurrent);
});
const counts = {
  contentComplete: effective.filter(row => row.content === 'complete').length,
  implementationComplete: effective.filter(row => row.implementation === 'complete').length,
  prepared: effective.filter(row => row.content === 'complete' && row.implementation === 'not-started').length,
};
check('Current delivery totals preserve the separate phases', () => {
  assert.deepEqual(counts, { contentComplete: 177, implementationComplete: 140, prepared: 37 });
});
const manuscriptFiles = baseline.scope.flatMap(id => ['lesson.md', 'design.md', 'visual-specifications.md'].map(name => `docs/teaching/drafts/${id}/${name}`));
const pythonFiles = [...new Set(baseline.scope.flatMap(id => Object.keys(ledger.topics[id].content.files || {}).filter(file => file.endsWith('.py'))))];
check('Prepared instructional Python files parse without importing or executing them', () => {
  assert.ok(pythonFiles.length > 0);
  const python = process.env.LEARNING_PYTHON || 'scratch/lesson-tools/Scripts/python.exe';
  const program = 'import ast,json,pathlib,sys\nfor name in json.load(sys.stdin):\n ast.parse(pathlib.Path(name).read_text(encoding="utf-8-sig"),filename=name)\n';
  execFileSync(python, ['-B', '-c', program], { input: JSON.stringify(pythonFiles), encoding: 'utf8' });
});
let localLinks = 0;
check('Required packet documents and their local file links exist', () => {
  for (const file of manuscriptFiles) {
    assert.ok(fs.statSync(file).size > 0, file);
    const source = fs.readFileSync(file, 'utf8')
      .replace(/^```[^\n]*\n[\s\S]*?^```\s*$/gm, '')
      .replace(/(`+)[\s\S]*?\1/g, ''); // An inline Python expression such as layers[i](x) is not a link.
    for (const match of source.matchAll(/\[[^\]\n]*\]\(([^\s)]+)(?:\s+"[^"]*")?\)/g)) {
      const href = match[1].replace(/^<|>$/g, '');
      if (/^(?:[a-z]+:|\/|#)/i.test(href)) continue;
      const target = decodeURIComponent(href.split('#')[0].split('?')[0]);
      if (!target) continue;
      assert.ok(fs.existsSync(path.resolve(path.dirname(file), target)), `${file}: missing ${href}`);
      localLinks++;
    }
  }
});

const report = {
  status: 'passed', checkedAt: new Date().toISOString(), phase: 'content-only', checks: cases,
  topics: baseline.scope, counts, unchangedPhaseRows: unchanged, unchangedRuntimeFiles: Object.keys(baseline.runtimeFiles).length,
  localFileLinksChecked: localLinks,
  instructionalPythonFilesParsed: pythonFiles.length,
  authorManifests: Object.fromEntries(reports.map(([, filename]) => [filename, hash(filename)])),
  ledgerSha256: hash('docs/teaching/lesson-delivery-progress.json'),
  verifierSha256: hash('scripts/verify-prepared-depth-writing.mjs'),
  limits: 'These are content handoff/identity/link checks. Author reports state actual reading, research and bounded correctness probes. No full native/training/GPU/performance, independent implementation review, browser or build campaign is claimed.',
};
fs.writeFileSync('docs/teaching/evidence/prepared-depth-writing-final.json', JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ status: report.status, checks: cases.length, counts, unchangedPhaseRows: unchanged, localLinks }));

import fs from 'node:fs';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';

const topicId = 'complex-numbers-fourier-laplace-transforms';
const inventory = JSON.parse(execFileSync(process.execPath, ['scripts/build-curriculum-inventory.mjs', '--topic', topicId], { encoding: 'utf8' }));
const sourcePath = 'src/learn/data/curriculum/cross-domain-expansion.js';
const source = fs.readFileSync(sourcePath, 'utf8');
const beginning = source.indexOf('    plan("Complex Numbers, Fourier & Laplace Transforms"');
const end = source.indexOf('    plan("Conditioning, Stability & Numerical Analysis"', beginning);
const target = 'docs/teaching/evidence/complex-transforms-original-plan.json';
if (!fs.existsSync(target)) {
  fs.writeFileSync(target, `${JSON.stringify({
    capturedAt: new Date().toISOString(),
    topicId,
    command: `node scripts/build-curriculum-inventory.mjs --topic ${topicId}`,
    publicationStatus: inventory.topic.publicationStatus,
    bodyExisted: fs.existsSync(`src/learn/data/topics/${topicId}.jsx`),
    individualBlueprintExisted: fs.existsSync(`src/learn/data/curriculum/blueprints/${topicId}.js`),
    originalSource: { path: sourcePath, sha256: createHash('sha256').update(fs.readFileSync(sourcePath)).digest('hex') },
    originalPlanExcerpt: source.slice(beginning, end),
    inventory,
  }, null, 2)}\n`);
}
console.log(target);

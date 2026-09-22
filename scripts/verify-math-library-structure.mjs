import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { pathToFileURL } from 'node:url';
import { parse } from '@babel/parser';

const manifestPath = 'docs/teaching/implementation-depth/math-library-remediation.json';
const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
const hash = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
assert.equal(manifest.topics.length, 7);
for (const topic of manifest.topics) {
  const source = fs.readFileSync(topic.body, 'utf8');
  parse(source, { sourceType: 'module', plugins: ['jsx'] });
  assert(source.includes(`source="${topic.programUrl}"`));
  assert(source.includes('output={libraryOutput}'));
  for (const [file, expected] of Object.entries(topic.sourceHashes)) {
    assert.equal(hash(file), expected, `${topic.id}: changed evidence source ${file}`);
  }
  const native = JSON.parse(fs.readFileSync(topic.evidencePaths[0], 'utf8'));
  assert.equal(native.exitCode, 0);
  assert.equal(hash(native.source), native.sha256);
  const outputPath = topic.changedFiles.find(file => file.endsWith('-library-output.js'));
  const displayed = (await import(pathToFileURL(path.resolve(outputPath)))).default;
  assert.equal(displayed, native.stdout.trim());
  assert.equal(`/${native.source.slice('public/'.length)}`, topic.programUrl);
  console.log(`${topic.id}: JSX, canonical program, recorded output and source hashes passed`);
}
